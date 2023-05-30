mod api;
mod config;

use api::GenerateError;
use api::GenerateRequest;
use api::GenerateResponse;
use api::StatusResponse;
use async_stream::stream;
use axum::extract::Path;
use axum::extract::Query;
use axum::extract::State;
use axum::http::header::CONTENT_TYPE;
use axum::http::HeaderValue;
use axum::http::Method;
use axum::response::sse::Event;
use axum::response::IntoResponse;
use axum::response::Sse;
use axum::routing::get;
use axum::routing::post;
use axum::Json;
use axum::Router;
use clap::Parser;
use config::Args;
use config::Config;
use futures_util::Stream;
use llm::InferenceParameters;
use llm::InferenceRequest;
use llm::InferenceSessionConfig;
use llm::ModelParameters;
use std::collections::HashMap;
use std::convert::Infallible;
use std::io::Write;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use std::{fs::File, io::Read};
use tower_http::cors::Any;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;

use crate::config::DEFAULT_THREADS_PER_SESSION;

pub struct ServerState {
	config: Config,
	models: HashMap<String, Box<dyn llm::Model>>,
}

#[tokio::main]
async fn main() {
	env_logger::init();
	// Read config file
	let args = Args::parse();
	let mut config_file = File::open(args.config_path).expect("open config file");
	let mut config_string = String::new();
	config_file.read_to_string(&mut config_string).expect("read config file");
	let config: Config = toml::from_str(&config_string).unwrap();
	let bind_address: SocketAddr = config.bind_address.parse().unwrap();
	log::info!("Starting llmd; bind address: {bind_address}",);

	let mut state = ServerState {
		models: HashMap::new(),
		config,
	};

	// Load models
	for (endpoint_name, endpoint) in &state.config.endpoints {
		let params = ModelParameters {
			prefer_mmap: true,
			context_size: 512,
			lora_adapters: None,
		};

		let model = llm::load_dynamic(endpoint.architecture, &endpoint.model_path, params, None, |progress| {
			log::debug!("Loading endpoint {endpoint_name}: {progress:#?}");
		})
		.expect("load model");

		state.models.insert(endpoint_name.clone(), model);
		log::info!("Loaded model for endpoint {}", endpoint_name);
	}

	// Set up CORS
	let mut cors_layer = CorsLayer::new();
	if let Some(ref origins) = state.config.allowed_origins {
		for origin in origins.iter() {
			if origin == "*" {
				cors_layer = cors_layer.allow_origin(Any);
			} else {
				cors_layer = cors_layer.allow_origin(origin.parse::<HeaderValue>().unwrap());
			}
		}
	} else {
		// Allow any origin by default
		cors_layer = cors_layer.allow_origin(Any);
		cors_layer = cors_layer.allow_headers([CONTENT_TYPE]);
		cors_layer = cors_layer.allow_methods([Method::GET, Method::POST]);
	}

	// Set up API server
	let app = Router::new()
		.nest_service("/", ServeDir::new("public"))
		.route("/status", get(status_handler))
		.route("/model/:endpoint/live", get(sse_handler))
		.route("/model/:endpoint/completion", post(post_model_completion_handler))
		.route("/model/:endpoint/completion", get(get_model_completion_handler))
		.with_state(Arc::new(state))
		.layer(cors_layer);
	axum::Server::bind(&bind_address).serve(app.into_make_service()).await.unwrap();
}

async fn status_handler(State(state): State<Arc<ServerState>>) -> impl IntoResponse {
	Json(StatusResponse {
		endpoints: state.config.endpoints.keys().cloned().collect(),
	})
}

async fn sse_handler(
	State(state): State<Arc<ServerState>>,
	Path(endpoint_name): Path<String>,
	Query(request): Query<GenerateRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, GenerateError> {
	log::debug!("New live connection for endpoint '{}'", endpoint_name.as_str());

	if !state.models.contains_key(&endpoint_name) {
		return Err(GenerateError::EndpointNotFound(endpoint_name));
	};

	let (tx, mut rx) = tokio::sync::mpsc::channel(32);

	tokio::spawn(async move {
		let model = state.models.get(&endpoint_name).unwrap();
		let inference_config = InferenceSessionConfig::default();
		let mut session = model.start_session(inference_config);
		let mut inference_parameters: InferenceParameters = request.clone().into();
		inference_parameters.n_threads = state.config.endpoints[&endpoint_name]
			.threads_per_session
			.unwrap_or(DEFAULT_THREADS_PER_SESSION);
		log::trace!("SSE request {:#?}", request);

		session
			.infer(
				model.as_ref(),
				&mut rand::thread_rng(),
				&InferenceRequest {
					prompt: llm::Prompt::Text(&request.prompt),
					parameters: &inference_parameters,
					play_back_previous_tokens: false,
					maximum_token_count: Some(request.max_tokens),
				},
				&mut Default::default(),
				|r| -> Result<_, GenerateError> {
					match r {
						llm::InferenceResponse::PromptToken(t) | llm::InferenceResponse::InferredToken(t) => {
							log::trace!("{t}");
							let tx = tx.clone();

							// Do not continue when client has disconnected
							if tx.is_closed() {
								log::debug!("client has disconnected live session, halting generation");
								return Ok(llm::InferenceFeedback::Halt);
							}
							tokio::spawn(async move {
								// This may fail when a client disconnects while we are generating a token, but we don't care (anymore).
								tx.send(t).await
							});
							Ok(llm::InferenceFeedback::Continue)
						}
						_ => Ok(llm::InferenceFeedback::Continue),
					}
				},
			)
			.unwrap();
	});

	let stream = stream! {
		loop {
			match rx.recv().await {
				Some(token) => {
					let evt = Event::default().id("token").data(token);
					yield Ok(evt);
				},
				None => return
			}

		}
	};

	Ok(Sse::new(stream).keep_alive(
		axum::response::sse::KeepAlive::new()
			.interval(Duration::from_secs(1))
			.text("keep-alive-text"),
	))
}

async fn get_model_completion_handler(
	State(state): State<Arc<ServerState>>,
	Path(endpoint_name): Path<String>,
	Query(request): Query<GenerateRequest>,
) -> Result<Json<GenerateResponse>, GenerateError> {
	completion_handler(state, &endpoint_name, &request).await
}

async fn post_model_completion_handler(
	State(state): State<Arc<ServerState>>,
	Path(endpoint_name): Path<String>,
	Json(request): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, GenerateError> {
	completion_handler(state, &endpoint_name, &request).await
}

async fn completion_handler(
	state: Arc<ServerState>,
	endpoint_name: &str,
	request: &GenerateRequest,
) -> Result<Json<GenerateResponse>, GenerateError> {
	let Some(model) = state.models.get(endpoint_name) else {
		return Err(GenerateError::EndpointNotFound(endpoint_name.to_string()));
	};

	let inference_config = InferenceSessionConfig::default();
	let mut session = model.start_session(inference_config);
	let mut inference_parameters: InferenceParameters = request.clone().into();
	inference_parameters.n_threads = state.config.endpoints[endpoint_name]
		.threads_per_session
		.unwrap_or(DEFAULT_THREADS_PER_SESSION);
	let mut text = String::new();

	log::trace!("Completion request {:?}", request);

	session
		.infer(
			model.as_ref(),
			&mut rand::thread_rng(),
			&InferenceRequest {
				prompt: llm::Prompt::Text(&request.prompt),
				parameters: &inference_parameters,
				play_back_previous_tokens: false,
				maximum_token_count: Some(request.max_tokens),
			},
			&mut Default::default(),
			|r| -> Result<_, GenerateError> {
				match r {
					llm::InferenceResponse::PromptToken(t) | llm::InferenceResponse::InferredToken(t) => {
						log::trace!("Output: {t}");
						text += &t;
						std::io::stdout().flush().unwrap();
						Ok(llm::InferenceFeedback::Continue)
					}
					_ => Ok(llm::InferenceFeedback::Continue),
				}
			},
		)
		.unwrap();

	Ok(Json(GenerateResponse { text }))
}
