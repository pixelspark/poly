mod api;
mod config;

use api::GenerateError;
use api::GenerateRequest;
use api::GenerateResponse;
use async_stream::stream;
use axum::extract::Path;
use axum::extract::State;
use axum::response::sse::Event;
use axum::response::Sse;
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
	for endpoint in &state.config.endpoints {
		let params = ModelParameters {
			prefer_mmap: true,
			context_size: 512,
			lora_adapters: None,
		};

		let model = llm::load_dynamic(endpoint.architecture, &endpoint.model_path, params, None, |progress| {
			log::debug!("Loading: {progress:#?}");
		})
		.expect("load model");

		state.models.insert(endpoint.name.clone(), model);
		log::info!("Loaded model {}", endpoint.name);
	}

	// Set up API server
	let app = Router::new()
		.route("/:endpoint/live", post(sse_handler))
		.route("/:endpoint/completion", post(completion_handler))
		.with_state(Arc::new(state));
	axum::Server::bind(&bind_address).serve(app.into_make_service()).await.unwrap();
}

async fn sse_handler(
	State(state): State<Arc<ServerState>>,
	Path(endpoint_name): Path<String>,
	Json(request): Json<GenerateRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, GenerateError> {
	println!("New live connection for endpoint '{}'", endpoint_name.as_str());

	if !state.models.contains_key(&endpoint_name) {
		return Err(GenerateError::EndpointNotFound(endpoint_name));
	};

	let (tx, mut rx) = tokio::sync::mpsc::channel(32);

	tokio::spawn(async move {
		let model = state.models.get(&endpoint_name).unwrap();
		let inference_config = InferenceSessionConfig::default();
		let mut session = model.start_session(inference_config);
		let inference_parameters: InferenceParameters = request.clone().into();

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
							print!("{t}");
							let tx = tx.clone();
							tokio::spawn(async move {
								tx.send(t).await.unwrap();
							});
							std::io::stdout().flush().unwrap();
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

async fn completion_handler(
	State(state): State<Arc<ServerState>>,
	Path(endpoint_name): Path<String>,
	Json(request): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, GenerateError> {
	let Some(model) = state.models.get(&endpoint_name) else {
		return Err(GenerateError::EndpointNotFound(endpoint_name));
	};

	let inference_config = InferenceSessionConfig::default();
	let mut session = model.start_session(inference_config);
	let inference_parameters = request.clone().into();
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
