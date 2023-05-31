mod api;
mod backend;
mod config;

use crate::backend::Backend;
use api::EmbeddingResponse;
use api::GenerateError;
use api::GenerateRequest;
use api::GenerateResponse;
use api::ModelsResponse;
use api::Status;
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
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use std::{fs::File, io::Read};
use tower::limit::ConcurrencyLimitLayer;
use tower_http::cors::Any;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;
use tower_http::trace::TraceLayer;
use tracing::{debug, info, trace};

#[tokio::main]
async fn main() {
	tracing_subscriber::fmt::init();
	// Read config file
	let args = Args::parse();
	let mut config_file = File::open(args.config_path).expect("open config file");
	let mut config_string = String::new();
	config_file.read_to_string(&mut config_string).expect("read config file");
	let config: Config = toml::from_str(&config_string).unwrap();
	let bind_address: SocketAddr = config.bind_address.parse().unwrap();
	info!("Starting llmd; bind address: {bind_address}",);

	// Set up CORS
	let mut cors_layer = CorsLayer::new();
	if let Some(ref origins) = config.allowed_origins {
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

	let state = Backend::from(config);

	// Set up API server
	let app = Router::new()
		.nest_service("/", ServeDir::new("public"))
		.route("/status", get(status_handler))
		.route("/model", get(models_handler))
		.route("/model/:endpoint/live", get(sse_handler))
		.route("/model/:endpoint/completion", post(post_model_completion_handler))
		.route("/model/:endpoint/completion", get(get_model_completion_handler))
		.route("/model/:endpoint/embedding", post(post_model_embedding_handler))
		.route("/model/:endpoint/embedding", get(get_model_embedding_handler))
		.layer(ConcurrencyLimitLayer::new(state.config.max_concurrent))
		.layer(TraceLayer::new_for_http())
		.layer(cors_layer)
		.with_state(Arc::new(state));

	axum::Server::bind(&bind_address).serve(app.into_make_service()).await.unwrap();
}

async fn status_handler() -> impl IntoResponse {
	Json(StatusResponse { status: Status::Ok })
}

async fn models_handler(State(state): State<Arc<Backend>>) -> impl IntoResponse {
	Json(ModelsResponse {
		models: state.config.endpoints.keys().cloned().collect(),
	})
}

async fn sse_handler(
	State(backend): State<Arc<Backend>>,
	Path(endpoint_name): Path<String>,
	Query(request): Query<GenerateRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, GenerateError> {
	debug!("New live connection for endpoint '{}'", endpoint_name.as_str());

	let (tx, mut rx) = tokio::sync::mpsc::channel(32);

	tokio::spawn(async move {
		backend
			.complete(&endpoint_name, &request, |r| -> Result<_, GenerateError> {
				match r {
					llm::InferenceResponse::InferredToken(t) => {
						trace!("{t}");
						let tx = tx.clone();

						// Do not continue when client has disconnected
						if tx.is_closed() {
							debug!("client has disconnected live session, halting generation");
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
			})
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

async fn get_model_embedding_handler(
	State(state): State<Arc<Backend>>,
	Path(endpoint_name): Path<String>,
	Query(request): Query<GenerateRequest>,
) -> Result<Json<EmbeddingResponse>, GenerateError> {
	embedding_handler(state, &endpoint_name, &request).await
}

async fn post_model_embedding_handler(
	State(state): State<Arc<Backend>>,
	Path(endpoint_name): Path<String>,
	Json(request): Json<GenerateRequest>,
) -> Result<Json<EmbeddingResponse>, GenerateError> {
	embedding_handler(state, &endpoint_name, &request).await
}

async fn get_model_completion_handler(
	State(state): State<Arc<Backend>>,
	Path(endpoint_name): Path<String>,
	Query(request): Query<GenerateRequest>,
) -> Result<Json<GenerateResponse>, GenerateError> {
	completion_handler(state, &endpoint_name, &request).await
}

async fn post_model_completion_handler(
	State(state): State<Arc<Backend>>,
	Path(endpoint_name): Path<String>,
	Json(request): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, GenerateError> {
	completion_handler(state, &endpoint_name, &request).await
}

async fn completion_handler(backend: Arc<Backend>, endpoint_name: &str, request: &GenerateRequest) -> Result<Json<GenerateResponse>, GenerateError> {
	let mut text = String::new();
	backend.complete(endpoint_name, request, |r| -> Result<_, GenerateError> {
		match r {
			llm::InferenceResponse::InferredToken(t) => {
				trace!("Output: {t}");
				text += &t;
				Ok(llm::InferenceFeedback::Continue)
			}
			_ => Ok(llm::InferenceFeedback::Continue),
		}
	})?;
	Ok(Json(GenerateResponse { text }))
}

async fn embedding_handler(backend: Arc<Backend>, endpoint_name: &str, request: &GenerateRequest) -> Result<Json<EmbeddingResponse>, GenerateError> {
	Ok(Json(backend.embedding(endpoint_name, request)?))
}
