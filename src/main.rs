mod api;
mod backend;
pub mod bias;
mod config;

use crate::backend::Backend;
use api::EmbeddingResponse;
use api::GenerateError;
use api::GenerateResponse;
use api::KeyQuery;
use api::ModelsResponse;
use api::PromptRequest;
use api::SessionAndPromptRequest;
use api::SessionRequest;
use api::Status;
use api::StatusResponse;
use api::TasksResponse;
use async_stream::stream;
use axum::extract::ws::Message;
use axum::extract::ws::WebSocket;
use axum::extract::ws::WebSocketUpgrade;
use axum::extract::Path;
use axum::extract::Query;
use axum::extract::State;
use axum::http;
use axum::http::header::CONTENT_TYPE;
use axum::http::HeaderValue;
use axum::http::Method;
use axum::http::Request;
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::sse::Event;
use axum::response::IntoResponse;
use axum::response::Sse;
use axum::routing::get;
use axum::routing::post;
use axum::Extension;
use axum::Json;
use axum::Router;
use clap::Parser;
use config::Args;
use config::Config;
use futures_util::Stream;
use llm::InferenceResponse;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use std::{fs::File, io::Read};
use tower::limit::ConcurrencyLimitLayer;
use tower_http::cors::Any;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;
use tower_http::trace::TraceLayer;
use tracing::{debug, info, trace};

#[derive(Clone)]
struct CurrentUser {
	key: Option<String>,
}

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
	}
	cors_layer = cors_layer.allow_headers([CONTENT_TYPE]);
	cors_layer = cors_layer.allow_methods([Method::GET, Method::POST]);

	let state = Arc::new(Backend::from(config));

	// Set up API server
	let app = Router::new()
		.nest_service("/", ServeDir::new("client/dist/"))
		.route("/status", get(status_handler))
		.nest(
			"/v1",
			Router::new()
				.nest(
					"/model",
					Router::new()
						.route("/", get(models_handler))
						.route("/:model/embedding", post(post_model_embedding_handler))
						.route("/:model/embedding", get(get_model_embedding_handler))
						.layer(axum::middleware::from_fn_with_state(state.clone(), authorize)),
				)
				.nest(
					"/task",
					Router::new()
						.route("/", get(tasks_handler))
						.route("/:task/chat", get(ws_task_handler))
						.route("/:task/status", get(status_with_user_handler))
						.route("/:task/live", get(sse_task_handler))
						.route("/:task/completion", post(post_task_completion_handler))
						.route("/:task/completion", get(get_task_completion_handler))
						.layer(axum::middleware::from_fn_with_state(state.clone(), authorize)),
				),
		)
		.layer(cors_layer)
		.layer(ConcurrencyLimitLayer::new(state.config.max_concurrent))
		.layer(TraceLayer::new_for_http())
		.with_state(state);

	axum::Server::bind(&bind_address).serve(app.into_make_service()).await.unwrap();
}

async fn status_handler() -> impl IntoResponse {
	Json(StatusResponse { status: Status::Ok })
}

async fn status_with_user_handler(Extension(current_user): Extension<CurrentUser>) -> impl IntoResponse {
	tracing::info!("task request from user {:?}", current_user.key);
	Json(StatusResponse { status: Status::Ok })
}

async fn models_handler(State(state): State<Arc<Backend>>) -> impl IntoResponse {
	Json(ModelsResponse {
		models: state.config.models.keys().cloned().collect(),
	})
}

async fn tasks_handler(State(state): State<Arc<Backend>>) -> impl IntoResponse {
	Json(TasksResponse {
		tasks: state.config.tasks.keys().cloned().collect(),
	})
}

async fn ws_task_handler(
	ws: WebSocketUpgrade,
	State(backend): State<Arc<Backend>>,
	Path(task_name): Path<String>,
	Query(request): Query<SessionRequest>,
) -> impl IntoResponse {
	debug!("New websocket connection for task '{}'", task_name.as_str());
	ws.on_upgrade(move |socket| socket_task_handler(socket, backend, task_name, request))
}

async fn socket_task_handler(mut ws: WebSocket, backend: Arc<Backend>, task_name: String, request: SessionRequest) {
	// Spawn a blocking thread
	let (tx_prompt, rx_prompt) = std::sync::mpsc::channel();
	let (tx_response, mut rx_response) = tokio::sync::mpsc::channel::<Result<String, String>>(32);
	thread::spawn(move || {
		let mut session = backend.start(&task_name, &request).unwrap();
		while let Ok(prompt) = rx_prompt.recv() {
			let prompt_request = PromptRequest { prompt };
			let res = session.complete(&prompt_request, |r| match r {
				InferenceResponse::InferredToken(token) => {
					if tx_response.blocking_send(Ok(token)).is_err() {
						// Connection is likely closed
						return Ok(llm::InferenceFeedback::Halt);
					}
					Ok(llm::InferenceFeedback::Continue)
				}
				InferenceResponse::EotToken => Ok(llm::InferenceFeedback::Halt),
				InferenceResponse::PromptToken(_) | InferenceResponse::SnapshotToken(_) => Ok(llm::InferenceFeedback::Continue),
			});

			match res {
				Ok(_) => {
					// Send empty token to signal this cycle has ended
					if tx_response.blocking_send(Ok("".to_string())).is_err() {
						// Output channel was probably dropped
						break;
					}
				}
				Err(e) => {
					if tx_response.blocking_send(Err(e.to_string())).is_err() {
						// Output channel was probably dropped
						break;
					}
				}
			}
		}
		tracing::info!("ending model thread");
	});

	tokio::spawn(async move {
		loop {
			tokio::select! {
				msg = ws.recv() => {
					let Some(msg) = msg else {
						// WebSocket closed?
						break;
					};

					match msg.unwrap() {
						Message::Text(prompt) => {
							tracing::trace!("WebSocket receive prompt text: {prompt}");
							tx_prompt.send(prompt).unwrap();
						},
						Message::Close(_close_frame) => {
							_ = ws.close().await;
							break;
						},
						Message::Binary(_) => {
							// Invalid binary message
							_ = ws.close().await;
							break;
						},
						Message::Ping(p) => {
							_ = ws.send(Message::Pong(p)).await;
						},
						Message::Pong(_) => {},
					}
				},
				response = rx_response.recv() => {
					match response.unwrap() {
						Ok(txt) => {
							if let Err(e) = ws.send(Message::Text(txt)).await {
								tracing::error!("WebSocket: send reported error: {e}");
									break;
							}
						},
						Err(e) => {
							tracing::error!("WebSocket: backend thread reported error: {e}");
							break;
						}
					}

				}
			}
		}
		tracing::info!("WebSocket connection closed");
	});
}

async fn sse_task_handler(
	State(backend): State<Arc<Backend>>,
	Path(task_name): Path<String>,
	Query(request): Query<SessionRequest>,
	Query(prompt): Query<PromptRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, GenerateError> {
	debug!("New live connection for task '{}'", task_name.as_str());

	let (tx, mut rx) = tokio::sync::mpsc::channel(32);
	let active = Arc::new(AtomicBool::new(true));
	let active_clone = active.clone();

	tokio::spawn(async move {
		backend
			.start(&task_name, &request)
			.unwrap()
			.complete(&prompt, |r| -> Result<_, GenerateError> {
				match r {
					llm::InferenceResponse::InferredToken(t) => {
						trace!("{t}");
						let tx = tx.clone();

						// Do not continue when client has disconnected
						if tx.is_closed() || !active_clone.load(Ordering::SeqCst) {
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

	struct Guard {
		flag: Arc<AtomicBool>,
	}
	impl Drop for Guard {
		fn drop(&mut self) {
			tracing::info!("SSE disconnected");
			self.flag.store(false, Ordering::SeqCst);
		}
	}

	let stream = stream! {
		let _guard = Guard{flag: active};
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
	Query(request): Query<SessionRequest>,
	Query(prompt): Query<PromptRequest>,
) -> Result<Json<EmbeddingResponse>, GenerateError> {
	embedding_handler(state, &endpoint_name, &request, &prompt).await
}

async fn post_model_embedding_handler(
	State(state): State<Arc<Backend>>,
	Path(endpoint_name): Path<String>,
	Json(request): Json<SessionAndPromptRequest>,
) -> Result<Json<EmbeddingResponse>, GenerateError> {
	embedding_handler(state, &endpoint_name, &request.session, &request.prompt).await
}

async fn get_task_completion_handler(
	State(state): State<Arc<Backend>>,
	Path(task_name): Path<String>,
	Query(request): Query<SessionRequest>,
	Query(prompt): Query<PromptRequest>,
) -> Result<Json<GenerateResponse>, GenerateError> {
	task_completion_handler(state, &task_name, &request, &prompt).await
}

async fn post_task_completion_handler(
	State(state): State<Arc<Backend>>,
	Path(task_name): Path<String>,
	Json(request): Json<SessionAndPromptRequest>,
) -> Result<Json<GenerateResponse>, GenerateError> {
	task_completion_handler(state, &task_name, &request.session, &request.prompt).await
}

async fn task_completion_handler(
	backend: Arc<Backend>,
	task_name: &str,
	request: &SessionRequest,
	prompt: &PromptRequest,
) -> Result<Json<GenerateResponse>, GenerateError> {
	let mut text = String::new();
	backend.start(task_name, request)?.complete(prompt, |r| -> Result<_, GenerateError> {
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

async fn embedding_handler(
	backend: Arc<Backend>,
	endpoint_name: &str,
	request: &SessionRequest,
	prompt: &PromptRequest,
) -> Result<Json<EmbeddingResponse>, GenerateError> {
	Ok(Json(backend.embedding(endpoint_name, request, prompt)?))
}

async fn authorize<T>(
	State(state): State<Arc<Backend>>,
	Query(key): Query<KeyQuery>,
	mut req: Request<T>,
	next: Next<T>,
) -> Result<impl IntoResponse, StatusCode> {
	if !state.config.allowed_keys.is_empty() {
		let auth_header = req
			.headers()
			.get(http::header::AUTHORIZATION)
			.and_then(|header| header.to_str().ok())
			.map(|s| s.to_string());

		let auth_header = if let Some(auth_header) = auth_header {
			auth_header
				.strip_prefix("Bearer ")
				.ok_or(StatusCode::UNAUTHORIZED)
				.map(|x| x.to_string())?
		} else {
			match key.api_key {
				Some(k) => k,
				None => return Err(StatusCode::UNAUTHORIZED),
			}
		};

		// Check if key is allowed
		if !state.config.allowed_keys.contains(&auth_header) {
			return Err(StatusCode::UNAUTHORIZED);
		}

		req.extensions_mut().insert(CurrentUser { key: Some(auth_header) });
	} else {
		req.extensions_mut().insert(CurrentUser { key: None });
	}

	Ok(next.run(req).await)
}
