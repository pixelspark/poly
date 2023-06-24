use async_stream::stream;
use axum::extract::{ws::Message, ws::WebSocket, ws::WebSocketUpgrade, Path, Query, State};
use axum::http::header::{AUTHORIZATION, CONTENT_TYPE};
use axum::http::{HeaderValue, Method, StatusCode};
use axum::response::sse::Event;
use axum::response::{IntoResponse, Sse};
use axum::routing::{get, post};
use axum::{Extension, Json, Router};
use clap::Parser;
use futures_util::Stream;
use llm::InferenceResponse;
use llmd::api::{
	EmbeddingResponse, GenerateError, GenerateResponse, JwtClaims, ModelsResponse, PromptRequest, SessionAndPromptRequest, SessionRequest,
	StatsResponse, Status, StatusResponse, TasksResponse,
};
use llmd::backend::Backend;
use llmd::config::{Args, Config};
use llmd::middleware::{authenticate, authorize};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use std::{fs::File, io::Read};
use tower::limit::ConcurrencyLimitLayer;
use tower_http::cors::{Any, CorsLayer};
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
	}
	cors_layer = cors_layer.allow_headers([CONTENT_TYPE, AUTHORIZATION]);
	cors_layer = cors_layer.allow_methods([Method::GET, Method::POST, Method::OPTIONS]);

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
						.layer(axum::middleware::from_fn_with_state(state.clone(), authenticate)),
				)
				.nest(
					"/task",
					Router::new()
						.route("/", get(tasks_handler))
						.nest(
							"/:task",
							Router::new()
								.route("/chat", get(ws_task_handler))
								.route("/status", get(status_with_user_handler))
								.route("/live", get(sse_task_handler))
								.route("/completion", post(post_task_completion_handler))
								.route("/completion", get(get_task_completion_handler))
								.layer(axum::middleware::from_fn(authorize)),
						)
						.layer(axum::middleware::from_fn_with_state(state.clone(), authenticate)),
				)
				.route("/stats", get(stats_handler)),
		)
		.fallback(handler_not_found)
		.layer(cors_layer)
		.layer(ConcurrencyLimitLayer::new(state.config.max_concurrent))
		.layer(TraceLayer::new_for_http())
		.with_state(state);

	axum::Server::bind(&bind_address).serve(app.into_make_service()).await.unwrap();
}

async fn stats_handler(State(state): State<Arc<Backend>>) -> impl IntoResponse {
	let task_stats = state.stats.task_stats.lock().unwrap().clone();
	Json(StatsResponse { tasks: task_stats })
}

async fn status_handler() -> impl IntoResponse {
	Json(StatusResponse { status: Status::Ok })
}

async fn status_with_user_handler(Extension(current_user): Extension<JwtClaims>) -> impl IntoResponse {
	tracing::info!("task request from user {:?}", current_user.sub);
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

async fn handler_not_found() -> impl IntoResponse {
	(StatusCode::NOT_FOUND, "not found")
}

async fn embedding_handler(
	backend: Arc<Backend>,
	endpoint_name: &str,
	_request: &SessionRequest,
	prompt: &PromptRequest,
) -> Result<Json<EmbeddingResponse>, GenerateError> {
	Ok(Json(backend.embedding(endpoint_name, prompt)?))
}
