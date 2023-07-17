use std::{
	convert::Infallible,
	sync::{
		atomic::{AtomicBool, Ordering},
		Arc,
	},
	time::Duration,
};

use async_stream::stream;
use axum::{
	extract::{
		ws::{Message, WebSocket},
		Path, Query, State, WebSocketUpgrade,
	},
	http::{Request, StatusCode},
	middleware::Next,
	response::{sse::Event, IntoResponse, Sse},
	routing::{get, post},
	Extension, Json, Router,
};
use futures_util::Stream;
use llm::InferenceResponse;
use poly_backend::types::{GenerateResponse, PromptRequest, SessionAndPromptRequest, SessionRequest, Status, StatusResponse, TasksResponse};
use tracing::{debug, trace};

use crate::{
	api::{BackendError, JwtClaims},
	server::Server,
};

pub fn router() -> Router<Arc<Server>, axum::body::Body> {
	Router::new().route("/", get(tasks_handler)).nest(
		"/:task",
		Router::new()
			.route("/chat", get(ws_task_handler))
			.route("/status", get(status_with_user_handler))
			.route("/live", get(sse_task_handler))
			.route("/completion", post(post_task_completion_handler))
			.route("/completion", get(get_task_completion_handler))
			.layer(axum::middleware::from_fn(authorize)),
	)
}

async fn tasks_handler(State(state): State<Arc<Server>>) -> impl IntoResponse {
	Json(TasksResponse {
		tasks: state.config.backend_config.tasks.keys().cloned().collect(),
	})
}

async fn status_with_user_handler(Extension(current_user): Extension<JwtClaims>) -> impl IntoResponse {
	tracing::info!("task request from user {:?}", current_user.sub);
	Json(StatusResponse { status: Status::Ok })
}

async fn get_task_completion_handler(
	State(state): State<Arc<Server>>,
	Path(task_name): Path<String>,
	Query(request): Query<SessionRequest>,
	Query(prompt): Query<PromptRequest>,
) -> Result<Json<GenerateResponse>, BackendError> {
	task_completion_handler(state, task_name, request, prompt).await
}

async fn post_task_completion_handler(
	State(state): State<Arc<Server>>,
	Path(task_name): Path<String>,
	Json(request): Json<SessionAndPromptRequest>,
) -> Result<Json<GenerateResponse>, BackendError> {
	task_completion_handler(state, task_name, request.session, request.prompt).await
}

async fn task_completion_handler(
	state: Arc<Server>,
	task_name: String,
	request: SessionRequest,
	prompt: PromptRequest,
) -> Result<Json<GenerateResponse>, BackendError> {
	tokio::task::spawn_blocking(move || {
		let mut text = String::new();
		state
			.backend
			.start(&task_name, &request, state.backend.clone())?
			.complete(&prompt, |r| -> Result<_, poly_backend::types::BackendError> {
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
	})
	.await
	.unwrap()
}

async fn ws_task_handler(
	ws: WebSocketUpgrade,
	State(state): State<Arc<Server>>,
	Path(task_name): Path<String>,
	Query(request): Query<SessionRequest>,
) -> impl IntoResponse {
	debug!("New websocket connection for task '{}'", task_name.as_str());
	ws.on_upgrade(move |socket| socket_task_handler(socket, state, task_name, request))
}

async fn socket_task_handler(mut ws: WebSocket, state: Arc<Server>, task_name: String, request: SessionRequest) {
	// Spawn a blocking thread
	let (tx_prompt, mut rx_prompt) = tokio::sync::mpsc::channel(16);
	let (tx_response, mut rx_response) = tokio::sync::mpsc::channel::<Result<String, String>>(32);
	let t = tokio::task::spawn_blocking(move || {
		let mut session = state.backend.start(&task_name, &request, state.backend.clone()).unwrap();
		while let Some(prompt) = rx_prompt.blocking_recv() {
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
							tx_prompt.send(prompt).await.unwrap();
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
	});
	t.await.unwrap();
	tracing::info!("WebSocket connection closed");
}

async fn sse_task_handler(
	State(state): State<Arc<Server>>,
	Path(task_name): Path<String>,
	Query(request): Query<SessionRequest>,
	Query(prompt): Query<PromptRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, BackendError> {
	debug!("New live connection for task '{}'", task_name.as_str());

	let (tx, mut rx) = tokio::sync::mpsc::channel(32);
	let active = Arc::new(AtomicBool::new(true));
	let active_clone = active.clone();

	let mut session = state.backend.start(&task_name, &request, state.backend.clone()).unwrap();

	tokio::task::spawn_blocking(move || {
		session.complete(&prompt, |r| -> Result<_, poly_backend::types::BackendError> {
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
		let _guard = Guard{ flag: active };
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

/// Middleware that checks whether the user has access to a certain task.
pub async fn authorize<T>(
	Path(task_name): Path<String>,
	Extension(claims): Extension<JwtClaims>,
	req: Request<T>,
	next: Next<T>,
) -> Result<impl IntoResponse, StatusCode> {
	if let Some(tasks) = &claims.tasks {
		if !tasks.contains(&task_name) {
			return Err(StatusCode::UNAUTHORIZED);
		}
	}

	Ok(next.run(req).await)
}
