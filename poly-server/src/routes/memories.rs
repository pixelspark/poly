use std::sync::Arc;

use axum::{
	extract::{Path, Query, RawBody, State},
	http::{Request, StatusCode},
	middleware::Next,
	response::IntoResponse,
	routing::{get, post},
	Extension, Json, Router,
};
use poly_backend::types::MemoriesResponse;
use serde::{Deserialize, Serialize};

use crate::{
	api::{GenerateError, JwtClaims},
	middleware::Server,
};

pub fn router() -> Router<Arc<Server>, axum::body::Body> {
	Router::new().route("/", get(memories_handler)).nest(
		"/:memory",
		Router::new()
			// 		.route("/chat", get(ws_task_handler))
			// 		.route("/status", get(status_with_user_handler))
			// 		.route("/live", get(sse_task_handler))
			// 		.route("/completion", post(post_task_completion_handler))
			.route("/", get(get_memory_recall_handler))
			.route("/", post(post_memory_remember_handler))
			.layer(axum::middleware::from_fn(authorize)),
	)
}

async fn memories_handler(State(state): State<Arc<Server>>) -> impl IntoResponse {
	Json(MemoriesResponse {
		memories: state.config.backend_config.memories.keys().cloned().collect(),
	})
}

#[derive(Deserialize)]
pub struct RecallRequest {
	pub prompt: String,
	pub n: Option<usize>,
}

#[derive(Serialize)]
pub struct RecallResponse {
	pub memories: Vec<String>,
}

#[derive(Serialize)]
pub struct RememberResponse {}

async fn post_memory_remember_handler(
	State(state): State<Arc<Server>>,
	Path(memory_name): Path<String>,
	RawBody(body): RawBody,
) -> Result<Json<RememberResponse>, GenerateError> {
	let Ok(bytes) = hyper::body::to_bytes(body).await else {
		return Err(poly_backend::types::GenerateError::InvalidDocument.into());
	};

	let data = std::str::from_utf8(&bytes).map_err(|_| poly_backend::types::GenerateError::InvalidDocument)?;
	state.backend.memorize(&memory_name, data).await?;
	Ok(Json(RememberResponse {}))
}

async fn get_memory_recall_handler(
	State(state): State<Arc<Server>>,
	Path(memory_name): Path<String>,
	Query(request): Query<RecallRequest>,
) -> Result<Json<RecallResponse>, GenerateError> {
	memory_recall_handler(state, &memory_name, request).await.map(Json)
}

async fn memory_recall_handler(state: Arc<Server>, memory_name: &str, request: RecallRequest) -> Result<RecallResponse, GenerateError> {
	let backend = state.backend.clone();
	Ok(RecallResponse {
		memories: backend.recall(memory_name, &request.prompt, request.n.unwrap_or(1)).await?,
	})
}

/// Middleware that checks whether the user has access to a certain model.
pub async fn authorize<T>(
	Path(memory_name): Path<String>,
	Extension(claims): Extension<JwtClaims>,
	req: Request<T>,
	next: Next<T>,
) -> Result<impl IntoResponse, StatusCode> {
	if let Some(memories) = &claims.memories {
		if !memories.contains(&memory_name) {
			return Err(StatusCode::UNAUTHORIZED);
		}
	}

	Ok(next.run(req).await)
}
