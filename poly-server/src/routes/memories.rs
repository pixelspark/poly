use std::sync::Arc;

use axum::{
	extract::{Path, Query, State},
	http::{Request, StatusCode},
	middleware::Next,
	response::IntoResponse,
	routing::{get, post, put},
	Extension, Json, Router,
};
use poly_backend::types::MemoriesResponse;
use poly_extract::middleware::Plaintext;
use serde::{Deserialize, Serialize};

use crate::{
	api::{GenerateError, JwtClaims},
	server::{IngestItem, Server},
};

pub fn router() -> Router<Arc<Server>, axum::body::Body> {
	Router::new().route("/", get(memories_handler)).nest(
		"/:memory",
		Router::new()
			.route("/", get(get_memory_recall_handler))
			.route("/", post(post_memory_recall_handler))
			.route("/", put(put_memory_ingest_handler))
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
	pub chunks: Vec<String>,
}

#[derive(Serialize)]
pub struct RememberResponse {}

#[derive(Deserialize)]
pub struct IngestRequest {
	#[serde(default = "default_wait")]
	pub wait: bool,
}

const fn default_wait() -> bool {
	true
}

async fn put_memory_ingest_handler(
	State(state): State<Arc<Server>>,
	Path(memory_name): Path<String>,
	Query(params): Query<IngestRequest>,
	Plaintext(body): Plaintext,
) -> Result<Json<RememberResponse>, GenerateError> {
	if params.wait {
		state.backend.memorize(&memory_name, &body).await?;
	} else {
		// Defer to a background job
		state
			.ingest(IngestItem {
				memory_name,
				plaintext: body,
			})
			.await;
	}
	Ok(Json(RememberResponse {}))
}

async fn post_memory_recall_handler(
	State(state): State<Arc<Server>>,
	Path(memory_name): Path<String>,
	Json(request): Json<RecallRequest>,
) -> Result<Json<RecallResponse>, GenerateError> {
	memory_recall_handler(state, &memory_name, request).await.map(Json)
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
		chunks: backend.recall(memory_name, &request.prompt, request.n.unwrap_or(1)).await?,
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
