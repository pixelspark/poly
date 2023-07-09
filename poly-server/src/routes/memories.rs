use std::sync::Arc;

use axum::{
	extract::{Path, State},
	http::{Request, StatusCode},
	middleware::Next,
	response::IntoResponse,
	routing::get,
	Extension, Json, Router,
};
use poly_backend::types::MemoriesResponse;

use crate::{api::JwtClaims, middleware::Server};

pub fn router() -> Router<Arc<Server>, axum::body::Body> {
	Router::new().route("/", get(memories_handler)).nest(
		"/:memory",
		Router::new()
			// 		.route("/chat", get(ws_task_handler))
			// 		.route("/status", get(status_with_user_handler))
			// 		.route("/live", get(sse_task_handler))
			// 		.route("/completion", post(post_task_completion_handler))
			// 		.route("/completion", get(get_task_completion_handler)),
			.layer(axum::middleware::from_fn(authorize)),
	)
}

async fn memories_handler(State(state): State<Arc<Server>>) -> impl IntoResponse {
	Json(MemoriesResponse {
		memories: state.config.backend_config.memories.keys().cloned().collect(),
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
