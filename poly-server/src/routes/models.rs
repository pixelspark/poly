use std::sync::Arc;

use axum::{
	extract::{Path, Query, State},
	http::{Request, StatusCode},
	middleware::Next,
	response::IntoResponse,
	routing::{get, post},
	Extension, Json, Router,
};
use poly_backend::types::{EmbeddingResponse, ModelsResponse, PromptRequest, SessionAndPromptRequest, SessionRequest, TokenizationResponse};

use crate::{
	api::{BackendError, JwtClaims},
	server::Server,
};

pub fn router() -> Router<Arc<Server>, axum::body::Body> {
	Router::new().route("/", get(models_handler)).nest(
		"/:model",
		Router::new()
			.route("/embedding", post(post_model_embedding_handler))
			.route("/embedding", get(get_model_embedding_handler))
			.route("/tokenization", post(post_model_tokenize_handler))
			.route("/tokenization", get(get_model_tokenize_handler))
			.layer(axum::middleware::from_fn(authorize)),
	)
}

async fn models_handler(State(state): State<Arc<Server>>) -> impl IntoResponse {
	Json(ModelsResponse {
		models: state.config.backend_config.models.keys().cloned().collect(),
	})
}

async fn get_model_embedding_handler(
	State(state): State<Arc<Server>>,
	Path(endpoint_name): Path<String>,
	Query(request): Query<SessionAndPromptRequest>,
) -> Result<Json<EmbeddingResponse>, BackendError> {
	let SessionAndPromptRequest { session, prompt } = request;
	embedding_handler(state, &endpoint_name, &session, &prompt)
}

async fn post_model_embedding_handler(
	State(state): State<Arc<Server>>,
	Path(endpoint_name): Path<String>,
	Json(request): Json<SessionAndPromptRequest>,
) -> Result<Json<EmbeddingResponse>, BackendError> {
	let SessionAndPromptRequest { session, prompt } = request;
	embedding_handler(state, &endpoint_name, &session, &prompt)
}

fn embedding_handler(
	state: Arc<Server>,
	endpoint_name: &str,
	_request: &SessionRequest,
	prompt: &PromptRequest,
) -> Result<Json<EmbeddingResponse>, BackendError> {
	Ok(Json(state.backend.embedding(endpoint_name, prompt)?))
}

async fn get_model_tokenize_handler(
	State(state): State<Arc<Server>>,
	Path(endpoint_name): Path<String>,
	Query(request): Query<SessionAndPromptRequest>,
) -> Result<Json<TokenizationResponse>, BackendError> {
	let SessionAndPromptRequest { session, prompt } = request;
	tokenize_handler(state, &endpoint_name, &session, &prompt)
}

async fn post_model_tokenize_handler(
	State(state): State<Arc<Server>>,
	Path(endpoint_name): Path<String>,
	Json(request): Json<SessionAndPromptRequest>,
) -> Result<Json<TokenizationResponse>, BackendError> {
	let SessionAndPromptRequest { session, prompt } = request;
	tokenize_handler(state, &endpoint_name, &session, &prompt)
}

fn tokenize_handler(
	state: Arc<Server>,
	endpoint_name: &str,
	_request: &SessionRequest,
	prompt: &PromptRequest,
) -> Result<Json<TokenizationResponse>, BackendError> {
	Ok(Json(state.backend.tokenize(endpoint_name, prompt)?))
}

/// Middleware that checks whether the user has access to a certain model.
pub async fn authorize<T>(
	Path(model_name): Path<String>,
	Extension(claims): Extension<JwtClaims>,
	req: Request<T>,
	next: Next<T>,
) -> Result<impl IntoResponse, StatusCode> {
	if let Some(models) = &claims.models {
		if !models.contains(&model_name) {
			return Err(StatusCode::UNAUTHORIZED);
		}
	}

	Ok(next.run(req).await)
}
