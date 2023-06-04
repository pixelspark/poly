use std::sync::Arc;

use axum::{http::StatusCode, response::IntoResponse};
use llm::{samplers::TopPTopK, InferenceError, InferenceParameters, TokenBias, TokenizationError};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Deserialize, Clone, Debug)]
pub struct KeyQuery {
	pub api_key: Option<String>,
}

#[derive(Deserialize, Clone, Debug)]
#[serde(default)]
pub struct SessionRequest {
	pub max_tokens: usize,
	pub batch_size: usize,
	pub repeat_last_n: usize,
	pub repeat_penalty: f32,
	pub temperature: f32,
	pub top_k: usize,
	pub top_p: f32,
}

#[derive(Deserialize, Clone, Debug)]
pub struct PromptRequest {
	pub prompt: String,
}

#[derive(Deserialize, Clone, Debug)]
pub struct SessionAndPromptRequest {
	#[serde(flatten)]
	pub session: SessionRequest,

	#[serde(flatten)]
	pub prompt: PromptRequest,
}

#[derive(Serialize, Clone, Debug)]
pub struct EmbeddingResponse {
	pub embedding: Vec<f32>,
}

impl Default for SessionRequest {
	fn default() -> Self {
		Self {
			max_tokens: 128,
			batch_size: 8,
			repeat_last_n: 64,
			repeat_penalty: 1.30,
			temperature: 0.80,
			top_k: 40,
			top_p: 0.95,
		}
	}
}

impl From<SessionRequest> for InferenceParameters {
	fn from(val: SessionRequest) -> Self {
		let sampler = TopPTopK {
			top_k: val.top_k,
			top_p: val.top_p,
			repeat_penalty: val.repeat_penalty,
			temperature: val.temperature,
			repetition_penalty_last_n: val.repeat_last_n,
			bias_tokens: TokenBias::empty(),
		};

		InferenceParameters {
			n_batch: val.batch_size,
			sampler: Arc::new(sampler),
			..Default::default()
		}
	}
}

#[derive(Serialize)]
pub struct ModelsResponse {
	pub models: Vec<String>,
}

#[derive(Serialize)]
pub struct TasksResponse {
	pub tasks: Vec<String>,
}

#[derive(Serialize)]
pub struct GenerateResponse {
	pub text: String,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
	Ok,
}

#[derive(Serialize)]
pub struct StatusResponse {
	pub status: Status,
}

#[derive(Error, Debug)]
pub enum GenerateError {
	#[error("task not found: {0}")]
	TaskNotFound(String),

	#[error("model not found: {0}")]
	ModelNotFound(String),

	#[error("inference error: {0}")]
	InferenceError(#[from] InferenceError),

	#[error("tokenization error: {0}")]
	TokenizationError(#[from] TokenizationError),

	#[error("illegal token encountered")]
	IllegalToken,
}

impl GenerateError {
	fn status_code(&self) -> StatusCode {
		match self {
			GenerateError::TaskNotFound(_) | GenerateError::ModelNotFound(_) => StatusCode::NOT_FOUND,
			GenerateError::InferenceError(_) | GenerateError::TokenizationError(_) => StatusCode::INTERNAL_SERVER_ERROR,
			GenerateError::IllegalToken => StatusCode::BAD_REQUEST,
		}
	}
}

impl IntoResponse for GenerateError {
	fn into_response(self) -> axum::response::Response {
		(self.status_code(), format!("{}", self)).into_response()
	}
}
