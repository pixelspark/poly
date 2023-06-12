use std::{collections::HashMap, sync::Arc};

use axum::{http::StatusCode, response::IntoResponse};
use llm::{samplers::TopPTopK, InferenceError, InferenceParameters, TokenBias, TokenizationError};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{config::TaskConfig, stats::TaskStats};

#[derive(Deserialize, Clone, Debug)]
pub struct KeyQuery {
	pub api_key: Option<String>,
}

#[derive(Serialize, Clone, Debug)]
pub struct StatsResponse {
	pub tasks: HashMap<String, TaskStats>,
}

#[derive(Deserialize, Clone, Debug, Default)]
#[serde(default)]
pub struct SessionRequest {}

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

#[derive(Serialize, Clone, Debug, Default)]
pub struct EmbeddingResponse {
	pub embedding: Vec<f32>,
}

impl From<TaskConfig> for InferenceParameters {
	fn from(val: TaskConfig) -> Self {
		let sampler = TopPTopK {
			top_k: val.top_k,
			top_p: val.top_p,
			repeat_penalty: val.repeat_penalty,
			temperature: val.temperature,
			repetition_penalty_last_n: val.repetition_penalty_last_n,
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
