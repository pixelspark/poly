use llm::{InferenceError, InferenceParameters, TokenId, TokenizationError};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use thiserror::Error;

use crate::{config::TaskConfig, memory::MemoryError};

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

#[derive(Serialize, Clone, Debug, Default)]
pub struct TokenizationResponse {
	pub tokens: Vec<TokenResponse>,
}

#[derive(Serialize, Clone, Debug, Default)]
pub struct TokenResponse {
	pub text: String,
	pub token: TokenId,
}

impl From<TaskConfig> for InferenceParameters {
	fn from(val: TaskConfig) -> Self {
		InferenceParameters {
			sampler: Arc::new(Mutex::new(val.sampler_chain())),
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
pub struct MemoriesResponse {
	pub memories: Vec<String>,
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
pub enum BackendError {
	#[error("task not found: {0}")]
	TaskNotFound(String),

	#[error("model not found: {0}")]
	ModelNotFound(String),

	// llm_base::InferenceError is not Send
	#[error("inference error: {0}")]
	InferenceError(String),

	#[error("tokenization error: {0}")]
	TokenizationError(#[from] TokenizationError),

	#[error("illegal token encountered")]
	IllegalToken,

	#[error("memory error: {0}")]
	Memory(#[from] MemoryError),

	#[error("memory not found: {0}")]
	MemoryNotFound(String),

	#[error("invalid document supplied")]
	InvalidDocument,

	#[error("chunk separator '{0}' invalid: must consist of exactly one token")]
	InvalidChunkSeparator(String),
}

impl From<InferenceError> for BackendError {
	fn from(e: InferenceError) -> BackendError {
		BackendError::InferenceError(e.to_string())
	}
}
