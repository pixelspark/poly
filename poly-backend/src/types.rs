use llm::{samplers::TopPTopK, InferenceError, InferenceParameters, TokenBias, TokenizationError};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
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

	// llm_base::InferenceError is not Send
	#[error("inference error: {0}")]
	InferenceError(String),

	#[error("tokenization error: {0}")]
	TokenizationError(#[from] TokenizationError),

	#[error("illegal token encountered")]
	IllegalToken,

	#[error("memory error: {0}")]
	Memory(#[from] MemoryError),
}

impl From<InferenceError> for GenerateError {
	fn from(e: InferenceError) -> GenerateError {
		GenerateError::InferenceError(e.to_string())
	}
}
