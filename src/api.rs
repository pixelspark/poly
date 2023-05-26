use axum::{http::StatusCode, response::IntoResponse};
use llm::InferenceParameters;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Deserialize, Clone, Debug)]
#[serde(default)]
pub struct GenerateRequest {
	pub prompt: String,
	pub max_tokens: usize,
	pub batch_size: usize,
	pub repeat_last_n: usize,
	pub repeat_penalty: f32,
	pub temperature: f32,
	pub top_k: usize,
	pub top_p: f32,
}

impl Default for GenerateRequest {
	fn default() -> Self {
		Self {
			prompt: Default::default(),
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

impl From<GenerateRequest> for InferenceParameters {
	fn from(val: GenerateRequest) -> Self {
		InferenceParameters {
			n_batch: val.batch_size,
			top_k: val.top_k,
			top_p: val.top_p,
			repeat_penalty: val.repeat_penalty,
			temperature: val.temperature,
			repetition_penalty_last_n: val.repeat_last_n,
			..Default::default()
		}
	}
}

#[derive(Serialize)]
pub struct GenerateResponse {
	pub text: String,
}

#[derive(Serialize)]
pub struct StatusResponse {
	pub endpoints: Vec<String>,
}

#[derive(Error, Debug)]
pub enum GenerateError {
	#[error("endpont not found: {0}")]
	EndpointNotFound(String),
}

impl IntoResponse for GenerateError {
	fn into_response(self) -> axum::response::Response {
		(StatusCode::INTERNAL_SERVER_ERROR, format!("{}", self)).into_response()
	}
}
