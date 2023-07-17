use axum::{http::StatusCode, response::IntoResponse};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use poly_backend::stats::TaskStats;
use poly_backend::types::BackendError as OriginalGenerateError;

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct JwtClaims {
	pub exp: Option<usize>,            // Expiry time
	pub sub: Option<String>,           // User identifier (currently only used for logging)
	pub tasks: Option<Vec<String>>,    // Optional list of tasks this token is allowed to use
	pub models: Option<Vec<String>>,   // Optional list of models this token is allowed to use
	pub memories: Option<Vec<String>>, // Optional list of memories this token is allowed to use
}

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

trait ToStatusCode {
	fn status_code(&self) -> StatusCode;
}

pub struct BackendError(OriginalGenerateError);

impl BackendError {
	fn status_code(&self) -> StatusCode {
		match self.0 {
			OriginalGenerateError::TaskNotFound(_) | OriginalGenerateError::ModelNotFound(_) | OriginalGenerateError::MemoryNotFound(_) => {
				StatusCode::NOT_FOUND
			}
			OriginalGenerateError::InferenceError(_) | OriginalGenerateError::TokenizationError(_) => StatusCode::INTERNAL_SERVER_ERROR,
			OriginalGenerateError::Memory(_) => StatusCode::INTERNAL_SERVER_ERROR,
			OriginalGenerateError::IllegalToken | OriginalGenerateError::InvalidDocument => StatusCode::BAD_REQUEST,
			OriginalGenerateError::InvalidChunkSeparator(_) => StatusCode::INTERNAL_SERVER_ERROR,
		}
	}
}

impl IntoResponse for BackendError {
	fn into_response(self) -> axum::response::Response {
		(self.status_code(), format!("{}", self.0)).into_response()
	}
}

impl From<OriginalGenerateError> for BackendError {
	fn from(t: OriginalGenerateError) -> BackendError {
		BackendError(t)
	}
}
