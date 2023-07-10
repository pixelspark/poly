use std::sync::Arc;

use axum::{
	async_trait,
	extract::{FromRequest, Query, State},
	http::{header::AUTHORIZATION, Request, StatusCode},
	middleware::Next,
	response::{IntoResponse, Response},
};
use hyper::header::CONTENT_TYPE;
use jsonwebtoken::Validation;

use crate::{
	api::{JwtClaims, KeyQuery},
	config::Config,
};

use poly_backend::backend::Backend;

pub struct Server {
	pub backend: Arc<Backend>,
	pub config: Config,
}

/// Middleware that authenticates a user using static pre-shared API keys or a JWT
pub async fn authenticate<T>(
	State(state): State<Arc<Server>>,
	Query(key): Query<KeyQuery>,
	mut req: Request<T>,
	next: Next<T>,
) -> Result<impl IntoResponse, (StatusCode, &'static str)> {
	// Obtain contents of the Authorization header
	let auth_header = req
		.headers()
		.get(AUTHORIZATION)
		.and_then(|header| header.to_str().ok())
		.map(|s| s.to_string());

	// If there was an Authorization header, extract the Bearer token (if any)
	let auth_token = if let Some(auth_header) = auth_header {
		Some(
			auth_header
				.strip_prefix("Bearer ")
				.ok_or((StatusCode::UNAUTHORIZED, "invalid bearer token"))
				.map(|x| x.to_string())?,
		)
	} else if key.api_key.as_ref().is_some_and(|s| !s.is_empty()) {
		key.api_key
	} else {
		None
	};

	let claims: JwtClaims = match auth_token {
		Some(auth_token) => {
			// Check if key is allowed
			if state.config.allowed_keys.contains(&auth_token) {
				// OK
				JwtClaims {
					sub: Some(auth_token),
					..Default::default()
				}
			} else if let Some(jwt_key) = &state.config.jwt_private_key {
				// Attempt to decode and validate JWT token
				let mut validation = Validation::new(jwt_key.algorithm());
				validation.validate_nbf = true;

				match jsonwebtoken::decode::<JwtClaims>(&auth_token, &jwt_key.decoding_key(), &validation) {
					Ok(valid_token) => {
						tracing::debug!(sub = valid_token.claims.sub, "valid JWT token");
						valid_token.claims
					}
					Err(e) => {
						tracing::debug!("error validating JWT token: {e}");
						return Err((StatusCode::UNAUTHORIZED, "invalid JWT token"));
					}
				}
			} else {
				return Err((StatusCode::UNAUTHORIZED, "no acceptable auth token provided"));
			}
		}
		None => {
			if !state.config.public {
				return Err((StatusCode::UNAUTHORIZED, "no auth token provided and not a public server"));
			}

			// Unauthenticated but access granted
			JwtClaims::default()
		}
	};

	req.extensions_mut().insert(claims);

	Ok(next.run(req).await)
}

/// Extractor that converts various body file types to plain text string
pub struct Plaintext(pub String);

#[async_trait]
impl<S> FromRequest<S, axum::body::Body> for Plaintext
where
	S: Send + Sync,
{
	type Rejection = Response;

	async fn from_request(mut req: Request<axum::body::Body>, _state: &S) -> Result<Self, Self::Rejection> {
		let content_type_header = req.headers().get(CONTENT_TYPE).cloned();
		let content_type = content_type_header.and_then(|value| value.to_str().map(|x| x.to_string()).ok());

		if let Some(content_type) = content_type {
			if content_type.starts_with("text/plain") {
				let Ok(bytes) = hyper::body::to_bytes(req.body_mut()).await else {
					return Err(StatusCode::UNPROCESSABLE_ENTITY.into_response());
				};

				return Ok(Self(
					std::str::from_utf8(&bytes)
						.map_err(|_| StatusCode::UNPROCESSABLE_ENTITY.into_response())?
						.to_string(),
				));
			} else {
				tracing::warn!("invalid content type: {content_type}");
			}
		}

		Err(StatusCode::UNSUPPORTED_MEDIA_TYPE.into_response())
	}
}
