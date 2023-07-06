use std::sync::Arc;

use axum::{
	extract::{Path, Query, State},
	http::{header::AUTHORIZATION, Request, StatusCode},
	middleware::Next,
	response::IntoResponse,
	Extension,
};
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

/// Middleware that checks whether the user has access to a certain task.
pub async fn authorize<T>(
	Path(task_name): Path<String>,
	Extension(claims): Extension<JwtClaims>,
	req: Request<T>,
	next: Next<T>,
) -> Result<impl IntoResponse, StatusCode> {
	if let Some(tasks) = &claims.tasks {
		if !tasks.contains(&task_name) {
			return Err(StatusCode::UNAUTHORIZED);
		}
	}

	Ok(next.run(req).await)
}

/// Middleware that authenticates a user using static pre-shared API keys or a JWT
pub async fn authenticate<T>(
	State(state): State<Arc<Server>>,
	Query(key): Query<KeyQuery>,
	mut req: Request<T>,
	next: Next<T>,
) -> Result<impl IntoResponse, StatusCode> {
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
				.ok_or(StatusCode::UNAUTHORIZED)
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
						return Err(StatusCode::UNAUTHORIZED);
					}
				}
			} else {
				return Err(StatusCode::UNAUTHORIZED);
			}
		}
		None => {
			if !state.config.public {
				return Err(StatusCode::UNAUTHORIZED);
			}

			// Unauthenticated but access granted
			JwtClaims::default()
		}
	};

	req.extensions_mut().insert(claims);

	Ok(next.run(req).await)
}
