use axum::{
	async_trait,
	extract::FromRequest,
	http::{header::CONTENT_TYPE, Request, StatusCode},
	response::IntoResponse,
};

/// Extractor that converts various body file types to plain text string
pub struct Plaintext(pub String);

#[async_trait]
impl<S> FromRequest<S, axum::body::Body> for Plaintext
where
	S: Send + Sync,
{
	type Rejection = axum::response::Response;

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
			} else if content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" {
				let Ok(bytes) = hyper::body::to_bytes(req.body_mut()).await else {
					return Err(StatusCode::UNPROCESSABLE_ENTITY.into_response());
				};
				let text = tokio::task::spawn_blocking(|| {
					let mut cur = std::io::Cursor::new(bytes);
					crate::docx::get_text_from_docx(&mut cur)
				})
				.await
				.unwrap();

				match text {
					Some(text) => return Ok(Self(text)),
					None => return Err(StatusCode::UNPROCESSABLE_ENTITY.into_response()),
				}
			} else if content_type == "application/pdf" {
				let Ok(bytes) = hyper::body::to_bytes(req.body_mut()).await else {
					return Err(StatusCode::UNPROCESSABLE_ENTITY.into_response());
				};
				match crate::pdf::get_text_from_pdf(&bytes) {
					Some(text) => return Ok(Self(text)),
					None => return Err(StatusCode::UNPROCESSABLE_ENTITY.into_response()),
				}
			}
		}

		Err(StatusCode::UNSUPPORTED_MEDIA_TYPE.into_response())
	}
}
