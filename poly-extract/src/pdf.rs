/// Retrieve plain text from a Word DOCX file
pub fn get_text_from_pdf(bytes: &[u8]) -> Option<String> {
	match std::panic::catch_unwind(move || pdf_extract::extract_text_from_mem(bytes).ok()) {
		Ok(output) => output,
		Err(err) => {
			tracing::debug!("error reading pdf: {:?}", err);
			None
		}
	}
}
