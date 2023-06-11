use llm::{TokenId, Vocabulary};

pub mod json;
pub mod sampler;

pub const TOKEN_ALLOWED: f32 = 10000.0;
pub const TOKEN_FORBIDDEN: f32 = -10000.0;

pub trait Biaser {
	/// Return the current set of token biases
	fn bias(&self, vocabulary: &Vocabulary, eot_token: TokenId) -> Vec<(TokenId, f32)>;

	/// Advance the biaser by feeding it a single next token (must be one of the tokens allowed as described by the
	/// result of a call to `bias`)
	fn advance(&mut self, vocabulary: &Vocabulary, token: TokenId);
}

pub struct NullBiaser {}

impl Biaser for NullBiaser {
	fn bias(&self, _vocabulary: &Vocabulary, _eot_token: TokenId) -> Vec<(TokenId, f32)> {
		vec![]
	}

	fn advance(&mut self, _vocabulary: &Vocabulary, _token: TokenId) {}
}
