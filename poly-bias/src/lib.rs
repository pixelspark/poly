use llm::{TokenId, Tokenizer};

pub mod json;

/// Logit value to indicate a token is allowed to be present in the result
pub const TOKEN_ALLOWED: f32 = 10000.0;

/// Logit value to indicate a token is forbidden to be present in the result
pub const TOKEN_FORBIDDEN: f32 = -10000.0;

/// An object that holds state for producting biases during inference
pub trait Biaser {
	/// Return the current set of token biases
	fn bias(&self, vocabulary: &Tokenizer, eot_token: TokenId) -> Vec<(TokenId, f32)>;

	/// Advance the biaser by feeding it a single next token (must be one of the tokens allowed as described by the
	/// result of a call to `bias`)
	fn advance(&mut self, vocabulary: &Tokenizer, token: TokenId);
}

/// A biaser that does not bias in any way
pub struct NullBiaser {}

impl Biaser for NullBiaser {
	fn bias(&self, _vocabulary: &Tokenizer, _eot_token: TokenId) -> Vec<(TokenId, f32)> {
		vec![]
	}

	fn advance(&mut self, _vocabulary: &Tokenizer, _token: TokenId) {}
}
