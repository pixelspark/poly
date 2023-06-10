use std::borrow::Cow;
use std::fmt::Display;

use llm::TokenizationError;
use llm::{TokenId, Vocabulary};
use serde::{Deserialize, Serialize};
use serde_json::json;
use serde_json::Value;
use thiserror::Error;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum JSONSchema {
	Boolean,
	Null,
	Object,
	Number,
	Array { items: Box<JSONSchema>, min_items: Option<usize> },
}

#[derive(Clone)]
struct JSONParserArrayState {
	items: Vec<Value>,
	value_state: Box<JSONBiaser>,
}

// Temp, to hide schema in logs
impl std::fmt::Debug for JSONParserArrayState {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("JSONParserArrayState").field("state", &self.value_state.state).finish()
	}
}

#[derive(Debug, Clone)]
enum JSONParserState {
	/// No input received yet
	Start,

	/// Object opening curly brace encountered
	InObject,

	/// Array opening bracket encountered
	InArray(JSONParserArrayState),

	/// Expect a comma for the next array item
	ExpectComma,

	/// Inside an integer (true = positive, false = negative)
	InInteger(bool, isize),

	/// JSON value is finished, no further input acceptable
	End(Value),
}

#[derive(Debug, Clone)]
pub struct JSONBiaser {
	schema: JSONSchema,
	state: JSONParserState,
}

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum JSONToken {
	True,
	False,
	Null,
	CurlyOpen,
	CurlyClose,
	BracketOpen,
	BracketClose,
	Comma,
	Minus,
	Number(usize),
	Eot,
}

impl JSONToken {
	pub fn from_text(s: &str) -> Option<JSONToken> {
		Some(match s {
			"true" => JSONToken::True,
			"false" => JSONToken::False,
			"null" => JSONToken::Null,
			"{" => JSONToken::CurlyOpen,
			"}" => JSONToken::CurlyClose,
			"[" => JSONToken::BracketOpen,
			"]" => JSONToken::BracketClose,
			"," => JSONToken::Comma,
			"-" => JSONToken::Minus,
			s => {
				if let Ok(n) = s.parse() {
					JSONToken::Number(n)
				} else {
					return None;
				}
			}
		})
	}

	pub fn to_string(&self) -> Cow<'static, str> {
		match self {
			JSONToken::True => Cow::from("true"),
			JSONToken::False => Cow::from("false"),
			JSONToken::Null => Cow::from("null"),
			JSONToken::CurlyOpen => Cow::from("{"),
			JSONToken::CurlyClose => Cow::from("}"),
			JSONToken::BracketOpen => Cow::from("["),
			JSONToken::BracketClose => Cow::from("]"),
			JSONToken::Comma => Cow::from(","),
			JSONToken::Minus => Cow::from("-"),
			JSONToken::Number(n) => Cow::from(format!("{n}")),
			JSONToken::Eot => Cow::from(""),
		}
	}

	pub fn from_token(vocab: &Vocabulary, eot_token_id: TokenId, token: TokenId) -> Result<JSONToken, TokenizationError> {
		if token == eot_token_id {
			Ok(JSONToken::Eot)
		} else {
			let bytes = vocab.decode(vec![token], false);
			let s = String::from_utf8(bytes).map_err(|_e| TokenizationError::InvalidTokenId(token))?;
			Self::from_text(&s).ok_or(TokenizationError::InvalidTokenId(token))
		}
	}

	pub fn token_id(&self, eot_token_id: TokenId, vocab: &Vocabulary) -> Option<TokenId> {
		if let JSONToken::Eot = self {
			return Some(eot_token_id);
		}

		match vocab.tokenize(&self.to_string(), false) {
			Ok(tokens) => {
				if tokens.len() != 1 {
					None
				} else {
					Some(tokens[0].1)
				}
			}
			Err(_) => None,
		}
	}

	pub fn numbers() -> Vec<JSONToken> {
		(0..=9).map(JSONToken::Number).collect()
	}
}

impl Display for JSONToken {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "{}", self.to_string())
	}
}

pub fn single_token_id_for(vocab: &Vocabulary, s: &str) -> Option<u32> {
	let ts = vocab.tokenize(s, false).unwrap();
	if ts.len() != 1 {
		None
	} else {
		Some(ts[0].1)
	}
}

#[derive(Error, Debug)]
pub enum BiaserError {
	#[error("invalid next token {0}")]
	InvalidToken(JSONToken),
}

impl JSONParserState {
	pub fn advance(&mut self, input: JSONToken, item_schema: Option<JSONSchema>) -> Result<(), BiaserError> {
		*self = match self {
			JSONParserState::Start => match input {
				JSONToken::True => JSONParserState::End(json! { true }),
				JSONToken::False => JSONParserState::End(json! { false }),
				JSONToken::Null => JSONParserState::End(json! { null }),
				JSONToken::CurlyOpen => JSONParserState::InObject,
				JSONToken::BracketOpen => JSONParserState::InArray(JSONParserArrayState {
					items: vec![],
					value_state: Box::new(JSONBiaser::new(item_schema.unwrap())),
				}),
				JSONToken::Minus => JSONParserState::InInteger(false, 0),
				JSONToken::Number(n) => JSONParserState::InInteger(true, n as isize),
				_ => return Err(BiaserError::InvalidToken(input)),
			},
			JSONParserState::InInteger(sign, digits) => match input {
				JSONToken::Eot => {
					let n = (if *sign { 1 } else { -1 }) * *digits;
					JSONParserState::End(json! { n })
				}
				JSONToken::Number(n) => JSONParserState::InInteger(*sign, *digits * 10 + (n as isize)),
				_ => return Err(BiaserError::InvalidToken(input)),
			},
			JSONParserState::InObject => match input {
				JSONToken::CurlyClose => JSONParserState::End(json! { {} }),
				_ => return Err(BiaserError::InvalidToken(input)),
			},
			JSONParserState::InArray(array_state) => {
				let mut array_state: JSONParserArrayState = array_state.clone();
				let next_valid_item_tokens = array_state.value_state.next_valid_tokens();

				let old_array_state = array_state.value_state.state.clone();

				let next_state = {
					if let JSONParserState::End(v) = array_state.value_state.state {
						array_state.items.push(v);
						if input == JSONToken::Comma {
							array_state.value_state.state = JSONParserState::Start;
						} else {
							array_state.value_state.state = JSONParserState::ExpectComma;
						}
						JSONParserState::InArray(array_state)
					} else if let JSONParserState::ExpectComma = array_state.value_state.state {
						array_state.value_state.state = JSONParserState::Start;
						JSONParserState::InArray(array_state)
					} else if next_valid_item_tokens.contains(&input) {
						array_state.value_state.advance(input)?;
						JSONParserState::InArray(array_state)
					} else if input == JSONToken::BracketClose {
						JSONParserState::End(json! { array_state.items })
					} else {
						return Err(BiaserError::InvalidToken(input));
					}
				};
				println!(
					"advance array {input:?}: array_state was ArrayState({:?}) => {:?}",
					old_array_state, next_state
				);
				next_state
			}
			JSONParserState::ExpectComma => match input {
				JSONToken::Comma => JSONParserState::Start,
				_ => return Err(BiaserError::InvalidToken(input)),
			},
			JSONParserState::End(_) => return Err(BiaserError::InvalidToken(input)),
		};
		Ok(())
	}
}

impl JSONBiaser {
	pub fn new(schema: JSONSchema) -> JSONBiaser {
		JSONBiaser {
			schema,
			state: JSONParserState::Start,
		}
	}

	fn child_item_schema(&self) -> Option<JSONSchema> {
		match &self.schema {
			JSONSchema::Array { items, .. } => Some(*items.clone()),
			_ => None,
		}
	}

	pub fn advance(&mut self, input: JSONToken) -> Result<(), BiaserError> {
		self.state.advance(input, self.child_item_schema())
	}

	pub fn next_valid_tokens(&self) -> Vec<JSONToken> {
		match &self.state {
			JSONParserState::End(_) => vec![],
			JSONParserState::ExpectComma => vec![JSONToken::Comma],
			JSONParserState::InObject => vec![JSONToken::CurlyClose],
			JSONParserState::InArray(array_state) => {
				let mut valid = array_state.value_state.next_valid_tokens();

				// If a value is done, we expect a comma first
				if let JSONParserState::End(_) = array_state.value_state.state {
					valid.clear();
					// TODO: max_items
					valid.push(JSONToken::Comma);
				}

				let JSONSchema::Array { min_items, .. } = self.schema else {
					panic!();
				};

				if let Some(min_items) = min_items {
					if min_items < array_state.items.len() {
						// We have enough items, you may close the array
						valid.push(JSONToken::BracketClose);
					}
				} else {
					// No minimum number of items, you may close the array
					valid.push(JSONToken::BracketClose);
				}
				valid
			}
			JSONParserState::InInteger(_sign, digits) => {
				// Limit the length of a number literal to what fits in a 32 bit integer
				if *digits >= u32::MAX as isize {
					return vec![JSONToken::Eot];
				}
				// TODO: EOT token?
				JSONToken::numbers()
			}
			JSONParserState::Start => match self.schema {
				JSONSchema::Boolean => {
					vec![JSONToken::True, JSONToken::False]
				}
				JSONSchema::Null => {
					vec![JSONToken::Null]
				}
				JSONSchema::Object => {
					vec![JSONToken::CurlyOpen]
				}
				JSONSchema::Number => {
					let mut d = JSONToken::numbers();
					d.push(JSONToken::Minus);
					d.push(JSONToken::Eot);
					d
				}
				JSONSchema::Array { .. } => {
					vec![JSONToken::BracketOpen]
				}
			},
		}
	}
}
