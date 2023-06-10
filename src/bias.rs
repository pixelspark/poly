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
	Array {
		items: Box<JSONSchema>,
		min_items: Option<usize>,
		max_items: Option<usize>,
	},
}

impl JSONSchema {
	pub fn is_valid(&self, value: &Value) -> bool {
		match (self, value) {
			(JSONSchema::Boolean, Value::Bool(_)) => true,
			(JSONSchema::Null, Value::Null) => true,
			(JSONSchema::Object, Value::Object(_)) => true,
			(JSONSchema::Array { items, min_items, max_items }, Value::Array(array_items)) => {
				if let Some(min_items) = min_items {
					if *min_items > array_items.len() {
						return false;
					}
				}

				if let Some(max_items) = max_items {
					if *max_items < array_items.len() {
						return false;
					}
				}
				return array_items.iter().all(|item| items.is_valid(item));
			}
			(JSONSchema::Number, Value::Number(_s)) => true,
			_ => false,
		}
	}
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

	/// Inside an integer (true = positive, false = negative)
	InInteger(String),

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
		}
	}

	pub fn from_token(vocab: &Vocabulary, token: TokenId) -> Result<JSONToken, TokenizationError> {
		let bytes = vocab.decode(vec![token], false);
		let s = String::from_utf8(bytes).map_err(|_e| TokenizationError::InvalidTokenId(token))?;
		Self::from_text(&s).ok_or(TokenizationError::InvalidTokenId(token))
	}

	pub fn token_id(&self, vocab: &Vocabulary) -> Option<TokenId> {
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
}

impl Display for JSONToken {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "{}", self.to_string())
	}
}

#[derive(Error, Debug)]
pub enum BiaserError {
	#[error("invalid next token {0}")]
	InvalidToken(JSONToken),
}

impl JSONParserState {
	pub fn value(&self) -> Option<Value> {
		match self {
			JSONParserState::Start => None,
			JSONParserState::InObject => Some(json!({})),
			JSONParserState::InArray(array_state) => {
				let mut items = array_state.items.clone();
				if let Some(v) = array_state.value_state.state.value() {
					items.push(v);
				}
				Some(Value::Array(items))
			}
			JSONParserState::InInteger(s) => Some(json! { s.parse::<f32>().unwrap() }),
			JSONParserState::End(v) => Some(v.clone()),
		}
	}

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
				JSONToken::Minus => JSONParserState::InInteger(String::from("-")),
				JSONToken::Number(n) => JSONParserState::InInteger(format!("{n}")),
				_ => return Err(BiaserError::InvalidToken(input)),
			},
			JSONParserState::InInteger(num_string) => match input {
				JSONToken::Number(n) => JSONParserState::InInteger(format!("{num_string}{n}")),
				_ => return Err(BiaserError::InvalidToken(input)),
			},
			JSONParserState::InObject => match input {
				JSONToken::CurlyClose => JSONParserState::End(json! { {} }),
				_ => return Err(BiaserError::InvalidToken(input)),
			},
			JSONParserState::InArray(array_state) => {
				let mut array_state: JSONParserArrayState = array_state.clone();
				let next_valid_item_tokens = array_state.value_state.next_valid_tokens();

				match input {
					JSONToken::Comma if array_state.value_state.can_end() => {
						if let Some(v) = array_state.value_state.state.value() {
							array_state.items.push(v);
						}
						array_state.value_state.state = JSONParserState::Start;
						JSONParserState::InArray(array_state)
					}
					JSONToken::BracketClose if array_state.value_state.can_end() => {
						if let Some(v) = array_state.value_state.state.value() {
							array_state.items.push(v);
						}
						JSONParserState::End(Value::Array(array_state.items))
					}
					t if next_valid_item_tokens.contains(&t) => {
						array_state.value_state.advance(input)?;
						JSONParserState::InArray(array_state)
					}
					t => return Err(BiaserError::InvalidToken(t)),
				}
			}

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

	pub fn can_end(&self) -> bool {
		match self.state {
			JSONParserState::Start => false,
			JSONParserState::InObject => false,
			JSONParserState::InArray(ref _array_state) => false,
			JSONParserState::InInteger(ref s) => s.parse::<f32>().is_ok(),
			JSONParserState::End(_) => true,
		}
	}

	pub fn next_valid_tokens(&self) -> Vec<JSONToken> {
		match &self.state {
			JSONParserState::End(_) => vec![],
			JSONParserState::InObject => vec![JSONToken::CurlyClose],
			JSONParserState::InArray(array_state) => {
				let JSONSchema::Array { min_items, max_items, .. } = self.schema else {
					panic!();
				};

				let mut valid = array_state.value_state.next_valid_tokens();

				if array_state.value_state.can_end() {
					// If the inner value can end (or must end, then valid = []), expect a comma (if we can accomodate more items)
					if max_items.is_none() || (array_state.items.len() + 1) <= max_items.unwrap() {
						valid.push(JSONToken::Comma);
					}

					// If we have enough items, also allow bracket close
					let has_enough_items = (array_state.items.len() + 1) >= min_items.unwrap_or(0);
					if has_enough_items {
						valid.push(JSONToken::BracketClose);
					}
				}

				valid
			}
			JSONParserState::InInteger(s) => {
				// Limit the length of a number literal to what fits in a 32 bit integer
				if let Ok(v) = s.parse::<f32>() {
					if v >= (u32::MAX as f32) {
						return vec![];
					}
				}

				// First digit cannot be zero
				if s == "-" {
					(1..=9).map(JSONToken::Number).collect()
				} else {
					(0..=9).map(JSONToken::Number).collect()
				}
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
					// First digit cannot be zero
					let mut d: Vec<JSONToken> = (1..=9).map(JSONToken::Number).collect();
					d.push(JSONToken::Minus);
					d
				}
				JSONSchema::Array { .. } => {
					vec![JSONToken::BracketOpen]
				}
			},
		}
	}
}
