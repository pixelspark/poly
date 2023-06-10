use std::borrow::Cow;
use std::fmt::Display;

use llm::TokenizationError;
use llm::{TokenId, Vocabulary};
use serde::{Deserialize, Serialize};
use serde_json::json;
use serde_json::Value;
use thiserror::Error;

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum JSONSchema {
	Boolean,
	Null,
	Object,
	Number {
		min: Option<f64>,
		max: Option<f64>,
		max_decimals: Option<usize>,
	},
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
			(JSONSchema::Number { min, max, .. }, Value::Number(v)) => {
				if let Some(min) = min {
					if v.as_f64().unwrap() < *min {
						return false;
					}
				}
				if let Some(max) = max {
					if v.as_f64().unwrap() > *max {
						return false;
					}
				}
				true
			}
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

const TOKEN_ALLOWED: f32 = 1000.0;

pub trait Biaser {
	fn bias(&self, vocabulary: &Vocabulary, eot_token: TokenId) -> Vec<(TokenId, f32)>;
	fn advance(&mut self, vocabulary: &Vocabulary, token: TokenId);
}

impl Biaser for JSONBiaser {
	fn bias(&self, vocabulary: &Vocabulary, eot_token: TokenId) -> Vec<(TokenId, f32)> {
		let next_valid_json_tokens = self.next_valid_tokens();
		tracing::trace!(
			"next valid tokens: {}",
			next_valid_json_tokens
				.iter()
				.map(|x| x.to_string().to_string())
				.collect::<Vec<String>>()
				.join(" ")
		);
		let mut next_valid_tokens: Vec<(TokenId, f32)> = next_valid_json_tokens
			.iter()
			.map(|t| (t.token_id(vocabulary).unwrap_or_else(|| panic!("token id for {t}")), TOKEN_ALLOWED))
			.collect();
		if self.can_end() {
			next_valid_tokens.push((eot_token, TOKEN_ALLOWED));
		}
		next_valid_tokens
	}

	fn advance(&mut self, vocabulary: &Vocabulary, token: TokenId) {
		let out_json_token = JSONToken::from_token(vocabulary, token).expect("valid token");
		self.advance(out_json_token).unwrap();
		tracing::debug!(
			"Token: {:?}, next valid tokens: {:?}",
			out_json_token,
			self.next_valid_tokens()
				.iter()
				.map(|x| x.to_string().to_string())
				.collect::<Vec<String>>()
				.join(" "),
		);
	}
}

pub struct NullBiaser {}

impl Biaser for NullBiaser {
	fn bias(&self, _vocabulary: &Vocabulary, _eot_token: TokenId) -> Vec<(TokenId, f32)> {
		vec![]
	}

	fn advance(&mut self, _vocabulary: &Vocabulary, _token: TokenId) {}
}

#[derive(Debug, Clone)]
pub struct JSONBiaser {
	schema: JSONSchema,
	state: JSONParserState,
}

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum JSONToken {
	BracketClose,
	BracketOpen,
	Comma,
	CurlyClose,
	CurlyOpen,
	Decimal,
	False,
	Minus,
	Null,
	Number(usize),
	True,
}

impl JSONToken {
	pub fn from_text(s: &str) -> Option<JSONToken> {
		Some(match s {
			"true" => JSONToken::True,
			"false" => JSONToken::False,
			"null" => JSONToken::Null,
			"." => JSONToken::Decimal,
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
			JSONToken::Decimal => Cow::from("."),
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
				JSONToken::Decimal => JSONParserState::InInteger(format!("{num_string}.")),
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
			JSONParserState::InInteger(ref s) => !s.is_empty() && s.parse::<f32>().is_ok() && !s.ends_with('.'),
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
				let JSONSchema::Number { max_decimals, min, max } = self.schema else {
					panic!();
				};
				let max_decimals = max_decimals.unwrap_or(0);
				let has_decimal = s.contains('.');

				if max_decimals == 0 && has_decimal {
					panic!("have decimal while not allowed");
				}

				// Limit the length of a number literal to what fits in a 32 bit integer
				if let Ok(v) = s.parse::<f64>() {
					if v >= (u32::MAX as f64) {
						return vec![];
					}
					if let Some(max) = max {
						if v >= max {
							return vec![];
						}

						// Can't just add numbers if that would make us go over max
						// TODO: fix this logic
						if !has_decimal && v * 10.0 >= max {
							return if max_decimals > 0 { vec![JSONToken::Decimal] } else { vec![] };
						}

						// TODO: if we have a decimal, we should limit adding digits after it
					}

					if let Some(min) = min {
						if v <= min {
							return vec![];
						}

						// Can't just add numbers if that would make us go over max
						// TODO: fix this logic
						if !has_decimal && v * 10.0 <= min {
							return if max_decimals > 0 { vec![JSONToken::Decimal] } else { vec![] };
						}

						// TODO: if we have a decimal, we should limit adding digits after it
					}
				}

				if s.contains('.') && max_decimals > 0 {
					let decimals = s.split_once('.').unwrap().1;
					if decimals.len() >= max_decimals {
						return vec![];
					}
				}

				// First digit cannot be zero
				let mut digits: Vec<JSONToken> = if s == "-" {
					(1..=9).map(JSONToken::Number).collect()
				} else {
					(0..=9).map(JSONToken::Number).collect()
				};

				if !has_decimal && max_decimals > 0 {
					digits.push(JSONToken::Decimal);
				}
				digits
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
				JSONSchema::Number { max, min, max_decimals: _ } => {
					// First digit cannot be zero
					let mut d: Vec<JSONToken> = (1..=9).map(JSONToken::Number).collect();

					if min.unwrap_or(-1.0) < 0.0 || max.unwrap_or(-1.0) < 0.0 {
						d.push(JSONToken::Minus);
					}
					d
				}
				JSONSchema::Array { .. } => {
					vec![JSONToken::BracketOpen]
				}
			},
		}
	}
}
