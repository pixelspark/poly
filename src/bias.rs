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
	String {},
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
			(JSONSchema::String { .. }, Value::String(_s)) => true,
			_ => false,
		}
	}
}

#[derive(Clone)]
struct JSONParserArrayState<'schema> {
	items: Vec<Value>,
	value_state: Box<JSONBiaser<'schema>>,
}

// Temp, to hide schema in logs
impl<'schema> std::fmt::Debug for JSONParserArrayState<'schema> {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("JSONParserArrayState").field("state", &self.value_state.state).finish()
	}
}

#[derive(Debug, Clone)]
enum JSONParserState<'schema> {
	/// No input received yet
	Start,

	/// Object opening curly brace encountered
	InObject,

	/// Array opening bracket encountered
	InArray(JSONParserArrayState<'schema>),

	/// Inside an integer (true = positive, false = negative)
	InInteger(String),

	/// JSON value is finished, no further input acceptable
	End(Value),

	/// Inside a string
	InString(String),
}

const TOKEN_ALLOWED: f32 = 10000.0;
const TOKEN_FORBIDDEN: f32 = -10000.0;

pub trait Biaser {
	fn bias(&self, vocabulary: &Vocabulary, eot_token: TokenId) -> Vec<(TokenId, f32)>;
	fn advance(&mut self, vocabulary: &Vocabulary, token: TokenId);
}

impl<'schema> Biaser for JSONBiaser<'schema> {
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

		// If the set contains String(""), then basically anything is allowed (no bias). Signal this by only returning
		// that the end token is forbidden
		if next_valid_json_tokens.contains(&JSONToken::AnyString) {
			let mut valid_tokens: Vec<TokenId> = (0..=(vocabulary.len() - 1) as TokenId)
				.filter(|token_id| {
					if *token_id == eot_token {
						return false;
					}
					let bytes = vocabulary.token(*token_id as usize);
					//println!("VOCAB TOKEN {token_id}={:?}", String::from_utf8(bytes.clone()));
					let Ok(s) = String::from_utf8(bytes) else {
						return false;
					};

					if s.contains('\"') || s.contains('\n') || s.contains('\t') || s.contains('\r') {
						return false;
					}
					true
				})
				.collect();

			valid_tokens.push(JSONToken::DoubleQuote.token_id(vocabulary).unwrap());

			println!("total tokens: {} valid: {}", vocabulary.len(), valid_tokens.len());

			return valid_tokens.iter().map(|vt| (*vt, TOKEN_ALLOWED)).collect();

			//return vec![(eot_token, TOKEN_FORBIDDEN)];
		}

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
		self.advance(&out_json_token).unwrap();
		tracing::debug!(
			"Token: {:?}, next valid tokens: {:?}",
			&out_json_token,
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
pub struct JSONBiaser<'schema> {
	schema: &'schema JSONSchema,
	state: JSONParserState<'schema>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JSONToken {
	AnyString, // Any string except double quote (used in next_valid_token)
	BracketClose,
	BracketOpen,
	Comma,
	CurlyClose,
	CurlyOpen,
	Decimal,
	Digit(usize),
	DoubleQuote,
	False,
	Minus,
	Null,
	String(String), // Anything except the double quote
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
			"\"" => JSONToken::DoubleQuote,
			s => {
				if let Ok(n) = s.parse() {
					JSONToken::Digit(n)
				} else if s != "\\" {
					JSONToken::String(s.to_string())
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
			JSONToken::Digit(n) => Cow::from(format!("{n}")),
			JSONToken::DoubleQuote => Cow::from("\""),
			JSONToken::String(s) => Cow::from(s.clone()),
			JSONToken::AnyString => panic!(),
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

impl<'schema> JSONParserState<'schema> {
	pub fn value(&self) -> Option<Value> {
		match self {
			JSONParserState::Start => None,
			JSONParserState::InString(s) => Some(Value::String(s.clone())),
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

	pub fn advance(&mut self, input: &JSONToken, item_schema: Option<&'schema JSONSchema>) -> Result<(), BiaserError> {
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
				JSONToken::Digit(n) => JSONParserState::InInteger(format!("{n}")),
				JSONToken::DoubleQuote => JSONParserState::InString(String::from("")),
				_ => return Err(BiaserError::InvalidToken(input.clone())),
			},
			JSONParserState::InString(s) => match input {
				JSONToken::DoubleQuote => JSONParserState::End(json! { s }),
				JSONToken::String(new_string) => {
					if new_string.ends_with('\"') {
						let string_value = format!("{s}{}", new_string.strip_suffix('\"').unwrap_or(""));
						JSONParserState::End(Value::String(string_value))
					} else {
						assert!(!new_string.contains('\"'), "String token may not contain double quote");
						JSONParserState::InString(format!("{s}{new_string}"))
					}
				}
				t => {
					// This could be any other token but now inside the string
					let new_string = t.to_string();
					assert!(!new_string.contains('\"'), "String token may not contain double quote");
					JSONParserState::InString(format!("{s}{new_string}"))
				}
			},
			JSONParserState::InInteger(num_string) => match input {
				JSONToken::Digit(n) => JSONParserState::InInteger(format!("{num_string}{n}")),
				JSONToken::Decimal => JSONParserState::InInteger(format!("{num_string}.")),
				_ => return Err(BiaserError::InvalidToken(input.clone())),
			},
			JSONParserState::InObject => match input {
				JSONToken::CurlyClose => JSONParserState::End(json! { {} }),
				_ => return Err(BiaserError::InvalidToken(input.clone())),
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
					t if next_valid_item_tokens.contains(t) => {
						array_state.value_state.advance(input)?;
						JSONParserState::InArray(array_state)
					}
					t => return Err(BiaserError::InvalidToken(t.clone())),
				}
			}

			JSONParserState::End(_) => return Err(BiaserError::InvalidToken(input.clone())),
		};
		Ok(())
	}
}

impl<'schema> JSONBiaser<'schema> {
	pub fn new(schema: &'schema JSONSchema) -> JSONBiaser<'schema> {
		JSONBiaser {
			schema,
			state: JSONParserState::Start,
		}
	}

	fn child_item_schema(&self) -> Option<&'schema JSONSchema> {
		match &self.schema {
			JSONSchema::Array { items, .. } => Some(items.as_ref()),
			_ => None,
		}
	}

	pub fn advance(&mut self, input: &JSONToken) -> Result<(), BiaserError> {
		self.state.advance(input, self.child_item_schema())
	}

	pub fn can_end(&self) -> bool {
		match self.state {
			JSONParserState::Start => false,
			JSONParserState::InObject => false,
			JSONParserState::InArray(ref _array_state) => false,
			JSONParserState::InInteger(ref s) => !s.is_empty() && s.parse::<f32>().is_ok() && !s.ends_with('.'),
			JSONParserState::End(_) => true,
			JSONParserState::InString(_) => false,
		}
	}

	pub fn next_valid_tokens(&self) -> Vec<JSONToken> {
		match &self.state {
			JSONParserState::End(_) => vec![],
			JSONParserState::InObject => vec![JSONToken::CurlyClose],
			JSONParserState::InString(_) => {
				vec![JSONToken::DoubleQuote, JSONToken::AnyString] // Any string
			}
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

				// Check if we are below the set maximum number of decimals
				if s.contains('.') && max_decimals > 0 {
					let decimals = s.split_once('.').unwrap().1;
					if decimals.len() >= max_decimals {
						return vec![];
					}
				}

				// First digit cannot be zero
				let mut digits: Vec<JSONToken> = if s == "-" {
					(1..=9).map(JSONToken::Digit).collect()
				} else {
					(0..=9).map(JSONToken::Digit).collect()
				};

				// Limit the length of a number literal to what fits in a 32 bit integer
				if let Ok(v) = s.parse::<f64>() {
					if v >= (u32::MAX as f64) {
						return vec![];
					}

					if let Some(max) = max {
						if v >= *max {
							return vec![];
						}

						digits.retain_mut(|digit| {
							// Try to append the digit and see if we still meet the minimum
							match format!("{s}{}", digit).parse::<f64>() {
								Err(_) => false,
								Ok(v) => v <= *max,
							}
						});
					}

					if let Some(min) = min {
						if v <= *min {
							return vec![];
						}

						digits.retain_mut(|digit| {
							// Try to append the digit and see if we still meet the minimum
							match format!("{s}{}", digit).parse::<f64>() {
								Err(_) => false,
								Ok(v) => v >= *min,
							}
						});
					}
				}

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
				JSONSchema::String {} => {
					vec![JSONToken::DoubleQuote]
				}
				JSONSchema::Number { max, min, max_decimals: _ } => {
					// First digit cannot be zero
					let mut d: Vec<JSONToken> = (1..=9)
						.filter(|d| {
							let df = *d as f64;
							df <= max.unwrap_or(df) && df >= min.unwrap_or(df)
						})
						.map(JSONToken::Digit)
						.collect();

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
