use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::Display;

use llm::TokenizationError;
use llm::{TokenId, Tokenizer};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use serde_json::{json, Map};
use thiserror::Error;

use crate::{Biaser, TOKEN_ALLOWED};

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum JsonSchema {
	Boolean,
	Null,
	Object {
		required: Vec<String>,
		properties: HashMap<String, Box<JsonSchema>>,
	},
	Number {
		min: Option<f64>,
		max: Option<f64>,
		max_decimals: Option<usize>,
	},
	Array {
		items: Box<JsonSchema>,
		min_items: Option<usize>,
		max_items: Option<usize>,
	},
	String {
		max_length: Option<usize>,
		r#enum: Option<Vec<String>>,
	},
}

impl JsonSchema {
	pub fn is_valid(&self, value: &Value) -> bool {
		match (self, value) {
			(JsonSchema::Boolean, Value::Bool(_)) => true,
			(JsonSchema::Null, Value::Null) => true,
			(JsonSchema::Object { required, properties }, Value::Object(object_value)) => {
				// All required keys must be present
				if !required.iter().all(|field| object_value.contains_key(field)) {
					false
				} else {
					// All keys that are in the object must conform to their schemas
					object_value.iter().all(|(field, field_value)| {
						let Some(field_schema) = properties.get(field) else {
							return false; // No schema for this field
						};

						field_schema.is_valid(field_value)
					})
				}
			}
			(JsonSchema::Array { items, min_items, max_items }, Value::Array(array_items)) => {
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
			(JsonSchema::Number { min, max, .. }, Value::Number(v)) => {
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
			(JsonSchema::String { .. }, Value::String(_s)) => true,
			_ => false,
		}
	}
}

#[derive(Clone)]
struct JsonParserArrayState<'schema> {
	items: Vec<Value>,
	value_state: Box<JsonBiaser<'schema>>,
}

// Temp, to hide schema in logs
impl<'schema> std::fmt::Debug for JsonParserArrayState<'schema> {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("JSONParserArrayState").field("state", &self.value_state.state).finish()
	}
}

#[derive(Debug, Clone)]
enum JsonParserObjectPartState<'schema> {
	BeforeKey,
	InKey(String),
	AfterKey(String),
	InValue { key: String, value: Box<JsonBiaser<'schema>> },
	Finished,
}

#[derive(Debug, Clone)]
struct JsonParserObjectState<'schema> {
	so_far: Map<String, Value>,
	object_schema: &'schema JsonSchema,
	part_state: JsonParserObjectPartState<'schema>,
}

#[derive(Debug, Clone)]
enum JsonParserState<'schema> {
	/// No input received yet
	Start,

	/// Object opening curly brace encountered
	InObject(JsonParserObjectState<'schema>),

	/// Array opening bracket encountered
	InArray(JsonParserArrayState<'schema>),

	/// Inside an integer (true = positive, false = negative)
	InInteger(String),

	/// JSON value is finished, no further input acceptable
	End(Value),

	/// Inside a string
	InString(String),
}

impl<'schema> Biaser for JsonBiaser<'schema> {
	fn bias(&self, vocabulary: &Tokenizer, eot_token: TokenId) -> Vec<(TokenId, f32)> {
		let next_valid_json_tokens = self.next_valid_tokens();
		tracing::trace!("next valid tokens: {:?}", next_valid_json_tokens);

		// Translate the next valid JSON tokens to model tokens
		let mut next_valid_tokens: Vec<(TokenId, f32)> = next_valid_json_tokens
			.iter()
			.flat_map(|json_token| match json_token {
				// Any string value from the list, or a prefix of one of the values, is allowed
				JsonToken::AnyOf(string_values) => {
					debug_assert!(
						string_values.iter().all(|s| !s.is_empty()),
						"no empty strings allowed in JSONToken::AnyOf"
					);

					let valid_tokens: Vec<TokenId> = (0..=(vocabulary.len() - 1) as TokenId)
						.filter(|token_id| {
							if *token_id == eot_token {
								return false;
							}
							let bytes = vocabulary.token(*token_id as usize);
							let Ok(s) = String::from_utf8(bytes) else {
								return false;
							};

							if s.is_empty() {
								return false;
							}

							if s.contains('\"') || s.contains('\n') || s.contains('\t') || s.contains('\r') {
								return false;
							}

							return string_values.iter().any(|sv| sv.starts_with(&s));
						})
						.collect();

					tracing::debug!("any-of: total tokens: {} valid: {}", vocabulary.len(), valid_tokens.len());
					tracing::trace!("any-of prefixes: {string_values:?} valid: {valid_tokens:?}");

					valid_tokens.iter().map(|vt| (*vt, TOKEN_ALLOWED)).collect()
				}

				// Basically any token is allowed if it fits the max length. Filter them from the vocabulary
				JsonToken::AnyString { max_length } => {
					let mut valid_tokens: Vec<TokenId> = (0..=(vocabulary.len() - 1) as TokenId)
						.filter(|token_id| {
							if *token_id == eot_token {
								return false;
							}
							let bytes = vocabulary.token(*token_id as usize);
							let Ok(s) = String::from_utf8(bytes) else {
								return false;
							};

							// Reject tokens that would make the string go over the maximum length
							if let Some(max_length) = max_length {
								if *max_length < s.len() {
									return false;
								}
							}

							if s.contains('\"') || s.contains('\n') || s.contains('\t') || s.contains('\r') {
								return false;
							}
							true
						})
						.collect();

					valid_tokens.push(JsonToken::DoubleQuote.token_id(vocabulary).unwrap());

					tracing::debug!("total tokens: {} valid: {}", vocabulary.len(), valid_tokens.len());

					valid_tokens.iter().map(|vt| (*vt, TOKEN_ALLOWED)).collect()
				}
				json_token => {
					vec![(
						(*json_token).token_id(vocabulary).unwrap_or_else(|| panic!("token id for {json_token}")),
						TOKEN_ALLOWED,
					)]
				}
			})
			.collect();

		if self.can_end() {
			next_valid_tokens.push((eot_token, TOKEN_ALLOWED));
		}
		next_valid_tokens
	}

	fn advance(&mut self, vocabulary: &Tokenizer, token: TokenId) {
		let out_json_token = JsonToken::from_token(vocabulary, token).expect("valid token");
		self.advance(&out_json_token).unwrap();
		tracing::debug!("Token: {:?}, next valid tokens: {:?}", &out_json_token, self.next_valid_tokens());
	}
}

#[derive(Debug)]
pub struct JsonBiaser<'schema> {
	schema: &'schema JsonSchema,
	state: JsonParserState<'schema>,
}

impl<'schema> Clone for JsonBiaser<'schema> {
	fn clone(&self) -> Self {
		Self {
			schema: self.schema,
			state: JsonParserState::Start,
		}
	}
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JsonToken {
	AnyString { max_length: Option<usize> }, // Any string except double quote (used in next_valid_token)
	AnyOf(Vec<String>),                      // Any string from the list (or a prefix of it)
	BracketClose,
	BracketOpen,
	Colon,
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

impl JsonToken {
	pub fn from_text(s: &str) -> Option<JsonToken> {
		Some(match s {
			"true" => JsonToken::True,
			"false" => JsonToken::False,
			"null" => JsonToken::Null,
			":" => JsonToken::Colon,
			"." => JsonToken::Decimal,
			"{" => JsonToken::CurlyOpen,
			"}" => JsonToken::CurlyClose,
			"[" => JsonToken::BracketOpen,
			"]" => JsonToken::BracketClose,
			"," => JsonToken::Comma,
			"-" => JsonToken::Minus,
			"\"" => JsonToken::DoubleQuote,
			s => {
				if let Ok(n) = s.parse() {
					JsonToken::Digit(n)
				} else if s != "\\" {
					JsonToken::String(s.to_string())
				} else {
					return None;
				}
			}
		})
	}

	pub fn to_string(&self) -> Option<Cow<'static, str>> {
		Some(match self {
			JsonToken::True => Cow::from("true"),
			JsonToken::False => Cow::from("false"),
			JsonToken::Null => Cow::from("null"),
			JsonToken::Colon => Cow::from(":"),
			JsonToken::CurlyOpen => Cow::from("{"),
			JsonToken::CurlyClose => Cow::from("}"),
			JsonToken::BracketOpen => Cow::from("["),
			JsonToken::BracketClose => Cow::from("]"),
			JsonToken::Comma => Cow::from(","),
			JsonToken::Minus => Cow::from("-"),
			JsonToken::Decimal => Cow::from("."),
			JsonToken::Digit(n) => Cow::from(format!("{n}")),
			JsonToken::DoubleQuote => Cow::from("\""),
			JsonToken::String(s) => Cow::from(s.clone()),
			JsonToken::AnyString { .. } | JsonToken::AnyOf(_) => return None,
		})
	}

	pub fn from_token(vocab: &Tokenizer, token: TokenId) -> Result<JsonToken, TokenizationError> {
		let bytes = vocab.decode(vec![token], false);
		let s = String::from_utf8(bytes).map_err(|_e| TokenizationError::InvalidTokenId(token))?;
		Self::from_text(&s).ok_or(TokenizationError::InvalidTokenId(token))
	}

	pub fn token_id(&self, vocab: &Tokenizer) -> Option<TokenId> {
		let s = self.to_string()?;

		match vocab.tokenize(&s, false) {
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

impl Display for JsonToken {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			JsonToken::AnyOf(s) => write!(f, "<any of: {}>", s.join(", ")),
			JsonToken::AnyString { max_length } => write!(f, "<any string max_length={max_length:?}>"),
			JsonToken::BracketClose
			| JsonToken::BracketOpen
			| JsonToken::Comma
			| JsonToken::Colon
			| JsonToken::CurlyClose
			| JsonToken::CurlyOpen
			| JsonToken::Decimal
			| JsonToken::Digit(_)
			| JsonToken::DoubleQuote
			| JsonToken::False
			| JsonToken::Minus
			| JsonToken::Null
			| JsonToken::String(_)
			| JsonToken::True => write!(f, "{}", self.to_string().unwrap()),
		}
	}
}

#[derive(Error, Debug)]
pub enum BiaserError {
	#[error("invalid next token {0}")]
	InvalidToken(JsonToken),
}

impl<'schema> JsonParserObjectState<'schema> {
	pub fn advance(&mut self, input: &JsonToken) -> Result<(), BiaserError> {
		let JsonSchema::Object { required: _, properties } = self.object_schema else {
			panic!("parsing a JSON object with some other schema than an object schema");
		};

		// Replace self with a temporary value so we can work with our owned copy
		let old_state = std::mem::replace(&mut self.part_state, JsonParserObjectPartState::Finished);

		self.part_state = match (old_state, input) {
			(JsonParserObjectPartState::BeforeKey, JsonToken::CurlyClose) => JsonParserObjectPartState::Finished,
			(JsonParserObjectPartState::BeforeKey, JsonToken::DoubleQuote) => JsonParserObjectPartState::InKey(String::from("")),
			(JsonParserObjectPartState::InKey(k), JsonToken::DoubleQuote) => JsonParserObjectPartState::AfterKey(k),
			// TODO: accept other tokens (e.g. comma?) as next token
			(JsonParserObjectPartState::InKey(k), JsonToken::String(s)) => JsonParserObjectPartState::InKey(format!("{k}{s}")),
			(JsonParserObjectPartState::AfterKey(key), JsonToken::Colon) => {
				let Some(value_schema) = properties.get(&key) else {
					panic!("invalid key");
				};
				JsonParserObjectPartState::InValue {
					key,
					value: Box::new(JsonBiaser::new(value_schema)),
				}
			}
			(JsonParserObjectPartState::InValue { key, value }, JsonToken::Comma) if value.can_end() => {
				self.so_far.insert(key, value.state.value().unwrap());
				JsonParserObjectPartState::BeforeKey
			}
			(JsonParserObjectPartState::InValue { key, value }, JsonToken::CurlyClose)
				if value.can_end() && self.remaining_required_keys().len() == 1 =>
			{
				self.so_far.insert(key, value.state.value().unwrap());
				JsonParserObjectPartState::Finished
			}
			(JsonParserObjectPartState::InValue { key, mut value }, t) => {
				value.advance(t)?;
				JsonParserObjectPartState::InValue { key, value }
			}

			_ => return Err(BiaserError::InvalidToken(input.clone())),
		};
		Ok(())
	}

	fn remaining_required_keys(&self) -> Vec<&'schema String> {
		let JsonSchema::Object { required, properties: _ } = self.object_schema else {
			panic!("parsing a JSON object with some other schema than an object schema");
		};

		required.iter().filter(|r| !self.so_far.contains_key(*r)).collect()
	}

	pub fn next_valid_tokens(&self) -> Vec<JsonToken> {
		match &self.part_state {
			JsonParserObjectPartState::Finished => vec![],
			JsonParserObjectPartState::BeforeKey => {
				if self.remaining_required_keys().is_empty() {
					return vec![JsonToken::CurlyClose];
				}
				vec![JsonToken::DoubleQuote]
			}
			JsonParserObjectPartState::InKey(k) => {
				let rk = self.remaining_required_keys();
				let next_key = rk.first().unwrap();
				let key_remainder = next_key.strip_prefix(k).unwrap_or("");
				if key_remainder.is_empty() {
					// key is finished
					vec![JsonToken::DoubleQuote]
				} else {
					// waiting for a part of the next key still
					vec![JsonToken::AnyOf(vec![key_remainder.to_string()])]
				}
			}
			JsonParserObjectPartState::InValue { key: _, value } => {
				let mut valid_next = value.next_valid_tokens();
				if value.can_end() {
					if self.remaining_required_keys().len() == 1 {
						valid_next.push(JsonToken::CurlyClose);
					} else {
						valid_next.push(JsonToken::Comma);
					}
				}
				valid_next
			}
			JsonParserObjectPartState::AfterKey(_) => vec![JsonToken::Colon],
		}
	}

	fn can_end(&self) -> bool {
		matches!(self.part_state, JsonParserObjectPartState::Finished)
	}
}

impl<'schema> JsonParserState<'schema> {
	pub fn value(&self) -> Option<Value> {
		match self {
			JsonParserState::Start => None,
			JsonParserState::InString(s) => Some(Value::String(s.clone())),
			JsonParserState::InObject(object_state) => {
				let mut object_value = object_state.so_far.clone();
				match &object_state.part_state {
					JsonParserObjectPartState::BeforeKey => {}
					JsonParserObjectPartState::Finished => return Some(Value::Object(object_value)),
					JsonParserObjectPartState::AfterKey(_) => return None, // Would return half an object
					JsonParserObjectPartState::InKey(_) => return None,    // Would return half an object
					JsonParserObjectPartState::InValue { key, value } => {
						if !value.can_end() {
							return None; // Would return half a value
						}
						let Some(jv) = value.state.value() else {
							return None; // No value for key
						};
						object_value.insert(key.clone(), jv);
					}
				}
				Some(Value::Object(object_value))
			}
			JsonParserState::InArray(array_state) => {
				let mut items = array_state.items.clone();
				if let Some(v) = array_state.value_state.state.value() {
					items.push(v);
				}
				Some(Value::Array(items))
			}
			JsonParserState::InInteger(s) => Some(json! { s.parse::<f32>().unwrap() }),
			JsonParserState::End(v) => Some(v.clone()),
		}
	}

	pub fn advance(&mut self, input: &JsonToken, item_schema: Option<&'schema JsonSchema>) -> Result<(), BiaserError> {
		// Replace self with a temporary value so we can work with our owned copy
		let old_self = std::mem::replace(self, JsonParserState::Start);
		*self = match old_self {
			JsonParserState::Start => match input {
				JsonToken::True => JsonParserState::End(json! { true }),
				JsonToken::False => JsonParserState::End(json! { false }),
				JsonToken::Null => JsonParserState::End(json! { null }),
				JsonToken::CurlyOpen => JsonParserState::InObject(JsonParserObjectState {
					so_far: Map::new(),
					object_schema: item_schema.unwrap(),
					part_state: JsonParserObjectPartState::BeforeKey,
				}),
				JsonToken::BracketOpen => JsonParserState::InArray(JsonParserArrayState {
					items: vec![],
					value_state: Box::new(JsonBiaser::new(item_schema.unwrap())),
				}),
				JsonToken::Minus => JsonParserState::InInteger(String::from("-")),
				JsonToken::Digit(n) => JsonParserState::InInteger(format!("{n}")),
				JsonToken::DoubleQuote => JsonParserState::InString(String::from("")),
				_ => return Err(BiaserError::InvalidToken(input.clone())),
			},
			JsonParserState::InString(s) => match input {
				JsonToken::DoubleQuote => JsonParserState::End(json! { s }),
				JsonToken::String(new_string) => {
					if new_string.ends_with('\"') {
						let string_value = format!("{s}{}", new_string.strip_suffix('\"').unwrap_or(""));
						JsonParserState::End(Value::String(string_value))
					} else {
						assert!(!new_string.contains('\"'), "String token may not contain double quote");
						JsonParserState::InString(format!("{s}{new_string}"))
					}
				}
				t => {
					// This could be any other token but now inside the string
					let new_string = t.to_string().unwrap_or(Cow::from(""));
					assert!(!new_string.contains('\"'), "String token may not contain double quote");
					JsonParserState::InString(format!("{s}{new_string}"))
				}
			},
			JsonParserState::InInteger(num_string) => match input {
				JsonToken::Digit(n) => JsonParserState::InInteger(format!("{num_string}{n}")),
				JsonToken::Decimal => JsonParserState::InInteger(format!("{num_string}.")),
				_ => return Err(BiaserError::InvalidToken(input.clone())),
			},
			JsonParserState::InObject(mut object_state) => {
				object_state.advance(input)?;
				JsonParserState::InObject(object_state)
			}
			JsonParserState::InArray(mut array_state) => match input {
				JsonToken::Comma if array_state.value_state.can_end() => {
					if let Some(v) = array_state.value_state.state.value() {
						array_state.items.push(v);
					}
					array_state.value_state.state = JsonParserState::Start;
					JsonParserState::InArray(array_state)
				}
				JsonToken::BracketClose if array_state.value_state.can_end() => {
					if let Some(v) = array_state.value_state.state.value() {
						array_state.items.push(v);
					}
					JsonParserState::End(Value::Array(array_state.items))
				}
				t => {
					if array_state.value_state.advance(input).is_ok() {
						JsonParserState::InArray(array_state)
					} else {
						return Err(BiaserError::InvalidToken(t.clone()));
					}
				}
			},

			JsonParserState::End(_) => return Err(BiaserError::InvalidToken(input.clone())),
		};
		Ok(())
	}
}

impl<'schema> JsonBiaser<'schema> {
	pub fn new(schema: &'schema JsonSchema) -> JsonBiaser<'schema> {
		JsonBiaser {
			schema,
			state: JsonParserState::Start,
		}
	}

	fn child_item_schema(&self) -> Option<&'schema JsonSchema> {
		match &self.schema {
			JsonSchema::Array { items, .. } => Some(items.as_ref()),
			JsonSchema::Object { .. } => Some(self.schema),
			_ => None,
		}
	}

	pub fn advance(&mut self, input: &JsonToken) -> Result<(), BiaserError> {
		self.state.advance(input, self.child_item_schema())
	}

	pub fn can_end(&self) -> bool {
		match self.state {
			JsonParserState::Start => false,
			JsonParserState::InObject(ref object_state) => object_state.can_end(),
			JsonParserState::InArray(ref _array_state) => false,
			JsonParserState::InInteger(ref s) => !s.is_empty() && s.parse::<f32>().is_ok() && !s.ends_with('.'),
			JsonParserState::End(_) => true,
			JsonParserState::InString(_) => false,
		}
	}

	pub fn next_valid_tokens(&self) -> Vec<JsonToken> {
		match &self.state {
			JsonParserState::End(_) => vec![],
			JsonParserState::InObject(object_state) => object_state.next_valid_tokens(),
			JsonParserState::InString(string_so_far) => {
				let JsonSchema::String {
					max_length,
					r#enum: string_values,
				} = self.schema
				else {
					panic!("in string without string schema");
				};

				let max_next_length = max_length.as_ref().map(|max_length| max_length - string_so_far.len());
				if max_next_length == Some(0) {
					// Must end string now
					return vec![JsonToken::DoubleQuote];
				}

				// There are pre-set string values
				if let Some(string_values) = string_values {
					let mut has_valid = false;
					let possible_remainders: Vec<String> = string_values
						.iter()
						.filter_map(|ps| {
							// Remove any strings that are too long to begin with
							if let Some(max_length) = max_length {
								if ps.len() > *max_length {
									return None;
								}
							}

							if ps == string_so_far {
								has_valid = true;
								None
							} else {
								// Find strings that have the current string 'so far' as prefix. Then remove that prefix
								if ps.starts_with(string_so_far) {
									ps.strip_prefix(string_so_far).map(|s| s.to_string())
								} else {
									None
								}
							}
						})
						.collect();

					let mut next_tokens = vec![];
					if !possible_remainders.is_empty() {
						next_tokens.push(JsonToken::AnyOf(possible_remainders));
					}
					if has_valid {
						next_tokens.push(JsonToken::DoubleQuote);
					}
					return next_tokens;
				}

				// Any string
				vec![JsonToken::DoubleQuote, JsonToken::AnyString { max_length: max_next_length }]
			}
			JsonParserState::InArray(array_state) => {
				let JsonSchema::Array { min_items, max_items, .. } = self.schema else {
					panic!();
				};

				let mut valid = array_state.value_state.next_valid_tokens();

				if array_state.value_state.can_end() {
					// If the inner value can end (or must end, then valid = []), expect a comma (if we can accomodate more items)
					if max_items.is_none() || (array_state.items.len() + 1) <= max_items.unwrap() {
						valid.push(JsonToken::Comma);
					}

					// If we have enough items, also allow bracket close
					let has_enough_items = (array_state.items.len() + 1) >= min_items.unwrap_or(0);
					if has_enough_items {
						valid.push(JsonToken::BracketClose);
					}
				}

				valid
			}
			JsonParserState::InInteger(s) => {
				let JsonSchema::Number { max_decimals, min, max } = self.schema else {
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
				let mut digits: Vec<JsonToken> = if s == "-" {
					(1..=9).map(JsonToken::Digit).collect()
				} else {
					(0..=9).map(JsonToken::Digit).collect()
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
					digits.push(JsonToken::Decimal);
				}
				digits
			}
			JsonParserState::Start => match self.schema {
				JsonSchema::Boolean => {
					vec![JsonToken::True, JsonToken::False]
				}
				JsonSchema::Null => {
					vec![JsonToken::Null]
				}
				JsonSchema::Object { .. } => {
					vec![JsonToken::CurlyOpen]
				}
				JsonSchema::String { .. } => {
					vec![JsonToken::DoubleQuote]
				}
				JsonSchema::Number { max, min, max_decimals: _ } => {
					// First digit cannot be zero
					let mut d: Vec<JsonToken> = (1..=9)
						.filter(|d| {
							let df = *d as f64;
							df <= max.unwrap_or(df) && df >= min.unwrap_or(df)
						})
						.map(JsonToken::Digit)
						.collect();

					if min.unwrap_or(-1.0) < 0.0 || max.unwrap_or(-1.0) < 0.0 {
						d.push(JsonToken::Minus);
					}
					d
				}
				JsonSchema::Array { .. } => {
					vec![JsonToken::BracketOpen]
				}
			},
		}
	}
}
