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
	json_vocab: JSONVocabulary,
	state: JSONParserState,
}

#[derive(Debug, Clone)]
pub struct JSONVocabulary {
	true_token: TokenId,
	false_token: TokenId,
	null_token: TokenId,
	curly_open_token: TokenId,
	curly_close_token: TokenId,
	bracket_open_token: TokenId,
	bracket_close_token: TokenId,
	comma_token: TokenId,
	minus_token: TokenId,
	number_tokens: Vec<TokenId>,
	eot_token: TokenId,
}

impl JSONVocabulary {
	pub fn from(vocab: &Vocabulary, eot_token: TokenId) -> JSONVocabulary {
		JSONVocabulary {
			true_token: single_token_id_for(vocab, "true").unwrap(),
			false_token: single_token_id_for(vocab, "false").unwrap(),
			null_token: single_token_id_for(vocab, "null").unwrap(),
			curly_open_token: single_token_id_for(vocab, "{").unwrap(),
			curly_close_token: single_token_id_for(vocab, "}").unwrap(),
			bracket_open_token: single_token_id_for(vocab, "[").unwrap(),
			bracket_close_token: single_token_id_for(vocab, "]").unwrap(),
			comma_token: single_token_id_for(vocab, ",").unwrap(),
			minus_token: single_token_id_for(vocab, "-").unwrap(),
			number_tokens: (0..=9).map(|n| single_token_id_for(vocab, &format!("{n}")).unwrap()).collect(),
			eot_token,
		}
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
	InvalidToken(TokenId),
}

impl JSONParserState {
	pub fn advance(&mut self, json_vocab: &JSONVocabulary, input: TokenId, item_schema: Option<JSONSchema>) -> Result<(), BiaserError> {
		*self = match self {
			JSONParserState::Start => {
				if input == json_vocab.true_token {
					JSONParserState::End(json! { true })
				} else if input == json_vocab.false_token {
					JSONParserState::End(json! { false })
				} else if input == json_vocab.null_token {
					JSONParserState::End(json! { null })
				} else if input == json_vocab.curly_open_token {
					JSONParserState::InObject
				} else if input == json_vocab.bracket_open_token {
					JSONParserState::InArray(JSONParserArrayState {
						items: vec![],
						value_state: Box::new(JSONBiaser::new(item_schema.unwrap(), json_vocab.clone())),
					})
				} else if input == json_vocab.minus_token {
					JSONParserState::InInteger(false, 0)
				} else if let Some(n) = json_vocab.number_tokens.iter().find(|x| **x == input) {
					JSONParserState::InInteger(true, *n as isize)
				} else {
					return Err(BiaserError::InvalidToken(input));
				}
			}
			JSONParserState::InInteger(sign, digits) => {
				if input == json_vocab.eot_token {
					let n = (if *sign { 1 } else { -1 }) * *digits;
					JSONParserState::End(json! { n })
				} else if let Some(n) = json_vocab.number_tokens.iter().find(|x| **x == input) {
					JSONParserState::InInteger(*sign, *digits * 10 + (*n as isize))
				} else {
					return Err(BiaserError::InvalidToken(input));
				}
			}
			JSONParserState::InObject => {
				if input == json_vocab.curly_close_token {
					JSONParserState::End(json! { {} })
				} else {
					return Err(BiaserError::InvalidToken(input));
				}
			}
			JSONParserState::InArray(array_state) => {
				let mut array_state: JSONParserArrayState = array_state.clone();
				let next_valid_item_tokens = array_state.value_state.next_valid_tokens();

				let old_array_state = array_state.value_state.state.clone();

				let next_state = {
					if let JSONParserState::End(v) = array_state.value_state.state {
						array_state.items.push(v);
						if input == json_vocab.comma_token {
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
					} else if input == json_vocab.bracket_close_token {
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
			JSONParserState::ExpectComma => {
				if input == json_vocab.comma_token {
					JSONParserState::Start
				} else {
					return Err(BiaserError::InvalidToken(input));
				}
			}
			JSONParserState::End(_) => return Err(BiaserError::InvalidToken(input)),
		};
		Ok(())
	}
}

impl JSONBiaser {
	pub fn new(schema: JSONSchema, json_vocab: JSONVocabulary) -> JSONBiaser {
		JSONBiaser {
			schema,
			json_vocab,
			state: JSONParserState::Start,
		}
	}

	fn child_item_schema(&self) -> Option<JSONSchema> {
		match &self.schema {
			JSONSchema::Array { items, .. } => Some(*items.clone()),
			_ => None,
		}
	}

	pub fn advance(&mut self, input: TokenId) -> Result<(), BiaserError> {
		self.state.advance(&self.json_vocab, input, self.child_item_schema())
	}

	pub fn next_valid_tokens(&self) -> Vec<TokenId> {
		match &self.state {
			JSONParserState::End(_) => vec![],
			JSONParserState::ExpectComma => vec![self.json_vocab.comma_token],
			JSONParserState::InObject => vec![self.json_vocab.curly_close_token],
			JSONParserState::InArray(array_state) => {
				let mut valid = array_state.value_state.next_valid_tokens();

				// If a value is done, we expect a comma first
				if let JSONParserState::End(_) = array_state.value_state.state {
					valid.clear();
					// TODO: max_items
					valid.push(self.json_vocab.comma_token);
				}

				let JSONSchema::Array { min_items, .. } = self.schema else {
					panic!();
				};

				if let Some(min_items) = min_items {
					if min_items < array_state.items.len() {
						// We have enough items, you may close the array
						valid.push(self.json_vocab.bracket_close_token);
					}
				} else {
					// No minimum number of items, you may close the array
					valid.push(self.json_vocab.bracket_close_token);
				}
				valid
			}
			JSONParserState::InInteger(_sign, digits) => {
				// Limit the length of a number literal to what fits in a 32 bit integer
				if *digits >= u32::MAX as isize {
					return vec![self.json_vocab.eot_token];
				}
				let mut d = self.json_vocab.number_tokens.clone();
				d.push(self.json_vocab.eot_token);
				d
			}
			JSONParserState::Start => match self.schema {
				JSONSchema::Boolean => {
					vec![self.json_vocab.true_token, self.json_vocab.false_token]
				}
				JSONSchema::Null => {
					vec![self.json_vocab.null_token]
				}
				JSONSchema::Object => {
					vec![self.json_vocab.curly_open_token]
				}
				JSONSchema::Number => {
					let mut d = self.json_vocab.number_tokens.clone();
					d.push(self.json_vocab.minus_token);
					d.push(self.json_vocab.eot_token);
					d
				}
				JSONSchema::Array { .. } => {
					vec![self.json_vocab.bracket_open_token]
				}
			},
		}
	}
}

#[cfg(test)]
mod tests {
	use std::{path::Path, sync::Arc};

	use llm::{
		samplers, InferenceFeedback, InferenceParameters, InferenceSessionConfig, Model, ModelArchitecture, ModelParameters, OutputRequest, Prompt,
		TokenBias, TokenUtf8Buffer,
	};

	use crate::bias::{single_token_id_for, BiaserError, JSONVocabulary};

	use super::{JSONBiaser, JSONSchema};

	#[test]
	pub fn test_json_biaser() {
		let model = llm::load_dynamic(
			ModelArchitecture::GptNeoX,
			Path::new("data/pythia-160m-q4_0.bin"),
			llm::VocabularySource::Model,
			ModelParameters::default(),
			|_progress| {},
		)
		.unwrap();
		let vocab = model.vocabulary();
		let json_vocab = JSONVocabulary::from(vocab, model.eot_token_id());
		dbg!(&json_vocab, model.eot_token_id());

		let mut bias = JSONBiaser::new(JSONSchema::Boolean, json_vocab.clone());
		bias.advance(single_token_id_for(vocab, "false").unwrap()).unwrap();
		dbg!(&bias);

		// test_json_bias(JSONSchema::Boolean, model.as_ref(), json_vocab.clone());

		// test_json_bias(JSONSchema::Null, model.as_ref(), json_vocab.clone());

		// test_json_bias(JSONSchema::Object, model.as_ref(), json_vocab.clone());

		// test_json_bias(JSONSchema::Number, model.as_ref(), json_vocab.clone());

		// Array-of-bools
		test_json_bias(
			JSONSchema::Array {
				items: Box::new(JSONSchema::Boolean),
				min_items: Some(2),
			},
			model.as_ref(),
			json_vocab,
		);

		// // Array-of-array-of-bools
		// test_json_bias(
		// 	JSONSchema::Array {
		// 		items: Box::new(JSONSchema::Array {
		// 			items: Box::new(JSONSchema::Boolean),
		// 			min_items: Some(2),
		// 		}),
		// 		min_items: Some(2),
		// 	},
		// 	model.as_ref(),
		// 	json_vocab.clone(),
		// );
	}

	fn test_json_bias(schema: JSONSchema, model: &dyn Model, json_vocab: JSONVocabulary) {
		let mut bias = JSONBiaser::new(schema, json_vocab);
		let mut session = model.start_session(InferenceSessionConfig::default());
		let vocab = model.vocabulary();

		session
			.feed_prompt(
				model,
				&InferenceParameters::default(),
				Prompt::Text("Feyenoord is better than Ajax. "),
				&mut OutputRequest::default(),
				|_| -> Result<InferenceFeedback, BiaserError> { Ok(InferenceFeedback::Continue) },
			)
			.unwrap();

		let mut rng = rand::thread_rng();
		let mut result = String::new();
		let mut result_buffer = TokenUtf8Buffer::new();
		loop {
			let next_valid_tokens = bias.next_valid_tokens();
			if next_valid_tokens.is_empty() {
				break;
			}
			let sampler = samplers::TopPTopK {
				bias_tokens: TokenBias::new(next_valid_tokens.iter().map(|t| (*t, 10000.0)).collect()),
				..Default::default()
			};
			let inference_params = InferenceParameters {
				sampler: Arc::new(sampler),
				..InferenceParameters::default()
			};

			if let Ok(out) = session.infer_next_token(model, &inference_params, &mut OutputRequest::default(), &mut rng) {
				let out_token = vocab.id(&out).unwrap();
				if out_token == model.eot_token_id() {
					println!("END OF TEXT");
					break;
				}
				println!(
					"out_token={out_token} = {}",
					String::from_utf8_lossy(&vocab.decode(vec![out_token], false))
				);
				bias.advance(out_token).unwrap();
				if let Some(output) = result_buffer.push(&out) {
					result.push_str(&output);
				}
				println!(
					"== TOKEN: {:?} next valid tokens: {:?} state: {:?}\n",
					String::from_utf8_lossy(&vocab.decode(vec![out_token], false)),
					String::from_utf8_lossy(&vocab.decode(bias.next_valid_tokens(), false)),
					bias.state
				);
			} else {
				// End of text
				bias.advance(model.eot_token_id()).unwrap();
				break;
			}
		}
		println!("Result={} json state={:?}", result, bias.state);
	}
}
