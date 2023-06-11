use std::collections::HashMap;
#[cfg(test)]
use std::{path::Path, sync::Arc};

use llm::{
	samplers, InferenceFeedback, InferenceParameters, InferenceSessionConfig, Model, ModelArchitecture, ModelParameters, OutputRequest, Prompt,
	TokenBias, TokenUtf8Buffer,
};

use llmd::bias::{Biaser, BiaserError, JSONToken};

use llmd::bias::{JSONBiaser, JSONSchema};
use rand::SeedableRng;
use serde_json::Value;
use tracing_test::traced_test;

#[traced_test]
#[test]
pub fn test_parser() {
	let schema = JSONSchema::Boolean;
	let bias = JSONBiaser::new(&schema);
	assert_eq!(bias.next_valid_tokens(), vec![JSONToken::True, JSONToken::False]);
}

#[traced_test]
#[test]
pub fn test_string_parser() {
	let schema = JSONSchema::String {
		max_length: Some(10),
		r#enum: None,
	};
	let mut bias = JSONBiaser::new(&schema);
	assert_eq!(bias.next_valid_tokens(), vec![JSONToken::DoubleQuote]);
	bias.advance(&JSONToken::DoubleQuote).unwrap();
	bias.advance(&JSONToken::String(String::from("hello"))).unwrap();
	bias.advance(&JSONToken::DoubleQuote).unwrap();
	assert_eq!(bias.next_valid_tokens(), vec![]);
}

#[traced_test]
#[test]
pub fn test_string_enum_parser() {
	let words = vec!["foo".to_string(), "bar".to_string(), "baz".to_string()];
	let schema = JSONSchema::String {
		max_length: Some(10),
		r#enum: Some(words.clone()),
	};
	let mut bias = JSONBiaser::new(&schema);
	assert_eq!(bias.next_valid_tokens(), vec![JSONToken::DoubleQuote]);
	bias.advance(&JSONToken::DoubleQuote).unwrap();
	assert_eq!(bias.next_valid_tokens(), vec![JSONToken::AnyOf(words)]);
	bias.advance(&JSONToken::String(String::from("foo"))).unwrap();
	bias.advance(&JSONToken::DoubleQuote).unwrap();
	assert_eq!(bias.next_valid_tokens(), vec![]);
}

#[traced_test]
#[test]
pub fn test_empty_object_parser() {
	let schema = JSONSchema::Object {
		required: vec![],
		properties: HashMap::new(),
	};

	let mut biaser = JSONBiaser::new(&schema);

	// '{}'
	assert_eq!(biaser.next_valid_tokens(), vec![JSONToken::CurlyOpen]);
	biaser.advance(&JSONToken::CurlyOpen).unwrap();
	assert_eq!(biaser.next_valid_tokens(), vec![JSONToken::CurlyClose]);
	biaser.advance(&JSONToken::CurlyClose).unwrap();
	assert_eq!(biaser.next_valid_tokens(), vec![]);
}

#[traced_test]
#[test]
pub fn test_object_parser() {
	let mut fields = HashMap::new();
	fields.insert(
		"first_name".to_string(),
		Box::new(JSONSchema::String {
			max_length: Some(5),
			r#enum: None,
		}),
	);
	fields.insert(
		"last_name".to_string(),
		Box::new(JSONSchema::String {
			max_length: Some(7),
			r#enum: None,
		}),
	);
	let schema = JSONSchema::Object {
		required: vec!["first_name".to_string(), "last_name".to_string()],
		properties: fields,
	};

	let mut biaser = JSONBiaser::new(&schema);

	// '{"'
	assert_eq!(biaser.next_valid_tokens(), vec![JSONToken::CurlyOpen]);
	biaser.advance(&JSONToken::CurlyOpen).unwrap();
	assert_eq!(biaser.next_valid_tokens(), vec![JSONToken::DoubleQuote]);
	biaser.advance(&JSONToken::DoubleQuote).unwrap();

	// First we expect the 'first_name' key
	assert_eq!(biaser.next_valid_tokens(), vec![JSONToken::String("first_name".to_string())]);
	biaser.advance(&JSONToken::String("first_".to_string())).unwrap();
	assert_eq!(biaser.next_valid_tokens(), vec![JSONToken::String("name".to_string())]);
	biaser.advance(&JSONToken::String("name".to_string())).unwrap();
	assert_eq!(biaser.next_valid_tokens(), vec![JSONToken::DoubleQuote]);
	biaser.advance(&JSONToken::DoubleQuote).unwrap();
	assert_eq!(biaser.next_valid_tokens(), vec![JSONToken::Colon]);
	biaser.advance(&JSONToken::Colon).unwrap(); // {"first_name": at this point
	assert_eq!(biaser.next_valid_tokens(), vec![JSONToken::DoubleQuote]);
	biaser.advance(&JSONToken::DoubleQuote).unwrap();
	biaser.advance(&JSONToken::String("tommy".to_string())).unwrap();
	biaser.advance(&JSONToken::DoubleQuote).unwrap(); // {"first_name":"tommy" at this point
	assert_eq!(biaser.next_valid_tokens(), vec![JSONToken::Comma]); // comma, nothing else, because we need that last_name key
	biaser.advance(&JSONToken::Comma).unwrap(); // {"first_name":"tommy", at this point

	assert_eq!(biaser.next_valid_tokens(), vec![JSONToken::DoubleQuote]);
	biaser.advance(&JSONToken::DoubleQuote).unwrap();
	assert_eq!(biaser.next_valid_tokens(), vec![JSONToken::String("last_name".to_string())]);
	biaser.advance(&JSONToken::String("last_name".to_string())).unwrap();
	assert_eq!(biaser.next_valid_tokens(), vec![JSONToken::DoubleQuote]);
	biaser.advance(&JSONToken::DoubleQuote).unwrap(); // {"first_name":"tommy","last_name" at this point
	assert_eq!(biaser.next_valid_tokens(), vec![JSONToken::Colon]);
	biaser.advance(&JSONToken::Colon).unwrap();
	biaser.advance(&JSONToken::DoubleQuote).unwrap();
	biaser.advance(&JSONToken::String("vorst".to_string())).unwrap();
	biaser.advance(&JSONToken::DoubleQuote).unwrap();
	assert_eq!(biaser.next_valid_tokens(), vec![JSONToken::CurlyClose]); // All keys have been gathered
	biaser.advance(&JSONToken::CurlyClose).unwrap();
	assert_eq!(biaser.next_valid_tokens(), vec![]); // Object is done
	assert!(biaser.can_end());

	println!("{:?}", biaser.next_valid_tokens());
}

#[traced_test]
#[test]
pub fn test_array_parser() {
	let schema = JSONSchema::Array {
		items: Box::new(JSONSchema::Boolean),
		min_items: Some(2),
		max_items: Some(3),
	};
	let mut bias = JSONBiaser::new(&schema);

	assert_eq!(bias.next_valid_tokens(), vec![JSONToken::BracketOpen]);
	bias.advance(&JSONToken::BracketOpen).unwrap();
	assert_eq!(bias.next_valid_tokens(), vec![JSONToken::True, JSONToken::False]);
	bias.advance(&JSONToken::True).unwrap();
	assert_eq!(bias.next_valid_tokens(), vec![JSONToken::Comma]);
	bias.advance(&JSONToken::Comma).unwrap();
	assert_eq!(bias.next_valid_tokens(), vec![JSONToken::True, JSONToken::False]);
	bias.advance(&JSONToken::False).unwrap();
	assert_eq!(bias.next_valid_tokens(), vec![JSONToken::Comma, JSONToken::BracketClose]);
	bias.advance(&JSONToken::Comma).unwrap();
	assert_eq!(bias.next_valid_tokens(), vec![JSONToken::True, JSONToken::False]);
	bias.advance(&JSONToken::False).unwrap();
	assert_eq!(bias.next_valid_tokens(), vec![JSONToken::Comma, JSONToken::BracketClose]);
	bias.advance(&JSONToken::Comma).unwrap();
	assert_eq!(bias.next_valid_tokens(), vec![JSONToken::True, JSONToken::False]);
	bias.advance(&JSONToken::False).unwrap();
	assert_eq!(bias.next_valid_tokens(), vec![JSONToken::BracketClose]);
	bias.advance(&JSONToken::BracketClose).unwrap();
	assert_eq!(bias.next_valid_tokens(), vec![]);
	assert!(bias.can_end());
}

#[traced_test]
#[test]
pub fn test_json_biaser_objects() {
	let model = llm::load_dynamic(
		ModelArchitecture::GptNeoX,
		Path::new("data/pythia-160m-q4_0.bin"),
		llm::VocabularySource::Model,
		ModelParameters::default(),
		|_progress| {},
	)
	.unwrap();

	test_json_bias(
		JSONSchema::Object {
			required: vec![],
			properties: HashMap::new(),
		},
		model.as_ref(),
	);

	let mut fields = HashMap::new();
	fields.insert(
		"first_name".to_string(),
		Box::new(JSONSchema::String {
			max_length: Some(5),
			r#enum: None,
		}),
	);
	fields.insert(
		"last_name".to_string(),
		Box::new(JSONSchema::String {
			max_length: Some(7),
			r#enum: None,
		}),
	);

	test_json_bias(
		JSONSchema::Object {
			required: fields.keys().cloned().collect(),
			properties: fields,
		},
		model.as_ref(),
	);
}

#[traced_test]
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

	test_json_bias(JSONSchema::Boolean, model.as_ref());

	test_json_bias(JSONSchema::Null, model.as_ref());

	test_json_bias(
		JSONSchema::String {
			max_length: Some(20),
			r#enum: Some(vec![
				"The quick brown fox".to_string(),
				"Jumped over the".to_string(),
				"The quick".to_string(),
			]),
		},
		model.as_ref(),
	);

	test_json_bias(
		JSONSchema::String {
			max_length: Some(20),
			r#enum: None,
		},
		model.as_ref(),
	);

	test_json_bias(
		JSONSchema::Number {
			max_decimals: Some(2),
			min: Some(-0.32),
			max: Some(5.87),
		},
		model.as_ref(),
	);

	// Array-of-bools
	test_json_bias(
		JSONSchema::Array {
			items: Box::new(JSONSchema::Boolean),
			min_items: Some(2),
			max_items: Some(5),
		},
		model.as_ref(),
	);

	// Array-of-array-of-numbers
	test_json_bias(
		JSONSchema::Array {
			items: Box::new(JSONSchema::Array {
				items: Box::new(JSONSchema::Number {
					max_decimals: Some(2),
					min: Some(-10.0),
					max: Some(10.0),
				}),
				min_items: Some(2),
				max_items: Some(4),
			}),
			min_items: Some(1),
			max_items: Some(3),
		},
		model.as_ref(),
	);
}

fn test_json_bias(schema: JSONSchema, model: &dyn Model) {
	for seed in [1340, 1338, 1339] {
		let mut rng = rand::rngs::StdRng::seed_from_u64(seed); // Deterministic for tests

		let mut bias = JSONBiaser::new(&schema);
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

		let mut result = String::new();
		let mut result_buffer = TokenUtf8Buffer::new();

		loop {
			let next_valid_tokens = bias.bias(vocab, model.eot_token_id());
			if next_valid_tokens.is_empty() {
				break;
			}

			if next_valid_tokens.len() < 15 {
				println!(
					"next_valid_tokens={}",
					next_valid_tokens
						.iter()
						.map(|tid| String::from_utf8_lossy(&vocab.decode(vec![tid.0], false)).to_string())
						.collect::<Vec<String>>()
						.join(" ")
				);
			} else {
				println!("next_valid_tokens: {} tokens", next_valid_tokens.len());
			}

			let sampler = samplers::TopPTopK {
				bias_tokens: TokenBias::new(next_valid_tokens),
				..Default::default()
			};
			let inference_params = InferenceParameters {
				sampler: Arc::new(sampler),
				..InferenceParameters::default()
			};

			match session.infer_next_token(model, &inference_params, &mut OutputRequest::default(), &mut rng) {
				Ok(out) => {
					let out_token = vocab.id(&out).unwrap();
					if out_token == model.eot_token_id() {
						println!("EOT token");
						break;
					}
					let out_json_token = JSONToken::from_token(vocab, out_token).expect("valid token");

					bias.advance(&out_json_token).expect("advance");
					if let Some(output) = result_buffer.push(&out) {
						result.push_str(&output);
					}

					if vocab.decode(vec![out_token], false).is_empty() {
						panic!("empty token generated: {out_token}");
					}

					println!(
						"Token: {}, RESULT: {result} next valid tokens: {:?}",
						String::from_utf8_lossy(&vocab.decode(vec![out_token], false)),
						bias.next_valid_tokens(),
					);
				}
				Err(e) => {
					// End of text
					println!("End {e:?}");
					break;
				}
			}
		}
		println!("Finish: {}\n", result);
		serde_json::from_str::<Value>(&result).expect("valid JSON");
	}
}
