use std::collections::HashMap;
#[cfg(test)]
use std::{path::Path, sync::Arc};

use llm::{
	samplers, InferenceFeedback, InferenceParameters, InferenceSessionConfig, Model, ModelArchitecture, ModelParameters, OutputRequest, Prompt,
	TokenBias, TokenUtf8Buffer,
};

use llm_bias::{
	json::{BiaserError, JsonBiaser, JsonSchema, JsonToken},
	Biaser,
};
use rand::SeedableRng;
use serde_json::Value;
use tracing_test::traced_test;

#[traced_test]
#[test]
pub fn test_parser() {
	let schema = JsonSchema::Boolean;
	let bias = JsonBiaser::new(&schema);
	assert_eq!(bias.next_valid_tokens(), vec![JsonToken::True, JsonToken::False]);
}

#[traced_test]
#[test]
pub fn test_string_parser() {
	let schema = JsonSchema::String {
		max_length: Some(10),
		r#enum: None,
	};
	let mut bias = JsonBiaser::new(&schema);
	assert_eq!(bias.next_valid_tokens(), vec![JsonToken::DoubleQuote]);
	bias.advance(&JsonToken::DoubleQuote).unwrap();
	bias.advance(&JsonToken::String(String::from("hello"))).unwrap();
	bias.advance(&JsonToken::DoubleQuote).unwrap();
	assert_eq!(bias.next_valid_tokens(), vec![]);
}

#[traced_test]
#[test]
pub fn test_string_enum_parser() {
	let words = vec!["foo".to_string(), "bar".to_string(), "baz".to_string()];
	let schema = JsonSchema::String {
		max_length: Some(10),
		r#enum: Some(words.clone()),
	};
	let mut bias = JsonBiaser::new(&schema);
	assert_eq!(bias.next_valid_tokens(), vec![JsonToken::DoubleQuote]);
	bias.advance(&JsonToken::DoubleQuote).unwrap();
	assert_eq!(bias.next_valid_tokens(), vec![JsonToken::AnyOf(words)]);
	bias.advance(&JsonToken::String(String::from("foo"))).unwrap();
	bias.advance(&JsonToken::DoubleQuote).unwrap();
	assert_eq!(bias.next_valid_tokens(), vec![]);
}

#[traced_test]
#[test]
pub fn test_empty_object_parser() {
	let schema = JsonSchema::Object {
		required: vec![],
		properties: HashMap::new(),
	};

	let mut biaser = JsonBiaser::new(&schema);

	// '{}'
	assert_eq!(biaser.next_valid_tokens(), vec![JsonToken::CurlyOpen]);
	biaser.advance(&JsonToken::CurlyOpen).unwrap();
	assert_eq!(biaser.next_valid_tokens(), vec![JsonToken::CurlyClose]);
	biaser.advance(&JsonToken::CurlyClose).unwrap();
	assert_eq!(biaser.next_valid_tokens(), vec![]);
}

#[traced_test]
#[test]
pub fn test_object_parser() {
	let mut fields = HashMap::new();
	fields.insert(
		"first_name".to_string(),
		Box::new(JsonSchema::String {
			max_length: Some(5),
			r#enum: None,
		}),
	);
	fields.insert(
		"last_name".to_string(),
		Box::new(JsonSchema::String {
			max_length: Some(7),
			r#enum: None,
		}),
	);
	let schema = JsonSchema::Object {
		required: vec!["first_name".to_string(), "last_name".to_string()],
		properties: fields,
	};

	let mut biaser = JsonBiaser::new(&schema);

	// '{"'
	assert_eq!(biaser.next_valid_tokens(), vec![JsonToken::CurlyOpen]);
	biaser.advance(&JsonToken::CurlyOpen).unwrap();
	assert_eq!(biaser.next_valid_tokens(), vec![JsonToken::DoubleQuote]);
	biaser.advance(&JsonToken::DoubleQuote).unwrap();

	// First we expect the 'first_name' key
	assert_eq!(biaser.next_valid_tokens(), vec![JsonToken::AnyOf(vec!["first_name".to_string()])]);
	biaser.advance(&JsonToken::String("first_".to_string())).unwrap();
	assert_eq!(biaser.next_valid_tokens(), vec![JsonToken::AnyOf(vec!["name".to_string()])]);
	biaser.advance(&JsonToken::String("name".to_string())).unwrap();
	assert_eq!(biaser.next_valid_tokens(), vec![JsonToken::DoubleQuote]);
	biaser.advance(&JsonToken::DoubleQuote).unwrap();
	assert_eq!(biaser.next_valid_tokens(), vec![JsonToken::Colon]);
	biaser.advance(&JsonToken::Colon).unwrap(); // {"first_name": at this point
	assert_eq!(biaser.next_valid_tokens(), vec![JsonToken::DoubleQuote]);
	biaser.advance(&JsonToken::DoubleQuote).unwrap();
	biaser.advance(&JsonToken::String("tommy".to_string())).unwrap();
	biaser.advance(&JsonToken::DoubleQuote).unwrap(); // {"first_name":"tommy" at this point
	assert_eq!(biaser.next_valid_tokens(), vec![JsonToken::Comma]); // comma, nothing else, because we need that last_name key
	biaser.advance(&JsonToken::Comma).unwrap(); // {"first_name":"tommy", at this point

	assert_eq!(biaser.next_valid_tokens(), vec![JsonToken::DoubleQuote]);
	biaser.advance(&JsonToken::DoubleQuote).unwrap();
	assert_eq!(biaser.next_valid_tokens(), vec![JsonToken::AnyOf(vec!["last_name".to_string()])]);
	biaser.advance(&JsonToken::String("last_name".to_string())).unwrap();
	assert_eq!(biaser.next_valid_tokens(), vec![JsonToken::DoubleQuote]);
	biaser.advance(&JsonToken::DoubleQuote).unwrap(); // {"first_name":"tommy","last_name" at this point
	assert_eq!(biaser.next_valid_tokens(), vec![JsonToken::Colon]);
	biaser.advance(&JsonToken::Colon).unwrap();
	biaser.advance(&JsonToken::DoubleQuote).unwrap();
	biaser.advance(&JsonToken::String("vorst".to_string())).unwrap();
	biaser.advance(&JsonToken::DoubleQuote).unwrap();
	assert_eq!(biaser.next_valid_tokens(), vec![JsonToken::CurlyClose]); // All keys have been gathered
	biaser.advance(&JsonToken::CurlyClose).unwrap();
	assert_eq!(biaser.next_valid_tokens(), vec![]); // Object is done
	assert!(biaser.can_end());
}

#[traced_test]
#[test]
pub fn test_array_parser() {
	let schema = JsonSchema::Array {
		items: Box::new(JsonSchema::Boolean),
		min_items: Some(2),
		max_items: Some(3),
	};
	let mut bias = JsonBiaser::new(&schema);

	assert_eq!(bias.next_valid_tokens(), vec![JsonToken::BracketOpen]);
	bias.advance(&JsonToken::BracketOpen).unwrap();
	assert_eq!(bias.next_valid_tokens(), vec![JsonToken::True, JsonToken::False]);
	bias.advance(&JsonToken::True).unwrap();
	assert_eq!(bias.next_valid_tokens(), vec![JsonToken::Comma]);
	bias.advance(&JsonToken::Comma).unwrap();
	assert_eq!(bias.next_valid_tokens(), vec![JsonToken::True, JsonToken::False]);
	bias.advance(&JsonToken::False).unwrap();
	assert_eq!(bias.next_valid_tokens(), vec![JsonToken::Comma, JsonToken::BracketClose]);
	bias.advance(&JsonToken::Comma).unwrap();
	assert_eq!(bias.next_valid_tokens(), vec![JsonToken::True, JsonToken::False]);
	bias.advance(&JsonToken::False).unwrap();
	assert_eq!(bias.next_valid_tokens(), vec![JsonToken::Comma, JsonToken::BracketClose]);
	bias.advance(&JsonToken::Comma).unwrap();
	assert_eq!(bias.next_valid_tokens(), vec![JsonToken::True, JsonToken::False]);
	bias.advance(&JsonToken::False).unwrap();
	assert_eq!(bias.next_valid_tokens(), vec![JsonToken::BracketClose]);
	bias.advance(&JsonToken::BracketClose).unwrap();
	assert_eq!(bias.next_valid_tokens(), vec![]);
	assert!(bias.can_end());
}

static MODEL_PATH: &str = "../data/pythia-160m-q4_0.bin";

#[traced_test]
#[test]
pub fn test_json_biaser_objects() {
	let model = llm::load_dynamic(
		ModelArchitecture::GptNeoX,
		Path::new(MODEL_PATH),
		llm::VocabularySource::Model,
		ModelParameters::default(),
		|_progress| {},
	)
	.unwrap();

	test_json_bias(
		JsonSchema::Object {
			required: vec![],
			properties: HashMap::new(),
		},
		model.as_ref(),
	);

	let mut fields = HashMap::new();
	fields.insert(
		"first_name".to_string(),
		Box::new(JsonSchema::String {
			max_length: Some(5),
			r#enum: None,
		}),
	);
	fields.insert(
		"last_name".to_string(),
		Box::new(JsonSchema::String {
			max_length: Some(7),
			r#enum: None,
		}),
	);

	test_json_bias(
		JsonSchema::Object {
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
		Path::new(MODEL_PATH),
		llm::VocabularySource::Model,
		ModelParameters::default(),
		|_progress| {},
	)
	.unwrap();

	test_json_bias(JsonSchema::Boolean, model.as_ref());

	test_json_bias(JsonSchema::Null, model.as_ref());

	test_json_bias(
		JsonSchema::String {
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
		JsonSchema::String {
			max_length: Some(20),
			r#enum: None,
		},
		model.as_ref(),
	);

	test_json_bias(
		JsonSchema::Number {
			max_decimals: Some(2),
			min: Some(-0.32),
			max: Some(5.87),
		},
		model.as_ref(),
	);

	// Array-of-bools
	test_json_bias(
		JsonSchema::Array {
			items: Box::new(JsonSchema::Boolean),
			min_items: Some(2),
			max_items: Some(5),
		},
		model.as_ref(),
	);

	// Array-of-array-of-numbers
	test_json_bias(
		JsonSchema::Array {
			items: Box::new(JsonSchema::Array {
				items: Box::new(JsonSchema::Number {
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

fn test_json_bias(schema: JsonSchema, model: &dyn Model) {
	for seed in [1340, 1338, 1339] {
		let mut rng = rand::rngs::StdRng::seed_from_u64(seed); // Deterministic for tests

		let mut bias = JsonBiaser::new(&schema);
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
					let out_json_token = JsonToken::from_token(vocab, out_token).expect("valid token");

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
