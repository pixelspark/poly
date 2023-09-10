use std::{
	collections::HashMap,
	str::FromStr,
	sync::{Mutex, Once},
};
#[cfg(test)]
use std::{path::Path, sync::Arc};

use llm::{
	samplers::{llm_samplers::types::SamplerChain, ConfiguredSamplers},
	InferenceError, InferenceFeedback, InferenceParameters, InferenceSessionConfig, Model, ModelArchitecture, ModelParameters, OutputRequest, Prompt,
	TokenUtf8Buffer,
};

use poly_bias::{
	json::{BiaserError, JsonBiaser, JsonSchema, JsonToken},
	Biaser,
};
use rand::SeedableRng;
use serde_json::Value;

static INIT: Once = Once::new();

pub fn setup() {
	INIT.call_once(|| {
		tracing_subscriber::fmt::init();
	});
}

#[test]
pub fn test_parser() {
	setup();
	let schema = JsonSchema::Boolean;
	let bias = JsonBiaser::new(&schema);
	assert_eq!(bias.next_valid_tokens(), vec![JsonToken::True, JsonToken::False]);
}

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

#[test]
pub fn test_string_enum_parser() {
	setup();
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

#[test]
pub fn test_empty_object_parser() {
	setup();
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

#[test]
pub fn test_nested_object_parser() {
	setup();
	let schema = JsonSchema::Object {
		required: vec!["car".to_string()],
		properties: {
			let mut hn = HashMap::new();
			hn.insert(
				"car".to_string(),
				Box::new(JsonSchema::Object {
					required: vec!["name".to_string()],
					properties: {
						let mut hn = HashMap::new();
						hn.insert(
							"name".to_string(),
							Box::new(JsonSchema::String {
								max_length: None,
								r#enum: None,
							}),
						);
						hn
					},
				}),
			);
			hn
		},
	};

	let mut biaser = JsonBiaser::new(&schema);

	// {"car":{"name":"car mccarface"}}
	let stream = vec![
		JsonToken::CurlyOpen,
		JsonToken::DoubleQuote,
		JsonToken::String("car".to_string()),
		JsonToken::DoubleQuote,
		JsonToken::Colon,
		JsonToken::CurlyOpen,
		JsonToken::DoubleQuote,
		JsonToken::String("name".to_string()),
		JsonToken::DoubleQuote,
		JsonToken::Colon,
		JsonToken::DoubleQuote,
		JsonToken::String("car mccarface".to_string()),
		JsonToken::DoubleQuote,
		JsonToken::CurlyClose,
		JsonToken::CurlyClose,
	];

	for token in stream.iter() {
		biaser.advance(token).unwrap();
	}
	assert!(biaser.next_valid_tokens().is_empty());
}

#[test]
pub fn test_object_parser() {
	setup();
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

#[test]
pub fn test_array_parser() {
	setup();
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

static MODEL_PATH: &str = "../data/gpt2.bin";

#[test]
pub fn test_json_biaser_objects() {
	setup();
	let model = llm::load_dynamic(
		Some(ModelArchitecture::Gpt2),
		Path::new(MODEL_PATH),
		llm::TokenizerSource::Embedded,
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

#[test]
pub fn test_json_biaser() {
	setup();
	let model = llm::load_dynamic(
		Some(ModelArchitecture::Gpt2),
		Path::new(MODEL_PATH),
		llm::TokenizerSource::Embedded,
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
	setup();
	for seed in [1340, 1338, 1339] {
		println!("Run with seed {seed}");
		let mut rng = rand::rngs::StdRng::seed_from_u64(seed); // Deterministic for tests

		let mut bias = JsonBiaser::new(&schema);
		let mut session = model.start_session(InferenceSessionConfig::default());
		let vocab = model.tokenizer();

		session
			.feed_prompt(
				model,
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

			let cs = ConfiguredSamplers::from_str("").unwrap();

			println!("next_valid_tokens: {next_valid_tokens:?}");
			let flat_bias = llm::samplers::llm_samplers::samplers::SampleFlatBias::new(next_valid_tokens);
			let mut samplers = SamplerChain::new();
			samplers.push_sampler(flat_bias);
			samplers += cs.builder.into_chain();

			let inference_params = InferenceParameters {
				sampler: Arc::new(Mutex::new(samplers)),
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
				Err(InferenceError::EndOfText) => {
					break;
				}
				Err(e) => {
					panic!("inference error: {e:?}");
				}
			}
		}
		println!("Finish: {}\n", result);
		serde_json::from_str::<Value>(&result).expect("valid JSON");
	}
}
