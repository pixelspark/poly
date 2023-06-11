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

#[test]
pub fn test_parser() {
	let schema = JSONSchema::Boolean;
	let bias = JSONBiaser::new(&schema);
	assert_eq!(bias.next_valid_tokens(), vec![JSONToken::True, JSONToken::False]);
}

#[test]
pub fn test_string_parser() {
	let schema = JSONSchema::String { max_length: Some(10) };
	let mut bias = JSONBiaser::new(&schema);
	assert_eq!(bias.next_valid_tokens(), vec![JSONToken::DoubleQuote]);
	bias.advance(&JSONToken::DoubleQuote).unwrap();
	bias.advance(&JSONToken::String(String::from("hello"))).unwrap();
	bias.advance(&JSONToken::DoubleQuote).unwrap();
	assert_eq!(bias.next_valid_tokens(), vec![]);
}

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

	test_json_bias(JSONSchema::String { max_length: Some(20) }, model.as_ref());

	test_json_bias(JSONSchema::Boolean, model.as_ref());

	test_json_bias(JSONSchema::Null, model.as_ref());

	test_json_bias(JSONSchema::Object, model.as_ref());

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
