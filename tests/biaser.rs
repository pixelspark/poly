#[cfg(test)]
use std::{path::Path, sync::Arc};

use llm::{
	samplers, InferenceFeedback, InferenceParameters, InferenceSessionConfig, Model, ModelArchitecture, ModelParameters, OutputRequest, Prompt,
	TokenBias, TokenUtf8Buffer,
};

use llmd::bias::{BiaserError, JSONToken};

use llmd::bias::{JSONBiaser, JSONSchema};
use rand::SeedableRng;
use serde_json::Value;

#[test]
pub fn test_parser() {
	let schema = JSONSchema::Boolean;
	let bias = JSONBiaser::new(schema);
	assert_eq!(bias.next_valid_tokens(), vec![JSONToken::True, JSONToken::False]);
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
	test_json_bias(JSONSchema::Boolean, model.as_ref());

	test_json_bias(JSONSchema::Null, model.as_ref());

	test_json_bias(JSONSchema::Object, model.as_ref());

	test_json_bias(JSONSchema::Number, model.as_ref());

	// Array-of-bools
	test_json_bias(
		JSONSchema::Array {
			items: Box::new(JSONSchema::Boolean),
			min_items: Some(2),
		},
		model.as_ref(),
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
	// 	model.as_ref()
	// );
}

fn test_json_bias(schema: JSONSchema, model: &dyn Model) {
	let mut bias = JSONBiaser::new(schema);
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

	let mut rng = rand::rngs::StdRng::seed_from_u64(1337); // Deterministic for tests
	let mut result = String::new();
	let mut result_buffer = TokenUtf8Buffer::new();
	loop {
		let next_valid_tokens = bias.next_valid_tokens();
		if next_valid_tokens.is_empty() {
			break;
		}
		let sampler = samplers::TopPTopK {
			bias_tokens: TokenBias::new(
				next_valid_tokens
					.iter()
					.map(|t| {
						(
							t.token_id(model.eot_token_id(), vocab).unwrap_or_else(|| panic!("token id for {t}")),
							10000.0,
						)
					})
					.collect(),
			),
			..Default::default()
		};
		let inference_params = InferenceParameters {
			sampler: Arc::new(sampler),
			..InferenceParameters::default()
		};

		if let Ok(out) = session.infer_next_token(model, &inference_params, &mut OutputRequest::default(), &mut rng) {
			let out_token = vocab.id(&out).unwrap();
			let out_json_token = JSONToken::from_token(vocab, model.eot_token_id(), out_token).expect("valid token");

			if let JSONToken::Eot = out_json_token {
				println!("End of text");
				break;
			}

			bias.advance(out_json_token).expect("advance");
			if let Some(output) = result_buffer.push(&out) {
				result.push_str(&output);
			}
			println!(
				"== TOKEN: {:?}, RESULT: {result}, next valid tokens: {:?}\n",
				String::from_utf8_lossy(&vocab.decode(vec![out_token], false)),
				bias.next_valid_tokens(),
			);
		} else {
			// End of text
			bias.advance(JSONToken::Eot).unwrap();
			break;
		}
	}
	println!("== FINISH {}\n\n", result);
	serde_json::from_str::<Value>(&result).expect("valid JSON");
}
