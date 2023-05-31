use std::collections::HashMap;

use llm::{InferenceFeedback, InferenceParameters, InferenceRequest, InferenceResponse, InferenceSessionConfig, InferenceStats, ModelParameters};

use crate::{
	api::{GenerateError, GenerateRequest},
	config::{Config, DEFAULT_THREADS_PER_SESSION},
};

use tracing::log::*;

pub struct Backend {
	pub config: Config,
	pub models: HashMap<String, Box<dyn llm::Model>>,
}

impl Backend {
	pub fn from(config: Config) -> Backend {
		let mut backend = Backend {
			config,
			models: HashMap::new(),
		};

		// Load models
		for (endpoint_name, endpoint) in &backend.config.endpoints {
			let params = ModelParameters {
				prefer_mmap: true,
				context_size: 512,
				lora_adapters: None,
			};

			let model = llm::load_dynamic(
				endpoint.architecture,
				&endpoint.model_path,
				llm::VocabularySource::Model,
				params,
				|progress| {
					debug!("Loading endpoint {endpoint_name}: {progress:#?}");
				},
			)
			.expect("load model");

			backend.models.insert(endpoint_name.clone(), model);
			info!("Loaded model for endpoint {}", endpoint_name);
		}

		backend
	}

	pub fn complete(
		&self,
		endpoint_name: &str,
		request: &GenerateRequest,
		callback: impl FnMut(InferenceResponse) -> Result<InferenceFeedback, GenerateError>,
	) -> Result<InferenceStats, GenerateError> {
		info!("Completion request {} {:?}", endpoint_name, request);

		if !self.models.contains_key(endpoint_name) {
			return Err(GenerateError::EndpointNotFound(endpoint_name.to_string()));
		};

		let model = self.models.get(endpoint_name).unwrap();
		let inference_config = InferenceSessionConfig::default();
		let mut session = model.start_session(inference_config);
		let mut inference_parameters: InferenceParameters = request.clone().into();
		inference_parameters.n_threads = self.config.endpoints[endpoint_name]
			.threads_per_session
			.unwrap_or(DEFAULT_THREADS_PER_SESSION);

		let stats = session.infer(
			model.as_ref(),
			&mut rand::thread_rng(),
			&InferenceRequest {
				prompt: llm::Prompt::Text(&request.prompt),
				parameters: &inference_parameters,
				play_back_previous_tokens: false,
				maximum_token_count: Some(request.max_tokens),
			},
			&mut Default::default(),
			callback,
		)?;
		info!(
			"Completion request completed: {} tok/s, {:?}",
			(stats.predict_tokens as f64) / stats.predict_duration.as_secs_f64(),
			stats
		);
		Ok(stats)
	}
}
