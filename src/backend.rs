use std::{collections::HashMap, sync::Arc};

use llm::{
	InferenceFeedback, InferenceParameters, InferenceRequest, InferenceResponse, InferenceSessionConfig, InferenceStats, ModelParameters,
	OutputRequest,
};

use crate::{
	api::{EmbeddingResponse, GenerateError, GenerateRequest},
	config::{Config, DEFAULT_THREADS_PER_SESSION},
};

use tracing::log::*;

pub struct Backend {
	pub config: Config,
	pub models: HashMap<String, Arc<Box<dyn llm::Model>>>,
}

impl Backend {
	pub fn from(config: Config) -> Backend {
		let mut backend = Backend {
			config,
			models: HashMap::new(),
		};

		// Load models
		for (endpoint_name, endpoint) in &backend.config.models {
			let params = ModelParameters {
				prefer_mmap: true,
				context_size: 512,
				lora_adapters: None,
			};

			let model = Arc::new(
				llm::load_dynamic(
					endpoint.architecture,
					&endpoint.model_path,
					llm::VocabularySource::Model,
					params,
					|progress| {
						debug!("Loading endpoint {endpoint_name}: {progress:#?}");
					},
				)
				.expect("load model"),
			);

			backend.models.insert(endpoint_name.clone(), model);
			info!("Loaded model for endpoint {}", endpoint_name);
		}

		// Verify tasks
		for (task_name, task_config) in &backend.config.tasks {
			if !backend.models.contains_key(&task_config.model) {
				panic!("model {} not found for task {}", task_config.model, task_name);
			}
		}

		backend
	}

	pub fn embedding(&self, model_name: &str, request: &GenerateRequest) -> Result<EmbeddingResponse, GenerateError> {
		info!("Embedding request {} {:?}", model_name, request);

		if !self.models.contains_key(model_name) {
			return Err(GenerateError::TaskNotFound(model_name.to_string()));
		};

		let model = self.models.get(model_name).unwrap();
		let inference_config = InferenceSessionConfig::default();
		let mut session = model.start_session(inference_config);
		let mut inference_parameters: InferenceParameters = request.clone().into();
		inference_parameters.n_threads = self.config.models[model_name].threads_per_session.unwrap_or(DEFAULT_THREADS_PER_SESSION);

		let mut output_request = OutputRequest {
			embeddings: Some(Vec::new()),
			all_logits: None,
		};

		let vocab = model.vocabulary();
		let beginning_of_sentence = true;
		let query_token_ids = vocab
			.tokenize(&request.prompt, beginning_of_sentence)
			.unwrap()
			.iter()
			.map(|(_, tok)| *tok)
			.collect::<Vec<_>>();
		model.evaluate(&mut session, &inference_parameters, &query_token_ids, &mut output_request);
		Ok(EmbeddingResponse {
			embedding: output_request.embeddings.unwrap(),
		})
	}

	pub fn complete(
		&self,
		task_name: &str,
		request: &GenerateRequest,
		callback: impl FnMut(InferenceResponse) -> Result<InferenceFeedback, GenerateError>,
	) -> Result<InferenceStats, GenerateError> {
		info!("Completion request {} {:?}", task_name, request);

		if !self.config.tasks.contains_key(task_name) {
			return Err(GenerateError::TaskNotFound(task_name.to_string()));
		};

		let task_config = self.config.tasks.get(task_name).unwrap();
		let model = self.models.get(&task_config.model).unwrap();
		let inference_config = InferenceSessionConfig::default();
		let mut session = model.start_session(inference_config);
		let mut inference_parameters: InferenceParameters = request.clone().into();
		inference_parameters.n_threads = self.config.models[&task_config.model]
			.threads_per_session
			.unwrap_or(DEFAULT_THREADS_PER_SESSION);

		let prompt = format!(
			"{}{}{}",
			&task_config.prefix.as_ref().unwrap_or(&String::from("")),
			request.prompt,
			&task_config.postfix.as_ref().unwrap_or(&String::from(""))
		);

		let stats = session.infer(
			model.as_ref().as_ref(),
			&mut rand::thread_rng(),
			&InferenceRequest {
				prompt: llm::Prompt::Text(&prompt),
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
