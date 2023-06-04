use std::{collections::HashMap, sync::Arc};

use llm::{
	InferenceFeedback, InferenceParameters, InferenceRequest, InferenceResponse, InferenceSessionConfig, InferenceStats, ModelParameters,
	OutputRequest, Prompt,
};

use crate::{
	api::{EmbeddingResponse, GenerateError, PromptRequest, SessionRequest},
	config::{Config, TaskConfig, DEFAULT_THREADS_PER_SESSION},
};

use tracing::log::*;

pub struct Backend {
	pub config: Config,
	pub models: HashMap<String, Arc<Box<dyn llm::Model>>>,
}

pub struct BackendSession {
	model: Arc<Box<dyn llm::Model>>,
	session: llm::InferenceSession,
	inference_parameters: InferenceParameters,
	max_tokens: Option<usize>,
	task_config: TaskConfig,
}

impl BackendSession {
	pub fn complete(
		&mut self,
		request: &PromptRequest,
		mut callback: impl FnMut(InferenceResponse) -> Result<InferenceFeedback, GenerateError>,
	) -> Result<InferenceStats, GenerateError> {
		// Generate tokens (prefix + prompt + postfix)
		tracing::debug!("beginning-of-text token is {:?}", self.model.bot_token_id());
		let beginning_of_sentence = self.model.bot_token_id().is_some() && self.session.n_past == 0;
		let mut tokens = vec![];

		// Append prefix tokens
		if let Some(ref prefix) = self.task_config.prefix {
			tokens.append(&mut Prompt::Text(prefix).to_tokens(self.model.vocabulary(), beginning_of_sentence)?);
		}

		// Generate user prompt tokens
		let mut user_tokens = Prompt::Text(&request.prompt).to_tokens(self.model.vocabulary(), beginning_of_sentence && tokens.is_empty())?;

		// Check for private tokens in user prompt
		let private_tokens = self.task_config.private_tokens.clone().unwrap_or(vec![]);
		if !private_tokens.is_empty() {
			let private_token_ids: Vec<u32> = private_tokens
				.iter()
				.map(|token_str| {
					let toks = self.model.vocabulary().tokenize(token_str, false).unwrap();
					if toks.len() != 1 {
						panic!("invalid forbidden token configured: {token_str}");
					}
					toks[0].1
				})
				.collect();
			if user_tokens.iter().any(|t| private_token_ids.contains(t)) {
				return Err(GenerateError::IllegalToken);
			}
		}
		tokens.append(&mut user_tokens);

		// Append postfix tokens
		if let Some(ref postfix) = self.task_config.postfix {
			tokens.append(&mut Prompt::Text(postfix).to_tokens(self.model.vocabulary(), beginning_of_sentence && tokens.is_empty())?);
		}

		tracing::trace!("tokens: {tokens:?}");

		let stats = self.session.infer(
			self.model.as_ref().as_ref(),
			&mut rand::thread_rng(),
			&InferenceRequest {
				prompt: llm::Prompt::Tokens(&tokens),
				parameters: &self.inference_parameters,
				play_back_previous_tokens: false,
				maximum_token_count: self.max_tokens,
			},
			&mut Default::default(),
			|r| -> Result<InferenceFeedback, GenerateError> {
				match r {
					// Swallow private tokens in output
					InferenceResponse::InferredToken(t) if private_tokens.contains(&t) => Ok(InferenceFeedback::Continue),
					_ => callback(r),
				}
			},
		)?;
		info!(
			"Completion request completed: {} tok/s, {:?}",
			(stats.predict_tokens as f64) / stats.predict_duration.as_secs_f64(),
			stats
		);
		Ok(stats)
	}
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
				context_size: endpoint.context_size.unwrap_or(512),
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

	pub fn embedding(&self, model_name: &str, request: &SessionRequest, prompt: &PromptRequest) -> Result<EmbeddingResponse, GenerateError> {
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
			.tokenize(&prompt.prompt, beginning_of_sentence)
			.unwrap()
			.iter()
			.map(|(_, tok)| *tok)
			.collect::<Vec<_>>();
		model.evaluate(&mut session, &inference_parameters, &query_token_ids, &mut output_request);
		Ok(EmbeddingResponse {
			embedding: output_request.embeddings.unwrap(),
		})
	}

	pub fn start(&self, task_name: &str, request: &SessionRequest) -> Result<BackendSession, GenerateError> {
		info!("Start session {task_name}");

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

		if let Some(ref prelude_prompt) = task_config.prelude {
			tracing::debug!("feeding prelude prompt: '{prelude_prompt}'");
			session.feed_prompt(
				model.as_ref().as_ref(),
				&inference_parameters,
				Prompt::Text(&prelude_prompt.clone()),
				&mut OutputRequest::default(),
				|r| -> Result<InferenceFeedback, GenerateError> {
					tracing::trace!("Feed prompt: received {r:?}");
					Ok(InferenceFeedback::Continue)
				},
			)?;
		}

		Ok(BackendSession {
			model: model.clone(),
			session,
			inference_parameters,
			max_tokens: Some(request.max_tokens),
			task_config: task_config.clone(),
		})
	}
}
