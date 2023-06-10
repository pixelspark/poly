use std::{collections::HashMap, sync::Arc};

use llm::{
	samplers, InferenceFeedback, InferenceParameters, InferenceResponse, InferenceSessionConfig, ModelParameters, OutputRequest, Prompt, TokenBias,
	TokenUtf8Buffer,
};

use crate::{
	api::{EmbeddingResponse, GenerateError, PromptRequest, SessionRequest},
	bias::{Biaser, JSONBiaser, NullBiaser},
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
	) -> Result<(), GenerateError> {
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

		// Feed initial prompt
		self.session
			.feed_prompt(
				self.model.as_ref().as_ref(),
				&InferenceParameters::default(),
				Prompt::Tokens(&tokens),
				&mut OutputRequest::default(),
				|_| -> Result<InferenceFeedback, GenerateError> { Ok(InferenceFeedback::Continue) },
			)
			.unwrap();

		// Set up biaser
		let mut biaser: Box<dyn Biaser> = if let Some(ref schema) = self.task_config.schema {
			// TODO: reference to schema, no clone
			Box::new(JSONBiaser::new(schema.clone()))
		} else {
			Box::new(NullBiaser {})
		};

		// Inference loop
		let mut result_buffer = TokenUtf8Buffer::new();
		let vocabulary = self.model.vocabulary();
		let eot_token = self.model.eot_token_id();
		let mut rng = rand::thread_rng();
		let mut inference_params = self.inference_parameters.clone();
		let mut tokens_generated: usize = 0;

		loop {
			let sampler = samplers::TopPTopK {
				bias_tokens: TokenBias::new(biaser.bias(vocabulary, eot_token)),
				temperature: self.task_config.temperature,
				top_k: self.task_config.top_k,
				top_p: self.task_config.top_p,
				repeat_penalty: self.task_config.repeat_penalty,
				repetition_penalty_last_n: self.task_config.repetition_penalty_last_n,
			};

			inference_params.sampler = Arc::new(sampler);

			if let Ok(out) = self
				.session
				.infer_next_token(self.model.as_ref().as_ref(), &inference_params, &mut OutputRequest::default(), &mut rng)
			{
				tokens_generated += 1;
				let out_token = vocabulary.id(&out).unwrap();
				if out_token == eot_token {
					break;
				}

				// Advance biaser
				biaser.advance(vocabulary, out_token);

				// Add token to result
				if let Some(output) = result_buffer.push(&out) {
					if !private_tokens.contains(&output) {
						// Swallow private tokens
						match callback(InferenceResponse::InferredToken(output))? {
							InferenceFeedback::Continue => {}
							InferenceFeedback::Halt => break,
						}
					}
				}

				// Stop once we have enough tokens
				if let Some(max_tokens) = self.max_tokens {
					if tokens_generated >= max_tokens {
						break;
					}
				}
			} else {
				// End of text
				break;
			}
		}
		Ok(())
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
			return Err(GenerateError::ModelNotFound(model_name.to_string()));
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
