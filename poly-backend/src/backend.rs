use std::{
	collections::HashMap,
	sync::{Arc, Mutex},
};

use llm::{
	samplers::TopPTopK, InferenceParameters, InferenceSessionConfig, InferenceStats, Model, ModelParameters, OutputRequest, Prompt, TokenId,
	TokenizerSource,
};
pub use llm::{InferenceFeedback, InferenceResponse};
use tokio::task::spawn_blocking;

use crate::{
	config::{BackendConfig, MemoryConfig, MemoryStoreConfig, ModelConfig},
	memory::hora::HoraMemory,
	memory::{Memory, MemoryError},
	session::BackendSession,
	stats::TaskStats,
	types::{EmbeddingResponse, GenerateError, PromptRequest, SessionRequest},
};

use tracing::*;

pub struct BackendStats {
	pub task_stats: Mutex<HashMap<String, TaskStats>>,
}

pub struct Backend {
	pub config: BackendConfig,
	pub models: HashMap<String, Arc<Box<dyn llm::Model>>>,
	pub memories: HashMap<String, Arc<Box<dyn Memory>>>,
	pub stats: Arc<BackendStats>,
}

impl Backend {
	pub fn from(config: BackendConfig, mut load_progress: impl FnMut(f64)) -> Backend {
		let mut backend = Backend {
			config,
			models: HashMap::new(),
			stats: Arc::new(BackendStats::default()),
			memories: HashMap::new(),
		};

		// Load models
		let n_models = backend.config.models.len();
		for (index, (model_name, model_config)) in backend.config.models.iter().enumerate() {
			let params = ModelParameters {
				prefer_mmap: true,
				context_size: model_config.context_size,
				lora_adapters: None,
				use_gpu: model_config.use_gpu,
			};

			let model = Arc::new(
				llm::load_dynamic(
					Some(model_config.architecture),
					&model_config.model_path,
					TokenizerSource::Embedded,
					params,
					|progress| {
						let fp = match progress {
							llm::LoadProgress::HyperparametersLoaded => 0.0,
							llm::LoadProgress::ContextSize { .. } => 0.0,
							llm::LoadProgress::LoraApplied { .. } => 0.0,
							llm::LoadProgress::TensorLoaded {
								current_tensor,
								tensor_count,
							} => (current_tensor as f64) / (tensor_count as f64),
							llm::LoadProgress::Loaded { .. } => 1.0,
						};
						load_progress((index as f64 + fp) / n_models as f64);
						trace!("Loading model {model_name}: {progress:#?}");
					},
				)
				.expect("load model"),
			);

			backend.models.insert(model_name.clone(), model);
			info!("Loaded model {} use_gpu={:?}", model_name, model_config.use_gpu);
		}

		info!("All models loaded");

		// Load memories
		for (memory_name, memory_config) in backend.config.memories.iter() {
			info!("Loading memory {memory_name}");
			if !backend.models.contains_key(&memory_config.embedding_model) {
				panic!("embedding model {} not found for memory {}", memory_config.embedding_model, memory_name);
			}
			let mem = memory_config.store.from(memory_config).expect("memory construction");
			backend.memories.insert(memory_name.clone(), Arc::new(mem));
		}

		// Verify tasks
		for (task_name, task_config) in &backend.config.tasks {
			if !backend.models.contains_key(&task_config.model) {
				panic!("model {} not found for task {}", task_config.model, task_name);
			}

			if let Some(memorization) = &task_config.memorization {
				if !backend.memories.contains_key(&memorization.memory) {
					panic!("memory {} not found for task {}", memorization.memory, task_name);
				}
			}
		}

		backend
	}

	pub fn embedding(&self, model_name: &str, prompt: &PromptRequest) -> Result<EmbeddingResponse, GenerateError> {
		info!("Embedding request {} ", model_name);

		if !self.models.contains_key(model_name) {
			return Err(GenerateError::ModelNotFound(model_name.to_string()));
		};

		let model = self.models.get(model_name).unwrap();
		let inference_config = InferenceSessionConfig {
			use_gpu: self.config.models[model_name].use_gpu,
			..InferenceSessionConfig::default()
		};
		let mut session = model.start_session(inference_config);
		let inference_parameters: InferenceParameters = InferenceParameters {
			n_threads: self.config.models[model_name].threads_per_session,
			n_batch: 8,
			sampler: Arc::new(TopPTopK::default()),
		};

		let mut output_request = OutputRequest {
			embeddings: Some(Vec::new()),
			all_logits: None,
		};

		let vocab = model.tokenizer();
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

	pub async fn recall(&self, memory_name: &str, prompt: &str, top_n: usize) -> Result<Vec<String>, GenerateError> {
		if !self.memories.contains_key(memory_name) {
			return Err(GenerateError::MemoryNotFound(memory_name.to_string()));
		}

		let memory_config = &self.config.memories[memory_name];

		// Generate embedding for prompt
		let embedding = self.embedding(&memory_config.embedding_model, &PromptRequest { prompt: prompt.to_string() })?;
		let memory = self.memories.get(memory_name).unwrap();
		memory.get(&embedding.embedding, top_n).await.map_err(GenerateError::Memory)
	}

	pub async fn memorize(&self, memory_name: &str, data: &str) -> Result<(), GenerateError> {
		// Obtain memorization configuration
		let memory_config = &self.config.memories[memory_name];
		let memory = self.memories[memory_name].clone();
		let model_name = &memory_config.embedding_model;

		// Get embedding model
		if !self.models.contains_key(model_name) {
			return Err(GenerateError::ModelNotFound(model_name.to_string()));
		};

		let model = self.models.get(model_name).unwrap().clone();
		let model_config = self.config.models[model_name].clone();

		// Split the input by all separator
		let vocab = model.tokenizer();
		let mut splits = vec![data];
		for splitter in &memory_config.chunk_separators {
			splits = splits.into_iter().flat_map(|s| s.split_inclusive(splitter)).collect();
		}

		let mut current_chunk = vec![];
		let mut current_chunk_text = String::new();
		let mut first = true;
		for split in splits {
			let split_tokens: Vec<(Vec<u8>, TokenId)> = vocab.tokenize(split, first).map_err(GenerateError::TokenizationError)?;

			// If this split exceeds the limits by itself, just chop it up until it fits
			for tokens in split_tokens.chunks(memory_config.chunk_max_tokens) {
				// Fill 'current_chunk' up to the max amount
				if tokens.len() + current_chunk.len() > memory_config.chunk_max_tokens {
					// This chunk is finished, commit to memory
					let model = model.clone();
					let memory = memory.clone();

					let current_chunk_to_store: Vec<u32> = std::mem::take(&mut current_chunk);
					let current_chunk_text_to_store: String = std::mem::take(&mut current_chunk_text);
					Self::memorize_chunk(model, &model_config, &current_chunk_text_to_store, current_chunk_to_store, memory).await?;

					current_chunk_text.clear();
				}

				current_chunk.extend(tokens.iter().map(|x| x.1));
				let tokens_bytes: Vec<u8> = tokens.iter().flat_map(|t| t.0.clone()).collect();
				let tokens_text = String::from_utf8_lossy(&tokens_bytes);
				current_chunk_text.push_str(&tokens_text);
				first = false;
			}
		}

		if !current_chunk.is_empty() {
			Self::memorize_chunk(model, &model_config, &current_chunk_text, current_chunk, memory).await?;
		}

		Ok(())
	}

	async fn memorize_chunk(
		model: Arc<Box<dyn Model>>,
		model_config: &ModelConfig,
		text: &str,
		tokens: Vec<TokenId>,
		memory: Arc<Box<dyn Memory>>,
	) -> Result<(), MemoryError> {
		// Calculate embedding
		tracing::debug!(n_tokens = tokens.len(), ?text, "memorize chunk");

		let inference_config = InferenceSessionConfig {
			use_gpu: model_config.use_gpu,
			..InferenceSessionConfig::default()
		};

		let mut session = model.start_session(inference_config);
		let inference_parameters: InferenceParameters = InferenceParameters {
			n_threads: model_config.threads_per_session,
			n_batch: 8,
			sampler: Arc::new(TopPTopK::default()),
		};

		let embeddings = spawn_blocking(move || {
			let mut output_request = OutputRequest {
				embeddings: Some(Vec::new()),
				all_logits: None,
			};
			model.evaluate(&mut session, &inference_parameters, &tokens, &mut output_request);
			output_request.embeddings.unwrap()
		})
		.await
		.unwrap();

		memory.store(text, &embeddings).await?;
		Ok(())
	}

	pub fn start(&self, task_name: &str, _request: &SessionRequest, backend: Arc<Backend>) -> Result<BackendSession, GenerateError> {
		info!("Start session {task_name}");

		if !self.config.tasks.contains_key(task_name) {
			return Err(GenerateError::TaskNotFound(task_name.to_string()));
		};

		let task_config = self.config.tasks.get(task_name).unwrap();

		let memory = task_config.memorization.as_ref().map(|mc| self.memories.get(&mc.memory).unwrap());

		let model = self.models.get(&task_config.model).unwrap();
		let inference_config = InferenceSessionConfig {
			use_gpu: self.config.models[&task_config.model].use_gpu,
			..InferenceSessionConfig::default()
		};
		let mut session = model.start_session(inference_config);

		let mut inference_parameters: InferenceParameters = task_config.clone().into();
		inference_parameters.n_threads = self.config.models[&task_config.model].threads_per_session;

		if let Some(ref prelude_prompt) = task_config.prelude {
			if !prelude_prompt.is_empty() {
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
		}

		Ok(BackendSession {
			model: model.clone(),
			memory: memory.cloned(),
			session,
			inference_parameters,
			task_config: task_config.clone(),
			stats: self.stats.clone(),
			task_name: task_name.to_string(),
			backend,
		})
	}
}

impl BackendStats {
	pub fn add(&self, task_name: &str, stats: &InferenceStats, n_threads: usize) {
		let mut ts = self.task_stats.lock().unwrap();
		if let Some(task_stats) = ts.get_mut(task_name) {
			task_stats.add_cycle(stats, n_threads);
		} else {
			let mut task_stats = TaskStats::default();
			task_stats.add_cycle(stats, n_threads);
			ts.insert(task_name.to_string(), task_stats);
		}
	}
}

impl Default for BackendStats {
	fn default() -> Self {
		BackendStats {
			task_stats: Mutex::new(HashMap::new()),
		}
	}
}

impl MemoryStoreConfig {
	pub fn from(&self, memory_config: &MemoryConfig) -> Result<Box<dyn Memory>, MemoryError> {
		match self {
			Self::Hora { path } => Ok(Box::new(HoraMemory::new(path, memory_config.dimensions)?)),
		}
	}
}
