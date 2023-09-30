use std::{
	borrow::Cow,
	collections::{HashMap, HashSet},
	path::PathBuf,
	sync::{Arc, Mutex, RwLock},
};

use directories::ProjectDirs;
use futures_util::StreamExt;
pub use llm::{InferenceFeedback, InferenceResponse};
use llm::{
	InferenceParameters, InferenceSession, InferenceSessionConfig, InferenceSnapshot, InferenceStats, Model, ModelParameters, OutputRequest, Prompt,
	TokenId, TokenizerSource,
};
use regex::Regex;
use tokio::{fs::File, io::AsyncWriteExt, sync::mpsc::Sender, task::spawn_blocking};

use crate::{
	config::{BackendConfig, ModelConfig},
	memory::{hierarchically_chunk, Memory, MemoryError},
	session::BackendSession,
	stats::TaskStats,
	types::{BackendError, EmbeddingResponse, PromptRequest, SessionRequest, TokenResponse, TokenizationResponse},
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
	pub prelude_snapshots: RwLock<HashMap<String, InferenceSnapshot>>,
}

const CACHE_MODELS_DIR: &str = "models";

impl Backend {
	pub async fn from(mut config: BackendConfig, progress: Option<Sender<f64>>) -> Backend {
		// Determine cache path
		if config.cache_path.is_none() {
			if let Some(pd) = ProjectDirs::from("nl.dialogic", "Dialogic", "Poly") {
				config.cache_path = Some(pd.cache_dir().to_path_buf());
			}
		}

		// Ensure cache directory exists (if there is one)
		let cache_path = config.cache_path.clone();
		if let Some(ref cache_path) = cache_path {
			tokio::fs::create_dir_all(cache_path.join(CACHE_MODELS_DIR)).await.unwrap();
		}

		tracing::info!(
			metal = cfg!(feature = "metal"),
			cache_path = cache_path.as_ref().map(|x| x.to_str().map(|y| y.to_string())),
			"backend instantiating"
		);
		let mut backend = Backend {
			config,
			models: HashMap::new(),
			stats: Arc::new(BackendStats::default()),
			memories: HashMap::new(),
			prelude_snapshots: RwLock::new(HashMap::new()),
		};

		// Load models
		let n_models = backend.config.models.len();
		for (index, (model_name, model_config)) in backend.config.models.iter().enumerate() {
			// Warn about invalid configurations
			if !model_config.use_gpu && model_config.gpu_layers.is_some() {
				tracing::warn!("gpu_layers set but ignored because use_gpu is not set to true");
			}
			if cfg!(feature = "metal") && model_config.use_gpu && model_config.gpu_layers.is_some() {
				tracing::warn!("gpu_layers set but ignored because with the Metal backend, all layers are run on the GPU");
			}

			// Check if we already have a copy of the model, or download it
			let actual_model_path = model_config.model_path.clone().unwrap_or_else(|| {
				cache_path
					.clone()
					.expect("cache path is set when models without path are specified")
					.join(CACHE_MODELS_DIR)
					.join(format!("{model_name}.bin"))
			});

			if !actual_model_path.exists() {
				// See if we can download this file
				if let Some(ref url) = model_config.url {
					// Download
					tracing::info!("downloading model {model_name} from {url}");
					Self::download_model(url, &actual_model_path).await.expect("could not download model");
					if !actual_model_path.exists() {
						panic!("model file not found for model {model_name} at path {actual_model_path:?} even after downloading");
					}
				} else {
					panic!("model file not found for model {model_name} at path {actual_model_path:?}");
				}
			}

			// Set up hyperparameters
			let params = ModelParameters {
				prefer_mmap: true,
				context_size: model_config.context_size,
				lora_adapters: model_config.lora_adapters.clone(),
				use_gpu: model_config.use_gpu,
				gpu_layers: model_config.gpu_layers,
				rope_overrides: None,
				n_gqa: None,
			};

			// Actually load the model
			let model_config = model_config.clone();
			let model_name_copy = model_name.clone();

			let progress_sender = progress.clone();
			let model = spawn_blocking(move || {
				Arc::new(
					llm::load_dynamic(
						Some(model_config.architecture),
						&actual_model_path,
						TokenizerSource::Embedded,
						params,
						|load_progress| {
							let fp: f64 = match load_progress {
								llm::LoadProgress::HyperparametersLoaded => 0.0,
								llm::LoadProgress::ContextSize { .. } => 0.0,
								llm::LoadProgress::LoraApplied { .. } => 0.0,
								llm::LoadProgress::TensorLoaded {
									current_tensor,
									tensor_count,
								} => (current_tensor as f64) / (tensor_count as f64),
								llm::LoadProgress::Loaded { .. } => 1.0,
							};
							if let Some(ref p) = progress_sender {
								_ = p.blocking_send((index as f64 + fp) / n_models as f64);
							}
							trace!("Loading model {model_name_copy}: {load_progress:#?}");
						},
					)
					.expect("load model"),
				)
			})
			.await
			.unwrap();

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

		info!("All memories loaded");

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

		info!("All tasks loaded");

		if let Some(ref p) = progress {
			_ = p.send(1.0).await;
		}

		backend
	}

	/// Downloads a file to the indicated location
	async fn download_model(url: &str, target_path: &PathBuf) -> Result<(), String> {
		let client = reqwest::Client::new();
		let res = client.get(url).send().await.map_err(|x| x.to_string())?;

		let mut temp_path = target_path.clone();
		temp_path.set_extension("download");
		let mut file = File::create(&temp_path)
			.await
			.map_err(|x| format!("could not create temp file at {temp_path:?}: {x}"))?;
		let total_size = res.content_length().ok_or(format!("Failed to get content length from '{}'", &url))? as usize;

		let mut stream = res.bytes_stream();
		let mut downloaded: usize = 0;
		while let Some(item) = stream.next().await {
			let chunk = item.or(Err("Error while downloading file".to_string()))?;
			file.write_all(&chunk).await.or(Err("Error while writing to file".to_string()))?;
			downloaded += chunk.len();
			tracing::debug!(url, "download: {}/{} bytes", downloaded, total_size);
		}
		if downloaded != total_size {
			tracing::debug!(
				url,
				"download completed, but size mismatches: {downloaded} downloaded bytes, {total_size} total size bytes",
			);
		}
		tracing::debug!(url, "download completed");

		// Move the temp file to the right location
		tokio::fs::rename(temp_path, target_path).await.map_err(|x| x.to_string())?;
		Ok(())
	}

	pub fn embedding(&self, model_name: &str, prompt: &PromptRequest) -> Result<EmbeddingResponse, BackendError> {
		info!(model_name, "embedding request");

		if !self.models.contains_key(model_name) {
			return Err(BackendError::ModelNotFound(model_name.to_string()));
		};

		let model = self.models.get(model_name).unwrap();
		let inference_config = InferenceSessionConfig {
			n_threads: self.config.models[model_name].threads_per_session,
			n_batch: 8,
			..InferenceSessionConfig::default()
		};
		let mut session = model.start_session(inference_config);
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
		model.evaluate(&mut session, &query_token_ids, &mut output_request);
		Ok(EmbeddingResponse {
			embedding: output_request.embeddings.unwrap(),
		})
	}

	pub fn tokenize(&self, model_name: &str, prompt: &PromptRequest) -> Result<TokenizationResponse, BackendError> {
		info!(model_name, "tokenization request");

		if !self.models.contains_key(model_name) {
			return Err(BackendError::ModelNotFound(model_name.to_string()));
		};

		let model = self.models.get(model_name).unwrap();
		let res = model.tokenizer().tokenize(&prompt.prompt, true)?;
		Ok(TokenizationResponse {
			tokens: res
				.iter()
				.map(|t| TokenResponse {
					text: String::from_utf8_lossy(&t.0).to_string(),
					token: t.1,
				})
				.collect(),
		})
	}

	pub async fn forget(&self, memory_name: &str) -> Result<(), BackendError> {
		if !self.memories.contains_key(memory_name) {
			return Err(BackendError::MemoryNotFound(memory_name.to_string()));
		}
		let memory = self.memories.get(memory_name).unwrap();
		tracing::info!("clearing memory {memory_name}");
		memory.clear().await.map_err(BackendError::Memory)
	}

	pub async fn recall(&self, memory_name: &str, prompt: &str, top_n: usize) -> Result<Vec<String>, BackendError> {
		if !self.memories.contains_key(memory_name) {
			return Err(BackendError::MemoryNotFound(memory_name.to_string()));
		}

		let memory_config = &self.config.memories[memory_name];

		// Generate embedding for prompt
		let embedding = self.embedding(&memory_config.embedding_model, &PromptRequest { prompt: prompt.to_string() })?;
		let memory = self.memories.get(memory_name).unwrap();
		memory.get(&embedding.embedding, top_n).await.map_err(BackendError::Memory)
	}

	pub async fn memorize(&self, memory_name: &str, data: &str) -> Result<(), BackendError> {
		// Obtain memorization configuration
		tracing::info!(memory_name, data_length = data.len(), "memorize");
		let memory_config = &self.config.memories[memory_name];
		let memory = self.memories[memory_name].clone();
		let model_name = &memory_config.embedding_model;

		// Get embedding model
		if !self.models.contains_key(model_name) {
			return Err(BackendError::ModelNotFound(model_name.to_string()));
		};

		let model = self.models.get(model_name).unwrap().clone();
		let model_config = self.config.models[model_name].clone();

		// Apply pre-filter
		let mut data = Cow::from(data);
		if !memory_config.pre_filter.is_empty() {
			for filter_string in memory_config.pre_filter.iter() {
				let regex = Regex::new(filter_string).unwrap();
				let out = regex.replace_all(&data, " ").to_string();
				data = Cow::Owned(out);
			}

			// Replace double spaces with single spaces
			data = Cow::Owned(data.replace("  ", " "));
		}

		// Split the input by all separators
		let vocab = model.tokenizer();
		let separator_tokens: Vec<TokenId> = memory_config
			.chunk_separators
			.iter()
			.map(|s| {
				let tokens = vocab.tokenize(s, false)?;
				if tokens.len() != 1 {
					return Err(BackendError::InvalidChunkSeparator(s.clone()));
				}
				Ok(tokens[0].1)
			})
			.collect::<Result<Vec<TokenId>, BackendError>>()?;

		let body_tokens = vocab.tokenize(data.as_ref(), false)?;
		let chunks = hierarchically_chunk(body_tokens, &separator_tokens, memory_config.chunk_max_tokens);

		let post_filter_tokens = memory_config
			.post_filter
			.iter()
			.map(|s| {
				let tokens = vocab.tokenize(s, false)?;
				if tokens.len() != 1 {
					return Err(BackendError::InvalidChunkSeparator(s.clone()));
				}
				Ok(tokens[0].1)
			})
			.collect::<Result<HashSet<TokenId>, BackendError>>()?;

		for mut chunk in chunks {
			assert!(
				chunk.len() <= memory_config.chunk_max_tokens,
				"chunk size ({}) must not exceed maximum ({})",
				chunk.len(),
				memory_config.chunk_max_tokens
			);
			// Apply post filter
			chunk.retain(|t| !post_filter_tokens.contains(&t.1));

			if !chunk.is_empty() {
				let chunk_tokens: Vec<TokenId> = chunk.iter().map(|x| x.1).collect();
				let chars: Vec<u8> = chunk.iter().flat_map(|x| x.0.clone()).collect();
				let chunk_text = String::from_utf8_lossy(&chars);
				tracing::trace!(?chunk_text, chunk_size_tokens = chunk_tokens.len(), "chunk for ingest");
				Self::memorize_chunk(model.clone(), &model_config, &chunk_text, chunk_tokens, memory.clone()).await?;
			}
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
		tracing::trace!(n_tokens = tokens.len(), ?text, "memorize chunk");

		let inference_config = InferenceSessionConfig {
			n_threads: model_config.threads_per_session,
			n_batch: model_config.batch_size,
			..InferenceSessionConfig::default()
		};

		let mut session = model.start_session(inference_config);

		let embeddings = spawn_blocking(move || {
			let mut output_request = OutputRequest {
				embeddings: Some(Vec::new()),
				all_logits: None,
			};
			model.evaluate(&mut session, &tokens, &mut output_request);
			output_request.embeddings.unwrap()
		})
		.await
		.unwrap();

		memory.store(text, &embeddings).await?;
		Ok(())
	}

	pub fn start(&self, task_name: &str, _request: &SessionRequest, backend: Arc<Backend>) -> Result<BackendSession, BackendError> {
		info!("Start session {task_name}");

		if !self.config.tasks.contains_key(task_name) {
			return Err(BackendError::TaskNotFound(task_name.to_string()));
		};

		let task_config = self.config.tasks.get(task_name).unwrap();

		let memory = task_config.memorization.as_ref().map(|mc| self.memories.get(&mc.memory).unwrap());

		let model = self.models.get(&task_config.model).unwrap().clone();
		let n_threads = self.config.models[&task_config.model].threads_per_session;
		let inference_config: InferenceSessionConfig = InferenceSessionConfig {
			n_threads,
			n_batch: self.config.models[&task_config.model].batch_size,
			..InferenceSessionConfig::default()
		};

		let inference_parameters: InferenceParameters = task_config.clone().into();

		let session = if let Some(ref prelude_prompt) = task_config.prelude {
			if !prelude_prompt.is_empty() {
				// Do we have a snapshot?
				let cache = self.prelude_snapshots.read().unwrap();
				if let Some(snapshot) = cache.get(task_name) {
					// We have a snapshot
					tracing::debug!("Re-using prelude snapshot for task {task_name}");
					InferenceSession::from_snapshot(snapshot.clone(), model.as_ref().as_ref()).expect("restore prelude")
				} else {
					// We are dropping the read lock here because further on we want to acquire a write lock, and RwLock
					// has no way to upgrade the read lock to a write lock. This is fine for now - it might cause us to
					// generate the prelude twice but that's okay.
					drop(cache);
					let mut session = model.start_session(inference_config);

					tracing::debug!("feeding prelude prompt: '{prelude_prompt}'");
					session.feed_prompt(
						model.as_ref().as_ref(),
						Prompt::Text(&prelude_prompt.clone()),
						&mut OutputRequest::default(),
						|r| -> Result<InferenceFeedback, BackendError> {
							tracing::trace!("Feed prompt: received {r:?}");
							Ok(InferenceFeedback::Continue)
						},
					)?;

					// Save snapshot
					tracing::trace!("Caching prelude snapshot for task {task_name}");
					let snapshot = unsafe { session.get_snapshot().to_owned() };
					{
						let mut cache = self.prelude_snapshots.write().unwrap();
						cache.insert(task_name.to_string(), snapshot);
					}
					session
				}
			} else {
				// Just a plain session
				model.start_session(inference_config)
			}
		} else {
			// Just a plain session
			model.start_session(inference_config)
		};

		Ok(BackendSession {
			model: model.clone(),
			memory: memory.cloned(),
			session,
			inference_parameters,
			task_config: task_config.clone(),
			stats: self.stats.clone(),
			task_name: task_name.to_string(),
			n_threads,
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
