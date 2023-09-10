use llm::samplers::{
	llm_samplers::{
		configure::{SamplerChainBuilder, SamplerSlot},
		samplers::{SampleRandDistrib, SampleRepetition, SampleTemperature, SampleTopK, SampleTopP},
		types::SamplerChain,
	},
	ConfiguredSamplers,
};
pub use llm::ModelArchitecture;
use poly_bias::json::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};
use std::{collections::HashMap, path::PathBuf, str::FromStr};

use crate::memory::MemoryStoreConfig;

fn architecture_from_str<'de, D>(deserializer: D) -> Result<ModelArchitecture, D::Error>
where
	D: Deserializer<'de>,
{
	let s: String = Deserialize::deserialize(deserializer)?;
	match s.as_str() {
		"gptneox" => Ok(ModelArchitecture::GptNeoX),
		"mpt" => Ok(ModelArchitecture::Mpt),
		"llama" => Ok(ModelArchitecture::Llama),
		"gpt2" => Ok(ModelArchitecture::Gpt2),
		"gptj" => Ok(ModelArchitecture::GptJ),
		"bloom" => Ok(ModelArchitecture::Bloom),
		_ => Err(serde::de::Error::custom("invalid model architecture name")),
	}
}

#[derive(Deserialize, Debug, Clone, Serialize)]
pub struct MemoryConfig {
	/// The type of memory to be constructed
	pub store: MemoryStoreConfig,

	/// Number of dimensions for embedding vectors
	pub dimensions: usize,

	/// Model to use for embedding
	pub embedding_model: String,

	/// Separators to use while chunking
	#[serde(default = "default_chunk_separators")]
	pub chunk_separators: Vec<String>,

	/// Maximum length for a chunk (in tokens)
	#[serde(default = "default_chunk_max_tokens")]
	pub chunk_max_tokens: usize,

	/// Remove the following patterns (regular expressions) before chunking, replacing them with a single space (after
	/// which double spaces are eliminated)
	#[serde(default = "default_pre_filter")]
	pub pre_filter: Vec<String>,

	/// Remove the following tokens after chunking (strings must refer to single tokens)
	#[serde(default = "default_post_filter")]
	pub post_filter: Vec<String>,
}

fn default_pre_filter() -> Vec<String> {
	vec![
		"[\\r\\t]".to_string(),          // Any carriage return or tab
		"[\\ \\n._|-]{2,}".to_string(),  // Specific characters occuring more than once
		"^[\\r\\n\\ ._|-]+".to_string(), // Specific characters at the beginning of a chunk
	]
}

fn default_post_filter() -> Vec<String> {
	vec!["\n".to_string()]
}

#[derive(Deserialize, Debug, Clone)]
pub struct ModelConfig {
	/// The model architecture type
	#[serde(deserialize_with = "architecture_from_str")]
	pub architecture: ModelArchitecture,

	/// Path to the model file
	pub model_path: Option<PathBuf>,

	/// URL where to download the model from. This is used in case no file is found at `model_path` or when `model_path`
	///  is not specified (in which case a cache location will be used)
	pub url: Option<String>,

	/// The [LoRA](https://arxiv.org/abs/2106.09685) adapters to use when loading the model. Note that these cannot currently
	/// be downloaded automatically on-demand.
	pub lora_adapters: Option<Vec<PathBuf>>,

	/// Threads per session
	#[serde(default = "default_threads_per_session")]
	pub threads_per_session: usize,

	/// Context size
	#[serde(default = "default_context_size")]
	pub context_size: usize,

	/// Whether to use GPU acceleration, if available
	#[serde(default = "default_use_gpu")]
	pub use_gpu: bool,

	/// Number of layers to offload to the GPU (ignored when `use_gpu` is false; when this is `None`, all layers wil
	///  be offloaded. For Metal, all layers will always be offloaded regardless of this setting)
	pub gpu_layers: Option<usize>,

	/// Controls batch/chunk size for prompt ingestion in [InferenceSession::feed_prompt].
	///
	/// This is the number of tokens that will be ingested at once. This is useful for
	/// trying to speed up the ingestion of prompts, as it allows for parallelization.
	/// However, you will be fundamentally limited by your machine's ability to evaluate
	/// the transformer model, so increasing the batch size will not always help.
	///
	/// A reasonable default value is 8.
	#[serde(default = "default_batch_size")]
	pub batch_size: usize,
}

const fn default_use_gpu() -> bool {
	false
}

const fn default_threads_per_session() -> usize {
	8
}

const fn default_context_size() -> usize {
	512
}

const fn default_chunk_max_tokens() -> usize {
	255
}

fn default_chunk_separators() -> Vec<String> {
	vec![String::from(" ")]
}

#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
pub enum BiaserConfig {
	/// Configure Biaser from JSON schema included directly in the configuration
	JsonSchema(JsonSchema),

	/// Configure Biaser using an external file containing a JSON schema (in JSON)
	JsonSchemaFile(PathBuf),
}

#[derive(Deserialize, Debug, Clone)]
pub struct TaskMemorizationConfig {
	/// The memory to use
	pub memory: String,

	/// Whether to store prompts
	pub store_prompts: bool,

	/// How many items from the memory to retrieve
	pub retrieve: Option<usize>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct TaskConfig {
	pub model: String,

	/// Text to start each conversation with
	pub prelude: Option<String>,

	/// Text to prefix each user input with
	pub prefix: Option<String>,

	/// Text to postfix each user input with
	pub postfix: Option<String>,

	/// Tokens that users should not be able to input as they are used for signalling
	pub private_tokens: Option<Vec<String>>,

	/// Maximum number of tokens to be generated (when biaser is enabled: applies only to unbiased phase when bias_prompt is used)
	pub max_tokens: Option<usize>,

	/// Biaser: the biaser to apply to the output (if any)
	pub biaser: Option<BiaserConfig>,

	/// When configured, first (up to max_tokens) tokens are inferred without bias, then this prompt is fed, after which
	/// a biased response is generated.
	pub bias_prompt: Option<String>,

	/// Sequences that when they occur end generation (just like end-of-text token)
	#[serde(default = "default_stop_sequences")]
	pub stop_sequences: Vec<String>,

	/// Sampler configuration
	#[serde(flatten)]
	pub sampler: SamplerConfig,

	/// Memorization config
	pub memorization: Option<TaskMemorizationConfig>,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum SamplerConfig {
	Advanced(AdvancedSamplerConfig),
	Standard(StandardSamplerConfig),
}

#[derive(Deserialize, Debug, Clone)]
pub struct AdvancedSamplerConfig {
	// Samplers to apply
	pub samplers: Vec<String>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct StandardSamplerConfig {
	/// The top K words by score are kept during sampling.
	#[serde(default = "default_top_k")]
	pub top_k: usize,

	/// The cumulative probability after which no more words are kept for sampling.
	#[serde(default = "default_top_p")]
	pub top_p: f32,

	/// The penalty for repeating tokens. Higher values make the generation less
	/// likely to get into a loop, but may harm results when repetitive outputs
	/// are desired.
	#[serde(default = "default_repeat_penalty")]
	pub repeat_penalty: f32,

	/// Temperature (randomness) used for sampling. A higher number is more random.
	#[serde(default = "default_temperature")]
	pub temperature: f32,

	/// The number of tokens to consider for the repetition penalty.
	#[serde(default = "default_repetition_penalty_last_n")]
	pub repetition_penalty_last_n: usize,
}

impl SamplerConfig {
	pub(crate) fn sampler_chain(&self) -> SamplerChain {
		match self {
			SamplerConfig::Standard(st) => st.sampler_chain(),
			SamplerConfig::Advanced(a) => a.sampler_chain(),
		}
	}
}

impl AdvancedSamplerConfig {
	pub(crate) fn sampler_chain(&self) -> SamplerChain {
		let sampler_options = self
			.samplers
			.iter()
			.map(|s| s.trim())
			.filter(|s| !s.is_empty())
			.map(|s| "/".to_string() + s)
			.collect::<String>();
		let configured_samplers = ConfiguredSamplers::from_str(&sampler_options).expect("valid sampler chain");
		configured_samplers.builder.into_chain()
	}
}

impl StandardSamplerConfig {
	pub(crate) fn sampler_chain(&self) -> SamplerChain {
		let StandardSamplerConfig {
			repeat_penalty,
			repetition_penalty_last_n,
			top_k,
			top_p,
			temperature,
			..
		} = self.clone();

		SamplerChainBuilder::from([
			(
				"repetition",
				SamplerSlot::new_chain(
					move || Box::new(SampleRepetition::default().penalty(repeat_penalty).last_n(repetition_penalty_last_n)),
					[],
				),
			),
			(
				"topk",
				SamplerSlot::new_single(move || Box::new(SampleTopK::default().k(top_k)), Option::<SampleTopK>::None),
			),
			(
				"topp",
				SamplerSlot::new_single(move || Box::new(SampleTopP::default().p(top_p)), Option::<SampleTopP>::None),
			),
			(
				"temperature",
				SamplerSlot::new_single(
					move || Box::new(SampleTemperature::default().temperature(temperature)),
					Option::<SampleTemperature>::None,
				),
			),
			("randdistrib", SamplerSlot::new_static(|| Box::<SampleRandDistrib>::default())),
		])
		.into_chain()
	}
}

impl TaskConfig {
	pub(crate) fn sampler_chain(&self) -> SamplerChain {
		self.sampler.sampler_chain()
	}
}

const fn default_stop_sequences() -> Vec<String> {
	vec![]
}

const fn default_batch_size() -> usize {
	8
}

const fn default_top_k() -> usize {
	40
}

const fn default_top_p() -> f32 {
	0.95
}

const fn default_repeat_penalty() -> f32 {
	1.30
}

const fn default_temperature() -> f32 {
	0.80
}

const fn default_repetition_penalty_last_n() -> usize {
	64
}

#[derive(Deserialize, Clone, Debug, Default)]
#[serde(default)]
pub struct BackendConfig {
	/// Models that are used
	pub models: HashMap<String, ModelConfig>,

	/// Tasks that are made available
	pub tasks: HashMap<String, TaskConfig>,

	/// Memories
	pub memories: HashMap<String, MemoryConfig>,

	/// Directory to store downloaded assets
	pub cache_path: Option<PathBuf>,
}
