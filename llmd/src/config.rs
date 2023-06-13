use clap::Parser;
use llm::ModelArchitecture;
use llm_bias::json::JsonSchema;
use serde::{Deserialize, Deserializer};
use std::{collections::HashMap, path::PathBuf};

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

#[derive(Deserialize, Debug, Clone)]
pub struct ModelConfig {
	/// The model architecture type
	#[serde(deserialize_with = "architecture_from_str")]
	pub architecture: ModelArchitecture,

	/// Path to the model file
	pub model_path: PathBuf,

	/// Threads per session
	#[serde(default = "default_threads_per_session")]
	pub threads_per_session: usize,

	/// Context size
	#[serde(default = "default_context_size")]
	pub context_size: usize,
}

const fn default_threads_per_session() -> usize {
	8
}

const fn default_context_size() -> usize {
	512
}

#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
pub enum BiaserConfig {
	JsonSchema(JsonSchema),
	JsonSchemaFile(PathBuf),
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

	/// Sequences that when they occur end generation (just like end-of-text token)
	#[serde(default = "default_stop_sequences")]
	pub stop_sequences: Vec<String>,
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
	512
}

#[derive(Deserialize, Clone, Debug)]
#[serde(default)]
pub struct Config {
	/// Address and port to bind the server to ("0.0.0.0:1234")
	pub bind_address: String,

	/// Models that are used
	pub models: HashMap<String, ModelConfig>,

	/// Tasks that are made available
	pub tasks: HashMap<String, TaskConfig>,

	/// CORS allowed origins
	pub allowed_origins: Option<Vec<String>>,

	/// The maximum number of concurrent requests serviced
	pub max_concurrent: usize,

	/// Allowed API keys. When empty, all keys will be allowed.
	pub allowed_keys: Vec<String>,
}

impl Default for Config {
	fn default() -> Self {
		Self {
			bind_address: String::from("0.0.0.0:3000"),
			models: HashMap::new(),
			tasks: HashMap::new(),
			allowed_origins: None,
			max_concurrent: 8,
			allowed_keys: vec![],
		}
	}
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
	/// Where to load the config file from
	#[arg(long, short = 'm', default_value = "config.toml")]
	pub config_path: PathBuf,
}
