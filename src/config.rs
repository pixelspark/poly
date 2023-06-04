use clap::Parser;
use llm::ModelArchitecture;
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

pub const DEFAULT_THREADS_PER_SESSION: usize = 8;

#[derive(Deserialize, Debug, Clone)]
pub struct ModelConfig {
	/// The model architecture type
	#[serde(deserialize_with = "architecture_from_str")]
	pub architecture: ModelArchitecture,

	/// Path to the model file
	pub model_path: PathBuf,

	/// Threads per session
	pub threads_per_session: Option<usize>,
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
