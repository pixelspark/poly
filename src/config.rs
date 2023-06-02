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
pub struct Endpoint {
	/// The model architecture type
	#[serde(deserialize_with = "architecture_from_str")]
	pub architecture: ModelArchitecture,

	/// Path to the model file
	pub model_path: PathBuf,

	/// Threads per session
	pub threads_per_session: Option<usize>,
}

#[derive(Deserialize, Clone, Debug)]
#[serde(default)]
pub struct Config {
	/// Address and port to bind the server to ("0.0.0.0:1234")
	pub bind_address: String,

	/// Endpoints to serve
	pub endpoints: HashMap<String, Endpoint>,

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
			endpoints: HashMap::new(),
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
