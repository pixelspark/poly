use clap::Parser;
use llm::ModelArchitecture;
use serde::{Deserialize, Deserializer};
use std::path::PathBuf;

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
pub struct Endpoint {
	#[serde(deserialize_with = "architecture_from_str")]
	pub architecture: ModelArchitecture,
	pub model_path: PathBuf,

	/// Path for the endpoint URL
	pub name: String,
}

#[derive(Deserialize, Clone, Debug)]
pub struct Config {
	pub bind_address: String,

	pub endpoints: Vec<Endpoint>,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
	/// Where to load the config file from
	#[arg(long, short = 'm', default_value = "config.toml")]
	pub config_path: PathBuf,
}
