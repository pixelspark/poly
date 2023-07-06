use clap::Parser;
use jsonwebtoken::{Algorithm, DecodingKey, EncodingKey};
pub use llm::ModelArchitecture;
use poly_backend::config::BackendConfig;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize, Clone, Debug)]
#[serde(rename_all = "snake_case")]
pub enum JwtPrivateKey {
	Symmetric(String),
}

#[derive(Deserialize, Clone, Debug)]
#[serde(default)]
pub struct Config {
	/// Address and port to bind the server to ("0.0.0.0:1234")
	pub bind_address: String,

	#[serde(flatten)]
	pub backend_config: BackendConfig,

	/// CORS allowed origins
	pub allowed_origins: Option<Vec<String>>,

	/// The maximum number of concurrent requests serviced
	pub max_concurrent: usize,

	/// Whether access is allowed without keys
	pub public: bool,

	/// Allowed static API keys
	pub allowed_keys: Vec<String>,

	/// Key for JWT signed keys
	pub jwt_private_key: Option<JwtPrivateKey>,
}

impl Default for Config {
	fn default() -> Self {
		Self {
			bind_address: String::from("0.0.0.0:3000"),
			backend_config: BackendConfig::default(),
			allowed_origins: None,
			max_concurrent: 8,
			allowed_keys: vec![],
			public: false,
			jwt_private_key: None,
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

impl JwtPrivateKey {
	pub fn algorithm(&self) -> Algorithm {
		match self {
			Self::Symmetric(_) => Algorithm::HS256,
		}
	}

	pub fn decoding_key(&self) -> DecodingKey {
		match self {
			JwtPrivateKey::Symmetric(s) => DecodingKey::from_secret(s.as_bytes()),
		}
	}

	pub fn encoding_key(&self) -> EncodingKey {
		match self {
			JwtPrivateKey::Symmetric(s) => EncodingKey::from_secret(s.as_bytes()),
		}
	}
}
