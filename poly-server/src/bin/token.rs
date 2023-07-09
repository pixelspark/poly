use std::{fs::File, io::Read, path::PathBuf};

use clap::Parser;
use jsonwebtoken::{get_current_timestamp, Header};
use poly_server::{api::JwtClaims, config::Config};
use rand::Rng;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
	/// Where to load the config file from
	#[arg(long, short = 'm', default_value = "config.toml")]
	pub config_path: PathBuf,

	/// When supplied, list of tasks that this token can use
	#[arg(long, short = 't')]
	pub tasks: Option<Vec<String>>,

	/// When supplied, list of models that this token can use
	#[arg(long, short = 'm')]
	pub models: Option<Vec<String>>,

	/// When supplied, list of memories that this token can use
	#[arg(long, short = 'n')]
	pub memories: Option<Vec<String>>,

	/// User ID (`sub` claim) in token
	#[arg(long, short = 's')]
	pub sub: Option<String>,
}

pub fn main() {
	tracing_subscriber::fmt::init();
	// Read config file
	let args = Args::parse();
	let mut config_file = File::open(args.config_path).expect("open config file");
	let mut config_string = String::new();
	config_file.read_to_string(&mut config_string).expect("read config file");
	let config: Config = toml::from_str(&config_string).unwrap();

	match config.jwt_private_key {
		Some(jwk_key) => {
			let ek = jwk_key.encoding_key();
			let token = jsonwebtoken::encode(
				&Header::default(),
				&JwtClaims {
					exp: Some(get_current_timestamp() as usize + 3600),
					sub: args.sub,
					tasks: args.tasks,
					models: args.models,
					memories: args.memories,
				},
				&ek,
			)
			.unwrap();
			println!("{token}");
		}
		None => {
			tracing::error!("no private key for JWT configured; here is a suggested random key:");
			let s: String = rand::thread_rng()
				.sample_iter(&rand::distributions::Alphanumeric)
				.take(64)
				.map(char::from)
				.collect();
			tracing::error!("jwt_private_key = {{ symmetric = \"{s}\" }}");
		}
	}
}
