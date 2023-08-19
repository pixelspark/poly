use axum::extract::State;
use axum::http::header::{AUTHORIZATION, CONTENT_TYPE};
use axum::http::{HeaderValue, Method, StatusCode};

use axum::response::IntoResponse;
use axum::routing::get;
use axum::{Json, Router};
use clap::Parser;
use poly_backend::backend::Backend;
use poly_backend::types::{Status, StatusResponse};
use poly_server::api::StatsResponse;
use poly_server::config::{Args, Config};
use poly_server::middleware::authenticate;
use poly_server::routes;
use poly_server::server::Server;

use std::net::SocketAddr;
use std::sync::Arc;
use std::{fs::File, io::Read};
use tower::limit::ConcurrencyLimitLayer;
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::ServeDir;
use tower_http::trace::TraceLayer;
use tracing::info;
use tracing_subscriber::EnvFilter;

pub use llm::InferenceFeedback;

#[tokio::main]
async fn main() {
	tracing_subscriber::fmt()
		.with_env_filter(EnvFilter::try_from_default_env().or_else(|_| EnvFilter::try_new("info")).unwrap())
		.init();
	// Read config file
	let args = Args::parse();
	let mut config_file = File::open(args.config_path).expect("open config file");
	let mut config_string = String::new();
	config_file.read_to_string(&mut config_string).expect("read config file");
	let config: Config = toml::from_str(&config_string).unwrap();
	let bind_address: SocketAddr = config.bind_address.parse().unwrap();
	info!("Starting llmd; bind address: {bind_address}",);

	// Set up CORS
	let mut cors_layer = CorsLayer::new();
	if let Some(ref origins) = config.allowed_origins {
		for origin in origins.iter() {
			if origin == "*" {
				cors_layer = cors_layer.allow_origin(Any);
			} else {
				cors_layer = cors_layer.allow_origin(origin.parse::<HeaderValue>().unwrap());
			}
		}
	} else {
		// Allow any origin by default
		cors_layer = cors_layer.allow_origin(Any);
	}
	cors_layer = cors_layer.allow_headers([CONTENT_TYPE, AUTHORIZATION]);
	cors_layer = cors_layer.allow_methods([Method::GET, Method::POST, Method::OPTIONS, Method::PUT, Method::DELETE]);

	let backend = Arc::new(Backend::from(config.backend_config.clone(), None).await);
	let state = Arc::new(Server::new(backend, config));

	// Set up API server
	let app = Router::new()
		.nest_service("/", ServeDir::new("client/dist/"))
		.route("/status", get(status_handler))
		.nest(
			"/v1",
			Router::new()
				.nest("/model", routes::models::router())
				.nest("/task", routes::tasks::router())
				.nest("/memory", routes::memories::router())
				.route("/stats", get(stats_handler))
				.layer(axum::middleware::from_fn_with_state(state.clone(), authenticate)),
		)
		.fallback(handler_not_found)
		.layer(cors_layer)
		.layer(ConcurrencyLimitLayer::new(state.config.max_concurrent))
		.layer(TraceLayer::new_for_http())
		.with_state(state);

	axum::Server::bind(&bind_address).serve(app.into_make_service()).await.unwrap();
}

async fn stats_handler(State(state): State<Arc<Server>>) -> impl IntoResponse {
	let task_stats = state.backend.stats.task_stats.lock().unwrap().clone();
	Json(StatsResponse { tasks: task_stats })
}

async fn status_handler() -> impl IntoResponse {
	Json(StatusResponse { status: Status::Ok })
}

async fn handler_not_found() -> impl IntoResponse {
	(StatusCode::NOT_FOUND, "not found")
}
