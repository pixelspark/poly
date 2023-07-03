use std::{fs::File, io::Read};

use directories::ProjectDirs;
use iced::{
	futures::{channel::mpsc, SinkExt, StreamExt},
	subscription, Subscription,
};
use llmd::{
	api::{PromptRequest, SessionRequest},
	backend::Backend,
	config::BackendConfig,
};
use tokio::task::spawn_blocking;

use crate::util::resource_path;

#[derive(Debug, Clone)]
pub enum LLMWorkerEvent {
	Loading(f64),
	Ready(mpsc::Sender<LLMWorkerCommand>),
	Running(bool),
	ResponseToken(String),
}

pub enum LLMWorkerCommand {
	Prompt(String),
	Reset,
}

enum LLMWorkerState {
	Starting,
	Ready(mpsc::Receiver<LLMWorkerCommand>),
}

pub fn llm_worker() -> Subscription<LLMWorkerEvent> {
	struct LLMWorker;

	subscription::channel(std::any::TypeId::of::<LLMWorker>(), 100, move |mut output| async move {
		let mut state = LLMWorkerState::Starting;

		let mut config_file_path = resource_path("config.toml");

		// Check if the user has a local override config
		if let Some(proj_dirs) = ProjectDirs::from("nl", "Dialogic", "LLM") {
			let config_dir = proj_dirs.config_dir();
			let user_config_path = config_dir.join("config.toml");
			tracing::info!("Looking for configuration file at {}", user_config_path.to_str().unwrap());
			if user_config_path.exists() {
				config_file_path = user_config_path;
				tracing::info!("Using user configuration file");
			} else {
				tracing::info!("Using built-in configuration file");
			}
		}

		// Load the config file
		let mut config_file = File::open(config_file_path).expect("open config file");
		let mut config_string = String::new();
		config_file.read_to_string(&mut config_string).expect("read config file");

		let mut config: BackendConfig = toml::from_str(&config_string).unwrap();

		// Update model paths
		for (_k, model_config) in config.models.iter_mut() {
			let model_path_str = model_config.model_path.to_str().unwrap();
			// Paths that are prefixed with '@' are relative to the data folder
			if model_path_str.starts_with('@') {
				model_config.model_path = resource_path(model_path_str.strip_prefix('@').unwrap());
			}
		}

		// Load backend
		let main_task_name = config.tasks.keys().next().unwrap().clone();
		let backend = {
			let (ptx, mut prx) = tokio::sync::mpsc::channel(16);
			let backend_future = spawn_blocking(move || {
				Backend::from(config, |progress| {
					ptx.blocking_send(progress).unwrap();
				})
			});

			while let Some(progress) = prx.recv().await {
				output.send(LLMWorkerEvent::Loading(progress)).await.unwrap();
				// For testing the loading progress bar
				// tokio::time::sleep(std::time::Duration::from_millis(50)).await;
			}

			backend_future.await.unwrap()
		};
		let mut session = backend.start(&main_task_name, &SessionRequest {}).unwrap();

		loop {
			match &mut state {
				LLMWorkerState::Starting => {
					// Create channel
					let (sender, receiver) = mpsc::channel(100);

					// Send the sender back to the application
					output.send(LLMWorkerEvent::Ready(sender)).await.unwrap();

					// We are ready to receive messages
					state = LLMWorkerState::Ready(receiver);
				}
				LLMWorkerState::Ready(receiver) => {
					// Read next input sent from `Application`
					let input = receiver.select_next_some().await;

					match input {
						LLMWorkerCommand::Reset => {
							// Create a new session
							session = backend.start(&main_task_name, &SessionRequest {}).unwrap();
						}

						LLMWorkerCommand::Prompt(prompt) => {
							let (ptx, mut prx) = tokio::sync::mpsc::channel(16);

							// Do some async work...
							output.send(LLMWorkerEvent::Running(true)).await.unwrap();
							let session_fut = spawn_blocking(move || {
								// Swallow errors. Typically 'context full'
								// TODO handle this in a better way
								let _ = session.complete(&PromptRequest { prompt }, |feo| {
									match feo {
										llmd::backend::InferenceResponse::SnapshotToken(_) => {}
										llmd::backend::InferenceResponse::PromptToken(_) => {}
										llmd::backend::InferenceResponse::InferredToken(ft) => {
											ptx.blocking_send(ft).unwrap();
										}
										llmd::backend::InferenceResponse::EotToken => return Ok(llmd::backend::InferenceFeedback::Halt),
									}
									Ok(llmd::backend::InferenceFeedback::Continue)
								});
								session
							});

							while let Some(token) = prx.recv().await {
								output.send(LLMWorkerEvent::ResponseToken(token)).await.unwrap();
							}

							session = session_fut.await.unwrap();
							output.send(LLMWorkerEvent::Running(false)).await.unwrap();
						}
					}
				}
			}
		}
	})
}
