use std::{
	fs::File,
	io::Read,
	sync::{
		atomic::{AtomicBool, Ordering},
		Arc,
	},
};

use directories::ProjectDirs;
use iced::{
	futures::{channel::mpsc, SinkExt, StreamExt},
	subscription, Subscription,
};
use poly_backend::{
	backend::{Backend, InferenceFeedback, InferenceResponse},
	config::BackendConfig,
	types::{PromptRequest, SessionRequest},
};
use tokio::{select, task::spawn_blocking};

use crate::util::resource_path;

#[derive(Debug, Clone)]
pub enum LLMWorkerEvent {
	Loading(f64),
	Ready {
		sender: mpsc::Sender<LLMWorkerCommand>,
		tasks: Vec<String>,
		selected_task: String,
	},
	Running(bool),
	ResponseToken(String),
}

pub enum LLMWorkerCommand {
	Prompt(String),
	Interrupt,
	Reset { task_name: String },
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
		if let Some(proj_dirs) = ProjectDirs::from("nl", "Dialogic", "Poly") {
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
			if let Some(ref model_path) = model_config.model_path {
				let model_path_str = model_path.to_str().unwrap();
				// Paths that are prefixed with '@' are relative to the data folder
				if model_path_str.starts_with('@') {
					model_config.model_path = Some(resource_path(model_path_str.strip_prefix('@').unwrap()));
				}
			}
		}

		let mut task_names: Vec<String> = config.tasks.keys().cloned().collect();
		task_names.sort();
		let mut selected_task_name = config.tasks.keys().next().unwrap().clone();

		// Load backend
		let backend = Arc::new({
			let (ptx, mut prx) = tokio::sync::mpsc::channel(32);
			let backend_future = Backend::from(config, Some(ptx));

			let mut output2 = output.clone();
			tokio::spawn(async move {
				while let Some(progress) = prx.recv().await {
					output2.send(LLMWorkerEvent::Loading(progress)).await.unwrap();
				}
			});

			tokio::spawn(backend_future).await.unwrap()
		});
		let mut session = backend.start(&selected_task_name, &SessionRequest {}, backend.clone()).unwrap();

		loop {
			match &mut state {
				LLMWorkerState::Starting => {
					// Create channel
					let (sender, receiver) = mpsc::channel(100);

					// Send the sender back to the application
					output
						.send(LLMWorkerEvent::Ready {
							sender,
							tasks: task_names.clone(),
							selected_task: selected_task_name.clone(),
						})
						.await
						.unwrap();

					// We are ready to receive messages
					state = LLMWorkerState::Ready(receiver);
				}
				LLMWorkerState::Ready(receiver) => {
					// Read next input sent from `Application`
					let input = receiver.select_next_some().await;

					match input {
						LLMWorkerCommand::Reset { task_name } => {
							// Create a new session
							selected_task_name = task_name;
							session = backend.start(&selected_task_name, &SessionRequest {}, backend.clone()).unwrap();
						}

						LLMWorkerCommand::Interrupt => {}

						LLMWorkerCommand::Prompt(prompt) => {
							let (ptx, mut prx) = tokio::sync::mpsc::channel(16);

							// Do some async work...
							let cancelled = Arc::new(AtomicBool::new(false));
							let cancelled_clone = cancelled.clone();

							output.send(LLMWorkerEvent::Running(true)).await.unwrap();
							let session_fut = spawn_blocking(move || {
								// Swallow errors. Typically 'context full'
								// TODO handle this in a better way
								let _ = session.complete(&PromptRequest { prompt }, |feo| {
									match feo {
										InferenceResponse::SnapshotToken(_) => {}
										InferenceResponse::PromptToken(_) => {}
										InferenceResponse::InferredToken(ft) => {
											ptx.blocking_send(ft).unwrap();
										}
										InferenceResponse::EotToken => return Ok(InferenceFeedback::Halt),
									}
									if cancelled_clone.load(Ordering::SeqCst) {
										return Ok(InferenceFeedback::Halt);
									}
									Ok(InferenceFeedback::Continue)
								});
								session
							});

							loop {
								select! {
									token = prx.recv() => {
										match token {
											Some(token) => output.send(LLMWorkerEvent::ResponseToken(token)).await.unwrap(),
											None => break
										};
									},
									LLMWorkerCommand::Interrupt = receiver.select_next_some() => {
										tracing::info!("Interrupted");
										cancelled.store(true, Ordering::SeqCst);
										break;
									},
									else => break
								}
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
