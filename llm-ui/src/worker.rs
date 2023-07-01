use std::{fs::File, io::Read};

use iced::{
	futures::{channel::mpsc, SinkExt, StreamExt},
	subscription, Subscription,
};
use llmd::{
	api::{PromptRequest, SessionRequest},
	backend::Backend,
	config::Config,
};
use tokio::task::spawn_blocking;

#[derive(Debug, Clone)]
pub enum LLMWorkerEvent {
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

		let mut config_file = File::open("config.toml").expect("open config file");
		let mut config_string = String::new();
		config_file.read_to_string(&mut config_string).expect("read config file");
		let config: Config = toml::from_str(&config_string).unwrap();
		let backend = Backend::from(config);
		let first_task_name = backend.config.tasks.keys().next().unwrap();
		let mut session = backend.start(first_task_name, &SessionRequest {}).unwrap();

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
							session = backend.start(first_task_name, &SessionRequest {}).unwrap();
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
