use crate::components::chatmessage::{ChatMessage, ChatMessageMessage};
use crate::worker::{LLMWorkerCommand, LLMWorkerEvent};
use iced::alignment::Horizontal;
use iced::futures::channel::mpsc::Sender;
use iced::widget::scrollable::RelativeOffset;
use iced::widget::{button, column, pick_list, progress_bar, row, scrollable, text, text_input};
use iced::{clipboard, executor, Alignment, Application, Color, Command, Element, Subscription, Theme};
use iced::{widget::container, Length};
use once_cell::sync::Lazy;

static CHAT_MESSAGES_SCROLLABLE_ID: Lazy<scrollable::Id> = Lazy::new(scrollable::Id::unique);
static CHAT_INPUT_ID: Lazy<text_input::Id> = Lazy::new(text_input::Id::unique);

pub struct App {
	message: String,
	messages: Vec<ChatMessage>,
	sender: Option<Sender<LLMWorkerCommand>>,
	tasks: Vec<String>,
	selected_task: Option<String>,
	running: bool,
	loading_progress: f64,
}

#[derive(Debug, Clone)]
pub enum AppMessage {
	ChangeTask(String),
	CopyText(String),
	Interrupt,
	Reset,
	Send,
	Type(String),
	WorkerEvent(LLMWorkerEvent),
}

impl Application for App {
	type Message = AppMessage;
	type Executor = executor::Default;
	type Theme = Theme;
	type Flags = ();

	fn new(_flags: Self::Flags) -> (Self, Command<AppMessage>) {
		(
			App {
				message: String::new(),
				messages: vec![],
				sender: None,
				running: false,
				loading_progress: 0.0,
				tasks: vec![],
				selected_task: None,
			},
			Command::none(),
		)
	}

	fn title(&self) -> String {
		String::from("Poly")
	}

	fn subscription(&self) -> Subscription<Self::Message> {
		crate::worker::llm_worker().map(AppMessage::WorkerEvent)
	}

	fn update(&mut self, message: Self::Message) -> Command<AppMessage> {
		match message {
			AppMessage::Type(t) => self.message = t,
			AppMessage::ChangeTask(t) => {
				if !self.selected_task.as_ref().is_some_and(|x| x == &t) {
					self.selected_task = Some(t);
					self.messages.clear();
					if let Some(ref mut sender) = self.sender {
						sender
							.try_send(LLMWorkerCommand::Reset {
								task_name: self.selected_task.clone().unwrap(),
							})
							.unwrap();
					}
				}
			}
			AppMessage::CopyText(t) => return clipboard::write(t),
			AppMessage::Interrupt => {
				if let Some(ref mut sender) = self.sender {
					sender.try_send(LLMWorkerCommand::Interrupt).unwrap();
				}
			}

			AppMessage::WorkerEvent(wevt) => {
				match wevt {
					LLMWorkerEvent::Loading(progress) => {
						self.loading_progress = progress;
					}
					LLMWorkerEvent::Ready {
						sender,
						tasks,
						selected_task,
					} => {
						self.sender = Some(sender);
						self.tasks = tasks;
						self.selected_task = Some(selected_task);
					}
					LLMWorkerEvent::Running(r) => {
						self.running = r;
						return iced::widget::text_input::focus(CHAT_INPUT_ID.clone());
					}
					LLMWorkerEvent::ResponseToken(rt) => {
						if let Some(last) = self.messages.last_mut() {
							if !last.from_user {
								last.text.push_str(&rt);
							} else {
								self.messages.push(ChatMessage { text: rt, from_user: false });
							}
						} else {
							self.messages.push(ChatMessage { text: rt, from_user: false });
						}

						return scrollable::snap_to(CHAT_MESSAGES_SCROLLABLE_ID.clone(), RelativeOffset::END);
					}
				};
			}
			AppMessage::Send => {
				if let Some(ref mut sender) = self.sender {
					let message = std::mem::take(&mut self.message);
					self.messages.push(ChatMessage {
						text: message.clone(),
						from_user: true,
					});
					sender.try_send(LLMWorkerCommand::Prompt(message)).unwrap();
				}
			}
			AppMessage::Reset => {
				self.messages.clear();
				if let Some(ref mut sender) = self.sender {
					sender
						.try_send(LLMWorkerCommand::Reset {
							task_name: self.selected_task.clone().unwrap(),
						})
						.unwrap();
				}
			}
		};

		text_input::focus(CHAT_INPUT_ID.clone())
	}

	fn view(&self) -> Element<AppMessage> {
		if self.sender.is_none() {
			return container(
				column![
					text("Loading models...").size(25).horizontal_alignment(Horizontal::Center),
					progress_bar(0.0..=1.0, self.loading_progress as f32)
				]
				.spacing(10),
			)
			.height(Length::Fill)
			.width(Length::Fill)
			.align_y(iced::alignment::Vertical::Center)
			.align_x(Horizontal::Center)
			.padding(30)
			.into();
		}

		let input: Element<AppMessage> = if self.running {
			Element::new(text("Working..."))
		} else {
			Element::new(
				text_input("type a message...", &self.message)
					.on_input(AppMessage::Type)
					.on_submit(AppMessage::Send)
					.id(CHAT_INPUT_ID.clone()),
			)
		};

		container(
			column![
				// Toolbar
				row![
					if self.messages.is_empty() || self.running {
						Element::new(text(""))
					} else {
						button("Restart").on_press(AppMessage::Reset).into()
					},
					if self.running {
						button("Stop").on_press(AppMessage::Interrupt).into()
					} else {
						Element::new(text(""))
					},
					if self.tasks.is_empty() {
						Element::new(text(self.selected_task.clone().unwrap_or("".to_string())))
					} else {
						pick_list(&self.tasks, self.selected_task.clone(), AppMessage::ChangeTask)
							.width(Length::Fill)
							.into()
					}
				]
				.spacing(5)
				.align_items(Alignment::Center)
				.width(Length::Fill),
				// Messages
				scrollable(if self.messages.is_empty() {
					Element::new(
						text("Ready to chat.")
							.horizontal_alignment(Horizontal::Center)
							.width(Length::Fill)
							.style(iced::theme::Text::Color(Color::from_rgb8(77, 77, 77))),
					)
				} else {
					Element::new(
						column(
							self.messages
								.iter()
								.map(|m| -> Element<AppMessage> {
									m.view().map(|cmm| match cmm {
										ChatMessageMessage::CopyText(t) => AppMessage::CopyText(t),
									})
								})
								.collect(),
						)
						.spacing(5),
					)
				})
				.width(Length::Fill)
				.height(Length::Fill)
				.id(CHAT_MESSAGES_SCROLLABLE_ID.clone()),
				// Text input
				input
			]
			.spacing(5),
		)
		.padding(10)
		.height(Length::Fill)
		.width(Length::Fill)
		.into()
	}
}
