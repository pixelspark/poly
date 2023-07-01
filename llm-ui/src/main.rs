use iced::alignment::Horizontal;
use iced::futures::channel::mpsc::Sender;
use iced::widget::scrollable::RelativeOffset;
use iced::widget::{button, column, row, scrollable, text, text_input};
use iced::{executor, Application, Color, Command, Element, Subscription, Theme};
use iced::{widget::container, Length, Settings};
use once_cell::sync::Lazy;
use worker::{LLMWorkerCommand, LLMWorkerEvent};

mod worker;

static CHAT_MESSAGES_SCROLLABLE_ID: Lazy<scrollable::Id> = Lazy::new(scrollable::Id::unique);
static CHAT_INPUT_ID: Lazy<text_input::Id> = Lazy::new(text_input::Id::unique);

struct ChatMessage {
	text: String,
	from_user: bool,
}

struct App {
	message: String,
	messages: Vec<ChatMessage>,
	sender: Option<Sender<LLMWorkerCommand>>,
	running: bool,
}

#[derive(Debug, Clone)]
pub enum AppMessage {
	Reset,
	Type(String),
	Send,
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
			},
			Command::none(),
		)
	}

	fn title(&self) -> String {
		String::from("LLM")
	}

	fn subscription(&self) -> Subscription<Self::Message> {
		worker::llm_worker().map(AppMessage::WorkerEvent)
	}

	fn update(&mut self, message: Self::Message) -> Command<AppMessage> {
		match message {
			AppMessage::Type(t) => self.message = t,

			AppMessage::WorkerEvent(wevt) => {
				match wevt {
					LLMWorkerEvent::Ready(sender) => self.sender = Some(sender),
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
					sender.try_send(LLMWorkerCommand::Reset).unwrap();
				}
			}
		};

		Command::none()
	}

	fn view(&self) -> Element<AppMessage> {
		if self.sender.is_none() {
			return container(container(text("Loading...").size(25)).center_x().center_y())
				.height(Length::Fill)
				.width(Length::Fill)
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

		container(column![
			// Toolbar
			row![button("Reset").on_press(AppMessage::Reset)].spacing(5),
			// Messages
			scrollable(column(self.messages.iter().map(|m| -> Element<AppMessage> { m.view() }).collect()).spacing(5))
				.width(Length::Fill)
				.height(Length::Fill)
				.id(CHAT_MESSAGES_SCROLLABLE_ID.clone()),
			// Text input
			input
		])
		.padding(10)
		.height(Length::Fill)
		.width(Length::Fill)
		.into()
	}
}

impl ChatMessage {
	pub fn view(&self) -> Element<AppMessage> {
		container(
			text(&self.text)
				.horizontal_alignment(if self.from_user { Horizontal::Right } else { Horizontal::Left })
				.width(Length::Fill)
				.style(if self.from_user {
					Color::from_rgb8(0, 55, 100)
				} else {
					Color::from_rgb8(236, 0, 0)
				}),
		)
		.padding(5)
		.into()
	}
}

pub fn main() -> iced::Result {
	tracing_subscriber::fmt::init();
	App::run(Settings::default())
}
