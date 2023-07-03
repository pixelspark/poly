use iced::{Application, Settings};

mod app;
mod components;
mod util;
mod worker;
use app::App;

pub fn main() -> iced::Result {
	tracing_subscriber::fmt::init();
	App::run(Settings {
		window: iced::window::Settings {
			size: (400, 700),
			min_size: Some((200, 200)),
			..Default::default()
		},
		..Default::default()
	})
}
