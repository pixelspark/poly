use iced::{Application, Settings};

mod app;
mod util;
mod worker;
use app::App;

pub fn main() -> iced::Result {
	tracing_subscriber::fmt::init();
	App::run(Settings::default())
}
