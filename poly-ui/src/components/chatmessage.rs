use iced::{
	widget::{button, container, text},
	Color, Element, Length,
};

pub struct ChatMessage {
	pub text: String,
	pub from_user: bool,
}

#[derive(Debug, Clone)]
pub enum ChatMessageMessage {
	CopyText(String),
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ChatMessageTheme {
	is_from_user: bool,
}

impl ChatMessageTheme {
	fn color(&self) -> Color {
		if self.is_from_user {
			Color::from_rgba8(0, 55, 100, 0.1)
		} else {
			Color::from_rgba8(236, 0, 0, 0.1)
		}
	}
}

impl container::StyleSheet for ChatMessageTheme {
	type Style = iced::Theme;

	fn appearance(&self, _style: &Self::Style) -> container::Appearance {
		container::Appearance {
			text_color: Some(if self.is_from_user {
				Color::from_rgb8(0, 55, 100)
			} else {
				Color::from_rgb8(236, 0, 0)
			}),
			background: Some(iced::Background::Color(self.color())),
			border_radius: 5.0.into(),
			border_width: 0.0,
			border_color: Color::TRANSPARENT,
		}
	}
}

impl ChatMessage {
	pub fn view(&self) -> Element<ChatMessageMessage> {
		container(
			button(text(&self.text))
				//.horizontal_alignment(if self.from_user { Horizontal::Right } else { Horizontal::Left })
				.width(Length::Fill)
				.style(iced::theme::Button::Text)
				.on_press(ChatMessageMessage::CopyText(self.text.clone())),
		)
		.padding(5)
		.style(iced::theme::Container::Custom(Box::new(ChatMessageTheme {
			is_from_user: self.from_user,
		})))
		.into()
	}
}
