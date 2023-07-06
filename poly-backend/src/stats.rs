use std::time::Duration;

use llm::InferenceStats;
use serde::Serialize;

pub trait InferenceStatsAdd {
	fn add(&mut self, stats: &InferenceStats);
}

impl InferenceStatsAdd for InferenceStats {
	fn add(&mut self, stats: &InferenceStats) {
		self.predict_duration += stats.predict_duration;
		self.predict_tokens += stats.predict_tokens;
		self.prompt_tokens += stats.prompt_tokens;
		self.feed_prompt_duration += stats.feed_prompt_duration;
	}
}

#[derive(Serialize, Debug, Clone)]
pub struct TaskStats {
	/// Number of completion cycles (`Backend::completion`) that were completed for this model
	cycles: usize,

	/// Total duration of prediction measured in thread-time
	predict_duration: Duration,
	predict_duration_threads: Duration,
	predict_tokens: usize,

	/// Total duration of prompt feeding measured in thread-time
	prompt_duration: Duration,
	prompt_duration_threads: Duration,
	prompt_tokens: usize,
}

impl Default for TaskStats {
	fn default() -> Self {
		Self {
			cycles: 0,

			predict_duration: Duration::ZERO,
			predict_duration_threads: Duration::ZERO,
			predict_tokens: 0,

			prompt_duration: Duration::ZERO,
			prompt_duration_threads: Duration::ZERO,
			prompt_tokens: 0,
		}
	}
}

impl TaskStats {
	pub fn add_cycle(&mut self, stats: &InferenceStats, n_threads: usize) {
		self.predict_tokens += stats.predict_tokens;
		self.prompt_tokens += stats.prompt_tokens;
		self.prompt_duration += stats.feed_prompt_duration;
		self.prompt_duration_threads += stats.feed_prompt_duration * (n_threads as u32);

		self.predict_duration += stats.predict_duration;
		self.predict_duration_threads += stats.predict_duration * (n_threads as u32);
		self.cycles += 1;
	}
}
