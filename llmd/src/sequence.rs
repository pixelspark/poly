#[derive(Debug)]
pub struct Sequence {
	tokens: String,
	state: usize,
}

impl Sequence {
	pub fn new(tokens: String) -> Sequence {
		Sequence { tokens, state: 0 }
	}

	fn is_complete(&self) -> bool {
		self.state == self.tokens.len()
	}

	pub fn advance(&mut self, token: &str) -> bool {
		if self.state >= self.tokens.len() {
			true // Already complete
		} else {
			if token.as_bytes().starts_with(&self.tokens.as_bytes()[self.state..]) {
				self.state += token.len();
			} else {
				// Reset back to zero
				self.state = 0;
			}
			self.is_complete()
		}
	}

	pub fn reset(&mut self) {
		self.state = 0;
	}
}

#[derive(Debug)]
pub struct SequenceSet {
	sequences: Vec<Sequence>,
}

impl SequenceSet {
	pub fn new(sequences: Vec<Sequence>) -> SequenceSet {
		SequenceSet { sequences }
	}

	pub fn reset(&mut self) {
		self.sequences.iter_mut().for_each(|s| s.reset());
	}

	/// Advance the sequences. If any of them is completed (or there are none), returns true
	pub fn advance(&mut self, token: &str) -> bool {
		if self.sequences.is_empty() {
			return true;
		}

		let mut any_complete = false;
		self.sequences.iter_mut().for_each(|s| {
			any_complete = s.advance(token) || any_complete;
		});
		any_complete
	}
}

#[cfg(test)]
mod test {
	use super::Sequence;
	use super::SequenceSet;

	#[test]
	fn test_sequences() {
		let mut s = SequenceSet::new(vec![Sequence::new("def".to_string()), Sequence::new("a".to_string())]);

		assert!(s.advance("a"));
		s.reset();
		assert!(!s.advance("d"));
		assert!(!s.advance("e"));
		assert!(s.advance("f"));
	}
}
