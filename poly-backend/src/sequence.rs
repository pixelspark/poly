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
			let remainder = &self.tokens.as_bytes()[self.state..];
			let overlap_length = remainder.len().min(token.len());
			if (remainder.len() == token.len() && remainder == token.as_bytes()) || remainder.starts_with(&token.as_bytes()[0..overlap_length]) {
				self.state += overlap_length;
				// The unused part of the token (if it was longer than our remainder) can be used to advance once more
				if token.len() > remainder.len() && self.is_complete() {
					self.state = 0;
					self.advance(&token[remainder.len()..]);
					return true;
				}
			} else {
				// Reset back to zero
				if self.state != 0 {
					// Try again from the beginning if we weren't at zero already
					self.state = 0;
					return self.advance(token);
				} else {
					// Just reset back to zero
					self.state = 0;
				}
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

		s.reset();
		assert!(s.advance("defq"));

		s.reset();
		assert!(s.advance("defde"));
		assert!(s.advance("f"));

		s.reset();
		assert!(s.advance("defde"));
		assert!(s.advance("def"));

		s.reset();
		assert!(s.advance("defde"));
		println!("{s:?}");
		assert!(!s.advance("ef"));
	}
}
