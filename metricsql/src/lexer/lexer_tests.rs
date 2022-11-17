#[cfg(test)]
mod tests {
	use crate::lexer::{Lexer};


	#[test]
	fn test_lexer_error() {
		// Invalid identifier
		test_error(".foo");

		// Incomplete string
		test_error("\"foobar");
		test_error("'");
		test_error("`");

		// Unrecognized char
		test_error("тест");

		// Invalid numbers
		test_error(".");
		test_error("123.");
		test_error("12e");
		test_error("1.2e");
		test_error("1.2E+");
		test_error("1.2E-");
	}

	fn test_error(s: &str) {
		let mut lex = Lexer::new(s);
		loop {
			match lex.next() {
				// Expected error
				None => break,
				_=> {}
			}
			if lex.is_eof() {
				panic!("expecting error during parse")
			}
		}

		// Try calling Next again. It must return error.
		match lex.next() {
			None => panic!("expecting non-nil error"),
			_ => {}
		}
	}

}