#[cfg(test)]
mod tests {
	use crate::lexer::duration;

	#[test]
	fn test_is_special_integer_prefix() {
		fn f(s: String, expected: bool) {
			let result = is_special_integer_prefix(s);
			assert_eq!(result, expected, "unexpected result for is_special_integer_prefix({}); got {}; want {}", s, result, expected)
		}

		f("", false);
		f("1", false);
		f("0", false);

		// octal numbers
		f("03", true);
		f("0o1", true);
		f("0O12", true);

		// binary numbers
		f("0b1110", true);
		f("0B0", true);

		// hex number
		f("0x1ffa", true);
		f("0X4", true);
	}

	#[test]
	fn test_scan_ident() {
		fn f(s: &str, result_expected: &str) {
			let result = scan_ident(s);
			assert_eq!(result, result_expected,
					   "unexpected result for scanIdent({}): got {}; want {}}", s, result, result_expected)
		}

		f("a", "a");
		f("foo.bar:baz_123", "foo.bar:baz_123");
		f("a+b", "a");
		f("foo()", "foo");
		f(r"a\-b+c", r"a\-b");
		f(r"a\ b\\\ c\", r"a\ b\\\ c\");
		f(r"\п\р\и\в\е\т123", r"\п\р\и\в\е\т123");
	}


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
		let mut lex: Lexer::new(s);
		loop {
			match lex.next() {
				// Expected error
				Err(_) => break,
				_=> {}
			}
			if isEOF(lex.Token) {
				panic!("expecting error during parse")
			}
		}

		// Try calling Next again. It must return error.
		match lex.next() {
			Ok(_) => panic!("expecting non-nil error"),
			_ => {}
		}
	}

}