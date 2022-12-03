#[cfg(test)]
mod tests {
    use logos::Logos;
    use crate::parser::tokens::Token;

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
        let mut lex = Token::lexer(s);
        loop {
            match lex.next() {
                // Expected error
                None => break,
                Some(Err(_)) => {}
                _ => panic!("expecting error during parse"),
            }
        }
    }
}
