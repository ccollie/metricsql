use regex::{Regex};
use crate::parser::{ParseError, ParseResult};

/// CompileRegexp returns compile regexp re.
pub fn compile_regexp(re: &str) -> ParseResult<Regex> {
    match Regex::new(re) {
        Err(_) => {
           return Err(ParseError::InvalidRegex(re.to_string()));
        }
        Ok(regex) => Ok(regex)
    }
}
