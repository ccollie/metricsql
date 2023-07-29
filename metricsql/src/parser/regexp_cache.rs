use crate::parser::{ParseError, ParseResult};
use regex::Regex;

/// todo: have cache.
pub fn compile_regexp(re: &str) -> ParseResult<Regex> {
    match Regex::new(re) {
        Err(_) => Err(ParseError::InvalidRegex(re.to_string())),
        Ok(regex) => Ok(regex),
    }
}

pub fn is_empty_regex(re: &str) -> bool {
    match re {
        "" | "?:" | "^$" | "^.*$" | "^.*" | ".*$" | ".*" | ".^" => true, // cheap check
        _ => match compile_regexp(format!("^{}$", re).as_str()) {
            Err(_) => false,
            Ok(regex) => regex.is_match(""),
        },
    }
}

pub(crate) fn is_regex_match(re: &str, s: &str) -> bool {
    match compile_regexp(re) {
        Err(_) => false,
        Ok(regex) => regex.is_match(s),
    }
}
