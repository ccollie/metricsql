use regex::{Error, Regex};

/// CompileRegexp returns compile regexp re.
pub fn compile_regexp(re: &str) -> Result<Regex, Error> {
    Regex::new(re)
}
