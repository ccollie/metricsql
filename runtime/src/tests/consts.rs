use std::cell::OnceCell;

use regex::Regex;

pub(crate) fn space_regex() -> &'static Regex {
    static PATTERN_SPACE: OnceCell<Regex> = OnceCell::new();
    PATTERN_SPACE.get_or_init(|| Regex::new("[\t ]+").unwrap())
}

pub(crate) fn load_regex() -> &'static Regex {
    static PATTERN_LOAD: OnceCell<Regex> = OnceCell::new();
    PATTERN_LOAD.get_or_init(|| Regex::new(r"^load\s+(.+?)$").unwrap())
}

pub(crate) fn eval_instant_regex() -> &'static Regex {
    static PATTERN_EVAL_INSTANT: OnceCell<Regex> = OnceCell::new();
    PATTERN_EVAL_INSTANT.get_or_init(|| {
        Regex::new(r"^eval(?:_(fail|ordered))?\s+instant\s+(?:at\s+(.+?))?\s+(.+)$").unwrap()
    })
}

pub(crate) fn upper_bucket_range() -> &'static str {
    static UPPER_BUCKET_RANGE: OnceCell<String> = OnceCell::new();
    UPPER_BUCKET_RANGE.get_or_init(|| format!("{:.3}...+Inf", UPPER_MAX))
}

/// Relative error allowed for sample values.
const EPSILON: f64 = 0.000001;
