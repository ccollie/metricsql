use once_cell::sync::Lazy;
use regex::Regex;

const PATTERN_SPACE: Lazy<Regex> = Regex::new("[\t ]+");
const PATTERN_LOAD: Lazy<Regex> = Regex::new(r"^load\s+(.+?)$");
const PATTERN_EVAL_INSTANT: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^eval(?:_(fail|ordered))?\s+instant\s+(?:at\s+(.+?))?\s+(.+)$"));

static UPPER_BUCKET_RANGE: Lazy<String> = Lazy::new(|| format!("{:.3}...+Inf", UPPER_MAX));

/// Relative error allowed for sample values.
const EPSILON: f64 = 0.000001;
