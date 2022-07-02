
// Note: callbacks can return `Option` or `Result`
pub fn parse_float(str: &str) -> Option<f64> {
    if str.starts_with("0x") {

    } else if str.starts_with("0b") {

    }
    match str.to_lowercase().as_str() {
        "+inf" | "inf" => Some(f64::INFINITY),
        "-inf" => Some(f64::NEG_INFINITY),
        "+nan" | "nan" => Some(f64::NAN),
        "-nan" => Some(f64::NAN),
        _ => Some(str.parse().ok()?)
    }
}