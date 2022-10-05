use crate::parser::ParseError;

// todo: rename
pub fn parse_float(str: &str) -> Result<f64, ParseError> {
    if str.starts_with("0x") {
        match u64::from_str_radix(str, 16) {
            Ok(n) => Ok(n as f64),
            Err(_) => Err(ParseError::InvalidNumber(str.to_string())),
        }
    } else if str.starts_with("0b") {
        match u64::from_str_radix(str, 2) {
            Ok(n) => Ok(n as f64),
            Err(_) => Err(ParseError::InvalidNumber(str.to_string())),
        }
    } else {
        match str.to_lowercase().as_str() {
            "+inf" | "inf" => Ok(f64::INFINITY),
            "-inf" => Ok(f64::NEG_INFINITY),
            "-nan" | "+nan" | "nan" => Ok(f64::NAN),
            _ => match str.parse::<f64>() {
                Ok(n) => Ok(n),
                Err(_) => Err(ParseError::InvalidNumber(str.to_string())),
            },
        }
    }
}
