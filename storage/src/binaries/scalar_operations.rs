use datafusion::error::{DataFusionError, Result};

use metricsql::common::Operator;

/// Supported operation between two float type values.
pub fn scalar_binary_operations(token: Operator, lhs: f64, rhs: f64) -> Result<f64> {
    use Operator::*;
    let value = match token {
        Add => lhs + rhs,
        Sub => lhs - rhs,
        Mul => lhs * rhs,
        Div => lhs / rhs,
        Pow => lhs.powf(rhs),
        Mod => lhs % rhs,
        Eql => (lhs == rhs) as u32 as f64,
        NotEq => (lhs != rhs) as u32 as f64,
        Gt => (lhs > rhs) as u32 as f64,
        Lt => (lhs < rhs) as u32 as f64,
        Gte => (lhs >= rhs) as u32 as f64,
        Lte => (lhs <= rhs) as u32 as f64,
        Atan2 => lhs.atan2(rhs),
        _ => {
            return Err(DataFusionError::NotImplemented(format!(
                "Unsupported scalar operation: {:?} {:?} {:?}",
                token, lhs, rhs
            )))
        }
    };
    Ok(value)
}
