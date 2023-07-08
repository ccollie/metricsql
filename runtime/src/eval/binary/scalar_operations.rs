use metricsql::common::Operator;

use crate::{RuntimeError, RuntimeResult};

/// Supported operation between two float type values.
pub(crate) fn scalar_binary_operations(op: Operator, lhs: f64, rhs: f64) -> RuntimeResult<f64> {
    use Operator::*;
    let value = match op {
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
            return Err(RuntimeError::NotImplemented(format!(
                "Unsupported scalar operation: {:?} {:?} {:?}",
                op, lhs, rhs
            )))
        }
    };
    Ok(value)
}
