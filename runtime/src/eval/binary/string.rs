use metricsql::binaryop::string_compare;
use metricsql::common::Operator;

use crate::{QueryValue, RuntimeError, RuntimeResult};

// move to metricsql binop module ?
pub(crate) fn eval_string_string_op(
    op: Operator,
    left: &str,
    right: &str,
    is_bool: bool,
) -> RuntimeResult<QueryValue> {
    match op {
        Operator::Add => Ok(QueryValue::String(format!("{}{}", left, right))),
        _ => {
            if op.is_comparison() {
                let cmp = string_compare(right, left, op, is_bool).map_err(|_| {
                    RuntimeError::Internal(format!("string compare failed: op = {op}"))
                })?;
                Ok(QueryValue::Scalar(cmp))
            } else {
                Err(RuntimeError::NotImplemented(format!(
                    "Unimplemented string operator: {} {} {}",
                    op, left, right
                )))
            }
        }
    }
}
