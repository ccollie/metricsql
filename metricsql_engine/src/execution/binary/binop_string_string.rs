use metricsql_parser::binaryop::string_compare;
use metricsql_parser::common::Operator;

use crate::{QueryValue, RuntimeError, RuntimeResult};

// move to metricsql_parser binop module ?
pub(crate) fn eval_string_string_binop(
    op: Operator,
    left: &str,
    right: &str,
    is_bool: bool,
) -> RuntimeResult<QueryValue> {
    match op {
        Operator::Add => {
            let mut res = String::with_capacity(left.len() + right.len());
            res += left;
            res += right;
            Ok(QueryValue::String(res))
        }
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
