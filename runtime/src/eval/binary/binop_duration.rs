use metricsql::ast::DurationExpr;
use metricsql::prelude::Operator;

use crate::{QueryValue, RuntimeError, RuntimeResult};

pub(crate) fn eval_duration_scalar_binop(
    dur: &DurationExpr,
    scalar: f64,
    op: Operator,
    step: i64,
) -> RuntimeResult<QueryValue> {
    let d = dur.value(step);
    match op {
        Operator::Add => {
            let millis = scalar as i64 * 1000_i64;
            Ok(QueryValue::Scalar((d + millis) as f64))
        }
        Operator::Sub => {
            let millis = scalar as i64 * 1000_i64;
            Ok(QueryValue::Scalar((d - millis) as f64))
        }
        Operator::Mul => {
            let n = d as f64 * scalar; // todo: saturating_mul
            Ok(QueryValue::Scalar(n))
        }
        Operator::Div => {
            let n = d as f64 / scalar; // todo: saturating_mul
            Ok(QueryValue::Scalar(n))
        }
        _ => Err(RuntimeError::NotImplemented(format!(
            "Invalid operator for duration: {:?}",
            op
        ))),
    }
}

pub(crate) fn eval_duration_duration_binop(
    dur_a: &DurationExpr,
    dur_b: &DurationExpr,
    op: Operator,
    step: i64,
) -> RuntimeResult<QueryValue> {
    let a = dur_a.value(step);
    let b = dur_b.value(step);
    match op {
        Operator::Add => Ok(QueryValue::Scalar((a + b) as f64)),
        Operator::Sub => Ok(QueryValue::Scalar((a - b) as f64)),
        _ => Err(RuntimeError::NotImplemented(format!(
            "Invalid operation: {dur_a} {op} {dur_b}"
        ))),
    }
}
