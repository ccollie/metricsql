use metricsql::ast::{DurationExpr, Expr, NumberLiteral};
use metricsql::prelude::Operator;

use crate::{RuntimeError, RuntimeResult};

// todo: add dur * scalar, scalar * dur, dur / scalar, scalar / dur to optimizer
pub(crate) fn duration_op_scalar(
    dur: &DurationExpr,
    scalar: f64,
    op: Operator,
    step: i64,
) -> RuntimeResult<Expr> {
    let d = dur.value(step);
    match op {
        Operator::Add => {
            let millis = scalar as i64 * 1000_i64;
            let dur = DurationExpr::new(d + millis, false);
            Ok(Expr::Duration(dur))
        }
        Operator::Sub => {
            let millis = scalar as i64 * 1000_i64;
            let dur = DurationExpr::new(d - millis, false);
            Ok(Expr::Duration(dur))
        }
        Operator::Mul => {
            let n = d as f64 * scalar; // todo: saturating_mul
            let dur = DurationExpr::new(n as i64, false);
            Ok(Expr::Duration(dur))
        }
        Operator::Div => {
            let n = d as f64 / scalar; // todo: saturating_mul
            let dur = DurationExpr::new(n as i64, false);
            Ok(Expr::Duration(dur))
        }
        _ => Err(RuntimeError::NotImplemented(format!(
            "Invalid operator for duration: {:?}",
            op
        ))),
    }
}

pub(crate) fn eval_duration_op_duration(
    dur_a: &DurationExpr,
    dur_b: DurationExpr,
    op: Operator,
    step: i64,
) -> RuntimeResult<Expr> {
    let a = dur_a.value(step);
    let b = dur_b.value(step);
    match op {
        Operator::Add => {
            let dur = DurationExpr::new(a + b, false);
            Ok(Expr::Duration(dur))
        }
        Operator::Sub => {
            let dur = DurationExpr::new(a - b, false);
            Ok(Expr::Duration(dur))
        }
        _ => Err(RuntimeError::NotImplemented(format!(
            "Invalid operator for duration: {:?}",
            op
        ))),
    }
}
