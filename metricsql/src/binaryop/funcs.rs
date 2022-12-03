use crate::common::Operator;
use crate::parser::{ParseError, ParseResult};

/// Eq returns true of left == right.
#[inline]
pub fn eq(left: f64, right: f64) -> bool {
    // Special handling for nan == nan.
    // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/150 .
    if left.is_nan() {
        return right.is_nan();
    }
    left == right
}

/// Neq returns true of left != right.
pub fn neq(left: f64, right: f64) -> bool {
    // Special handling for comparison with nan.
    // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/150 .
    if left.is_nan() {
        return !right.is_nan();
    }
    if right.is_nan() {
        return true;
    }
    left != right
}

/// Gt returns true of left > right
#[inline]
pub fn gt(left: f64, right: f64) -> bool {
    left > right
}

/// Lt returns true if left < right
#[inline]
pub fn lt(left: f64, right: f64) -> bool {
    left < right
}

/// Gte returns true if left >= right
#[inline]
pub fn gte(left: f64, right: f64) -> bool {
    left >= right
}

/// Lte returns true if left <= right
#[inline]
pub fn lte(left: f64, right: f64) -> bool {
    left <= right
}

/// Plus returns left + right
#[inline]
pub fn plus(left: f64, right: f64) -> f64 {
    left + right
}

/// Minus returns left - right
#[inline]
pub fn minus(left: f64, right: f64) -> f64 {
    left - right
}

/// Mul returns left * right
#[inline]
pub fn mul(left: f64, right: f64) -> f64 {
    left * right
}

/// Div returns left / right
/// Todo: protect against div by zero
#[inline]
pub fn div(left: f64, right: f64) -> f64 {
    left / right
}

/// Mod returns mod(left, right)
#[inline]
pub fn mod_(left: f64, right: f64) -> f64 {
    left % right
}

/// Pow returns pow(left, right)
#[inline]
pub fn pow(left: f64, right: f64) -> f64 {
    left.powf(right)
}

/// atan2 returns atan2(left, right)
#[inline]
pub fn atan2(left: f64, right: f64) -> f64 {
    left.atan2(right)
}

/// default returns left or right if left is NaN.
pub fn default(left: f64, right: f64) -> f64 {
    if left.is_nan() {
        return right;
    }
    left
}

/// If returns left if right is not NaN. Otherwise NaN is returned.
#[inline]
pub fn if_(left: f64, right: f64) -> f64 {
    if right.is_nan() {
        return f64::NAN;
    }
    left
}

/// Ifnot returns left if right is NaN. Otherwise NaN is returned.
#[inline]
pub fn if_not(left: f64, right: f64) -> f64 {
    if right.is_nan() {
        return left;
    }
    f64::NAN
}

pub fn eval_binary_op(left: f64, right: f64, op: Operator, is_bool: bool) -> f64 {
    return if op.is_comparison() {
        fn eval_cmp(
            left: f64,
            right: f64,
            is_bool: bool,
            cf: fn(left: f64, right: f64) -> bool,
        ) -> f64 {
            if is_bool {
                return if cf(left, right) { 1_f64 } else { 0_f64 };
            }
            return if cf(left, right) { left } else { f64::NAN };
        }

        match op {
            Operator::Eql => eval_cmp(left, right, is_bool, eq),
            Operator::NotEq => eval_cmp(left, right, is_bool, neq),
            Operator::Gt => eval_cmp(left, right, is_bool, gt),
            Operator::Lt => eval_cmp(left, right, is_bool, lt),
            Operator::Gte => eval_cmp(left, right, is_bool, gte),
            Operator::Lte => eval_cmp(left, right, is_bool, lte),
            _ => panic!("BUG: unexpected comparison binaryOp: {}", op),
        }
    } else {
        match op {
            Operator::Add => plus(left, right),
            Operator::Sub => minus(left, right),
            Operator::Mul => mul(left, right),
            Operator::Div => div(left, right),
            Operator::Mod => mod_(left, right),
            Operator::Pow => pow(left, right),
            Operator::Atan2 => atan2(left, right),
            Operator::And | Operator::Or => left,
            Operator::Unless => f64::NAN, // nothing to do
            Operator::Default => default(left, right),
            Operator::If => if_(left, right),
            Operator::IfNot => if_not(left, right),
            _ => panic!("unexpected non-comparison op: {:?}", op),
        }
    };
}

pub fn string_compare(a: &str, b: &str, op: Operator) -> ParseResult<bool> {
    match op {
        Operator::Eql => Ok(a == b),
        Operator::NotEq => Ok(a != b),
        Operator::Lt => Ok(a < b),
        Operator::Gt => Ok(a > b),
        Operator::Lte => Ok(a <= b),
        Operator::Gte => Ok(a >= b),
        _ => Err(ParseError::General(format!(
            "unexpected operator {} in string comparison",
            op
        ))),
    }
}
