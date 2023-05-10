use crate::common::Operator;
use crate::parser::{ParseError, ParseResult};

// see https://github.com/VictoriaMetrics/metricsql/blob/master/binary_op.go

pub type BinopFunc = fn(left: f64, right: f64) -> f64;

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

/// unless returns NaN if left is NaN. Otherwise right is returned.
pub fn unless(left: f64, right: f64) -> f64 {
    // todo: is this correct?
    if left.is_nan() {
        return f64::NAN;
    }
    right
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

fn return_left(left: f64, _right: f64) -> f64 {
    left
}

fn return_nan(_left: f64, _right: f64) -> f64 {
    f64::NAN
}

macro_rules! make_comparison_func {
    ($name: ident, $func: expr) => {
        pub fn $name(left: f64, right: f64) -> f64 {
            if $func(left, right) { left } else { f64::NAN }
        }
    };
}

macro_rules! make_comparison_func_bool {
    ($name: ident, $func: expr) => {
        pub fn $name(left: f64, right: f64) -> f64 {
            return if $func(left, right) { 1_f64 } else { 0_f64 }
        }
    };
}

make_comparison_func!(compare_eq, eq);
make_comparison_func!(compare_neq, neq);
make_comparison_func!(compare_gt, gt);
make_comparison_func!(compare_lt, lt);
make_comparison_func!(compare_gte, gte);
make_comparison_func!(compare_lte, lte);

make_comparison_func_bool!(compare_eq_bool, eq);
make_comparison_func_bool!(compare_neq_bool, neq);
make_comparison_func_bool!(compare_gt_bool, gt);
make_comparison_func_bool!(compare_lt_bool, lt);
make_comparison_func_bool!(compare_gte_bool, gte);
make_comparison_func_bool!(compare_lte_bool, lte);

fn get_comparison_handler(op: Operator, is_bool: bool) -> BinopFunc {
    if is_bool {
        match op {
            Operator::Eql => compare_eq_bool,
            Operator::NotEq => compare_neq_bool,
            Operator::Gt => compare_gt_bool,
            Operator::Lt => compare_lt_bool,
            Operator::Gte => compare_gte_bool,
            Operator::Lte => compare_lte_bool,
            _ => panic!("unexpected non-comparison op: {:?}", op),
        }
    } else {
        match op {
            Operator::Eql => compare_eq,
            Operator::NotEq => compare_neq,
            Operator::Gt => compare_gt,
            Operator::Lt => compare_lt,
            Operator::Gte => compare_gte,
            Operator::Lte => compare_lte,
            _ => panic!("unexpected non-comparison op: {:?}", op),
        }
    }
}

pub fn get_scalar_binop_handler(op: Operator, is_bool: bool) -> BinopFunc {
    if op.is_comparison() {
        return get_comparison_handler(op, is_bool);
    }

    match op {
        Operator::Add => plus,
        Operator::Atan2 => atan2,
        Operator::Default => default,
        Operator::Div => div,
        Operator::Mod => mod_,
        Operator::Mul => mul,
        Operator::Pow => pow,
        Operator::Sub => minus,
        Operator::If => if_,
        Operator::IfNot => if_not,
        Operator::Unless => return_nan,
        Operator::And | Operator::Or => return return_left,
        _ => panic!("unexpected non-comparison op: {:?}", op),
    }
}

pub fn eval_binary_op(left: f64, right: f64, op: Operator, is_bool: bool) -> f64 {
    let handler = get_scalar_binop_handler(op, is_bool);
    handler(left, right)
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
