use crate::types::BinaryOp;

// Eq returns true of left == right.
#[inline]
pub fn eq(left: f64, right: f64) -> bool {
    // Special handling for nan == nan.
    // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/150 .
    if left.is_nan() {
        return right.is_nan()
    }
    return left == right
}

// Neq returns true of left != right.
pub fn neq(left: f64, right: f64) -> bool {
    // Special handling for comparison with nan.
    // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/150 .
    if left.is_nan() {
        return !right.is_nan()
    }
    if right.is_nan() {
        return true
    }
    return left != right
}

// Gt returns true of left > right
#[inline]
pub fn gt(left: f64, right: f64) -> bool {
    return left > right
}

// Lt returns true if left < right
#[inline]
pub fn lt(left: f64, right: f64) -> bool {
    left < right
}

// Gte returns true if left >= right
#[inline]
pub fn gte(left: f64, right: f64) -> bool {
    left >= right
}

// Lte returns true if left <= right
#[inline]
pub fn lte(left: f64, right: f64) -> bool {
    left <= right
}

// Plus returns left + right
#[inline]
pub fn plus(left: f64, right: f64) -> f64 {
     left + right
}

// Minus returns left - right
#[inline]
pub fn minus(left: f64, right: f64) -> f64 {
     left - right
}

// Mul returns left * right
#[inline]
pub fn mul(left: f64, right: f64) -> f64 {
     left * right
}

// Div returns left / right
#[inline]
pub fn div(left: f64, right: f64) -> f64 {
     left / right
}

// Mod returns mod(left, right)
#[inline]
pub fn mod_(left: f64, right: f64) -> f64 {
    left % right
}

// Pow returns pow(left, right)
#[inline]
pub fn pow(left: f64, right: f64) -> f64 {
    left.powf(right)
}

// Atan2 returns atan2(left, right)
pub fn atan2(left: f64, right: f64) -> f64 {
    left.atan2(right)
}

// Default returns left or right if left is NaN.
pub fn default(left: f64, right: f64) -> f64 {
    if left.is_nan() {
        return right
    }
    return left
}

// If returns left if right is not NaN. Otherwise NaN is returned.
pub fn if_(left: f64, right: f64) -> f64 {
    if right.is_nan() {
        return f64::NAN
    }
    return left
}

// Ifnot returns left if right is NaN. Otherwise NaN is returned.
#[inline]
pub fn ifnot(left: f64, right: f64) -> f64 {
    if right.is_nan() {
        return left
    }
    f64::NAN
}

pub fn eval_binary_op(left: f64, right: f64, op: BinaryOp, is_bool: bool) -> f64 {
    use BinaryOp::*;

    fn fixup_comparison(left: f64, right: f64, cf: fn(left: f64, right: f64) -> bool) -> f64 {
        let val = cf(left, right);
        if val == 1 {
            return 1.0
        }
        return 0.0;
    }

    match op {
        Add => plus(left, right),
        Sub => minus(left, right),
        Mul => mul(left, right),
        Div => div(left, right),
        Mod => mod_(left, right),
        Pow => pow(left, right),
        Atan2 => atan2(left, right),
        Eql => fixup_comparison(left, right, eq),
        Neq => fixup_comparison(left, right,neq),
        Gt => fixup_comparison(left, right,gt),
        Lt => fixup_comparison(left, right,lt),
        Gte =>fixup_comparison(left, right,gte),
        Lte =>fixup_comparison(left, right,lte),
        Default => default(left, right),
        If => if_(left, right),
        IfNot => ifnot(left, right),
        _ => panic!("unexpected non-comparison op: {:?}", op),
    }
}

