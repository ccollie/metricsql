use crate::BinaryOp;
use crate::BinaryOpKind::Logical;

// Eq returns true of left == right.
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
pub fn gt(left: f64, right: f64) -> bool {
    return left > right
}

// Lt returns true if left < right
pub fn lt(left: f64, right: f64) -> bool {
    return left < right
}

// Gte returns true if left >= right
pub fn gte(left: f64, right: f64) -> bool {
    return left >= right
}

// Lte returns true if left <= right
pub fn lte(left: f64, right: f64) -> bool {
    return left <= right
}

// Plus returns left + right
pub fn plus(left: f64, right: f64) -> f64 {
    return left + right
}

// Minus returns left - right
pub fn minus(left: f64, right: f64) -> f64 {
    return left - right
}

// Mul returns left * right
pub fn mul(left: f64, right: f64) -> f64 {
    return left * right
}

// Div returns left / right
pub fn div(left: f64, right: f64) -> f64 {
    return left / right
}

// Mod returns mod(left, right)
pub fn mod_(left: f64, right: f64) -> f64 {
    return math.Mod(left, right)
}

// Pow returns pow(left, right)
pub fn pow(left: f64, right: f64) -> f64 {
    return math.Pow(left, right)
}

// Atan2 returns atan2(left, right)
pub fn atan2(left: f64, right: f64) -> f64 {
    return left.atan2(right)
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
pub fn ifnot(left: f64, right: f64) -> f64 {
    if right.is_nan() {
        return left
    }
    return f64::NAN
}

pub fn eval_binary_op(left: f64, right: f64, op: BinaryOp, is_bool: bool) -> f64 {
    use BinaryOp::*;
    use BinaryOpKind::*;

    fn fixup_comparison(cf: fn(left: f64, right: f64) -> bool) -> f64 {
        let val = cf(left, right);
        if val == 1 {
            return 1 as f64
        }
        return 0 as f64;
    }

    match op {
        Add => plus(left, right),
        Sub => minus(left, right),
        Mul => mul(left, right),
        Div => div(left, right),
        Mod => mod_(left, right),
        Pow => pow(left, right),
        Atan2 => atan2(left, right),
        Eq => fixup_comparison(eq),
        Neq => fixup_comparison(neq),
        Gt => fixup_comparison(gt),
        Lt => fixup_comparison(lt),
        Gte =>fixup_comparison(gte),
        Lte =>fixup_comparison(lte),
        Default => default(left, right),
        If => if_(left, right),
        Ifnot => ifnot(left, right),
        _ => panic!("unexpected non-comparison op: {:?}", op),
    }
}

