use crate::ast::{BExpression, Expression, BinaryOp, FuncExpr, BinaryOpExpr, StringExpr};
use crate::ast::BinaryOp::Add;
use crate::ast::StringTokenType::String;
use crate::binaryop::{eval_binary_op, string_compare};
use crate::prelude::NumberExpr;


pub(crate) fn simplify_expr(expr: Expression) -> Expression {
    let exp = remove_parens_expr(expr);
    simplify_constants(exp)
}

/// removes parensExpr for (Expr) case.
pub(crate) fn remove_parens_expr(e: Expression) -> Expression {
    use Expression::*;

    fn remove_parens_args(args: &mut Vec<BExpression>) {
        for i in 0 .. args.len() {
            let arg = &args[i];
            if let Some(removed) = remove_boxed(&arg) {
                args[i] = removed;
            }
        }
    }

    fn remove_boxed(expr: &BExpression) -> Option<BExpression> {
        let expr = expr.as_ref();
        if should_remove_parens(expr) {
            // cloned because we can't extract a value from a Box
            let cloned = expr.clone();
            let res  = Box::new(remove_parens_expr(cloned));
            return Some(res)
        }
        None
    }

    let mut expr = e;
    match expr {
        Rollup(ref mut re) => {
            if let Some(expr) = remove_boxed(&re.expr) {
                re.expr = expr;
            }
            if let Some(at) = &re.at {
                if let Some(expr) = remove_boxed(&at) {
                    re.at = Some(expr);
                }
            }
        }
        BinaryOperator(ref mut be) => {
            if let Some(expr) = remove_boxed(&be.left) {
                be.left = expr;
            }
            if let Some(expr) = remove_boxed(&be.right) {
                be.right = expr;
            }
        }
        Aggregation(ref mut agg) => {
            remove_parens_args(&mut agg.args);
        }
        Function(ref mut f) => {
            remove_parens_args(&mut f.args);
        }
        Parens(ref mut parens) => {
            remove_parens_args(&mut parens.expressions);
            if parens.len() == 1 {
                expr = std::mem::take(parens.expressions[0].as_mut());
            } else {
                // Treat parensExpr as a function with empty name, i.e. union()
                // todo: how to avoid clone
                let fe = FuncExpr::new("", parens.expressions.clone(),
                                       expr.span())
                    .unwrap(); // todo: remove this
                expr = Function(fe)
            }
        }
        _ => {},
    }
    expr
}


fn should_remove_parens(e: &Expression) -> bool {
    use Expression::*;

    match e {
        Rollup(re) => {
            if !should_remove_parens(re.expr.as_ref()) {
                return false;
            }
            if let Some(at) = &re.at {
                return should_remove_parens(at.as_ref())
            }
            return false
        }
        BinaryOperator(be) => {
            should_remove_parens(be.left.as_ref()) ||
            should_remove_parens(be.right.as_ref())
        }
        Aggregation(agg) => {
            agg.args.iter().any(|x| should_remove_parens(x.as_ref()))
        }
        Function(func) => {
            func.args.iter().any(|x| should_remove_parens(x.as_ref()))
        }
        Parens(parens) => {
            if parens.len() == 1 {
                return true;
            }
            parens.expressions.iter().any(|x| should_remove_parens(x.as_ref()))
        }
        _ => false
    }
}

pub(crate) fn can_simplify_constants(expr: &Expression) -> bool {
    use Expression::*;

    match expr {
        Rollup(re) => {
            if !can_simplify_constants(re.expr.as_ref()) {
                return false
            }
            if let Some(at) = &re.at {
                return can_simplify_constants(at.as_ref());
            }
            return false;
        }
        BinaryOperator(be) => {
            let left = be.left.as_ref();
            let right = be.right.as_ref();
            match (left, right) {
                (Number(_), Number(_)) => true,
                (String(_), String(_)) => {
                    be.op == Add || be.op.is_comparison()
                },
                _ => {
                    can_simplify_constants(left) ||
                    can_simplify_constants(right)
                }
            }
        }
        Aggregation(agg) => {
            agg.args.iter().any(|x| can_simplify_constants(x.as_ref()))
        }
        Function(fe) => {
            fe.args.iter().any(|x| can_simplify_constants(x.as_ref()))
        }
        Parens(parens) => {
            if parens.len() == 1 {
                return true
            }
            parens.expressions.iter().any(|x| can_simplify_constants(x.as_ref()))
        }
        _ => true,
    }
}

// todo: use a COW?
pub(crate) fn simplify_constants(expr: Expression) -> Expression {
    use Expression::*;

    fn simplify_boxed(expr: &BExpression) -> BExpression {
        // todo(perf) can we take possession of the contents of the box rather than clone
        let cloned = expr.as_ref().clone();
        let simple = simplify_constants(cloned);
        return Box::new(simple);
    }

    let mut expr = expr;
    match expr {
        Rollup(ref mut re) => {
            if can_simplify_constants(&re.expr) {
                re.expr = simplify_boxed(&re.expr);
            }
            if let Some(at) = &re.at {
                if can_simplify_constants(at) {
                    re.at = Some( simplify_boxed(at) );
                }
            }
        }
        BinaryOperator(ref mut be) => {
            if can_simplify_constants(&be.left) {
                be.left = simplify_boxed(&be.left);
            }

            if can_simplify_constants(&be.right) {
                be.right = simplify_boxed(&be.right);
            }

            if let Some( folded ) = constant_fold_binary(be) {
                return folded
            }
        }
        Aggregation(ref mut agg) => {
            simplify_args_in_place(&mut agg.args);
        }
        Function(ref mut fe) => {
            simplify_args_in_place(&mut fe.args);
        }
        Parens(ref mut parens) => {
            if parens.len() == 1 {
                let single = parens.expressions.remove(0).as_ref().clone();
                // todo: how to avoid clone ?
                return simplify_constants(single.clone());
            }
            simplify_args_in_place(&mut parens.expressions);
        }
        _ => {},
    }
    expr
}


pub(crate) fn simplify_constants_inplace(expr: &mut Expression) {
    use Expression::*;

    match expr {
        Rollup(ref mut re) => {
            simplify_constants_inplace(&mut re.expr);
            if let Some(at) = &re.at {
                // todo: how to avoid clone
                let clone = at.as_ref().clone();
                re.at = Some( Box::new( simplify_constants(clone)) );
            }
        }
        BinaryOperator(be) => {
            if let Some(constant) = constant_fold_binary(be) {
                *expr = constant
            }
        }
        Aggregation(agg) => {
            simplify_args_in_place(&mut agg.args);
        }
        Function(fe) => {
            simplify_args_in_place(&mut fe.args);
        }
        Parens(parens) => {
            simplify_args_in_place(&mut parens.expressions);
        }
        _ => {},
    }
}

#[inline]
fn simplify_args_in_place(args: &mut Vec<BExpression>) {
    for arg in args {
        simplify_constants_inplace(arg.as_mut());
    }
}
/// Perform constant-folding on binary op if possible
pub(super) fn constant_fold_binary(be: &BinaryOpExpr) -> Option<Expression> {

    match (be.left.as_ref(), be.right.as_ref()) {
        (Expression::Number(ln), Expression::Number(rn)) => {
            let n = eval_binary_op(ln.value, rn.value, be.op, be.bool_modifier);
            let expr = Expression::Number( NumberExpr::new(n, be.left.span()));
            Some(expr)
        }
        (Expression::String(left), Expression::String(right)) => {
            if be.op == BinaryOp::Add {
                let val = format!("{}{}", left.value, right.value);
                let expr = Expression::String( StringExpr::new(val, be.left.span()));
                return Some(expr)
            }
            if be.op.is_comparison() {
                let n = if string_compare(&left.value, &right.value, be.op)
                    .unwrap_or(false) {
                    1.0
                } else if !be.bool_modifier {
                    f64::NAN
                } else {
                    0.0
                };
                let expr = Expression::Number( NumberExpr::new(n, be.left.span()));
                return Some(expr)
            }
            None
        }
        _ => None,
    }
}

// todo: tests