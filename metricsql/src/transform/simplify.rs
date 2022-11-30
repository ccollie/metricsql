use crate::ast::{BExpression, Expression, BinaryOp, FuncExpr};
use crate::binaryop::{eval_binary_op, string_compare};
use crate::prelude::NumberExpr;


pub(crate) fn simplify_expr(expr: Expression) -> Expression {
    let exp = remove_parens_expr(expr);
    simplify_constants(exp)
}

/// removes parensExpr for (Expr) case.
pub(crate) fn remove_parens_expr(e: Expression) -> Expression {
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
        Expression::Rollup(ref mut re) => {
            if let Some(expr) = remove_boxed(&re.expr) {
                re.expr = expr;
            }
            if let Some(at) = &re.at {
                if let Some(expr) = remove_boxed(&at) {
                    re.at = Some(expr);
                }
            }
        }
        Expression::BinaryOperator(ref mut be) => {
            if let Some(expr) = remove_boxed(&be.left) {
                be.left = expr;
            }
            if let Some(expr) = remove_boxed(&be.right) {
                be.right = expr;
            }
        }
        Expression::Aggregation(ref mut agg) => {
            remove_parens_args(&mut agg.args);
        }
        Expression::Function(ref mut f) => {
            remove_parens_args(&mut f.args);
        }
        Expression::Parens(ref mut parens) => {
            remove_parens_args(&mut parens.expressions);
            if parens.len() == 1 {
                expr = std::mem::take(parens.expressions[0].as_mut());
            } else {
                // Treat parensExpr as a function with empty name, i.e. union()
                // todo: how to avoid clone
                let fe = FuncExpr::new("", parens.expressions.clone(),
                                       expr.span())
                    .unwrap(); // todo: remove this
                expr = Expression::Function(fe)
            }
        }
        _ => {},
    }
    expr
}


fn should_remove_parens(e: &Expression) -> bool {
    match e {
        Expression::Rollup(re) => {
            if !should_remove_parens(re.expr.as_ref()) {
                return false;
            }
            if let Some(at) = &re.at {
                return should_remove_parens(at.as_ref())
            }
            return false
        }
        Expression::BinaryOperator(be) => {
            should_remove_parens(be.left.as_ref()) ||
            should_remove_parens(be.right.as_ref())
        }
        Expression::Aggregation(agg) => {
            agg.args.iter().any(|x| should_remove_parens(x.as_ref()))
        }
        Expression::Function(func) => {
            func.args.iter().any(|x| should_remove_parens(x.as_ref()))
        }
        Expression::Parens(parens) => {
            if parens.len() == 1 {
                return true;
            }
            parens.expressions.iter().any(|x| should_remove_parens(x.as_ref()))
        }
        _ => false
    }
}

pub(crate) fn can_simplify_constants(expr: &Expression) -> bool {
    match expr {
        Expression::Rollup(re) => {
            if !can_simplify_constants(re.expr.as_ref()) {
                return false
            }
            if let Some(at) = &re.at {
                return can_simplify_constants(at.as_ref());
            }
            return false;
        }
        Expression::BinaryOperator(be) => {
            can_simplify_constants(be.left.as_ref()) ||
            can_simplify_constants(be.right.as_ref())
        }
        Expression::Aggregation(agg) => {
            agg.args.iter().any(|x| can_simplify_constants(x.as_ref()))
        }
        Expression::Function(fe) => {
            fe.args.iter().any(|x| can_simplify_constants(x.as_ref()))
        }
        Expression::Parens(parens) => {
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

    fn simplify_boxed(expr: &BExpression) -> BExpression {
        // todo(perf) can we take possession of the contents of the box rather
        // than clone
        let cloned = expr.as_ref().clone();
        let simple = simplify_constants(cloned);
        return Box::new(simple);
    }

    let mut expr = expr;
    match expr {
        Expression::Rollup(ref mut re) => {
            if can_simplify_constants(&re.expr) {
                re.expr = simplify_boxed(&re.expr);
            }
            if let Some(at) = &re.at {
                if can_simplify_constants(at) {
                    re.at = Some( simplify_boxed(at) );
                }
            }
        }
        Expression::BinaryOperator(ref mut be) => {
            if can_simplify_constants(&be.left) {
                be.left = simplify_boxed(&be.left);
            }

            if can_simplify_constants(&be.right) {
                be.right = simplify_boxed(&be.right);
            }

            match (&be.left.as_ref(), &be.right.as_ref()) {
                (Expression::Number(ln), Expression::Number(rn)) => {
                    let n = eval_binary_op(ln.value, rn.value, be.op, be.bool_modifier);
                    return Expression::Number( NumberExpr::new(n, be.left.span()))
                }
                (Expression::String(left), Expression::String(right)) => {
                    if be.op == BinaryOp::Add {
                        let val = format!("{}{}", left.value, right.value);
                        return Expression::from(val)
                    }
                    if be.op.is_comparison() {
                        // Note:: the `or` branch should not be reached because of
                        // the comparison above
                        let n = if string_compare(&left.value, &right.value, be.op)
                            .unwrap_or(false) {
                            1.0
                        } else if !be.bool_modifier {
                            f64::NAN
                        } else {
                            0.0
                        };
                        return Expression::from(n)
                    }
                }
                _ => {},
            }
        }
        Expression::Aggregation(ref mut agg) => {
            simplify_args_in_place(&mut agg.args);
        }
        Expression::Function(ref mut fe) => {
            simplify_args_in_place(&mut fe.args);
        }
        Expression::Parens(ref mut parens) => {
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
    match expr {
        Expression::Rollup(ref mut re) => {
            simplify_constants_inplace(&mut re.expr);
            if let Some(at) = &re.at {
                // todo: how to avoid clone
                let clone = at.as_ref().clone();
                re.at = Some( Box::new( simplify_constants(clone)) );
            }
        }
        Expression::BinaryOperator(be) => {
            simplify_constants_inplace(&mut be.left);
            simplify_constants_inplace(&mut be.right);
        }
        Expression::Aggregation(agg) => {
            simplify_args_in_place(&mut agg.args);
        }
        Expression::Function(fe) => {
            simplify_args_in_place(&mut fe.args);
        }
        Expression::Parens(parens) => {
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