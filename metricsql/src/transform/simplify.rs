use crate::ast::{Expression, Operator, FuncExpr, BinaryExpr, StringExpr};
use crate::ast::Operator::Add;
use crate::ast::expr_rewriter::rewrite_expr;
use crate::binaryop::{eval_binary_op, string_compare};
use crate::prelude::{NumberExpr, ParseResult};

pub fn simplify_expr(expr: Expression) -> ParseResult<Expression> {
    rewrite_expr(expr, |exp| {
        match &exp { 
            Expression::BinaryOperator(be) => {
                if let Some( folded ) = constant_fold_binary_expression(&be) {
                    return Ok(folded)
                }
                Ok(exp)
            }
            Expression::Parens(pe) => {
                if pe.len() == 1 {
                    Ok(std::mem::take(pe.expressions[0].as_mut()))
                } else {
                    // Treat parensExpr as a function with empty name, i.e. union()
                    // todo: how to avoid clone
                    let fe = FuncExpr::new("", pe.expressions.clone(), expr.span())?;
                    Ok( Expression::Function(fe) )
                }
            }
            _ => Ok(exp)
        }
    })
}

pub fn simplify_constants(expr: Expression) -> ParseResult<Expression> {
    rewrite_expr(expr, |exp| {
        match &exp {
            Expression::BinaryOperator(be) => {
                if let Some( folded ) = constant_fold_binary_expression(&be) {
                    return Ok(folded)
                } else {
                    Ok(exp)
                }
            }
            _ => Ok(exp)
        }
    })
}

/// Perform constant-folding on binary op if possible
pub fn constant_fold_binary_expression(be: &BinaryExpr) -> Option<Expression> {
    constant_fold_internal(be.left.as_ref(), be.right.as_ref(), be.op, be.bool_modifier)
}

pub(crate) fn constant_fold_internal(left: &Expression,
                                     right: &Expression,
                                     op: Operator,
                                     bool_modifier: bool) -> Option<Expression> {

    match (left, right) {
        (Expression::Number(ln), Expression::Number(rn)) => {
            let n = eval_binary_op(ln.value, rn.value, op, bool_modifier);
            let expr = Expression::Number( NumberExpr::new(n, left.span()));
            Some(expr)
        }
        (Expression::String(left), Expression::String(right)) => {
            if op == Add {
                let val = format!("{}{}", left.value, right.value);
                let expr = Expression::String( StringExpr::new(val, left.span));
                return Some(expr)
            }
            if op.is_comparison() {
                let n = if string_compare(&left.value, &right.value, op)
                    .unwrap_or(false) {
                    1.0
                } else if !bool_modifier {
                    f64::NAN
                } else {
                    0.0
                };
                let expr = Expression::Number( NumberExpr::new(n, left.span));
                return Some(expr)
            }
            None
        }
        _ => None,
    }
}

// todo: tests