use crate::ast::{BExpression, Expression, FuncExpr, ParensExpr, ExpressionNode, BinaryOp};
use crate::binaryop::{eval_binary_op, string_compare};
use crate::parser::{ParseResult};


pub(crate) fn simplify_expr(expr: &Expression) -> ParseResult<Expression> {
    let exp = remove_parens_expr(&expr);
    // if we have a parens expr, simplify it further
    let res = match exp {
        Expression::Parens(pe) => simplify_parens_expr(pe)?,
        _ => exp,
    };
    simplify_constants(&res)
}

/// removes parensExpr for (Expr) case.
pub(crate) fn remove_parens_expr(e: &Expression) -> Expression {
    fn remove_parens_args(args: &[BExpression]) -> Vec<BExpression> {
        return args
            .iter()
            .map(|x| Box::new(remove_parens_expr(x)))
            .collect();
    }

    match e {
        Expression::Rollup(re) => {
            let expr = remove_parens_expr(&*re.expr);
            let at: Option<BExpression> = match &re.at {
                Some(at) => {
                    let expr = remove_parens_expr(at);
                    Some(Box::new(expr))
                }
                None => None,
            };
            let mut res = re.clone();
            res.at = at;
            res.expr = Box::new(expr);
            Expression::Rollup(res)
        }
        Expression::BinaryOperator(be) => {
            let left = remove_parens_expr(&be.left);
            let right = remove_parens_expr(&be.right);
            let mut res = be.clone();
            res.left = Box::new(left);
            res.right = Box::new(right);
            Expression::BinaryOperator(res)
        }
        Expression::Aggregation(agg) => {
            let mut expr = agg.clone();
            expr.args = remove_parens_args(&agg.args);
            Expression::Aggregation(expr)
        }
        Expression::Function(f) => {
            let mut res = f.clone();
            res.args = remove_parens_args(&res.args);
            Expression::Function(res)
        }
        Expression::Parens(parens) => {
            let mut res = parens.clone();
            res.expressions = remove_parens_args(&res.expressions);
            Expression::Parens(res)
        }
        _ => e.clone(),
    }
}

#[inline]
fn should_remove_parens(expr: &Expression) -> bool {
    match expr {
        Expression::Rollup(re) => {
            if !should_remove_parens(&*re.expr) {
                return false
            }
            if let Some(at) = &re.at {
                return should_remove_parens(at.as_ref());
            };
            return false
        }
        Expression::BinaryOperator(be) => {
            return should_remove_parens(&be.left) ||
                should_remove_parens(&be.right)
        }
        Expression::Aggregation(agg) => {
            return agg.args.iter()
                .any(|x| should_remove_parens(x.as_ref()));
        }
        Expression::Function(f) => {
            return f.args.iter()
                .any(|x| should_remove_parens(x.as_ref()));
        }
        Expression::Parens(parens) => {
            return parens.expressions.iter()
                .any(|x| should_remove_parens(x.as_ref()));
        }
        _ => false
    }
}

pub(crate) fn simplify_parens_expr(expr: ParensExpr) -> ParseResult<Expression> {
    if expr.len() == 1 {
        let res = *expr.expressions[0].clone();
        return Ok(res);
    }
    // Treat parensExpr as a function with empty name, i.e. union()
    let span = expr.span.clone();
    let fe = FuncExpr::new("union", expr.expressions, span)?;
    Ok(Expression::Function(fe))
}

// todo: use a COW?
pub(crate) fn simplify_constants(expr: &Expression) -> ParseResult<Expression> {
    match expr {
        Expression::Rollup(re) => {
            let mut clone = re.clone();
            let expr = simplify_constants(&re.expr)?;
            clone.expr = Box::new(expr);
            match &re.at {
                Some(at) => {
                    let simplified = simplify_constants(at)?;
                    clone.at = Some(Box::new(simplified));
                }
                None => {}
            }
            Ok(Expression::Rollup(clone))
        }
        Expression::BinaryOperator(be) => {
            let left = simplify_constants(&*be.left)?;
            let right = simplify_constants(&*be.right)?;

            match (&left, &right) {
                (Expression::Number(ln), Expression::Number(rn)) => {
                    let n = eval_binary_op(ln.value, rn.value, be.op, be.bool_modifier);
                    // TODO: construct a span covering left and right
                    return Ok(Expression::from(n))
                }
                (Expression::String(left), Expression::String(right)) => {
                    if be.op == BinaryOp::Add {
                        let val = format!("{}{}", left.value, right.value);
                        return Ok(Expression::from(val))
                    }
                    if be.op.is_comparison() {
                        let n = if string_compare(&left.value, &right.value, be.op)? {
                            1.0
                        } else if !be.bool_modifier {
                            f64::NAN
                        } else {
                            0.0
                        };
                        return Ok(Expression::from(n));
                    }
                }
                _ => {},
            }

            let mut clone = be.clone();
            clone.left = Box::new(left);
            clone.right = Box::new(right);

            return Ok(Expression::BinaryOperator(clone));

        }
        Expression::Aggregation(agg) => {
            let mut res = agg.clone();
            res.args = simplify_args_constants(&res.args)?;
            Ok(Expression::Aggregation(res))
        }
        Expression::Function(fe) => {
            let mut res = fe.clone();
            res.args = simplify_args_constants(&res.args)?;
            Ok(Expression::Function(res))
        }
        Expression::Parens(parens) => {
            let args = simplify_args_constants(&parens.expressions)?;
            if parens.len() == 1 {
                let expr = args.into_iter().next();
                return Ok(*expr.unwrap());
            }
            // Treat parensExpr as a function with empty name, i.e. union()
            Ok(Expression::Function(FuncExpr::new("union", args, expr.span())?))
        }
        _ => Ok(expr.clone().cast()),
    }
}

pub(crate) fn simplify_args_constants(args: &[BExpression]) -> ParseResult<Vec<BExpression>> {
    let mut res: Vec<BExpression> = Vec::with_capacity(args.len());
    for arg in args {
        let simple = simplify_constants(arg)?;
        res.push(Box::new(simple));
    }

    Ok(res)
}
