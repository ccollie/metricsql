use std::ops::{Deref};

use crate::ast::*;
use crate::common::{AggregateModifier, GroupModifier, JoinModifier};
use crate::parser::{ArgCountError, dedupe_label_filters, ParseError, ParseResult};


pub fn expand_with_expr(was: &Vec<WithArgExpr>,
                        expr: &Expression) -> ParseResult<Expression> {
    use Expression::*;

    match expr {
        BinaryOperator(_) => expand_binary(expr, was),
        Function(_) => expand_function(expr, was),
        Aggregation(_) => expand_aggregation(expr, &was),
        With(with) => {
            let mut was_new: Vec<WithArgExpr> = Vec::with_capacity(was.len() + with.was.len());
            was_new.extend_from_slice(was);
            was_new.extend_from_slice(&*with.was);
            let res = expand_with_expr(&was_new, with.expr.as_ref())?;
            return Ok(res.clone());
        }
        _ => Ok(expr.clone())
    }
}

// https://stackoverflow.com/questions/67674841/is-there-a-macro-i-can-use-to-expect-a-variant-of-an-enum-and-extract-its-data
// todo: move elsewhere
macro_rules! expect_variant {
    ($e:expr, $p:path) => {
        match $e {
            $p(value) => value,
            _ => panic!("expected {}", stringify!($p)),
        }
    };
}

fn expand_binary(expr: &Expression,
                 was: &Vec<WithArgExpr>) -> ParseResult<Expression> {

    let be = expect_variant!(expr, Expression::BinaryOperator);

    let left = expand_with_expr(was, be.left.as_ref())?;
    let right = expand_with_expr(was, be.right.as_ref())?;

    let mut new_be = be.clone();
    new_be.left = BExpression::from(left);
    new_be.right = BExpression::from(right);

    if let Some(modifier) = &be.group_modifier {
        let labels = expand_modifier_args(was, &modifier.labels)?;
        if labels != modifier.labels {
            new_be.group_modifier = Some(GroupModifier::new(modifier.op, labels));
        }
    }

    if let Some(ref modifier) = &be.join_modifier {
        let labels = expand_modifier_args(was, &modifier.labels)?;
        if labels != modifier.labels {
            new_be.join_modifier = Some(JoinModifier::new(modifier.op, labels));
        }
    }

    Ok(Expression::BinaryOperator(new_be))
}

fn expand_function(expr: &Expression, was: &Vec<WithArgExpr>) -> ParseResult<Expression> {

    let func = expect_variant!(expr, Expression::Function);
    let arg_expr = get_with_arg_expr(was, &func.with_name);
    if let Some(wa) = arg_expr {
        let args = expand_with_args(was, &func.args)?;
        return expand_with_expr_ext(was, wa, Some(&args))
    }

    Ok(expr.clone())
}

fn expand_aggregation(expr: &Expression, was: &Vec<WithArgExpr>) -> ParseResult<Expression> {

    let aggregate = expect_variant!(expr, Expression::Aggregation);
    let args = expand_with_args(was, &aggregate.args)?;

    let wa = get_with_arg_expr(was, &aggregate.name);
    if let Some(with_arg_expression) = wa {
        return expand_with_expr_ext(was, with_arg_expression, Some(&args));
    }

    let mut aggr = aggregate.clone();

    if let Some(modifier) = &aggregate.modifier {
        let new_args =  expand_modifier_args(was, &modifier.args)?;
        if new_args != modifier.args {
            aggr.modifier = Some(AggregateModifier::new(modifier.op.clone(), new_args));
        }
    }

    Ok(Expression::Aggregation(aggr))
}

fn expand_metric_expr(expr: &Expression,
                      was: &Vec<WithArgExpr>) -> ParseResult<Expression> {

    let me = expect_variant!(expr, Expression::MetricExpression);
    let me = expand_metric_labels(&me, was)?;

    if !me.has_non_empty_metric_group() {
        let expr = Expression::MetricExpression(me);
        return Ok(expr);
    }

    let wa = {
        let k = &me.label_filters[0].value;  // name
        get_with_arg_expr(was, k.to_string().as_str())
    };

    if wa.is_none() {
        let expr = Expression::MetricExpression(me);
        return Ok(expr);
    }

    let wa = wa.unwrap();
    let e_new = expand_with_expr_ext(was, wa, None)?;

    let wme = match &e_new {
        Expression::MetricExpression(me) => Some(me),
        Expression::Rollup(e) => {
            match e.expr.as_ref() {
                Expression::MetricExpression(me) => Some(me),
                _ => None,
            }
        },
        _ => None,
    };

    if wme.is_none() {
        if !me.is_only_metric_group() {
            let msg = format!("cannot expand {:?} to non-metric expression {}", me, wa);
            return Err(ParseError::WithExprExpansionError(msg));
        }
        return Ok(e_new);
    }

    let wme = wme.unwrap();

    let mut label_filters = wme.label_filters.clone(); // do we need the original ??
    let other = me.label_filters[1..].to_vec();
    for label in other {
        label_filters.push(label.clone());
    }

    dedupe_label_filters(&mut label_filters);

    let result = Expression::MetricExpression( MetricExpr::with_filters(label_filters) );

    match e_new {
        Expression::Rollup(mut re) => {
            re.expr = Box::new(result);
            let expr = Expression::Rollup(re);
            Ok(expr)
        }
        _ => {
            Ok(result)
        }
    }
}

pub(crate) fn expand_metric_labels(me: &MetricExpr,
                        was: &Vec<WithArgExpr>) -> ParseResult<MetricExpr> {

    // Populate me.label_filters
    let mut me_new = MetricExpr::default();
    if me.label_filters.len() > 0 {
        me_new.label_filters.extend_from_slice(&me.label_filters);
    }

    for lfe in me.label_filters.iter() {
        if !lfe.is_resolved() {
            // Expand lfe.label into Vec<LabelFilter>.
            let wa = get_with_arg_expr(was, &lfe.label);
            if wa.is_none() {
                let msg = format!("missing {} MetricExpr", lfe.label);
                return Err(ParseError::WithExprExpansionError(msg));
            }

            let wa = wa.unwrap();
            let e_new = expand_with_expr_ext(was, wa, None)?;
            let error_msg = format!(
                "{} must be filters expression inside {}: got {}",
                lfe.label, me, e_new
            );

            let wae = expand_with_expr_ext(was, wa, None)?;
            match wae {
                Expression::MetricExpression(wme) => {
                    if me.is_only_metric_group() {
                        return Err(ParseError::WithExprExpansionError(error_msg));
                    }

                    for lfe in wme.label_filters.iter() {
                        me_new.label_filters.push(lfe.clone());
                    }
                    continue;
                }
                _ => {
                    return Err(ParseError::WithExprExpansionError(error_msg));
                }
            }
        }

        // convert lfe to LabelFilter.

        // todo(perf) - this seems wasteful to create a wrapped variant only to unpack again.
        // maybe extract logic from expand_string and use that
        // this whole block should be redone
        let str_expr = Expression::from(lfe.value.to_string());
        let se = expand_with_expr(was, &str_expr)?;
        let se_extract = expect_variant!(&se, Expression::String);

        let lf = LabelFilterExpr::new(lfe.op, &lfe.label, se_extract.clone())?;
        me_new.label_filters.push(lf);
    }

    me_new.label_filters.clear();
    dedupe_label_filters(&mut me_new.label_filters);

    return Ok(me_new);
}

fn expand_modifier_args(was: &Vec<WithArgExpr>, args: &[String]) -> ParseResult<Vec<String>> {
    if args.is_empty() {
        return Ok(vec![]);
    }

    let mut dst_args: Vec<String> = Vec::with_capacity(1);

    let handle_metric_expr = |me: &MetricExpr, arg: &str, dst_args: &mut Vec<String>| -> ParseResult<()> {
        if !me.is_only_metric_group() {
            let msg = format!(
                "cannot use {:?} instead of {:?} in {}",
                me,
                arg,
                args.join(",")
            );
            return Err(ParseError::WithExprExpansionError(msg));
        }
        let dst_arg = &me.label_filters[0].value;
        dst_args.push(dst_arg.clone());
        Ok(())
    };


    for arg in args.iter() {
        match get_with_arg_expr(was, arg) {
            None => {
                // Leave the arg as is.
                dst_args.push(arg.to_string());
                continue;
            }
            Some(wa) => {
                if !wa.args.is_empty() {
                    // Template functions cannot be used inside modifier list. Leave the arg as is.
                    dst_args.push(arg.to_string());
                    continue;
                }
                match &wa.expr.deref() {
                    Expression::MetricExpression(me) => {
                        handle_metric_expr(me, arg, &mut dst_args)?
                    }
                    Expression::Parens(pe) => {
                        for exp in pe.expressions.iter() {
                            match exp.deref() {
                                Expression::MetricExpression(me) => {
                                    handle_metric_expr(me, arg, &mut dst_args)?;
                                }
                                _ => {
                                    let msg = "expected metric selector as WITH argument".to_string();
                                    return Err(ParseError::WithExprExpansionError(msg));
                                }
                            }
                        }
                    }
                    _ => {
                        let msg = format!(
                            "cannot use {:?} instead of {:?} in {:?}",
                            wa.expr, arg, args
                        );
                        return Err(ParseError::WithExprExpansionError(msg));
                    }
                }
            }
        }
    }

    Ok(dst_args)
}

fn expand_with_expr_ext(was: &Vec<WithArgExpr>,
                        wa: &WithArgExpr,
                        args: Option<&Vec<BExpression>>) -> ParseResult<Expression> {

    let args_len = if let Some(expressions) = args {
        expressions.len()
    } else {
        0
    };

    if wa.args.len() != args_len {
        if args.is_none() {
            // Just return MetricExpr with the wa.name name.
            let me = Expression::MetricExpression(MetricExpr::new(&wa.name));
            return Ok(me);
        }
        let err = ParseError::InvalidArgCount(ArgCountError::new(&*wa.name, args_len, args_len));
        return Err(err);
    }

    let mut was_new: Vec<WithArgExpr> = Vec::with_capacity(was.len() + args_len);
    for wa_tmp in was.iter() {
        if wa_tmp.name == wa.name {
            break;
        }
        was_new.push(wa_tmp.clone());
    }

    if let Some(args) = args {
        for (i, arg) in args.iter().enumerate() {
            let wae = WithArgExpr {
                name: wa.args[i].clone(),
                args: vec![],
                expr: arg.clone(),
                is_function: false
            };
            was_new.push(wae);
        }
    }

    expand_with_expr(&was_new, &wa.expr)
}

fn expand_with_args(was: &Vec<WithArgExpr>, args: &Vec<BExpression>) -> ParseResult<Vec<BExpression>> {
    let mut result: Vec<BExpression> = Vec::with_capacity(args.len());

    for arg in args.iter() {
        let expanded_arg = expand_with_expr(was, arg.as_ref())?;
        result.push( Box::new(expanded_arg) );
    }

    Ok(result)
}

fn get_with_arg_expr<'a>(was: &'a Vec<WithArgExpr>, name: &str) -> Option<&'a WithArgExpr> {
    // Scan was backwards, since certain expressions may override
    // previously defined expressions
    for expr in was.iter().rev() {
        if expr.name == name {
            return Some(expr);
        }
    }
    None
}
