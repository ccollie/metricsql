use std::borrow::{BorrowMut, Cow};
use std::collections::HashSet;
use std::ops::{Deref};

use crate::ast::*;
use crate::binaryop::{eval_binary_op, string_compare};
use crate::lexer::{TextSpan};
use crate::parser::{ArgCountError, ParseError, ParseResult};

pub(crate) fn expand_with_expr<'a>(was: &Vec<WithArgExpr>,
                                   expr: &'a Expression) -> ParseResult<Cow<'a, Expression>> {
    let mut expr = expr;
    expand_with_expr_internal(was, &mut expr)

}


pub(super) fn expand_with_expr_internal<'a>(was: &Vec<WithArgExpr>,
                                            expr: &'a mut Expression) -> ParseResult<&'a mut Expression> {
    use Expression::*;

    match expr {
        BinaryOperator(_) => {
            if let Some(mut exp) = expand_binary(expr, was)? {
                return Ok(&mut exp);
            }
            return Ok(expr);
        },
        Function(_) => expand_function(expr, was),
        Aggregation(_) => expand_aggregation(expr, was),
        MetricExpression(ref mut metric) => expand_metric_expr(expr, metric, was),
        Rollup(rollup) => {
            expand_rollup(rollup, was)?;
            Ok(expr)
        },
        Parens(parens) => expand_parens_expr(expr, parens, was),
        String(str) => expand_string(expr, str, was),
        With(with) => {
            let mut was_new: Vec<WithArgExpr> = Vec::with_capacity(was.len() + with.was.len());
            was_new.extend_from_slice(was);
            was_new.extend(with.was.clone());
            return expand_with_expr_internal(&was_new, &mut with.expr)
        }
        _ => Ok(expr)
    }
}

fn expand_binary(expr: &mut Expression,
                 was: &Vec<WithArgExpr>) -> ParseResult<Option<Expression>> {

    let mut be = match expr {
        Expression::BinaryOperator(be) => be,
        _ => unreachable!()
    };

    let left = expand_with_expr_internal(was, be.left.as_mut())?;
    let right = expand_with_expr_internal(was, be.right.as_mut())?;

    match (left, right) {
        (Expression::Number(ln), Expression::Number(rn)) => {
            let n = eval_binary_op(ln.value, rn.value, be.op, be.bool_modifier);
            let expr = Expression::Number( NumberExpr::new(n, be.left.span()));
            return Ok(Some(expr))
        }
        (Expression::String(left), Expression::String(right)) => {
            if be.op == BinaryOp::Add {
                let val = format!("{}{}", left.value, right.value);
                let expr = Expression::from(val);
                return Ok(Some(expr))
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
                let expr = Expression::from(n);
                return Ok(Some(expr))
            }
        }
        _ => {},
    }


    if let Some(modifier) = &be.group_modifier {
        let labels = expand_modifier_args(was, &modifier.labels)?;
        be.group_modifier = Some( GroupModifier::new(modifier.op, labels) );
    }

    if let Some(ref modifier) = &be.join_modifier {
        let labels = expand_modifier_args(was, &modifier.labels)?;
        be.join_modifier = Some( JoinModifier::new(modifier.op, labels) );
    }

    Ok(None)
}

fn expand_function<'a>(expr: &'a mut Expression,
                       was: &Vec<WithArgExpr>) -> ParseResult<&'a mut Expression> {

    let mut func = match expr {
        Expression::Function(fe) => fe,
        _ => unreachable!()
    };

    expand_with_args(was, &mut func.args)?;

    match get_with_arg_expr(was, &func.with_name) {
        Some(wa) => {
            return expand_with_expr_ext(expr, was, wa, Some(&func.args))
        }
        None => Ok(expr)
    }
}

fn expand_aggregation<'a>(expr: &'a mut Expression,
                          was: &Vec<WithArgExpr>) -> ParseResult<&'a mut Expression> {

    let mut aggregate = match expr {
        Expression::Aggregation(fe) => fe,
        _ => unreachable!()
    };

    expand_with_args(was, &mut aggregate.args)?;
    let wa = get_with_arg_expr(was, &aggregate.name);
    if let Some(wae) = wa {
        return expand_with_expr_ext(expr, was, wae, Some(&aggregate.args));
    }
    let empty_vec = vec![];

    let mod_args = match &aggregate.modifier {
        Some(modifier) => &modifier.args,
        None => &empty_vec,
    };

    match expand_modifier_args(was, mod_args) {
        Err(e) => Err(e),
        Ok(modifier_args) => {
            if let Some(ref mut modifier) = aggregate.modifier {
                modifier.args = modifier_args;
            }
            Ok(expr)
        }
    }
}

fn expand_parens_expr<'a>(expr: &'a mut Expression,
                          parens: &mut ParensExpr,
                          was: &Vec<WithArgExpr>) -> ParseResult<&'a mut Expression> {
    let mut span = was[0].expr.span();
    for e in parens.expressions.iter_mut() {
        let expr = expand_with_expr_internal(was, e)?;
        *e = Box::new(expr);
        span = span.cover(e.span());
    }

    Ok(expr)
}

fn expand_string<'a>(expr: &'a mut Expression,
                     e: &mut StringExpr,
                     was: &Vec<WithArgExpr>) -> ParseResult<Cow<'a, Expression>> {
    if e.is_expanded() {
        // Already expanded. Copying should be cheap
        return Ok(Cow::Borrowed(expr));
    }
    let start = e.span.start;

    // todo(perf): preallocate string capacity
    let mut b: String = String::new();

    for token in e.tokens.iter() {
        let ident: &str;

        match token {
            StringTokenType::String(v) => {
                b.push_str(&v);
                continue;
            }
            StringTokenType::Ident(v) => ident = v.as_str()
        };

        let wa = get_with_arg_expr(was, ident);
        if wa.is_none() {
            let msg = format!("missing {} value inside StringExpr", ident);
            return Err(ParseError::WithExprExpansionError(msg));
        }
        let wa = wa.unwrap();
        let e_new = match expand_with_expr_ext(expr, was, wa, None)? {
            Expression::String(e) => e,
            _ => {
                let msg = format!("{} is not a string expression", ident);
                return Err(ParseError::WithExprExpansionError(msg));
            }
        };
        if e_new.has_tokens() {
            let msg = format!(
                "BUG: string.tokens must be empty; got {}",
                e_new.token_count()
            );
            return Err(ParseError::WithExprExpansionError(msg));
        }

        b.push_str(e_new.value.as_str());
    }

    e.value = b;
    e.span = TextSpan::at(start, b.len());

    return Ok(Cow::Borrowed(expr));
}

fn expand_rollup(rollup: &mut RollupExpr, was: &Vec<WithArgExpr>) -> ParseResult<()> {

    let expanded = expand_with_expr_internal(was, &mut rollup.expr.as_mut())?;
    rollup.expr = Box::new(expanded);

    if let Some(ref mut at) = rollup.at {
        let val = expand_with_expr_internal(was, at.as_mut())?;
        rollup.at = Some(Box::new(*val))

    }

    Ok(())
}

fn expand_metric_expr<'a>(expr: &'a mut Expression,
                          me: &'a mut MetricExpr,
                          was: &Vec<WithArgExpr>) -> ParseResult<&'a mut Expression> {
    if me.is_expanded() {
        // Already expanded.
        return Ok(expr);
    }
    expand_metric_labels(expr, me, was)?;
    if !me.has_non_empty_metric_group() {
        return Ok(expr);
    }
    let k = &me.label_filters[0].value;
    let wa = get_with_arg_expr(was, k);
    if wa.is_none() {
        return Ok(expr);
    }

    let wa = wa.unwrap();
    let e_new = expand_with_expr_ext(expr, was, wa, None)?;
    let re: Option<&mut RollupExpr> = None;

    let wme = match e_new {
        Expression::MetricExpression(ref mut me) => Some(me),
        Expression::Rollup(ref mut e) => match *e.expr {
            Expression::MetricExpression(ref mut me) => Some(me),
            _ => None,
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
    if !wme.label_filter_exprs.is_empty() {
        let msg = format!(
            "BUG: wme.label_filters must be empty; got {:?}",
            wme.label_filter_exprs
        );
        return Err(ParseError::WithExprExpansionError(msg));
    }

    let mut label_filters = wme.label_filters.clone(); // do we need the original ??
    let other = me.label_filters[1..].to_vec();
    for label in other {
        label_filters.push(label.clone());
    }

    remove_duplicate_label_filters(&mut label_filters);

    let new_me = MetricExpr::with_filters(label_filters);

    match re {
        None => Ok(expr),
        Some(rollup) => {
            rollup.expr = Box::new(Expression::MetricExpression(new_me));
            Ok(rollup.borrow_mut())
        }
    }
}

fn expand_metric_labels(expr: &mut Expression, me: &mut MetricExpr, was: &Vec<WithArgExpr>) -> ParseResult<()> {

    if me.label_filters.len() > 0 {
        // already expanded
        return Ok(());
    }

    // Populate me.label_filters
    for lfe in me.label_filter_exprs.iter_mut() {
        if !lfe.is_init() {
            // Expand lfe.label into Vec<LabelFilter>.
            let wa = get_with_arg_expr(was, &lfe.label);
            if wa.is_none() {
                let msg = format!("missing {} MetricExpr", lfe.label);
                return Err(ParseError::WithExprExpansionError(msg));
            }

            let wa = wa.unwrap();
            let e_new = expand_with_expr_ext(expr, was, wa, None)?;
            let error_msg = format!(
                "{} must be filters expression inside {}: got {}",
                lfe.label, me, e_new
            );

            match expand_with_expr_ext(expr, was, wa, None)? {
                Expression::MetricExpression(wme) => {
                    if me.is_only_metric_group() {
                        return Err(ParseError::WithExprExpansionError(error_msg));
                    }
                    if !me.label_filter_exprs.is_empty() {
                        let msg = format!(
                            "BUG: wme.label_filter_exprs must be empty; got {:?}",
                            me.label_filter_exprs
                        );
                        return Err(ParseError::WithExprExpansionError(msg));
                    }
                    for lfe in wme.label_filters.into_iter() {
                        me.label_filters.push(lfe);
                    }
                    continue;
                }
                _ => {
                    return Err(ParseError::WithExprExpansionError(error_msg));
                }
            }
        }

        // convert lfe to LabelFilter.
        let se = expand_string(expr, &mut lfe.value, was)?;
        let lfe_new = LabelFilterExpr::new_tag(
            lfe.label.to_string(),
            lfe.op,
            se.to_string(), TextSpan::default());
        let lf = lfe_new.to_label_filter();
        me.label_filters.push(lf);
    }

    me.label_filter_exprs.clear();
    remove_duplicate_label_filters(&mut me.label_filters);

    return Ok(());
}

fn expand_modifier_args(was: &Vec<WithArgExpr>, args: &[String]) -> Result<Vec<String>, ParseError> {
    if args.is_empty() {
        return Ok(vec![]);
    }

    let mut dst_args: Vec<String> = Vec::with_capacity(1);
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
                    }
                    Expression::Parens(pe) => {
                        for arg in pe.expressions.iter() {
                            match arg.deref() {
                                Expression::MetricExpression(me) => {
                                    if !me.is_only_metric_group() {
                                        let msg = format!(
                                            "cannot use {:?} instead of {:?} in {:?}",
                                            pe, me, &pe.expressions
                                        );
                                        return Err(ParseError::WithExprExpansionError(msg));
                                    }
                                    let dst_arg = me.label_filters[0].value.clone();
                                    dst_args.push(dst_arg);
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

fn expand_with_expr_ext<'a>(expr: &'a mut Expression,
                            was: &Vec<WithArgExpr>,
                            wa: &'a WithArgExpr,
                            args: Option<&Vec<BExpression>>) -> ParseResult<&'a mut Expression> {

    let args_len = if let Some(expressions) = args {
        expressions.len()
    } else {
        0
    };

    if wa.args.len() != args_len {
        if args.is_none() {
            // Just return MetricExpr with the wa.name name.
            let mut me = Expression::MetricExpression(MetricExpr::new(&wa.name)).borrow_mut();
            return Ok(me);
        }
        let err = ParseError::InvalidArgCount(ArgCountError::new(&*wa.name, args_len, args_len));
        return Err(err);
    }

    let mut was_new: Vec<WithArgExpr> = Vec::with_capacity(was.len() + args_len);
    for wa_tmp in was.iter() {
        // let a: *const &WithArgExpr = &wa_tmp;
        // let b: *const &WithArgExpr = &wa;
        // if a == b {
        //     break;
        // }
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
            };
            was_new.push(wae);
        }
    }

    expand_with_expr_internal(&was_new, expr)
}

fn expand_with_args(was: &Vec<WithArgExpr>, args: &mut Vec<BExpression>) -> ParseResult<()> {
    for arg in args.iter_mut() {
        let expanded = expand_with_expr_internal(was, arg)?;
        *arg = Box::new(expanded);
    }
    Ok(())
}

fn get_with_arg_expr<'a>(was: &Vec<WithArgExpr>, name: &'a str) -> Option<&'a WithArgExpr> {
    // Scan wes backwards, since certain expressions may override
    // previously defined expressions
    for i in was.iter().rev() {
        if i.name == name {
            return Some(i);
        }
    }
    None
}

fn remove_duplicate_label_filters(lfs: &mut Vec<LabelFilter>) {
    let mut set: HashSet<String> = HashSet::with_capacity(lfs.len());
    lfs.retain(|lf| {
        let key = lf.to_string();
        if set.contains(&key) {
            return false;
        }
        set.insert(key);
        true
    })
}
