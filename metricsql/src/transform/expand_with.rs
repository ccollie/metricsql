use std::borrow::{Cow};
use std::collections::HashSet;
use std::ops::{Deref};

use crate::ast::*;
use crate::binaryop::{eval_binary_op, string_compare};
use crate::lexer::{TextSpan};
use crate::parser::{ArgCountError, ParseError, ParseResult};

pub(crate) fn expand_with_expr(was: &Vec<WithArgExpr>,
                               expr: Expression) -> ParseResult<Expression> {
    match expand_with_expr_internal(was, &expr)? {
        Cow::Owned(val) => Ok(val),
        Cow::Borrowed(val) => Ok(val.clone())   // todo: how to avoid clone ?
    }
}


pub(super) fn expand_with_expr_internal<'a>(was: &Vec<WithArgExpr>,
                                            expr: &'a Expression) -> ParseResult<Cow<'a, Expression>> {
    use Expression::*;

    match expr {
        BinaryOperator(_) => expand_binary(expr, was),
        Function(_) => expand_function(&expr, &was),
        Aggregation(_) => expand_aggregation(&expr, was),
        MetricExpression(_) => expand_metric_expr(&expr, was),
        Rollup(_) => expand_rollup(expr, was),
        Parens(_) => expand_parens_expr(expr, was),
        String(_) => expand_string(expr, was),
        With(with) => {
            let mut was_new: Vec<WithArgExpr> = Vec::with_capacity(was.len() + with.was.len());
            was_new.extend_from_slice(was);
            was_new.extend(with.was.clone());
            return expand_with_expr_internal(&was_new, with.expr.as_ref())
        }
        _ => Ok(Cow::Borrowed(expr))
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

fn expand_binary<'a>(expr: &'a Expression,
                 was: &Vec<WithArgExpr>) -> ParseResult<Cow<'a, Expression>> {

    let be = expect_variant!(expr, Expression::BinaryOperator);

    let left = expand_with_expr_internal(was, be.left.as_ref())?;
    let right = expand_with_expr_internal(was, be.right.as_ref())?;

    match (left.as_ref(), right.as_ref()) {
        (Expression::Number(ln), Expression::Number(rn)) => {
            let n = eval_binary_op(ln.value, rn.value, be.op, be.bool_modifier);
            let expr = Expression::Number( NumberExpr::new(n, be.left.span()));
            return Ok(Cow::Owned(expr))
        }
        (Expression::String(left), Expression::String(right)) => {
            if be.op == BinaryOp::Add {
                let val = format!("{}{}", left.value, right.value);
                let expr = Expression::from(val);
                return Ok(Cow::Owned(expr))
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
                return Ok(Cow::Owned(expr))
            }
        }
        _ => {},
    }

    // todo(perf) determine if we need to clone
    let mut changed = false;
    let mut group_modifier: Option<GroupModifier> = None;
    let mut join_modifier: Option<JoinModifier> = None;

    if let Some(modifier) = &be.group_modifier {
        let labels = expand_modifier_args(was, &modifier.labels)?;
        if labels != modifier.labels {
            changed = true;
            group_modifier = Some(GroupModifier::new(modifier.op, labels));
        }
    }

    if let Some(ref modifier) = &be.join_modifier {
        let labels = expand_modifier_args(was, &modifier.labels)?;
        if labels != modifier.labels {
            changed = true;
            join_modifier = Some(JoinModifier::new(modifier.op, labels));
        }
    }

    if changed {
        let mut new_be = be.clone();

        new_be.join_modifier = join_modifier;
        new_be.group_modifier = group_modifier;

        return Ok(Cow::Owned(Expression::BinaryOperator(new_be)))
    }

    Ok(Cow::Borrowed(expr))
}

fn expand_function<'a>(expr: &'a Expression,
                       was: &Vec<WithArgExpr>) -> ParseResult<Cow<'a, Expression>> {

    let func = expect_variant!(expr, Expression::Function);
    let arg_expr = get_with_arg_expr(was, &func.with_name);
    if let Some(wa) = arg_expr {
        let args = expand_with_args(was, &func.args)?;
        return expand_with_expr_ext(expr, was, wa, Some(&args))
    }

    Ok(Cow::Borrowed(expr))
}

fn expand_aggregation<'a>(expr: &'a Expression, was: &Vec<WithArgExpr>) -> ParseResult<Cow<'a, Expression>> {

    let aggregate = expect_variant!(expr, Expression::Aggregation);

    let wa = get_with_arg_expr(was, &aggregate.name);
    if let Some(wae) = wa {
        let args = expand_with_args(was, &aggregate.args)?;
        return expand_with_expr_ext(expr, was, wae, Some(&args));
    }

    let mut new_modifier: Option<AggregateModifier> = None;
    if let Some(modifier) = &aggregate.modifier {
        let new_args =  expand_modifier_args(was, &modifier.args)?;
        if new_args != modifier.args {
            new_modifier = Some(AggregateModifier::new(modifier.op.clone(), new_args));
        }
    }

    if new_modifier.is_some() {
        let mut aggr = aggregate.clone();
        aggr.modifier = new_modifier;
        return Ok(Cow::Owned(Expression::Aggregation(aggr)));
    }

    Ok(Cow::Borrowed(expr))
}

fn expand_parens_expr<'a>(expr: &'a Expression,
                          was: &Vec<WithArgExpr>) -> ParseResult<Cow<'a, Expression>> {
    let parens = expect_variant!(expr, Expression::Parens);
    let mut span = was[0].expr.span();
    let mut exprs : Vec<BExpression> = Vec::with_capacity(parens.expressions.len());
    for e in parens.expressions.iter() {
        let expr = expand_with_expr_internal(was, e)?;
        match expr {
            Cow::Borrowed(borrowed) => {
                let val = borrowed.clone();
                exprs.push(Box::new(val));
            }
            Cow::Owned(val) => {
                exprs.push(Box::new(val));
            }
        }
        span = span.cover(e.span());
    }

    let new_parens = ParensExpr::new(exprs, span);

    Ok(Cow::Owned(Expression::Parens(new_parens)))
}

fn expand_string<'a>(expr: &'a Expression,
                     was: &Vec<WithArgExpr>) -> ParseResult<Cow<'a, Expression>> {

    let e = expect_variant!(expr, Expression::String);

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
        let e_new = expand_with_expr_ext(expr, was, wa, None)?;

        let e_new = match e_new.as_ref() {
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

    let len = b.len();
    let res = Expression::String(StringExpr::new(b, TextSpan::at(start, len)));

    return Ok(Cow::Owned(res));
}

fn expand_rollup<'a>(expr: &'a Expression, was: &Vec<WithArgExpr>) -> ParseResult<Cow<'a, Expression>> {

    let rollup = expect_variant!(expr, Expression::Rollup);

    let mut new_expr: Option<BExpression> = None;
    let mut new_at: Option<BExpression> = None;
    let mut changed: bool = false;

    // todo: use Box::into_inner instead of clone ?
    match expand_with_expr_internal(was, rollup.expr.as_ref())? {
        Cow::Owned(expanded) => {
            changed = true;
            new_expr = Some(Box::new(expanded));
        }
        _ => {}
    }

    if let Some(at) = &rollup.at {
        match expand_with_expr_internal(was, at.as_ref())? {
            Cow::Owned(expanded) => {
                changed = true;
                new_at = Some(Box::new(expanded)) // todo: use .into ?
            }
            _ => {}
        }
    }

    if changed {
        let mut res = rollup.clone();

        if new_at.is_some() {
            res.at = new_at;
        }

        if let Some(_expr) = new_expr {
            res.expr = _expr;
        }

        return Ok(Cow::Owned( Expression::Rollup(res)))
    }

    Ok(Cow::Borrowed(expr))
}

fn expand_metric_expr<'a>(expr: &'a Expression,
                      was: &Vec<WithArgExpr>) -> ParseResult<Cow<'a, Expression>> {

    let me = expect_variant!(expr, Expression::MetricExpression);
    let me = expand_metric_labels(expr, &me, was)?;

    if !me.has_non_empty_metric_group() {
        return Ok(Cow::Borrowed(expr));
    }

    let wa = {
        let k = &me.label_filters[0].value;
        get_with_arg_expr(was, k)
    };

    if wa.is_none() {
        return match me {
            Cow::Owned(owned) => {
                let expr = Expression::MetricExpression(owned);
                Ok(Cow::Owned(expr))
            }
            Cow::Borrowed(_) => {
                Ok(Cow::Borrowed(expr))
            }
        }
    }

    let wa = wa.unwrap();
    let e_new = expand_with_expr_ext(expr, was, wa, None)?;
    let re: Option<&mut RollupExpr> = None;

    let wme = match e_new.as_ref() {
        Expression::MetricExpression(me) => Some(me),
        Expression::Rollup(e) => match &e.expr.as_ref() {
            Expression::MetricExpression(me) => Some(me),
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
        None => Ok(Cow::Borrowed(expr)),
        Some(rollup) => {
            rollup.expr = Box::new(Expression::MetricExpression(new_me));
            let expr = Expression::Rollup(rollup.clone());
            Ok(Cow::Owned(expr))
        }
    }
}

fn expand_metric_labels<'a>(expr: &'a Expression,
                            me: &'a MetricExpr,
                            was: &Vec<WithArgExpr>) -> ParseResult<Cow<'a, MetricExpr>> {

    if me.label_filters.len() > 0 {
        // already expanded
        return Ok(Cow::Borrowed(me));
    }

    // Populate me.label_filters
    let mut me_new = MetricExpr::default();
    for lfe in me.label_filter_exprs.iter() {
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

            let wae = expand_with_expr_ext(e_new.as_ref(), was, wa, None)?;
            match wae.as_ref() {
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
        let str_expr = Expression::from(lfe.value.as_str());
        let se = expand_with_expr_internal(was, &str_expr)?;
        let lfe_new = LabelFilterExpr::new_tag(
            lfe.label.to_string(),
            lfe.op,
            se.to_string(), TextSpan::default());
        let lf = lfe_new.to_label_filter();
        me_new.label_filters.push(lf);
    }

    me_new.label_filter_exprs.clear();
    remove_duplicate_label_filters(&mut me_new.label_filters);

    return Ok(Cow::Owned(me_new));
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

fn expand_with_expr_ext<'a>(expr: &'a Expression,
                            was: &Vec<WithArgExpr>,
                            wa: &WithArgExpr,
                            args: Option<&Vec<BExpression>>) -> ParseResult<Cow<'a, Expression>> {

    let args_len = if let Some(expressions) = args {
        expressions.len()
    } else {
        0
    };

    if wa.args.len() != args_len {
        if args.is_none() {
            // Just return MetricExpr with the wa.name name.
            let me = Expression::MetricExpression(MetricExpr::new(&wa.name));
            return Ok(Cow::Owned(me));
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

fn expand_with_args(was: &Vec<WithArgExpr>, args: &Vec<BExpression>) -> ParseResult<Vec<BExpression>> {
    let mut result: Vec<BExpression> = Vec::with_capacity(args.len());

    for arg in args.iter() {
        match expand_with_expr_internal(was, arg.as_ref())? {
            Cow::Owned(value) => {
                result.push( Box::new(value) )
            },
            Cow::Borrowed(borrowed) => {
                let value = borrowed.clone(); // can be do better ??
                result.push( Box::new(value) );
            }
        }
    }

    Ok(result)
}

fn get_with_arg_expr<'a>(was: &'a Vec<WithArgExpr>, name: &str) -> Option<&'a WithArgExpr> {
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

/// type_changed checks for the case where the type of an expression changes, in our case
/// because of a binary op simplification (eg. a > b or "foo" + "bar") i.o.w binary op to
/// scalar or string
fn type_changed(before: &Expression, after: &Expression) -> bool {
    let (first_type, second_type) = (before.type_name(), after.type_name());
    first_type != second_type
}