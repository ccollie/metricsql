use std::collections::HashSet;
use std::ops::{Deref};

use crate::ast::*;
use crate::lexer::{TextSpan};
use crate::parser::{ArgCountError, ParseError, ParseResult};

pub(crate) fn expand_with_expr(was: &[WithArgExpr], expr: &Expression) -> ParseResult<Expression> {
    use Expression::*;

    match expr {
        BinaryOperator(binary) => expand_binary(binary, was),
        Function(function) => expand_function(function, was),
        Aggregation(aggregation) => expand_aggregation(aggregation, was),
        MetricExpression(metric) => expand_metric_expr(metric, was),
        Rollup(rollup) => expand_rollup(rollup, was),
        Parens(parens) => expand_parens_expr(parens, was),
        String(str) => match expand_string(str, was) {
            Err(e) => Err(e),
            Ok(s) => Ok(String(s)),
        },
        With(with) => {
            let mut was_new: Vec<WithArgExpr> = Vec::with_capacity(was.len() + with.was.len());
            was_new.extend_from_slice(was);
            was_new.extend(with.was.clone());
            //todo: find a better way. this is ridiculous
            let expr = with.expr.clone();
            expand_with_expr(&was_new, &expr)
        }
        _ => Ok(expr.clone()),
    }
}

fn expand_binary(e: &BinaryOpExpr, was: &[WithArgExpr]) -> ParseResult<Expression> {
    let left = expand_with_expr(was, e.left.deref())?;
    let right = expand_with_expr(was, e.right.deref())?;

    if e.op == BinaryOp::Add {
        match (&left, &right) {
            (Expression::String(left), Expression::String(right)) => {
                let concat = format!("{}{}", left.value, right.value);
                let span = TextSpan::at(left.span.start, concat.len());
                let expr = StringExpr::new(concat, span);
                return Ok(Expression::String(expr))
            },
            (Expression::Number(left), Expression::Number(right)) => {
                let sum = left.value + right.value;
                let text = format!("{}", sum);
                let span = TextSpan::at(left.span.start, text.len());
                let num = NumberExpr::new(sum, span);
                return Ok(Expression::Number(num))
            }
            _ => {
                // Err(ParseError::General(
                //     "+ operator can only be used with string or number arguments".to_string(),
                // ))
            },
        };
    }

    let mut be = BinaryOpExpr::new(e.op, left, right)?;
    be.bool_modifier = e.bool_modifier;
    be.span = e.span;

    if let Some(modifier) = &e.group_modifier {
        let labels = expand_modifier_args(was, &modifier.labels)?;
        be.group_modifier = Some( GroupModifier::new(modifier.op, labels) );
    }

    if let Some(ref modifier) = &e.join_modifier {
        let labels = expand_modifier_args(was, &modifier.labels)?;
        be.join_modifier = Some( JoinModifier::new(modifier.op, labels) );
    }

    let bin_expr = Expression::BinaryOperator(be);
    // let args = vec![Box::new(bin_expr)];
    // Ok(Expression::Parens(ParensExpr::new(args, e.span)))
    Ok(bin_expr)
}

fn expand_function(func: &FuncExpr, was: &[WithArgExpr]) -> ParseResult<Expression> {
    let args = expand_with_args(was, &func.args);

    // TODO: !!!!!!! fill out impl of udf/withexpr on BuiltinFunction
    match get_with_arg_expr(was, &func.with_name) {
        Some(wa) => {
            let expr = expand_with_expr_ext(was, wa, Some(&args))?;
            Ok(expr)
        }
        None => {
            let fe = FuncExpr::new(&func.name(), args, func.span.clone())?;
            Ok(Expression::Function(fe))
        }
    }
}

fn expand_aggregation(aggregate: &AggrFuncExpr, was: &[WithArgExpr]) -> ParseResult<Expression> {
    let args = expand_with_args(was, &aggregate.args);
    let wa = get_with_arg_expr(was, &aggregate.name);
    if let Some(wae) = wa {
        return expand_with_expr_ext(was, wae, Some(&args));
    }
    let empty_vec = vec![];

    let mod_args = match &aggregate.modifier {
        Some(modifier) => &modifier.args,
        None => &empty_vec,
    };

    match expand_modifier_args(was, mod_args) {
        Err(e) => Err(e),
        Ok(modifier_args) => {
            let mut ae = aggregate.clone();
            ae.args = args;
            if let Some(ref mut modifier) = ae.modifier {
                modifier.args = modifier_args;
            }
            Ok(Expression::Aggregation(ae))
        }
    }
}

fn expand_parens_expr(parens: &ParensExpr, was: &[WithArgExpr]) -> ParseResult<Expression> {
    let mut result: Vec<BExpression> = Vec::with_capacity(parens.expressions.len());
    let mut span = was[0].expr.span();
    for e in parens.expressions.iter() {
        let expr = expand_with_expr(was, e)?;
        span = span.cover(expr.span());
        result.push(Box::new(expr));
    }

    Ok(Expression::Parens(ParensExpr::new(result, span)))
}

fn expand_string(e: &StringExpr, was: &[WithArgExpr]) -> ParseResult<StringExpr> {
    if e.is_expanded() {
        // Already expanded. Copying should be cheap
        return Ok(e.clone());
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
        let e_new = match expand_with_expr_ext(was, wa, None)? {
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

    let span = TextSpan::at(start, b.len());
    Ok(StringExpr::new(b, span))
}

fn expand_rollup(rollup: &RollupExpr, was: &[WithArgExpr]) -> ParseResult<Expression> {
    let mut re = rollup.clone();
    let e_new = expand_with_expr(was, &re.expr)?;
    re.expr = BExpression::from(e_new);
    if let Some(at) = &re.at {
        let at = expand_with_expr(was, at)?;
        re.at = Some(Box::new(at));
    }
    Ok(Expression::Rollup(re))
}

fn expand_metric_expr(e: &MetricExpr, was: &[WithArgExpr]) -> ParseResult<Expression> {
    if e.is_expanded() {
        // todo: COW
        // Already expanded.
        return Ok(Expression::MetricExpression(e.clone()));
    }
    let e = expand_metric_labels(e, was)?;
    if !e.has_non_empty_metric_group() {
        return Ok(Expression::MetricExpression(e));
    }
    let k = &e.label_filters[0].value;
    let wa = get_with_arg_expr(was, k);
    if wa.is_none() {
        return Ok(Expression::MetricExpression(e));
    }

    let wa = wa.unwrap();
    let mut e_new = expand_with_expr_ext(was, wa, None)?;
    let re: Option<RollupExpr> = None;

    let wme = match e_new {
        Expression::MetricExpression(ref mut me) => Some(me),
        Expression::Rollup(ref mut e) => match *e.expr {
            Expression::MetricExpression(ref mut me) => Some(me),
            _ => None,
        },
        _ => None,
    };

    if wme.is_none() {
        if !e.is_only_metric_group() {
            let msg = format!("cannot expand {:?} to non-metric expression {}", e, wa);
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

    let mut me = MetricExpr::default();
    let mut label_filters = wme.label_filters.clone(); // do we need the original ??
    let other = e.label_filters[1..].to_vec();
    for label in other {
        label_filters.push(label.clone());
    }

    remove_duplicate_label_filters(&mut label_filters);
    me.label_filters = label_filters;

    match re {
        None => Ok(Expression::MetricExpression(me)),
        Some(t) => {
            let mut re_new = t;
            re_new.set_expr(me);
            Ok(Expression::Rollup(re_new))
        }
    }
}

fn expand_metric_labels(expr: &MetricExpr, was: &[WithArgExpr]) -> ParseResult<MetricExpr> {

    if expr.label_filters.len() > 0 {
        // already expanded
        return Ok(expr.clone());   // todo: use COW to avoid this clone
    }

    let mut me: MetricExpr = MetricExpr::default();

    // Populate me.label_filters
    for lfe in expr.label_filter_exprs.iter() {
        if !lfe.is_init() {
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
                lfe.label, expr, e_new
            );

            match expand_with_expr_ext(was, wa, None)? {
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
        let se = expand_string(&lfe.value, was)?;
        let lfe_new = LabelFilterExpr::new(lfe.label.clone(), se, lfe.op);
        let lf = lfe_new.to_label_filter();
        me.label_filters.push(lf);
    }

    me.label_filter_exprs.clear();
    remove_duplicate_label_filters(&mut me.label_filters);

    Ok(me)
}

fn expand_modifier_args(was: &[WithArgExpr], args: &[String]) -> Result<Vec<String>, ParseError> {
    if args.is_empty() {
        return Ok(vec![]);
    }

    let mut dst_args: Vec<String> = Vec::with_capacity(1);
    for arg in args {
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

fn expand_with_expr_ext(
    was: &[WithArgExpr],
    wa: &WithArgExpr,
    args: Option<&Vec<BExpression>>,
) -> ParseResult<Expression> {
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
        let a: *const &WithArgExpr = &wa_tmp;
        let b: *const &WithArgExpr = &wa;
        if a == b {
            break;
        }
        was_new.push(wa_tmp.clone());
    }

    for (i, arg) in args.unwrap().iter().enumerate() {
        let wae = WithArgExpr {
            name: wa.args[i].clone(),
            args: vec![],
            expr: arg.clone(),
        };
        was_new.push(wae);
    }

    expand_with_expr(&was_new, wa.expr.deref())
}

fn expand_with_args(was: &[WithArgExpr], args: &Vec<BExpression>) -> Vec<BExpression> {
    let mut res: Vec<BExpression> = Vec::with_capacity(args.len());
    for arg in args.iter() {
        let expanded = expand_with_expr(was, arg).unwrap();
        res.push(Box::new(expanded));
    }
    res
}

fn get_with_arg_expr<'a>(was: &'a [WithArgExpr], name: &'a str) -> Option<&'a WithArgExpr> {
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
    // Type inference lets us omit an explicit type signature (which
    // would be `HashSet<String>` in this example).
    let mut set: HashSet<String> = HashSet::with_capacity(lfs.len());
    lfs.retain(|lf| {
        let key = lf.label.to_string();
        if set.contains(&key) {
            return false;
        }
        set.insert(key);
        true
    })
}
