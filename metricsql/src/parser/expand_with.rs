use crate::ast::*;
use crate::lexer::{is_string_prefix, quote};
use crate::parser::{ArgCountError, ParseError, ParseResult};
use enquote::unquote;
use std::collections::HashSet;
use std::ops::Deref;

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
        return match (left, right) {
            (Expression::String(left), Expression::String(right)) => {
                let expr = StringExpr::new(format!("{}{}", left.s, right.s));
                Ok(Expression::String(expr))
            }
            _ => Err(ParseError::General(
                "+ operator can only be used with string arguments".to_string(),
            )),
        };
    }

    let mut group_modifier_args: Vec<String> = vec![];
    let mut join_modifier_args: Vec<String> = vec![];

    if let Some(modifier) = &e.group_modifier {
        group_modifier_args = expand_modifier_args(was, &modifier.labels)?;
    }

    if let Some(modifier) = &e.join_modifier {
        join_modifier_args = expand_modifier_args(was, &modifier.labels)?;
    }

    let mut be = BinaryOpExpr::new(e.op, left, right);
    be.bool_modifier = e.bool_modifier;

    if let Some(ref mut group_mod) = be.group_modifier {
        group_mod.labels = group_modifier_args;
    }

    if let Some(ref mut join_mod) = be.join_modifier {
        join_mod.labels = join_modifier_args;
    }

    let args = vec![Box::new(Expression::BinaryOperator(be))];
    Ok(Expression::Parens(ParensExpr::new(args)))
}

fn expand_function(func: &FuncExpr, was: &[WithArgExpr]) -> ParseResult<Expression> {
    let args = expand_with_args(was, &func.args);
    match get_with_arg_expr(was, &func.name) {
        Some(wa) => {
            let expr = expand_with_expr_ext(was, wa, Some(&args))?;
            Ok(expr)
        }
        None => {
            let fe = FuncExpr::new(&func.name, args);
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
    for e in parens.expressions.iter() {
        let expr = expand_with_expr(was, e)?;
        result.push(Box::new(expr));
    }

    Ok(Expression::Parens(ParensExpr::new(result)))
}

fn expand_string(e: &StringExpr, was: &[WithArgExpr]) -> ParseResult<StringExpr> {
    if e.is_expanded() {
        // Already expanded. Copying should be cheap
        return Ok(e.clone());
    }
    let mut b: String = String::new();
    for token in e.tokens.as_ref().unwrap() {
        if is_string_prefix(token) {
            let s = extract_string_value(token)?;
            b.push_str(s.as_str());
            continue;
        }
        let wa = get_with_arg_expr(was, token);
        if wa.is_none() {
            let msg = format!("missing {} value inside StringExpr", token);
            return Err(ParseError::WithExprExpansionError(msg));
        }
        let wa = wa.unwrap();
        let e_new = match expand_with_expr_ext(was, wa, None)? {
            Expression::String(e) => e,
            _ => {
                let msg = format!("{} is not a string expression", token);
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

        b.push_str(e_new.s.as_str());
    }

    Ok(StringExpr::new(b))
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
    if !e.label_filters.is_empty() {
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

fn expand_metric_labels(e: &MetricExpr, was: &[WithArgExpr]) -> ParseResult<MetricExpr> {
    let mut me: MetricExpr = MetricExpr::default();
    // Populate me.LabelFilters
    for lfe in e.label_filter_exprs.iter() {
        if !lfe.is_expanded() {
            // Expand lfe.Label into Vec<LabelFilter>.
            let wa = get_with_arg_expr(was, &lfe.label);
            if wa.is_none() {
                let msg = format!("missing {} value inside MetricExpr", lfe.label);
                return Err(ParseError::WithExprExpansionError(msg));
            }

            let wa = wa.unwrap();
            let e_new = expand_with_expr_ext(was, wa, None)?;
            let error_msg = format!(
                "{} must be filters expression inside {}: got {}",
                lfe.label, e, e_new
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
        let lfe_new = LabelFilterExpr {
            label: lfe.label.clone(),
            value: se,
            op: lfe.op,
        };
        let lf = lfe_new.to_label_filter();
        me.label_filters.push(lf);
    }
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
            // Just return MetricExpr with the wa.Name name.
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

fn extract_string_value(token: &str) -> Result<String, ParseError> {
    if !is_string_prefix(token) {
        let msg = format!(
            "StringExpr must contain only string literals; got {}",
            token
        );
        return Err(ParseError::WithExprExpansionError(msg));
    }

    let ch = token.chars().next().unwrap();

    // See https://prometheus.io/docs/prometheus/latest/querying/basics/#string-literals
    if ch == '\'' {
        let msg = format!(
            "StringExpr must contain only string literals; got {}",
            token
        );
        if token.len() < 2 {
            return Err(ParseError::WithExprExpansionError(msg));
        }
        let ch = token.chars().last().unwrap();
        if ch != '\'' {
            return Err(ParseError::WithExprExpansionError(msg));
        }
        let s = &token[1..token.len() - 1];
        let _ = s.replace("\\'", "'").replace("\\\"", "\"");
        return Ok(quote(s));
    }
    match unquote(token) {
        Ok(res) => Ok(res),
        Err(_) => {
            Err(ParseError::WithExprExpansionError(
                "Cannot extract string value".to_string(),
            ))
        }
    }
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
