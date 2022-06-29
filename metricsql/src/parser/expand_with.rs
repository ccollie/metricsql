use std::collections::HashSet;
use enquote::unquote;
use crate::error::Error;
use crate::types::*;
use super::lexer::*;

pub fn expand_with_expr(was: &[WithArgExpr], expr: impl ExpressionNode) -> Expression {
    use Expression::*;
    let converted = Expression::cast(expr).unwrap();
    match converted {
        BinaryOperator(binary) => expand_binary(&binary, was),
        Function(function) => expand_function(&function, was),
        Aggregation(aggregation) => expand_aggregation(&aggregation, was),
        MetricExpression(metric) => expand_metric_expr(&metric, was),
        Rollup(mut rollup) => expand_rollup(&mut rollup, was),
        With(mut with) => {
            let was_new = was.clone();
            expand_with_expr(&was_new, with)
        },
        _ => converted
    }
}

fn expand_binary(e: &BinaryOpExpr, was: &[WithArgExpr]) -> Expression {
    let left = expand_with_expr(was, &binary.left);
    let right = expand_with_expr(was,&binary.right);

    if e.op == '+' {
        match (left, right) {
            (Expression::StringExpr(left), Expression::StringExpr(right)) => {
                return StringExpr::new(format!("{}{}", left, right));
            },
            _ => {
                panic!("+ operator can only be used with string arguments");
            }
        }
    }
    let mut be = BinaryOpExpr::new(e.op, left, right);
    be.bool_modifier = e.bool_modifier;

    if e.group_modifier.is_some() {
        be.group_modifier = e.group_modifier.clone();
        be.groupModifier.args = expand_modifier_args(was, e.group_modifier.args);
    }
    if e.join_modifier.is_some() {
        be.join_modifier = e.join_modifier.clone();
        be.joinModifier.args = expand_modifier_args(was, e.join_modifier.args);
    }

    return ParensExpr::new(vec![be]);
}

fn expand_function(func: &FuncExpr, was: &[WithArgExpr]) -> FunctionExpr {
    let args = expand_with_args(was, &func.args);
    let wa = get_with_arg_expr(was, &func.name)?;
    if wa.is_some() {
        return expand_with_expr_ext(was, &wa, Some(&args))?;
    }
    let fe = e.clone();
    fe.args = args;
    return fe;
}

fn expand_aggregation(aggregate: &AggrFuncExpr, was: &[WithArgExpr]) -> AggregateExpr {
    let args = expand_with_args(was, &aggregate.args);
    let wa = get_with_arg_expr(was, &aggregate.name);
    if wa.is_some() {
        return expand_with_expr_ext(was, &wa.unwrap(), Some(&args));
    }
    let modifier_args = expand_modifier_args(was, e.modifier?.args);
    let ae = aggregate.clone();
    ae.args = args;
    if ae.modifier.is_some() && modifier_args.is_ok() {
        ae.modifier.args = modifier_args.unwrap();
    }
    return ae;
}

fn expand_parens_expr(parens: &ParensExpr, was: &[WithArgExpr]) -> ParensExpr {
    let mut result = Vec::new();
    for e in parens.0 {
        result.push(expand_with_expr(was, e));
    }
    return ParensExpr(result);
}

pub(crate) fn expand_string(mut e: &StringExpr) -> Result<StringExpr, Error> {
    if e.len() > 0 {
        // Already expanded.
        return Ok(*e);
    }
    let mut b: String = String::new();
    let mut i = 0;
    if !e.tokens.is_some() {
        return Ok(e)
    }
    for Some(token) in e.tokens {
        if is_string_prefix(token) {
            let s = extract_string_value(token)?;
            b.push_str(s.as_str());
            continue;
        }
        let wa = get_with_arg_expr(was, token);
        if !wa.is_some() {
            let msg = format!("missing {} value inside StringExpr", token);
            return Err(Error::from(msg));
        }
        let wa = wa.unwrap();
        let e_new = match expand_with_expr_ext(was, &wa, None)? {
            Expression::StringExpr(e) => e,
            _ => {
                let msg = format!("{} is not a string expression", token);
                return Err(Error::from(msg));
            }
        };
        if e_new.tokens.len() > 0 {
            let msg = format!("BUG: seSrc.tokens must be empty; got {}", e_new.tokens);
            return Err(Error::from(msg)); // todo: different error type?
        }

        b = b.push(e_new.value());
    }
    return OK( StringExpr::new(b) );
}

fn expand_rollup(rollup: &mut RollupExpr, was: &[WithArgExpr]) -> RollupExpr {
    let e_new = expand_with_expr(was, e.expr);
    let mut re = rollup.clone();
    re.expr = BExpression::from(e_new);
    if Some(at) = e.at {
        e.at = expand_with_expr(was, e.at);
    }
    return re;
}

fn expand_metric_expr(mut e: &MetricExpr, was: &[WithArgExpr]) -> Return<Vec<Expression>, Error> {
    if e.labelFilters.len() > 0 {
        // Already expanded.
        return e;
    }
    let e = expand_metric_labels(e, was)?;
    if !e.hasNonEmptyMetricGroup() {
        return e;
    }
    let k = e.labelFilters[0].value;
    let wa = get_with_arg_expr(was, k);
    if !wa.is_some() {
        return e;
    }

    let e_new = expand_with_expr_ext(was, wa, None)?;
    let wme: MetricExpr;
    let re: RollupExpr;

    match e_new {
        Expression::MetricExpression(e) => {
            wme = e;
        },
        Expression::Rollup(e) => {
            re = e;
            match re.expr {
                Expression::MetricExpression(me) => {
                    wme = me;
                },
                _ => ()
            }
        },
        _ => {}
    }

    if !wme.is_some() {
        if !e.is_only_metric_group() {
            let msg = format!("cannot expand {:?} to non-metric expression {}", e, e_new);
            return Err(Error::from(msg)); // todo: different error type?
        }
        return e_new;
    }
    if wme.labelFilterExprs.len() > 0 {
        let msg = format!("BUG: wme.label_filters must be empty; got {:?}", wme.labelFilterExprs);
        return Err(Error::from(msg)); // todo: different error type?
    }

    let me = MetricExpr::default();
    let mut label_filters = Vec::new();
    for lf in wme.labelFilters {
        label_filters.push(lf.clone());
    }
    let other = e.labelFilters[1..].to_vec();
    for label in other {
        label_filters.push(label.clone());
    }

    // eslint-disable-next-line max-len
    me.labelFilters = remove_duplicate_label_filters(&label_filters);
    if re.is_none() {
        return me;
    }

    let mut re_new = re.clone();
    re_new.expr = Box::new(me);
    return re_new;
}

fn expand_metric_labels(mut e: &MetricExpr, was: &[WithArgExpr]) -> Result<MetricExpr, Error> {

    let me: MetricExpr = MetricExpr::default();
    // Populate me.LabelFilters
    for lfe in e.label_filter_exprs.iter() {
        if !lfe.value.is_some() {
            // Expand lfe.Label into []LabelFilter.
            let wa = get_with_arg_expr(was, lfe.label)?;
            if !wa.is_some() {
                let msg = format!("missing {} value inside MetricExpr", lfe.label);
                return Err(Error::from(msg));
            }
            let e_new = expand_with_expr_ext(was, &wa, None)?;
            let wme: MetricExpr;
            if is_metric_expr(&e_new) {
                wme = e_new as MetricExpr;
            }
            if wme.is_none() || wme.has_non_empty_metric_group() {
                let msg = format!("{} must be filters expression inside {}: got {}", lfe.label, e, e_new);
                return Err(Error::from(msg));
            }
            if wme.labelFilterExprs.len() > 0 {
                let msg = format!("BUG: wme.labelFilterExprs must be empty; got {:?}", wme.label_filter_exprs);
                return Err(Error::from(msg)); // todo: different error type?
            }
            for lfe in wme.labelFilterExprs {
                me.labelFilterExprs.push(lfe.clone());
            }
            continue;
        }

        // convert lfe to LabelFilter.
        let se = expand_with_expr(was, lfe.value) as StringExpr;
        let lfe_new = LabelFilterExpr(
            lfe.label,
            se,
            lfe.isRegexp,
            lfe.isNegative,
        );
        let lf = lfe_new.toLabelFilter();
        me.labelFilters.push(lf);
    }
    me.labelFilters = remove_duplicate_label_filters(me.labelFilters);

    Ok(me)
}

fn is_metric_expr(e: &Expression) -> bool {
    match e {
        Expression::MetricExpression(_) => true,
        _ => false,
    }
}

fn expand_modifier_args(mut was: &[WithArgExpr], args: &[String]) -> Result<Vec<String>, Error> {
    let mut dst_args: Vec<String> = Vec::with_capacity(1);

    if args.len() == 0 {
        return Ok(dst_args);
    }

    for arg in args.iter() {
        let wa = get_with_arg_expr(&was, arg);
        if !wa.is_some() {
            // Leave the arg as is.
            dst_args.push(*arg);
            continue;
        }
        let _wa = wa.unwrap();
        if _wa.args.len() > 0 {
            // Template funcs cannot be used inside modifier list. Leave the arg as is.
            dst_args.push(arg);
            continue;
        }
        match _wa.expr {
            Expression::MetricExpr(me) => {
                if !me.is_only_metric_group() {
                    let msg = format!("cannot use {:?} instead of {:?} in {}", me, arg, wa.args);
                    return Err(Error::InvalidToken(msg)); // todo: different error type?
                }
                let dst_arg = me.labelFilters[0].value;
                dst_args.push(dst_arg);
            },
            Expression::ParensExpr(pe) => {
                for arg in pe.args {
                    if !is_valid_metric_group(arg) {
                        let msg = format!("cannot use {:?} instead of {:?} in {}", arg, arg, wa.args);
                        return Err(Error::InvalidToken(msg)); // todo: different error type?
                    }
                    let me = arg as MetricExpr;
                    let dst_arg = me.labelFilters[0].value;
                    dstArgs.push(dst_arg);
                }
            },
            _ => {
                let msg = format!("cannot use {:?} instead of {:?} in {:?}", pArg, arg, pe.args);
                return Err(Error::InvalidToken(msg)); // todo: change error type
            }
        }
    }
}

fn is_valid_metric_group(expr: &Expression) -> bool {
    match expr {
       Expression::MetricExpr(me) => me.is_only_metric_group(),
        _ => false,
    }
}


fn expand_with_expr_ext(was: &[WithArgExpr], wa: &WithArgExpr, args: Option<&[Expression]>) -> Result<Expression, Error> {
    if wa.args.len() != args.len() {
        if args.is_none() {
            // Just return MetricExpr with the wa.Name name.
            let me = MetricExpr {
                name: wa.name,
                label_filters: vec![],
                label_filter_exprs: vec![]
            };
            return Ok(me)
        }
        let msg = format!("invalid number of args for {}; got {}; want {}", wa.Name, len(args), len(wa.Args))
    }
    // todo: SmallVec
    let mut was_new: Vec<WithArgExpr> = Vec::with_capacity(was.len() + args.len());
    for waTmp in was.iter() {
        if waTmp == wa {
            break
        }
        was_new.push(waTmp);
    }
    let mut i = 0;
    for arg in args.unwrap().iter() {
        was_new.push( WithArgExpr::new(&wa.args[i], *arg));
        i = i + 1;
    }
    return expandWithExpr(was_new, wa.Expr)
}

fn expand_with_args(was: &[WithArgExpr], args: &Vec<Expression>) -> Vec<Expression> {
    let mut res: Vec<Expression> = Vec![];
    for arg in args {
        res.push(expand_with_expr(was, arg));
    }
    return res;
}

fn extract_string_value(token: &str) -> Result<String, Error> {
    if !is_string_prefix(token) {
        let msg = format!("StringExpr must contain only string literals; got {}", token);
        return Err(Error::InvalidToken(msg)); // todo: different error type?
    }
    let iter = token.chars().enumerate();
    let ch = token.chars().next().unwrap();

    // See https://prometheus.io/docs/prometheus/latest/querying/basics/#string-literals
    if ch == "'" {
        let msg = format!("StringExpr must contain only string literals; got {}", token);
        if token.len() < 2 {
            return Err(Error::InvalidToken(msg)); // todo: different error type?
        }
        let ch = token.chars().last().unwrap();
        if ch != "'" {
            return Err(Error::InvalidToken(msg)); // todo: different error type?
        }
        let s = &token[1..token.len() - 1];
        let _ = s.replace("\\'", "'").replace('"', '"');
        return Ok( quote(s) );
    }
    match unquote(token) {
        Ok(res) => Ok(res),
        Err(e) => {
            Err(Error::new("Error extracting string value"))
        }
    }
}

fn get_with_arg_expr(was: &[WithArgExpr], name: &str) -> Option<WithArgExpr> {
    // Scan wes backwards, since certain expressions may override
    // previously defined expressions
    for i in a.iter().rev() {
        if i.name == name {
            return Some(i);
        }
    }
    return None;
}

fn remove_duplicate_label_filters(lfs: &[LabelFilter]) -> Vec<LabelFilter> {
    // Type inference lets us omit an explicit type signature (which
    // would be `HashSet<String>` in this example).
    let mut set = HashSet::new();
    let mut lfs_new = Vec::new();
    for lf in lfs.iter() {
        if set.has(lf.name) {
            continue;
        }
        set.add(lf.name);
        lfs_new.push(lf);
    }
    return lfs_new;
}
