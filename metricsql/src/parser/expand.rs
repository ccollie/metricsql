use std::collections::HashSet;

use crate::ast::{
    AggregationExpr, BinaryExpr, Expr, FunctionExpr, MetricExpr, ParensExpr, RollupExpr,
    WithArgExpr, WithExpr,
};
use crate::common::{
    AggregateModifier, GroupModifier, JoinModifier, LabelFilter, remove_duplicate_label_filters,
    StringExpr, StringSegment,
};
use crate::parser::{ParseError, ParseResult, syntax_error};
use crate::parser::symbol_provider::SymbolProviderRef;

pub fn expand_with(
    symbols: &SymbolProviderRef,
    was: &Vec<WithArgExpr>,
    we: WithExpr,
) -> ParseResult<Expr> {
    let mut was_new = Vec::with_capacity(was.len() + we.was.len());
    was_new.append(&mut was.clone());
    was_new.append(&mut we.was.clone());
    expand_with_expr(symbols, &was_new, *we.expr)
}

pub(super) fn expand_with_expr(
    symbols: &SymbolProviderRef,
    was: &Vec<WithArgExpr>,
    expr: Expr,
) -> ParseResult<Expr> {
    use Expr::*;
    match expr {
        BinaryOperator(be) => expand_binary_operator(symbols, was, be),
        Function(fe) => expand_function(symbols, was, fe),
        Aggregation(ae) => expand_aggregation(symbols, was, ae),
        Parens(pe) => expand_parens(symbols, was, pe),
        Rollup(re) => expand_rollup(symbols, was, re),
        StringExpr(se) => expand_string_expr(symbols, was, &se),
        MetricExpression(me) => expand_metric_expression(symbols, was, me),
        With(we) => expand_with(symbols, was, we),
        _ => Ok(expr),
    }
}

pub fn should_expand(expr: &Expr) -> bool {
    use Expr::*;

    match expr {
        StringLiteral(_) | Number(_) | Duration(_) => false,
        BinaryOperator(be) => should_expand(&be.left) || should_expand(&be.right),
        StringExpr(se) => !se.is_expanded(),
        Parens(pe) => !pe.expressions.is_empty() && pe.expressions.iter().any(|x| should_expand(x)),
        Rollup(re) => {
            if should_expand(&re.expr) {
                return true;
            }
            if let Some(at) = &re.at {
                return should_expand(at);
            }
            false
        }
        _ => true,
    }
}

fn expand_with_args(
    symbols: &SymbolProviderRef,
    was: &Vec<WithArgExpr>,
    args: Vec<Expr>,
) -> ParseResult<Vec<Expr>> {
    let mut dst_args: Vec<Expr> = Vec::with_capacity(args.len());
    for arg in args {
        let dst_arg = expand_with_expr(symbols, was, arg)?;
        dst_args.push(dst_arg);
    }
    Ok(dst_args)
}

fn expand_with_expr_box(
    symbols: &SymbolProviderRef,
    was: &Vec<WithArgExpr>,
    expr: Box<Expr>,
) -> ParseResult<Box<Expr>> {
    if should_expand(&expr) {
        let dst_expr = expand_with_expr(symbols, was, *expr)?;
        Ok(Box::new(dst_expr))
    } else {
        Ok(expr)
    }
}

fn expand_binary_operator(
    symbols: &SymbolProviderRef,
    was: &Vec<WithArgExpr>,
    be: BinaryExpr,
) -> ParseResult<Expr> {
    let mut be = be;

    be.left = expand_with_expr_box(symbols, was, be.left)?;
    be.right = expand_with_expr_box(symbols, was, be.right)?;

    if let Some(modifier) = &be.group_modifier {
        let labels = expand_modifier_args(symbols, was, &modifier.labels)?;
        if labels != modifier.labels {
            be.group_modifier = Some(GroupModifier::new(modifier.op, labels));
        }
    }

    if let Some(ref modifier) = &be.join_modifier {
        let labels = expand_modifier_args(symbols, was, &modifier.labels)?;
        if labels != modifier.labels {
            be.join_modifier = Some(JoinModifier::new(modifier.op, labels));
        }
    }

    // parens(expr)
    Ok(Expr::BinaryOperator(be))
}

pub(super) fn expand_metric_expression(
    symbols: &SymbolProviderRef,
    was: &Vec<WithArgExpr>,
    me: MetricExpr,
) -> ParseResult<Expr> {
    fn ensure_filters_resolved(me: &MetricExpr) -> ParseResult<()> {
        if !me.label_filter_expressions.is_empty() {
            let msg = format!(
                "BUG: wme.label_filters must be empty; got {}",
                me.label_filter_expressions
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            return Err(ParseError::General(msg));
        }
        Ok(())
    }

    fn handle_expanded(wme: &MetricExpr, src: &mut MetricExpr) -> ParseResult<Expr> {
        ensure_filters_resolved(&wme)?;

        let mut filters = wme.label_filters.clone();

        for i in (1..src.label_filters.len()).rev() {
            filters.push(src.label_filters.remove(i));
        }

        remove_duplicate_label_filters(&mut filters);

        Ok(Expr::MetricExpression(MetricExpr::with_filters(filters)))
    }

    if me.is_expanded() {
        // Already expanded.
        return Ok(Expr::MetricExpression(me));
    }

    let mut new_selector: MetricExpr = MetricExpr::default();
    new_selector.label_filters = me.label_filters.clone();

    // Populate me.LabelFilters
    for lfe in me.label_filter_expressions {
        if lfe.value.is_empty() {
            // Expand lfe.Label into vec<LabelFilter>.
            // we have something like: foo{commonFilters} and we want to expand it into
            // foo{bar="bax", job="trace"}
            let wa = get_with_arg_expr(symbols, was, &lfe.label);
            if wa.is_none() {
                let msg = format!("missing {} value inside {}", lfe.label, new_selector);
                return Err(ParseError::General(msg));
            }
            let e_new = expand_with_expr_ext(symbols, was, wa.unwrap(), vec![])?;

            let mut has_non_empty_metric_group = false;
            let wme = match e_new {
                Expr::MetricExpression(ref me) => {
                    has_non_empty_metric_group = me.has_non_empty_metric_group();
                    Some(me)
                }
                _ => None,
            };
            if wme.is_none() || has_non_empty_metric_group {
                let msg = format!(
                    "{} must be filters expression inside {}; got {}",
                    lfe.label, new_selector, &e_new
                );
                return Err(ParseError::General(msg));
            }
            let wme = wme.unwrap();
            ensure_filters_resolved(&wme)?;

            new_selector
                .label_filters
                .append(&mut wme.label_filters.clone());
            continue;
        }

        // convert lfe to LabelFilter.
        let se = expand_string_expr(symbols, was, &lfe.value)?;
        let lf = LabelFilter {
            label: lfe.label,
            op: lfe.op,
            value: get_expr_as_string(&se)?,
        };

        new_selector.label_filters.push(lf);
    }

    remove_duplicate_label_filters(&mut new_selector.label_filters);
    if !new_selector.has_non_empty_metric_group() {
        return Ok(Expr::MetricExpression(new_selector));
    }
    let k = &new_selector.label_filters[0].value;
    let wa = get_with_arg_expr(symbols, was, k);
    if wa.is_none() {
        return Ok(Expr::MetricExpression(new_selector));
    }

    let expanded = expand_with_expr_ext(symbols, was, wa.unwrap(), vec![])?;

    let wme = match &expanded {
        Expr::MetricExpression(me) => {
            let res = handle_expanded(me, &mut new_selector)?;
            Some(res)
        }
        Expr::Rollup(re) => match re.expr.as_ref() {
            Expr::MetricExpression(me) => {
                let mut rollup = re.clone();
                rollup.expr = Box::new(handle_expanded(me, &mut new_selector)?);
                Some(Expr::Rollup(rollup))
            }
            _ => None,
        },
        _ => None,
    };

    match wme {
        None => {
            if !new_selector.is_only_metric_group() {
                let msg = format!(
                    "cannot expand {} to non-metric expression {}",
                    new_selector, expanded
                );
                return Err(ParseError::General(msg));
            }
            return Ok(expanded);
        }
        Some(e) => Ok(e),
    }
}

fn get_expr_as_string(expr: &Expr) -> ParseResult<String> {
    match expr {
        Expr::StringExpr(se) => {
            if !se.is_literal_only() {
                let msg = format!("BUG: string expression segments must be empty; got {}", se);
                return Err(ParseError::General(msg));
            }
            Ok(se.to_string())
        }
        Expr::StringLiteral(s) => Ok(s.to_string()),
        _ => {
            let msg = format!("must be string expression; got {}", expr);
            return Err(ParseError::General(msg));
        }
    }
}

fn expand_function(
    symbols: &SymbolProviderRef,
    was: &Vec<WithArgExpr>,
    func: FunctionExpr,
) -> ParseResult<Expr> {
    let wa = get_with_arg_expr(symbols, was, &func.name);
    let args = expand_with_args(symbols, was, func.args)?;
    if wa.is_some() {
        return expand_with_expr_ext(symbols, was, wa.unwrap(), args);
    }
    let res = FunctionExpr {
        name: func.name,
        args,
        arg_idx_for_optimization: func.arg_idx_for_optimization,
        keep_metric_names: func.keep_metric_names,
        is_scalar: func.is_scalar,
        function: func.function,
        return_type: func.return_type,
    };
    Ok(Expr::Function(res))
}

fn expand_aggregation(
    symbols: &SymbolProviderRef,
    was: &Vec<WithArgExpr>,
    ae: AggregationExpr,
) -> ParseResult<Expr> {
    let mut ae = ae;
    let args = expand_with_args(symbols, was, ae.args)?;
    let wa = get_with_arg_expr(symbols, was, &ae.name);
    if wa.is_some() {
        // TODO:: if were in this method at all, Its a confirmed aggregate, so we should ensure
        // new name is also an aggregate
        return expand_with_expr_ext(symbols, was, wa.unwrap(), args);
    }
    ae.args = args;

    if let Some(modifier) = &ae.modifier {
        match modifier {
            AggregateModifier::By(args) => {
                let new_args = expand_modifier_args(symbols, was, &args)?;
                if args != &new_args {
                    ae.modifier = Some(AggregateModifier::By(new_args));
                }
            }
            AggregateModifier::Without(args) => {
                let new_args = expand_modifier_args(symbols, was, &args)?;
                if args != &new_args {
                    ae.modifier = Some(AggregateModifier::Without(new_args));
                }
            }
        };
    }

    Ok(Expr::Aggregation(ae))
}

fn expand_parens(
    symbols: &SymbolProviderRef,
    was: &Vec<WithArgExpr>,
    p: ParensExpr,
) -> ParseResult<Expr> {
    let mut exprs = expand_with_args(symbols, was, p.expressions)?;
    if exprs.len() == 1 {
        return Ok(exprs.remove(0));
    }
    let pe = ParensExpr::new(exprs);
    Ok(Expr::Parens(pe))
}

fn expand_rollup(
    symbols: &SymbolProviderRef,
    was: &Vec<WithArgExpr>,
    re: RollupExpr,
) -> ParseResult<Expr> {
    let mut re = re;

    re.expr = expand_with_expr_box(symbols, was, re.expr)?;
    if let Some(at) = re.at {
        let expr = expand_with_expr(symbols, was, *at)?;
        re.at = Some(Box::new(expr));
    }
    Ok(Expr::Rollup(re))
}

pub(super) fn expand_string_expr(
    symbols: &SymbolProviderRef,
    was: &Vec<WithArgExpr>,
    se: &StringExpr,
) -> ParseResult<Expr> {
    if se.is_expanded() {
        // Already expanded.
        return match se.get_literal()? {
            Some(s) => Ok(Expr::from(s.to_string())),
            None => Ok(Expr::from("".to_string())),
        };
    }

    if se.is_empty() {
        return Ok(Expr::StringLiteral("".to_string()));
    }

    // todo: calculate size
    let mut b = String::with_capacity(64);
    for token in se.iter() {
        match token {
            StringSegment::Literal(s) => {
                b.push_str(s);
                continue;
            }
            StringSegment::Ident(ident) => {
                let expr = resolve_ident(symbols, was, ident, vec![])?;
                if expr.is_none() {
                    let msg = format!("missing {} value inside string expression", ident);
                    return Err(ParseError::General(msg));
                }
                let expr = expr.unwrap();
                let value = get_expr_as_string(&expr)?;
                b.push_str(&value);
            }
        }
    }

    Ok(Expr::StringLiteral(b))
}

pub(super) fn resolve_ident(
    symbols: &SymbolProviderRef,
    was: &Vec<WithArgExpr>,
    ident: &str,
    args: Vec<Expr>,
) -> ParseResult<Option<Expr>> {
    let wa = get_with_arg_expr(symbols, was, ident);
    if wa.is_none() {
        return Ok(None);
    }
    let expr = expand_with_expr_ext(symbols, was, wa.unwrap(), args)?;
    Ok(Some(expr))
}

fn expand_modifier_args(
    symbols: &SymbolProviderRef,
    was: &Vec<WithArgExpr>,
    args: &[String],
) -> ParseResult<Vec<String>> {
    fn handle_metric_expr(expr: &Expr, arg: &String, args: &[String]) -> ParseResult<String> {
        fn error(expr: &Expr, arg: &String, args: &[String]) -> ParseResult<String> {
            let msg = format!("cannot use {expr} instead of {arg} in {}", args.join(", "));
            // let err = syntax_error(&msg, &expr.token_range, "".to_string());
            Err(ParseError::General(msg))
        }

        match expr {
            Expr::MetricExpression(me) => {
                if !me.is_only_metric_group() {
                    return error(&expr, arg, args);
                }
                Ok(me.label_filters[0].value.clone())
            }
            _ => {
                return error(&expr, arg, args);
            }
        }
    }

    if args.is_empty() {
        return Ok(vec![]);
    }

    let mut dst_args: Vec<String> = Vec::with_capacity(args.len());
    for arg in args {
        let wa = get_with_arg_expr(symbols, was, arg);
        if wa.is_none() {
            // Leave the arg as is.
            dst_args.push(arg.clone());
            continue;
        }
        let wa = wa.unwrap();
        if !wa.args.is_empty() {
            // Template funcs cannot be used inside modifier list. Leave the arg as is.
            dst_args.push(arg.clone());
            continue;
        }
        match &wa.expr {
            Expr::MetricExpression(_) => {
                let resolved = handle_metric_expr(&wa.expr, &arg, args)?;
                dst_args.push(resolved);
                continue;
            }
            Expr::Parens(pe) => {
                for p_arg in &pe.expressions {
                    let resolved = handle_metric_expr(&p_arg, &arg, args)?;
                    dst_args.push(resolved);
                }
                continue;
            }
            _ => {
                let msg = format!(
                    "cannot use {} instead of {} in {}",
                    wa.expr,
                    arg,
                    args.join(", ")
                );
                let err = syntax_error(&msg, &wa.token_range, "".to_string());
                return Err(err);
            }
        }
    }

    // Remove duplicate args from dst_args
    let m: HashSet<&String> = HashSet::from_iter(dst_args.iter());
    Ok(m.iter().map(|x| x.to_string()).collect::<Vec<String>>())
}

pub(super) fn get_with_arg_expr<'a>(
    symbols: &'a SymbolProviderRef,
    was: &'a [WithArgExpr],
    name: &str,
) -> Option<&'a WithArgExpr> {
    // Scan wes backwards, since certain expressions may override
    // previously defined expressions
    for wa in was.iter().rev() {
        if wa.name == name {
            return Some(wa);
        }
    }
    symbols.get_symbol(name)
}

fn expand_with_expr_ext(
    symbols: &SymbolProviderRef,
    was: &Vec<WithArgExpr>,
    wa: &WithArgExpr,
    args: Vec<Expr>,
) -> ParseResult<Expr> {
    if wa.args.len() != args.len() {
        if args.is_empty() {
            // Just return MetricExpr with the wa.Name name.
            return Ok(Expr::MetricExpression(MetricExpr::new(&wa.name)));
        }
        let msg = format!(
            "invalid number of args for {}(); got {}; want {}",
            wa.name,
            args.len(),
            wa.args.len()
        );
        return Err(ParseError::General(msg));
    }

    let mut was_new: Vec<WithArgExpr> = Vec::with_capacity(was.len() + args.len());

    for wa_tmp in was {
        if wa_tmp == wa {
            break;
        }
        was_new.push(wa_tmp.clone())
    }

    let arg_names = &wa.args[..];

    for (i, arg) in args.iter().enumerate() {
        was_new.push(WithArgExpr {
            name: arg_names[i].clone(),
            args: vec![],
            expr: arg.clone(),
            token_range: Default::default(),
        });
    }

    expand_with_expr(symbols, &was_new, wa.expr.clone())
}
