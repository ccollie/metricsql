use std::collections::{HashMap, HashSet};
use std::hash::Hasher;
use std::ops::Deref;

use xxhash_rust::xxh3::Xxh3;

use crate::ast::{
    AggregationExpr, BinaryExpr, Expr, FunctionExpr, MetricExpr, ParensExpr, RollupExpr,
    WithArgExpr, WithExpr,
};
use crate::common::{
    AggregateModifier, LabelFilter, Labels, StringExpr, StringSegment, VectorMatchCardinality,
    VectorMatchModifier,
};
use crate::parser::symbol_provider::SymbolProviderRef;
use crate::parser::{syntax_error, ParseError, ParseResult};
use crate::prelude::InterpolatedSelector;

pub(super) fn expand_with_expr(
    symbols: &SymbolProviderRef,
    was: &Vec<WithArgExpr>,
    expr: Expr,
) -> ParseResult<Expr> {
    use Expr::*;

    //print!("expanding {} => ", &expr);

    let res = match expr {
        BinaryOperator(be) => expand_binary_operator(symbols, was, be),
        Function(fe) => expand_function(symbols, was, fe),
        Aggregation(ae) => expand_aggregation(symbols, was, ae),
        MetricExpression(me) => expand_metric_expression(symbols, was, me),
        Parens(pe) => expand_parens(symbols, was, pe),
        Rollup(re) => expand_rollup(symbols, was, re),
        StringExpr(se) => expand_string_expr(symbols, was, &se),
        With(we) => expand_with(symbols, was, we),
        WithSelector(ws) => expand_with_selector_expression(symbols, was, ws),
        _ => Ok(expr),
    }?;

    // println!("{}", res);

    Ok(res)
}

fn expand_with(
    symbols: &SymbolProviderRef,
    was: &Vec<WithArgExpr>,
    we: WithExpr,
) -> ParseResult<Expr> {
    let mut was_new = Vec::with_capacity(was.len() + we.was.len());
    was_new.append(&mut was.clone());
    was_new.append(&mut we.was.clone());

    expand_with_expr(symbols, &was_new, *we.expr)
}

pub fn should_expand(expr: &Expr) -> bool {
    use Expr::*;

    match expr {
        StringLiteral(_) | Number(_) | Duration(_) => false,
        BinaryOperator(be) => should_expand(&be.left) || should_expand(&be.right),
        MetricExpression(me) => me.is_only_metric_group(),
        StringExpr(se) => !se.is_expanded(),
        Parens(pe) => !pe.expressions.is_empty() && pe.expressions.iter().any(should_expand),
        Rollup(re) => {
            if should_expand(&re.expr) {
                return true;
            }
            if let Some(at) = &re.at {
                return should_expand(at);
            }
            false
        }
        WithSelector(_) => true,
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

    if let Some(modifier) = &mut be.modifier {
        if let Some(matching) = &mut modifier.matching {
            match matching {
                VectorMatchModifier::On(labels) => {
                    let new_labels = expand_modifier_args(symbols, was, labels.as_ref())?;
                    if labels != &new_labels {
                        *labels = Labels::from_iter(new_labels);
                    }
                }
                VectorMatchModifier::Ignoring(labels) => {
                    let new_labels = expand_modifier_args(symbols, was, labels.as_ref())?;
                    if labels != &new_labels {
                        *labels = Labels::from_iter(new_labels);
                    }
                }
            }

            match &modifier.card {
                VectorMatchCardinality::ManyToOne(labels) => {
                    let new_labels = expand_modifier_args(symbols, was, labels.as_ref())?;
                    modifier.card =
                        VectorMatchCardinality::ManyToOne(Labels::from_iter(new_labels));
                }
                VectorMatchCardinality::OneToMany(labels) => {
                    let new_labels = expand_modifier_args(symbols, was, labels.as_ref())?;
                    modifier.card =
                        VectorMatchCardinality::ManyToOne(Labels::from_iter(new_labels));
                }
                _ => {}
            }
        }
    }

    Ok(Expr::BinaryOperator(be))
}

fn remove_dupes(filters: &mut Vec<LabelFilter>) {
    fn get_hash(hasher: &mut Xxh3, filter: &LabelFilter) -> u64 {
        hasher.reset();
        hasher.write(filter.label.as_bytes());
        hasher.write(filter.op.as_str().as_bytes());
        hasher.finish()
    }

    let mut hasher = Xxh3::new();
    let mut hash_map: HashMap<u64, bool> = HashMap::with_capacity(filters.len());

    for i in (0..filters.len()).rev() {
        let hash = get_hash(&mut hasher, &filters[i]);
        if let std::collections::hash_map::Entry::Vacant(e) = hash_map.entry(hash) {
            e.insert(true);
        } else {
            filters.remove(i);
        }
    }
}

fn merge_selectors(dst: &mut MetricExpr, src: &mut MetricExpr) {
    src.label_filters.retain(|x| x.is_metric_name_filter());

    let mut items = src.label_filters.drain(..).collect::<Vec<_>>();
    dst.label_filters.append(&mut items);

    remove_dupes(&mut dst.label_filters);
}

fn expand_with_selector_expression(
    symbols: &SymbolProviderRef,
    was: &Vec<WithArgExpr>,
    me: InterpolatedSelector,
) -> ParseResult<Expr> {
    fn handle_expanded(dst: &MetricExpr, src: &mut MetricExpr) -> ParseResult<Expr> {
        let mut dst = dst.clone();
        merge_selectors(&mut dst, src);
        Ok(Expr::MetricExpression(dst))
    }

    if me.is_resolved() {
        // Already expanded.
        let filters = me.to_label_filters()?;
        let res = MetricExpr::with_filters(filters);
        return Ok(Expr::MetricExpression(res));
    }

    let mut new_selector: MetricExpr = MetricExpr::default();

    // Populate me.LabelFilters
    for lfe in me.matchers {
        if lfe.value.is_empty() || lfe.is_variable() {
            // Expand lfe.Label into vec<LabelFilter>.
            // we have something like: foo{commonFilters} and we want to expand it into
            // foo{bar="bax", job="trace"}
            let label = lfe.name();
            let wa = get_with_arg_expr(symbols, was, &label);
            if wa.is_none() {
                let msg = format!("missing {label} value inside {new_selector}");
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
                    "{label} must be filters expression inside {new_selector}; got {e_new}"
                );
                return Err(ParseError::General(msg));
            }
            let mut labels = wme.unwrap().label_filters.clone();

            new_selector.label_filters.append(&mut labels);

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

    remove_dupes(&mut new_selector.label_filters);
    new_selector.sort_filters();

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
        Expr::Rollup(re) => match re.expr.deref() {
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
                let msg =
                    format!("cannot expand {new_selector} to non-metric expression {expanded}",);
                return Err(ParseError::General(msg));
            }
            Ok(expanded)
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
            Err(ParseError::General(msg))
        }
    }
}

fn expand_metric_expression(
    symbols: &SymbolProviderRef,
    was: &Vec<WithArgExpr>,
    me: MetricExpr,
) -> ParseResult<Expr> {
    if !me.is_only_metric_group() {
        // Already expanded.
        return Ok(Expr::MetricExpression(me));
    }

    let k = &me.label_filters[0].value;
    if let Some(wa) = get_with_arg_expr(symbols, was, k) {
        return expand_with_expr_ext(symbols, was, wa, vec![]);
    }

    Ok(Expr::MetricExpression(me))
}

fn expand_function(
    symbols: &SymbolProviderRef,
    was: &Vec<WithArgExpr>,
    func: FunctionExpr,
) -> ParseResult<Expr> {
    let args = expand_with_args(symbols, was, func.args)?;
    if let Some(wa) = get_with_arg_expr(symbols, was, &func.name) {
        return expand_with_expr_ext(symbols, was, wa, args);
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
    if let Some(wa) = get_with_arg_expr(symbols, was, &ae.name) {
        // TODO:: if were in this method at all, Its a confirmed aggregate, so we should ensure
        // new name is also an aggregate
        return expand_with_expr_ext(symbols, was, wa, args);
    }
    ae.args = args;

    if let Some(modifier) = &ae.modifier {
        match modifier {
            AggregateModifier::By(args) => {
                let new_args = expand_modifier_args(symbols, was, args)?;
                if args != &new_args {
                    ae.modifier = Some(AggregateModifier::By(new_args));
                }
            }
            AggregateModifier::Without(args) => {
                let new_args = expand_modifier_args(symbols, was, args)?;
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

fn expand_string_expr(
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
                if let Some(expr) = resolve_ident(symbols, was, ident, vec![])? {
                    let value = get_expr_as_string(&expr)?;
                    b.push_str(&value);
                } else {
                    let msg = format!("missing {} value inside string expression", ident);
                    return Err(ParseError::General(msg));
                }
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
    if let Some(wa) = get_with_arg_expr(symbols, was, ident) {
        let expr = expand_with_expr_ext(symbols, was, wa, args)?;
        return Ok(Some(expr));
    }

    Ok(None)
}

fn expand_modifier_args(
    symbols: &SymbolProviderRef,
    was: &[WithArgExpr],
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
                    return error(expr, arg, args);
                }
                Ok(me.label_filters[0].value.clone())
            }
            _ => error(expr, arg, args),
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
                let resolved = handle_metric_expr(&wa.expr, arg, args)?;
                dst_args.push(resolved);
                continue;
            }
            Expr::Parens(pe) => {
                for p_arg in &pe.expressions {
                    let resolved = handle_metric_expr(p_arg, arg, args)?;
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

fn get_with_arg_expr<'a>(
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
