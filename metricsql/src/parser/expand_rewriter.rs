use std::collections::HashSet;
use crate::ast::*;
use crate::ast::expr_rewriter::{ExprRewritable, ExprRewriter, RewriteRecursion};
use crate::ast::label_filter_expr::LabelFilterExpr;
use crate::parser::{ParseError, ParseResult};
use crate::prelude::segmented_string::SegmentedString;

pub struct ExpandRewriter<F> {
    resolve: F
}

impl<F> ExprRewriter for ExpandRewriter<F> {
    fn pre_visit(&mut self, expr: &Expression) -> ParseResult<RewriteRecursion> {
        use Expression::*;

        match expr {
            BinaryOperator(_) |
            Function(_) |
            With(_) |
            Aggregation(_) => Ok(RewriteRecursion::Mutate),
            MetricExpression(me) => {
                if me.is_expanded() {
                    Ok(RewriteRecursion::Skip)
                } else {
                    Ok(RewriteRecursion::Mutate)
                }
            },
            _ => Ok(RewriteRecursion::Skip)
        }
    }

    fn mutate(&mut self, expr: Expression) -> ParseResult<Expression> {
        use Expression::*;

        match expr {
            BinaryOperator(_) => self.expand_binary(expr),
            Function(_) => self.expand_function(&expr),
            Aggregation(_) => self.expand_aggregation(expr),
            MetricExpression(_) => self.expand_metric_expr(expr),
            _ => Ok(expr)
        }
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


impl<F> ExpandRewriter<F>
    where F: FnMut(&str) -> Option<Expression>
{
    pub fn new(f: F) -> Self {
        Self { resolve: f }
    }

    pub fn expand(&self, expr: Expression) -> ParseResult<Expression> {
        let mut expander = ExpandRewriter { resolve: &self.resolve };
        expr.rewrite(&mut expander)
    }

    fn expand_binary(&self, expr: Expression) -> ParseResult<Expression> {
        let be = expect_variant!(expr, Expression::BinaryOperator);

        let mut group_modifier: Option<GroupModifier> = None;
        let mut join_modifier: Option<JoinModifier> = None;

        if let Some(modifier) = &be.group_modifier {
            let labels = self.expand_modifier_args(&modifier.labels)?;
            if labels != modifier.labels {
                group_modifier = Some(GroupModifier::new(modifier.op, labels));
            }
        }

        if let Some(ref modifier) = &be.join_modifier {
            let labels = self.expand_modifier_args(&modifier.labels)?;
            if labels != modifier.labels {
                join_modifier = Some(JoinModifier::new(modifier.op, labels));
            }
        }

        Ok(Expression::BinaryOperator(be))
    }

    fn resolve_segmented_string(&self, ss: &SegmentedString) -> ParseResult<String> {
        if !ss.is_expanded() {
            let expanded = ss.resolve(&self.resolve)?;
            return Ok(expanded)
        }
        Ok(ss.to_string())
    }

    fn resolve_metric_name_as_string(&self, me: &MetricExpr, arg: &str) -> ParseResult<String> {
        if !me.is_only_metric_group() {
            let msg = format!("cannot use {:?} instead of {}", me, arg);
            return Err(ParseError::WithExprExpansionError(msg));
        }
        let value = self.resolve_segmented_string(&me.label_filters[0].value)?;
        Ok(value)
    }

    fn resolve_strings(&self, expr: &Expression, arg: &str) -> ParseResult<Option<Vec<String>>> {

        let handle_metric_expr = |me: &MetricExpr| -> ParseResult<Option<Vec<String>>> {
            let str = self.resolve_metric_name_as_string(me, arg)?;
            return Ok(Some(vec![str]));
        };

        match expr {
            Expression::String(se) => Ok(Some(vec![se.to_string()])),
            Expression::MetricExpression(me) => handle_metric_expr(me),
            Expression::Parens(pe) => {
                let mut res = Vec::with_capacity(pe.len());
                for (i, _arg) in pe.expressions.iter().enumerate() {
                    let name = format!("{}[{}]", arg, i);
                    if let Some(expanded) = self.resolve_strings(_arg, &name)? {
                        res.extend_from_slice(&expanded);
                    } else {
                        // not found. todo error
                    }
                }
                Ok(Some(res))
            }
            _ => {
                let msg = "expected metric selector as WITH argument".to_string();
                return Err(ParseError::General(msg));
            }
        }
    }

    fn resolve_string(&self, expr: &Expression, name: &str) -> ParseResult<Option<String>> {
        let mut values = self.resolve_strings(expr, name)?;
        if let Some(vals) = values.as_mut() {
            if vals.len() == 1 {
                return Ok( Some(vals.remove(0)) )
            } else if vals.len() > 1 {
                // todo: err
            }
        }
        return Ok(None)
    }

    fn expand_aggregation(&self, expr: Expression) -> ParseResult<Expression> {
        let aggregate = expect_variant!(expr, Expression::Aggregation);

        let wa = self.resolve(&aggregate.name);
        if let Some(wae) = wa {
            // if were in this method at all, Its a confirmed aggregate, so we should ensure
            // new name is also an aggregate
            if let Some(name) = self.resolve_string(wae, &aggregate.name)? {
                todo!()
            }
        }

        let mut new_modifier: Option<AggregateModifier> = None;
        if let Some(modifier) = &aggregate.modifier {
            let new_args = self.expand_modifier_args(&modifier.args)?;
            if new_args != modifier.args {
                new_modifier = Some(AggregateModifier::new(modifier.op.clone(), new_args));
            }
        }

        if new_modifier.is_some() {
            let mut aggr = aggregate.clone();
            aggr.modifier = new_modifier;
            return Ok(Expression::Aggregation(aggr));
        }

        Ok(Expression::Aggregation(aggregate))
    }

    fn expand_metric_expr(&self, expr: Expression) -> ParseResult<Expression> {

        let me = expect_variant!(expr, Expression::MetricExpression);
        let mut me = self.expand_metric_labels(&me)?;

        if me.is_only_metric_group() {
            Ok(Expression::MetricExpression(me))
        }

        let resolved_name = self.resolve_segmented_string(&me.label_filters[0].value)
            .or_else(|x| {
                let msg = format!("cannot expand {:?} to non-metric expression", me);
                return Err(ParseError::WithExprExpansionError(msg));
            })?;

        me.add_tag(NAME_LABEL, &resolved_name);

        if !me.label_filters.is_empty() {
            let msg = format!("BUG: wme.label_filters must be empty; got {:?}", me.label_filters);
            return Err(ParseError::WithExprExpansionError(msg));
        }

        // todo: avoid clone
        Ok(Expression::MetricExpression(me.clone()))
    }

    fn expand_metric_labels(&self, me: &MetricExpr) -> ParseResult<MetricExpr> {
        // Populate me.label_filters
        let mut me_new = MetricExpr::default();
        if me.label_filters.len() > 0 {
            me_new.label_filters.extend_from_slice(&me.label_filters);
        }

        for lfe in me_new.label_filters.iter_mut() {
                if lfe.is_resolved() {
                    continue;
                }
                // Expand lfe.label into Vec<LabelFilter>.
                let wa = (self.resolve)(&lfe.label);
                if wa.is_none() {
                    let msg = format!("missing {} MetricExpr", lfe.label);
                    return Err(ParseError::WithExprExpansionError(msg));
                }

                let value = self.resolve_segmented_string(&lfe.value)?;
                lfe.value.set_from_string(value);
            }


        me_new.label_filters.clear();
        remove_duplicate_label_filters(&mut me_new.label_filters);

        return Ok(me_new);
    }

    fn expand_modifier_args(&self, args: &[String]) -> ParseResult<Vec<String>> {
        if args.is_empty() {
            return Ok(vec![]);
        }

        let mut dst_args: Vec<String> = Vec::with_capacity(1);
        for arg in args.iter() {
            match self.resolve(arg) {
                None => {
                    // Leave the arg as is.
                    dst_args.push(arg.to_string());
                    continue;
                }
                Some(wa) => {
                    dst_args.push(wa)
                }
            }
        }

        Ok(dst_args)
    }
}

pub(crate) fn expand_with_expr(was: &Vec<WithArgExpr>,
                               expr: Expression) -> ParseResult<Expression> {
    todo!()
}

fn remove_duplicate_label_filters(filters: &mut Vec<LabelFilterExpr>) {
    let mut set: HashSet<String> = HashSet::with_capacity(filters.len());
    filters.retain(|filters| {
        let key = filters.to_string();
        if !set.contains(&key) {
            set.insert(key);
            true
        } else {
            false
        }
    })
}

/// type_changed checks for the case where the type of an expression changes, in our case
/// because of a binary op simplification (eg. a > b or "foo" + "bar") i.o.w binary op to
/// scalar or string
fn type_changed(before: &Expression, after: &Expression) -> bool {
    let (first_type, second_type) = (before.variant_name(), after.variant_name());
    first_type != second_type
}