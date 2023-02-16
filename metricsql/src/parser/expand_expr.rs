use crate::ast::*;
use crate::common::{AggregateModifier, GroupModifier, JoinModifier, NAME_LABEL};
use crate::parser::{ParseError, ParseResult};
use std::collections::HashSet;

pub(crate) fn expand_with_expr<'a>(
    expr: Expression,
    resolver: impl Fn(&str) -> Option<Expression> + 'a,
) -> ParseResult<Expression> {
    let mut rewriter = ExpandRewriter {
        resolve: Box::new(resolver),
    };
    rewriter.expand(expr)
}

pub struct ExpandRewriter<'a> {
    resolve: Box<dyn Fn(&str) -> Option<Expression> + 'a>,
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

impl<'a> ExpandRewriter<'a> {
    pub fn expand(&mut self, expr: Expression) -> ParseResult<Expression> {
        use Expression::*;

        match expr {
            BinaryOperator(_) => self.expand_binary(expr),
            Function(_) => self.expand_function(expr),
            Aggregation(_) => self.expand_aggregation(expr),
            MetricExpression(_) => self.expand_metric_expr(expr),
            _ => Ok(expr),
        }
    }

    fn resolve_string(&self, value: &str) -> ParseResult<Option<String>> {
        if let Some(resolved) = (self.resolve)(value) {
            self.resolve_string_internal(&resolved, value)
        } else {
            Ok(None)
        }
    }
    fn expand_binary(&self, expr: Expression) -> ParseResult<Expression> {
        let mut be = expect_variant!(expr, Expression::BinaryOperator);

        if let Some(modifier) = &be.group_modifier {
            let labels = self.expand_modifier_args(&modifier.labels)?;
            if labels != modifier.labels {
                be.group_modifier = Some(GroupModifier::new(modifier.op, labels));
            }
        }

        if let Some(ref modifier) = &be.join_modifier {
            let labels = self.expand_modifier_args(&modifier.labels)?;
            if labels != modifier.labels {
                be.join_modifier = Some(JoinModifier::new(modifier.op, labels));
            }
        }

        Ok(Expression::BinaryOperator(be))
    }

    fn resolve_string_expr(&self, ss: &StringExpr) -> ParseResult<String> {
        if !ss.is_resolved() {
            let capacity = ss.estimate_result_capacity();
            let mut result = String::with_capacity(capacity);
            for segment in ss.iter() {
                match segment {
                    StringSegment::Literal(lit) => result.push_str(lit.as_str()),
                    StringSegment::Ident(identifier) => {
                        let expanded = self.resolve_string(identifier)?;
                        if let Some(ident_value) = expanded {
                            result.push_str(&ident_value);
                        } else {
                            let msg = format!(
                                "unknown identifier {:?} in string expression of {}",
                                identifier,
                                ss.to_string()
                            );
                            return Err(ParseError::WithExprExpansionError(msg));
                        }
                    } //
                }
            }
            return Ok(result);
        }
        Ok(ss.to_string())
    }

    fn resolve_metric_name_as_string(&self, me: &MetricExpr, arg: &str) -> ParseResult<String> {
        if !me.is_only_metric_group() {
            let msg = format!("cannot use {:?} instead of {}", me, arg);
            return Err(ParseError::WithExprExpansionError(msg));
        }
        let value = self.resolve_string_expr(&me.label_filters[0].value)?;
        Ok(value)
    }

    fn resolve_strings(&self, expr: &Expression, arg: &str) -> ParseResult<Option<Vec<String>>> {
        let handle_metric_expr = |me: &MetricExpr| -> ParseResult<Option<Vec<String>>> {
            let str = self.resolve_metric_name_as_string(me, arg)?;
            return Ok(Some(vec![str]));
        };

        match expr {
            Expression::String(se) => return Ok(Some(vec![se.to_string()])),
            Expression::MetricExpression(me) => return handle_metric_expr(me),
            Expression::Rollup(rollup) => {
                if rollup.for_subquery() {
                    let msg = format!(
                        "Cannot substitute {:?} for {} string in {:?}",
                        rollup, arg, expr
                    );
                    return Err(ParseError::General(msg));
                }
                match rollup.expr.as_ref() {
                    Expression::MetricExpression(me) => return handle_metric_expr(me),
                    _ => {}
                }
            }
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
                return Ok(Some(res));
            }
            _ => {}
        }
        let msg = "expected metric selector as WITH argument".to_string();
        return Err(ParseError::General(msg));
    }

    fn resolve_string_internal(
        &self,
        expr: &Expression,
        name: &str,
    ) -> ParseResult<Option<String>> {
        let mut values = self.resolve_strings(expr, name)?;
        if let Some(vals) = values.as_mut() {
            if vals.len() == 1 {
                return Ok(Some(vals.remove(0)));
            } else if vals.len() > 1 {
                // todo: err
            }
        }
        return Ok(None);
    }

    fn expand_aggregation(&self, expr: Expression) -> ParseResult<Expression> {
        let aggregate = expect_variant!(expr, Expression::Aggregation);

        let wa = self.resolve_string(&aggregate.name)?;
        if let Some(_wae) = wa {
            // TODO:: if were in this method at all, Its a confirmed aggregate, so we should ensure
            // new name is also an aggregate
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
            return Ok(Expression::MetricExpression(me));
        }

        let resolved_name = self
            .resolve_string_expr(&me.label_filters[0].value)
            .or_else(|_x| {
                let msg = format!("cannot expand {:?} to non-metric expression", me);
                return Err(ParseError::WithExprExpansionError(msg));
            })?;

        me.add_tag(NAME_LABEL, &resolved_name);

        if !me.label_filters.is_empty() {
            let msg = format!(
                "BUG: wme.label_filters must be empty; got {:?}",
                me.label_filters
            );
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

            let value = self.resolve_string_expr(&lfe.value)?;
            lfe.value = StringExpr::new(value);
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
            let resolved = self.resolve_string(arg)?;
            dst_args.push(resolved.unwrap_or(arg.to_string()))
        }

        Ok(dst_args)
    }

    fn expand_function(&mut self, expr: Expression) -> ParseResult<Expression> {
        let mut func = expect_variant!(expr, Expression::Function);
        if let Some(_wa) = self.resolve_string(&func.with_name)? {
            // todo: replace name. Validate that new name is valid function of same type as orig
            todo!()
        }
        func.args = self.expand_args(&func.args)?;

        return Ok(Expression::Function(func));
    }

    fn expand_args(&mut self, args: &Vec<BExpression>) -> ParseResult<Vec<BExpression>> {
        let mut result: Vec<BExpression> = Vec::with_capacity(args.len());

        for arg in args.iter() {
            let expanded = self.expand(*arg.clone())?;
            result.push(Box::new(expanded));
        }

        Ok(result)
    }
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
