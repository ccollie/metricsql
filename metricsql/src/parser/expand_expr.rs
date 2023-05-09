use crate::ast::*;
use crate::common::{AggregateModifier, GroupModifier, JoinModifier, LabelFilter, LabelFilterOp, NAME_LABEL, StringExpr};
use crate::parser::{ParseError, ParseResult};
use std::collections::HashSet;

pub(crate) fn expand_with_expr<'a>(
    expr: Expr,
    resolver: impl Fn(&str) -> Option<Expr> + 'a,
) -> ParseResult<Expr> {
    let mut rewriter = ExpandRewriter {
        resolve: Box::new(resolver),
    };
    rewriter.expand(expr)
}

pub struct ExpandRewriter<'a> {
    resolve: Box<dyn Fn(&str) -> Option<Expr> + 'a>,
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
    pub fn expand(&mut self, expr: Expr) -> ParseResult<Expr> {
        use Expr::*;

        match expr {
            BinaryOperator(be) => self.expand_binary(be),
            Function(_) => self.expand_function(expr),
            Aggregation(_) => self.expand_aggregation(expr),
            MetricExpression(_) => self.expand_metric_expr(expr),
            StringExpr(str) => {
                let ss = self.resolve_string_expr(&str)?;
                Ok(StringLiteral(ss))
            },
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

    fn expand_binary(&self, be: BinaryExpr) -> ParseResult<Expr> {
        let mut be = be;

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

        Ok(Expr::BinaryOperator(be))
    }

    fn resolve_string_expr(&self, ss: &StringExpr) -> ParseResult<String> {
        ss.resolve(|ident| self.resolve_string(ident))
    }

    fn resolve_metric_name_as_string(&self, me: &MetricExpr, arg: &str) -> ParseResult<String> {
        if !me.is_only_metric_group() {
            let msg = format!("cannot use {:?} instead of {}", me, arg);
            return Err(ParseError::WithExprExpansionError(msg));
        }
        let value = self.resolve_string_expr(&me.label_filter_expressions[0].value)?;
        Ok(value)
    }

    fn resolve_strings(&self, expr: &Expr, arg: &str) -> ParseResult<Option<Vec<String>>> {
        let handle_metric_expr = |me: &MetricExpr| -> ParseResult<Option<Vec<String>>> {
            let str = self.resolve_metric_name_as_string(me, arg)?;
            return Ok(Some(vec![str]));
        };

        match expr {
            Expr::StringExpr(se) => {
                let value = self.resolve_string_expr(se)?;
                return Ok(Some(vec![value]))
            },
            Expr::MetricExpression(me) => return handle_metric_expr(me),
            Expr::Rollup(rollup) => {
                if rollup.for_subquery() {
                    let msg = format!(
                        "Cannot substitute {:?} for {} string in {:?}", rollup, arg, expr
                    );
                    return Err(ParseError::General(msg));
                }
                match rollup.expr.as_ref() {
                    Expr::MetricExpression(me) => return handle_metric_expr(me),
                    _ => {}
                }
            }
            Expr::Parens(pe) => {
                let mut res = Vec::with_capacity(pe.len());
                for (i, _arg) in pe.expressions.iter().enumerate() {
                    let name = format!("{}[{}]", _arg, i);
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
        expr: &Expr,
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

    fn expand_aggregation(&self, expr: Expr) -> ParseResult<Expr> {
        let aggregate = expect_variant!(expr, Expr::Aggregation);

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
            return Ok(Expr::Aggregation(aggr));
        }

        Ok(Expr::Aggregation(aggregate))
    }

    fn expand_metric_expr(&self, expr: Expr) -> ParseResult<Expr> {
        let me = expect_variant!(expr, Expr::MetricExpression);
        if me.is_resolved() {
            return Ok(Expr::MetricExpression(me));
        }
        let mut me = self.expand_metric_labels(&me)?;

        if me.is_only_metric_group() {
            return Ok(Expr::MetricExpression(me));
        }

        let resolved_name = self
            .resolve_string_expr(&me.label_filter_expressions[0].value)
            .or_else(|_x| {
                let msg = format!("cannot expand {:?} to non-metric Expr", me);
                return Err(ParseError::WithExprExpansionError(msg));
            })?;

        Self::add_metric_tag(&mut me, NAME_LABEL, resolved_name);

        if !me.label_filter_expressions.is_empty() {
            let msg = format!(
                "BUG: wme.label_filter_expressions must be empty; got {:?}",
                me.label_filter_expressions
            );
            return Err(ParseError::WithExprExpansionError(msg));
        }

        // todo: avoid clone
        Ok(Expr::MetricExpression(me.clone()))
    }

    fn expand_metric_labels(&self, me: &MetricExpr) -> ParseResult<MetricExpr> {
        // Populate me.label_filters
        let mut me_new = MetricExpr::default();
        if !me.label_filters.is_empty() {
            // todo: Take ownership of me.filters
            me_new.label_filters = me.label_filters.clone();
        }

        for lfe in me.label_filter_expressions.iter() {
            if lfe.is_resolved() {
                me_new.label_filters.push(lfe.to_label_filter()?);
                continue;
            }
            // Expand lfe.label into Vec<LabelFilter>.
            let wa = (self.resolve)(&lfe.label);
            if wa.is_none() {
                let msg = format!("missing {} MetricExpr", lfe.label);
                return Err(ParseError::WithExprExpansionError(msg));
            }

            let label = self.resolve_string(&lfe.label)?;
            let value = self.resolve_string_expr(&lfe.value)?;
            me_new.label_filters.push(LabelFilter {
                op: lfe.op,
                label: label.unwrap_or(lfe.label.to_string()),
                value
            });
        }

        remove_duplicate_label_filters(&mut me_new.label_filters);

        return Ok(me_new);
    }

    fn add_metric_tag<S: Into<String>>(selector: &mut MetricExpr, name: S, value: String) {
        let name_str = name.into();
        for label in selector.label_filters.iter_mut() {
            if label.label == name_str {
                label.value.clear();
                label.value.push_str(&value);
                return;
            }
        }
        selector.label_filters.push(LabelFilter {
            op: LabelFilterOp::Equal,
            label: name_str,
            value,
        });
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

    fn expand_function(&mut self, expr: Expr) -> ParseResult<Expr> {
        let mut func = expect_variant!(expr, Expr::Function);
        if let Some(_wa) = self.resolve_string(&func.name)? {
            // todo: replace name. Validate that new name is valid function of same type as orig
            todo!()
        }
        func.args = self.expand_args(&func.args)?;

        return Ok(Expr::Function(func));
    }

    fn expand_args(&mut self, args: &Vec<Expr>) -> ParseResult<Vec<Expr>> {
        let mut result: Vec<Expr> = Vec::with_capacity(args.len());
        for expr in args.into_iter() {
            result.push(self.expand(expr.clone())?);
        }
        Ok(result)
    }
}

fn remove_duplicate_label_filters(filters: &mut Vec<LabelFilter>) {
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
