use crate::ast::{
    AggregationExpr, BinaryExpr, Expr, FunctionExpr, ParensExpr, RollupExpr, UnaryExpr,
    WithArgExpr, WithExpr,
};
use crate::common::TreeNodeRewriter;
use crate::parser::ParseResult;

#[allow(rustdoc::private_intra_doc_links)]
/// Remove unnecessary parentheses from an `Expr`s
pub struct ParensRemover {}

impl TreeNodeRewriter for ParensRemover {
    type N = Expr;

    fn mutate(&mut self, node: Self::N) -> ParseResult<Self::N> {
        Ok(remove_parens_expr(node))
    }
}

impl ParensRemover {
    pub fn new() -> Self {
        ParensRemover {}
    }

    pub fn remove_parens(&self, e: Expr) -> Expr {
        remove_parens_expr(e)
    }
}

fn unnest_parens(pe: &mut ParensExpr) {
    while pe.expressions.len() == 1 {
        if let Some(Expr::Parens(pe2)) = pe.expressions.get_mut(0) {
            *pe = pe2.clone(); // todo: take
        } else {
            break;
        }
    }
}

fn remove_parens_args(args: Vec<Expr>) -> Vec<Expr> {
    args.into_iter()
        .map(|arg| remove_parens_expr(arg))
        .collect()
}

// remove_parens_expr removes parensExpr for (Expr) case.
pub fn remove_parens_expr(e: Expr) -> Expr {
    return match e {
        Expr::Rollup(re) => Expr::Rollup(RollupExpr {
            expr: Box::new(remove_parens_expr(*re.expr)),
            at: re.at.map(|at| Box::new(remove_parens_expr(*at))),
            window: re.window,
            step: re.step,
            offset: re.offset,
            inherit_step: re.inherit_step,
        }),
        Expr::BinaryOperator(be) => Expr::BinaryOperator(BinaryExpr {
            left: Box::new(remove_parens_expr(*be.left)),
            right: Box::new(remove_parens_expr(*be.right)),
            op: be.op,
            modifier: be.modifier,
        }),
        Expr::UnaryOperator(ue) => Expr::UnaryOperator(UnaryExpr {
            expr: Box::new(remove_parens_expr(*ue.expr)),
        }),
        Expr::Aggregation(ae) => Expr::Aggregation(AggregationExpr {
            name: ae.name,
            function: ae.function,
            args: ae
                .args
                .into_iter()
                .map(|arg| remove_parens_expr(arg))
                .collect(),
            modifier: ae.modifier,
            limit: ae.limit,
            keep_metric_names: ae.keep_metric_names,
        }),
        Expr::Function(fe) => Expr::Function(FunctionExpr {
            name: fe.name,
            args: remove_parens_args(fe.args),
            keep_metric_names: fe.keep_metric_names,
            function: fe.function,
            return_type: fe.return_type,
        }),
        Expr::Parens(pe) => {
            let mut pe = pe;
            unnest_parens(&mut pe);
            let len = pe.len();
            let mut args = remove_parens_args(pe.expressions);
            if len == 1 {
                return args.remove(0);
            }
            Expr::Parens(ParensExpr { expressions: args })
        }
        Expr::With(with) => Expr::With(WithExpr {
            expr: Box::new(remove_parens_expr(*with.expr)),
            was: with
                .was
                .into_iter()
                .map(|wa| WithArgExpr {
                    name: wa.name,
                    args: vec![],
                    expr: remove_parens_expr(wa.expr),
                    token_range: Default::default(),
                })
                .collect(),
        }),
        _ => e,
    };
}

#[cfg(test)]
mod tests {
    use crate::ast::{Expr, ParensExpr};
    use crate::optimizer::parens_remover::remove_parens_expr;

    #[test]
    fn test_remove_parens_expr() {
        let empty_parens = Expr::Parens(ParensExpr::new(vec![]));
        let actual = Expr::Parens(ParensExpr::new(vec![empty_parens.clone()]));

        let result = remove_parens_expr(actual);
        assert_expr_eq(&empty_parens, &result);
    }

    fn assert_expr_eq(expected: &Expr, actual: &Expr) {
        assert_eq!(
            expected, actual,
            "expected: \n{}\n but got: \n{}",
            expected, actual
        );
    }
}
