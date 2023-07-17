use crate::ast::Expr;

/// Convert 'num cmpOp query' expression to `query reverseCmpOp num` expression
/// like Prometheus does. For instance, `0.5 < foo` must be converted to `foo > 0.5`
/// in order to return valid values for `foo` that are bigger than 0.5.
pub fn adjust_comparison_ops(expr: &mut Expr) {
    match expr {
        Expr::Aggregation(agg) => {
            for arg in agg.args.iter_mut() {
                adjust_comparison_ops(arg);
            }
        }
        Expr::BinaryOperator(be) => {
            adjust_comparison_ops(&mut be.left);
            adjust_comparison_ops(&mut be.right);
            be.adjust_comparison_op();
        }
        Expr::Function(fe) => {
            for arg in fe.args.iter_mut() {
                adjust_comparison_ops(arg);
            }
        }
        Expr::Parens(pe) => {
            for e in pe.expressions.iter_mut() {
                adjust_comparison_ops(e);
            }
        }
        Expr::Rollup(re) => {
            adjust_comparison_ops(&mut re.expr);
            match re.at {
                Some(ref mut at) => {
                    adjust_comparison_ops(at.as_mut());
                }
                _ => {}
            }
        }
        _ => {}
    }
}
