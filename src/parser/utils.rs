use crate::Expression;

pub fn visit_all(e: &Expression, visitor: fn(&Expression) -> void) {
    match e {
        Expression::BinaryOperator(be) => {
            visit_all(be.left, visitor);
            visit_all(be.right, visitor);
        },
        Expression::Function(fe) => {
            for arg in fe.args {
                visit_all(arg, visitor)
            }
        },
        Expression::Aggregation(ae) => {
            for arg in ae.args {
                visit_all(arg, visitor)
            }
        },
        Expression::Rollup(re) => {
            visit_all(re.expr, visitor)
        }
    }
    visitor(e);
}