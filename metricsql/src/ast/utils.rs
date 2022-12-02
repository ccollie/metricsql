use crate::ast::Expression;

pub fn visit_all(e: &mut Expression, visitor: fn(&mut Expression) -> ()) {
    match e {
        Expression::BinaryOperator(be) => {
            visit_all(&mut be.left, visitor);
            visit_all(&mut be.right, visitor);
        }
        Expression::Function(fe) => {
            for arg in fe.args.iter_mut() {
                visit_all(arg, visitor)
            }
        }
        Expression::Aggregation(ae) => {
            for arg in ae.args.iter_mut() {
                visit_all(arg, visitor)
            }
        }
        Expression::Rollup(re) => visit_all(&mut re.expr, visitor),
        _ => {}
    }
    visitor(e);
}
