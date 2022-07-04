use crate::parse;
use crate::types::Expression;

pub fn visit_all(e: &Expression, visitor: &fn(&Expression) -> ()) {
    match e {
        Expression::BinaryOperator(be) => {
            visit_all(&be.left, visitor);
            visit_all(&be.right, visitor);
        },
        Expression::Function(fe) => {
            for arg in fe.args {
                visit_all(&arg, visitor)
            }
        },
        Expression::Aggregation(ae) => {
            for arg in ae.args {
                visit_all(&arg, visitor)
            }
        },
        Expression::Rollup(re) => {
            visit_all(&re.expr, visitor)
        }
        _ => {}
    }
    visitor(e);
}
// ExpandWithExprs expands WITH expressions inside q and returns the resulting
// PromQL without WITH expressions.
pub fn expand_with_exprs(q: &str) -> Result<String, Error> {
    let e = parse(q)?;
    Ok( format!("{}", e) )
}