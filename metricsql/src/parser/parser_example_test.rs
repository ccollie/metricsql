#[cfg(test)]
mod tests {
    use std::fmt::Display;
    use std::ops::Deref;
    use crate::ast::{AggrFuncExpr, Expression, FuncExpr, MetricExpr, RollupExpr};
    use crate::parser::parse;

    fn show_aggregate(ae: AggrFuncExpr) {
        let arg = ae.args.get(0).expect("ae.args[0] should not be None");
        println!("aggr func: name={}, arg={}, modifier={}", ae.name, arg,
                 option_to_string(&ae.modifier));
        match arg.deref() {
            Expression::Function(fe) => show_func_expr(fe),
            _ => panic!("expected rollup as func expr. Got {}", arg)
        }
    }

    fn show_func_expr(fe: &FuncExpr) {
        let arg = fe.args.get(0).expect("fe.args[0] should not be None");
        println!("func: name={}, arg={}\n", fe.name, arg);
        match arg.deref() {
            Expression::Rollup(re) => show_rollup(re),
            _ => {
                panic!("expected rollup as func.args[0]. Got {}", arg);
            }
        }
    }

    fn show_rollup(re: &RollupExpr) {
        println!("rollup: expr={}, window={}\n", re.expr, option_to_string(&re.window));
        match &re.expr.deref() {
            Expression::MetricExpression(me) => show_metric_expr(me),
            _ => {
                panic!("expected MetricExpr in rollup. Got {}", re.expr);
            }
        }
    }

    fn show_metric_expr(me: &MetricExpr) {
        println!("metric: labelFilter1={}, labelFilter2={}", me.label_filters[0], me.label_filters[1]);
    }

    fn option_to_string<T: Display>(val: &Option<T>) -> String {
        match val {
            Some(v) => v.to_string(),
            None => "".to_string()
        }
    }

    #[test]
    fn example_parse() {
        let expr = parse(r#"sum(rate(foo{bar="baz"}[5m])) by (x,y)"#).unwrap();
        println!("parsed expr: {}\n", expr);

        match expr {
            Expression::Aggregation(ae) => show_aggregate(ae),
            _ => {
                panic!("parsed expression should be an AggrFuncExpr. Got {}", expr)
            }
        }
// Output:
// parsed expr: sum(rate(foo{bar="baz"}[5m])) by (x, y)
// aggr func: name=sum, arg=rate(foo{bar="baz"}[5m]), modifier=by (x, y)
// func: name=rate, arg=foo{bar="baz"}[5m]
// rollup: expr=foo{bar="baz"}, window=5m
// metric: labelFilter1=__name__="foo", labelFilter2=bar="baz"
    }

}