#[cfg(test)]
mod tests {
    use crate::ast::{AggregationExpr, Expr, FunctionExpr, MetricExpr, RollupExpr};
    use crate::parser::parse;
    use std::fmt::Display;
    use std::ops::Deref;

    fn show_aggregate(ae: AggregationExpr) {
        let arg = ae.args.first().expect("ae.args[0] should not be None");
        println!(
            "aggr func: name={}, arg={}, modifier={}",
            ae.name(),
            arg,
            option_to_string(&ae.modifier)
        );
        match arg {
            Expr::Function(fe) => show_func_expr(fe),
            _ => panic!("expected rollup as func expr. Got {}", arg),
        }
    }

    fn show_func_expr(fe: &FunctionExpr) {
        let arg = fe.args.first().expect("fe.args[0] should not be None");
        println!("func: name={}, arg={}\n", fe.function, arg);
        match arg {
            Expr::Rollup(re) => show_rollup(re),
            _ => {
                panic!("expected rollup as func.args[0]. Got {}", arg);
            }
        }
    }

    fn show_rollup(re: &RollupExpr) {
        println!(
            "rollup: expr={}, window={}\n",
            re.expr,
            option_to_string(&re.window)
        );
        match &re.expr.deref() {
            Expr::MetricExpression(me) => show_metric_expr(me),
            _ => {
                panic!("expected MetricExpr in rollup. Got {}", re.expr);
            }
        }
    }

    fn show_metric_expr(me: &MetricExpr) {
        println!(
            "metric: {me}",
        );
    }

    fn option_to_string<T: Display>(val: &Option<T>) -> String {
        match val {
            Some(v) => v.to_string(),
            None => "".to_string(),
        }
    }

    #[test]
    fn example_parse() {
        let expr = parse(r#"sum(rate(foo{bar="baz"}[5m])) by (x,y)"#).unwrap();
        println!("parsed expr: {}\n", expr);

        match expr {
            Expr::Aggregation(ae) => show_aggregate(ae),
            _ => {
                panic!("parsed Expr should be an AggrFunctionExpr. Got {}", expr)
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
