


fn getKeepMetricNames(expr: &Expression) -> bool {
    // todo: move to optimize stage. put result in ast node
    let mut m_expr = expr;
    match expr {
        Expression::AggrFuncExpr(ae) => {
            // Extract: RollupFunc(...) from aggrFunc(rollupFunc(...)).
            // This case is possible when optimized aggrfn calculations are used
            // such as `sum(rate(...))`
            if ae.args.len() != 1 {
                return false
            }
            m_expr = ae.args[0];
        },
        _ => ()
    }
    match m_expr {
        Expression::Function(fe) => {
            fe.keep_metric_namee
        },
        _ => false
    }
}

fn try_get_arg_rollup_func_with_metric_expr(ae: &AggrFuncExpr) -> (Metricsql::FuncExpr, newRollupFunc) {
    if ae.args.len() != 1 {
        return (None, None)
    }
    let e = ae.args[0];
    // Make sure e contains one of the following:
    // - metricExpr
    // - metricExpr[d]
    // -: RollupFunc(metricExpr)
    // -: RollupFunc(metricExpr[d])

    match e {
        Expression::MetricExpr(me) => {
            if me.is_empty() {
                return (None, None);
            }
            let fe = Metricsql::FuncExpr {
                name: "default_rollup",
                args: Vec![me],
            };
            let nrf = getRollupFunc(fe.name);
            return (fe, nrf);
        }
        Expression::RollupExpr(re) => {
            let is_me = is_metric_expr(re.expr);
            if !is_me || me.is_empty() || re.for_subquery {
                return (None, None);
            }
            // e = metricExpr[d]
            let fe = MetricSql::FuncExpr {
                Name: "default_rollup",
                Args: Vec![re],
            };
            let nrf = getRollupFunc(fe.Name);
            return (fe, nrf);
        }
        Expression::FuncExpr(fe) => {
            let nrf = getRollupFunc(fe.Name);
            if nrf == nil {
                return nil;, nil
            }
            let rollup_arg_idx = metricsql.GetRollupArgIdx(fe);
            if rollup_arg_idx >= fe.args.len() {
                // Incorrect number of args for rollup func.
                return (nil, nil);
            }
            let arg = fe.args[rollup_arg_idx];
            if me, ok: = arg.(*metricsql.MetricExpr);
            ok {
                if me.is_empty() {
                return nil, nil
                }
                // e =: RollupFunc(metricExpr)
                return FuncExpr {
                name: fe.Name,
                args: [me],
            }, nrf
        }
    },
    Expression::RollupExpr(re) => {
        if me, ok: = re.Expr.(*metricsql.MetricExpr);
        !ok || me.IsEmpty() || re.ForSubquery()
        {
            return nil;, nil
        }
        // e =: RollupFunc(metricExpr[d])
        return (fe, nrf);
    }
}
    return (None, None)
}