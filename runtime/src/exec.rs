use std::collections::HashSet;
use lib::error::{Error, Result};
use metricsql::{parse, optimize, visit_all};
use metricsql::types::{BinaryOp, Expression};
use crate::eval::EvalConfig;
use crate::parser_cache::{get_cache, get_parse_cache, ParseCacheValue};
use crate::timeseries::Timeseries;

// runtime executes q for the given ec.
pub fn exec(qt: Querytracer, ec: &EvalConfig, q: &str, is_first_point_only: bool) -> Result<Vec<QueryResult>> {
    if querystats.enabled() {
        startTime = time.Now();
        defer querystats.RegisterQuery(q, ec.End-ec.Start, startTime)
    }

    ec.validate()?;

    let e = parse_promql_with_cache(q)?;
    let qid = activeQueriesV.Add(ec, q);
    let rv = evalExpr(qt, ec, e);
    activeQueriesV.Remove(qid);
    if rv.is_error() {
        return  rv
    }
    if is_first_point_only {
        // Remove all the points except the first one from every time series.
        for ts in rv.iter() {
            ts.values = ts.values[0..1];
            ts.timestamps = ts.timestamps[0..1];
        }
        qt.Printf("leave only the first point in every series")
    }
    let may_sort = may_sort_results(e, rv);
    let mut result = timeseries_to_result(rv, may_sort)?;
    if may_sort {
        qt.printf("sort series by metric name and labels")
    } else {
        qt.printf("do not sort series by metric name and labels")
    }
    if let n = ec.roundDigits && n < 100 {
        for r in result.iter() {
            let mut values = r.values;
            for (j, v) in values.iter().enumerate() {
                values[j] = decimal.roundToDecimalDigits(v, n)
            }
        }
        qt.Printf("round series values to %d decimal digits after the point", n)
    }
    Ok(result)
}

fn may_sort_results(e: &Expression, tss: &[Timeseries]) -> bool {
    match e {
        Expression::Function(fe) => {
            let lower = fe.name.to_lower().as_string();
            match lower {
                "sort" | "sort_desc" | "sort_by_label" | "sort_by_label_desc" => false,
                _ => true
            }
        },
        Expression::Aggregation(ae) => {
            let lower = ae.name.to_lower().as_string();
            match lower {
                "topk" | "bottomk" | "outliersk" |
                "topk_max" | "topk_min" | "topk_avg" |
                "topk_median" | "topk_last" | "bottomk_max" |
                "bottomk_min" | "bottomk_avg" | "bottomk_median" | "bottomk_last" => false,
                _ => true
            }
        },
        _ => true
    }
    return true
}

pub(crate) fn timeseries_to_result(mut tss: &Vec<Timeseries>, may_sort: bool) -> Result<Vec<QueryResult>> {
    let tss = remove_empty_series(tss);
    let result: Vec<QueryResult> = Vec::with_capacity(tss.len());
    let m:  HashSet<String> = HashSet::with_capacity(tss.len());
    let mut bb = bbPool.Get();
    for ts in tss.iter() {
        let key = marshalMetricNameSorted(bb, &ts.metric_name).as_string();
        if Some(m.contains(key)) {
            return Err(Error::from(format!("duplicate output timeseries: {}", ts.metric_name)));
        }
        m.insert(key);
        let mut rs = &result[i];
        rs.metric_name.copyFrom(&ts.MetricName);
        rs.values = ts.values;
        rs.timestamps = append(rs.timestamps[:0], ts.timestamps...)
    }
    bbPool.Put(bb);

    if may_sort {
        result.sort_by(|a, b| {
            a.metric_name.cmp(b.metric_name)
        })
    }

    return result
}

pub(super) fn remove_empty_series(mut tss: &Vec<Timeseries>) {
    tss.retain(|ts| {
        !ts.values.all(|v| v.is_nan())
    });
}

// todo: put in optimize phase
fn adjust_cmp_ops(mut e: &Expression) {
    visit_all(e, |mut expr: &Expression|
        {
            match expr {
                Expression::BinaryOpExpr(be) => {
                    if !be.op.is_comparison() {
                        return
                    }
                    if is_number_expr(be.right) || !is_scalar_expr(be.left) {
                        return
                    }
                    // Convert 'num cmpOp query' expression to `query reverseCmpOp num` expression
                    // like Prometheus does. For instance, `0.5 < foo` must be converted to `foo > 0.5`
                    // in order to return valid values for `foo` that are bigger than 0.5.
                    be.op = get_reverse_cmp(be.op);
                    be.swap_operands();
                },
                _ => {}
            }
        });
}

fn is_number_expr(e: &Expression) -> bool {
    match e {
        Expression::Number(..) => true,
        _ => false
    }
}

fn is_scalar_expr(e: Expression) -> bool {
    match e {
        Expression::Number(..) => true,
        Expression::Function(fe) => {
            // time() returns scalar in PromQL - see https://prometheus.io/docs/prometheus/latest/querying/functions/#time
            let name = fe.name.to_lowercase().as_string();
            return name == "time"
        },
        _ => false
    }
}

fn get_reverse_cmp(op: BinaryOp) -> BinaryOp {
    match op {
        BinaryOp::Gt => BinaryOp::Lt,
        BinaryOp::Lt => BinaryOp::Gt,
        BinaryOp::Gte => BinaryOp::Lte,
        BinaryOp::Lte => BinaryOp::Gte,
        // there is no need in changing `==` and `!=`.
        _ => op
    }
}

pub fn parse_promql_with_cache(q: &str) -> Result<&Expression> {
    let mut cache = get_parse_cache();
    let mut pcv: ParseCacheValue;
    let mut expression: &Expression;

    match cache().get(q) {
        Some(v) => {
            pcv = v;
        },
        None(..) => {
            let mut e = parse(&q);
            match e {
                Ok(mut expr) => {
                    expression = optimize(expr);
                    adjust_cmp_ops(&expression);
                    pcv = ParseCacheValue {
                        expr: Some(Expression),
                        err: None
                    }
                },
                Err(err) => {
                    pcv = ParseCacheValue {
                        expr: None,
                        err: Some(err)
                    };
                }
            }
            cache.put(q, pcv);
        }
    }
    if pcv.err.is_some() {
        Err(pcv.err.unwrap())
    } else {
        Ok(&pcv.expr.unwrap())
    }
}