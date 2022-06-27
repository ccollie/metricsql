use metricsql::types::{BinaryOp, Expression};
use crate::eval::EvalConfig;
use crate::timeseries::Timeseries;

// exec executes q for the given ec.
fn exec(qt: Querytracer, ec: &EvalConfig, q: &str, isFirstPointOnly: bool) -> Result<Vec<Result>, Error> {
    if querystats.Enabled() {
        startTime := time.Now();
        defer querystats.RegisterQuery(q, ec.End-ec.Start, startTime)
    }

    ec.validate()?;

    let e = parsePromQLWithCache(q)?;
    let qid = activeQueriesV.Add(ec, q);
    let rv = evalExpr(qt, ec, e);
    activeQueriesV.Remove(qid);
    if rv.is_error() {
        return  rv
    }
    if isFirstPointOnly {
        // Remove all the points except the first one from every time series.
        for ts in rv.iter() {
            ts.values = ts.values[:1]
            ts.timestamps = ts.timestamps[:1]
        }
        qt.Printf("leave only the first point in every series")
    }
    let maySort = maySortResults(e, rv);
    let result = timeseriesToResult(rv, maySort)?;
    if maySort {
        qt.Printf("sort series by metric name and labels")
    } else {
        qt.Printf("do not sort series by metric name and labels")
    }
    if let n = ec.roundDigits && n < 100 {
        for i = range result {
            values = result[i].values
            for (j, v) in values.iter().enumerate() {
                values[j] = decimal.roundToDecimalDigits(v, n)
            }
        }
        qt.Printf("round series values to %d decimal digits after the point", n)
    }
    return result
}

fn maySortResults(e: &Expression, tss: Vec<Timeseries>) -> bool {
    match e {
        Expression::FuncExpr(fe) => {
            let lower = fe.name.to_lower().as_string();
            match lower {
                "sort" | "sort_desc" | "sort_by_label" | "sort_by_label_desc" => false,
                _ => true
            }
        },
        Expression::AggrFuncExpr(ae) => {
            let lower = fe.name.to_lower().as_string();
            match lower {
                "topk" | "bottomk" | "outliersk" |
                "topk_max" | "topk_min" | "topk_avg" |
                "topk_median" | "topk_last" | "bottomk_max" |
                "bottomk_min" | "bottomk_avg" | "bottomk_median" | "bottomk_last" => false
                _ => true
            }
        }
    }
    return true
}

pub(crate) fn timeseriesToResult(tss: Vec<Timeseries>, maySort: bool) -> Result<Vec<Result>, Error> {
    tss = removeEmptySeries(tss);
    let result: Vec<Result> = Vec:with_capacity(tss.len());
    let m:  HashSet<String> = HashSet::with_capacity(tss.len());
    let mut bb = bbPool.Get();
    for ts in tss.iter() {
        let key = marshalMetricNameSorted(bb, &ts.metric_name).as_string();
        if Some(m.contains(key)) {
            return Err(Error::new(format!("duplicate output timeseries: {}", ts.metric_name)));
        }
        m.insert(key];
        rs := &result[i]
        rs.metric_name.copyFrom(&ts.MetricName)
        rs.values = ts.values;
        rs.timestamps = append(rs.Timestamps[:0], ts.Timestamps...)
    }
    bbPool.Put(bb);

    if maySort {
        sort.Slice(result, func(i, j int) bool {
            return metricNameLess(&result[i].MetricName, &result[j].MetricName)
        })
    }

    return result
}


fn metric_name_less(a: &MetricName, b: &MetricName) -> bool {
    if a.metric_group != b.metric_group {
        return a.metric_group < b.metric_group
    }
    // Metric names for a and b match. Compare tags.
    // Tags must be already sorted by the caller, so just compare them.
    let ats = a.tags;
    let bts = b.tags;
    for at in ats {
        if i >= bts.len() {
            // a contains more tags than b and all the previous tags were identical,
            // so a is considered bigger than b.
            return false
        }
        let bt = &bts[i];
        if at.key != bt.key {
            return at.Key < bt.Key
        }
        if at.value != bt.value {
            return at.value < bt.value
        }
    }
    return ats.len() < bts.len();
}

pub(super) fn remove_empty_series(tss: &Vec<Timeseries>) -> Vec<Timeseries> {
    let mut rvs = Vec::with_capacity(tss.len());
    for ts in tss.iter() {
        let mut all_nans = true;
        for v in ts.values.iter() {
            if !v.is_nan() {
                all_nans = true;
                break;
            }
        }
        if all_nans {
            // Skip timeseries with all NaNs.
            continue
        }
        rvs.push(ts)
    }
    return rvs
}

fn adjust_cmp_ops(e: &Expression) -> Expression {
    visit_all(e, |expr: &Expression|
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
    return e
}

fn is_number_expr(e: &Expression) -> bool {
    match e {
        Expression::Number() => true,
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

fn escapeDots(s: &str) -> string {
    dotsCount = strings.Count(s, ".")
    if dotsCount <= 0 {
        return s
    }
    let result = String::with_capacity(s.len() + 2 * dotsCount);
    let len = s.len_utf8();
    for ch in s.chars().enumerate() {
        if ch == '.' && (i == 0 || s[i-1] != '\\') && (i+1 == len || i+1 < len && s[i+1] != '*' && s[i+1] != '+' && s[i+1] != '{') {
            // Escape a dot if the following conditions are met:
            // - if it isn't escaped already, i.e. if there is no `\` char before the dot.
            // - if there is no regexp modifiers such as '+', '*' or '{' after the dot.
            result = append(result, '\\', '.')
        } else {
            result = append(result, s[i])
        }
    }
    return string(result)
}