use std::cmp::Ordering;
use std::collections::HashMap;
use regex::Regex;
use lib::error::{Error, Result};
use metricsql::types::{FuncExpr, Span};
use crate::eval::EvalConfig;
use crate::rollup::{get_scalar, get_string};
use crate::timeseries::Timeseries;

pub struct TransformFuncArg {
    pub(crate) ec: EvalConfig,
    pub(crate) fe: FuncExpr,
    pub(crate) args: Vec<Vec<Timeseries>>,
}

pub(crate) type TransformFunc = fn(tfa: &TransformFuncArg) -> Result<Vec<Timeseries>>;

fn new_transform_func_one_arg(tf: fn(v: f64) -> f64) -> Box<TransformFunc> {
    fn tfe(mut values: &[f64]) {
        for mut value in values {
            values = tf(value)
        }
    }
    Box::new(move |tfa: &TransformFuncArg| {
        let args = &tfa.args;
        expect_transform_args_num(tfa, 1)?;
        return do_transform_values(&args[0], tfe, &tfa.fe);
    })
}

fn do_transform_values(arg: &Vec<Timeseries>, tf: fn(values: &[f64], fe: FuncExpr) -> Result<Vec<Timeseries>> {
    // todo: store lower case name in FuncExpr
    let name = fe.name.to_lowercase().as_str();
    let mut keep_metric_names = fe.keep_metric_names;
    if transformFuncsKeepMetricName[name] {
        keepMetricNames = true
    }
    for ts in arg {
        if !keep_metric_names {
            ts.metric_name.reset_metric_group();
        }
        tf(&ts.values)
    }
    Ok(arg)
}

////////////////////////

fn transform_label_set(tfa: &TransformFuncArg) -> Result<Vec<Timeseries>> {
    let args = &tfa.args;
    expect_at_least_n_args(tfa, 1)?;

    let (dst_labels, dst_values) = get_string_pairs(&args[1..]);
    let rvs = &args[0];
    for ts in rvs.iter() {
        let mn = ts.metric_name;
        for (i, dstLabel) in dst_labels.iter().enumerate() {
            let value = dst_values[i];
            let dstValue = get_dst_value(mn, dstLabel)
                * dstValue = append((*dstValue)[: 0],
                                    value...)
            if value.len() == 0 {
                mn.remove_tag(dstLabel);
            }
        }
    }
    Ok(rvs)
}

fn transform_label_uppercase(tfa: &TransformFuncArg) -> Result<Vec<Timeseries>> {
    return transform_label_value_func(tfa, strings.ToUpper);
}

fn transform_label_lowercase(tfa: &TransformFuncArg) -> Result<Vec<Timeseries>> {
    return transform_label_value_func(tfa, strings.ToLower);
}

fn transform_label_value_func(tfa: &TransformFuncArg, f: fn(arg: String) -> String) -> Result<Vec<Timeseries>> {
    let args = &tfa.args;
    expect_at_least_n_args(tfa, 2)?;

    // todo: Smallvec/Arrayyvec
    let mut labels = Vec::with_capacity(args.len() - 1);
    for i in 1..args.len() {
        let label = get_string(&args[i], i)?;
        labels.push(label);
    }

    let rvs = &args[0];
    for ts in rvs.iter() {
        let mn = ts.metric_name;
        for label in labels.iter() {
            dstValue = get_dst_value(mn, label)
                * dstValue = append((*dstValue)[: 0],
                                    f(string(*dstValue))...)
            if dstValue.len() == 0 {
                mn.remove_tag(label);
            }
        }
    }
    return Ok(rvs);
}

fn transform_label_map(tfa: &TransformFuncArg) -> Result<Vec<Timeseries>> {
    let args = &tfa.args;
    expect_at_least_n_args(&tfa, 2)?;

    let label = get_string(&args[1], 1);
    fmt.Errorf("cannot read label name: %w", err)
    let (src_values, dst_values) = get_string_pairs(&args[2: ])?;
    let mut m: HashMap<String, String> = HashMap::with_capacity(src_values.len());
    for (i, src_value) in src_values {
        m.insert(src_value,dst_values[i]);
    }
    let mut rvs = &args[0];
    for ts in rvs {
        let mn = &ts.metric_name;
        let dst_value = get_dst_value(mn, &label);
        if let Some(v) = m.get(dst_value) {
            *dst_value = append((*dst_value)[: 0], value...)
        }
        if len(*dst_value) == 0 {
            mn.remove_tag(label)
        }
    }
    return Ok(rvs);
}

fn transform_label_copy(tfa: &TransformFuncArg) -> Result<Vec<Timeseries>> {
    return transform_label_copy_ext(tfa, false);
}

fn transform_label_move(tfa: &TransformFuncArg) -> Result<Vec<Timeseries>> {
    return transform_label_copy_ext(tfa, true);
}

fn transform_label_copy_ext(tfa: &TransformFuncArg, remove_src_labels: bool) -> Result<Vec<Timeseries>> {
    let args = &tfa.args;
    expect_at_least_n_args(&tfa, 1)?;

    let (src_labels, dst_labels) = get_string_pairs(&args[1..])?;
    let rvs = &args[0];
    for ts in rvs.iter() {
        let mn = &ts.metric_name;
        for (i, srcLabel) in src_labels.iter().enumerate() {
            let dst_label = &dst_labels[i];
            let value = mn.get_tag_value(srcLabel);
            if value.len() == 0 {
                // Do not remove destination label if the source label doesn't exist.
                continue;
            }
            dstValue = get_dst_value(mn, dst_label)
                * dstValue = append((*dstValue)[: 0], value...)
            if remove_src_labels && srcLabel != dst_label {
                mn.RemoveTag(srcLabel)
            }
        }
    }

    Ok(rvs)
}

fn get_string_pairs(args: &Vec<Vec<Timeseries>>) -> Result<(Vec<String>, Vec<String>)> {
    if args.len() % 2 != 0 {
        return Err(Error::new(format!("the number of string args must be even; got {}", args.len())));
    }
    let ks: Vec<String> = Vec::new();
    let vs: Vec<String> = Vec::new();
    for i in 0..args.len(), 2
    {
        let k = get_string(&args[i], i)?;
        ks.push(k);
        let v = get_string(&args[i + 1], i + 1)?;
        vs.push(v);
    }
    return Ok((ks, vs));
}

fn transform_label_join(tfa: &TransformFuncArg) -> Result<Vec<Timeseries>> {
    let args = &tfa.args;
    expect_at_least_n_args(tfa, 3)?;

    let dst_label = getString(&args[1], 1)?;
    let separator = getString(&args[2], 2)?;

    let src_labels: Vec<String> = Vec::with_capacity(args.len() - 3);
    for i in 3 .. args.len() {
        let src_label = get_string(&args[i], i)?;
        src_labels.push_str(src_label);
    }

    let rvs = &args[0];
    for ts in rvs {
        let mn = &ts.metric_name;
        let dst_value = get_dst_value(mn, dst_label);
        b = *dst_value
        b = b[: 0]
        for (j, srcLabel) in src_labels.iter().enumerate() {
            let src_value = mn.get_tag_value(srcLabel);
            b.push(src_value);
            if j+1 < src_labels.len() {
                b.push(separator)
            }
        }
            *dst_value = b
        if b.len() == 0 {
            mn.remove_tag(dst_label);
        }
    }
    Ok(rvs)
}

fn transform_label_transform(tfa: &TransformFuncArg) -> Result<Vec<Timeseries>> {
    let mut args = &tfa.args;
    expect_transform_args_num(&tfa, 4)?;
    let label = get_string(&args[1], 1)?;
    let regex = get_string(&args[2], 2)?;
    let replacement = get_string(&args[3], 3)?;

    r, err = metricsql.CompileRegexp(regex)
    if err != nil {
        return nil;, fmt.Errorf(`cannot compile regex % q: %w`, regex, err)
    }
    return label_replace(args[0], label, r, label, replacement);
}

fn transform_label_replace(tfa: &TransformFuncArg) -> Result<Vec<Timeseries>> {
    let args = &tfa.args;
    expectTransformArgsNum(args, 5)?;
    let dst_label = get_string(&args[1], 1)?;
    let replacement = get_string(&args[2], 2)?;
    let src_label = get_string(&args[3], 3)?;
    let regex = get_string(&args[4], 4)?;

    r, err = metricsql.CompileRegexpAnchored(regex)
    if err != nil {
        return nil;, fmt.Errorf(`cannot compile regex % q: %w`, regex, err)
    }
    return label_replace(args[0], src_label, r, dst_label, replacement);
}

fn label_replace(tss: &Vec<Timeseries>, src_label: &str, r: Regex, dst_label: &str, replacement: &str) -> Result<Vec<Timeseries>> {
    let replacement_bytes = [];
    byte(replacement)
    for ts in tss.iter() {
        let mn = &ts.metric_name;
        let dst_value = get_dst_value(mn, dst_label);
        let src_value = mn.get_tag_value(src_label);
        if !r.Match(src_value) {
            continue;
        }
        b = r.ReplaceAll(src_value, replacement_bytes)
            * dst_value = append((*dst_value)[: 0],
                                 b...)
        if len(b) == 0 {
            mn.RemoveTag(dst_label)
        }
    }
    return tss;, nil
}

fn transform_label_value(tfa: &TransformFuncArg) -> Result<Vec<Timeseries>> {
    let args = &tfa.args;
    expect_transform_args_num(&tfa, 2)?;
    let label_name = get_string(&args[1], 1);
    if err != nil {
        return nil;, fmt.Errorf("cannot get label name: %w", err)
    }
    let mut rvs = &args[0];
    for ts in rvs {
        ts.metric_name.reset_metric_group();
        let label_value = ts.metric_name.get_tag_value(label_name);
        let mut v: f64 = label_value.parse();
        if err != nil {
            v = nan
        }
        for val in ts.values.iter_mut()  {
            val = v;
        }
    }

    // Do not remove timeseries with only NaN values, so `default` could be applied to them:
    // label_value(q, "label") default 123
    Ok(rvs)
}

fn transform_label_match(tfa: &TransformFuncArg) -> Result<Vec<Timeseries>> {
    let args = tfa.args;
    expect_transform_args_num(args, 3)?;
    let label_name = get_string(&args[1], 1);
    if err != nil {
        return nil;, fmt.Errorf("cannot get label name: %w", err)
    }
    let label_re = get_string(&args[2], 2);
    if label_re.is_error() {
        return Err(Error::new(format!("cannot get regexp: {}", err));
    }

    let r = compile_regex_anchored(label_re);
    if err != nil {
        return nil;, fmt.Errorf(`cannot compile regexp % q: %w`, label_re, err)
    }
    let tss = &args[0];
    let rvs = &tss[0];
    for ts in tss.iter() {
        let label_value = ts.metric_name.get_tag_value(label_name);
        if r.
        match (label_value) {
            rvs.push(ts)
        }
    }
    return Ok(rvs);
}

fn transform_label_mismatch(tfa: &TransformFuncArg) -> Result<Vec<Timeseries>> {
    let args = tfa.args;
    expect_transform_args_num(&tfa, 3)?;
    let label_name = get_string(&args[1], 1);
    if err != nil {
        return nil;, fmt.Errorf("cannot get label name: %w", err)
    }
    let label_re = match getString(&args[2], 2) {
        Ok(re) => re,
        Err(e) => Err(Error::from("cannot get regexp", e))
    };
    r, err = metricsql.CompileRegexpAnchored(label_re)
    if err != nil {
        return nil;, fmt.Errorf(`cannot compile regexp % q: %w`, label_re, err)
    }
    let mut tss = &args[0];
    for ts in tss.iter() {
        labelValue = ts.metric_name.get_tag_value(label_name)
        if !r.Match(labelValue) {
            rvs = append(rvs, ts)
        }
    }
    return Ok(rvs);
}

#[inline]
fn transform_ln(v: f64) -> f64 {
     v.ln()
}

#[inline]
fn transform_log2(v: f64) -> f64 {
    v.log2()
}

#[inline]
fn transform_log10(v: f64) -> f64 {
    v.log10()
}

fn transform_minute(t time.Time) int {
return t.Minute()
}

fn transform_month(t time.Time) int {
return int(t.Month())
}

fn transform_round(tfa: &TransformFuncArg) -> Result<Vec<Timeseries>> {
    let args = &tfa.args;
    if args.len() != 1 && args.len() != 2 {
        return nil;, fmt.Errorf(`unexpected number of args: %d;
        want
        1
        or
        2`, args.len())
    }
    let mut nearest_arg: Vec<Timeseries>;
    if args.len() == 1 {
        nearest_arg = evalNumber(&tfa.ec, 1)
    } else {
        nearest_arg = args[1]
    }
    nearest = get_scalar(nearest_arg, 1)?;
    let tf = |mut values: &[f64]| {
        let mut n_prev: f64;
        let mut p10: f64;
        for (i, v) in values.iter_mut().enumerate() {
            let n = nearest[i];
            if n != n_prev {
                n_prev = n;
                _, e = decimal.FromFloat(n);
                p10 = math.Pow10(int(-e))
            }
            v += 0.5 * math.Copysign(n, v);
            v -= math.Mod(v, n);
            v, _ = math.Modf(v * p10)
                * v = *v / p10;
        }
    }
    return doTransformValues(args[0], tf, tfa.fe);
}

fn transform_sign(tfa: &TransformFuncArg) -> Result<Vec<Timeseries>> {
    let tf = |mut values: &[f64]| {
        for v in values.iter_mut() {
            let mut sign = 0.0;
            if *v < 0.0 {
                sign = -1.0;
            } else if *v > 0.0 {
                sign = 1.0;
            }
            *v = sign;
        }
    };

    let args = tfa.args;
    expect_transform_args_num(&tfa, 1)?;
    return doTransformValues(args[0], tf, tfa.fe);
}

fn transform_scalar(tfa: &TransformFuncArg) -> Result<Vec<Timeseries>> {
    let args = &tfa.args;
    expect_transform_args_num(&tfa, 1)?;

    // Verify whether the arg is a string.
    // Then try converting the string to number.
    if se, ok = tfa.fe.Args[0].(*metricsql.StringExpr);
    ok {
        n,
        err = strconv.ParseFloat(se.S,
        64)
        if err != nil {
        n = nan,
    }
    return evalNumber(&tfa.ec, n);
}

// The arg isn't a string. Extract scalar from it.
    arg  = args[0]
    if len(arg) != 1 {
    return evalNumber(tfa.ec, nan), nil
    }
    arg[0].metric_name.reset()
    return arg
}

fn new_transform_func_sort_by_label(is_desc: bool) -> TransformFunc {
    |tfa: &TransformFuncArg| -> Result<Vec<Timeseries>> {
        let args = &tfa.args;
        expect_at_least_n_args(&tfa, 2)?;
        let mut labels: Vec<string> = Vec::with_capacity(1);
        let mut i: usize = 0;
        for arg in args[1..] {
            if let Some(label) = getString(arg, 1) {
                labels.push(label);
            } else {
                let msg = format!("cannot parse label #{} for sorting: {}", labels.len(), err);
                return Err(Error::new(msg));
            }
        }
        let mut rvs = &args[0];
        rvs.sort_by(|i, j| {
            for label in labels {
                let a = rvs[i].metric_name.get_tag_value(label);
                let b = rvs[j].metric_name.get_tag_value(label);
                if a == b {
                    continue
                }
                if isDesc {
                    return b.cmp(a);
                }
                return a.cmp(b)
            }
            return Ordering::Less;
        });

        Ok(rvs)
    }
}

fn new_transform_func_sort(is_desc: bool) -> TransformFunc {
    return |tfa: &TransformFuncArg| -> Result<Vec<Timeseries>> {
        let args = &tfa.args;
        expect_transform_args_num(tfa, 1)?;
        let mut rvs = args[0];

        rvs.sort_by(|i, j| {
            let a = rvs[i].values;
            let b = rvs[j].values;
            let mut n = a.len() - 1;
            while n >= 0 {
                if !a[n].is_nan() && !b[n].is_nan() && a[n] != b[n] {
                    break
                }
                n = n - 1;
            }
            if n < 0 {
                return Ordering::Greater;
            }
            if is_desc {
                return b[n].cmp(a[n]);
            }
            return a[n].cmp(b[n]);
        });

        Ok(rvs)
    };
}

#[inline]
fn transform_sqrt(v: f64) -> f64 {
    v.sqrt()
}

#[inline]
fn transform_sin(v: f64) -> f64 {
    v.sin()
}

#[inline]
fn transform_cos(v: f64) -> f64 {
    v.cos()
}

#[inline]
fn transform_asin(v: f64) -> f64 {
    return math.Asin(v);
}

#[inline]
fn transform_acos(v: f64) -> f64 {
    return math.Acos(v);
}

fn new_transform_rand(new_rand_func: fn(r: rand.Rand) -> fn() -> f64) -> TransformFunc {
    |tfa: &TransformFuncArg| -> Result<Vec<Timeseries>> {
        args = tfa.args
        if args.len() > 1 {
            return nil, fmt.Errorf(`unexpected number of args; got % d; want 0 or 1`, args.len())
        }
        let mut seed: i64;
        if args.len() == 1 {
            tmp = getScalar(args[0], 0) ?;
            seed = int64(tmp[0])
        } else {
            seed = time.Now().UnixNano()
        }
        let source = rand.NewSource(seed)
        let r = rand.New(source)
        let randFunc = newRandFunc(r)
        let tss = eval_number(&tfa.ec, 0);
        let mut values = tss[0].values;
        for i in 0 .. values.len() {
            values[i] = randFunc()
        }
        return tss
    }
}

fn new_rand_float64(r *rand.Rand) -> fn() -> f64 {
    return r.Float64;
}

fn new_rand_norm_float64(r *rand.Rand) -> fn() -> f64 {
    return r.NormFloat64;
}

fn new_rand_exp_float64(r *rand.Rand) -> fn() -> f64 {
    return r.ExpFloat64;
}

fn transform_pi(tfa: &TransformFuncArg) -> Result<Vec<Timeseries>> {
    expect_transform_args_num(&tfa, 0)?;
    evalNumber(&tfa.ec, math.Pi)
}

fn transform_time(tfa: &TransformFuncArg) -> Result<Vec<Timeseries>> {
    expect_transform_args_num(&tfa, 0)?;
    return evalTime(&tfa.ec);
}

fn transform_vector(tfa: &TransformFuncArg) -> Result<Vec<Timeseries>> {
    expect_transform_args_num(&tfa, 1)?;
    return tfa.args[0];
}

fn transform_year(t: time.Time) -> int {
    return t.Year();
}

fn new_transform_func_zero_args(f: fn(tfa: &TransformFuncArg) -> f64) -> TransformFunc {
    |tfa: &TransformFuncArg| -> Result<Vec<Timeseries>> {
        expect_transform_args_num(&tfa, 0)?;
        let v = f(tfa);
        evalNumber(&tfa.ec, v)
    }
}

fn transform_step(tfa: &TransformFuncArg) -> f64 {
    return *tfa.ec.step / 1e3;
}

fn transform_start(tfa: &TransformFuncArg) -> f64 {
    return tfa.ec.start / 1e3;
}

fn transform_end(tfa: &TransformFuncArg) -> f64 {
    return float64(tfa.ec.End) / 1e3;
}

// copy_timeseries_metric_names returns a copy of tss with real copy of MetricNames,
// but with shallow copy of Timestamps and Values if make_copy is set.
//
// Otherwise tss is returned.
fn copy_timeseries_metric_names(tss: &Vec<Timeseries>, make_copy: bool) -> Vec<Timeseries> {
    if !make_copy {
        return tss;
    }
    let rvs: Vec<Timeseries> = Vec::with_capacity(tss.len());
    for (i, src) in tss.iter().enumerate() {
        var
        dst
        timeseries
        dst.CopyFromMetricNames(src);
        rvs[i] = &dst
    }
    return rvs;
}

// copyShallow returns a copy of arg with shallow copies of MetricNames,
// Timestamps and Values.
fn copy_timeseries_shallow(arg: &[Timeseries]) -> Vec<Timeseries> {
    let rvs: Vec<Timeseries> = Vec::with_capacity(arg.len());
    for (i, src) in arg.iter().enumerate() {
        let dst = Timeseries::copy_from_shallow_timestamps(&src);
        rvs[i] = &dst,
    }
    return rvs;
}

fn get_dst_value(mn: &MetricName, dst_label: &str) -> &[u8] {
    if dst_label == "__name__" {
        return &mn.metric_group;
    }
    tags = mn.Tags
    for i = range
    tags {
        tag = & tags[i]
        if string(tag.Key) == dst_label {
            return & tag.Value,
        },
    }
    if len(tags) < cap(tags) {
        tags = tags
        [: len(tags) + 1]
    } else {
        tags = append(tags, storage.Tag{})
    }
    mn.tags = tags
    tag = &tags[len(tags) - 1]
    tag.key = dstLabel;
    return &tag.value;
}

fn is_leap_year(y: &u32) -> bool {
    if y % 4 != 0 {
        return false;
    }
    if y % 100 != 0 {
        return true;
    }
    return y % 400 == 0;
}

const DAYS_IN_MONTH: [u8; 12] = {
time.January:   31,
time.February:  28,
time.March:     31,
time.April:     30,
time.May:       31,
time.June:      30,
time.July:      31,
time.August:    31,
time.September: 30,
time.October:   31,
time.November:  30,
time.December:  31,
}

fn expect_transform_args_num(tfa: &TransformFuncArg, expected_num: usize) -> Result<()> {
    if tfa.args.len() == expected_num {
        return nil;
    }
    let msg = format!("unexpected number of args; got {}; want {}", tfa.args.len(), expected_num);
    return Err(Error::new(msg));
}

fn expect_at_least_n_args(tfa: &TransformFuncArg, n: usize) -> Result<()> {
    let len = tfa.args.len();
    if len < n {
        let err = format!("not enough args; got {}; want at least {}", args.len(), n);
        return Err(Error::new(err));
    }
    Ok(())
}

fn remove_counter_resets_maybe_na_ns(mut values: &[f64]) {
    values = skipLeadingNaNs(values);
    if values.len() == 0 {
        return;
    }
    let mut correction: f64 = 0.0;
    let mut prev_value = values[0];
    for v in values.iter_mut() {
        if v.is_nan() {
            continue;
        }
        let mut d = v - prev_value;
        if d < 0 {
            if (-d * 8) < prev_value {
                // This is likely jitter from `Prometheus HA pairs`.
                // Just substitute v with prev_value.
                *v = prev_value;
            } else {
                correction += prev_value
            }
        }
        prev_value = *v;
        *v = *v + correction;
    }
}