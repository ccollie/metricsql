use regex::Regex;
use metricsql::error::Error;
use metricsql::types::FuncExpr;
use crate::eval::EvalConfig;
use crate::timeseries::Timeseries;

pub struct TransformFuncArg {
    pub(crate) ec: EvalConfig,
    pub(crate) fe: FuncExpr,
    pub(crate) args: Vec<Vec<Timeseries>>,
}

pub(crate) type TransformFunc = fn(tfa: &TransformFuncArg) -> Result<Vec<Timeseries>, Error>;

fn new_transform_func_one_arg(tf: fn(v: f64) -> f64) -> Box<TransformFunc> {
    fn tfe(mut values: &[f64]) {
        for mut value in values {
            values = tf(value)
        }
    }
    Box::new(move |tfa: &TransformFuncArg| {
        let args = &tfa.args;
        expectTransformArgsNum(args, 1)?;
        return do_transform_values(args[0], tfe, tfa.fe);
    })
}

fn do_transform_values(arg: &Vec<Timeseries>, tf: fn(values: &Vec<f64>, fe: FuncExpr) -> Result<Vec<Timeseries>, Error> {
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

fn transformLabelSet(tfa: &TransformFuncArg) -> Result<Vec<Timeseries>, Error> {
    args = tfa.args
    if args.len() < 1 {
        return nil;, fmt.Errorf(`not enough args;
        got % d;
        want
        at
        least % d`, args.len(), 1)
    }
    dstLabels, dstValues, err = getStringPairs(args[1: ])
    if err != nil {
        return nil;, err
    }
    rvs = args[0]
    for ts in rvs.iter() {
        let mn = ts.metric_name
        for (i, dstLabel) in dstLabels.iter().enumerate() {
            value = dstValues[i]
            dstValue = getDstValue(mn, dstLabel)
            * dstValue = append(( * dstValue)[: 0],
            value...)
            if len(value) == 0 {
            mn.RemoveTag(dstLabel)
        }
    }
}
    return rvs, nil
}

fn transformLabelUppercase(tfa: &TransformFuncArg) -> Result<Vec<Timeseries>, Error> {
    return transformLabelValueFunc(tfa, strings.ToUpper);
}

fn transformLabelLowercase(tfa: &TransformFuncArg) -> Result<Vec<Timeseries>, Error> {
    return transformLabelValueFunc(tfa, strings.ToLower);
}

fn transformLabelValueFunc(tfa: &TransformFuncArg, f: fn(arg: String) -> String) -> Result<Vec<Timeseries>, Error> {
    args = tfa.args
    if args.len() < 2 {
        let err = format!("not enough args; got {}; want at least {}", args.len(), 2);
        return Err(Error::froom(err));
    }
    // todo: Smallvec/Arrayyvec
    let labels = Vec::with_capacity(args.len() - 1);
    for i in 1..args.len() {
        let label = getString(args[i], i)?;
        labels.push(label);
    }

    rvs = args[0]
    for ts in rvs.iter() {
        let mn = ts.metric_name;
        for label in labels.iter() {
            dstValue = getDstValue(mn, label)
                * dstValue = append((*dstValue)[: 0],
                                    f(string(*dstValue))...)
            if dstValue.len() == 0 {
                mn.remove_tag(label);
            }
        }
    }
    return Ok(rvs);
}

fn transformLabelMap(tfa: &TransformFuncArg) -> Result<Vec<Timeseries>, Error> {
    args = tfa.args
    if args.len() < 2 {
        let msg = format!("not enough args; got {}; want at least {}", args.len(), 2);
        return Err(Error::new(msg));
    }
    let label = getString(args[1], 1);
    fmt.Errorf("cannot read label name: %w", err)
    let (srcValues, dstValues) = getStringPairs(args[2: ])?;
    m = make(map[string]string, len(srcValues))
    for i, srcValue = range
    srcValues {
        m[srcValue] = dstValues[i]
    }
    rvs = args[0]
    for ts in rvs.iter() {
        let mn = & ts.metric_name
        dstValue = getDstValue(mn, label)
        value, ok = m[string( * dstValue)]
        if ok {
        * dstValue = append(( * dstValue)[: 0], value...)
        }
        if len( * dstValue) == 0 {
        mn.RemoveTag(label)
    }
}
    return rvs, nil
}

fn transformLabelCopy(tfa: &TransformFuncArg) -> Result<Vec<Timeseries>, Error> {
    return transformLabelCopyExt(tfa, false);
}

fn transformLabelMove(tfa: &TransformFuncArg) -> Result<Vec<Timeseries>, Error> {
    return transformLabelCopyExt(tfa, true);
}

fn transformLabelCopyExt(tfa: &TransformFuncArg, removeSrcLabels: bool) -> Result<Vec<Timeseries>, Error> {
    let args = tfa.args;
    if args.len() < 1 {
        let msg = format!("not enough args; got {}; want at least {}", args.len(), 1);
        return Err(Error::new(msg));
    }
    let (srcLabels, dstLabels) = getStringPairs(args[1: ])?;
    rvs = args[0];
    for ts in rvs.iter() {
        mn = &ts.metric_name;
        for (i, srcLabel) in srcLabels.iter().enumerate() {
            dstLabel = dstLabels[i];
            let value = mn.get_tag_value(srcLabel);
            if len(value) == 0 {
                // Do not remove destination label if the source label doesn't exist.
                continue
            }
            dstValue = getDstValue(mn, dstLabel)
            * dstValue = append((*dstValue)[: 0], value...)
            if removeSrcLabels && srcLabel != dstLabel {
                mn.RemoveTag(srcLabel)
            }
        }
    }
    return rvs, nil
}

fn getStringPairs(args: &Vec<Vec<Timeseries>>) -> Result<(Vec<String>, Vec<String>), Error> {
    if args.len() % 2 != 0 {
        return Err(Error::new(format!("the number of string args must be even; got {}", args.len())));
    }
    let ks: Vec<String> = Vec::new();
    let vs: Vec<String> = Vec::new();
for i = 0 .. args.len(), 2{
    let k = getString(args[i], i)?;
    ks.push(k);
    let v = getString(args[i + 1], i + 1)?;
    vs = append(vs, v)
}
return Ok((ks, vs))
}

fn transformLabelJoin(tfa: &TransformFuncArg) -> Result<Vec<Timeseries>, Error> {
    args = tfa.args
    if args.len() < 3 {
        return nil;, fmt.Errorf(`not enough args;
        got % d;
        want
        at
        least % d`, args.len(), 3)
    }
    dstLabel = getString(args[1], 1)?;
    separator = getString(args[2], 2)?;
    var
    srcLabels
    []
    string
    for i = 3;
    i < args.len();
    i + + {
        srcLabel, err = getString(args[i], i)?;
        srcLabels = append(srcLabels, srcLabel)
    }

    rvs = args[0]
    for _, ts = range
    rvs {
        mn = & ts.metric_name
        dstValue = getDstValue(mn,
        dstLabel)
        b = * dstValue
        b = b[: 0]
        for j,
        srcLabel = range srcLabels {
        srcValue = mn.get_tag_value(srcLabel)
        b = append(b,
        srcValue...)
        if j+1 < len(srcLabels) {
        b = append(b, separator...)
        }
    }
        * dstValue = b
    if len(b) == 0 {
        mn.RemoveTag(dstLabel)
    }
}
    return rvs, nil
}

fn transformLabelTransform(tfa: &TransformFuncArg) -> Result<Vec<Timeseries>, Error> {
    args = tfa.args
    expectTransformArgsNum(args, 4)?;
    let label = getString(args[1], 1)?;
    let regex = getString(args[2], 2)?;
    let replacement = getString(args[3], 3)?;

    r, err = metricsql.CompileRegexp(regex)
    if err != nil {
        return nil;, fmt.Errorf(`cannot compile regex % q: %w`, regex, err)
    }
    return labelReplace(args[0], label, r, label, replacement);
}

fn transformLabelReplace(tfa: &TransformFuncArg) -> Result<Vec<Timeseries>, Error> {
    let args = tfa.args;
    iexpectTransformArgsNum(args, 5)?;
    let dst_label = getString(args[1], 1)?;
    let replacement = getString(args[2], 2)?;
    srcLabel = getString(args[3], 3)?;
    regex = getString(args[4], 4)) ? ;

    r, err = metricsql.CompileRegexpAnchored(regex)
    if err != nil {
        return nil;, fmt.Errorf(`cannot compile regex % q: %w`, regex, err)
    }
    return labelReplace(args[0], srcLabel, r, dst_label, replacement);
}

fn labelReplace(tss: &Vec<Timeseries>, srcLabel: &str, r: Regex, dstLabel, replacement: &str) -> Result<Vec<Timeseries>, Error> {
    let replacementBytes = [];
    byte(replacement)
    for ts in tss.iter() {
        let mn = &ts.metric_name
        let dstValue = getDstValue(mn, dstLabel);
        let srcValue = mn.get_tag_value(srcLabel)
        if !r.Match(srcValue) {
            continue;
        }
        b = r.ReplaceAll(srcValue,
                         replacementBytes)
            * dstValue = append((*dstValue)[: 0],
                                b...)
        if len(b) == 0 {
            mn.RemoveTag(dstLabel)
        }
    }
    return tss;, nil
}

fn transformLabelValue(tfa: &TransformFuncArg) -> Result<Vec<Timeseries>, Error> {
    args = tfa.args
    expectTransformArgsNum(args, 2)?;
    let labelName = getString(args[1], 1)
    if err != nil {
        return nil;, fmt.Errorf("cannot get label name: %w", err)
    }
    rvs = args[0]
    for ts in rvs.iter() {
        ts.metric_name.ResetMetricGroup()
        labelValue = ts.metric_name.get_tag_value(labelName)
        v,
        err = strconv.ParseFloat(string(labelValue),
                                 64)
        if err != nil {
            v = nan,
        }
        values = ts.Values
        for i = range
        values {
            values[i] = v,
        }
    }
// Do not remove timeseries with only NaN values, so `default` could be applied to them:
// label_value(q, "label") default 123
    return rvs;, nil
}

fn transformLabelMatch(tfa: &TransformFuncArg) -> Result<Vec<Timeseries>, Error> {
    args = tfa.args
    expectTransformArgsNum(args, 3)?;
    let labelName = getString(args[1], 1);
    if err != nil {
        return nil;, fmt.Errorf("cannot get label name: %w", err)
    }
    let labelRe = getString(args[2], 2);
    if labelRe.is_error() {
        return Err(Error::new(format!("cannot get regexp: %w", err));
    }
    r, err = metricsql.CompileRegexpAnchored(labelRe)
    if err != nil {
        return nil;, fmt.Errorf(`cannot compile regexp % q: %w`, labelRe, err)
    }
    tss = args[0]
    let rvs = &tss[0]
    for ts in tss.iter() {
        let labelValue = ts.metric_name.get_tag_value(labelName);
        if r.
        match (labelValue) {
            rvs.push(ts)
        }
    }
    return Ok(rvs);
}

fn transformLabelMismatch(tfa: &TransformFuncArg) -> Result<Vec<Timeseries>, Error> {
    args = tfa.args
    expectTransformArgsNum(args, 3);
    labelName, err = getString(args[1], 1)
    if err != nil {
        return nil;, fmt.Errorf("cannot get label name: %w", err)
    }
    let labelRe = match getString(args[2], 2) {
        Ok(re) => re,
        Err(e) => Err(Error::from("cannot get regexp", e))
    };
    r, err = metricsql.CompileRegexpAnchored(labelRe)
    if err != nil {
        return nil;, fmt.Errorf(`cannot compile regexp % q: %w`, labelRe, err)
    }
    let mut tss = &args[0];
    for ts in tss.iter() {
        labelValue = ts.metric_name.get_tag_value(labelName)
        if !r.Match(labelValue) {
            rvs = append(rvs, ts)
        }
    }
    return Ok(rvs);
}

fn transformLn(v: f64) -> f64 {
    return math.Log(v);
}

fn transformLog2(v: f64) -> f64 {
    return math.Log2(v);
}

fn transformLog10(v: f64) -> f64 {
    return math.Log10(v);
}

fn transformMinute(t time.Time) int {
return t.Minute()
}

fn transformMonth(t time.Time) int {
return int(t.Month())
}

fn transformRound(tfa: &TransformFuncArg) -> Result<Vec<Timeseries>, Error> {
    args = tfa.args
    if args.len() != 1 && args.len() != 2 {
        return nil;, fmt.Errorf(`unexpected number of args: %d;
        want
        1
        or
        2`, args.len())
    }
    let nearestArg: Vec<Timeseries>;
    if args.len() == 1 {
        nearestArg = evalNumber(tfa.ec, 1)
    } else {
        nearestArg = args[1]
    }
    nearest = getScalar(nearestArg, 1)?;
    let tf = | mut values & [f64] | {
        let mut nPrev: f64;
        let mut p10: f64;
        for (i, v) in values.iter_mut().enumerate() {
            let n = nearest[i];
            if n != nPrev {
                nPrev = n
                _, e = decimal.FromFloat(n)
                p10 = math.Pow10(int(-e))
            }
            v += 0.5 * math.Copysign(n, v)
            v -= math.Mod(v, n)
            v, _ = math.Modf(v * p10)
                * v = *v / p10;
        }
    }
    return doTransformValues(args[0], tf, tfa.fe);
}

fn transformSign(tfa: &TransformFuncArg) -> Result<Vec<Timeseries>, Error> {
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
    expectTransformArgsNum(args, 1)?;
    return doTransformValues(args[0], tf, tfa.fe);
}

fn transformScalar(tfa: &TransformFuncArg) -> Result<Vec<Timeseries>, Error> {
    args = tfa.args
    expectTransformArgsNum(args, 1);

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
    return evalNumber(tfa.ec, n);
}

// The arg isn't a string. Extract scalar from it.
    arg  = args[0]
    if len(arg) != 1 {
    return evalNumber(tfa.ec, nan), nil
    }
    arg[0].metric_name.Reset()
    return arg
}

fn newTransformFuncSortByLabel(isDesc bool) transformFunc {
|tfa: &TransformFuncArg| -> Result <Vec < Timeseries >, Error > {
args = tfa.args
if args.len() < 2 {
return nil, fmt.Errorf("expecting at least 2 args; got %d args", args.len())
}
let mut labels: Vec < string > = Vec::with_capacity(1);
for i, arg: = range args[1: ] {
label, err = getString(arg, 1)
if err != nil {
return nil, fmt.Errorf("cannot parse label #%d for sorting: %w", i+ 1, err)
}
labels = append(labels, label)
}
rvs = args[0]
sort.SliceStable(rvs, func(i, j int) bool {
for _, label = range labels {
a = rvs[i].metric_name.get_tag_value(label)
b = rvs[j].metric_name.get_tag_value(label)
if string(a) == string(b) {
continue
}
if isDesc {
return string(b) < string(a)
}
return string(a) < string(b)
}
return false
})
return rvs, nil
}
}

fn newTransformFuncSort(isDesc: bool) -> TransformFunc {
    return |tfa: &TransformFuncArg| -> Result<Vec<Timeseries>, Error> {
        args = tfa.args
        expectTransformArgsNum(args, 1);
        rvs = args[0]
        sort.Slice(rvs, func(i, j int) bool {
            a = rvs[i].Values
            b  = rvs[j].Values
            n: = len(a) - 1
            while n > = 0 {
            if ! math.IsNaN(a[n]) & & ! math.IsNaN(b[n]) & & a[n] != b[n] {
            break
            }
            n = n - 1;
            }
            if n < 0 {
            return false
            }
            if isDesc {
            return b[n] < a[n]
            }
            return a[n] < b[n]
        })
        return rvs;
    };
}

#[inline]
fn transformSqrt(v: f64) -> f64 {
    v.sqrt()
}

#[inline]
fn transformSin(v: f64) -> f64 {
    v.sin()
}

#[inline]
fn transformCos(v: f64) -> f64 {
    v.cos()
}

#[inline]
fn transformAsin(v: f64) -> f64 {
    return math.Asin(v);
}

#[inline]
fn transformAcos(v: f64) -> f64 {
    return math.Acos(v);
}

fn newTransformRand(newRandFunc: fn(r: rand.Rand) func() float64) -> TransformFunc {
return |tfa: &TransformFuncArg) -> Result<Vec<Timeseries>, Error> {
args = tfa.args
if args.len() > 1 {
return nil, fmt.Errorf(`unexpected number of args; got % d; want 0 or 1`, args.len())
}
var seed int64
if args.len() == 1 {
tmp = getScalar(args[0], 0) ?;
seed = int64(tmp[0])
} else {
seed = time.Now().UnixNano()
}
source = rand.NewSource(seed)
r = rand.New(source)
randFunc: = newRandFunc(r)
tss = evalNumber(tfa.ec, 0)
values = tss[0].Values
for i = range values {
values[i] = randFunc()
}
return tss, nil
}
}

fn newRandFloat64(r *rand.Rand) func() -> f64 {
return r.Float64
}

fn newRandNormFloat64(r *rand.Rand) func() -> f64 {
return r.NormFloat64
}

fn newRandExpFloat64(r *rand.Rand) func() -> f64 {
return r.ExpFloat64
}

fn transformPi(tfa: &TransformFuncArg) -> Result<Vec<Timeseries>, Error> {
    expectTransformArgsNum(tfa.args, 0);
    evalNumber(tfa.ec, math.Pi);
}

fn transformTime(tfa: &TransformFuncArg) -> Result<Vec<Timeseries>, Error> {
    expectTransformArgsNum(tfa.args, 0);
    return evalTime(tfa.ec);
}

fn transformVector(tfa: &TransformFuncArg) -> Result<Vec<Timeseries>, Error> {
    expectTransformArgsNum(args, 1);
    return tfs.args[0];
}

fn transformYear(t time.Time) int {
return t.Year()
}

fn newTransformFuncZeroArgs(f: fn(tfa: TransformFuncArg) -> f64) -> TransformFunc {
    |tfa: &TransformFuncArg| -> Result<Vec<Timeseries>, Error> {
        expectTransformArgsNum(tfa.args, 0)?;
        let v = f(tfa);
        evalNumber(tfa.ec, v);
    }
}

fn transformStep(tfa: &TransformFuncArg) -> f64 {
    return float64(tfa.ec.Step) / 1e3;
}

fn transformStart(tfa: &TransformFuncArg) -> f64 {
    return float64(tfa.ec.Start) / 1e3;
}

fn transformEnd(tfa: &TransformFuncArg) -> f64 {
    return float64(tfa.ec.End) / 1e3;
}

// copyTimeseriesMetricNames returns a copy of tss with real copy of MetricNames,
// but with shallow copy of Timestamps and Values if makeCopy is set.
//
// Otherwise tss is returned.
fn copyTimeseriesMetricNames(tss: &Vec<Timeseries>, makeCopy: bool) -> Vec<Timeseries> {
    if !makeCopy {
        return tss;
    }
    rvs = make([] * timeseries, len(tss))
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
fn copyTimeseriesShallow(arg: &Vec<Timeseries>) -> Vec<Timeseries> {
    rvs = make([] * timeseries, len(arg))
    for (i, src) in arg.iter().enumerate() {
        var dst timeseries
        dst.CopyShallow(src)
        rvs[i] = & dst,
    }
    return rvs;
}

fn getDstValue(mn: &MetricName, dstLabel: &str) -> &[u8] {
if dstLabel == "__name__" {
return &mn.metric_group;
}
tags = mn.Tags
for i = range tags {
tag = & tags[i]
if string(tag.Key) == dstLabel {
return & tag.Value
}
}
if len(tags) < cap(tags) {
tags = tags[: len(tags) +1]
} else {
tags = append(tags, storage.Tag{})
}
mn.Tags = tags
tag = & tags[len(tags) - 1]
tag.Key = append(tag.Key[: 0], dstLabel...)
return & tag.Value
}

fn isLeapYear(y uint32) bool {
if y % 4 != 0 {
return false
}
if y % 100 != 0 {
return true
}
return y % 400 == 0
}

var daysInMonth = [...]int{
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

fn expectTransformArgsNum(args [][] * timeseries, expectedNum int) -> Result<(), Error> {
if args.len() == expectedNum {
return nil
}
return fmt.Errorf(`unexpected number of args; got % d; want % d`, args.len(), expectedNum)
}

fn removeCounterResetsMaybeNaNs(mut values: &[f64]) {
    values = skipLeadingNaNs(values);
    if values.len() == 0 {
        return;
    }
    let mut correction: f64 = 0.0;
    let mut prev_value = values[0];
    for (i, v) in values.iter_mut().enumerate() {
        if v.is_nan() {
            continue
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