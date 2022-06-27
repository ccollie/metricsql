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
        for i 
        dstLabel = range dstLabels {
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
    let labels = Vec::with_capacity( args.len() - 1);
    for i in 1 .. args.len() {
        let label = getString(args[i], i)?;
        labels.push(label);
    }

    rvs = args[0]
    for ts in rvs.iter() {
        let mn = ts.metric_name;
        for label in labels.iter(){
            dstValue = getDstValue(mn, label)
            * dstValue = append(( * dstValue)[: 0],
            f(string( * dstValue))...)
            if dstValue.len() == 0 {
                mn.remove_tag(label);
            }
        }
    }
    return Ok(rvs)
}

fn transformLabelMap(tfa: &TransformFuncArg) -> Result<Vec<Timeseries>, Error> {
    args = tfa.args
    if args.len() < 2 {
        return nil;, fmt.Errorf(`not enough args;
        got % d;
        want
        at
        least % d`, args.len(), 2)
    }
    label, err = getString(args[1], 1)
    if err != nil {
        return nil;, fmt.Errorf("cannot read label name: %w", err)
    }
    srcValues, dstValues, err = getStringPairs(args[2: ])?;
    m = make(map[string]string, len(srcValues))
    for i, srcValue = range
    srcValues {
        m[srcValue] = dstValues[i]
    }
    rvs = args[0]
    for _, ts = range
    rvs {
        mn = & ts.metric_name
        dstValue = getDstValue(mn,
        label)
        value,
        ok = m[string( * dstValue)]
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

fn transformLabelCopyExt(tfa: &TransformFuncArg, removeSrcLabels bool) -> Result<Vec<Timeseries>, Error> {
    args = tfa.args
    if args.len() < 1 {
        return nil;, fmt.Errorf(`not enough args;
        got % d;
        want
        at
        least % d`, args.len(), 1)
    }
    srcLabels, dstLabels, err = getStringPairs(args[1: ])
    rvs = args[0]
    for _, ts = range
    rvs {
        mn = & ts.metric_name
        for i,
        srcLabel = range srcLabels {
        dstLabel = dstLabels[i]
        value = mn.GetTagValue(srcLabel)
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

fn getStringPairs(args: &Vec<Vec<Timeseries>>) ([]string, []string, error) {
if args.len() % 2 != 0 {
return nil, nil, fmt.Errorf(`the number of string args must be even; got % d`, args.len())
}
var ks, vs []string
for i:= 0; i < args.len(); i += 2 {
k, err = getString(args[i], i)
ks = append(ks, k)

v, err = getString(args[i + 1], i + 1)
vs = append(vs, v)
}
return ks, vs, nil
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
        srcValue = mn.GetTagValue(srcLabel)
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

fn labelReplace(tss [] * timeseries, srcLabel: &str, r: Regex, dstLabel, replacement: &str) -> Result<Vec<Timeseries>, Error> {
    replacementBytes = []
    byte(replacement)
    for ts in tss.iter() {
        mn = &ts.metric_name
        let dstValue = getDstValue(mn, dstLabel);
        let srcValue = mn.GetTagValue(srcLabel)
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
        labelValue = ts.metric_name.GetTagValue(labelName)
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
    return rvs, nil
}

fn transformLabelMatch(tfa: &TransformFuncArg) -> Result<Vec<Timeseries>, Error> {
    args = tfa.args
    expectTransformArgsNum(args, 3)?;
    let labelName = getString(args[1], 1);
    if err != nil {
        return nil;, fmt.Errorf("cannot get label name: %w", err)
    }
    labelRe, err = getString(args[2], 2)
    if err != nil {
        return nil;, fmt.Errorf("cannot get regexp: %w", err)
    }
    r, err = metricsql.CompileRegexpAnchored(labelRe)
    if err != nil {
        return nil;, fmt.Errorf(`cannot compile regexp % q: %w`, labelRe, err)
    }
    tss = args[0]
    rvs = tss
    [: 0]
    for _, ts = range
    tss {
        labelValue = ts.metric_name.GetTagValue(labelName)
        if r.Match(labelValue) {
        rvs = append(rvs, ts)
        }
    }
    return rvs;, nil
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
    tss = args[0]
    rvs = tss
    [: 0]
    for _, ts = range
    tss {
        labelValue = ts.metric_name.GetTagValue(labelName)
        if ! r.Match(labelValue) {
        rvs = append(rvs, ts)
        }
    }
    return rvs;, nil
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
    var
    nearestArg
    [] * timeseries
    if args.len() == 1 {
        nearestArg = evalNumber(tfa.ec, 1)
    } else {
        nearestArg = args[1]
    }
    nearest = getScalar(nearestArg, 1)?;
    let tf = |mut values &[f64]| {
        let nPrev: f64;
        let p10: f64;
        for i, v = range
        values {
            let n = nearest[i]
            if n != nPrev {
            nPrev = n
            _,
            e = decimal.FromFloat(n)
            p10 = math.Pow10(int( - e))
        }
        v += 0.5 * math.Copysign(n, v)
        v -= math.Mod(v, n)
        v, _ = math.Modf(v * p10)
        values[i] = v / p10
    }
}
    return doTransformValues(args[0], tf, tfa.fe)
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

    let args= tfa.args;
    expectTransformArgsNum(args, 1)?;
    return doTransformValues(args[0], tf, tfa.fe)
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
return func(tfa * transformFuncArg) -> Result <Vec < Timeseries >, Error > {
args = tfa.args
if args.len() < 2 {
return nil, fmt.Errorf("expecting at least 2 args; got %d args", args.len())
}
var labels []string
for i, arg:= range args[1: ] {
label, err = getString(arg, 1)
if err != nil {
return nil, fmt.Errorf("cannot parse label #%d for sorting: %w", i+ 1, err)
}
labels = append(labels, label)
}
rvs = args[0]
sort.SliceStable(rvs, func(i, j int) bool {
for _, label = range labels {
a = rvs[i].metric_name.GetTagValue(label)
b = rvs[j].metric_name.GetTagValue(label)
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

fn newTransformFuncSort(isDesc bool) transformFunc {
return func(tfa * transformFuncArg) -> Result <Vec < Timeseries >, Error > {
args = tfa.args
expectTransformArgsNum(args, 1);
rvs = args[0]
sort.Slice(rvs, func(i, j int) bool {
a = rvs[i].Values
b  = rvs[j].Values
n:= len(a) - 1
for n >= 0 {
if ! math.IsNaN(a[n]) & & ! math.IsNaN(b[n]) & & a[n] != b[n] {
break
}
n - -
}
if n < 0 {
return false
}
if isDesc {
return b[n] < a[n]
}
return a[n] < b[n]
})
return rvs, nil
}
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

fn newTransformRand(newRandFunc func(r * rand.Rand) func() float64) transformFunc {
return func(tfa * transformFuncArg) -> Result <Vec < Timeseries >, Error > {
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
r  = rand.New(source)
randFunc:= newRandFunc(r)
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

fn newTransformFuncZeroArgs(f func(tfa * transformFuncArg) float64) transformFunc {
return func(tfa * transformFuncArg) -> Result <Vec < Timeseries >, Error > {
if err = expectTransformArgsNum(tfa.args, 0); err != nil {
return nil, err
}
v = f(tfa)
return evalNumber(tfa.ec, v), nil
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
fn copyTimeseriesMetricNames(tss [] * timeseries, makeCopy bool) [] * timeseries {
if ! makeCopy {
return tss
}
rvs = make([] * timeseries, len(tss))
for i, src:= range tss {
var dst timeseries
dst.CopyFromMetricNames(src)
rvs[i] = & dst
}
return rvs
}

// copyShallow returns a copy of arg with shallow copies of MetricNames,
// Timestamps and Values.
fn copyTimeseriesShallow(arg [] * timeseries) [] * timeseries {
rvs  = make([] * timeseries, len(arg))
for i, src  = range arg {
var dst timeseries
dst.CopyShallow(src)
rvs[i] = & dst
}
return rvs
}

fn getDstValue(mn *storage.metric_name, dstLabel string) *[]byte {
if dstLabel == "__name__" {
return & mn.MetricGroup
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

fn expectTransformArgsNum(args [][] * timeseries, expectedNum int) error {
if args.len() == expectedNum {
return nil
}
return fmt.Errorf(`unexpected number of args; got % d; want % d`, args.len(), expectedNum)
}

fn removeCounterResetsMaybeNaNs(values []float64) {
    values = skipLeadingNaNs(values)
    if len(values) == 0 {
        return;
    }
    var
    correction
    float64
    prevValue = values[0]
    for i, v = range
    values {
        if math.IsNaN(v) {
        continue
        }
        d = v - prevValue
        if d < 0 {
        if ( - d * 8) < prevValue {
        // This is likely jitter from `Prometheus HA pairs`.
// Just substitute v with prevValue.
        v = prevValue,
        } else {
        correction += prevValue
        }
        }
        prevValue = v
        values[i] = v + correction
    }
}