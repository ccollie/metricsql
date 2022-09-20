use std::sync::Arc;

use chrono::Duration;

use metricsql::ast::*;
use metricsql::functions::{DataType, Signature, TypeSignature, Volatility};

use crate::context::Context;
use crate::eval::aggregate::{AggregateEvaluator, create_aggr_evaluator};
use crate::eval::binaryop::BinaryOpEvaluator;
use crate::eval::duration::DurationEvaluator;
use crate::eval::function::{create_function_evaluator, TransformEvaluator};
use crate::eval::number::NumberEvaluator;
use crate::eval::string::StringEvaluator;
use crate::functions::types::ParameterValue;
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::search::Deadline;
use crate::timeseries::Timeseries;
use crate::traits::Timestamp;

use super::rollup::RollupEvaluator;
use super::traits::{Evaluator, NullEvaluator};

pub(crate) enum ExprEvaluator {
    Null(NullEvaluator),
    Aggregate(AggregateEvaluator),
    BinaryOp(BinaryOpEvaluator),
    Duration(DurationEvaluator),
    Function(TransformEvaluator),
    Number(NumberEvaluator),
    Rollup(RollupEvaluator),
    String(StringEvaluator),
}

impl Evaluator for ExprEvaluator {
    fn eval(&self, ctx: &mut Context, ec: &EvalConfig) -> RuntimeResult<Vec<Timeseries>> {
        match self {
            ExprEvaluator::Null(e) => e.eval(ctx, ec),
            ExprEvaluator::Aggregate(ae) => ae.eval(ctx, ec),
            ExprEvaluator::BinaryOp(bo) => bo.eval(ctx, ec),
            ExprEvaluator::Duration(de) => de.eval(ctx, ec),
            ExprEvaluator::Function(fe) => fe.eval(ctx, ec),
            ExprEvaluator::Number(n) => n.eval(ctx, ec),
            ExprEvaluator::Rollup(ref mut re) => re.eval(ctx, ec),
            ExprEvaluator::String(se) => se.eval(ctx, ec),
        }
    }
}

impl Default for ExprEvaluator {
    fn default() -> Self {
        ExprEvaluator::Null(NullEvaluator{})
    }
}

impl From<i64> for ExprEvaluator {
    fn from(val: i64) -> Self {
        Self::Number(NumberEvaluator::from(val as f64))
    }
}

impl From<f64> for ExprEvaluator {
    fn from(val: f64) -> Self {
        Self::Number(NumberEvaluator::from(val))
    }
}

impl From<String> for ExprEvaluator {
    fn from(val: String) -> Self {
        Self::String(StringEvaluator::new(&val))
    }
}

impl From<&str> for ExprEvaluator {
    fn from(val: &str) -> Self {
        Self::String(StringEvaluator::new(val))
    }
}

pub fn create_evaluator(expr: &Expression) -> RuntimeResult<ExprEvaluator> {
    match expr {
        Expression::Aggregation(ae) => create_aggr_evaluator(ae),
        Expression::MetricExpression(me) => {
            Ok(ExprEvaluator::Rollup(RollupEvaluator::from_metric_expression(me.clone())?))
        }
        Expression::Rollup(re) => Ok(ExprEvaluator::Rollup(RollupEvaluator::new(re)?)),
        Expression::Function(fe) => create_function_evaluator(fe),
        Expression::BinaryOperator(be) => Ok(
            ExprEvaluator::BinaryOp(BinaryOpEvaluator::new(be)?)
        ),
        Expression::Number(ne) => Ok(ExprEvaluator::from(ne.value)),
        Expression::String(se) => Ok(ExprEvaluator::from(se.value())),
        Expression::Duration(de) => {
            if de.requires_step {
                Ok(ExprEvaluator::Duration(DurationEvaluator::new(de)))
            } else {
                Ok(ExprEvaluator::from(de.const_value ))
            }
        },
        Expression::With(_) => {
            panic!("unexpected WITH expression - {}: Should have been expanded during parsing", expr);
        }
        _ => {
            panic!("unexpected expression {}: ", expr);
        }
    }
}

pub(crate) fn create_evaluators(vec: &Vec<BExpression>) -> RuntimeResult<Vec<ExprEvaluator>> {
    let mut res: Vec<ExprEvaluator> = Vec::with_capacity(vec.len());
    for arg in vec {
        match create_evaluator(arg) {
            Err(e) => return Err(e),
            Ok(eval) => { res.push(eval) },
        }
    }
    Ok(res)
}

/// The minimum number of points per timeseries for enabling time rounding.
/// This improves cache hit ratio for frequently requested queries over
/// big time ranges.
pub static MIN_TIMESERIES_POINTS_FOR_TIME_ROUNDING: usize = 50;

/// validate_max_points_per_timeseries checks the maximum number of points that
/// may be returned per each time series.
///
/// The number mustn't exceed -search.maxPointsPerTimeseries.
pub(crate) fn validate_max_points_per_timeseries(
    start: Timestamp,
    end: Timestamp,
    step: i64,
    max_points_per_timeseries: usize,
) -> RuntimeResult<()> {
    let points = (end - start) / step + 1;
    if (max_points_per_timeseries > 0) && points > max_points_per_timeseries as i64 {
        let msg = format!("too many points for the given step={}, start={} and end={}: {}; cannot exceed -search.maxPointsPerTimeseries={}",
                          step, start, end, points, max_points_per_timeseries);
        Err(RuntimeError::from(msg))
    } else {
        Ok(())
    }
}

pub fn adjust_start_end(start: Timestamp, end: Timestamp, step: i64) -> (i64, i64) {
    // if disableCache {
    //     // do not adjust start and end values when cache is disabled.
    //     // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/563
    //     return (start, end);
    // }
    let points = (end - start) / step + 1;
    if points < MIN_TIMESERIES_POINTS_FOR_TIME_ROUNDING as i64 {
        // Too small number of points for rounding.
        return (start, end);
    }

    // Round start and end to values divisible by step in order
    // to enable response caching (see EvalConfig.mayCache).
    let (start, end) = align_start_end(start, end, step);

    // Make sure that the new number of points is the same as the initial number of points.
    let mut new_points = (end - start) / step + 1;
    let mut _end = end;
    while new_points > points {
        _end = end - step;
        new_points -= 1;
    }

    return (start, _end);
}

pub fn align_start_end(start: i64, end: i64, step: i64) -> (i64, i64) {
    // Round start to the nearest smaller value divisible by step.
    let new_start = start - start % step;
    // Round end to the nearest bigger value divisible by step.
    let adjust = end % step;
    let mut new_end = end;
    if adjust > 0 {
        new_end += step - adjust
    }
    return (new_start, new_end);
}

#[derive(Copy, Clone, Debug)]
pub struct EvalOptions {
    /// Set this flag to true if the database doesn't contain Prometheus stale markers, so there is
    /// no need in spending additional CPU time on its handling. Staleness markers may exist only in
    /// data obtained from Prometheus scrape targets
    pub no_stale_markers: bool,

    /// The maximum interval for staleness calculations. By default it is automatically calculated from
    /// the median interval between samples. This could be useful for tuning Prometheus data model
    /// closer to Influx-style data model.
    /// See https://prometheus.io/docs/prometheus/latest/querying/basics/#staleness for details.
    /// See also '-search.setLookbackToStep' flag
    pub max_staleness_interval: Duration,

    /// The minimum interval for staleness calculations. This could be useful for removing gaps on
    /// graphs generated from time series with irregular intervals between samples.
    pub min_staleness_interval: Duration,
}

impl EvalOptions {
    pub fn new() -> Self {
        EvalOptions::default()
    }
}

impl Default for EvalOptions {
    fn default() -> Self {
        Self {
            no_stale_markers: true,
            max_staleness_interval: Duration::milliseconds(0),
            min_staleness_interval: Duration::milliseconds(0),
        }
    }
}

#[derive(Clone)]
pub struct EvalConfig {
    pub start: Timestamp,
    pub end: Timestamp,
    pub step: Timestamp,

    /// max_series is the maximum number of time series, which can be scanned by the query.
    /// Zero means 'no limit'
    pub max_series: usize,

    /// quoted remote address.
    pub quoted_remote_addr: Option<String>,

    pub deadline: Deadline,

    /// Whether the response can be cached.
    _may_cache: bool,

    /// lookback_delta is analog to `-query.lookback-delta` from Prometheus.
    pub lookback_delta: i64,

    /// How many decimal digits after the point to leave in response.
    pub round_digits: i16,

    /// enforced_tag_filterss may contain additional label filters to use in the query.
    pub enforced_tag_filterss: Vec<Vec<LabelFilter>>,

    pub options: EvalOptions,

    /// The limit on the number of points which can be generated per each returned time series.
    pub max_points_per_series: usize,

    /// Whether to disable response caching. This may be useful during data backfilling
    pub disable_cache: bool,

    _timestamps: Arc<Vec<i64>>,
}

impl EvalConfig {
    pub fn new() -> Self {
        EvalConfig {
            start: 0,
            end: 0,
            step: 0,
            max_series: 0,
            quoted_remote_addr: None,
            deadline: Deadline::default(),
            _may_cache: false,
            lookback_delta: 0,
            round_digits: 0,
            enforced_tag_filterss: vec![],
            options: EvalOptions::default(),
            max_points_per_series: 0,
            disable_cache: false,
            _timestamps: Arc::new(vec![])
        }
    }

    pub fn copy_no_timestamps(self) -> EvalConfig {
        let ec = EvalConfig {
            start: self.start,
            end: self.end,
            step: self.step,
            deadline: self.deadline,
            max_series: self.max_series,
            quoted_remote_addr: self.quoted_remote_addr.clone(),
            _may_cache: self._may_cache,
            lookback_delta: self.lookback_delta,
            round_digits: self.round_digits,
            enforced_tag_filterss: vec![],
            // do not copy src.timestamps - they must be generated again.
            _timestamps: Arc::new(vec![]),
            options: EvalOptions::default(),
            max_points_per_series: self.max_points_per_series,
            disable_cache: self.disable_cache,
        };
        return ec;
    }

    pub fn validate(&self) -> RuntimeResult<()> {
        if self.start > self.end {
            let msg = format!(
                "BUG: start cannot exceed end; got {} vs {}",
                self.start, self.end
            );
            return Err(RuntimeError::from(msg));
        }
        if self.step <= 0 {
            let msg = format!("BUG: step must be greater than 0; got {}", self.step);
            return Err(RuntimeError::from(msg));
        }
        Ok(())
    }

    pub fn may_cache(&self) -> bool {
        if self.disable_cache {
            return false;
        }
        if self._may_cache {
            return true;
        }
        if self.start % self.step != 0 {
            return false;
        }
        if self.end % self.step != 0 {
            return false;
        }

        true
    }

    pub fn timestamps<'a>(&self) -> &'a Arc<Vec<i64>> {
        &self._timestamps
    }

    pub fn get_timestamps<'a>(&mut self) -> &'a Arc<Vec<i64>> {
        self.ensure_timestamps().unwrap(); //???
        &self._timestamps
    }

    pub(crate) fn ensure_timestamps(&mut self) -> RuntimeResult<()> {
        if self._timestamps.len() == 0 {
            let ts = get_timestamps(
                self.start,
                self.end,
                self.step,
                self.max_points_per_series as usize
            )?;
            self._timestamps = Arc::new(ts);
        }
        Ok(())
    }

    pub fn get_shared_timestamps(&mut self) -> &Arc<Vec<i64>> {
        self.get_timestamps()
    }
}

pub fn get_timestamps(start: i64, end: i64, step: i64, max_timestamps_per_timeseries: usize) -> RuntimeResult<Vec<i64>> {
    // Sanity checks.
    if step <= 0 {
        let msg = format!("BUG: Step must be bigger than 0; got {}", step);
        return Err(RuntimeError::from(msg));
    }
    if start > end {
        let msg = format!("BUG: Start cannot exceed End; got {} vs {}", start, end);
        return Err(RuntimeError::from(msg));
    }
    match validate_max_points_per_timeseries(start, end, step, max_timestamps_per_timeseries) {
        Err(err) => {
            let msg = format!(
                "BUG: {:?}; this must be validated before the call to get_timestamps",
                err
            );
            return Err(RuntimeError::from(msg));
        }
        _ => (),
    }

    // Prepare timestamps.
    let n: usize = (1 + (end - start) / step) as usize;
    let mut timestamps: Vec<i64> = Vec::with_capacity(n);
    let mut cursor = start;
    for i in 0..n {
        timestamps.push(cursor);
        cursor += step;
    }
    return Ok(timestamps);
}

// copy_eval_config returns src copy.
pub(crate) fn copy_eval_config(src: &EvalConfig) -> EvalConfig {
    src.copy_no_timestamps()
}

pub(crate) fn eval_args(
    ctx: &mut Context,
    ec: &mut EvalConfig,
    args: &Vec<ExprEvaluator>,
) -> RuntimeResult<Vec<Vec<Timeseries>>> {
    let mut res: Vec<Vec<Timeseries>> = Vec::with_capacity(args.len());
    for evaluator in args {
        match evaluator.eval(ctx, ec) {
            Err(e) => return Err(e),
            Ok(val) => res.push(val)
        }
    }
    Ok(res)
}

pub(crate) fn eval_number(ec: &EvalConfig, n: f64) -> Vec<Timeseries> {
    let timestamps = ec.timestamps();
    let values = vec![n; timestamps.len()];
    let ts = Timeseries::with_shared_timestamps(timestamps, &values);
    vec![ts]
}

pub(crate) fn eval_string(ec: &EvalConfig, s: &str) -> Vec<Timeseries> {
    let mut rv = eval_number(ec, f64::NAN);
    rv[0].metric_name.metric_group = s.to_string();
    rv
}

pub(crate) fn eval_time(ec: &EvalConfig) -> Vec<Timeseries> {
    let mut rv = eval_number(ec, f64::NAN);
    for (i, ts) in rv[0].timestamps.iter().enumerate() {
        rv[0].values[i] = *ts as f64 / 1e3_f64;
    }
    rv
}


fn get_string_arg(arg: &Vec<Timeseries>, arg_num: usize) -> RuntimeResult<String> {
    if arg.len() != 1 {
        let msg = format!(
            "arg # {} must contain a single timeseries; got {} timeseries",
            arg_num + 1,
            arg.len()
        );
        return Err(RuntimeError::ArgumentError(msg));
    }
    let ts = &arg[0];
    let all_nan = arg[0].values.iter().all(|x| x.is_nan());
    if !all_nan {
        let msg = format!("arg # {} contains non - string timeseries", arg_num + 1);
        return Err(RuntimeError::ArgumentError(msg));
    }
    // todo: return reference
    Ok(ts.metric_name.metric_group.clone())
}

fn get_label(arg: &Vec<Timeseries>, name: &str, arg_num: usize) -> RuntimeResult<String> {
    match get_string_arg(arg, arg_num) {
        Ok(lbl) => Ok(lbl),
        Err(err) => {
            let msg = format!("cannot read {} label name", name);
            return Err(RuntimeError::ArgumentError(msg));
        }
    }
}

pub(super) fn get_scalar(arg: &Vec<Timeseries>, arg_num: usize) -> RuntimeResult<&Vec<f64>> {
    if arg.len() != 1 {
        let msg = format!(
            "arg # {} must contain a single timeseries; got {} timeseries",
            arg_num + 1,
            arg.len()
        );
        return Err(RuntimeError::ArgumentError(msg))
    }
    Ok(&arg[arg_num].values)
}

#[inline]
fn get_int_number(arg: &Vec<Timeseries>, arg_num: usize) -> RuntimeResult<i64> {
    let v = get_float(arg, arg_num)?;
    Ok(v as i64)
}

#[inline]
fn get_float(arg: &Vec<Timeseries>, arg_num: usize) -> RuntimeResult<f64> {
    let v = get_scalar(arg, arg_num)?;
    let mut n = 0_f64;
    if v.len() > 0 {
        n = v[0];
    }
    Ok(n)
}

pub(crate) fn eval_param(
    ctx: &mut Context,
    ec: &EvalConfig,
    expr: &ExprEvaluator,
    expected: &DataType,
    arg_num: usize) -> RuntimeResult<ParameterValue> {
    let val = expr.eval(ctx, ec)?;
    Ok(match expected {
        DataType::Matrix => {
            // this is likely incorrect
            ParameterValue::Matrix(vec![val])
        }
        DataType::Series => {
            ParameterValue::Series(val)
        },
        DataType::Vector => {
            let val = get_scalar(&val, arg_num)?;
            ParameterValue::Vector(val.into_iter().collect())
        },
        DataType::Float => {
            let val = get_float(&val, arg_num)?;
            ParameterValue::Float(val)
        },
        DataType::Int => {
            let val = get_int_number(&val, arg_num)?;
            ParameterValue::Int(val)
        },
        DataType::String => {
            let val = get_string_arg(&val, arg_num)?;
            ParameterValue::String(val)
        }
    })
}


pub(crate) fn validate_params(
    name: &str,
    signature: &Signature,
    args: &[ExprEvaluator]
) -> RuntimeResult<()> {

    let (arg_types, min) = signature.expand_types();
    match signature.type_signature.validate_arg_count(name, args.len()) {
        Err(e) => {
            let msg = format!("{:?}", e);
            return Err(RuntimeError::ArgumentError(msg))
        },
        _ => {}
    }

    for (i, arg)  in args.iter().enumerate()  {
        let expected = arg_types[i];
        let actual = arg.return_type();

        // todo:
        let mut valid = false;
        todo!()
    }

    Ok(())
}

pub fn eval_params(
    ctx: &mut Context,
    ec: &EvalConfig,
    signature: &TypeSignature,
    args: &[ExprEvaluator]
) -> RuntimeResult<Vec<ParameterValue>> {

    fn eval_param_variadic(
        ctx: &mut Context,
        ec: &EvalConfig,
        args: &[ExprEvaluator],
        types: &Vec<DataType>
    ) -> RuntimeResult<Vec<ParameterValue>> {
        // todo: tinyvec
        let mut params: Vec<ParameterValue> = Vec::with_capacity(types.len());
        for (i, expected) in types.iter().enumerate() {
            let arg = eval_param(ctx, ec, &args[i], expected, i)?;
            params.push(arg);
        }
        Ok(params)
    }

    return match signature {
        TypeSignature::Variadic(valid_types, min) => {
            eval_param_variadic(ctx, ec, args, valid_types)
        },
        TypeSignature::Uniform(number, valid_type) => {
            let mut params: Vec<ParameterValue> = Vec::with_capacity(args.len());
            for i in 0 .. *number {
                let arg = eval_param(ctx, ec, &args[i], valid_type, i)?;
                params.push(arg);
            }
            Ok(params)
        },
        TypeSignature::VariadicEqual(data_type, min) => {
            let mut params: Vec<ParameterValue> = Vec::with_capacity(args.len());
            for (i, evaluator) in args.iter().enumerate() {
                let arg = eval_param(ctx, ec, evaluator, data_type, i)?;
                params.push(arg);
            }
            Ok(params)
        }
        TypeSignature::Exact(valid_types) => {
            eval_param_variadic(ctx, ec, args, valid_types)
        },
        TypeSignature::Any(number) => {
            let mut params: Vec<ParameterValue> = Vec::with_capacity(*number);
            for i in 0 .. *number {
                let arg = eval_param(ctx, ec, &args[i], &DataType::Series, i)?;
                params.push(arg);
            }
            Ok(params)
        }
    }
}

#[inline]
fn should_parallelize(t: &DataType) -> bool {
    !matches!(t, DataType::String | DataType::Float | DataType::Int | DataType::Vector )
}

/// Determines if we should parallelize parameter parsing. We ignore "lightweight"
/// parameter types like `String` or `Int`
pub(crate) fn should_parallelize_param_parsing(signature: &Signature) -> bool {
    let types = &signature.type_signature;

    fn check_args(valid_types: &[DataType]) -> bool {
        valid_types.iter().filter(|x| should_parallelize(*x)).count() > 1
    }

    return match types {
        TypeSignature::Variadic(valid_types, min) => {
            check_args(valid_types)
        },
        TypeSignature::Uniform(number, valid_type) => {
            let types = &[*valid_type];
            return *number >= 2 && check_args(types)
        },
        TypeSignature::VariadicEqual(data_type, _) => {
            let types = &[*data_type];
            check_args(types)
        }
        TypeSignature::Exact(valid_types) => {
            check_args(valid_types)
        },
        TypeSignature::Any(number) => {
            if *number < 2 {
                return false
            }
            true
        }
    }
}


pub(super) fn eval_volatility(sig: &Signature, args: &Vec<ExprEvaluator>) -> Volatility {
    if sig.volatility != Volatility::Immutable {
        return sig.volatility
    }

    let mut has_volatile = false;
    let mut mutable = false;

    for arg in args {
        let vol = arg.volatility();
        if vol != Volatility::Immutable {
            mutable = true;
            has_volatile = vol == Volatility::Volatile;
        }
    }

    if mutable {
        return if has_volatile {
            Volatility::Volatile
        } else {
            Volatility::Stable
        }
    } else {
        Volatility::Immutable
    }
}
