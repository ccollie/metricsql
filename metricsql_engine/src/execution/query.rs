use std::borrow::Cow;

use chrono::Duration;

use metricsql_parser::prelude::{DurationExpr, Expr, LabelFilter, Matchers};

use crate::execution::{
    adjust_start_end, exec, parse_promql_internal, validate_max_points_per_timeseries,
};
use crate::execution::{Context, EvalConfig};
use crate::provider::{join_matchers_with_extra_filters};
use crate::types::{Timestamp, TimestampTrait};
use crate::{
    remove_empty_values_and_timeseries, Deadline, QueryResult, QueryResults, RuntimeError,
    RuntimeResult, SearchQuery, MAX_DURATION_MSECS,
};

/// Default step used if not set.
const DEFAULT_STEP: i64 = 5 * 60 * 1000;
const TWO_DAYS_MSECS: i64 = 2 * 24 * 3600 * 1000;

#[derive(Clone, Debug)]
pub struct QueryParams {
    pub query: String,
    pub may_cache: bool,
    pub start: Timestamp, // todo: support into
    pub end: Timestamp,
    pub step: Duration,
    pub deadline: Deadline,
    pub round_digits: u8,
    pub required_tag_filters: Vec<Vec<LabelFilter>>,
}

impl Default for QueryParams {
    fn default() -> Self {
        Self {
            query: "".to_string(),
            may_cache: true,
            start: 0,
            end: 0,
            step: Duration::milliseconds(DEFAULT_STEP),
            deadline: Default::default(),
            round_digits: 100,
            required_tag_filters: vec![],
        }
    }
}

impl QueryParams {
    pub fn validate(&self, context: &Context) -> RuntimeResult<()> {
        if self.query.is_empty() {
            let msg = "no query specified".to_string();
            return Err(RuntimeError::General(msg));
        }

        if self.query.len() > context.config.max_query_len {
            let msg = format!(
                "too long query; got {} bytes; mustn't exceed `maxQueryLen={}` bytes",
                self.query.len(),
                context.config.max_query_len
            );
            return Err(RuntimeError::General(msg));
        }

        if self.start > self.end {
            let msg = format!(
                "BUG: start cannot exceed end; got {} vs {}",
                self.start, self.end
            );
            return Err(RuntimeError::from(msg));
        }

        if self.step.num_milliseconds() <= 0 {
            let msg = format!("BUG: step must be greater than 0; got {}", self.step);
            return Err(RuntimeError::from(msg));
        }

        // validate_duration('deadline', self.deadline);

        Ok(())
    }
}

pub struct QueryBuilder {
    query: String,
    start: Option<Timestamp>,
    end: Option<Timestamp>,
    step: Option<Duration>,
    timeout: Option<Duration>,
    extra_tag_filters: Vec<Vec<LabelFilter>>,
    round_digits: u8,
    no_cache: bool,
    trace_enabled: bool,
}

impl Default for QueryBuilder {
    fn default() -> Self {
        Self {
            query: "".to_string(),
            start: None,
            end: None,
            step: None,
            timeout: None,
            extra_tag_filters: vec![],
            round_digits: 100,
            no_cache: false,
            trace_enabled: false,
        }
    }
}

impl QueryBuilder {
    pub fn start<T: Into<Timestamp>>(&mut self, start: T) -> &mut Self {
        self.start = Some(start.into());
        self
    }

    pub fn end<T: Into<Timestamp>>(&mut self, end: T) -> &mut Self {
        self.end = Some(end.into());
        self
    }

    pub fn query<S: Into<String>>(&mut self, q: S) -> &mut Self {
        self.query = q.into();
        self
    }

    pub fn no_cache(&mut self) -> &mut Self {
        self.no_cache = true;
        self
    }

    pub fn enable_tracing(&mut self) -> &mut Self {
        self.trace_enabled = true;
        self
    }

    pub fn step<D: Into<Duration>>(&mut self, value: D) -> &mut Self {
        self.step = Some(value.into());
        self
    }

    pub fn timeout<D: Into<Duration>>(&mut self, value: D) -> &mut Self {
        self.timeout = Some(value.into());
        self
    }

    pub fn round_digits(&mut self, digits: u8) -> &mut Self {
        self.round_digits = digits;
        self
    }

    pub fn build(&self, context: &Context) -> RuntimeResult<QueryParams> {
        let mut q: QueryParams = QueryParams::default();

        let start = self.start.unwrap_or_else(Timestamp::now);
        let mut end = self.end.unwrap_or(start);

        // Limit the `end` arg to the current time +2 days to prevent possible timestamp overflow
        let max_ts = start + TWO_DAYS_MSECS;
        if end > max_ts {
            end = max_ts
        }
        if end < start {
            end = start
        }

        let timeout = self
            .timeout
            .unwrap_or_else(|| Duration::milliseconds(TWO_DAYS_MSECS));

        q.query = self.query.clone();
        q.start = start;
        q.end = end;
        q.may_cache = !self.no_cache;
        q.step = self
            .step
            .unwrap_or_else(|| Duration::milliseconds(DEFAULT_STEP));
        if !self.extra_tag_filters.is_empty() {
            q.required_tag_filters = self.extra_tag_filters.clone();
        }
        q.round_digits = self.round_digits;
        q.deadline = get_deadline_for_query(context, q.start, Some(timeout))?;
        Ok(q)
    }
}

/// CommonParams contains common parameters for all /api/v1/* handlers
#[derive(Debug, Default)]
struct CommonParams {
    deadline: Deadline,
    start: Timestamp,
    end: Timestamp,
    filters: Vec<Vec<LabelFilter>>,
}

/// Query handler for `Instant Queries`
///
/// See https://prometheus.io/docs/prometheus/latest/querying/api/#instant-queries
pub fn query(context: &Context, params: &QueryParams) -> RuntimeResult<Vec<QueryResult>> {
    let ct = Timestamp::now();
    let mut start = params.start;
    let mut end = params.end;
    let step_millis = params.step.num_milliseconds();

    let lookback_delta = get_max_lookback(context, step_millis, Some(TWO_DAYS_MSECS));
    let mut step = if step_millis == 0 {
        lookback_delta
    } else {
        step_millis
    };
    if step <= 0 {
        step = DEFAULT_STEP
    }

    let parsed = parse_promql_internal(context, &params.query)?;

    // Safety: at this point expr has a value. We error out above in case of failure
    let expr = parsed.expr.as_ref().unwrap();
    if let Some(rollup) = get_rollup(expr) {
        let window = rollup.window.value(step);
        let offset = rollup.offset.value(step);

        if let Some(filters) = rollup.filters {
            // metric expression without subquery
            start -= offset;
            end = start;
            start = end - window;
            // Do not include data point with a timestamp matching the lower boundary of the window as
            // Prometheus does.
            start += 1;
            if end < start {
                end = start
            }

            // Fetch the remaining part of the result.
            let tfs_list = join_matchers_with_extra_filters(&filters, &params.required_tag_filters);

            let cp = CommonParams {
                deadline: params.deadline,
                start,
                end,
                filters: tfs_list.to_vec(), // todo: avoid this
            };

            return match export_handler(context, cp) {
                Err(err) => {
                    let msg = format!("error when exporting data for query={} on the time range (start={}, end={}): {:?}",
                                      rollup.expr, start, end, err);
                    return Err(RuntimeError::General(msg));
                }
                Ok(v) => Ok(v.series),
            };
        }

        // we have a rollup with a non-empty window
        let new_step = rollup.step.value(step);
        if new_step > 0 {
            step = new_step
        }
        let window = rollup.window.value(step);
        let offset = rollup.offset.value(step);
        start -= offset;
        end = start;
        start = end - window;

        let mut params_copy = (*params).clone();
        params_copy.query = expr.to_string();
        params_copy.start = start;
        params_copy.end = end;

        return match query_range_handler(context, ct, params) {
            Err(err) => {
                let msg = format!("error when executing query={} on the time range (start={}, end={}, step={}): {:?}",
                                  &params_copy.query, start, end, step, err);
                return Err(RuntimeError::General(msg));
            }
            Ok(v) => Ok(v),
        };
    }

    let mut query_offset = get_latency_offset_milliseconds(context);
    if params.may_cache && ct - start < query_offset && start - ct < query_offset {
        // Adjust start time only if `nocache` arg isn't set.
        // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/241
        let start_prev = start;
        start = ct - query_offset;
        query_offset = start_prev - start
    } else {
        query_offset = 0
    }

    let mut ec = EvalConfig::new(start, end, step);

    if !ec.enforced_tag_filters.is_empty() {
        ec.enforced_tag_filters = params.required_tag_filters.clone(); // todo: .into()?
    }

    ec.deadline = params.deadline;

    ec.update_from_context(context);

    match exec(context, &mut ec, &params.query, true) {
        Err(err) => {
            let msg = format!(
                "error executing query={} for (time={}, step={}): {:?}",
                &params.query, start, step, err
            );
            Err(RuntimeError::General(msg))
        }
        Ok(mut result) => {
            if query_offset > 0 && !result.is_empty() {
                // todo: verify see https://github.com/VictoriaMetrics/VictoriaMetrics/commit/0771d57860d87806b2eda684cc2d3b640939bf6b
                for res in result.iter_mut() {
                    for j in res.timestamps.iter_mut() {
                        *j += query_offset;
                    }
                }
            }
            Ok(result)
        }
    }
}

fn export_handler(ctx: &Context, cp: CommonParams) -> RuntimeResult<QueryResults> {
    let max_series = &ctx.config.max_unique_timeseries;
    let sq = SearchQuery::new(cp.start, cp.end, cp.filters, *max_series);
    ctx.search(sq, cp.deadline)
}

/// query_range processes a range vector request
///
/// See https://prometheus.io/docs/prometheus/latest/querying/api/#range-queries
pub fn query_range(ctx: &Context, params: &QueryParams) -> RuntimeResult<Vec<QueryResult>> {
    let ct = Timestamp::now();
    let mut step = params.step.num_milliseconds();
    if step <= 0 {
        step = DEFAULT_STEP;
    }

    match query_range_handler(ctx, ct, params) {
        Err(err) => {
            let msg = format!(
                "error executing query={} on the time range (start={}, end={} step={}): {:?}",
                &params.query, params.start, params.end, step, err
            );
            Err(RuntimeError::General(msg))
        }
        Ok(v) => Ok(v),
    }
}

fn query_range_handler(
    ctx: &Context,
    ct: Timestamp,
    params: &QueryParams,
) -> RuntimeResult<Vec<QueryResult>> {
    let config = &ctx.config;

    let (mut start, mut end, mut step) = (params.start, params.end, params.step.num_milliseconds());

    if step <= 0 {
        step = DEFAULT_STEP
    }

    let lookback_delta = get_max_lookback(ctx, step, None);

    // Validate input args.
    if start > end {
        end = start + DEFAULT_STEP
    }

    let max_points = ctx.config.max_points_subquery_per_timeseries;
    validate_max_points_per_timeseries(start, end, step, max_points)?;

    if params.may_cache {
        (start, end) = adjust_start_end(start, end, step)
    }

    let mut ec = EvalConfig::new(start, end, step);
    ec.deadline = params.deadline;
    ec.set_caching(params.may_cache);
    ec.enforced_tag_filters = params.required_tag_filters.clone(); // todo: how to avoid this clone ??
    ec.round_digits = params.round_digits;
    ec.lookback_delta = lookback_delta;
    ec.update_from_context(ctx);

    let mut result = exec(ctx, &mut ec, &params.query, false)?;
    if step < config.max_step_for_points_adjustment.num_milliseconds() {
        let query_offset = get_latency_offset_milliseconds(ctx);
        if ct - query_offset < end {
            adjust_last_points(&mut result, ct - query_offset, ct + step)
        }
    }

    // Remove NaN values as Prometheus does.
    // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/153
    remove_empty_values_and_timeseries(&mut result);

    Ok(result)
}

/// adjust_last_points substitutes the last point values on the time range (start..end]
/// with the previous point values, since these points may contain incomplete values.
fn adjust_last_points(tss: &mut [QueryResult], start: Timestamp, end: Timestamp) {
    for ts in tss.iter_mut() {
        let n = ts.timestamps.len();
        if n <= 1 {
            continue;
        }
        let mut j = n - 1;
        if ts.timestamps[j] > end {
            // It looks like the `offset` is used in the query, which shifts time range beyond the `end`.
            // Leave such a time series as is, since it is unclear which points may be incomplete in it.
            // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/625
            continue;
        }

        for v in ts.timestamps.iter().rev() {
            if *v > start {
                j -= 1;
            } else {
                break;
            }
        }

        let mut last_value = f64::NAN;
        if j > 0 {
            last_value = ts.values[j - 1]
        }
        // todo(perf): user iter and zip to avoid bounds checks
        while j < ts.timestamps.len() && ts.timestamps[j] <= end {
            ts.values[j] = last_value;
            j += 1;
        }
    }
}

fn get_max_lookback(ctx: &Context, step: i64, max: Option<i64>) -> i64 {
    let config = &ctx.config;
    let mut d = config.max_lookback.num_milliseconds();
    if d == 0 {
        d = config.max_staleness_interval.num_milliseconds()
    }
    d = max.unwrap_or(d);
    if config.set_lookback_to_step && step > 0 {
        d = step;
    }
    d
}

struct DeconstructedRollup<'a> {
    filters: Option<&'a Matchers>,
    window: &'a DurationExpr,
    offset: Cow<'a, DurationExpr>,
    step: Cow<'a, DurationExpr>,
    expr: &'a Expr,
}

fn get_rollup(expr: &Expr) -> Option<DeconstructedRollup> {
    match &expr {
        Expr::Rollup(re) => {
            if let Some(window) = &re.window {
                let mut res = DeconstructedRollup {
                    filters: None,
                    window,
                    offset: get_duration_expr(&re.offset),
                    step: get_duration_expr(&re.step),
                    expr: &re.expr,
                };

                if re.step.is_none() {
                    // check whether expr contains PromQL metric selector wrapped into rollup.
                    if let Expr::MetricExpression(me) = &re.expr.as_ref() {
                        if !me.matchers.is_empty() {
                            res.filters = Some(&me.matchers);
                        }
                    }
                    // todo: see if we have default_rollup(metric{job="email"})
                }
                return Some(res);
            }
        }
        _ => return None,
    }

    None
}

fn get_duration_expr(offset: &Option<DurationExpr>) -> Cow<DurationExpr> {
    return match &offset {
        Some(ofs) => Cow::Borrowed(ofs),
        None => Cow::Owned(DurationExpr::default()),
    };
}

fn get_latency_offset_milliseconds(ctx: &Context) -> i64 {
    std::cmp::min(ctx.config.latency_offset.num_milliseconds(), 1000)
}

fn validate_duration(arg_key: &str, duration: Duration, default_value: i64) -> RuntimeResult<()> {
    let mut msecs = duration.num_milliseconds();
    if msecs == 0 {
        msecs = default_value
    }
    if msecs <= 0 || msecs > MAX_DURATION_MSECS {
        let msg = format!(
            "{}={}ms is out of allowed range [{} ... {}]",
            arg_key, msecs, 0, MAX_DURATION_MSECS
        );
        return Err(RuntimeError::from(msg));
    }
    Ok(())
}

/// get_deadline_for_query returns deadline for the given query r.
fn get_deadline_for_query<T, D>(
    ctx: &Context,
    start_time: T,
    duration: Option<D>,
) -> RuntimeResult<Deadline>
where
    T: Into<Timestamp>,
    D: Into<Duration>,
{
    let d_max = ctx.config.max_query_duration.num_milliseconds();
    get_deadline_with_max_duration(duration, start_time, d_max)
}

fn get_deadline_with_max_duration<D, T>(
    candidate: Option<D>,
    start_time: T,
    d_max: i64,
) -> RuntimeResult<Deadline>
where
    T: Into<Timestamp>,
    D: Into<Duration>,
{
    let mut d = if let Some(val) = candidate {
        val.into().num_milliseconds()
    } else {
        0
    };
    if d == 0 || d > d_max {
        d = d_max
    }
    Deadline::with_start_time(start_time, Duration::milliseconds(d))
}
