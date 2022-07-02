use phf::phf_ordered_set;
use crate::types::FuncExpr;

static ROLLUP_FUNCTIONS: phf::OrderedSet<&'static str> = phf_ordered_set! {
  "absent_over_time",
  "aggr_over_time",
  "ascent_over_time",
  "avg_over_time",
  "changes",
  "changes_prometheus",
  "count_eq_over_time",
  "count_gt_over_time",
  "count_le_over_time",
  "count_ne_over_time",
  "count_over_time",
  "decreases_over_time",
  "default_rollup",
  "delta",
  "delta_prometheus",
  "deriv",
  "deriv_fast",
  "descent_over_time",
  "distinct_over_time",
  "duration_over_time",
  "first_over_time",
  "geomean_over_time",
  "histogram_over_time",
  "hoeffding_bound_lower",
  "hoeffding_bound_upper",
  "holt_winters",
  "idelta",
  "ideriv",
  "increase",
  "increase_prometheus",
  "increase_pure",
  "increases_over_time",
  "integrate",
  "irate",
  "lag",
  "last_over_time",
  "lifetime",
  "max_over_time",
  "min_over_time",
  "mode_over_time",
  "predict_linear",
  "present_over_time",
  "quantile_over_time",
  "quantiles_over_time",
  "range_over_time",
  "rate",
  "rate_over_sum",
  "resets",
  "rollup",
  "rollup_candlestick",
  "rollup_delta",
  "rollup_deriv",
  "rollup_increase",
  "rollup_rate",
  "rollup_scrape_interval",
  "scrape_interval",
  "share_gt_over_time",
  "share_le_over_time",
  "stale_samples_over_time",
  "stddev_over_time",
  "stdvar_over_time",
  "sum_over_time",
  "sum2_over_time",
  "tfirst_over_time",
  // `timestamp` function must return timestamp for the last datapoint on the current window
  // in order to properly handle offset and timestamps unaligned to the current step.
  // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/415 for details.
  "timestamp",
  "timestamp_with_name",
  "tlast_change_over_time",
  "tlast_over_time",
  "tmax_over_time",
  "tmin_over_time",
  "zscore_over_time",
};

pub fn is_rollup_func(func: &str) -> bool {
  let lower = func.to_lowercase().as_str();
  ROLLUP_FUNCTIONS.contains(lower)
}

// GetRollupArgIdx returns the argument index for the given fe, which accepts the rollup argument.
//
// -1 is returned if fe isn't a rollup function.
pub fn get_rollup_arg_idx(fe: &FuncExpr) -> i32 {
  let lower = fe.name.to_lowercase().as_str();
  if !ROLLUP_FUNCTIONS.contains(lower) {
    return -1;
  }

  match lower {
    "quantile_over_time" | "aggr_over_time" | "hoeffding_bound_lower" | "hoeffding_bound_upper" => 1,
    "quantiles_over_time" => fe.args.len() - 1,
    _ => 0,
  }
}