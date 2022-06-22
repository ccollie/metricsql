use phf::phf_map;
use phf::phf_ordered_set;

static AGGR_FUNCTIONS: phf::OrderedSet<&'static str> = phf_ordered_set! {
  "any",
  "avg",
  "bottomk",
  "bottomk_avg",
  "bottomk_max",
  "bottomk_median",
  "bottomk_last",
  "bottomk_min",
  "count",
  "count_values",
  "distinct",
  "geomean",
  "group",
  "histogram",
  "limitk",
  "mad",
  "max",
  "median",
  "min",
  "mode",
  "outliers_mad",
  "outliersk",
  "quantile",
  "quantiles",
  "stddev",
  "stdvar",
  "sum",
  "sum2",
  "topk",
  "topk_avg",
  "topk_max",
  "topk_median",
  "topk_last",
  "topk_min",
  "zscore",
};

pub fn is_aggr_func(func: &str) -> bool {
  let lower = func.to_lowercase().as_str();;
  AGGR_FUNCTIONS.contains(lower)
}

pub fn is_aggr_func_modifier(s: &str) -> bool {
    let lower = s.to_lowercase().as_str();
    return lower == "by" || lower == "without";
}