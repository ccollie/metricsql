use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use phf::phf_map;
use metricsql::ast::{AggrFuncExpr};

use crate::runtime_error::RuntimeResult;
use crate::timeseries::Timeseries;

// todo: make it a trait ??
pub(crate) struct IncrementalAggrFuncCallbacks {
    update: fn(iac: &mut IncrementalAggrContext, values: &[f64]),
    merge: fn(dst: &mut IncrementalAggrContext, src: &IncrementalAggrContext),
    finalize: fn(iac: &mut IncrementalAggrContext),
    // Whether to keep the original MetricName for every time series during aggregation
    keep_original: bool,
}

/// callbacks for optimized incremental calculations for aggregate functions
/// over rollup over metricsql.MetricExpr.
///
/// These calculations save RAM for aggregates over big number of time series.
static INCREMENTAL_AGGR_FUNC_CALLBACKS_MAP: phf::Map<&'static str, IncrementalAggrFuncCallbacks> = phf_map! {
"sum" => IncrementalAggrFuncCallbacks {
update:   update_aggr_sum,
merge:    merge_aggr_sum,
finalize: finalize_aggr_common,
        keep_original: false
},
"min" => IncrementalAggrFuncCallbacks{
update:   update_aggr_min,
merge:    merge_aggr_min,
finalize: finalize_aggr_common,
        keep_original: false
},
"max" => IncrementalAggrFuncCallbacks {
update:   update_aggr_max,
merge:    merge_aggr_max,
finalize: finalize_aggr_common,
        keep_original: false
},
"avg" => IncrementalAggrFuncCallbacks{
update:   update_aggr_avg,
merge:    merge_aggr_avg,
finalize: finalize_aggr_avg,
        keep_original: false
},
"count" => IncrementalAggrFuncCallbacks{
update:   update_aggr_count,
merge:    merge_aggr_count,
finalize: finalize_aggr_count,
        keep_original: false
},
"sum2" => IncrementalAggrFuncCallbacks{
update:   update_aggr_sum2,
merge:    merge_aggr_sum2,
finalize: finalize_aggr_common,
        keep_original: false
},
"geomean" => IncrementalAggrFuncCallbacks{
update:   update_aggr_geomean,
merge:    merge_aggr_geomean,
finalize: finalize_aggr_geomean,
        keep_original: false
},
"any" => IncrementalAggrFuncCallbacks{
    update:   update_aggr_any,
    merge:    merge_aggr_any,
    finalize: finalize_aggr_common,
    keep_original: true,
},
"group" => IncrementalAggrFuncCallbacks{
    update:   update_aggr_count,
    merge:    merge_aggr_count,
    finalize: finalize_aggr_group,
        keep_original: false
},
};

#[derive(Default)]
pub(crate) struct IncrementalAggrContext {
    ts: Timeseries,
    values: Vec<f64>,
}

type ContextHash = HashMap<u64, HashMap<String, IncrementalAggrContext>>;

pub(crate) struct IncrementalAggrFuncContext<'a> {
    ae: &'a AggrFuncExpr,
    // todo: use Rc/Arc based on cfg
    context_map: RwLock<ContextHash>,
    callbacks: &'static IncrementalAggrFuncCallbacks,
}

impl<'a> IncrementalAggrFuncContext<'a> {
    pub(crate) fn new(ae: &'a AggrFuncExpr, callbacks: &'static IncrementalAggrFuncCallbacks) -> Self {
        let m: HashMap<u64, HashMap<String, IncrementalAggrContext>> = HashMap::new();

        Self {
            ae,
            context_map: RwLock::new(m),
            callbacks,
        }
    }

    pub fn update_timeseries(&mut self, ts_orig: &mut Timeseries, worker_id: u64) -> RuntimeResult<()> {
        let mut im = self.context_map.write().unwrap();
        let m = im.entry(worker_id).or_default();

        if self.ae.limit > 0 && m.len() >= self.ae.limit {
            // Skip this time series, since the limit on the number of output time series has been already reached.
            return Ok(());
        }

        let keep_original = self.callbacks.keep_original;

        // avoid temporary value dropped while borrowed
        let mut ts: &mut Timeseries = ts_orig;
        if !keep_original {
            ts.metric_name.remove_group_tags(&self.ae.modifier);
        }

        let key = ts.metric_name.to_string();

        let value_len = ts.values.len();

        match m.entry(key) {
            Vacant(entry) => {
                if keep_original {
                    ts = ts_orig
                }
                let ts_aggr = Timeseries {
                    metric_name: ts.metric_name.clone(),
                    values: vec![0_f64; value_len],
                    timestamps: Arc::clone(&ts.timestamps)
                };

                let mut iac = IncrementalAggrContext {
                    ts: ts_aggr,
                    values: vec![0_f64; value_len],
                };

                (self.callbacks.update)(&mut iac, &ts.values);
                entry.insert(iac);
            }
            Occupied(mut entry) => {
                let mut iac = entry.get_mut();
                (self.callbacks.update)(&mut iac, &ts.values);
            }
        };

        Ok(())
    }

    pub fn finalize_timeseries(&mut self) -> Vec<Timeseries> {
        // There is no need in iafc.mLock.lock here, since finalize_timeseries must be called
        // without concurrent threads touching iafc.
        let mut m_global: HashMap<&String, IncrementalAggrContext> = HashMap::new();
        let merge = self.callbacks.merge;
        let mut hash = self.context_map.write().unwrap();
        for (_, m) in hash.iter_mut() {
            for (k, iac) in m.into_iter() {
                match m_global.get_mut(k) {
                    Some(iac_global) => {
                        merge(iac_global, iac);
                    },
                    None => {
                        if self.ae.limit > 0 && m_global.len() >= self.ae.limit {
                            // Skip this time series, since the limit on the number of output time series
                            // has been already reached.
                            continue;
                        }
                        m_global.insert(k, std::mem::take(iac));
                    }
                }
            }
        }
        let mut tss: Vec<Timeseries> = Vec::with_capacity(m_global.len());
        let finalize_aggrfn = self.callbacks.finalize;
        for mut iac in m_global.into_values() {
            finalize_aggrfn(&mut iac);
            tss.push(std::mem::take(&mut iac.ts));
        }
        return tss;
    }
}

pub(crate) fn get_incremental_aggr_func_callbacks(
    name: &str,
) -> Option<&'static IncrementalAggrFuncCallbacks> {
    return INCREMENTAL_AGGR_FUNC_CALLBACKS_MAP.get(&name.to_lowercase());
}

fn finalize_aggr_common(iac: &mut IncrementalAggrContext) {
    let counts = &iac.values;
    for (i, v) in counts.iter().enumerate() {
        if *v == 0.0 {
            iac.ts.values[i] = f64::NAN
        }
    }
}

fn update_aggr_sum(iac: &mut IncrementalAggrContext, values: &[f64]) {
    for (i, v) in values.iter().enumerate() {
        if v.is_nan() {
            continue;
        }

        if iac.values[i] == 0.0 {
            iac.ts.values[i] = *v;
            iac.values[i] = 1.0;
            continue;
        }

        iac.ts.values[i] += v;
    }
}

fn merge_aggr_sum(dst: &mut IncrementalAggrContext, src: &IncrementalAggrContext) {
    let src_values = &src.ts.values;
    let src_counts = &src.values;

    for (i, v) in src_values.iter().enumerate() {
        if src_counts[i] == 0.0 {
            continue;
        }

        if dst.values[i] == 0.0 {
            dst.ts.values[i] = *v;
            dst.values[i] = 1.0;
            continue;
        }

        dst.ts.values[i] += v;
    }
}

fn update_aggr_min(iac: &mut IncrementalAggrContext, values: &[f64]) {
    for (i, v) in values.iter().enumerate() {
        if v.is_nan() {
            continue;
        }

        if iac.values[i] == 0.0 {
            iac.ts.values[i] = *v;
            iac.values[i] = 1.0;
            continue;
        }

        if v < &iac.ts.values[i] {
            iac.ts.values[i] = *v;
        }
    }
}

fn merge_aggr_min(dst: &mut IncrementalAggrContext, src: &IncrementalAggrContext) {
    let src_values = &src.ts.values;
    let src_counts = &src.values;

    for (i, v) in src_values.iter().enumerate() {
        if src_counts[i] == 0.0 {
            continue;
        }

        let dst_count = dst.values[i];
        let dst_value = dst.ts.values[i];
        if dst_count == 0.0 {
            dst.ts.values[i] = *v;
            dst.values[i] = 1.0;
            continue;
        }
        if *v < dst_value {
            dst.values[i] = *v
        }
    }
}

fn update_aggr_max(iac: &mut IncrementalAggrContext, values: &[f64]) {
    for (i, v) in values.iter().enumerate() {
        if v.is_nan() {
            continue;
        }

        let dst_value = iac.ts.values[i];

        if iac.values[i] == 0.0 {
            iac.ts.values[i] = *v;
            iac.values[i] = 1.0;
            continue;
        }
        if v > &dst_value {
            iac.ts.values[i] = *v;
        }
    }
}

fn merge_aggr_max(dst: &mut IncrementalAggrContext, src: &IncrementalAggrContext) {
    for i in 0 .. src.ts.values.len() {
        let v = src.ts.values[i];
        if src.values[i] == 0.0 {
            continue;
        }

        if dst.values[i] == 0.0 {
            dst.ts.values[i] = v;
            dst.values[i] = 1.0;
            continue;
        }

        if v > dst.ts.values[i] {
            dst.ts.values[i] = v;
        }
    }
}

fn update_aggr_avg(iac: &mut IncrementalAggrContext, values: &[f64]) {
    // do not use `Rapid calculation methods` at https://en.wikipedia.org/wiki/Standard_deviation,
    // since it is slower and has no obvious benefits in increased precision.
    for (i, v) in values.iter().enumerate() {
        if v.is_nan() {
            continue;
        }

        if iac.values[i] == 0.0 {
            iac.ts.values[i] = *v;
            iac.values[i] = 1.0;
            continue;
        }

        iac.ts.values[i] += v;
        iac.values[i] += 1.0;
    }
}

// TODO: check original !!!
fn merge_aggr_avg(dst: &mut IncrementalAggrContext, src: &IncrementalAggrContext) {
    let src_values = &src.ts.values;
    let src_counts = &src.values;

    for (i, v) in src_values.iter().enumerate() {
        if src_counts[i] == 0.0 {
            continue;
        }

        if dst.ts.values[i] == 0.0 {
            dst.ts.values[i] = *v;
            dst.values[i] = src_counts[i];
            continue;
        }

        dst.ts.values[i] += v;
        dst.values[i] += src_counts[i];
    }
}

fn finalize_aggr_avg(iac: &mut IncrementalAggrContext) {
    let counts = &iac.values;

    for (i, v) in counts.iter().enumerate() {
        let dst_value = iac.ts.values[i];
        if *v == 0.0 {
            iac.ts.values[i] = f64::NAN;
            continue;
        }
        iac.ts.values[i]  = dst_value / v;
    }
}

fn update_aggr_count(iac: &mut IncrementalAggrContext, values: &[f64]) {

    for (i, v) in values.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        iac.ts.values[i] += 1.0;
    }
}

fn merge_aggr_count(dst: &mut IncrementalAggrContext, src: &IncrementalAggrContext) {
    for (i, v) in src.ts.values.iter().enumerate() {
        dst.ts.values[i] += v;
    }
}

fn finalize_aggr_count(iac: &mut IncrementalAggrContext) {
    for v in iac.ts.values.iter_mut() {
        if *v == 0.0 {
            *v = f64::NAN
        }
    }
}

fn finalize_aggr_group(iac: &mut IncrementalAggrContext) {
    for v in iac.ts.values.iter_mut() {
        if *v == 0.0 {
            *v = f64::NAN;
        } else {
            *v = 1.0;
        }
    }
}

fn update_aggr_sum2(iac: &mut IncrementalAggrContext, values: &[f64]) {
    for (i, v) in values.iter().enumerate() {
        if v.is_nan() {
            continue;
        }

        let v_squared = *v * *v;
        if iac.values[i] == 0.0 {
            iac.ts.values[i] = v_squared;
            iac.values[i] = 1.0;
            continue;
        }

        iac.ts.values[i] += v_squared;
    }
}

fn merge_aggr_sum2(dst: &mut IncrementalAggrContext, src: &IncrementalAggrContext) {
    let src_values = &src.ts.values;
    let src_counts = &src.values;

    for (i, v) in src_values.iter().enumerate() {
        if src_counts[i] == 0.0 {
            continue;
        }

        if dst.values[i] == 0.0 {
            dst.ts.values[i] = *v;
            dst.values[i] = 1.0;
            continue;
        }

        dst.ts.values[i] += v;
    }
}

fn update_aggr_geomean(iac: &mut IncrementalAggrContext, values: &[f64]) {

    for (i, v) in values.iter().enumerate() {
        if v.is_nan() {
            continue;
        }

        if iac.values[i] == 0.0 {
            iac.ts.values[i] = *v;
            iac.values[i] = 1.0;
            continue;
        }

        iac.ts.values[i] *= v;
        iac.values[i] += 1.0;
    }
}

fn merge_aggr_geomean(dst: &mut IncrementalAggrContext, src: &IncrementalAggrContext) {
    let src_values = &src.ts.values;
    let src_counts = &src.values;

    for (i, v) in src_values.iter().enumerate() {
        if src_counts[i] == 0.0 {
            continue;
        }

        if dst.values[i] == 0.0 {
            dst.ts.values[i] = *v;
            dst.values[i] = src_counts[i];
            continue;
        }

        dst.ts.values[i] *= v;
        dst.values[i] += src_counts[i]
    }
}

fn finalize_aggr_geomean(iac: &mut IncrementalAggrContext) {
    let counts = &iac.values;

    for (i, v) in counts.iter().enumerate() {
        if *v == 0.0 {
            iac.ts.values[i] = f64::NAN;
            continue;
        }
        iac.ts.values[i] = iac.ts.values[i].powf(1.0 / v)
    }
}

fn update_aggr_any(iac: &mut IncrementalAggrContext, values: &[f64]) {
    if iac.values[0] > 0 as f64 {
        return;
    }
    for i in 0..values.len() {
        iac.values[i] = 1.0;
    }
    // ??
    iac.ts.values = Vec::from(values);
}

fn merge_aggr_any(dst: &mut IncrementalAggrContext, src: &IncrementalAggrContext) {
    let src_values = &src.ts.values;
    let src_counts = &src.values;
    if dst.values[0] > 0.0 {
        return;
    }
    dst.values[0] = src_counts[0];
    dst.ts.values = src_values.clone();
}
