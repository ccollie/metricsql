use super::context::{IncrementalAggrContext, IncrementalAggrHandler};

pub struct IncrementalAggrSum2{}

impl IncrementalAggrHandler for IncrementalAggrSum2 {
    fn update(&self, iac: &mut IncrementalAggrContext, values: &[f64]) {
        for ((v, count), dst) in values.iter()
            .zip(iac.values.iter_mut())
            .zip(iac.ts.values.iter_mut()) {
            if v.is_nan() {
                continue;
            }
            let v_squared = *v * *v;
            if *count == 0.0 {
                *count = 1.0;
                *dst = v_squared;
                continue;
            }

            *dst += v_squared;
        }
    }
    fn merge(&self, dst: &mut IncrementalAggrContext, src: &IncrementalAggrContext) {
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
    fn keep_original(&self) -> bool {
        false
    }
}