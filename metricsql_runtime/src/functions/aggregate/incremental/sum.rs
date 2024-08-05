use itertools::izip;

use super::context::{IncrementalAggrContext, IncrementalAggrHandler};

pub struct IncrementalAggrSum {}

impl IncrementalAggrHandler for IncrementalAggrSum {
    fn update(&self, iac: &mut IncrementalAggrContext, values: &[f64]) {
        for ((v, count), dst) in values
            .iter()
            .zip(iac.values.iter_mut())
            .zip(iac.ts.values.iter_mut())
        {
            if v.is_nan() {
                continue;
            }
            if *count == 0.0 {
                *dst = *v;
                *count = 1.0;
                continue;
            }
            *dst += *v;
        }
    }

    fn merge(&self, dst: &mut IncrementalAggrContext, src: &IncrementalAggrContext) {
        let iter = izip!(
            src.values.iter(),
            dst.values.iter_mut(),
            src.ts.values.iter(),
            dst.ts.values.iter_mut()
        );
        for (src_count, dst_count, v, dst) in iter {
            if *src_count == 0.0 {
                continue;
            }
            if *dst_count == 0.0 {
                *dst_count = 1.0;
                *dst = *v;
                continue;
            }
            *dst += *v;
        }
    }

    fn keep_original(&self) -> bool {
        false
    }
}
