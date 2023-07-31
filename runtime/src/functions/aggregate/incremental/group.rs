use super::context::{IncrementalAggrContext, IncrementalAggrHandler};
use super::count::{merge_count, update_count};

pub struct IncrementalAggrGroup {}

impl IncrementalAggrHandler for IncrementalAggrGroup {
    fn update(&self, iac: &mut IncrementalAggrContext, values: &[f64]) {
        update_count(iac, values);
    }
    fn merge(&self, dst: &mut IncrementalAggrContext, src: &IncrementalAggrContext) {
        merge_count(dst, src);
    }
    fn finalize(&self, iac: &mut IncrementalAggrContext) {
        for v in iac.ts.values.iter_mut() {
            if *v == 0.0 {
                *v = f64::NAN
            } else {
                *v = 1.0
            }
        }
    }
    fn keep_original(&self) -> bool {
        true
    }
}
