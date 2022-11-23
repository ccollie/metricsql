use super::context::{IncrementalAggrContext, IncrementalAggrHandler};

pub struct IncrementalAggrAny {}

impl IncrementalAggrHandler for IncrementalAggrAny {
    fn update(&self, iac: &mut IncrementalAggrContext, values: &[f64]) {
        if iac.values[0] > 0.0 {
            return;
        }
        iac.values.fill(1.0);
        for (dst, src) in iac.ts.values.iter_mut().zip(values.iter()) {
            *dst = *src;
        }
    }
    fn merge(&self, dst: &mut IncrementalAggrContext, src: &IncrementalAggrContext) {
        if dst.values[0] > 0.0 {
            return;
        }
        dst.values[0] = src.values[0];
        for (dst, src) in dst.ts.values.iter_mut().zip(src.ts.values.iter()) {
            *dst = *src;
        }
    }
    fn keep_original(&self) -> bool {
        false
    }
}
