mod any;
mod avg;
mod context;
mod count;
mod geomean;
mod group;
mod handler;
mod max;
mod min;
mod sum;
mod sum2;

pub use context::*;
pub use handler::*;

pub fn try_get_incremental_aggr_handler(name: &str) -> Option<Handler> {
    if let Ok(func) = IncrementalAggrFuncKind::try_from(name) {
        return Some(Handler::new(func));
    }
    None
}
