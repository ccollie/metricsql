mod context;
mod avg;
mod sum;
mod max;
mod min;
mod count;
mod any;
mod group;
mod sum2;
mod geomean;
mod handler;

pub use context::*;
pub use handler::*;

pub fn try_get_incremental_aggr_handler(
    name: &str,
) -> Option<Handler> {
    if let Ok(func) = IncrementalAggrFuncKind::try_from(name) {
        return Some(Handler::new(func));
    }
    None
}