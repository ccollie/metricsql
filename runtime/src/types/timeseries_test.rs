#[cfg(test)]
mod tests {
    use crate::{
        marshal_timeseries_fast, test_metric_names_equal, test_rows_equal, test_timeseries_equal,
        Timeseries,
    };
    use std::sync::Arc;

    fn compare_series(ts: &Timeseries, ts_expected: &Timeseries) {
        test_metric_names_equal(&ts.metric_name, &ts_expected.metric_name, 0);
        test_rows_equal(
            &ts.values,
            &ts.timestamps,
            &ts_expected.values,
            &ts_expected.timestamps,
        )
    }
}
