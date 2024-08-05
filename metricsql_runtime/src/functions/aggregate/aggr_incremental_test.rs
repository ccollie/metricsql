#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicU64, Ordering};

    use rayon::iter::IntoParallelRefMutIterator;

    use metricsql_parser::ast::AggregationExpr;

    use crate::functions::aggregate::IncrementalAggrFuncContext;
    use crate::rayon::iter::ParallelIterator;
    use crate::{compare_values, RuntimeError, RuntimeResult, Timeseries};

    const NAN: f64 = f64::NAN;

    const DEFAULT_TIMESTAMPS: [i64; 4] = [100000_i64, 200000_i64, 300000_i64, 400000_i64];
    const VALUES: [[f64; 4]; 7] = [
        [1.0, NAN, 2.0, NAN],
        [3.0, NAN, NAN, 4.0],
        [NAN, NAN, 5.0, 6.0],
        [7.0, NAN, 8.0, 9.0],
        [4.0, NAN, NAN, NAN],
        [2.0, NAN, 3.0, 2.0],
        [0.0, NAN, 1.0, 1.0],
    ];

    fn copy_timeseries(source: &[Timeseries]) -> Vec<Timeseries> {
        source
            .iter()
            .map(|x| x.clone())
            .collect::<Vec<Timeseries>>()
    }

    fn make_source_timeseries() -> Vec<Timeseries> {
        let mut tss_src: Vec<Timeseries> = Vec::with_capacity(VALUES.len());
        for vs in VALUES.iter() {
            let ts = Timeseries::new(DEFAULT_TIMESTAMPS.to_vec(), vs.to_vec());
            tss_src.push(ts);
        }
        tss_src
    }

    fn test_incremental(name: &str, values_expected: &[f64]) {
        let tss_src = make_source_timeseries();
        let ae = AggregationExpr::from_name(name)
            .expect(format!("{} is an invalid aggregate function", name).as_str());
        let tss_expected = [Timeseries::new(
            DEFAULT_TIMESTAMPS.to_vec(),
            Vec::from(values_expected),
        )];

        // run the test multiple times to make sure there are no side effects on concurrency
        (0..10).for_each(move |i| {
            let mut iafc = IncrementalAggrFuncContext::new(&ae).unwrap();
            let mut tss_src_copy = copy_timeseries(&tss_src);
            match test_incremental_parallel_aggr(&mut iafc, &mut tss_src_copy, &tss_expected) {
                Err(err) => panic!("unexpected error on iteration {}: {:?}", i, err),
                _ => {}
            }
        });
    }

    #[test]
    fn test_incremental_aggr_sum() {
        let values_expected = [17.0, NAN, 19.0, 22.0];
        test_incremental("sum", &values_expected);
    }

    #[test]
    fn test_incremental_aggr_min() {
        let values_expected = [0.0, NAN, 1.0, 1.0];
        test_incremental("min", &values_expected);
    }

    #[test]
    fn test_incremental_aggr_max() {
        let values_expected = [7.0, NAN, 8.0, 9.0];
        test_incremental("max", &values_expected)
    }

    #[test]
    fn test_incremental_aggr_avg() {
        let values_expected = [2.8333333333333335, NAN, 3.8, 4.4];
        test_incremental("avg", &values_expected)
    }

    #[test]
    fn test_incremental_aggr_count() {
        let values_expected = [6.0, NAN, 5.0, 5.0];
        test_incremental("count", &values_expected)
    }

    #[test]
    fn test_incremental_aggr_sum2() {
        let values_expected = [79.0, NAN, 103.0, 138.0];
        test_incremental("sum2", &values_expected)
    }

    #[test]
    fn test_incremental_aggr_geomean() {
        let values_expected = [0.0, NAN, 2.9925557394776896, 3.365865436338599];
        test_incremental("geomean", &values_expected)
    }

    fn test_incremental_parallel_aggr(
        iafc: &mut IncrementalAggrFuncContext,
        tss_src: &mut [Timeseries],
        tss_expected: &[Timeseries],
    ) -> RuntimeResult<()> {
        let worker_id: AtomicU64 = AtomicU64::new(1);
        tss_src
            .par_iter_mut()
            .for_each(|ts| {
                let id = worker_id.fetch_add(1, Ordering::SeqCst);
                iafc.update_single_timeseries(ts, id);
            });
        let tss_actual = iafc.finalize();

        match expect_timeseries_equal(&tss_actual, tss_expected) {
            Err(err) => {
                let msg = format!(
                    "{:?}; tssActual={:?}, tss_expected={:?}",
                    err, tss_actual, tss_expected
                );
                Err(RuntimeError::from(msg))
            }
            _ => Ok(()),
        }
    }

    fn expect_timeseries_equal(
        actual: &[Timeseries],
        expected: &[Timeseries],
    ) -> RuntimeResult<()> {
        if actual.len() != expected.len() {
            let msg = format!(
                "unexpected number of time series; got {}; want {}",
                actual.len(),
                expected.len()
            );
            return Err(RuntimeError::from(msg));
        }
        let m_actual = timeseries_to_map(actual);
        let m_expected = timeseries_to_map(expected);
        if m_actual.len() != m_expected.len() {
            let msg = format!(
                "unexpected number of time series after converting to map; got {}; want {}",
                m_actual.len(),
                m_expected.len()
            );
            return Err(RuntimeError::from(msg));
        }

        for (k, ts_expected) in m_expected.iter() {
            let ts_actual = m_actual.get(k);
            if ts_actual.is_none() {
                return Err(RuntimeError::from(format!(
                    "missing time series for key={}",
                    k
                )));
            }
            expect_ts_equal(&ts_actual.unwrap(), ts_expected)?;
        }
        Ok(())
    }

    fn timeseries_to_map<'a>(tss: &'a [Timeseries]) -> BTreeMap<String, &'a Timeseries> {
        let mut m: BTreeMap<String, &'a Timeseries> = BTreeMap::new();
        for ts in tss.iter() {
            let k = ts.metric_name.to_string();
            m.insert(k, ts);
        }
        return m;
    }

    fn expect_ts_equal(actual: &Timeseries, expected: &Timeseries) -> RuntimeResult<()> {
        let mn_actual = actual.metric_name.to_string();
        let mn_expected = expected.metric_name.to_string();
        if mn_actual != mn_expected {
            return Err(RuntimeError::from(format!(
                "unexpected metric name; got {}; want {}",
                mn_actual, mn_expected
            )));
        }
        if actual.timestamps != expected.timestamps {
            let msg = format!(
                "unexpected timestamps; got {:?}; want {:?}",
                &actual.timestamps, &expected.timestamps
            );
            return Err(RuntimeError::from(msg));
        }
        match compare_values(&actual.values, &expected.values) {
            Err(err) => {
                let msg = format!(
                    "{:?}; actual {:?}; expected {:?}",
                    err, &actual.values, &expected.values
                );
                return Err(RuntimeError::from(msg));
            }
            _ => {}
        }
        Ok(())
    }
}
