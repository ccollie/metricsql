#[cfg(test)]
mod tests {
    use crate::{
        marshal_timeseries_fast, test_metric_names_equal, test_rows_equal, test_timeseries_equal,
        Timeseries,
    };
    use crate::{unmarshal_fast_no_timestamps, unmarshal_timeseries_fast};
    use std::sync::Arc;

    #[test]
    fn test_timeseries_marshal_unmarshal_fast_single() {
        let mut buf: Vec<u8> = Vec::with_capacity(1024);

        let mut ts_orig: Timeseries = Timeseries::default();

        ts_orig.marshal_fast_no_timestamps(&mut buf);
        let n = ts_orig.marshaled_fast_size_no_timestamps();
        assert_eq!(
            n,
            buf.len(),
            "unexpected marshaled size; got {}; want {}",
            n,
            buf.len()
        );

        let mut ts_got: Timeseries = Timeseries::default();
        let tail = match unmarshal_fast_no_timestamps(&mut ts_got, &buf) {
            Err(err) => panic!("cannot unmarshal timeseries: {:?}", err),
            Ok(v) => v,
        };

        if tail.len() > 0 {
            panic!(
                "unexpected non-empty tail left: len(tail)={}; tail={:?}",
                tail.len(),
                tail
            );
        }
        ts_orig.metric_name.metric_group = "".to_string();
        test_timeseries_equal(&[ts_orig], &[ts_got]);
    }

    #[test]
    fn test_timeseries_marshal_unmarshal_fast_multiple() {
        let mut dst: Vec<u8> = vec![];

        let mut tss_orig: Vec<Timeseries> = vec![];
        let timestamps: Arc<Vec<i64>> = Arc::new(vec![2]);

        for i in 0..10 {
            let mut ts = Timeseries::default();
            ts.metric_name.metric_group = format!("metric_group {}", i);
            ts.metric_name.set_tag(
                format!("key {}", i).as_str(),
                format!("value {}", i).as_str(),
            );

            ts.values = vec![i as f64 + 0.2];
            ts.timestamps = Arc::clone(&timestamps);

            let dst_len = dst.len();
            ts.marshal_fast_no_timestamps(&mut dst);
            let n = ts.marshaled_fast_size_no_timestamps();
            if n != dst.len() - dst_len {
                panic!(
                    "unexpected marshaled size on iteration {}; got {}; want {}",
                    i,
                    n,
                    dst.len() - dst_len
                );
            }

            let mut ts_got = Timeseries::default();
            let tail: &[u8];

            ts_got.timestamps = Arc::clone(&ts.timestamps);
            match unmarshal_fast_no_timestamps(&mut ts_got, &dst[dst_len..]) {
                Ok(v) => {
                    tail = v;
                }
                Err(err) => {
                    panic!("cannot unmarshal timeseries on iteration {}: {:?}", i, err);
                }
            }

            assert_eq!(
                tail.len(),
                0,
                "unexpected non-empty tail left on iteration {}: len(tail)={}; tail={:?}",
                i,
                tail.len(),
                tail
            );

            compare_series(&ts_got, &ts);

            tss_orig.push(ts);
        }

        let mut buf: Vec<u8> = vec![];

        marshal_timeseries_fast(&mut buf, &tss_orig, 1_000_000, 123)
            .expect("marshal_timeseries_fast failed");

        match unmarshal_timeseries_fast(&buf) {
            Err(err) => {
                panic!("error in unmarshal_timeseries_fast: {:?}", err)
            }
            Ok(tss_got) => {
                test_timeseries_equal(&tss_got, &tss_orig);
            }
        }

        let mut src = &dst[0..];
        let mut i = 0;
        for ts_orig in tss_orig.iter_mut() {
            let mut ts: Timeseries = Timeseries::default();

            ts.timestamps = Arc::clone(&ts_orig.timestamps);
            src = unmarshal_fast_no_timestamps(&mut ts, src)
                .unwrap_or_else(|_| panic!("cannot unmarshal timeseries[{}]", i));
            compare_series(&ts, &ts_orig);
            i += 1;
        }

        assert_eq!(
            src.len(),
            0,
            "unexpected tail left; len(tail)={}; tail={:?}",
            src.len(),
            src
        );
    }

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
