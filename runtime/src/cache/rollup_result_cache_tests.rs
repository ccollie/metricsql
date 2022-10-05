#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use metricsql::ast::{AggrFuncExpr, Expression, ExpressionNode, FuncExpr, LabelFilter, LabelFilterOp, MetricExpr};
    use crate::cache::rollup_result_cache::{merge_timeseries, RollupResultCache};
    use crate::{EvalConfig, MetricName, Timeseries};

    struct TestContext {
        fe: Expression,
        ae: Expression,
        ec: EvalConfig,
        cache: RollupResultCache,
        window: i64
    }

    fn setup() -> TestContext {
        let window = 456_i64;
        let mut ec = EvalConfig::default();
        ec.start = 1000;
        ec.end = 2000;
        ec.step = 200;
        ec.max_points_per_series = 1e4 as usize;
        ec.may_cache = true;

        let mut me = MetricExpr::default();
        me.label_filters = vec![LabelFilter::new(LabelFilterOp::Equal, "aaa", "xxx").unwrap()];

        let fe = FuncExpr::create("foo", &[me.cast()], TextRange::default())?;

        let ae = AggrfnExpr::new(AggregationFunction::Sum);
        ae.args.push(fe);

        let cache = RollupResultCache::default();

        TestContext {
            ec,
            ae,
            fe: fe.cast(),
            cache,
            window
        }
    }

    fn create_ts(timestamps: &[i64], values: &[f64]) -> Timeseries {
        Timeseries{
            metric_name: MetricName::default(),
            timestamps: Arc::new( Vec::from(timestamps) ),
            values: Vec::from(values)
        }
    }

    // Try obtaining an empty value.
    #[test]
    fn test_empty() {
        let TestContext { mut cache, ec, fe, window, .. } = setup();
        let (tss, new_start) = cache.get(&ec, &fe, window);
        assert_eq!(new_start, ec.start, "unexpected new_start; got {}; want {}", new_start, ec.start);

        assert_eq!(tss.len(), 0, "got {} timeseries, while expecting zero", tss.len());
    }

    // Store timeseries overlapping with start
    #[test]
    fn test_start_overlap_no_ae() {
        let TestContext { mut cache, ec, fe, window, .. } = setup();
        let tss = vec![create_ts(&[800, 1000, 1200], &[0_f64, 1_f64, 2_f64])];

        cache.put(&ec, &fe, window, &tss)?;
        let (tss, new_start) = cache.get(&ec, &fe, window)?;
        assert_eq!(new_start, 1400, "unexpected new_start; got {}; want {}", new_start, 1400);

        let tss_expected = vec![
            create_ts(&[1000, 1200], &[1_f64, 2_f64])
        ];

        let tss = tss.unwrap();
        test_timeseries_equal(&tss, &tss_expected)
    }

    #[test]
    fn test_start_overlap_with_ae() {
        let tss = vec![
            create_ts(&[800, 1000, 1200], &[0_f64, 1_f64, 2_f64])
        ];

        let TestContext { mut cache, ec, ae, window, .. } = setup();
        let (tss, new_start) = cache.get(&ec, &ae, window)?;
        assert_eq!(new_start, 1400, "unexpected new_start; got {}; want {}", new_start, 1400);

        let tss_expected = vec![
            create_ts(&[1000, 1200], &[1_f64, 2_f64])
        ];

        let tss = tss.unwrap();
        test_timeseries_equal(&tss, &tss_expected)
    }

    // Store timeseries overlapping with end
    #[test]
    fn end_overlap() {
        let tss = vec![
            create_ts(&[1800, 2000, 2200, 2400], &[333_f64, 0_f64, 1_f64, 2_f64])
        ];

        let TestContext { mut cache, ec, fe, window, .. } = setup();
        let (tss, new_start) = cache.get(&ec, &fe, window);

        assert_eq!(new_start, 1000, "unexpected new_start; got {}; want {}", new_start, 1000);

        assert_eq!(tss.len(), 0, "got {} timeseries, while expecting zero", tss.len());
    }

    // Store timeseries covered by [start ... end]
    #[test]
    fn full_cover() {
        let tss = vec![
            create_ts(&[1200, 1400, 1600], &[0_f64, 1_f64, 2_f64])
        ];

        let TestContext { mut cache, ec, fe, window, .. } = setup();
        let (tss, new_start) = cache.get(&ec, &fe, window);

        assert_eq!(new_start, 1000, "unexpected new_start; got {}; want {}", new_start, 1000);

        assert_eq!(tss.len(), 0, "got {} timeseries, while expecting zero", tss.len());
    }

    // Store timeseries below start
    #[test]
    fn before_start() {
        let tss = vec![create_ts(&[200, 400, 600], &[0_f64, 1_f64, 2_f64]) ];

        let TestContext { mut cache, ec, fe, window, .. } = setup();
        let (tss, new_start) = cache.get(&ec, &fe, window);

        assert_eq!(new_start, 1000, "unexpected new_start; got {}; want {}", new_start, 1000);
        assert_eq!(tss.len(), 0, "got {} timeseries, while expecting zero", tss.len());
    }

    // Store timeseries after end
    #[test]
    fn after_end() {
        let tss = vec![
            create_ts(&[2200, 2400, 2600], &[0_f64, 1_f64, 2_f64])
        ];

        let TestContext { mut cache, ec, fe, window, .. } = setup();
        let (tss, new_start) = cache.get(&ec, &fe, window);

        assert_eq!(new_start, 1000, "unexpected new_start; got {}; want {}", new_start, 1000);
        assert_eq!(tss.len(), 0, "got {} timeseries, while expecting zero", tss.len());
    }

    // Store timeseries bigger than the interval [start ... end]
    #[test]
    fn bigger_than_start_end() {
        let tss = vec![
            create_ts(&[800, 1000, 1200, 1400, 1600, 1800, 2000, 2200],
                      &[0_f64, 1_f64, 2_f64, 3_f64, 4_f64, 5_f64, 6_f64, 7_f64])
        ];

        let TestContext { mut cache, ec, fe, window, .. } = setup();
        let (tss, new_start) = cache.get(&ec, &fe, window);

        assert_eq!(new_start, 2200, "unexpected new_start; got {}; want {}", new_start, 2200);

        let tss_expected = vec![
            create_ts(&[1000, 1200, 1400, 1600, 1800, 2000],
                      &[1_f64, 2_f64, 3_f64, 4_f64, 5_f64, 6_f64])
        ];

        testTimeseriesEqual(tss, tss_expected)
    }

    // Store timeseries matching the interval [start ... end]
    #[test]
    fn test_start_end_match() {
        let mut tss = vec![
            create_ts(&[1000, 1200, 1400, 1600, 1800, 2000],
                      &[1_f64, 2_f64, 3_f64, 4_f64, 5_f64, 6_f64])
        ];

        let TestContext { mut cache, ec, fe, window, .. } = setup();
        let (tss, new_start) = cache.get(&ec, &fe, window);

        assert_eq!(new_start, 2200, "unexpected new_start; got {}; want {}", new_start, 2200);

        let tss_expected = vec![
            create_ts(&[1000, 1200, 1400, 1600, 1800, 2000], &[1_f64, 2_f64, 3_f64, 4_f64, 5_f64, 6_f64])
        ];

        testTimeseriesEqual(tss, tss_expected)
    }

    // Store big timeseries, so their marshaled size exceeds 64Kb.
    #[test]
    fn big_timeseries() {
        let mut tss: Vec<Timeseries> = vec![];
        (0..1000).for_each(|| {
            let ts = create_ts(&[1000, 1200, 1400, 1600, 1800, 2000],
                               &[1_f64, 2_f64, 3_f64, 4_f64, 5_f64, 6_f64]);
            tss.push(ts);
        });

        let TestContext { mut cache, ec, fe, window, .. } = setup();
        let (tss, new_start) = cache.get(&ec, &fe, window);

        assert_eq!(new_start, 2200, "unexpected new_start; got {}; want {}", new_start, 2200);

        testTimeseriesEqual(tss_result, tss)
    }

    // Store multiple time series
    #[test]
    fn multi_timeseries() {
        let tss1 = vec![
            create_ts(&[800, 1000, 1200], &[0_f64, 1_f64, 2_f64])
        ];

        let tss2 = vec![
            create_ts(&[1800, 2000, 2200, 2400], &[333_f64, 0_f64, 1_f64, 2_f64])
        ];

        let tss3 = vec![
            create_ts(&[1200, 1400, 1600], &[0_f64, 1_f64, 2_f64])
        ];

        let TestContext { mut cache, ec, fe, window, .. } = setup();

        cache.put(&ec, &fe, window, &tss1)?;
        cache.put(&ec, &fe, window, &tss2)?;
        cache.put(&ec, &fe, window, &tss3)?;

        let (tss, new_start) = cache.get(&ec, &fe, window)?;
        assert_eq!(new_start, 1400, "unexpected new_start; got {}; want {}", new_start, 1400);

        let tss_expected = vec![
            create_ts(&[1000, 1200], &[1_f64, 2_f64])
        ];

        let tss = tss.unwrap();

        test_timeseries_equal(&tss, &tss_expected)
    }

    struct MergeTestContext {
        ec: EvalConfig,
        bstart: i64
    }

    fn setup_merge() -> MergeTestContext {
        let mut ec = EvalConfig::default();
        ec.start = 1000;
        ec.end = 2000;
        ec.step = 200;
        ec.max_points_per_series = 1e4 as usize;

        MergeTestContext {
            ec,
            bstart: 1400_i64
        }
    }


    #[test]
    fn merge_bstart_eq_ec_start() {
        let a: Vec<Timeseries> = vec![];
        let b = vec![
            create_ts(&[1000, 1200, 1400, 1600, 1800, 2000],&[1_f64, 2_f64, 3_f64, 4_f64, 5_f64, 6_f64]),
        ];

        let MergeTestContext { ec, .. } = setup_merge();
        let tss = merge_timeseries(a, b, 1000, &ec)?;
        let tss_expected = vec![
            create_ts(&[1000, 1200, 1400, 1600, 1800, 2000], &[1_f64, 2_f64, 3_f64, 4_f64, 5_f64, 6_f64])
        ];

        test_timeseries_equal(&tss, &tss_expected)
    }

    #[test]
    fn merge_a_empty() {
        let a: Vec<Timeseries> = vec![];
        let b = vec![
            create_ts(&[1400, 1600, 1800, 2000], &[3_f64, 4_f64, 5_f64, 6_f64])
        ];

        let MergeTestContext { ec, bstart } = setup_merge();
        let tss = merge_timeseries(a, b, bstart, &ec)?;
        let tss_expected = vec![
            create_ts(&[1000, 1200, 1400, 1600, 1800, 2000], &[nan, nan, 3_f64, 4_f64, 5_f64, 6_f64])
        ];

        test_timeseries_equal(&tss, &tss_expected)
    }

    #[test]
    fn merge_b_empty() {
        let a = vec![
            create_ts(&[1000, 1200], &[2_f64, 1_f64])
        ];

        let b = vec![ Timeseries::default() ];

        let MergeTestContext { ec, bstart } = setup_merge();
        let tss = merge_timeseries(a, b, bstart, &ec)?;

        let tss_expected = vec![
            create_ts(&[1000, 1200, 1400, 1600, 1800, 2000], &[2_f64, 1_f64, nan, nan, nan, nan])
        ];

        test_timeseries_equal(&tss, &tss_expected);
    }

    #[test]
    fn merge_non_empty() {
        let a = vec![
            create_ts(&[1000, 1200], &[2_f64, 1_f64])
        ];

        let b = vec![
            create_ts(&[1400, 1600, 1800, 2000], &[3_f64, 4_f64, 5_f64, 6_f64])
        ];

        let MergeTestContext { ec, bstart } = setup_merge();
        let tss = merge_timeseries(a, b, bstart, &ec)?;
        let tss_expected = vec![
            create_ts(&[1000, 1200, 1400, 1600, 1800, 2000], &[2_f64, 1_f64, 3_f64, 4_f64, 5_f64, 6_f64])
        ];

        test_timeseries_equal(&tss, &tss_expected)
    }

    #[test]
    fn merge_non_empty_distinct_metric_names() {
        let mut a = vec![
            create_ts(&[1000, 1200], &[2_f64, 1_f64])
        ];

        a.get_mut(0).metric_name.metric_group = "bar".to_string();

        let mut b = vec![
            create_ts(&[1400, 1600, 1800, 2000], &[3_f64, 4_f64, 5_f64, 6_f64])
        ];

        b.get_mut(0).metric_name.metric_group = "foo".to_string();

        let MergeTestContext { ec, bstart } = setup_merge();
        let tss = merge_timeseries(a, b, bstart, &ec)?;

        let mut foo = Timeseries::default();
        foo.metric_name.metric_group = "foo".to_string();

        let mut bar = Timeseries::default();
        bar.metric_name.metric_group = "bar".to_string();

        let tss_expected = vec![
            foo,
            create_ts(&[1000, 1200, 1400, 1600, 1800, 2000], &[nan, nan, 3_f64, 4_f64, 5_f64, 6_f64]),
            bar,
            create_ts(&[1000, 1200, 1400, 1600, 1800, 2000], &[2_f64, 1_f64, nan, nan, nan, nan])
        ];

        test_timeseries_equal(&tss, &tss_expected)
    }

    fn test_timeseries_equal(tss: &[Timeseries], tss_expected: &[Timeseries]) {
        assert_eq!(tss.len(), tss_expected.len(), "unexpected timeseries count; got {}; want {}",
                   tss.len(), tss_expected.len());

        for (i, ts) in tss.iter().enumerate() {
            let ts_expected = &tss_expected[i];
            testMetricNamesEqual(&ts.metric_name, &ts_expected.metric_name, i);
            testRowsEqual(ts.values, ts.timestamps, ts_expected.values, ts_expected.timestamps)
        }
    }
}