#[cfg(test)]
mod tests {
    use std::cmp::Ordering::Greater;
    use crate::common::math::mode_no_nans;
    use crate::functions::aggregate::aggr_fns::{aggregate_bottomk, aggregate_topk};
    use crate::functions::aggregate::AggrFuncArg;
    use crate::{QueryValue, Timeseries};
    use crate::functions::utils::{float_cmp_with_nans, float_cmp_with_nans_desc};

    const NAN: f64 = f64::NAN;

    #[test]
    fn test_mode_no_nans() {
        let f = |prev_value: f64, a: &[f64], expected_result: f64| {
            let mut values = Vec::from(a);
            let result = mode_no_nans(prev_value, &mut values);
            if result.is_nan() {
                assert!(
                    expected_result.is_nan(),
                    "unexpected result; got {}; want {}",
                    result,
                    expected_result
                );
                return;
            }
            if result != expected_result {
                panic!(
                    "unexpected result; got {}; want {}",
                    result, expected_result
                )
            }
        };

        f(NAN, &[], NAN);
        f(NAN, &[123.0], 123.0);
        f(NAN, &[1.0, 2.0, 3.0], 1.0);
        f(NAN, &[1.0, 2.0, 2.0], 2.0);
        f(NAN, &[1.0, 1.0, 2.0], 1.0);
        f(NAN, &[1.0, 1.0, 1.0], 1.0);
        f(NAN, &[1.0, 2.0, 2.0, 3.0], 2.0);
        f(NAN, &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0], 3.0);
        f(1.0, &[2.0, 3.0, 4.0, 5.0], 1.0);
        f(1.0, &[2.0, 2.0], 2.0);
        f(1.0, &[2.0, 3.0, 3.0], 3.0);
        f(1.0, &[2.0, 4.0, 3.0, 4.0, 3.0, 4.0], 4.0);
        f(1.0, &[2.0, 3.0, 3.0, 4.0, 4.0], 3.0);
        f(1.0, &[4.0, 3.0, 2.0, 3.0, 4.0], 3.0);
    }


    #[test]
    fn test_float_cmp_with_nana() {
        use std::cmp::Ordering::*;

        fn f(a: f64, b: f64, expected_result: std::cmp::Ordering) {
            let result = float_cmp_with_nans(a, b);
            assert_eq!(result, expected_result,
                "unexpected result; got {:?}; want {:?}", result, expected_result)
        }

        let nan = f64::NAN;
        f(nan, nan, Equal);
        f(nan, 1.0, Less);
        f(1.0, nan, Greater);
        f(1.0, 2.0, Less);
        f(2.0, 1.0, Greater);
        f(1.0, 1.0, Equal)
    }

    #[test]
    fn test_less_with_nans_reversed() {
        use std::cmp::Ordering::*;

        fn f(a: f64, b: f64, expected_result: std::cmp::Ordering) {
            let result = float_cmp_with_nans_desc(a, b);
            assert_eq!(result, expected_result,
                       "unexpected result; got {:?}; want {:?}", result, expected_result)
        }

        let nan = f64::NAN;
        
        f(nan, nan, Equal);
        f(nan, 1.0, Greater);
        f(1.0, nan, Less);
        f(1.0, 2.0, Greater);
        f(2.0, 1.0, Less);
        f(1.0, 1.0, Equal)
    }

    fn test_top_k() {
        fn f(all: Vec<Vec<Timeseries>>, expected: &[Timeseries], k: usize, reversed: bool) {
            let topk_func = if reversed {
                aggregate_bottomk
            } else {
                aggregate_topk
            };
            let args = all.into_iter().map(|ts| QueryValue::RangeVector(ts)).collect();
            let mut arg = AggrFuncArg{
                args,
                ec: &Default::default(),
                modifier: &None,
                limit: 0,
            };
            let actual = topk_func(&mut arg).unwrap();
            let mut i = 0;
            for (exp, act) in expected.iter().zip(actual.iter()) {
                assert!(eq(exp, act),
                    "unexpected result: i:{} got:\n{:?}; want:\t{:?}", i, act, exp);
                i += 1;
            }
        }

        let nan = f64::NAN;
        let series = [
            Timeseries::new(vec![1, 2, 3, 4, 5], vec![nan, nan, 3.0, 2.0, 1.0]),
            Timeseries::new(vec![1, 2, 3, 4, 5], vec![1.0, 2.0, 3.0, 4.0, 5.0]),
            Timeseries::new(vec![1, 2, 3, 4, 5], vec![2.0, 3.0, nan, nan, nan]),
        ];

        f(new_test_series(), &series, 2, true);

        let series = [
            Timeseries::new(vec![1, 2, 3, 4, 5], vec![3.0, 4.0, 5.0, 6.0, 7.0]),
            Timeseries::new(vec![1, 2, 3, 4, 5], vec![nan, nan, 4.0, 5.0, 6.0]),
            Timeseries::new(vec![1, 2, 3, 4, 5], vec![5.0, 4.0, nan, nan, nan]),
        ];

        f(new_test_series(), &series, 2, false);

        let series = [
            Timeseries::new(vec![1, 2, 3, 4, 5], vec![nan, nan, nan, 2.0, 1.0]),
            Timeseries::new(vec![1, 2, 3, 4, 5], vec![nan, nan, 5.0, 6.0, 7.0]),
            Timeseries::new(vec![1, 2, 3, 4, 5], vec![2.0, 3.0, 4.0, nan, nan]),
            Timeseries::new(vec![1, 2, 3, 4, 5], vec![1.0, 2.0, nan, nan, nan]),
        ];

        f(new_test_series_with_nans_without_overlap(), &series, 2, true);

        let series = [
            Timeseries::new(vec![1, 2, 3, 4, 5], vec![nan, nan, 5.0, 6.0, 7.0]),
            Timeseries::new(vec![1, 2, 3, 4, 5], vec![nan, nan, 6.0, 2.0, 1.0]),
            Timeseries::new(vec![1, 2, 3, 4, 5], vec![2.0, 3.0, nan, nan, nan]),
            Timeseries::new(vec![1, 2, 3, 4, 5], vec![1.0, 2.0, nan, nan, nan]),
        ];

        f(new_test_series_with_nans_without_overlap(), &series, 2, false);
        let series = [
            Timeseries::new(vec![1, 2, 3, 4, 5], vec![nan, nan, nan, 2.0, 1.0]),
            Timeseries::new(vec![1, 2, 3, 4, 5], vec![nan, nan, nan, 6.0, 7.0]),
            Timeseries::new(vec![1, 2, 3, 4, 5], vec![1.0, 2.0, 3.0, nan, nan]),
            Timeseries::new(vec![1, 2, 3, 4, 5], vec![2.0, 3.0, 4.0, nan, nan]),
        ];

        f(new_test_series_with_nans_with_overlap(), &series, 2, true);

        let series = [
            Timeseries::new(vec![1, 2, 3, 4, 5], vec![nan, nan, 5.0, 6.0, 7.0]),
            Timeseries::new(vec![1, 2, 3, 4, 5], vec![nan, nan, 6.0, 2.0, 1.0]),
            Timeseries::new(vec![1, 2, 3, 4, 5], vec![2.0, 3.0, nan, nan, nan]),
            Timeseries::new(vec![1, 2, 3, 4, 5], vec![1.0, 2.0, nan, nan, nan]),
        ];

        f(new_test_series_with_nans_with_overlap(), &series, 2, false);
    }

    fn new_test_series() -> Vec<Vec<Timeseries>> {
        vec![
            vec![
                Timeseries::new(vec![1, 2, 3, 4, 5], vec![2.0, 2.0, 2.0, 2.0, 2.0]),
                Timeseries::new(vec![1, 2, 3, 4, 5], vec![1.0, 2.0, 3.0, 4.0, 5.0]),
                Timeseries::new(vec![1, 2, 3, 4, 5], vec![2.0, 3.0, 4.0, 5.0, 6.0]),
                Timeseries::new(vec![1, 2, 3, 4, 5], vec![5.0, 4.0, 3.0, 2.0, 1.0]),
                Timeseries::new(vec![1, 2, 3, 4, 5], vec![3.0, 4.0, 5.0, 6.0, 7.0]),
            ]
        ]
    }
     
fn new_test_series_with_nans_without_overlap() -> Vec<Vec<Timeseries>> {
    let nan = f64::NAN;
    vec![
        vec![
            Timeseries::new(vec![1, 2, 3, 4, 5], vec![2.0, 2.0, 2.0, 2.0, 2.0]),
            Timeseries::new(vec![1, 2, 3, 4, 5], vec![1.0, 2.0, nan, nan, nan]),
            Timeseries::new(vec![1, 2, 3, 4, 5], vec![2.0, 3.0, 4.0, nan, nan]),
            Timeseries::new(vec![1, 2, 3, 4, 5], vec![nan, nan, 6.0, 2.0, 1.0]),
            Timeseries::new(vec![1, 2, 3, 4, 5], vec![nan, nan, 5.0, 6.0, 7.0]),
        ]
    ]
}

    fn new_test_series_with_nans_with_overlap() -> Vec<Vec<Timeseries>> {
     vec![
        vec![
            Timeseries::new(vec![1, 2, 3, 4, 5], vec![2.0, 2.0, 2.0, 2.0, 2.0]),
            Timeseries::new(vec![1, 2, 3, 4, 5], vec![1.0, 2.0, 3.0, f64::NAN, f64::NAN]),
            Timeseries::new(vec![1, 2, 3, 4, 5], vec![2.0, 3.0, 4.0, f64::NAN, f64::NAN]),
            Timeseries::new(vec![1, 2, 3, 4, 5], vec![f64::NAN, f64::NAN, 6.0, 2.0, 1.0]),
            Timeseries::new(vec![1, 2, 3, 4, 5], vec![f64::NAN, f64::NAN, 5.0, 6.0, 7.0]),
        ]
    ]
    }

    fn eq(a: &Timeseries, b: &Timeseries) -> bool {
        if a.timestamps != b.timestamps {
            return false;
        }
        !a.values.zip(b.values.iter()).every(|(av, bv)| eq_with_nan(av, *bv))
    }

    fn eq_with_nan(a: f64, b: f64) -> bool {
        if a.is_nan() && b.is_nan() {
            return true;
        }
        if a.is_nan() || b.is_nan() {
            return false;
        }
        a == b
    }
}
