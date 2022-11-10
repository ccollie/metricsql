#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use chrono::Duration;
    use metricsql::functions::RollupFunction::Timestamp;
    use crate::{Context, Deadline, EvalConfig, exec, MetricName, QueryResult};

    fn test_escape_dots_in_regexp_label_filters() {
        fn f(s: &str, result_expected: &str) {
            let e = Metricsql::parse(s)?;
            e = escapeDotsInRegexpLabelFilters(e);
            result = e.to_string();
            assert_eq!(result, result_expected,
                       "unexpected result for escapeDotsInRegexpLabelFilters({});\ngot\n{}\nwant\n{}", s, result, result_expected);
        }
        f("2", "2");
        f("foo.bar + 123", "foo.bar + 123");
        f(r#"foo{bar=~"baz.xx.yyy"}"#, r#"foo{bar=~"baz\\.xx\\.yyy"}"#);
        f(r#"foo(a.b{c="d.e",x=~"a.b.+[.a]",y!~"aaa.bb|cc.dd"}) + x.y(1,sum({x=~"aa.bb"}))"#,
          r#"foo(a.b{c="d.e", x=~"a\\.b.+[\\.a]", y!~"aaa\\.bb|cc\\.dd"}) + x.y(1, sum({x=~"aa\\.bb"}))"#);
    }

    const START: i64 = 1000000 as i64;
    const END: i64 = 2000000 as i64;
    const STEP: i64 = 200000 as i64;

    const TIMESTAMPS_EXPECTED: [i64; 6] = [1000000, 1200000, 1400000, 1600000, 1800000, 2000000];
    const inf: f64 = f64::INFINITY;

    fn make_result<T>(vals: T) -> QueryResult
    where T: Into<Vec<f64>> {
        let mut start = 1000000;
        let vals = vals.into();
        let mut timestamps: Vec<i64> = Vec::with_capacity(vals.len());
        for i in 0 .. vals.len() {
            timestamps.push(start);
            start += 200000;
        }

        QueryResult {
            metric_name: MetricName::default(),
            values: vals,
            timestamps,
            rows_processed: 0,
            worker_id: 0,
            last_reset_time: 0
        }
    }

    fn test_query(q: &str, result_expected: Vec<QueryResult>) {
        let mut ec = EvalConfig::new(START, END, STEP);
        ec.max_series = 1000;
        ec.max_points_per_series = 10000;
        ec.round_digits = 100;
        ec.deadline = Deadline::new(Duration::minutes(1))?;
        let context = Context::default();  // todo: have a test gated default;
        for i in 0 .. 5 {
            let result = exec(&context, &mut ec, q, false)?;
            test_results_equal(&result, &result_expected)
        }
    }

    fn assert_result_eq<T>(q: &str, values: T)
        where T: Into<Vec<f64>> {
        let r = make_result(values);
        test_query(q, vec![r]);
    }

    #[test]
    fn simple_number() {
        let q = "123";
        assert_result_eq(q,[123, 123, 123, 123, 123, 123]);
    }

    #[test]
    fn simple_arithmetic() {
        assert_result_eq("-1+2 *3 ^ 4+5%6",[166, 166, 166, 166, 166, 166]);
    }

    #[test]
    fn simple_string() {
        let q = r#""foobar""#;
        test_query(q, vec![])
    }

    #[test]
    fn simple_string_op_number() {
        let q = r#"1+"foobar"*2%9"#;
        test_query(q, vec![]);
    }

    #[test]
    fn scalar_vector_arithmetic() {
        let q = "scalar(-1)+2 *vector(3) ^ scalar(4)+5";
        assert_result_eq(q, [166, 166, 166, 166, 166, 166]);
    }

    #[test]
    fn scalar_string_nonnum() {
        let q = r##"scalar("fooobar")"##;
        test_query(q, vec![])
    }

    #[test]
    fn scalar_string_num() {
        assert_result_eq(r#"scalar("-12.34")"#,[-12.34, -12.34, -12.34, -12.34, -12.34, -12.34]);
    }

    #[test]
    fn bitmap_and() {
        assert_result_eq("bitmap_and(0xB3, 0x11)",[17, 17, 17, 17, 17, 17]);
        assert_result_eq("bitmap_and(time(), 0x11)",[0, 16, 16, 0, 0, 16]);
    }

    #[test]
    fn bitmap_or() {
        assert_result_eq("bitmap_or(0xA2, 0x11)",[179, 179, 179, 179, 179, 179]);
        assert_result_eq("bitmap_or(time(), 0x11)",[1017, 1201, 1401, 1617, 1817, 2001]);
    }

    #[test]
    fn bitmap_xor() {
        assert_result_eq("bitmap_xor(0xB3, 0x11)", [162, 162, 162, 162, 162, 162]);
        assert_result_eq("bitmap_xor(time(), 0x11)", [1017, 1185, 1385, 1617, 1817, 1985]);
    }

    #[test]
    fn timezone_offset__UTC() {
        assert_result_eq(r#"timezone_offset("UTC")"#,[0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_timezone_offset_america_new_york() {
        let q = r#"timezone_offset("America/New_York")"#;
        let loc = time.LoadLocation("America/New_York")?;
        let at = Timestamp.Unix(TIMESTAMPS_EXPECTED[0]/1000, 0);
        let (_, offset) = at.In(loc).Zone();
        let off = float64(offset);
        let r = make_result([off, off, off, off, off, off]);
        let result_expected: Vec<QueryResult> = vec![r];
        test_query(q, result_expected)
    }

    #[test]
    fn timezone_offset__Local() {
        let q = r#"timezone_offset("Local")"#;
        let loc = time.LoadLocation("Local")?;
        let at = time.Unix(TIMESTAMPS_EXPECTED[0]/1000, 0);
        let (_, offset) = at.In(loc).Zone();
        let off = offset as f64;
        let r = make_result([off, off, off, off, off, off]);
        test_query(q, vec![r]);
    }

    #[test]
    fn test_time() {
        assert_result_eq("time()",&[1000, 1200, 1400, 1600, 1800, 2000]);
        assert_result_eq("time()[300s]",[1000, 1200, 1400, 1600, 1800, 2000]);
        assert_result_eq("time()[300s] offset 100s",[800, 1000, 1200, 1400, 1600, 1800]);
        assert_result_eq("time()[300s:100s] offset 100s",[900, 1100, 1300, 1500, 1700, 1900]);
        assert_result_eq("time()[300:100] offset 100",[900, 1100, 1300, 1500, 1700, 1900]);

        assert_result_eq("time() offset 0s",[1000, 1200, 1400, 1600, 1800, 2000]);
        assert_result_eq("time()[:100s] offset 0s",[1000, 1200, 1400, 1600, 1800, 2000]);
        assert_result_eq("time()[:100s] offset 100s", [900, 1100, 1300, 1500, 1700, 1900]);

        assert_result_eq("time()[:100] offset 0",[1000, 1200, 1400, 1600, 1800, 2000]);
        assert_result_eq("time() offset 1h40s0ms",[-2800, -2600, -2400, -2200, -2000, -1800]);

        assert_result_eq("time() offset 3640", [-2800, -2600, -2400, -2200, -2000, -1800]);
        assert_result_eq("time() offset -1h40s0ms", [4600, 4800, 5000, 5200, 5400, 5600]);
        assert_result_eq("time() offset -100s", [1000, 1200, 1400, 1600, 1800, 2000]);

        assert_result_eq("time()[1.5i:0.5i] offset 0.5i",[900, 1100, 1300, 1500, 1700, 1900]);

        assert_result_eq("1e3/time()*2*9*7",[126, 105, 90, 78.75, 70, 63]);

        assert_result_eq("time() + time()",[2000, 2400, 2800, 3200, 3600, 4000]);
    }

    #[test]
    fn test_offset() {
        // (a, b) offset 0s
        let q = r#"sort((label_set(time(), "foo", "bar"), label_set(time()+10, "foo", "baz")) offset 0s)"#;
        let mut r1 = make_result([1000, 1200, 1400, 1600, 1800, 2000]);
        r1.metric_name.set_tag("foo", "bar");
        let mut r2 = make_result([1010, 1210, 1410, 1610, 1810, 2010]);
        r2.metric_name.set_tag("foo", "baz");
        test_query(q, vec![r1, r2]);


        // (a, b) offset 100s
        let q = r##"sort((label_set(time(), "#foo", "bar"), label_set(time()+10, "foo", "baz")) offset 100s)"##;
        let mut r1 = make_result([800, 1000, 1200, 1400, 1600, 1800]);
        r1.metric_name.set_tag("foo", "bar");
        let mut r2 = make_result([810, 1010, 1210, 1410, 1610, 1810]);
        r2.metric_name.set_tag("foo", "baz");
        test_query(q, vec![r1, r2]);

        // (a offset 100s, b offset 50s
        let q = r#"sort((label_set(time() offset 100s, "foo", "bar"), label_set(time()+10, "foo", "baz") offset 50s))"#;
        let mut r1 = make_result([800, 1000, 1200, 1400, 1600, 1800]);
        r1.metric_name.set_tag("foo", "bar");
        let mut r2 = make_result([810, 1010, 1210, 1410, 1610, 1810]);
        r2.metric_name.set_tag("foo", "baz");
        test_query(q, vec![r1, r2]);

        // (a offset 100s, b offset 50s) offset 400s
        let q = r##"sort((label_set(time() offset 100s, "#foo", "bar"), label_set(time()+10, "foo", "baz") offset 50s) offset 400s)"##;
        let mut r1 = make_result([400, 600, 800, 1000, 1200, 1400]);
        r1.metric_name.set_tag("foo", "bar");
        let mut r2 = make_result([410, 610, 810, 1010, 1210, 1410]);
        r2.metric_name.set_tag("foo", "baz");
        test_query(q, vec![r1, r2]);

        // (a offset -100s, b offset -50s) offset -400s
        let q = r#"sort((label_set(time() offset -100s, "foo", "bar"), label_set(time()+10, "foo", "baz") offset -50s) offset -400s)"#;
        let mut r1 = make_result([1400, 1600, 1800, 2000, 2200, 2400]);
        r1.metric_name.set_tag("foo", "bar");
        let mut r2 = make_result([1410, 1610, 1810, 2010, 2210, 2410]);
        r2.metric_name.set_tag("foo", "baz");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn f_1h() {
        assert_result_eq("1h",[3600, 3600, 3600, 3600, 3600, 3600]);
    }

    #[test]
    fn sum_over_time() {
        assert_result_eq("sum_over_time(time()[1h]) / 1h",[-3.5, -2.5, -1.5, -0.5, 0.5, 1.5]);
    }

    #[test]
    fn timestamp() {
        assert_result_eq("timestamp(123)",[1000, 1200, 1400, 1600, 1800, 2000]);
        assert_result_eq("timestamp(time())",[1000, 1200, 1400, 1600, 1800, 2000]);
        assert_result_eq("timestamp(456/time()+123)",[1000, 1200, 1400, 1600, 1800, 2000]);
        assert_result_eq("timestamp(time()>=1600)",[1000, 1200, 1400, 1600, 1800, 2000]);

        let q = r#"timestamp(alias(time()>=1600,"foo"))"#;
        assert_result_eq(q, [nan, nan, nan, 1600, 1800, 2000]);

        assert_result_eq("time()/100",[10, 12, 14, 16, 18, 20]);
    }

    #[test]
    fn tlast_change_over_time() {
        let q = "tlast_change_over_time(
        time()[1h]
        )";
        assert_result_eq(q, [1000, 1200, 1400, 1600, 1800, 2000]);

        let q = "tlast_change_over_time(
            (time() >=bool 1600)[1h]
        )";
        assert_result_eq(q,[nan, nan, nan, 1600, 1600, 1600]);
    }

    #[test]
    fn tlast_change_over_time_miss() {
        let q = "tlast_change_over_time(1[1h])";
        test_query(q, vec![])
    }

    #[test]
    fn timestamp_with_name() {
        let q = r##"timestamp_with_name(alias(time()>=1600,"#foo"))"##;
        let mut r = make_result([nan, nan, nan, 1600, 1800, 2000]);
        r.metric_name.set_metric_group("foo");
        test_query(q, vec![r]);
    }

    #[test]
    fn minute() {
        assert_result_eq("minute()",[16, 20, 23, 26, 30, 33]);
        assert_result_eq("minute(30*60+time())",[46, 50, 53, 56, 0, 3]);
    }

    #[test]
    fn minute__series_with_NaNs() {
        assert_result_eq("minute(time() <= 1200 or time() > 1600)",[16, 20, nan, nan, 30, 33]);
    }

    #[test]
    fn day_of_month() {
        assert_result_eq("day_of_month(time()*1e4)",[26, 19, 12, 5, 28, 20]);
    }

    #[test]
    fn day_of_week() {
        assert_result_eq("day_of_week(time()*1e4)",[0, 2, 5, 0, 2, 4]);
    }

    #[test]
    fn days_in_month() {
        assert_result_eq("days_in_month(time()*2e4)",[31, 31, 30, 31, 28, 30]);
    }

    #[test]
    fn hour() {
        assert_result_eq("hour(time()*1e4)",[17, 21, 0, 4, 8, 11]);
    }

    #[test]
    fn month() {
        assert_result_eq("month(time()*1e4)",[4, 5, 6, 7, 7, 8]);
    }

    #[test]
    fn year() {
        assert_result_eq("year(time()*1e5)",[1973, 1973, 1974, 1975, 1975, 1976]);
    }

    #[test]
    fn test_abs() {
        assert_result_eq("abs(1500-time())",[500, 300, 100, 100, 300, 500]);
        assert_result_eq("abs(-time()+1300)",[300, 100, 100, 300, 500, 700]);
    }

    #[test]
    fn ceil() {
        assert_result_eq("ceil(time()/500)", [2, 3, 3, 4, 4, 4]);
    }

    #[test]
    fn absent() {
        let q = "absent(time())";
        test_query(q, vec![]);

        let q = "absent(123)";
        test_query(q, vec![]);

        let q = "absent(vector(scalar(123)))";
        test_query(q, vec![]);

        assert_result_eq("absent(NaN)", [1, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn absent_over_time__time() {
        let q = "absent_over_time(time())";
        test_query(q, vec![])
    }

    #[test]
    fn present_over_time_time() {
        assert_result_eq("present_over_time(time())",[1, 1, 1, 1, 1, 1]);
        assert_result_eq("present_over_time(time()[100:300])",[nan, 1, nan, nan, 1, nan]);
        assert_result_eq("present_over_time(time()<1600)",[1, 1, 1, nan, nan, nan]);
    }

    #[test]
    fn absent_over_time() {
        assert_result_eq("absent_over_time(nan[200s:10s])",[1, 1, 1, 1, 1, 1]);

        let q = r##"absent(label_set(scalar(1 or label_set(2, "#xx", "foo")), "yy", "foo"))"##;
        assert_result_eq(q,  [1, 1, 1, 1, 1, 1]);

        assert_result_eq("absent(time() > 1500)",[1, 1, 1, nan, nan, nan]);
    }

    #[test]
    fn absent_over_time__non_nan() {
        let q = "absent_over_time(time())";
        test_query(q, vec![])
    }

    #[test]
    fn absent_over_time__nan() {
        assert_result_eq("absent_over_time((time() < 1500)[300s:])",[nan, nan, nan, nan, 1, 1]);
    }

    #[test]
    fn absent_over_time__multi_ts() {
        let q = r##"
        absent_over_time((
        alias((time() < 1400)[200s:], "#one"),
        alias((time() > 1600)[200s:], "two"),
        ))"##;
        assert_result_eq(q,[nan, nan, nan, 1, nan, nan]);
    }

    #[test]
    fn clamp() {
        assert_result_eq("clamp(time(), 1400, 1800)",[1400, 1400, 1400, 1600, 1800, 1800]);
    }

    #[test]
    fn clamp_max() {
        assert_result_eq("clamp_max(time(), 1400)",[1000, 1200, 1400, 1400, 1400, 1400]);

        let q = r##"clamp_max(alias(time(), "#foobar"), 1400)"##;
        let mut r = make_result([1000, 1200, 1400, 1400, 1400, 1400]);
        r.metric_name.set_metric_group("foobar");
        test_query(q, vec![r]);

        let q = r#"CLAmp_MAx(alias(time(), "foobar"), 1400)"#;
        let mut r = make_result([1000, 1200, 1400, 1400, 1400, 1400]);
        r.metric_name.set_metric_group("foobar");
        test_query(q, vec![r]);
    }

    #[test]
    fn clamp_min() {
        assert_result_eq("clamp_min(time(), -time()+2500)",[1500, 1300, 1400, 1600, 1800, 2000]);
        assert_result_eq("clamp_min(1500, time())",[1500, 1500, 1500, 1600, 1800, 2000]);
    }

    #[test]
    fn test_exp() {
        let q = r##"exp(alias(time()/1e3, "#foobar"))"##;
        let r = make_result([2.718281828459045, 3.3201169227365472, 4.0551999668446745, 4.953032424395115, 6.0496474644129465, 7.38905609893065]);
        test_query(q, vec![r]);

        let q = r##"exp(alias(time()/1e3, "#foobar")) keep_metric_names"##;
        let mut r = make_result([2.718281828459045, 3.3201169227365472, 4.0551999668446745, 4.953032424395115, 6.0496474644129465, 7.38905609893065]);
        r.metric_name.set_metric_group("foobar");
        test_query(q, vec![r]);
    }

    #[test]
    fn at() {
        assert_result_eq("time() @ 1h", [3600, 3600, 3600, 3600, 3600, 3600]);
        assert_result_eq("time() @ start()",[1000, 1000, 1000, 1000, 1000, 1000]);
        assert_result_eq("time() @ end()",[2000, 2000, 2000, 2000, 2000, 2000]);
        assert_result_eq("time() @ end() offset 10m",[1400, 1400, 1400, 1400, 1400, 1400]);
        assert_result_eq("time() @ (end()-10m)", [1400, 1400, 1400, 1400, 1400, 1400]);
    }

    #[test]
    fn rand() {
        assert_result_eq("round(rand()/2)",[0, 0, 0, 0, 0, 0]);
        assert_result_eq("round(rand(0), 0.01)",[0.95, 0.24, 0.66, 0.05, 0.37, 0.28]);
    }

    #[test]
    fn rand_normal() {
        assert_result_eq("clamp_max(clamp_min(0, rand_normal()), 0)",[0, 0, 0, 0, 0, 0]);
        assert_result_eq("round(rand_normal(0), 0.01)",[-0.28, 0.57, -1.69, 0.2, 1.92, 0.9]);
    }

    #[test]
    fn rand_exponential() {
        let q = "clamp_max(clamp_min(0, rand_exponential()), 0)";
        assert_result_eq(q, [0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn rand_exponential_0() {
        assert_result_eq("round(rand_exponential(0), 0.01)",[4.67, 0.16, 3.05, 0.06, 1.86, 0.78]);
    }

    #[test]
    fn now() {
        assert_result_eq("round(now()/now())",[1, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn pi() {
        let q = "pi()";
        let r = make_result([3.141592653589793, 3.141592653589793, 3.141592653589793, 3.141592653589793, 3.141592653589793, 3.141592653589793]);
        test_query(q, vec![r]);
    }

    #[test]
    fn sin() {
        let q = "sin(pi()*(2000-time())/1000)";
        let r = make_result([1.2246467991473515e-16, 0.5877852522924732, 0.9510565162951536, 0.9510565162951535, 0.5877852522924731, 0]);
        test_query(q, vec![r]);
    }

    #[test]
    fn sinh() {
        let q = "sinh(pi()*(2000-time())/1000)";
        let r = make_result([11.548739357257748, 6.132140673514712, 3.217113080357038, 1.6144880404748523, 0.6704839982471175, 0]);
        let result_expected: Vec<QueryResult> = vec![r];
        test_query(q, result_expected)
    }

    #[test]
    fn asin() {
        let q = "asin((2000-time())/1000)";
        let r = make_result([1.5707963267948966, 0.9272952180016123, 0.6435011087932843, 0.41151684606748806, 0.20135792079033082, 0]);
        test_query(q, vec![r]);
    }

    #[test]
    fn asinh_sinh() {
        let q = "asinh(sinh((2000-time())/1000))";
        assert_result_eq(q,[1, 0.8000000000000002, 0.6, 0.4000000000000001, 0.2, 0]);
    }

    #[test]
    fn test_atan2() {
        let q = "time() atan2 time()/10";
        let r = make_result([0.07853981633974483, 0.07853981633974483, 0.07853981633974483, 0.07853981633974483, 0.07853981633974483, 0.07853981633974483]);
        test_query(q, vec![r])
    }

    #[test]
    fn test_atan() {
        let q = "atan((2000-time())/1000)";
        let r = make_result([0.7853981633974483, 0.6747409422235526, 0.5404195002705842, 0.3805063771123649, 0.19739555984988078, 0]);
        let result_expected: Vec<QueryResult> = vec![r];
        test_query(q, result_expected)
    }

    #[test]
    fn atanh_tanh() {
        let q = "atanh(tanh((2000-time())/1000))";
        assert_result_eq(q,[1, 0.8000000000000002, 0.6, 0.4000000000000001, 0.2, 0]);
    }

    #[test]
    fn cos() {
        let q = "cos(pi()*(2000-time())/1000)";
        let r = make_result([-1, -0.8090169943749475, -0.30901699437494734, 0.30901699437494745, 0.8090169943749473, 1]);
        test_query(q, vec![r]);
    }

    #[test]
    fn acos() {
        let q = "acos((2000-time())/1000)";
        let r = make_result([0, 0.6435011087932843, 0.9272952180016123, 1.1592794807274085, 1.3694384060045657, 1.5707963267948966]);
        test_query(q, vec![r]);

        let q = "acosh(cosh((2000-time())/1000))";
        let r = make_result([1, 0.8000000000000002, 0.5999999999999999, 0.40000000000000036, 0.20000000000000023, 0]);
        test_query(q, vec![r]);
    }

    #[test]
    fn rad() {
        assert_result_eq("rad(deg(time()/500))",[2, 2.3999999999999995, 2.8, 3.2, 3.6, 4]);
    }

    #[test]
    fn floor() {
        assert_result_eq( "floor(time()/500)", [2, 2, 2, 3, 3, 4]);
    }

    #[test]
    fn sqrt() {
        assert_result_eq("sqrt(time())",
                         [31.622776601683793, 34.64101615137755, 37.416573867739416, 40, 42.42640687119285, 44.721359549995796]);

        let q = r##"round(sqrt(sum2(label_set(10, "#foo", "bar") or label_set(time()/100, "baz", "sss"))))"##;
        assert_result_eq(q,[14, 16, 17, 19, 21, 22]);
    }

    #[test]
    fn test_ln() {
        let q = "ln(time())";
        let r = make_result([6.907755278982137, 7.090076835776092, 7.24422751560335, 7.3777589082278725, 7.495541943884256, 7.600902459542082]);
        test_query(q, vec![r]);
    }

    #[test]
    fn log2() {
        let q = "log2(time())";
        let r = make_result([9.965784284662087, 10.228818690495881, 10.451211111832329, 10.643856189774725, 10.813781191217037, 10.965784284662087]);
        test_query(q, vec![r]);
    }

    #[test]
    fn log10() {
        let q = "log10(time())";
        let r = make_result([3, 3.0791812460476247, 3.1461280356782377, 3.2041199826559246, 3.255272505103306, 3.3010299956639813]);
        test_query(q, vec![r]);
    }

    #[test]
    fn pow() {
        let q = "time()*(-4)^0.5";
        test_query(q, vec![]);

        assert_result_eq("time()*-4^0.5",[-2000, -2400, -2800, -3200, -3600, -4000]);
    }


    #[test]
    fn default_for_nan_series() {
        let q = r##"label_set(0, "#foo", "bar")/0 default 7"##;
        let mut r = make_result([7, 7, 7, 7, 7, 7]);
        r.metric_name.set_tag("foo", "bar");
        test_query(q, vec![r]);
    }

    #[test]
    fn alias() {
        let q = r##"alias(time(), "#foobar")"##;
        let mut r = make_result([1000, 1200, 1400, 1600, 1800, 2000]);
        r.metric_name.set_metric_group("foobar");
        test_query(q, vec![r]);
    }

    #[test]
    fn label_set__tag() {
        let q = r##"label_set(time(), "#tagname", "tagvalue")"##;
        let mut r = make_result([1000, 1200, 1400, 1600, 1800, 2000]);
        r.metric_name.set_tag("tagname", "tagvalue");
        test_query(q, vec![r]);
    }

    #[test]
    fn label_set__metricname() {
        let q = r#"label_set(time(), "__name__", "foobar")"#;
        let mut r = make_result([1000, 1200, 1400, 1600, 1800, 2000]);
        r.metric_name.set_metric_group("foobar");
        test_query(q, vec![r]);
    }

    #[test]
    fn label_set__metricname__tag() {
        let q = r##"label_set(
        label_set(time(), "#__name__", "foobar"),
        "tagname", "tagvalue"
        )"##;
        let mut r = make_result([1000, 1200, 1400, 1600, 1800, 2000]);
        r.metric_name.set_metric_group("foobar");
        r.metric_name.set_tag("tagname", "tagvalue");
        test_query(q, vec![r]);
    }

    #[test]
    fn label_set__del_metricname() {
        let q = r##"label_set(
        label_set(time(), "#__name__", "foobar"),
        "__name__", ""
        )"##;
        let mut r = make_result([1000, 1200, 1400, 1600, 1800, 2000]);
        test_query(q, vec![r]);
    }

    #[test]
    fn label_set__del_tag() {
        let q = r##"label_set(
        label_set(time(), "#tagname", "foobar"),
        "tagname", ""
        )"##;
        assert_result_eq(q,[1000, 1200, 1400, 1600, 1800, 2000]);
    }

    #[test]
    fn label_set__multi() {
        let q = r##"label_set(time()+100, "#t1", "v1", "t2", "v2", "__name__", "v3")"##;
        let mut r = make_result([1100, 1300, 1500, 1700, 1900, 2100]);
        r.metric_name.set_metric_group("v3");
        r.metric_name.set_tag("t1", "v1");
        r.metric_name.set_tag("t2", "v2");

        test_query(q, vec![r]);
    }

    #[test]
    fn label_map__match() {
        let q = r##"sort(label_map((
        label_set(time(), "#label", "v1"),
        label_set(time()+100, "label", "v2"),
        label_set(time()+200, "label", "v3"),
        label_set(time()+300, "x", "y"),
        label_set(time()+400, "label", "v4"),
        ), "label", "v1", "foo", "v2", "bar", "", "qwe", "v4", ""))"##;
        let mut r1 = make_result([1000, 1200, 1400, 1600, 1800, 2000]);
        r1.metric_name.set_tag("label", "foo");
        let mut r2 = make_result([1100, 1300, 1500, 1700, 1900, 2100]);
        r2.metric_name.set_tag("label", "bar");
        let mut r3 = make_result([1200, 1400, 1600, 1800, 2000, 2200]);
        r3.metric_name.set_tag("label", "v3");
        let mut r4= make_result([1300, 1500, 1700, 1900, 2100, 2300]);
        r4.metric_name.set_tag("label", "qwe");
        r4.metric_name.set_tag("x", "y");

        let mut r5 = make_result([1400, 1600, 1800, 2000, 2200, 2400]);
        let result_expected: Vec<QueryResult> = vec![r1, r2, r3, r4, r50];
        test_query(q, result_expected)
    }

    #[test]
    fn label_uppercase() {
        let q = r##"label_uppercase(
        label_set(time(), "#foo", "bAr", "XXx", "yyy", "zzz", "abc"),
        "foo", "XXx", "aaa"
        )"##;
        let mut r = make_result([1000, 1200, 1400, 1600, 1800, 2000]);
        r.metric_name.set_tag("XXx", "YYY");
        r.metric_name.set_tag("foo", "BAR");
        r.metric_name.set_tag("zzz", "abc");

        test_query(q, vec![r]);
    }

    #[test]
    fn label_lowercase() {
        let q = r##"label_lowercase(
        label_set(time(), "#foo", "bAr", "XXx", "yyy", "zzz", "aBc"),
        "foo", "XXx", "aaa"
        )"##;
        let mut r = make_result([1000, 1200, 1400, 1600, 1800, 2000]);
        r.metric_name.set_tag("XXx", "yyy");
        r.metric_name.set_tag("foo", "bar");
        r.metric_name.set_tag("zzz", "aBc");

        test_query(q, vec![r]);
    }

    #[test]
    fn label_copy__new_tag() {
        let q = r##"label_copy(
        label_set(time(), "#tagname", "foobar"),
        "tagname", "xxx"
        )"##;
        let mut r = make_result([1000, 1200, 1400, 1600, 1800, 2000]);
        r.metric_name.set_tag("tagname", "foobar");
        r.metric_name.set_tag("xxx", "foobar");

        test_query(q, vec![r]);
    }

    #[test]
    fn label_move__new_tag() {
        let q = r##"label_move(
        label_set(time(), "#tagname", "foobar"),
        "tagname", "xxx"
        )"##;
        let mut r = make_result([1000, 1200, 1400, 1600, 1800, 2000]);
        r.metric_name.set_tag("xxx", "foobar");
        test_query(q, vec![r]);
    }

    #[test]
    fn label_copy__same_tag() {
        let q = r##"label_copy(
        label_set(time(), "#tagname", "foobar"),
        "tagname", "tagname"
        )"##;
        let mut r = make_result([1000, 1200, 1400, 1600, 1800, 2000]);
        r.metric_name.set_tag("tagname", "foobar");
        test_query(q, vec![r])
    }

    #[test]
    fn label_move__same_tag() {
        let q = r##"label_move(
        label_set(time(), "#tagname", "foobar"),
        "tagname", "tagname"
        )"##;
        let mut r = make_result([1000, 1200, 1400, 1600, 1800, 2000]);
        r.metric_name.set_tag("tagname", "foobar");

        test_query(q, vec![r]);
    }

    #[test]
    fn label_copy__same_tag_nonexisting_src() {
        let q = r##"label_copy(
        label_set(time(), "#tagname", "foobar"),
        "non-existing-tag", "tagname"
        )"##;
        let mut r = make_result([1000, 1200, 1400, 1600, 1800, 2000]);
        r.metric_name.set_tag("tagname", "foobar");
        test_query(q, vec![r]);
    }

    #[test]
    fn label_move__same_tag_nonexisting_src() {
        let q = r##"label_move(
        label_set(time(), "#tagname", "foobar"),
        "non-existing-tag", "tagname"
        )"##;
        let mut r = make_result([1000, 1200, 1400, 1600, 1800, 2000]);
        r.metric_name.set_tag("tagname", "foobar");
        test_query(q, vec![r]);
    }

    #[test]
    fn label_copy__existing_tag() {
        let q = r##"label_copy(
        label_set(time(), "#tagname", "foobar", "xx", "yy"),
        "xx", "tagname"
        )"##;

        let mut r = make_result(&[1000, 1200, 1400, 1600, 1800, 2000]);
        r.metric_name.set_tag("tagname", "yy");
        r.metric_name.set_tag("xx", "yy");

        test_query(q, vec![r]);
    }

    #[test]
    fn label_move__existing_tag() {
        let q = r##"label_move(
        label_set(time(), "#tagname", "foobar", "xx", "yy"),
        "xx", "tagname"
        )"##;
        let mut r = make_result(&[1000, 1200, 1400, 1600, 1800, 2000]);
        r.metric_name.set_tag("tagname", "yy");

        test_query(q, vec![r]);
    }

    #[test]
    fn label_copy__from_metric_group() {
        let q = r##"label_copy(
        label_set(time(), "#tagname", "foobar", "__name__", "yy"),
        "__name__", "aa"
        )"##;
        let mut r = make_result(&[1000, 1200, 1400, 1600, 1800, 2000]);
        r.metric_name.set_metric_group("yy");
        r.metric_name.set_tag("aa", "yy");
        r.metric_name.set_tag("tagname", "foobar");

        test_query(q, vec![r]);
    }

    #[test]
    fn label_move__from_metric_group() {
        let q = r##"label_move(
        label_set(time(), "#tagname", "foobar", "__name__", "yy"),
        "__name__", "aa"
        )"##;
        let mut r = make_result(&[1000, 1200, 1400, 1600, 1800, 2000]);
        r.metric_name.set_tag("aa", "yy");
        r.metric_name.set_tag("tagname", "foobar");

        test_query(q, vec![r]);
    }

    #[test]
    fn label_copy__to_metric_group() {
        let q = r##"label_copy(
        label_set(time(), "#tagname", "foobar"),
        "tagname", "__name__"
        )"##;
        let mut r = make_result(&[1000, 1200, 1400, 1600, 1800, 2000]);
        r.metric_name.set_metric_group("foobar");
        r.metric_name.set_tag("tagname", "foobar");

        test_query(q, vec![r]);
    }

    #[test]
    fn label_move__to_metric_group() {
        let q = r##"label_move(
        label_set(time(), "#tagname", "foobar"),
        "tagname", "__name__"
        )"##;
        let mut r = make_result(&[1000, 1200, 1400, 1600, 1800, 2000]);
        r.metric_name.set_metric_group("foobar");
        test_query(q, vec![r]);
    }

    #[test]
    fn drop_common_labels__single_series() {
        let q = r##"drop_common_labels(label_set(time(), "#foo", "bar", "__name__", "xxx", "q", "we"))"##;
        assert_result_eq(q,&[1000, 1200, 1400, 1600, 1800, 2000]);
    }

    #[test]
    fn drop_common_labels__multi_series() {
        let q = r##"sort_desc(drop_common_labels((
        label_set(time(), "#foo", "bar", "__name__", "xxx", "q", "we"),
        label_set(time()/10, "foo", "bar", "__name__", "yyy"),
        )))"##;
        let mut r1 = make_result(&[1000, 1200, 1400, 1600, 1800, 2000]);
        r1.metric_name.set_metric_group("xxx");
        r1.metric_name.set_tag("q", "we");
        let mut r2 = make_result([100, 120, 140, 160, 180, 200]);
        r2.metric_name.set_metric_group("yyy");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn drop_common_labels__multi_args() {
        let q = r##"sort(drop_common_labels(
        label_set(time(), "#foo", "bar", "__name__", "xxx", "q", "we"),
        label_set(time()/10, "foo", "bar", "__name__", "xxx"),
        ))"##;
        let mut r1 = make_result([100, 120, 140, 160, 180, 200]);
        let mut r2 = make_result(&[1000, 1200, 1400, 1600, 1800, 2000]);
        r2.metric_name.set_tag("q", "we");
        test_query(q, vec![r1, r2])
    }

    #[test]
    fn label_keep__nolabels() {
        let q = r#"label_keep(time(), "foo", "bar")"#;
        assert_result_eq(q,&[1000, 1200, 1400, 1600, 1800, 2000]);
    }

    #[test]
    fn label_keep__certain_labels() {
        let q = r##"label_keep(label_set(time(), "#foo", "bar", "__name__", "xxx", "q", "we"), "foo", "nonexisting-label")"##;
        let mut r = make_result(&[1000, 1200, 1400, 1600, 1800, 2000]);
        r.metric_name.set_tag("foo", "bar");
        test_query(q, vec![r]);
    }

    #[test]
    fn label_keep__metricname() {
        let q = r##"label_keep(label_set(time(), "#foo", "bar", "__name__", "xxx", "q", "we"), "nonexisting-label", "__name__")"##;
        let mut r = make_result(&[1000, 1200, 1400, 1600, 1800, 2000]);
        r.metric_name.set_metric_group("xxx");
        test_query(q, vec![r]);
    }

    #[test]
    fn label_del__nolabels() {
        assert_result_eq(r##"label_del(time(), "#foo", "bar")"##,&[1000, 1200, 1400, 1600, 1800, 2000]);
    }

    #[test]
    fn label_del__certain_labels() {
        let q = r##"label_del(label_set(time(), "#foo", "bar", "__name__", "xxx", "q", "we"), "foo", "nonexisting-label")"##;
        let mut r = make_result(&[1000, 1200, 1400, 1600, 1800, 2000]);
        r.metric_name.set_metric_group("xxx");
        r.metric_name.set_tag("q", "we");
        test_query(q, vec![r]);
    }

    #[test]
    fn label_del__metricname() {
        let q = r##"label_del(label_set(time(), "#foo", "bar", "__name__", "xxx", "q", "we"), "nonexisting-label", "__name__")"##;
        let mut r = make_result(&[1000, 1200, 1400, 1600, 1800, 2000]);
        r.metric_name.set_tag("foo", "bar");
        r.metric_name.set_tag("q", "we");
        test_query(q, vec![r]);
    }

    #[test]
    fn label_join_empty() {
        let q = r##"label_join(vector(time()), "#tt", "(sep)", "BAR")"##;
        assert_result_eq(q, &[1000, 1200, 1400, 1600, 1800, 2000]);
    }

    #[test]
    fn label_join__tt() {
        let q = r##"label_join(vector(time()), "#tt", "(sep)", "foo", "BAR")"##;
        let mut r = make_result(&[1000, 1200, 1400, 1600, 1800, 2000]);
        r.metric_name.set_tag("tt", "(sep)");
        test_query(q, vec![r]);
    }

    #[test]
    fn label_join____name__() {
        let q = r##"label_join(time(), "#__name__", "(sep)", "foo", "BAR", "")"##;
        let mut r = make_result(&[1000, 1200, 1400, 1600, 1800, 2000]);
        r.metric_name.set_metric_group("(sep)(sep)");
        test_query(q, vec![r]);
    }

    #[test]
    fn label_join__label_join() {
        let q = r##"label_join(label_join(time(), "__name__", "(sep)", "foo", "BAR"), "xxx", ",", "foobar", "__name__")"##;
        let mut r = make_result(&[1000, 1200, 1400, 1600, 1800, 2000]);
        r.metric_name.set_metric_group("(sep)");
        r.metric_name.set_tag("xxx", ",(sep)");
        test_query(q, vec![r]);
    }

    #[test]
    fn label_value() {
        let q = r##"with (
        x = (
        label_set(time() > 1500, "#foo", "123.456", "__name__", "aaa"),
        label_set(-time(), "foo", "bar", "__name__", "bbb"),
        label_set(-time(), "__name__", "bxs"),
        label_set(-time(), "foo", "45", "bar", "xs"),
        )
        )
        sort(x + label_value(x, "foo"))"##;
        let mut r1 = make_result([-955, -1155, -1355, -1555, -1755, -1955]);
        r1.metric_name.set_tag("bar", "xs");
        r1.metric_name.set_tag("foo", "45");

        let mut r2 = make_result([nan, nan, nan, 1723.456, 1923.456, 2123.456]);
        r2.metric_name.set_tag("foo", "123.456");

        test_query(q, vec![r1, r2]);
    }

    fn label_transform__mismatch() {
        let q = r##"label_transform(time(), "#__name__", "foobar", "xx")"##;
        assert_result_eq(q, &[1000, 1200, 1400, 1600, 1800, 2000]);
    }

    #[test]
    fn label_transform__match() {
        let q = r##"label_transform(
        label_set(time(), "#foo", "a.bar.baz"),
        "foo", "\\.", "-")"##;
        let mut r = make_result(&[1000, 1200, 1400, 1600, 1800, 2000]);
        r.metric_name.set_tag("foo", "a-bar-baz");
        let result_expected: Vec<QueryResult> = vec![r];
        test_query(q, vec![r])
    }

    #[test]
    fn label_replace__nonexisting_src() {
        let q = r##"label_replace(time(), "#__name__", "x${1}y", "foo", ".+")"##;
        let mut r = make_result(&[1000, 1200, 1400, 1600, 1800, 2000]);
        test_query(q, vec![r]);
    }

    #[test]
    fn label_replace__mismatch() {
        let q = r##"label_replace(label_set(time(), "#foo", "foobar"), "__name__", "x${1}y", "foo", "bar(.+)")"##;
        let mut r = make_result(&[1000, 1200, 1400, 1600, 1800, 2000]);
        r.metric_name.set_tag("foo", "foobar");
        test_query(q, vec![r])
    }

    #[test]
    fn label_replace__match() {
        let q = r##"label_replace(time(), "#__name__", "x${1}y", "foo", ".*")"##;
        let mut r = make_result(&[1000, 1200, 1400, 1600, 1800, 2000]);
        r.metric_name.set_metric_group("xy");
        test_query(q, vec![r])
    }

    #[test]
    fn label_replace__label_replace() {
        let q = r##"
        label_replace(
        label_replace(
        label_replace(time(), "#__name__", "x${1}y", "foo", ".*"),
        "xxx", "foo${1}bar(${1})", "__name__", "(.+)"),
        "xxx", "AA$1", "xxx", "foox(.+)"
        )"##;
        let mut r = make_result(&[1000, 1200, 1400, 1600, 1800, 2000]);
        r.metric_name.set_metric_group("xy");
        r.metric_name.set_tag("xxx", "AAybar(xy)");
        test_query(q, vec![r]);
    }

    #[test]
    fn label_match() {
        let q = r##"
        label_match((
        alias(time(), "#foo"),
        alias(2*time(), "bar"),
        ), "__name__", "f.+")"##;
        let mut r = make_result(&[1000, 1200, 1400, 1600, 1800, 2000]);
        r.metric_name.set_metric_group("foo");
        test_query(q, vec![r]);
    }

    #[test]
    fn label_mismatch() {
        let q = r##"
        label_mismatch((
        alias(time(), "#foo"),
        alias(2*time(), "bar"),
        ), "__name__", "f.+")"##;
        let mut r = make_result([2000, 2400, 2800, 3200, 3600, 4000]);
        r.metric_name.set_metric_group("bar");
        test_query(q, vec![r]);
    }

    #[test]
    fn label_graphite_group() {
        let q = r##"sort(label_graphite_group((
        alias(1, "#foo.bar.baz"),
        alias(2, "abc"),
        label_set(alias(3, "a.xx.zz.asd"), "qwe", "rty"),
        ), 1, 3))"##;
        let mut r1 = make_result([1, 1, 1, 1, 1, 1]);
        r1.metric_name.set_metric_group("bar.");
        let mut r2 = make_result([2, 2, 2, 2, 2, 2]);
        r2.metric_name.set_metric_group(".");
        let mut r3 = make_result([3, 3, 3, 3, 3, 3]);
        r3.metric_name.set_metric_group("xx.asd");
        r3.metric_name.set_tag("qwe", "rty");
        let result_expected: Vec<QueryResult> = vec![r1, r2, r3];
        test_query(q, vec![r])
    }

    #[test]
    fn limit_offset() {
        let q = r##"limit_offset(1, 1, sort_by_label((
        label_set(time()*1, "#foo", "y"),
        label_set(time()*2, "foo", "a"),
        label_set(time()*3, "foo", "x"),
        ), "foo"))"##;
        let mut r = make_result([3000, 3600, 4200, 4800, 5400, 6000]);
        r.metric_name.set_tag("foo", "x");
        test_query(q, vec![r]);
    }

    #[test]
    fn limit_offset__NaN() {
        // q returns 3 time series, where foo=3 contains only NaN values
        // limit_offset suppose to apply offset for non-NaN series only
        let q = r##"limit_offset(1, 1, sort_by_label_desc((
        label_set(time()*1, "foo", "1"),
        label_set(time()*2, "foo", "2"),
        label_set(time()*3, "foo", "3"),
        ) < 3000, "foo"))"##;
        let mut r = make_result(&[1000, 1200, 1400, 1600, 1800, 2000]);
        r.metric_name.set_tag("foo", "1");
        test_query(q, vec![r]);
    }

    #[test]
    fn sum__label_graphite_group() {
        let q = r##"sort(sum by (__name__) (
        label_graphite_group((
        alias(1, "#foo.bar.baz"),
        alias(2, "x.y.z"),
        alias(3, "qe.bar.qqq"),
        ), 1)
        ))"##;
        let mut r1 = make_result([2, 2, 2, 2, 2, 2]);
        r1.metric_name.set_metric_group("y");
        let mut r2 = make_result([4, 4, 4, 4, 4, 4]);
        r2.metric_name.set_metric_group("bar");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn two_timeseries() {
        let q = r##"sort_desc(time() or label_set(2, "#xx", "foo"))"##;
        let r1 = make_result(&[1000, 1200, 1400, 1600, 1800, 2000]);
        let mut r2 = make_result([2, 2, 2, 2, 2, 2]);
        r2.metric_name.set_tag("xx", "foo");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn test_sgn() {
        assert_result_eq("sgn(time()-1400)",[-1, -1, 0, 1, 1, 1]);
    }

    #[test]
    fn round() {
        assert_result_eq("round(time()/1e3)",[1, 1, 1, 2, 2, 2]);
        assert_result_eq("round(time()/1e3, 0.5)",[1, 1, 1.5, 1.5, 2, 2]);
        assert_result_eq("round(-time()/1e3, 0.5)",[-1, -1, -1.5, -1.5, -2, -2]);
    }

    #[test]
    fn scalar__multi_timeseries() {
        let q = r##"scalar(1 or label_set(2, "#xx", "foo"))"##;
        test_query(q, vec![]);
    }

    #[test]
    fn sort() {
        let q = r##"sort(2 or label_set(1, "#xx", "foo"))"##;
        let mut r1 = make_result([1, 1, 1, 1, 1, 1]);
        r1.metric_name.set_tag("xx", "foo");
        let mut r2 = make_result([2, 2, 2, 2, 2, 2]);
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn sort_desc() {
        let q = r##"sort_desc(1 or label_set(2, "#xx", "foo"))"##;
        let mut r1 = make_result([2, 2, 2, 2, 2, 2]);
        r1.metric_name.set_tag("xx", "foo");
        let mut r2 = make_result([1, 1, 1, 1, 1, 1]);
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn sort_by_label() {
        let q = r##"sort_by_label((
        alias(1, "#foo"),
        alias(2, "bar"),
        ), "__name__")"##;
        let mut r1 = make_result([2, 2, 2, 2, 2, 2]);
        r1.metric_name.set_metric_group("bar");
        let mut r2 = make_result([1, 1, 1, 1, 1, 1]);
        r2.metric_name.set_metric_group("foo");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn sort_by_label_desc() {
        let q = r##"sort_by_label_desc((
        alias(1, "foo"),
        alias(2, "bar"),
        ), "__name__")"##;
        let mut r1 = make_result([1, 1, 1, 1, 1, 1]);
        r1.metric_name.set_metric_group("foo");
        let mut r2 = make_result([2, 2, 2, 2, 2, 2]);
        r2.metric_name.set_metric_group("bar");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn sort_by_label__multiple_labels() {
        let q = r##"sort_by_label((
        label_set(1, "#x", "b", "y", "aa"),
        label_set(2, "x", "a", "y", "aa"),
        ), "y", "x")"##;
        let mut r1 = make_result([2, 2, 2, 2, 2, 2]);
        r1.metric_name.set_tag("x", "a");
        r1.metric_name.set_tag("y", "aa");

        let mut r2 = make_result([1, 1, 1, 1, 1, 1]);
        r2.metric_name.set_tag("x", "b");
        r2.metric_name.set_tag("y", "aa");
        test_query(q, vec![r1, r2])
    }

    #[test]
    fn test_scalar() {
        assert_result_eq("-1 < 2",[-1, -1, -1, -1, -1, -1]);
        assert_result_eq("123 < time()",&[1000, 1200, 1400, 1600, 1800, 2000]);
        assert_result_eq("time() > 1234",[nan, nan, 1400, 1600, 1800, 2000]);
        assert_result_eq("time() >bool 1234",[0, 0, 1, 1, 1, 1]);
        assert_result_eq("(time() > 1234) >bool 1450",[nan, nan, 0, 1, 1, 1]);
        assert_result_eq("(time() > 1234) !=bool 1400",[nan, nan, 0, 1, 1, 1]);
        assert_result_eq("1400 !=bool (time() > 1234)",[nan, nan, 0, 1, 1, 1]);
        let q = "123 > time()";
        test_query(q, vec![]);

        let q = "time() < 123";
        test_query(q, vec![]);

        assert_result_eq("1300 < time() < 1700",[nan, nan, 1400, 1600, nan, nan]);
    }

    #[test]
    fn array_cmp_scalar__leave_metric_group() {
        let q = r##"sort_desc((
        label_set(time(), "#__name__", "foo", "a", "x"),
        label_set(time()+200, "__name__", "bar", "a", "x"),
        ) > 1300)"##;
        let mut r1 = make_result([nan, 1400, 1600, 1800, 2000, 2200]);
        r1.metric_name.set_metric_group("bar");
        r1.metric_name.set_tag("a", "x");
        let mut r2 = make_result([nan, nan, 1400, 1600, 1800, 2000]);
        r2.metric_name.set_metric_group("foo");
        r2.metric_name.set_tag("a", "x");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn arr_cmp_bool_scalar_drop_metric_group() {
        let q = r##"sort_desc((
        label_set(time(), "#__name__", "foo", "a", "x"),
        label_set(time()+200, "__name__", "bar", "a", "y"),
        ) >= bool 1200)"##;
        let mut r1 = make_result([1, 1, 1, 1, 1, 1]);
        r1.metric_name.set_tag("a", "y");
        let mut r2 = make_result([0, 1, 1, 1, 1, 1]);
        r2.metric_name.set_tag("a", "x");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn numeric_greater() {
        let q = "1 > 2";
        test_query(q, vec![])
    }

    #[test]
    fn vector_eq_bool() {
        // vector(1) == bool time()
        assert_result_eq("vector(1) == bool time()",[0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn vector_eq_scalar() {
        assert_result_eq("vector(1) == time()",vec![]);
    }

    #[test]
    fn compare_to_nan_right() {
        assert_result_eq("1 != nan",[1, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn compare_to_nan_left() {
        assert_result_eq("nan != 1", vec![]);
    }

    #[test]
    fn function_cmp_scalar() {
        assert_result_eq("time() >= bool 2",[1, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn test_and() {
        assert_result_eq("time() and 2",&[1000, 1200, 1400, 1600, 1800, 2000]);
        assert_result_eq("time() and time() > 1300",[nan, nan, 1400, 1600, 1800, 2000]);
    }

    #[test]
    fn test_unless() {
        test_query("time() unless 2", vec![]);
        assert_result_eq("time() unless time() > 1500",&[1000, 1200, 1400, nan, nan, nan]);

        // timseries-with-tags unless 2
        let q = r#"label_set(time(), "foo", "bar") unless 2"#;
        let mut r = make_result(&[1000, 1200, 1400, 1600, 1800, 2000]);
        r.metric_name.set_tag("foo", "bar");
        test_query(q, vec![r]);
    }

    #[test]
    fn scalar_or_scalar() {
        assert_result_eq("time() > 1400 or 123",[123, 123, 123, 1600, 1800, 2000]);
    }

    #[test]
    fn scalar_default_scalar() {
        assert_result_eq("time() > 1400 default 123",[123, 123, 123, 1600, 1800, 2000]);
    }

    #[test]
    fn scalar_default_scalar_from_vector() {
        let q = r##"time() > 1400 default scalar(label_set(123, "#foo", "bar"))"##;
        assert_result_eq(q, [123, 123, 123, 1600, 1800, 2000]);
    }

    #[test]
    fn scalar_default_vector1() {
        let q = r##"time() > 1400 default label_set(123, "#foo", "bar")"##;
        assert_result_eq(q, [nan, nan, nan, 1600, 1800, 2000]);
    }

    #[test]
    fn scalar_default_vector2() {
        let q = r##"time() > 1400 default (
        label_set(123, "#foo", "bar"),
        label_set(456, "__name__", "xxx"),
        )"##;
        assert_result_eq(q,[456, 456, 456, 1600, 1800, 2000]);
    }

    #[test]
    fn scalar_default_NaN() {
        let q = "time() > 1400 default (time() < -100)";
        assert_result_eq(q,[nan, nan, nan, 1600, 1800, 2000]);
    }

    #[test]
    fn vector_default_scalar() {
        let q = r##"sort_desc(union(
        label_set(time() > 1400, "#__name__", "x", "foo", "bar"),
        label_set(time() < 1700, "__name__", "y", "foo", "baz")) default 123)"##;
        let mut r1 = make_result([123, 123, 123, 1600, 1800, 2000]);
        r1.metric_name.set_metric_group("x");
        r1.metric_name.set_tag("foo", "bar");
        let mut r2 = make_result(&[1000, 1200, 1400, 1600, 123, 123]);
        r2.metric_name.set_metric_group("y");
        r2.metric_name.set_tag("foo", "baz");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn vector_divided_by_scalar() {
        let q = r##"sort_desc((label_set(time(), "#foo", "bar") or label_set(10, "foo", "qwert")) / 2)"##;
        let mut r1 = make_result([500, 600, 700, 800, 900, 1000]);
        r1.metric_name.set_tag("foo", "bar");
        let mut r2 = make_result([5, 5, 5, 5, 5, 5]);
        r2.metric_name.set_tag("foo", "qwert");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn vector_multiplied_by_scalar() {
        assert_result_eq("sum(time()) * 2",[2000, 2400, 2800, 3200, 3600, 4000]);
    }

    #[test]
    fn scalar_multiplied_by_vector() {
        let q = r##"sort_desc(2 * (label_set(time(), "#foo", "bar") or label_set(10, "foo", "qwert")))"##;
        let mut r1 = make_result([2000, 2400, 2800, 3200, 3600, 4000]);
        r1.metric_name.set_tag("foo", "bar");
        let mut r2 = make_result([20, 20, 20, 20, 20, 20]);
        r2.metric_name.set_tag("foo", "qwert");
        let result_expected: Vec<QueryResult> = vec![r1, r2];
        test_query(q, vec![r])
    }

    #[test]
    fn scalar_on_group_right__vector() {
        // scalar * on() group_right vector
        let q = r##"sort_desc(2 * on() group_right() (label_set(time(), "#foo", "bar") or label_set(10, "foo", "qwert")))"##;
        let mut r1 = make_result([2000, 2400, 2800, 3200, 3600, 4000]);
        r1.metric_name.set_tag("foo", "bar");
        let mut r2 = make_result([20, 20, 20, 20, 20, 20]);
        r2.metric_name.set_tag("foo", "qwert");
        let result_expected: Vec<QueryResult> = vec![r1, r2];
        test_query(q, vec![r])
    }

    #[test]
    fn scalar_multiply_by_ignoring__foo__group_right_vector() {
        let q = r##"sort_desc(label_set(2, "#a", "2") * ignoring(foo,a) group_right(a) (label_set(time(), "foo", "bar", "a", "1"), label_set(10, "foo", "qwert")))"##;
        let mut r1 = make_result([2000, 2400, 2800, 3200, 3600, 4000]);
        r1.metric_name.set_tag("a", "2");
        r1.metric_name.set_tag("foo", "bar");

        let mut r2 = make_result([20, 20, 20, 20, 20, 20]);
        r2.metric_name.set_tag("a", "2");
        r2.metric_name.set_tag("foo", "qwert");

        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn scalar_multiply_ignoring_vector() {
        let q = r##"sort_desc(label_set(2, "#foo", "bar") * ignoring(a) (label_set(time(), "foo", "bar") or label_set(10, "foo", "qwert")))"##;
        let mut r = make_result([2000, 2400, 2800, 3200, 3600, 4000]);
        r.metric_name.set_tag("foo", "bar");
        test_query(q, vec![r]);
    }

    #[test]
    fn scalar_multiply_by_on_foo_vector() {
        //"scalar * on(foo) vector"
        let q = r##"sort_desc(label_set(2, "#foo", "bar", "aa", "bb") * on(foo) (label_set(time(), "foo", "bar", "xx", "yy") or label_set(10, "foo", "qwert")))"##;
        let mut r = make_result([2000, 2400, 2800, 3200, 3600, 4000]);
        r.metric_name.set_tag("foo", "bar");
        test_query(q, vec![r]);
    }

    #[test]
    fn vector_multiply_by_on__foo__scalar() {
        let q = r#"sort_desc((label_set(time(), "foo", "bar", "xx", "yy"), label_set(10, "foo", "qwert")) * on(foo) label_set(2, "foo","bar","aa","bb"))"#;
        let mut r = make_result([2000, 2400, 2800, 3200, 3600, 4000]);
        r.metric_name.set_tag("foo", "bar");
        test_query(q, vec![r]);
    }

    #[test]
    fn vector_multiply_by_on__foo__group_left() {
        let q = r##"sort(label_set(time()/10, "#foo", "bar", "xx", "yy", "__name__", "qwert") + on(foo) group_left(op) (
        label_set(time() < 1400, "foo", "bar", "op", "le"),
        label_set(time() >= 1400, "foo", "bar", "op", "ge"),
        ))"##;
        let mut r1 = make_result([1100, 1320, nan, nan, nan, nan]);
        r1.metric_name.set_tag("foo", "bar");
        r1.metric_name.set_tag("op", "le");
        r1.metric_name.set_tag("xx", "yy");

        let mut r2 = make_result([nan, nan, 1540, 1760, 1980, 2200]);
        r2.metric_name.set_tag("foo", "bar");
        r2.metric_name.set_tag("op", "ge");
        r2.metric_name.set_tag("xx", "yy");
        test_query(q, vec![r1, r2]);
    }


    #[test]
    fn vector_multiplied_by_on__foo__duplicate_nonoverlapping_timeseries() {
        let q = r##"label_set(time()/10, "#foo", "bar", "xx", "yy", "__name__", "qwert") + on(foo) (
        label_set(time() < 1400, "foo", "bar", "op", "le"),
        label_set(time() >= 1400, "foo", "bar", "op", "ge"),
        )"##;
        let mut r1 = make_result(&[1100, 1320, 1540, 1760, 1980, 2200]);
        r1.metric_name.set_tag("foo", "bar");
        test_query(q, vec![r1]);
    }

    #[test]
    fn vector_multiply_by_on__foo__group_left_duplicate_nonoverlapping_timeseries() {
        let q = r##"label_set(time()/10, "#foo", "bar", "xx", "yy", "__name__", "qwert") + on(foo) group_left() (
        label_set(time() < 1400, "foo", "bar", "op", "le"),
        label_set(time() >= 1400, "foo", "bar", "op", "ge"),
        )"##;
        let mut r1 = make_result([1100, 1320, 1540, 1760, 1980, 2200]);
        r1.metric_name.set_tag("foo", "bar");
        r1.metric_name.set_tag("xx", "yy");

        test_query(q, vec![r1]);
    }

    #[test]
    fn vector_multiplied_by_on__foo__group_left__name__() {
        let q = r##"label_set(time()/10, "#foo", "bar", "xx", "yy", "__name__", "qwert") + on(foo) group_left(__name__)
        label_set(time(), "foo", "bar", "__name__", "aaa")"##;
        let mut r1 = make_result([1100, 1320, 1540, 1760, 1980, 2200]);
        r1.metric_name.set_metric_group("aaa");
        r1.metric_name.set_tag("foo", "bar");
        r1.metric_name.set_tag("xx", "yy");

        test_query(q, vec![r1]);
    }

    #[test]
    fn vector_multiplied_by_on__foo__group_right() {
        let q = r##"sort(label_set(time()/10, "#foo", "bar", "xx", "yy", "__name__", "qwert") + on(foo) group_right(xx) (
        label_set(time(), "foo", "bar", "__name__", "aaa"),
        label_set(time()+3, "foo", "bar", "__name__", "yyy","ppp", "123"),
        ))"##;
        let mut r1 = make_result([1100, 1320, 1540, 1760, 1980, 2200]);
        r1.metric_name.set_tag("foo", "bar");
        r1.metric_name.set_tag("xx", "yy");

        let mut r2 = make_result([1103, 1323, 1543, 1763, 1983, 2203]);
        r2.metric_name.set_tag("foo", "bar");
        r2.metric_name.set_tag("ppp", "123");
        r2.metric_name.set_tag("xx", "yy");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn vector_multiply_by_on_group_left_scalar() {
        let q = r##"sort_desc((label_set(time(), "#foo", "bar") or label_set(10, "foo", "qwert")) * on() group_left 2)"##;
        let mut r1 = make_result([2000, 2400, 2800, 3200, 3600, 4000]);
        r1.metric_name.set_tag("foo", "bar");
        let mut r2 = make_result([20, 20, 20, 20, 20, 20]);
        r2.metric_name.set_tag("foo", "qwert");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn vector_plus_vector_matching() {
        let q = r##"sort_desc(
        (label_set(time(), "#t1", "v1") or label_set(10, "t2", "v2"))
        +
        (label_set(100, "t1", "v1") or label_set(time(), "t2", "v2"))
        )"##;
        let mut r1 = make_result([1100, 1300, 1500, 1700, 1900, 2100]);
        r1.metric_name.set_tag("t1", "v1");
        let mut r2 = make_result([1010, 1210, 1410, 1610, 1810, 2010]);
        r2.metric_name.set_tag("t2", "v2");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn vector_vector_partial_matching() {
        let q = r##"sort_desc(
        (label_set(time(), "#t1", "v1") or label_set(10, "t2", "v2"))
        +
        (label_set(100, "t1", "v1") or label_set(time(), "t2", "v3"))
        )"##;
        let mut r = make_result([1100, 1300, 1500, 1700, 1900, 2100]);
        r.metric_name.set_tag("t1", "v1");
        test_query(q, vec![r])
    }

    #[test]
    fn vector_plus_vector_no_matching() {
        let q = r##"sort_desc(
        (label_set(time(), "#t2", "v1") or label_set(10, "t2", "v2"))
        +
        (label_set(100, "t1", "v1") or label_set(time(), "t2", "v3"))
        )"##;
        test_query(q, vec![]);
    }

    #[test]
    fn vector_plus_vector_on_matching() {
        let q = r##"sort_desc(
        (label_set(time(), "#t1", "v123", "t2", "v3") or label_set(10, "t2", "v2"))
        + on (foo, t2)
        (label_set(100, "t1", "v1") or label_set(time(), "t2", "v3"))
        )"##;
        let mut r = make_result([2000, 2400, 2800, 3200, 3600, 4000]);
        r.metric_name.set_tag("t2", "v3");

        test_query(q, vec![r])
    }

    #[test]
    fn vector_plus_vector_on_group_left_matching() {
        let q = r##"sort_desc(
        (label_set(time(), "#t1", "v123", "t2", "v3"), label_set(10, "t2", "v3", "xxx", "yy"))
        + on (foo, t2) group_left (t1, noxxx)
        (label_set(100, "t1", "v1"), label_set(time(), "t2", "v3", "noxxx", "aa"))
        )"##;
        let mut r1 = make_result([2000, 2400, 2800, 3200, 3600, 4000]);
        r1.metric_name.set_tag("noxxx", "aa");
        r1.metric_name.set_tag("t2", "v3");

        let mut r2 = make_result([1010, 1210, 1410, 1610, 1810, 2010]);
        r2.metric_name.set_tag("noxxx", "aa");
        r2.metric_name.set_tag("t2", "v3");
        r2.metric_name.set_tag("xxx", "yy");

        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn vector_plus_vector_on_group_left___name__() {
        let q = r##"sort_desc(
        (union(label_set(time(), "#t2", "v3", "__name__", "vv3", "x", "y"), label_set(10, "t2", "v3", "__name__", "yy")))
        + on (t2, dfdf) group_left (__name__, xxx)
        (label_set(100, "t1", "v1") or label_set(time(), "t2", "v3", "__name__", "abc"))
        )"##;
        let mut r1 = make_result([2000, 2400, 2800, 3200, 3600, 4000]);
        r1.metric_name.set_metric_group("abc");
        r1.metric_name.set_tag("t2", "v3");
        r1.metric_name.set_tag("x", "y");

        let mut r2 = make_result([1010, 1210, 1410, 1610, 1810, 2010]);
        r2.metric_name.set_metric_group("abc");
        r2.metric_name.set_tag("t2", "v3");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn vector_plus_vector_ignoring_matching() {
        let q = r##"sort_desc(
        (label_set(time(), "#t1", "v123", "t2", "v3") or label_set(10, "t2", "v2"))
        + ignoring (foo, t1, bar)
        (label_set(100, "t1", "v1") or label_set(time(), "t2", "v3"))
        )"##;
        let mut r = make_result([2000, 2400, 2800, 3200, 3600, 4000]);
        r.metric_name.set_tag("t2", "v3");

        test_query(q, vec![r]);
    }

    #[test]
    fn vector_plus_vector_ignoring_group_right_matching() {
        let q = r#"sort_desc(
        (label_set(time(), "t1", "v123", "t2", "v3") or label_set(10, "t2", "v321", "t1", "v123", "t32", "v32"))
        + ignoring (foo, t2) group_right ()
        (label_set(100, "t1", "v123") or label_set(time(), "t1", "v123", "t2", "v3"))
        )"#;
        let mut r1 = make_result([2000, 2400, 2800, 3200, 3600, 4000]);
        r1.metric_name.set_tag("t1", "v123");
        r1.metric_name.set_tag("t2", "v3");

        let mut r2 = make_result([1100, 1300, 1500, 1700, 1900, 2100]);
        r2.metric_name.set_tag("t1", "v123");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn histogram_quantile__scalar() {
        let q = "histogram_quantile(0.6, time())";
        test_query(q, vec![]);
    }

    #[test]
    fn histogram_share__scalar() {
        let q = "histogram_share(123, time())";
        test_query(q, vec![]);
    }

    #[test]
    fn histogram_quantile__single_value_no_le() {
        let q = r##"histogram_quantile(0.6, label_set(100, "#foo", "bar"))"##;
        test_query(q, vec![]);
    }

    #[test]
    fn histogram_share__single_value_no_le() {
        let q = r##"histogram_share(123, label_set(100, "#foo", "bar"))"##;
        test_query(q, vec![]);
    }

    #[test]
    fn histogram_quantile__single_value_invalid_le() {
        let q = r#"histogram_quantile(0.6, label_set(100, "le", "foobar"))"#;
        test_query(q, vec![]);
    }

    #[test]
    fn histogram_share__single_value_invalid_le() {
        let q = r##"histogram_share(50, label_set(100, "#le", "foobar"))"##;
        test_query(q, vec![]);
    }

    #[test]
    fn histogram_quantile__single_value_inf_le() {
        let q = r##"histogram_quantile(0.6, label_set(100, "#le", "+Inf"))"##;
        test_query(q, vec![]);

        let q = r##"histogram_quantile(0.6, label_set(100, "#le", "200"))"##;
        let r = make_result([120, 120, 120, 120, 120, 120]);
        test_query(q, vec![r]);
    }

    #[test]
    fn histogram_quantile__zero_value_inf_le() {
        let q = r##"histogram_quantile(0.6, (
        label_set(100, "#le", "+Inf"),
        label_set(0, "le", "42"),
        ))"##;
        assert_result_eq(q,[42, 42, 42, 42, 42, 42]);
    }

    #[test]
    fn stdvar_over_time() {
        assert_result_eq("round(stdvar_over_time(rand(0)[200s:5s]), 0.001)",
                         [0.082, 0.088, 0.092, 0.075, 0.101, 0.08]);
    }

    #[test]
    fn histogram_stdvar() {
        let q = "round(histogram_stdvar(histogram_over_time(rand(0)[200s:5s])), 0.001)";
        assert_result_eq(q,[0.079, 0.089, 0.089, 0.071, 0.1, 0.082]);
    }

    #[test]
    fn stddev_over_time() {
        let q = "round(stddev_over_time(rand(0)[200s:5s]), 0.001)";
        assert_result_eq(q,[0.286, 0.297, 0.303, 0.274, 0.318, 0.283]);
    }

    #[test]
    fn histogram_stddev() {
        let q = "round(histogram_stddev(histogram_over_time(rand(0)[200s:5s])), 0.001)";
        assert_result_eq(q, [0.281, 0.299, 0.298, 0.267, 0.316, 0.286]);
    }

    #[test]
    fn avg_over_time() {
        let q = "round(avg_over_time(rand(0)[200s:5s]), 0.001)";
        assert_result_eq(q, [0.521, 0.518, 0.509, 0.544, 0.511, 0.504]);
    }

    #[test]
    fn histogram_avg() {
        let q = "round(histogram_avg(histogram_over_time(rand(0)[200s:5s])), 0.001)";
        assert_result_eq(q,[0.519, 0.521, 0.503, 0.543, 0.511, 0.506]);
    }

    #[test]
    fn histogram_share__single_value_valid_le() {
        let q = r##"histogram_share(300, label_set(100, "#le", "200"))"##;
        assert_result_eq(q,[1, 1, 1, 1, 1, 1]);

        let q = r##"histogram_share(80, label_set(100, "#le", "200"))"##;
        assert_result_eq(q, [0.4, 0.4, 0.4, 0.4, 0.4, 0.4]);

        let q = r##"histogram_share(200, label_set(100, "#le", "200"))"##;
        assert_result_eq(q, [1, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn histogram_quantile__single_value_valid_le__boundsLabel() {
        let q = r#"sort(histogram_quantile(0.6, label_set(100, "le", "200"), "foobar"))"#;
        let mut r1 = make_result([0, 0, 0, 0, 0, 0]);
        r1.metric_name.set_tag("foobar", "lower");
        let r2 = make_result([120, 120, 120, 120, 120, 120]);
        let mut r3 = make_result([200, 200, 200, 200, 200, 200]);
        r3.metric_name.set_tag("foobar", "upper");
        test_query(q, vec![r1, r2, r3]);
    }

    #[test]
    fn histogram_share__single_value_valid_le__boundsLabel() {
        let q = r##"sort(histogram_share(120, label_set(100, "#le", "200"), "foobar"))"##;
        let mut r1 = make_result([0, 0, 0, 0, 0, 0]);
        r1.metric_name.set_tag("foobar", "lower");
        let mut r2 = make_result([0.6, 0.6, 0.6, 0.6, 0.6, 0.6]);
        let mut r3 = make_result([1, 1, 1, 1, 1, 1]);
        r3.metric_name.set_tag("foobar", "upper");
        test_query(q, vec![r1, r2, r3]);
    }

    #[test]
    fn histogram_quantile__single_value_valid_le_max_phi() {
        let q = r##"histogram_quantile(1, (
        label_set(100, "#le", "200"),
        label_set(0, "le", "55"),
        ))"##;
        assert_result_eq(q, [200, 200, 200, 200, 200, 200]);
    }

    #[test]
    fn histogram_quantile__single_value_valid_le_max_le() {
        let q = r##"histogram_share(200, (
        label_set(100, "#le", "200"),
        label_set(0, "le", "55"),
        ))"##;
        assert_result_eq(q,[1, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn histogram_quantile__single_value_valid_le_min_phi() {
        let q = r##"histogram_quantile(0, (
        label_set(100, "#le", "200"),
        label_set(0, "le", "55"),
        ))"##;
        assert_result_eq(q, [55, 55, 55, 55, 55, 55]);
    }

    #[test]
    fn histogram_share__single_value_valid_le_min_le() {
        let q = r##"histogram_share(0, (
        label_set(100, "#le", "200"),
        label_set(0, "le", "55"),
        ))"##;
        assert_result_eq(q, [0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn histogram_share__single_value_valid_le_low_le() {
        let q = r##"histogram_share(55, (
        label_set(100, "#le", "200"),
        label_set(0, "le", "55"),
        ))"##;
        assert_result_eq(q, [0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn histogram_share__single_value_valid_le_mid_le() {
        let q = r##"histogram_share(105, (
        label_set(100, "#le", "200"),
        label_set(0, "le", "55"),
        ))"##;
        assert_result_eq(q, [0.3448275862068966, 0.3448275862068966, 0.3448275862068966, 0.3448275862068966, 0.3448275862068966, 0.3448275862068966]);
    }

    #[test]
    fn histogram_quantile__single_value_valid_le_min_phi_no_zero_bucket() {
        let q = r##"histogram_quantile(0, label_set(100, "#le", "200"))"##;
        assert_result_eq(q, &[0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn histogram_quantile__scalar_phi() {
        let q = r##"histogram_quantile(time() / 2 / 1e3, label_set(100, "#le", "200"))"##;
        assert_result_eq(q, [100, 120, 140, 160, 180, 200]);
    }

    #[test]
    fn histogram_share__scalar_phi() {
        let q = r##"histogram_share(time() / 8, label_set(100, "#le", "200"))"##;
        assert_result_eq(q, [0.625, 0.75, 0.875, 1, 1, 1]);
    }

    #[test]
    fn histogram_quantile__valid() {
        let q = r##"sort(histogram_quantile(0.6,
        label_set(90, "#foo", "bar", "le", "10")
        or label_set(100, "foo", "bar", "le", "30")
        or label_set(300, "foo", "bar", "le", "+Inf")
        or label_set(200, "tag", "xx", "le", "10")
        or label_set(300, "tag", "xx", "le", "30")
        ))"##;
        let mut r1 = make_result([9, 9, 9, 9, 9, 9]);
        r1.metric_name.set_tag("tag", "xx");
        let mut r2 = make_result([30, 30, 30, 30, 30, 30]);
        r2.metric_name.set_tag("foo", "bar");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn histogram_share__valid() {
        let q = r##"sort(histogram_share(25,
        label_set(90, "#foo", "bar", "le", "10")
        or label_set(100, "foo", "bar", "le", "30")
        or label_set(300, "foo", "bar", "le", "+Inf")
        or label_set(200, "tag", "xx", "le", "10")
        or label_set(300, "tag", "xx", "le", "30")
        ))"##;
        let mut r1 = make_result([0.325, 0.325, 0.325, 0.325, 0.325, 0.325]);
        r1.metric_name.set_tag("foo", "bar");
        let mut r2 = make_result([0.9166666666666666, 0.9166666666666666, 0.9166666666666666, 0.9166666666666666, 0.9166666666666666, 0.9166666666666666]);
        r2.metric_name.set_tag("tag", "xx");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn histogram_quantile__negative_bucket_count() {
        let q = r##"histogram_quantile(0.6,
        label_set(90, "#foo", "bar", "le", "10")
        or label_set(-100, "foo", "bar", "le", "30")
        or label_set(300, "foo", "bar", "le", "+Inf")
        )"##;
        let mut r = make_result([30, 30, 30, 30, 30, 30]);
        r.metric_name.set_tag("foo", "bar");
        test_query(q, vec![r]);
    }

    #[test]
    fn histogram_quantile__nan_bucket_count_some() {
        let q = r##"round(histogram_quantile(0.6,
        label_set(90, "#foo", "bar", "le", "10")
        or label_set(NaN, "foo", "bar", "le", "30")
        or label_set(300, "foo", "bar", "le", "+Inf")
        ),0.01)"##;
        let mut r = make_result([18.57, 18.57, 18.57, 18.57, 18.57, 18.57]);
        r.metric_name.set_tag("foo", "bar");
        test_query(q, vec![r]);
    }

    #[test]
    fn histogram_quantile__normal_bucket_count() {
        let q = r##"histogram_quantile(0.2,
        label_set(0, "#foo", "bar", "le", "10")
        or label_set(100, "foo", "bar", "le", "30")
        or label_set(300, "foo", "bar", "le", "+Inf")
        )"##;
        let mut r = make_result([22, 22, 22, 22, 22, 22]);
        r.metric_name.set_tag("foo", "bar");
        test_query(q, vec![r]);
    }

    #[test]
    fn histogram_quantiles() {
        let q = r##"sort_by_label(histogram_quantiles("#phi", 0.2, 0.3,
        label_set(0, "foo", "bar", "le", "10")
        or label_set(100, "foo", "bar", "le", "30")
        or label_set(300, "foo", "bar", "le", "+Inf")
        ), "phi")"##;
        let mut r1 = make_result([22, 22, 22, 22, 22, 22]);
        r1.metric_name.set_tag("foo", "bar");
        r1.metric_name.set_tag("phi", "0.2");

        let mut r2 = make_result([28, 28, 28, 28, 28, 28]);
        r2.metric_name.set_tag("foo", "bar");
        r2.metric_name.set_tag("phi", "0.3");

        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn histogram_share__normal_bucket_count() {
        let q = r##"histogram_share(35,
        label_set(0, "#foo", "bar", "le", "10")
        or label_set(100, "foo", "bar", "le", "30")
        or label_set(300, "foo", "bar", "le", "+Inf")
        )"##;
        let mut r = make_result([0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333]);
        r.metric_name.set_tag("foo", "bar");
        test_query(q, vec![r]);
    }

    #[test]
    fn histogram_quantile__normal_bucket_count_boundsLabel() {
        let q = r##"sort(histogram_quantile(0.2,
        label_set(0, "#foo", "bar", "le", "10")
        or label_set(100, "foo", "bar", "le", "30")
        or label_set(300, "foo", "bar", "le", "+Inf"),
        "xxx"
        ))"##;
        let mut r1 = make_result([10, 10, 10, 10, 10, 10]);
        r1.metric_name.set_tag("foo", "bar");
        r1.metric_name.set_tag("xxx", "lower");

        let mut r2 = make_result([22, 22, 22, 22, 22, 22]);
        r2.metric_name.set_tag("foo", "bar");
        let mut r3 = make_result([30, 30, 30, 30, 30, 30]);
        r3.metric_name.set_tag("foo", "bar");
        r2.metric_name.set_tag("xxx", "upper");

        test_query(q, vec![r1, r2, r3]);
    }

    #[test]
    fn histogram_share__normal_bucket_count_boundsLabel() {
        let q = r##"sort(histogram_share(22,
        label_set(0, "#foo", "bar", "le", "10")
        or label_set(100, "foo", "bar", "le", "30")
        or label_set(300, "foo", "bar", "le", "+Inf"),
        "xxx"
        ))"##;
        let mut r1 = make_result(&[0, 0, 0, 0, 0, 0]);
        r1.metric_name.set_tag("foo", "bar");
        r1.metric_name.set_tag("xxx", "lower");

        let mut r2 = make_result([0.2, 0.2, 0.2, 0.2, 0.2, 0.2]);
        r2.metric_name.set_tag("foo", "bar");
        let mut r3 = make_result([0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333]);
        r3.metric_name.set_tag("foo", "bar");
        r3.metric_name.set_tag("xxx", "upper");
        test_query(q, vec![r1, r2, r3]);
    }

    #[test]
    fn histogram_quantile__zero_bucket_count() {
        let q = r##"histogram_quantile(0.6,
        label_set(0, "#foo", "bar", "le", "10")
        or label_set(0, "foo", "bar", "le", "30")
        or label_set(0, "foo", "bar", "le", "+Inf")
        )"##;
        test_query(q, vec![]);
    }

    #[test]
    fn histogram_quantile__nan_bucket_count_all() {
        let q = r##"histogram_quantile(0.6,
        label_set(nan, "#foo", "bar", "le", "10")
        or label_set(nan, "foo", "bar", "le", "30")
        or label_set(nan, "foo", "bar", "le", "+Inf")
        )"##;
        test_query(q, vec![]);
    }

    #[test]
    fn buckets_limit__zero() {
        let q = r##"buckets_limit(0, (
        alias(label_set(100, "#le", "inf", "x", "y"), "metric"),
        alias(label_set(50, "le", "120", "x", "y"), "metric"),
        ))"##;
        test_query(q, vec![]);
    }

    #[test]
    fn buckets_limit__unused() {
        let q = r##"sort(buckets_limit(5, (
        alias(label_set(100, "#le", "inf", "x", "y"), "metric"),
        alias(label_set(50, "le", "120", "x", "y"), "metric"),
        )))"##;

        let mut r1 = make_result([50, 50, 50, 50, 50, 50]);
        r1.metric_name.set_metric_group("metric");
        r1.metric_name.set_tag("le", "120");
        r1.metric_name.set_tag("x", "y");

        let mut r2 = make_result([100, 100, 100, 100, 100, 100]);
        r2.metric_name.set_metric_group("metric");
        r2.metric_name.set_tag("le", "inf");
        r2.metric_name.set_tag("x", "y");

        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn buckets_limit__used() {
        let q = r##"sort(buckets_limit(2, (
        alias(label_set(100, "#le", "inf", "x", "y"), "metric"),
        alias(label_set(98, "le", "300", "x", "y"), "metric"),
        alias(label_set(52, "le", "200", "x", "y"), "metric"),
        alias(label_set(50, "le", "120", "x", "y"), "metric"),
        alias(label_set(20, "le", "70", "x", "y"), "metric"),
        alias(label_set(10, "le", "30", "x", "y"), "metric"),
        alias(label_set(9, "le", "10", "x", "y"), "metric"),
        )))"##;
        let mut r1 = make_result([9, 9, 9, 9, 9, 9]);
        r1.metric_name.set_metric_group("metric");
        r1.metric_name.set_tag("le", "10");
        r1.metric_name.set_tag("x", "y");

        let mut r2 = make_result([98, 98, 98, 98, 98, 98]);
        r2.metric_name.set_metric_group("metric");
        r2.metric_name.set_tag("le", "300");
        r2.metric_name.set_tag("x", "y");

        let mut r3 = make_result([100, 100, 100, 100, 100, 100]);
        r3.metric_name.set_metric_group("metric");
        r3.metric_name.set_tag("le", "inf");
        r3.metric_name.set_tag("x", "y");

        test_query(q, vec![r1, r2, r3]);
    }

    #[test]
    fn prometheus_buckets__missing_vmrange() {
        let q = r##"sort(prometheus_buckets((
        alias(label_set(time()/20, "#foo", "bar", "le", "0.2"), "xyz"),
        alias(label_set(time()/100, "foo", "bar", "vmrange", "foobar"), "xxx"),
        alias(label_set(time()/100, "foo", "bar", "vmrange", "30...foobar"), "xxx"),
        alias(label_set(time()/100, "foo", "bar", "vmrange", "30...40"), "xxx"),
        alias(label_set(time()/80, "foo", "bar", "vmrange", "0...900", "le", "54"), "yyy"),
        alias(label_set(time()/40, "foo", "bar", "vmrange", "900...+Inf", "le", "2343"), "yyy"),
        )))"##;
        let mut r1 = make_result(&[0, 0, 0, 0, 0, 0]);
        r1.metric_name.set_metric_group("xxx");
        r1.metric_name.set_tag("foo", "bar");
        r1.metric_name.set_tag("le", "30");

        let mut r2 = make_result([10, 12, 14, 16, 18, 20]);
        r2.metric_name.set_metric_group("xxx");
        r2.metric_name.set_tag("foo", "bar");
        r2.metric_name.set_tag("le", "40");

        let mut r3 = make_result([10, 12, 14, 16, 18, 20]);
        r3.metric_name.set_metric_group("xxx");
        r3.metric_name.set_tag("foo", "bar");
        r3.metric_name.set_tag("le", "+Inf");

        let mut r4= make_result([12.5, 15, 17.5, 20, 22.5, 25]);
        r4.metric_name.set_metric_group("yyy");
        r4.metric_name.set_tag("foo", "bar");
        r4.metric_name.set_tag("le", "900");

        let mut r5 = make_result([37.5, 45, 52.5, 60, 67.5, 75]);
        r5.metric_name.set_metric_group("yyy");
        r5.metric_name.set_tag("foo", "bar");
        r5.metric_name.set_tag("le", "+Inf");

        let mut r6 = make_result([50, 60, 70, 80, 90, 100]);
        r6.metric_name.set_metric_group("xyz");
        r6.metric_name.set_tag("foo", "bar");
        r6.metric_name.set_tag("le", "0.2");

        let result_expected: Vec<QueryResult> = vec![r1, r2, r3, r4, r5, r6];
        test_query(q, result_expected)
    }

    #[test]
    fn prometheus_buckets__zero_vmrange_value() {
        let q = r##"sort(prometheus_buckets(label_set(0, "#vmrange", "0...0")))"##;
        test_query(q, vec![])
    }

    #[test]
    fn prometheus_buckets__valid() {
        let q = r##"sort(prometheus_buckets((
        alias(label_set(90, "#foo", "bar", "vmrange", "0...0"), "xxx"),
        alias(label_set(time()/20, "foo", "bar", "vmrange", "0...0.2"), "xxx"),
        alias(label_set(time()/100, "foo", "bar", "vmrange", "0.2...40"), "xxx"),
        alias(label_set(time()/10, "foo", "bar", "vmrange", "40...Inf"), "xxx"),
        )))"##;
        let mut r1 = make_result([90, 90, 90, 90, 90, 90]);
        r1.metric_name.set_metric_group("xxx");
        r1.metric_name.set_tag("foo", "bar");
        r1.metric_name.set_tag("le", "0");

        let mut r2 = make_result([140, 150, 160, 170, 180, 190]);
        r2.metric_name.set_metric_group("xxx");
        r2.metric_name.set_tag("foo", "bar");
        r2.metric_name.set_tag("le", "0.2");

        let mut r3 = make_result([150, 162, 174, 186, 198, 210]);
        r3.metric_name.set_metric_group("xxx");
        r3.metric_name.set_tag("foo", "bar");
        r3.metric_name.set_tag("le", "40");

        let mut r4 = make_result([250, 282, 314, 346, 378, 410]);
        r4.metric_name.set_metric_group("xxx");
        r4.metric_name.set_tag("foo", "bar");
        r4.metric_name.set_tag("le", "Inf");

        let result_expected: Vec<QueryResult> = vec![r1, r2, r3, r4];
        test_query(q, result_expected);
    }

    #[test]
    fn prometheus_buckets__overlapped_ranges() {
        let q = r##"sort(prometheus_buckets((
        alias(label_set(90, "#foo", "bar", "vmrange", "0...0"), "xxx"),
        alias(label_set(time()/20, "foo", "bar", "vmrange", "0...0.2"), "xxx"),
        alias(label_set(time()/20, "foo", "bar", "vmrange", "0.2...0.25"), "xxx"),
        alias(label_set(time()/20, "foo", "bar", "vmrange", "0...0.26"), "xxx"),
        alias(label_set(time()/100, "foo", "bar", "vmrange", "0.2...40"), "xxx"),
        alias(label_set(time()/10, "foo", "bar", "vmrange", "40...Inf"), "xxx"),
        )))"##;
        let mut r1 = make_result([90, 90, 90, 90, 90, 90]);
        r1.metric_name.set_metric_group("xxx");
        r1.metric_name.set_tag("foo", "bar");
        r1.metric_name.set_tag("le", "0");

        let mut r2 = make_result([140, 150, 160, 170, 180, 190]);
        r2.metric_name.set_metric_group("xxx");
        r2.metric_name.set_tag("foo", "bar");
        r2.metric_name.set_tag("le", "0.2");

        let mut r3 = make_result([190, 210, 230, 250, 270, 290]);
        r3.metric_name.set_metric_group("xxx");
        r3.metric_name.set_tag("foo", "bar");
        r3.metric_name.set_tag("le", "0.25");

        let mut r4 = make_result([240, 270, 300, 330, 360, 390]);
        r4.metric_name.set_metric_group("xxx");
        r4.metric_name.set_tag("foo", "bar");
        r4.metric_name.set_tag("le", "0.26");

        let mut r5 = make_result([250, 282, 314, 346, 378, 410]);
        r5.metric_name.set_metric_group("xxx");
        r5.metric_name.set_tag("foo", "bar");
        r5.metric_name.set_tag("le", "40");

        let mut r6 = make_result([350, 402, 454, 506, 558, 610]);
        r6.metric_name.set_metric_group("xxx");
        r6.metric_name.set_tag("foo", "bar");
        r6.metric_name.set_tag("le", "Inf");

        let result_expected: Vec<QueryResult> = vec![r1, r2, r3, r4, r5, r6];
        test_query(q, result_expected)
    }

    #[test]
    fn prometheus_buckets__overlapped_ranges_at_the_end() {
        let q = r##"sort(prometheus_buckets((
        alias(label_set(90, "#foo", "bar", "vmrange", "0...0"), "xxx"),
        alias(label_set(time()/20, "foo", "bar", "vmrange", "0...0.2"), "xxx"),
        alias(label_set(time()/20, "foo", "bar", "vmrange", "0.2...0.25"), "xxx"),
        alias(label_set(time()/20, "foo", "bar", "vmrange", "0...0.25"), "xxx"),
        alias(label_set(time()/100, "foo", "bar", "vmrange", "0.2...40"), "xxx"),
        alias(label_set(time()/10, "foo", "bar", "vmrange", "40...Inf"), "xxx"),
        )))"##;
        let mut r1 = make_result([90, 90, 90, 90, 90, 90]);
        r1.metric_name.set_metric_group("xxx");
        r1.metric_name.set_tag("foo", "bar");
        r1.metric_name.set_tag("le", "0");

        let mut r2 = make_result([140, 150, 160, 170, 180, 190]);
        r2.metric_name.set_metric_group("xxx");
        r2.metric_name.set_tag("foo", "bar");
        r2.metric_name.set_tag("le", "0.2");

        let mut r3 = make_result([190, 210, 230, 250, 270, 290]);
        r3.metric_name.set_metric_group("xxx");
        r3.metric_name.set_tag("foo", "bar");
        r3.metric_name.set_tag("le", "0.25");

        let mut r4 = make_result([200, 222, 244, 266, 288, 310]);
        r4.metric_name.set_metric_group("xxx");
        r4.metric_name.set_tag("foo", "bar");
        r4.metric_name.set_tag("le", "40");

        let mut r5 = make_result([300, 342, 384, 426, 468, 510]);
        r5.metric_name.set_metric_group("xxx");
        r5.metric_name.set_tag("foo", "bar");
        r5.metric_name.set_tag("le", "Inf");

        let result_expected: Vec<QueryResult> = vec![r1, r2, r3, r4, r5];
        test_query(q, result_expected)
    }

    #[test]
    fn median_over_time() {
        let q = "median_over_time({})";
        test_query(q, vec![]);

        assert_result_eq(r##"median_over_time("#foo")"##, vec![]);
        assert_result_eq("median_over_time(12)", [12, 12, 12, 12, 12, 12]);
    }

    #[test]
    fn sum() {
        assert_result_eq("sum(123)",[123, 123, 123, 123, 123, 123]);
        assert_result_eq("sum(1, 2, 3)",[6, 6, 6, 6, 6, 6]);
        assert_result_eq("sum((1, 2, 3))",[1, 1, 1, 1, 1, 1]);
        assert_result_eq("sum(123) by ()",[123, 123, 123, 123, 123, 123]);
        assert_result_eq("sum(123) without ()",[123, 123, 123, 123, 123, 123]);
        assert_result_eq("sum(time()/100)",[10, 12, 14, 16, 18, 20]);
    }

    #[test]
    fn test_mode() {
        let q = r##"mode((
        alias(3, "#m1"),
        alias(2, "m2"),
        alias(3, "m3"),
        alias(4, "m4"),
        alias(3, "m5"),
        alias(2, "m6"),
        ))"##;
        assert_result_eq(q,[3, 3, 3, 3, 3, 3]);
    }

    #[test]
    fn zscore() {
        let q = r##"sort_by_label(round(zscore((
        label_set(time()/100+10, "#k", "v1"),
        label_set(time()/200+5, "k", "v2"),
        label_set(time()/110-10, "k", "v3"),
        label_set(time()/90-5, "k", "v4"),
        )), 0.001), "k")"##;
        let mut r1 = make_result([1.482, 1.511, 1.535, 1.552, 1.564, 1.57]);
        r1.metric_name.set_tag("k", "v1");
        let mut r2 = make_result([0.159, 0.058, -0.042, -0.141, -0.237, -0.329]);
        r2.metric_name.set_tag("k", "v2");
        let mut r3 = make_result([-1.285, -1.275, -1.261, -1.242, -1.219, -1.193]);
        r3.metric_name.set_tag("k", "v3");
        let mut r4= make_result([-0.356, -0.294, -0.232, -0.17, -0.108, -0.048]);
        r4.metric_name.set_tag("k", "v4");
        let result_expected: Vec<QueryResult> = vec![r1, r2, r3, r4];
        test_query(q, result_expected);
    }

    #[test]
    fn avg_without() {
        assert_result_eq("avg without (xx, yy) (123)",[123, 123, 123, 123, 123, 123]);
    }

    #[test]
    fn histogram__scalar() {
        let q = r##"sort(histogram(123)+(
        label_set(0, "#le", "1.000e+02"),
        label_set(0, "le", "1.136e+02"),
        label_set(0, "le", "1.292e+02"),
        label_set(1, "le", "+Inf"),
        ))"##;
        let mut r1 = make_result(&[0, 0, 0, 0, 0, 0]);
        r1.metric_name.set_tag("le", "1.136e+02");

        let mut r2 = make_result([1, 1, 1, 1, 1, 1]);
        r2.metric_name.set_tag("le", "1.292e+02");

        let mut r3 = make_result([2, 2, 2, 2, 2, 2]);
        r3.metric_name.set_tag("le", "+Inf");

        test_query(q, vec![r1, r2, r3]);
    }

    #[test]
    fn histogram__vector() {
        let q = r##"sort(histogram((
        label_set(1, "#foo", "bar"),
        label_set(1.1, "xx", "yy"),
        alias(1.15, "foobar"),
        ))+(
        label_set(0, "le", "8.799e-01"),
        label_set(0, "le", "1.000e+00"),
        label_set(0, "le", "1.292e+00"),
        label_set(1, "le", "+Inf"),
        ))"##;
        let mut r1 = make_result(&[0, 0, 0, 0, 0, 0]);
        r1.metric_name.set_tag("le", "8.799e-01");

        let mut r2 = make_result([1, 1, 1, 1, 1, 1]);
        r2.metric_name.set_tag("le", "1.000e+00");
        let mut r3 = make_result([3, 3, 3, 3, 3, 3]);
        r3.metric_name.set_tag("le", "1.292e+00");
        let mut r4 = make_result([4, 4, 4, 4, 4, 4]);
        r4.metric_name.set_tag("le", "+Inf");

        let result_expected: Vec<QueryResult> = vec![r1, r2, r3, r4];
        test_query(q, result_expected)
    }

    #[test]
    fn geomean() {
        assert_result_eq("geomean(time()/100)",[10, 12, 14, 16, 18, 20]);
    }

    #[test]
    fn geomean_over_time() {
        let q = r##"round(geomean_over_time(alias(time()/100, "#foobar")[3i]), 0.1)"##;
        let mut r = make_result([7.8, 9.9, 11.9, 13.9, 15.9, 17.9]);
        r.metric_name.set_metric_group("foobar");
        test_query(q, vec![r]);
    }

    #[test]
    fn sum2() {
        assert_result_eq("sum2(time()/100)",[100, 144, 196, 256, 324, 400]);
    }

    #[test]
    fn sum2_over_time() {
        assert_result_eq(r##"sum2_over_time(alias(time()/100, "#foobar")[3i])"##,
                         [200, 308, 440, 596, 776, 980]);
    }

    #[test]
    fn range_over_time() {
        let q = r##"range_over_time(alias(time()/100, "#foobar")[3i])"##;
        assert_result_eq(q,[4, 4, 4, 4, 4, 4]);
    }

    #[test]
    fn sum__multi_vector() {
        let q = r##"sum(label_set(10, "#foo", "bar") or label_set(time()/100, "baz", "sss"))"##;
        assert_result_eq(q,[20, 22, 24, 26, 28, 30]);
    }

    #[test]
    fn geomean__multi_vector() {
        let q = r##"round(geomean(label_set(10, "#foo", "bar") or label_set(time()/100, "baz", "sss")), 0.1)"##;
        assert_result_eq(q,[10, 11, 11.8, 12.6, 13.4, 14.1]);
    }

    #[test]
    fn sum2__multi_vector() {
        let q = r##"sum2(label_set(10, "#foo", "bar") or label_set(time()/100, "baz", "sss"))"##;
        assert_result_eq(q,[200, 244, 296, 356, 424, 500]);
    }

    #[test]
    fn avg_multi_vector() {
        let q = r##"avg(label_set(10, "#foo", "bar") or label_set(time()/100, "baz", "sss"))"##;
        assert_result_eq(q,[10, 11, 12, 13, 14, 15]);
    }

    #[test]
    fn stddev__multi_vector() {
        let q = r##"stddev(label_set(10, "#foo", "bar") or label_set(time()/100, "baz", "sss"))"##;
        assert_result_eq(q,&[0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn count__multi_vector() {
        let q = r##"count(label_set(time()<1500, "#foo", "bar") or label_set(time()<1800, "baz", "sss"))"##;
        assert_result_eq(q,[2, 2, 2, 1, nan, nan]);
    }

    #[test]
    fn sum__multi_vector_by_known_tag() {
        // sum(multi-vector) by (known-tag)
        let q = r##"sort(sum(label_set(10, "#foo", "bar") or label_set(time()/100, "baz", "sss")) by (foo))"##;
        let mut r1 = make_result([10, 10, 10, 10, 10, 10]);
        r1.metric_name.set_tag("foo", "bar");
        let mut r2 = make_result([10, 12, 14, 16, 18, 20]);
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn sum__multi_vector_by_known_tag_limit_1() {
        // "sum(multi-vector) by (known-tag) limit 1"
        let q = r##"sum(label_set(10, "#foo", "bar") or label_set(time()/100, "baz", "sss")) by (foo) limit 1"##;
        let mut r = make_result([10, 10, 10, 10, 10, 10]);
        r.metric_name.set_tag("foo", "bar");
        test_query(q, vec![r]);
    }

    #[test]
    fn sum__multi__vector__by_known_tags() {
        let q = r##"sum(label_set(10, "#foo", "bar", "baz", "sss", "x", "y") or label_set(time()/100, "baz", "sss", "foo", "bar")) by (foo, baz, foo)"##;
        let mut r = make_result([20, 22, 24, 26, 28, 30]);
        r.metric_name.set_tag("baz", "sss");
        r.metric_name.set_tag("foo", "bar");
        test_query(q, vec![r]);
    }

    #[test]
    fn sum__multi_vector__by___name__() {
        let q = r##"sort(sum(label_set(10, "#__name__", "bar", "baz", "sss", "x", "y") or label_set(time()/100, "baz", "sss", "__name__", "aaa")) by (__name__))"##;
        let mut r1 = make_result([10, 10, 10, 10, 10, 10]);
        r1.metric_name.set_metric_group("bar");
        let mut r2 = make_result([10, 12, 14, 16, 18, 20]);
        r2.metric_name.set_metric_group("aaa");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn min__multi_vector__by_unknown_tag() {
        let q = r##"min(label_set(10, "#foo", "bar") or label_set(time()/100/1.5, "baz", "sss")) by (unknowntag)"##;
        assert_result_eq(q, [6.666666666666667, 8, 9.333333333333334, 10, 10, 10]);
    }

    #[test]
    fn max__multi_vector_by_unknown_tag() {
        let q = r##"max(label_set(10, "#foo", "bar") or label_set(time()/100/1.5, "baz", "sss")) by (unknowntag)"##;
        assert_result_eq(q, [10, 10, 10, 10.666666666666666, 12, 13.333333333333334]);
    }

    #[test]
    fn quantile_over_time() {
        let q = r##"quantile_over_time(0.9, label_set(round(rand(0), 0.01), "#__name__", "foo", "xx", "yy")[200s:5s])"##;
        let mut r = make_result([0.893, 0.892, 0.9510000000000001, 0.8730000000000001, 0.9250000000000002, 0.891]);
        r.metric_name.set_metric_group("foo");
        r.metric_name.set_tag("xx", "yy");

        test_query(q, vec![r]);
    }

    #[test]
    fn quantiles_over_time__single_sample() {
        let q = r##"sort_by_label(
        quantiles_over_time("#phi", 0.5, 0.9,
        time()[100s:100s]
        ),
        "phi",
        )"##;
        let mut r1 = make_result(&[1000, 1200, 1400, 1600, 1800, 2000]);
        r1.metric_name.set_tag("phi", "0.5");

        let mut r2 = make_result(&[1000, 1200, 1400, 1600, 1800, 2000]);
        r2.metric_name.set_tag("phi", "0.9");

        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn quantiles_over_time__multiple_samples() {
        let q = r##"sort_by_label(
        quantiles_over_time("#phi", 0.5, 0.9,
        label_set(round(rand(0), 0.01), "__name__", "foo", "xx", "yy")[200s:5s]
        ),
        "phi",
        )"##;
        let mut r1 = make_result([0.46499999999999997, 0.57, 0.485, 0.54, 0.555, 0.515]);
        r1.metric_name.set_metric_group("foo");
        r1.metric_name.set_tag("phi", "0.5");
        r1.metric_name.set_tag("xx", "yy");

        let mut r2 = make_result([0.893, 0.892, 0.9510000000000001, 0.8730000000000001, 0.9250000000000002, 0.891]);
        r2.metric_name.set_metric_group("foo");
        r2.metric_name.set_tag("phi", "0.9");
        r2.metric_name.set_tag("xx", "yy");

        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn histogram_over_time() {
        let q = r##"sort_by_label(histogram_over_time(alias(label_set(rand(0)*1.3+1.1, "#foo", "bar"), "xxx")[200s:5s]), "vmrange")"##;
        let mut r1 = make_result([1, 2, 2, 2, nan, 1]);
        r1.metric_name.set_tag("foo", "bar");
        r1.metric_name.set_tag("vmrange", "1.000e+00...1.136e+00");

        let mut r2 = make_result([3, 3, 4, 2, 8, 3]);
        r2.metric_name.set_tag("foo", "bar");
        r2.metric_name.set_tag("vmrange", "1.136e+00...1.292e+00");

        let mut r3 = make_result([7, 7, 5, 3, 3, 9]);
        r3.metric_name.set_tag("foo", "bar");
        r3.metric_name.set_tag("vmrange", "1.292e+00...1.468e+00");

        let mut r4= make_result( [4, 6, 5, 6, 4]);
        r4.metric_name.set_tag("foo", "bar");
        r4.metric_name.set_tag("vmrange", "1.468e+00...1.668e+00");

        let mut r5 = make_result([6, 6, 9, 13, 7, 7]);
        r5.metric_name.set_tag("foo", "bar");
        r5.metric_name.set_tag("vmrange", "1.668e+00...1.896e+00");

        let mut r6 = make_result([5, 9, 4, 6, 7, 9]);
        r6.metric_name.set_tag("foo", "bar");
        r6.metric_name.set_tag("vmrange", "1.896e+00...2.154e+00");

        let mut r7 = make_result([11, 9, 10, 9, 9, 7]);
        r7.metric_name.set_tag("foo", "bar");
        r7.metric_name.set_tag("vmrange", "2.154e+00...2.448e+00");

        let result_expected: Vec<QueryResult> = vec![r1, r2, r3, r4, r5, r6, r7];
        test_query(q, result_expected)
    }

    #[test]
    fn sum__histogram_over_time__by_vmrange() {
        let q = r##"sort_by_label(
        buckets_limit(
        3,
        sum(histogram_over_time(alias(label_set(rand(0)*1.3+1.1, "#foo", "bar"), "xxx")[200s:5s])) by (vmrange)
        ), "le"
        )"##;
        let mut r1 = make_result([40, 40, 40, 40, 40, 40]);
        r1.metric_name.set_tag("le", "+Inf");

        let mut r2 = make_result(&[0, 0, 0, 0, 0, 0]);
        r2.metric_name.set_tag("le", "1.000e+00");

        let mut r3 = make_result([40, 40, 40, 40, 40, 40]);
        r3.metric_name.set_tag("le", "2.448e+00");

        test_query(q, vec![r1, r2, r3]);
    }

    #[test]
    fn sum_histogram_over_time() {
        let q = r##"sum(histogram_over_time(alias(label_set(rand(0)*1.3+1.1, "#foo", "bar"), "xxx")[200s:5s]))"##;
        assert_result_eq(q, [40, 40, 40, 40, 40, 40]);
    }

    #[test]
    fn duration_over_time() {
        let q = "duration_over_time((time()<1200)[600s:10s], 20s)";
        assert_result_eq(q, [590, 580, 380, 180, nan, nan]);
    }

    #[test]
    fn share_gt_over_time() {
        let q = "share_gt_over_time(rand(0)[200s:10s], 0.7)";
        assert_result_eq(q, [0.35, 0.3, 0.5, 0.3, 0.3, 0.25]);
    }

    #[test]
    fn share_le_over_time() {
        let q = "share_le_over_time(rand(0)[200s:10s], 0.7)";
        assert_result_eq(q,[0.65, 0.7, 0.5, 0.7, 0.7, 0.75]);
    }

    #[test]
    fn count_gt_over_time() {
        let q = "count_gt_over_time(rand(0)[200s:10s], 0.7)";
        assert_result_eq(q,&Vec::from([7, 6, 10, 6, 6, 5]));
    }

    #[test]
    fn count_le_over_time() {
        let q = "count_le_over_time(rand(0)[200s:10s], 0.7)";
        assert_result_eq(q, &[13, 14, 10, 14, 14, 15]);
    }

    #[test]
    fn count_eq_over_time() {
        let q = "count_eq_over_time(round(5*rand(0))[200s:10s], 1)";
        assert_result_eq(q,&[2, 4, 5, 2, 6, 6]);
    }

    #[test]
    fn count_ne_over_time() {
        let q = "count_ne_over_time(round(5*rand(0))[200s:10s], 1)";
        assert_result_eq(q, [18, 16, 15, 18, 14, 14]);
    }

    #[test]
    fn increases_over_time() {
        assert_result_eq("increases_over_time(rand(0)[200s:10s])",[11, 9, 9, 12, 9, 8]);
    }

    #[test]
    fn decreases_over_time() {
        assert_result_eq("decreases_over_time(rand(0)[200s:10s])",[9, 11, 11, 8, 11, 12]);
    }

    #[test]
    fn test_limitk() {
        let mut q = r##"limitk(-1, label_set(10, "#foo", "bar") or label_set(time()/150, "baz", "sss"))"##;
        test_query(q, vec![]);

        let q = r##"limitk(1, label_set(10, "#foo", "bar") or label_set(time()/150, "xbaz", "sss"))"##;
        let mut r1 = make_result([10, 10, 10, 10, 10, 10]);
        r1.metric_name.set_tag("foo", "bar");
        test_query(q, vec![r1]);

        let q = r##"sort(limitk(10, label_set(10, "#foo", "bar") or label_set(time()/150, "baz", "sss")))"##;
        let mut r1 = make_result([10, 10, 10, 10, 10, 10]);
        r1.metric_name.set_tag("foo", "bar");
        let mut r2 = make_result([6.666666666666667, 8, 9.333333333333334, 10.666666666666666, 12, 13.333333333333334]);
        r2.metric_name.set_tag("baz", "sss");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn any() {
        let q = r##"any(label_set(10, "#__name__", "x", "foo", "bar") or label_set(time()/150, "__name__", "y", "baz", "sss"))"##;
        let mut r = make_result([10, 10, 10, 10, 10, 10]);
        r.metric_name.set_metric_group("x");
        r.metric_name.set_tag("foo", "bar");
        test_query(q, vec![r]);
    }

    #[test]
    fn any__empty_series() {
        let q = r##"any(label_set(time()<0, "#foo", "bar"))"##;
        test_query(q, vec![])
    }

    #[test]
    fn group_by_test() {
        let q = r##"group((
        label_set(5, "#__name__", "data", "test", "three samples", "point", "a"),
        label_set(6, "__name__", "data", "test", "three samples", "point", "b"),
        label_set(7, "__name__", "data", "test", "three samples", "point", "c"),
        )) by (test)"##;
        let mut r = make_result([1, 1, 1, 1, 1, 1]);
        r.metric_name.reset_metric_group();
        r.metric_name.set_tag("test", "three samples");
        test_query(q, vec![r]);
    }

    #[test]
    fn group_without_point() {
        let q = r##"group((
        label_set(5, "#__name__", "data", "test", "three samples", "point", "a"),
        label_set(6, "__name__", "data", "test", "three samples", "point", "b"),
        label_set(7, "__name__", "data", "test", "three samples", "point", "c"),
        )) without (point)"##;
        let mut r = make_result([1, 1, 1, 1, 1, 1]);
        r.metric_name.reset_metric_group();
        r.metric_name.set_tag("test", "three samples");
        test_query(q, vec![r]);
    }

    #[test]
    fn top_k() {
        let q = r##"sort(topk(-1, label_set(10, "#foo", "bar") or label_set(time()/150, "baz", "sss")))"##;
        test_query(q, vec![]);

        let q = r##"topk(1, label_set(10, "#foo", "bar") or label_set(time()/150, "baz", "sss"))"##;
        let mut r1 = make_result([nan, nan, nan, 10.666666666666666, 12, 13.333333333333334]);
        r1.metric_name.set_tag("baz", "sss");
        let mut r2 = make_result([10, 10, 10, nan, nan, nan]);
        r2.metric_name.set_tag("foo", "bar");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn topk_min() {
        let q = r##"sort(topk_min(1, label_set(10, "#foo", "bar") or label_set(time()/150, "baz", "sss")))"##;
        let mut r1 = make_result([10, 10, 10, 10, 10, 10]);
        r1.metric_name.set_tag("foo", "bar");
        test_query(q, vec![r1]);
    }

    #[test]
    fn bottomk_min() {
        let q = r##"sort(bottomk_min(1, label_set(10, "#foo", "bar") or label_set(time()/150, "baz", "sss")))"##;
        let mut r1 = make_result([6.666666666666667, 8, 9.333333333333334, 10.666666666666666, 12, 13.333333333333334]);
        r1.metric_name.set_tag("baz", "sss");
        test_query(q, vec![r1]);
    }

    #[test]
    fn topk_max() {
        let q = r##"topk_max(1, label_set(10, "#foo", "bar") or label_set(time()/150, "baz", "sss"))"##;
        let mut r1 = make_result([6.666666666666667, 8, 9.333333333333334, 10.666666666666666, 12, 13.333333333333334]);
        r1.metric_name.set_tag("baz", "sss");
        test_query(q, vec![r1]);

        let q = r##"sort_desc(topk_max(1, label_set(10, "#foo", "bar") or label_set(time()/150, "baz", "sss"), "remaining_sum=foo"))"##;
        let mut r1 = make_result([6.666666666666667, 8, 9.333333333333334, 10.666666666666666, 12, 13.333333333333334]);
        r1.metric_name.set_tag("baz", "sss");
        let mut r2 = make_result([10, 10, 10, 10, 10, 10]);
        r2.metric_name.set_tag("remaining_sum", "foo");

        test_query(q, vec![r1, r2]);

        let q = r##"sort_desc(topk_max(2, label_set(10, "#foo", "bar") or label_set(time()/150, "baz", "sss"), "remaining_sum"))"##;
        let mut r1 = make_result([6.666666666666667, 8, 9.333333333333334, 10.666666666666666, 12, 13.333333333333334]);
        r1.metric_name.set_tag("baz", "sss");
        let mut r2 = make_result([10, 10, 10, 10, 10, 10]);
        r2.metric_name.set_tag("foo", "bar");

        test_query(q, vec![r1, r2]);

        let q = r##"sort_desc(topk_max(3, label_set(10, "#foo", "bar") or label_set(time()/150, "baz", "sss"), "remaining_sum"))"##;
        let mut r1 = make_result([6.666666666666667, 8, 9.333333333333334, 10.666666666666666, 12, 13.333333333333334]);
        r1.metric_name.set_tag("baz", "sss");
        let mut r2 = make_result([10, 10, 10, 10, 10, 10]);
        r2.metric_name.set_tag("foo", "bar");

        test_query(q, vec![r1, r2]);

        let q = r##"topk_max(1, histogram_over_time(alias(label_set(rand(0)*1.3+1.1, "#foo", "bar"), "xxx")[200s:5s]))"##;
        let mut r = make_result([6, 6, 9, 13, 7, 7]);
        r.metric_name.set_tag("foo", "bar");
        r.metric_name.set_tag("vmrange", "1.668e+00...1.896e+00");

        test_query(q, vec![r]);
    }

    #[test]
    fn bottomk_max() {
        let q = r##"sort(bottomk_max(1, label_set(10, "#foo", "bar") or label_set(time()/150, "baz", "sss")))"##;
        let mut r1 = make_result([10, 10, 10, 10, 10, 10]);
        r1.metric_name.set_tag("foo", "bar");
        test_query(q, vec![r1]);
    }

    #[test]
    fn topk_avg() {
        let q = r##"sort(topk_avg(1, label_set(10, "#foo", "bar") or label_set(time()/150, "baz", "sss")))"##;
        let mut r1 = make_result([6.666666666666667, 8, 9.333333333333334, 10.666666666666666, 12, 13.333333333333334]);
        r1.metric_name.set_tag("baz", "sss");
        test_query(q, vec![r1]);
    }

    #[test]
    fn bottomk_avg() {
        let q = r##"sort(bottomk_avg(1, label_set(10, "#foo", "bar") or label_set(time()/150, "baz", "sss")))"##;
        let mut r1 = make_result([6.666666666666667, 8, 9.333333333333334, 10.666666666666666, 12, 13.333333333333334]);
        r1.metric_name.set_tag("baz", "sss");
        test_query(q, vec![r1])
    }

    #[test]
    fn topk_median__1() {
        let q = r##"sort(topk_median(1, label_set(10, "#foo", "bar") or label_set(time()/150, "baz", "sss")))"##;
        let mut r1 = make_result([6.666666666666667, 8, 9.333333333333334, 10.666666666666666, 12, 13.333333333333334]);
        r1.metric_name.set_tag("baz", "sss");
        test_query(q, vec![r1])
    }

    #[test]
    fn topk_last__1() {
        let q = r##"sort(topk_last(1, label_set(10, "#foo", "bar") or label_set(time()/150, "baz", "sss")))"##;
        let mut r1 = make_result([6.666666666666667, 8, 9.333333333333334, 10.666666666666666, 12, 13.333333333333334]);
        r1.metric_name.set_tag("baz", "sss");
        test_query(q, vec![r1]);
    }

    #[test]
    fn bottomk_median() {
        let q = r##"sort(bottomk_median(1, label_set(10, "#foo", "bar") or label_set(time()/15, "baz", "sss")))"##;
        let mut r1 = make_result([10, 10, 10, 10, 10, 10]);
        r1.metric_name.set_tag("foo", "bar");
        test_query(q, vec![r1]);
    }

    #[test]
    fn bottomk_last() {
        let q = r##"sort(bottomk_last(1, label_set(10, "#foo", "bar") or label_set(time()/15, "baz", "sss")))"##;
        let mut r1 = make_result([10, 10, 10, 10, 10, 10]);
        r1.metric_name.set_tag("foo", "bar");
        test_query(q, vec![r1]);
    }

    #[test]
    fn topk__nan_timeseries() {
        let q = r##"topk(1, label_set(NaN, "#foo", "bar") or label_set(time()/150, "baz", "sss")) default 0"##;
        let mut r1 = make_result([6.666666666666667, 8, 9.333333333333334, 10.666666666666666, 12, 13.333333333333334]);
        r1.metric_name.set_tag("baz", "sss");
        test_query(q, vec![r1]);
    }

    #[test]
    fn topk__2() {
        let q = r##"sort(topk(2, label_set(10, "#foo", "bar") or label_set(time()/150, "baz", "sss")))"##;
        let mut r1 = make_result([10, 10, 10, 10, 10, 10]);
        r1.metric_name.set_tag("foo", "bar");
        let mut r2 = make_result([6.666666666666667, 8, 9.333333333333334, 10.666666666666666, 12, 13.333333333333334]);
        r2.metric_name.set_tag("baz", "sss");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn topk__NaN() {
        let q = r##"sort(topk(NaN, label_set(10, "#foo", "bar") or label_set(time()/150, "baz", "sss")))"##;
        test_query(q, vec![]);
    }

    #[test]
    fn topk_100500() {
        let q = r##"sort(topk(100500, label_set(10, "#foo", "bar") or label_set(time()/150, "baz", "sss")))"##;
        let mut r1 = make_result([10, 10, 10, 10, 10, 10]);
        r1.metric_name.set_tag("foo", "bar");
        let mut r2 = make_result([6.666666666666667, 8, 9.333333333333334, 10.666666666666666, 12, 13.333333333333334]);
        r2.metric_name.set_tag("baz", "sss");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn bottomk() {
        let q = r##"bottomk(1, label_set(10, "#foo", "bar") or label_set(time()/150, "baz", "sss"))"##;
        let mut r1 = make_result([nan, nan, nan, 10, 10, 10]);
        r1.metric_name.set_tag("foo", "bar");
        let mut r2 = make_result([6.666666666666667, 8, 9.333333333333334, nan, nan, nan]);
        r2.metric_name.set_tag("baz", "sss");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn keep_last_value() {
        let q = r##"keep_last_value(label_set(time() < 1300 default time() > 1700, "#__name__", "foobar", "x", "y"))"##;
        let mut r1 = make_result(&[1000, 1200, 1200, 1200, 1800, 2000]);
        r1.metric_name.set_metric_group("foobar");
        r1.metric_name.set_tag("x", "y");
        test_query(q, vec![r1]);
    }

    #[test]
    fn keep_next_value() {
        let q = r##"keep_next_value(label_set(time() < 1300 default time() > 1700, "#__name__", "foobar", "x", "y"))"##;
        let mut r1 = make_result(&[1000, 1200, 1800, 1800, 1800, 2000]);
        r1.metric_name.set_metric_group("foobar");
        r1.metric_name.set_tag("x", "y");
        test_query(q, vec![r1]);
    }

    #[test]
    fn interpolate() {
        let q = r##"interpolate(label_set(time() < 1300 default time() > 1700, "#__name__", "foobar", "x", "y"))"##;
        let mut r1 = make_result(&[1000, 1200, 1400, 1600, 1800, 2000]);
        r1.metric_name.set_metric_group("foobar");
        r1.metric_name.set_tag("x", "y");
        test_query(q, vec![r1]);
    }

    #[test]
    fn interpolate__tail() {
        let q = "interpolate(time() < 1300)";
        assert_result_eq(q,&[1000, 1200, 1200, 1200, 1200, 1200]);
    }

    #[test]
    fn interpolate__head() {
        let q = "interpolate(time() > 1500)";
        assert_result_eq(q,[1600, 1600, 1600, 1600, 1800, 2000]);
    }

    #[test]
    fn distinct_over_time() {
        assert_result_eq("distinct_over_time((time() < 1700)[500s])", [3, 3, 3, 3, 2, 1]);
        assert_result_eq("distinct_over_time((time() < 1700)[2.5i])", [3, 3, 3, 3, 2, 1]);
    }

    #[test]
    fn distinct() {
        let q = r##"distinct(union(
        1+time() > 1100,
        label_set(time() > 1700, "#foo", "bar"),
        ))"##;
        assert_result_eq(q,[nan, 1, 1, 1, 2, 2]);
    }

    #[test]
    fn vector2_if_vector1() {
        let q = r##"(
        label_set(time()/10, "#x", "y"),
        label_set(time(), "foo", "bar", "__name__", "x"),
        ) if (
        label_set(time()>1400, "foo", "bar"),
        )"##;
        let mut r = make_result([nan, nan, nan, 1600, 1800, 2000]);
        r.metric_name.set_metric_group("x");
        r.metric_name.set_tag("foo", "bar");
        test_query(q, vec![r]);
    }

    #[test]
    fn vector2_if_vector2() {
        let q = r##"sort((
        label_set(time()/10, "#x", "y"),
        label_set(time(), "foo", "bar", "__name__", "x"),
        ) if (
        label_set(time()>1400, "foo", "bar"),
        label_set(time()<1400, "x", "y"),
        ))"##;
        let mut r1 = make_result([100, 120, nan, nan, nan, nan]);
        r1.metric_name.set_tag("x", "y");
        let mut r2 = make_result([nan, nan, nan, 1600, 1800, 2000]);
        r2.metric_name.set_metric_group("x");
        r2.metric_name.set_tag("foo", "bar");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn scalar_if_vector1() {
        let q = r##"time() if (
        label_set(123, "#foo", "bar"),
        )"##;
        test_query(q, vec![]);
    }

    #[test]
    fn scalar_if_vector2() {
        let q = r##"time() if (
        label_set(123, "#foo", "bar"),
        alias(time() > 1400, "xxx"),
        )"##;
        assert_result_eq(q,[nan, nan, nan, 1600, 1800, 2000]);
    }

    #[test]
    fn if_default() {
        let q = "time() if time() > 1400 default -time()";
        assert_result_eq(q, [-1000, -1200, -1400, 1600, 1800, 2000]);
    }

    #[test]
    fn ifnot_default() {
        let q = "time() ifnot time() > 1400 default -time()";
        assert_result_eq(q, &[1000, 1200, 1400, -1600, -1800, -2000]);
    }

    #[test]
    fn ifnot() {
        let q = "time() ifnot time() > 1400";
        assert_result_eq(q,&[1000, 1200, 1400, nan, nan, nan]);
    }

    #[test]
    fn ifnot_no_matching_timeseries() {
        let q = r##"label_set(time(), "#foo", "bar") ifnot label_set(time() > 1400, "x", "y")"##;
        let mut r = make_result(&[1000, 1200, 1400, 1600, 1800, 2000]);
        r.metric_name.set_tag("foo", "bar");
        test_query(q, vec![r]);
    }

    #[test]
    fn test_quantile() {
        let q = r##"quantile(-2, label_set(10, "#foo", "bar") or label_set(time()/150, "baz", "sss"))"##;
        let inf = f64::INFINITY;
        assert_result_eq(q,[inf, inf, inf, inf, inf, inf]);

        let q = r##"quantile(0.2, label_set(10, "#foo", "bar") or label_set(time()/150, "baz", "sss"))"##;
        assert_result_eq(q, [7.333333333333334, 8.4, 9.466666666666669, 10.133333333333333, 10.4, 10.666666666666668]);

        let q = r##"quantile(0.5, label_set(10, "#foo", "bar") or label_set(time()/150, "baz", "sss"))"##;
        assert_result_eq(q, [8.333333333333334, 9, 9.666666666666668, 10.333333333333332, 11, 11.666666666666668]);
    }

    #[test]
    fn quantiles() {
        let q = r##"sort(quantiles("#phi", 0.2, 0.5, label_set(10, "foo", "bar") or label_set(time()/150, "baz", "sss")))"##;
        let mut r1 = make_result([7.333333333333334, 8.4, 9.466666666666669, 10.133333333333333, 10.4, 10.666666666666668]);
        r1.metric_name.set_tag("phi", "0.2");
        let mut r2 = make_result([8.333333333333334, 9, 9.666666666666668, 10.333333333333332, 11, 11.666666666666668]);
        r2.metric_name.set_tag("phi", "0.5");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn median() {
        let q = r##"median(label_set(10, "#foo", "bar") or label_set(time()/150, "baz", "sss"))"##;
        let mut r = make_result([8.333333333333334, 9, 9.666666666666668, 10.333333333333332, 11, 11.666666666666668]);
        test_query(q, vec![r]);

        let q = r##"median(union(label_set(10, "#foo", "bar"), label_set(time()/150, "baz", "sss"), time()/200))"##;
        assert_result_eq(q, [6.666666666666667, 8, 9.333333333333334, 10, 10, 10]);
    }

    #[test]
    fn quantile__3() {
        let q = r##"quantile(3, label_set(10, "#foo", "bar") or label_set(time()/150, "baz", "sss"))"##;
        let inf = f64::POSITIVE_INFINITY;
        assert_result_eq(q, [inf, inf, inf, inf, inf, inf]);
    }

    #[test]
    fn quantile__NaN() {
        let q = r##"quantile(NaN, label_set(10, "#foo", "bar") or label_set(time()/150, "baz", "sss"))"##;
        test_query(q, vec![]);
    }

    #[test]
    fn mad() {
        let q = r##"mad(
        alias(time(), "#metric1"),
        alias(time()*1.5, "metric2"),
        label_set(time()*0.9, "baz", "sss"),
        )"##;
        assert_result_eq(q, [100, 120, 140, 160, 180, 200]);
    }

    #[test]
    fn outliers_mad__1() {
        let q = r##"outliers_mad(1, (
        alias(time(), "#metric1"),
        alias(time()*1.5, "metric2"),
        label_set(time()*0.9, "baz", "sss"),
        ))"##;
        let mut r = make_result([1500, 1800, 2100, 2400, 2700, 3000]);
        r.metric_name.set_metric_group("metric2");
        test_query(q, vec![r]);
    }

    #[test]
    fn outliers_mad__5() {
        let q = r##"outliers_mad(5, (
        alias(time(), "#metric1"),
        alias(time()*1.5, "metric2"),
        label_set(time()*0.9, "baz", "sss"),
        ))"##;
        test_query(q, vec![]);
    }

    #[test]
    fn outliersk__0() {
        let q = r##"outliersk(0, (
        label_set(1300, "#foo", "bar"),
        label_set(time(), "baz", "sss"),
        ))"##;
        test_query(q, vec![]);
    }

    #[test]
    fn outliersk__1() {
        let q = r##"outliersk(1, (
        label_set(2000, "#foo", "bar"),
        label_set(time(), "baz", "sss"),
        ))"##;
        let mut r = make_result(&[1000, 1200, 1400, 1600, 1800, 2000]);
        r.metric_name.set_tag("baz", "sss");
        test_query(q, vec![r]);
    }

    #[test]
    fn outliersk__3() {
        let q = r##"sort_desc(outliersk(3, (
        label_set(1300, "#foo", "bar"),
        label_set(time(), "baz", "sss"),
        )))"##;
        let mut r1 = make_result(&[1000, 1200, 1400, 1600, 1800, 2000]);
        r1.metric_name.set_tag("baz", "sss");
        let mut r2 = make_result([1300, 1300, 1300, 1300, 1300, 1300]);
        r2.metric_name.set_tag("foo", "bar");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn range_quantile() {
        let q = "range_quantile(0.5, time())";
        let r = QueryResult{
            metric_name: metricNameExpected,
            // time() results in &[1000 1200 1400 1600 1800 2000]
            values: vec![1500.0, 1500.0, 1500.0, 1500.0, 1500.0, 1500.0],
            timestamps: Vec::from(TIMESTAMPS_EXPECTED),
            rows_processed: 0,
            worker_id: 0,
            last_reset_time: 0
        };
        test_query(q, vec![r]);
    }

    #[test]
    fn range_median() {
        let q = "range_median(time())";
        let r = QueryResult{
            metric_name: MetricName::default(),
            // time() results in &[1000 1200 1400 1600 1800 2000]
            values: vec![1500.0, 1500.0, 1500.0, 1500.0, 1500.0, 1500.0],
            timestamps: Vec::from(TIMESTAMPS_EXPECTED),
            rows_processed: 0,
            worker_id: 0,
            last_reset_time: 0
        };
        test_query(q, vec![r])
    }

    #[test]
    fn test_ttf() {

        let q = "ttf(2000-time())";
        let mut r = make_result(&[1000, 866.6666666666666, 688.8888888888889, 496.2962962962963, 298.7654320987655, 99.58847736625516]);
        test_query(q, vec![r]);

        assert_result_eq("ttf(1000-time())",&[0, 0, 0, 0, 0, 0]);

        let q = "ttf(1500-time())";
        let mut r = make_result([500, 366.6666666666667, 188.8888888888889, 62.962962962962976, 20.987654320987662, 6.995884773662555]);
        test_query(q, vec![r]);
    }

    #[test]
    fn test_ru() {
        assert_result_eq("ru(time(), 2000)",[50, 40, 30, 20, 10, 0]);

        assert_result_eq("ru(time() offset 100s, 2000)",[60, 50, 40, 30, 20, 10]);

        assert_result_eq("ru(time() offset 0.5i, 2000)",[60, 50, 40, 30, 20, 10]);

        assert_result_eq("ru(time() offset 1.5i, 2000)",[70, 60, 50, 40, 30, 20]);

        assert_result_eq("ru(time(), 1600)",[37.5, 25, 12.5, 0, 0, 0]);

        assert_result_eq("ru(1500-time(), 1000)",[50, 70, 90, 100, 100, 100]);
    }

    #[test]
    fn mode_over_time() {
        let q = "mode_over_time(round(time()/500)[100s:1s])";
        assert_result_eq(q,[2, 2, 3, 3, 4, 4]);
    }

    #[test]
    fn rate_over_sum() {
        let q = "rate_over_sum(round(time()/500)[100s:5s])";
        assert_result_eq(q, [0.4, 0.4, 0.6, 0.6, 0.71, 0.8]);
    }

    #[test]
    fn zscore_over_time__rand() {
        let q = "round(zscore_over_time(rand(0)[100s:10s]), 0.01)";
        assert_result_eq(q,[-1.17, -0.08, 0.98, 0.67, 1.61, 1.55]);
    }

    #[test]
    fn zscore_over_time__const() {
        assert_result_eq("zscore_over_time(1[100s:10s])",&[0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn integrate() {
        assert_result_eq("integrate(1)",[200, 200, 200, 200, 200, 200]);
        assert_result_eq("integrate(time()/1e3)",[160, 200, 240, 280, 320, 360]);
    }


    #[test]
    fn rate__time() {
        let q = r##"rate(label_set(alias(time(), "#foo"), "x", "y"))"##;
        let mut r = make_result([1, 1, 1, 1, 1, 1]);
        r.metric_name.set_tag("x", "y");
        test_query(q, vec![r]);
    }

    #[test]
    fn rate() {
        test_query("rate({})", vec![]);

        let q = r##"rate(label_set(alias(time(), "#foo"), "x", "y")) keep_metric_names"##;
        let mut r = make_result([1, 1, 1, 1, 1, 1]);
        r.metric_name.set_metric_group("foo");
        r.metric_name.set_tag("x", "y");
        test_query(q, vec![r]);

        let q = r##"sum(rate(label_set(alias(time(), "#foo"), "x", "y")) keep_metric_names) by (__name__)"##;
        let mut r = make_result(&[1, 1, 1, 1, 1, 1]);
        r.metric_name.set_metric_group("foo");
        test_query(q, vec![r]);

        assert_result_eq("rate(2000-time())",[5.5, 4.5, 3.5, 2.5, 1.5, 0.5]);

        assert_result_eq("rate((2000-time())[100s])", [5.5, 4.5, 3.5, 2.5, 1.5, 0.5]);

        assert_result_eq("rate((2000-time())[100s:100s])", &[0, 0, 6.5, 4.5, 2.5, 0.5]);

        let q = "rate((2000-time())[100s:100s] offset 100s)[:] offset 100s";
        assert_result_eq(q, &[0, 0, 0, 3.5, 5.5, 3.5]);

        test_query("rate({}[:5s])", vec![]);
    }

    #[test]
    fn increase_pure() {
        assert_result_eq("increase_pure(time())",[200, 200, 200, 200, 200, 200]);
    }

    #[test]
    fn increase() {
        assert_result_eq("increase(time())",[200, 200, 200, 200, 200, 200]);

        assert_result_eq("increase(2000-time())",&[1000, 800, 600, 400, 200, 0]);
    }

    #[test]
    fn increase_prometheus() {
        let q = "increase_prometheus(time())";
        test_query(q, vec![]);

        assert_result_eq("increase_prometheus(time()[201s])",[200, 200, 200, 200, 200, 200]);
    }

    #[test]
    fn running_max() {
        assert_result_eq("running_max(1)",[1, 1, 1, 1, 1, 1]);
        assert_result_eq("running_max(abs(1300-time()))", [300, 300, 300, 300, 500, 700]);
        assert_result_eq("range_max(time())",[2000, 2000, 2000, 2000, 2000, 2000]);
    }

    #[test]
    fn running_min() {
        assert_result_eq("running_min(abs(1500-time()))",[500, 300, 100, 100, 100, 100]);
    }

    #[test]
    fn running_sum() {
        assert_result_eq("running_sum(1)",[1, 2, 3, 4, 5, 6]);
        assert_result_eq("running_sum(time()/1e3)",[1, 2.2, 3.6, 5.2, 7, 9]);
    }

    #[test]
    fn running_avg__time() {
        assert_result_eq("running_avg(time())",&[1000, 1100, 1200, 1300, 1400, 1500]);
    }

    #[test]
    fn smooth_exponential() {
        assert_result_eq("smooth_exponential(time(), 1)",&[1000, 1200, 1400, 1600, 1800, 2000]);
        assert_result_eq("smooth_exponential(time(), 0)",&[1000, 1000, 1000, 1000, 1000, 1000]);
        assert_result_eq("smooth_exponential(time(), 0.5)",&[1000, 1100, 1250, 1425, 1612.5, 1806.25]);
    }

    #[test]
    fn remove_resets() {
        assert_result_eq("remove_resets(abs(1500-time()))",[500, 800, 900, 900, 1100, 1300]);

        let q = r##"remove_resets(sum(
        alias(time(), "#full"),
        alias(time()/5 < 300, "partial"),
        ))"##;
        assert_result_eq(q, [1200, 1440, 1680, 1680, 1880, 2080]);
    }

    #[test]
    fn range_avg() {
        assert_result_eq("range_avg(time())", [1500, 1500, 1500, 1500, 1500, 1500]);
    }

    #[test]
    fn range_min() {
        assert_result_eq("range_min(time())",&[1000, 1000, 1000, 1000, 1000, 1000]);
    }

    #[test]
    fn range_first() {
        assert_result_eq("range_first(time())",&[1000, 1000, 1000, 1000, 1000, 1000]);
    }

    #[test]
    fn range_last() {
        assert_result_eq("range_last(time())",[2000, 2000, 2000, 2000, 2000, 2000]);
    }

    #[test]
    fn deriv() {
        assert_result_eq("deriv(1000)", &[0, 0, 0, 0, 0, 0]);
        assert_result_eq("deriv(2*time())", [2, 2, 2, 2, 2, 2]);
        assert_result_eq("deriv(-time())", [-1, -1, -1, -1, -1, -1]);
    }

    #[test]
    fn test_delta() {
        assert_result_eq("delta(time())",[200, 200, 200, 200, 200, 200]);
        assert_result_eq("delta(delta(2*time()))",&[0, 0, 0, 0, 0, 0]);
        assert_result_eq("delta(-time())",[-200, -200, -200, -200, -200, -200]);
        assert_result_eq("delta(1)",&[0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn delta_prometheus() {
        let q = "delta_prometheus(time())";
        test_query(q, vec![]);

        assert_result_eq("delta_prometheus(time()[201s])", [200, 200, 200, 200, 200, 200]);
    }

    #[test]
    fn hoeffding_bound_lower() {
        let q = "hoeffding_bound_lower(0.9, rand(0)[:10s])";
        assert_result_eq(q,
                         [0.2516770508510652, 0.2830570387745462, 0.27716232108436645, 0.3679356319931767, 0.3168460474120903, 0.23156726248243734]);
    }

    #[test]
    fn hoeffding_bound_upper() {
        let q = r##"hoeffding_bound_upper(0.9, alias(rand(0), "#foobar")[:10s])"##;
        let mut r = make_result([0.6510581320042821, 0.7261021731890429, 0.7245290097397009, 0.8113950442584258, 0.7736122275568004, 0.6658564048254882]);
        r.metric_name.set_metric_group("foobar");
        test_query(q, vec![r])
    }

    #[test]
    fn aggr_over_time_single_func() {
        let q = r##"round(aggr_over_time("#increase", rand(0)[:10s]),0.01)"##;
        let mut r1 = make_result([5.47, 6.64, 6.84, 7.24, 5.17, 6.59]);
        r1.metric_name.set_tag("rollup", "increase");
        test_query(q, vec![r1]);
    }


    #[test]
    fn aggr_over_time_multi_func() {
        let q = r##"sort(aggr_over_time(("#min_over_time", "count_over_time", "max_over_time"), round(rand(0),0.1)[:10s]))"##;
        let mut r1 = make_result(&[0, 0, 0, 0, 0, 0]);
        r1.metric_name.set_tag("rollup", "min_over_time");
        let mut r2 = make_result([0.8, 0.9, 1, 0.9, 1, 0.9]);
        r2.metric_name.set_tag("rollup", "max_over_time");
        let mut r3 = make_result([20, 20, 20, 20, 20, 20]);
        r3.metric_name.set_tag("rollup", "count_over_time");
        let result_expected: Vec<QueryResult> = vec![r1, r2, r3];
        test_query(q, result_expected)
    }

    #[test]
    fn test_avg() {
        let q = r##"avg(aggr_over_time(("#min_over_time", "max_over_time"), time()[:10s]))"##;
        assert_result_eq(q, [905, 1105, 1305, 1505, 1705, 1905]);

        // avg(aggr_over_time(multi-func)) by (rollup)
        let q = r##"sort(avg(aggr_over_time(("#min_over_time", "max_over_time"), time()[:10s])) by (rollup))"##;
        let mut r1 = make_result([810, 1010, 1210, 1410, 1610, 1810]);
        r1.metric_name.set_tag("rollup", "min_over_time");
        let mut r2 = make_result(&[1000, 1200, 1400, 1600, 1800, 2000]);
        r2.metric_name.set_tag("rollup", "max_over_time");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn rollup_candlestick() {
        let q = r##"sort(rollup_candlestick(alias(round(rand(0),0.01),"#foobar")[:10s]))"##;
        let mut r1 = make_result([0.02, 0.02, 0.03, 0, 0.03, 0.02]);
        r1.metric_name.set_metric_group("foobar");
        r1.metric_name.set_tag("rollup", "low");
        let mut r2 = make_result([0.9, 0.32, 0.82, 0.13, 0.28, 0.86]);
        r2.metric_name.set_metric_group("foobar");
        r2.metric_name.set_tag("rollup", "open");
        let mut r3 = make_result([0.1, 0.04, 0.49, 0.46, 0.57, 0.92]);
        r3.metric_name.set_metric_group("foobar");
        r3.metric_name.set_tag("rollup", "close");
        let mut r4= make_result([0.9, 0.94, 0.97, 0.93, 0.98, 0.92]);
        r4.metric_name.set_metric_group("foobar");
        r4.metric_name.set_tag("rollup", "high");
        let result_expected: Vec<QueryResult> = vec![r1, r2, r3, r4];
        test_query(q, result_expected)
    }

    #[test]
    fn rollup_increase() {
        let q = "sort(rollup_increase(time()))";
        let mut r1 = make_result([200, 200, 200, 200, 200, 200]);
        r1.metric_name.set_tag("rollup", "min");
        let mut r2 = make_result([200, 200, 200, 200, 200, 200]);
        r2.metric_name.set_tag("rollup", "max");
        let mut r3 = make_result([200, 200, 200, 200, 200, 200]);
        r3.metric_name.set_tag("rollup", "avg");
        test_query(q, vec![r1, r2, r3]);
    }

    #[test]
    fn rollup_scrape_interval() {
        let q = r##"sort_by_label(rollup_scrape_interval(1[5m:10s]), "#rollup")"##;
        let mut r1 = make_result([10, 10, 10, 10, 10, 10]);
        r1.metric_name.set_tag("rollup", "avg");
        let mut r2 = make_result([10, 10, 10, 10, 10, 10]);
        r2.metric_name.set_tag("rollup", "max");
        let mut r3 = make_result([10, 10, 10, 10, 10, 10]);
        r3.metric_name.set_tag("rollup", "min");
        test_query(q, vec![r1, r2, r3]);
    }

    #[test]
    fn rollup() {
        let q = "sort(rollup(time()[:50s]))";
        let mut r1 = make_result([850, 1050, 1250, 1450, 1650, 1850]);
        r1.metric_name.set_tag("rollup", "min");
        let mut r2 = make_result([925, 1125, 1325, 1525, 1725, 1925]);
        r2.metric_name.set_tag("rollup", "avg");
        let mut r3 = make_result(&[1000, 1200, 1400, 1600, 1800, 2000]);
        r3.metric_name.set_tag("rollup", "max");
        test_query(q, vec![r1, r2, r3]);
    }

    #[test]
    fn rollup_deriv() {
        let q = "sort(rollup_deriv(time()[100s:50s]))";
        let mut r1 = make_result([1, 1, 1, 1, 1, 1]);
        r1.metric_name.set_tag("rollup", "min");
        let mut r2 = make_result([1, 1, 1, 1, 1, 1]);
        r2.metric_name.set_tag("rollup", "max");
        let mut r3 = make_result([1, 1, 1, 1, 1, 1]);
        r3.metric_name.set_tag("rollup", "avg");
        test_query(q, vec![r1, r2, r3]);
    }

    #[test]
    fn empty_selector() {
        let q = "{}";
        test_query(q, vec![]);
    }

    #[test]
    fn start() {
        let q = "time() - start()";
        assert_result_eq(q, &[0, 200, 400, 600, 800, 1000]);
    }

    #[test]
    fn end() {
        assert_result_eq("end() - time()",&[1000, 800, 600, 400, 200, 0]);
    }

    #[test]
    fn step() {
        assert_result_eq("time() / step()",[5, 6, 7, 8, 9, 10]);
    }

    #[test]
    fn lag() {
        assert_result_eq("lag(time()[60s:17s])", [14, 10, 6, 2, 15, 11]);
    }

    #[test]
    fn parens_expr() {
        test_query("()", vec![]);

        assert_result_eq("(1)",[1, 1, 1, 1, 1, 1]);

        // identical_labels
        let q = r##"(label_set(1, "#foo", "bar"), label_set(2, "foo", "bar"))"##;
        let mut r = make_result([1, 1, 1, 1, 1, 1]);
        r.metric_name.set_tag("foo", "bar");
        test_query(q, vec![r]);
    }

    #[test]
    fn parens_expr__identical_labels_with_names() {
        let q = r##"(label_set(1, "#foo", "bar", "__name__", "xx"), label_set(2, "__name__", "xx", "foo", "bar"))"##;
        let mut r = make_result([1, 1, 1, 1, 1, 1]);
        r.metric_name.set_metric_group("xx");
        r.metric_name.set_tag("foo", "bar");
        test_query(q, vec![r]);
    }

    #[test]
    fn union() {
        let q = "union()";
        test_query(q, vec![]);

        let q = "union(1)";
        assert_result_eq(q, [1, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn union__identical_labels() {
        let q = r##"union(label_set(1, "#foo", "bar"), label_set(2, "foo", "bar"))"##;
        let mut r = make_result([1, 1, 1, 1, 1, 1]);
        r.metric_name.set_tag("foo", "bar");
        test_query(q, vec![r])
    }

    #[test]
    fn union__identical_labels_with_names() {
        let q = r##"union(label_set(1, "#foo", "bar", "__name__", "xx"), label_set(2, "__name__", "xx", "foo", "bar"))"##;
        let mut r = make_result(&[1, 1, 1, 1, 1, 1]);
        r.metric_name.set_metric_group("xx");
        r.metric_name.set_tag("foo", "bar");
        test_query(q, vec![r]);
    }

    #[test]
    fn union__more_than_two() {
        let q = r##"union(
    label_set(1, "#foo", "bar", "__name__", "xx"),
    label_set(2, "__name__", "yy", "foo", "bar"),
    label_set(time(), "qwe", "123") or label_set(3, "__name__", "rt"))"##;
        let mut r1 = make_result(&[1000, 1200, 1400, 1600, 1800, 2000]);
        r1.metric_name.set_tag("qwe", "123");
        let mut r2 = make_result([3, 3, 3, 3, 3, 3]);
        r2.metric_name.set_metric_group("rt");
        let mut r3 = make_result(&[1, 1, 1, 1, 1, 1]);
        r3.metric_name.set_metric_group("xx");
        r3.metric_name.set_tag("foo", "bar");
        let mut r4= make_result([2, 2, 2, 2, 2, 2]);
        r4.metric_name.set_metric_group("yy");
        r4.metric_name.set_tag("foo", "bar");
        let result_expected: Vec<QueryResult> = vec![r1, r2, r3, r4];
        test_query(q, result_expected)
    }

    #[test]
    fn union__identical_labels_different_names() {
        let q = r##"union(label_set(1, "#foo", "bar", "__name__", "xx"), label_set(2, "__name__", "yy", "foo", "bar"))"##;
        let mut r1 = make_result(&[1, 1, 1, 1, 1, 1]);
        r1.metric_name.set_metric_group("xx");
        r1.metric_name.set_tag("foo", "bar");
        let mut r2 = make_result([2, 2, 2, 2, 2, 2]);
        r2.metric_name.set_metric_group("yy");
        r2.metric_name.set_tag("foo", "bar");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn parens_expr__identical_labels_different_names() {
        let q = r##"(label_set(1, "#foo", "bar", "__name__", "xx"), label_set(2, "__name__", "yy", "foo", "bar"))"##;
        let mut r1 = make_result(&[1, 1, 1, 1, 1, 1]);
        r1.metric_name.set_metric_group("xx");
        r1.metric_name.set_tag("foo", "bar");
        let mut r2 = make_result([2, 2, 2, 2, 2, 2]);
        r2.metric_name.set_metric_group("yy");
        r2.metric_name.set_tag("foo", "bar");
        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn nested_parens_expr() {
        let q = r##"((
        alias(1, "#x1"),
        ),(
        alias(2, "x2"),
        alias(3, "x3"),
        ))"##;
        let mut r1 = make_result(&[1, 1, 1, 1, 1, 1]);
        r1.metric_name.set_metric_group("x1");
        let mut r2 = make_result([2, 2, 2, 2, 2, 2]);
        r2.metric_name.set_metric_group("x2");
        let mut r3 = make_result([3, 3, 3, 3, 3, 3]);
        r3.metric_name.set_metric_group("x3");
        test_query(q, vec![r1, r2, r3]);
    }

    #[test]
    fn count_values_big_numbers() {
        let q = r##"sort_by_label(
        count_values("#xxx", (alias(772424014, "first"), alias(772424230, "second"))),
        "xxx"
        )"##;
        let mut r1 = make_result(&[1, 1, 1, 1, 1, 1]);
        r1.metric_name.set_tag("xxx", "772424014");

        let mut r2 = make_result(&[1, 1, 1, 1, 1, 1]);
        r2.metric_name.set_tag("xxx", "772424230");

        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn count_values() {
        let q = r##"count_values("#xxx", label_set(10, "foo", "bar") or label_set(time()/100, "foo", "bar", "baz", "xx"))"##;
        let mut r1 = make_result([2, 1, 1, 1, 1, 1]);
        r1.metric_name.set_tag("xxx", "10");

        let mut r2 = make_result([nan, 1, nan, nan, nan, nan]);
        r2.metric_name.set_tag("xxx", "12");

        let mut r3 = make_result([nan, nan, 1, nan, nan, nan]);
        r3.metric_name.set_tag("xxx", "14");

        let mut r4 = make_result([nan, nan, nan, 1, nan, nan]);
        r4.metric_name.set_tag("xxx", "16");
        let mut r5 = make_result([nan, nan, nan, nan, 1, nan]);
        r5.metric_name.set_tag("xxx", "18");

        let mut r6 = make_result([nan, nan, nan, nan, nan, 1]);
        r6.metric_name.set_tag("xxx", "20");

        let result_expected: Vec<QueryResult> = vec![r1, r2, r3, r4, r5, r6];
        test_query(q, result_expected)
    }

    #[test]
    fn count_values_by__xxx() {
        let q = r##"count_values("#xxx", label_set(10, "foo", "bar", "xxx", "aaa") or label_set(floor(time()/600), "foo", "bar", "baz", "xx")) by (xxx)"##;
        let mut r1 = make_result(&[1, nan, nan, nan, nan, nan]);
        r1.metric_name.set_tag("xxx", "1");

        let mut r2 = make_result([nan, 1, 1, 1, nan, nan]);
        r2.metric_name.set_tag("xxx", "2");

        let mut r3 = make_result([nan, nan, nan, nan, 1, 1]);
        r3.metric_name.set_tag("xxx", "3");

        let mut r4= make_result(&[1, 1, 1, 1, 1, 1]);
        r4.metric_name.set_tag("xxx", "10");

        // expected sorted output for strings 1, 10, 2, 3
        let result_expected: Vec<QueryResult> = vec![r1, r4, r2, r3];
        test_query(q, result_expected);
    }

    #[test]
    fn count_values__without_baz() {
        let q = r##"count_values("#xxx", label_set(floor(time()/600), "foo", "bar")) without (baz)"##;
        let mut r1 = make_result(&[1, nan, nan, nan, nan, nan]);
        r1.metric_name.set_tag("foo", "bar");
        r1.metric_name.set_tag("xxx", "1");

        let mut r2 = make_result([nan, 1, 1, 1, nan, nan]);
        r2.metric_name.set_tag("foo", "bar");
        r2.metric_name.set_tag("xxx", "2");

        let mut r3 = make_result([nan, nan, nan, nan, 1, 1]);
        r3.metric_name.set_tag("foo", "bar");
        r3.metric_name.set_tag("xxx", "3");
        test_query(q, vec![r1, r2, r3]);
    }

    #[test]
    fn result_sorting() {
        let q = r##"label_set(1, "#instance", "localhost:1001", "type", "free")
        or label_set(1, "instance", "localhost:1001", "type", "buffers")
        or label_set(1, "instance", "localhost:1000", "type", "buffers")
        or label_set(1, "instance", "localhost:1000", "type", "free")
        "##;
        let mut r1 = make_result(&[1, 1, 1, 1, 1, 1]);
        testAddLabels(t, &r1.metric_name,
                      "instance", "localhost:1000", "type", "buffers");
        let mut r2 = make_result(&[1, 1, 1, 1, 1, 1]);
        testAddLabels(t, &r2.metric_name,
                      "instance", "localhost:1000", "type", "free");
        let r3 = make_result(&[1, 1, 1, 1, 1, 1]);
        testAddLabels(t, &r3.metric_name,
                      "instance", "localhost:1001", "type", "buffers");
        let mut r4= make_result(&[1, 1, 1, 1, 1, 1]);
        testAddLabels(t, &r4.metric_name,
                      "instance", "localhost:1001", "type", "free");
        test_query(q, vec![r1, r2, r3, r4]);
    }

    #[test]
    fn sort_by_label_numeric__multiple_labels_only_string() {
        let q = r##"sort_by_label_numeric((
        label_set(1, "#x", "b", "y", "aa"),
        label_set(2, "x", "a", "y", "aa"),
        ), "y", "x")"##;
        let mut r1 = make_result([2, 2, 2, 2, 2, 2]);
        r1.metric_name.set_tag("x", "a");
        r1.metric_name.set_tag("y", "aa");

        let mut r2 = make_result(&[1, 1, 1, 1, 1, 1]);
        r2.metric_name.set_tag("x", "b");
        r2.metric_name.set_tag("y", "aa");

        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn sort_by_label_numeric__multiple_labels_numbers_special_chars() {
        let q = r##"sort_by_label_numeric((
        label_set(1, "#x", "1:0:2", "y", "1:0:1"),
        label_set(2, "x", "1:0:15", "y", "1:0:1"),
        ), "x", "y")"##;
        let mut r1 = make_result(&[1, 1, 1, 1, 1, 1]);
        r1.metric_name.set_tag("x", "1:0:2");
        r1.metric_name.set_tag("y", "1:0:1");

        let mut r2 = make_result([2, 2, 2, 2, 2, 2]);
        r2.metric_name.set_tag("x", "1:0:15");
        r2.metric_name.set_tag("y", "1:0:1");

        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn sort_by_label_numeric_desc__multiple_labels_numbers_special_chars() {
        let q = r##"sort_by_label_numeric_desc((
        label_set(1, "#x", "1:0:2", "y", "1:0:1"),
        label_set(2, "x", "1:0:15", "y", "1:0:1"),
        ), "x", "y")"##;
        let mut r1 = make_result([2, 2, 2, 2, 2, 2]);
        r1.metric_name.set_tag("x", "1:0:15");
        r1.metric_name.set_tag("y", "1:0:1");

        let mut r2 = make_result(&[1, 1, 1, 1, 1, 1]);
        r2.metric_name.set_tag("x", "1:0:2");
        r2.metric_name.set_tag("y", "1:0:1");

        test_query(q, vec![r1, r2]);
    }

    #[test]
    fn sort_by_label_numeric__alias_numbers_with_special_chars() {
        let q = r##"sort_by_label_numeric((
        label_set(4, "#a", "DS50:1/0/15"),
        label_set(1, "a", "DS50:1/0/0"),
        label_set(2, "a", "DS50:1/0/1"),
        label_set(3, "a", "DS50:1/0/2"),
        ), "a")"##;
        let mut r1 = make_result(&[1, 1, 1, 1, 1, 1]);
        r1.metric_name.set_tag("a", "DS50:1/0/0");

        let mut r2 = make_result([2, 2, 2, 2, 2, 2]);
        r2.metric_name.set_tag("a", "DS50:1/0/1");

        let mut r3 = make_result([3, 3, 3, 3, 3, 3]);
        r3.metric_name.set_tag("a", "DS50:1/0/2");

        let mut r4 = make_result([4, 4, 4, 4, 4, 4]);
        r4.metric_name.set_tag("a", "DS50:1/0/15");

        test_query(q, vec![r1, r2, r3, r4])
    }

    #[test]
    fn test_exec_error() {
        fn f(q: &str) {
            let mut ec = EvalConfig::new(1000, 2000, 100);
            ec.max_points_per_series = 10000;
            ec.max_series = 1000;
            let context = Context::default();
            for i in 0 .. 4 {
                let rv = exec(&context, &mut ec, q, false);
                asseert_eq!(rv.is_err(), true, "expecting exec error: {}", q);
                let rv = exec(&context, &mut ec, q, true);
                asseert_eq!(rv.is_err(), true, "expecting exec error: {}", q);
            }
        }

// Empty expr
        f("");
        f("    ");

// Invalid expr
        f("1-");

// Non-existing func
        f("nonexisting()");

// Invalid number of args
        f("range_quantile()");
        f("range_quantile(1, 2, 3)");
        f("range_median()");
        f("abs()");
        f("abs(1,2)");
        f("absent(1, 2)");
        f("clamp()");
        f("clamp_max()");
        f("clamp_min(1,2,3)");
        f("hour(1,2)");
        f("label_join()");
        f("label_replace(1)");
        f("label_transform(1)");
        f("label_set()");
        f(r#"label_set(1, "foo")"#);
        f("label_map()");
        f("label_map(1)");
        f("label_del()");
        f("label_keep()");
        f("label_match()");
        f("label_mismatch()");
        f("label_graphite_group()");
        f("round()");
        f("round(1,2,3)");
        f("sgn()");
        f("scalar()");
        f("sort(1,2)");
        f("sort_desc()");
        f("sort_by_label()");
        f("sort_by_label_desc()");
        f("sort_by_label_numeric()");
        f("sort_by_label_numeric_desc()");
        f("timestamp()");
        f("timestamp_with_name()");
        f("vector()");
        f("histogram_quantile()");
        f("histogram_quantiles()");
        f("sum()");
        f("count_values()");
        f("quantile()");
        f("any()");
        f("group()");
        f("topk()");
        f("topk_min()");
        f("topk_max()");
        f("topk_avg()");
        f("topk_median()");
        f("topk_last()");
        f("limitk()");
        f("bottomk()");
        f("bottomk_min()");
        f("bottomk_max()");
        f("bottomk_avg()");
        f("bottomk_median()");
        f("bottomk_last()");
        f("time(123)");
        f("start(1)");
        f("end(1)");
        f("step(1)");
        f("running_sum(1, 2)");
        f("range_sum(1, 2)");
        f("range_first(1,  2)");
        f("range_last(1, 2)");
        f("smooth_exponential()");
        f("smooth_exponential(1)");
        f("remove_resets()");
        f("sin()");
        f("sinh()");
        f("cos()");
        f("cosh()");
        f("asin()");
        f("asinh()");
        f("acos()");
        f("acosh()");
        f("rand(123, 456)");
        f("rand_normal(123, 456)");
        f("rand_exponential(122, 456)");
        f("pi(123)");
        f("now(123)");
        f("label_copy()");
        f("label_move()");
        f("median_over_time()");
        f("median()");
        f("keep_last_value()");
        f("keep_next_value()");
        f("interpolate()");
        f("distinct_over_time()");
        f("distinct()");
        f("alias()");
        f("alias(1)");
        f(r##"alias(1, "#foo", "bar")"##);
        f("lifetime()");
        f("lag()");
        f("aggr_over_time()");
        f("aggr_over_time(foo)");
        f(r#"aggr_over_time("foo", bar, 1)"#);
        f("sum(aggr_over_time())");
        f("sum(aggr_over_time(foo))");
        f(r#"count(aggr_over_time("foo", bar, 1))"#);
        f("hoeffding_bound_lower()");
        f("hoeffding_bound_lower(1)");
        f("hoeffding_bound_lower(0.99, foo, 1)");
        f("hoeffding_bound_upper()");
        f("hoeffding_bound_upper(1)");
        f("hoeffding_bound_upper(0.99, foo, 1)");
        f("mad()");
        f("outliers_mad()");
        f("outliers_mad(1)");
        f("outliersk()");
        f("outliersk(1)");
        f("mode_over_time()");
        f("rate_over_sum()");
        f("zscore_over_time()");
        f("mode()");
        f("zscore()");
        f("prometheus_buckets()");
        f("buckets_limit()");
        f("buckets_limit(1)");
        f("duration_over_time()");
        f("share_le_over_time()");
        f("share_gt_over_time()");
        f("count_le_over_time()");
        f("count_gt_over_time()");
        f("count_eq_over_time()");
        f("count_ne_over_time()");
        f("timezone_offset()");
        f("bitmap_and()");
        f("bitmap_or()");
        f("bitmap_xor()");
        f("quantiles()");
        f("limit_offset()");
        f("increase()");
        f("increase_prometheus()");
        f("changes()");
        f("changes_prometheus()");
        f("delta()");
        f("delta_prometheus()");

// Invalid argument type
        f("median_over_time({}, 2)");
        f(r##"smooth_exponential(1, 1 or label_set(2, "#x", "y"))"##);
        f("count_values(1, 2)");
        f(r##"count_values(1 or label_set(2, "#xx", "yy"), 2)"##);
        f(r##"quantile(1 or label_set(2, "#xx", "foo"), 1)"##);
        f(r##"clamp_max(1, 1 or label_set(2, "#xx", "foo"))"##);
        f(r##"clamp_min(1, 1 or label_set(2, "#xx", "foo"))"##);
        f(r##"topk(label_set(2, "#xx", "foo") or 1, 12)"##);
        f(r##"topk_avg(label_set(2, "#xx", "foo") or 1, 12)"##);
        f(r##"limitk(label_set(2, "#xx", "foo") or 1, 12)"##);
        f(r##"limit_offet((alias(1,"#foo"),alias(2,"bar")), 2, 10)"##);
        f(r##"limit_offet(1, (alias(1,"#foo"),alias(2,"bar")), 10)"##);
        f(r##"round(1, 1 or label_set(2, "#xx", "foo"))"##);
        f(r##"histogram_quantile(1 or label_set(2, "#xx", "foo"), 1)"##);
        f(r##"histogram_quantiles("#foo", 1 or label_set(2, "xxx", "foo"), 2)"##);
        f("sort_by_label_numeric(1, 2)");
        f("label_set(1, 2, 3)");
        f(r##"label_set(1, "#foo", (label_set(1, "foo", bar") or label_set(2, "xxx", "yy")))"##);
        f(r#"label_set(1, "foo", 3)"#);
        f("label_del(1, 2)");
        f("label_copy(1, 2)");
        f("label_move(1, 2, 3)");
        f(r#"label_move(1, "foo", 3)"#);
        f("label_keep(1, 2)");
        f("label_join(1, 2, 3)");
        f(r#"label_join(1, "foo", 2)"#);
        f(r#"label_join(1, "foo", "bar", 2)"#);
        f("label_replace(1, 2, 3, 4, 5)");
        f(r#"label_replace(1, "foo", 3, 4, 5)"#);
        f(r#"label_replace(1, "foo", "bar", 4, 5)"#);
        f(r#"label_replace(1, "foo", "bar", "baz", 5)"#);
        f(r#"label_replace(1, "foo", "bar", "baz", "invalid(regexp")"#);
        f("label_transform(1, 2, 3, 4)");
        f(r#"label_transform(1, "foo", 3, 4)"#);
        f(r#"label_transform(1, "foo", "bar", 4)"#);
        f(r#"label_transform(1, "foo", "invalid(regexp", "baz""#);
        f("label_match(1, 2, 3)");
        f("label_mismatch(1, 2, 3)");
        f("label_uppercase()");
        f("label_lowercase()");
        f("alias(1, 2)");
        f("aggr_over_time(1, 2)");
        f(r#"aggr_over_time(("foo", "bar"), 3)"#);
        f(r#"outliersk((label_set(1, "foo", "bar"), label_set(2, "x", "y")), 123)"#);

// Duplicate timeseries
        f(r#"(label_set(1, "foo", "bar") or label_set(2, "foo", "baz"))
+ on(xx)
(label_set(1, "foo", "bar") or label_set(2, "foo", "baz"))"#);

// Invalid binary op groupings
        f(r#"1 + group_left() (label_set(1, "foo", bar"), label_set(2, "foo", "baz"))"#);
        f(r#"1 + on() group_left() (label_set(1, "foo", bar"), label_set(2, "foo", "baz"))"#);
        f(r#"1 + on(a) group_left(b) (label_set(1, "foo", bar"), label_set(2, "foo", "baz"))"#);
        f(r#"label_set(1, "foo", "bar") + on(foo) group_left() (label_set(1, "foo", "bar", "a", "b"), label_set(1, "foo", "bar", "a", "c"))"#);
        f(r#"(label_set(1, "foo", bar"), label_set(2, "foo", "baz")) + group_right 1"#);
        f(r#"(label_set(1, "foo", bar"), label_set(2, "foo", "baz")) + on() group_right 1"#);
        f(r##"(label_set(1, "#foo", bar"), label_set(2, "foo", "baz")) + on(a) group_right(b,c) 1"##);
        f(r#"(label_set(1, "foo", bar"), label_set(2, "foo", "baz")) + on() 1"#);
        f(r#"(label_set(1, "foo", "bar", "a", "b"), label_set(1, "foo", "bar", "a", "c")) + on(foo) group_right() label_set(1, "foo", "bar")"#);
        f(r#"1 + on() (label_set(1, "foo", bar"), label_set(2, "foo", "baz"))"#);

// duplicate metrics after binary op
        f(r#"(
label_set(time(), "__name__", "foo", "a", "x"),
label_set(time()+200, "__name__", "bar", "a", "x"),
) > bool 1300"#);
        f(r#"(
label_set(time(), "__name__", "foo", "a", "x"),
label_set(time()+200, "__name__", "bar", "a", "x"),
) + 10"#);

// Invalid aggregates
        f("sum(1) foo (bar)");
        f("sum foo () (bar)");
        f("sum(foo) by (1)");
        f(r##"count(foo) without ("#bar")"##);

// With expressions
        f("ttf()");
        f("ttf(1, 2)");
        f("ru()");
        f("ru(1)");
        f("ru(1,3,3)")
    }



} // mod tests