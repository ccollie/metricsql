#[cfg(test)]
mod tests {
    use crate::ast::{MetricExpr};
    use crate::ast::Expression::MetricExpression;
    use crate::optimizer::{get_common_label_filters, optimize, pushdown_binary_op_filters};
    use crate::parser::parse;

    #[test]
    fn test_pushdown_binary_op_filters() {
        let f = |q: &str, filters: &str, result_expected: &str| {
            let e = parse(q).expect(format!("unexpected error in Parse({})", q).as_str());
            let orig = e.to_string();
            let filters_expr = parse(filters).expect(format!("cannot parse filters {}", filters).as_str());
            match filters_expr {
                MetricExpression(mut me) => {
                    let result_expr = pushdown_binary_op_filters(&e, &mut me.label_filters);
                    let result = result_expr.to_string();
                    assert_eq!(result, result_expected, "unexpected result for pushdown_binary_op_filters({}, {});\ngot\n{}\nwant\n{}", q, filters, result, result_expected);
                    // Verify that the original e didn't change after PushdownBinaryOpFilters() call
                    let s = e.to_string();
                    assert_eq!(s, orig, "the original expression has been changed;\ngot\n{}\nwant\n{}", s, orig)
                },
                _ => {
                    panic!("filters={} must be a metrics expression; got {}", filters, filters_expr)
                }
            }
        };

        f("foo", "{}", "foo");
        f("foo", r#"{a="b"}"#, r#"foo{a="b"}"#);
        f(r#"foo + bar{x="y"}"#, r#"{c="d",a="b"}"#, r#"foo{a="b", c="d"} + bar{a="b", c="d", x="y"}"#);
        f("sum(x)", r#"{a="b"}"#, "sum(x)");
        f(r#"foo or bar"#, r#"{a="b"}"#, r#"foo{a="b"} or bar{a="b"}"#);
        f(r#"foo or on(x) bar"#, r#"{a="b"}"#, r#"foo or on (x) bar"#);
        f(r#"foo == on(x) group_LEft bar"#, r#"{a="b"}"#, r#"foo == on (x) group_left () bar"#);
        f(r#"foo{x="y"} > ignoRIng(x) group_left(abc) bar"#, r#"{a="b"}"#, r#"foo{a="b", x="y"} > ignoring (x) group_left (abc) bar{a="b"}"#);
        f(r#"foo{x="y"} >bool ignoring(x) group_right(abc,def) bar"#, r#"{a="b"}"#, r#"foo{a="b", x="y"} > bool ignoring (x) group_right (abc, def) bar{a="b"}"#);
        f(r#"foo * ignoring(x) bar"#, r#"{a="b"}"#, r#"foo{a="b"} * ignoring (x) bar{a="b"}"#);
        f(r#"foo{f1!~"x"} UNLEss bar{f2=~"y.+"}"#, r#"{a="b",x=~"y"}"#, r#"foo{a="b", f1!~"x", x=~"y"} unless bar{a="b", f2=~"y.+", x=~"y"}"#);
        f(r#"a / sum(x)"#, r#"{a="b",c=~"foo|bar"}"#, r#"a{a="b", c=~"foo|bar"} / sum(x)"#);
        f(r#"round(rate(x[5m] offset -1h)) + 123 / {a="b"}"#, r#"{x!="y"}"#, r#"round(rate(x{x!="y"}[5m] offset -1h)) + (123 / {a="b", x!="y"})"#);
        f(r#"scalar(foo)+bar"#, r#"{a="b"}"#, r#"scalar(foo) + bar{a="b"}"#);
        f("vector(foo)", r#"{a="b"}"#, "vector(foo)");
        f(r#"{a="b"} + on() group_left() {c="d"}"#, r#"{a="b"}"#, r#"{a="b"} + on () group_left () {c="d"}"#);
    }

    #[test]
    fn test_get_common_label_filters() {
        let f = |q, result_expected: &str| {
            let e = parse(q).expect(format!("unexpected error in Parse({})", q).as_str());
            let lfs = get_common_label_filters(&e);
            let me = MetricExpr::with_filters(lfs);
            let result = me.to_string();
            assert_eq!(result, result_expected,
                       "unexpected result for getCommonLabelFilters({});\ngot\n{}\nwant\n{}", q, result, result_expected);
        };
        f("{}", "{}");
        f("foo", "{}");
        f(r#"{__name__="foo"}"#, "{}");
        f(r#"{__name__=~"bar"}"#, "{}");
        f(r#"{__name__=~"a|b",x="y"}"#, r#"{x="y"}"#);
        f(r#"foo{c!="d",a="b"}"#, r#"{c!="d", a="b"}"#);
        f(r#"1+foo"#, "{}");
        f(r#"foo + bar{a="b"}"#, r#"{a="b"}"#);
        f(r#"foo + bar / baz{a="b"}"#, r#"{a="b"}"#);
        f(r#"foo{x!="y"} + bar / baz{a="b"}"#, r#"{x!="y", a="b"}"#);
        f(r#"foo{x!="y"} + bar{x=~"a|b",q!~"we|rt"} / baz{a="b"}"#, r#"{x!="y", x=~"a|b", q!~"we|rt", a="b"}"#);
        f(r#"{a="b"} + on() {c="d"}"#, "{}");
        f(r#"{a="b"} + on() group_left() {c="d"}"#, r#"{a="b"}"#);
        f(r#"{a="b"} + on(a) group_left() {c="d"}"#, r#"{a="b"}"#);
        f(r#"{a="b"} + on(c) group_left() {c="d"}"#, r#"{a="b", c="d"}"#);
        f(r#"{a="b"} + on(a,c) group_left() {c="d"}"#, r#"{a="b", c="d"}"#);
        f(r#"{a="b"} + on(d) group_left() {c="d"}"#, r#"{a="b"}"#);
        f(r#"{a="b"} + on() group_right(s) {c="d"}"#, r#"{c="d"}"#);
        f(r#"{a="b"} + On(a) groUp_right() {c="d"}"#, r#"{a="b", c="d"}"#);
        f(r#"{a="b"} + on(c) group_right() {c="d"}"#, r#"{c="d"}"#);
        f(r#"{a="b"} + on(a,c) group_right() {c="d"}"#, r#"{a="b", c="d"}"#);
        f(r#"{a="b"} + on(d) group_right() {c="d"}"#, r#"{c="d"}"#);
        f(r#"{a="b"} or {c="d"}"#, "{}");
        f(r#"{a="b",x="y"} or {x="y",c="d"}"#, r#"{x="y"}"#);
        f(r#"{a="b",x="y"} Or on() {x="y",c="d"}"#, "{}");
        f(r#"{a="b",x="y"} Or on(a) {x="y",c="d"}"#, "{}");
        f(r#"{a="b",x="y"} Or on(x) {x="y",c="d"}"#, r#"{x="y"}"#);
        f(r#"{a="b",x="y"} Or oN(x,y) {x="y",c="d"}"#, r#"{x="y"}"#);
        f(r#"{a="b",x="y"} Or on(y) {x="y",c="d"}"#, "{}");
        f(r#"(foo{a="b"} + bar{c="d"}) or (baz{x="y"} <= x{a="b"})"#, r#"{a="b"}"#);
        f(r#"{a="b"} unless {c="d"}"#, r#"{a="b"}"#);
        f(r#"{a="b"} unless on() {c="d"}"#, "{}");
        f(r#"{a="b"} unLess on(a) {c="d"}"#, r#"{a="b"}"#);
        f(r#"{a="b"} unLEss on(c) {c="d"}"#, "{}");
        f(r#"{a="b"} unless on(a,c) {c="d"}"#, r#"{a="b"}"#);
        f(r#"{a="b"} Unless on(x) {c="d"}"#, "{}");
    }

    #[test]
    fn test_optimize() {
        let f = |q, optimized_expected: &str| {
            let e = parse(q).expect(format!("unexpected error in parse({})", q).as_str());
            let orig = e.to_string();
            let e_optimized = optimize(&e);
            let q_optimized = e_optimized.to_string();
            assert_eq!(q_optimized, optimized_expected, "unexpected q_optimized;\ngot\n{}\nwant\n{}",
                       q_optimized, optimized_expected);
            // Make sure the the original e didn't change after Optimize() call
            let s = e.to_string();
            assert_eq!(s, orig, "the original expression has been changed;\ngot\n{}\nwant\n{}", s, orig);
        };

        f("foo", "foo");

        // common binary expressions
        f("a + b", "a + b");
        f(r#"foo{label1="value1"} == bar"#,  r#"foo{label1="value1"} == bar{label1="value1"}"#);
        f(r#"foo{label1="value1"} == bar{label2="value2"}"#,  r#"foo{label1="value1", label2="value2"} == bar{label1="value1", label2="value2"}"#);
        f(r#"foo + bar{b=~"a.*", a!="ss"}"#,  r#"foo{a!="ss", b=~"a.*"} + bar{a!="ss", b=~"a.*"}"#);
        f(r#"foo{bar="1"} / 234"#,  r#"foo{bar="1"} / 234"#);
        f(r#"foo{bar="1"} / foo{bar="1"}"#,  r#"foo{bar="1"} / foo{bar="1"}"#);
        f(r#"123 + foo{bar!~"xx"}"#,  r#"123 + foo{bar!~"xx"}"#);
        f(r#"foo or bar{x="y"}"#,  r#"foo or bar{x="y"}"#);
        f(r#"foo{x="y"} * on() baz{a="b"}"#,  r#"foo{x="y"} * on () baz{a="b"}"#);
        f(r#"foo{x="y"} * on(a) baz{a="b"}"#,  r#"foo{a="b", x="y"} * on (a) baz{a="b"}"#);
        f(r#"foo{x="y"} * on(bar) baz{a="b"}"#,  r#"foo{x="y"} * on (bar) baz{a="b"}"#);
        f(r#"foo{x="y"} * on(x,a,bar) baz{a="b"}"#,  r#"foo{a="b", x="y"} * on (x, a, bar) baz{a="b", x="y"}"#);
        f(r#"foo{x="y"} * ignoring() baz{a="b"}"#,  r#"foo{a="b", x="y"} * ignoring () baz{a="b", x="y"}"#);
        f(r#"foo{x="y"} * ignoring(a) baz{a="b"}"#,  r#"foo{x="y"} * ignoring (a) baz{a="b", x="y"}"#);
        f(r#"foo{x="y"} * ignoring(bar) baz{a="b"}"#,  r#"foo{a="b", x="y"} * ignoring (bar) baz{a="b", x="y"}"#);
        f(r#"foo{x="y"} * ignoring(x,a,bar) baz{a="b"}"#,  r#"foo{x="y"} * ignoring (x, a, bar) baz{a="b"}"#);
        f(r#"foo{x="y"} * ignoring() group_left(foo,bar) baz{a="b"}"#,  r#"foo{a="b", x="y"} * ignoring () group_left (foo, bar) baz{a="b", x="y"}"#);
        f(r#"foo{x="y"} * on(a) group_left baz{a="b"}"#,  r#"foo{a="b", x="y"} * on (a) group_left () baz{a="b"}"#);
        f(r#"foo{x="y"} * on(a) group_right(x, y) baz{a="b"}"#,  r#"foo{a="b", x="y"} * on (a) group_right (x, y) baz{a="b"}"#);
        f(r#"f(foo, bar{baz=~"sdf"} + aa{baz=~"axx", aa="b"})"#,  r#"f(foo, bar{aa="b", baz=~"axx", baz=~"sdf"} + aa{aa="b", baz=~"axx", baz=~"sdf"})"#);
        f(r#"sum(foo, bar{baz=~"sdf"} + aa{baz=~"axx", aa="b"})"#,  r#"sum(foo, bar{aa="b", baz=~"axx", baz=~"sdf"} + aa{aa="b", baz=~"axx", baz=~"sdf"})"#);
        f(r#"foo AND bar{baz="aa"}"#,  r#"foo{baz="aa"} and bar{baz="aa"}"#);
        f(r#"{x="y",__name__="a"} + {a="b"}"#,  r#"a{a="b", x="y"} + {a="b", x="y"}"#);
        f(r#"{x="y",__name__=~"a|b"} + {a="b"}"#,  r#"{__name__=~"a|b", a="b", x="y"} + {a="b", x="y"}"#);
        f(r#"a{x="y",__name__=~"a|b"} + {a="b"}"#,  r#"a{__name__=~"a|b", a="b", x="y"} + {a="b", x="y"}"#);
        f(r#"{a="b"} + ({c="d"} * on() group_left() {e="f"})"#,  r#"{a="b", c="d"} + ({c="d"} * on () group_left () {e="f"})"#);
        f(r#"{a="b"} + ({c="d"} * on(a) group_left() {e="f"})"#,  r#"{a="b", c="d"} + ({a="b", c="d"} * on (a) group_left () {a="b", e="f"})"#);
        f(r#"{a="b"} + ({c="d"} * on(c) group_left() {e="f"})"#,  r#"{a="b", c="d"} + ({c="d"} * on (c) group_left () {c="d", e="f"})"#);
        f(r#"{a="b"} + ({c="d"} * on(e) group_left() {e="f"})"#,  r#"{a="b", c="d", e="f"} + ({c="d", e="f"} * on (e) group_left () {e="f"})"#);
        f(r#"{a="b"} + ({c="d"} * on(x) group_left() {e="f"})"#,  r#"{a="b", c="d"} + ({c="d"} * on (x) group_left () {e="f"})"#);
        f(r#"{a="b"} + ({c="d"} * on() group_right() {e="f"})"#,  r#"{a="b", e="f"} + ({c="d"} * on () group_right () {e="f"})"#);
        f(r#"{a="b"} + ({c="d"} * on(a) group_right() {e="f"})"#,  r#"{a="b", e="f"} + ({a="b", c="d"} * on (a) group_right () {a="b", e="f"})"#);
        f(r#"{a="b"} + ({c="d"} * on(c) group_right() {e="f"})"#,  r#"{a="b", c="d", e="f"} + ({c="d"} * on (c) group_right () {c="d", e="f"})"#);
        f(r#"{a="b"} + ({c="d"} * on(e) group_right() {e="f"})"#,  r#"{a="b", e="f"} + ({c="d", e="f"} * on (e) group_right () {e="f"})"#);
        f(r#"{a="b"} + ({c="d"} * on(x) group_right() {e="f"})"#,  r#"{a="b", e="f"} + ({c="d"} * on (x) group_right () {e="f"})"#);

        // specially handled binary expressions
        f(r#"foo{a="b"} or bar{x="y"}"#,  r#"foo{a="b"} or bar{x="y"}"#);
        f(r#"(foo{a="b"} + bar{c="d"}) or (baz{x="y"} <= x{a="b"})"#,  r#"(foo{a="b", c="d"} + bar{a="b", c="d"}) or (baz{a="b", x="y"} <= x{a="b", x="y"})"#);
        f(r#"(foo{a="b"} + bar{c="d"}) or on(x) (baz{x="y"} <= x{a="b"})"#,  r#"(foo{a="b", c="d"} + bar{a="b", c="d"}) or on (x) (baz{a="b", x="y"} <= x{a="b", x="y"})"#);
        f(r#"foo + (bar or baz{a="b"})"#,  r#"foo + (bar or baz{a="b"})"#);
        f(r#"foo + (bar{a="b"} or baz{a="b"})"#,  r#"foo{a="b"} + (bar{a="b"} or baz{a="b"})"#);
        f(r#"foo + (bar{a="b",c="d"} or baz{a="b"})"#,  r#"foo{a="b"} + (bar{a="b", c="d"} or baz{a="b"})"#);
        f(r#"foo{a="b"} + (bar OR baz{x="y"})"#,  r#"foo{a="b"} + (bar{a="b"} or baz{a="b", x="y"})"#);
        f(r#"foo{a="b"} + (bar{x="y",z="456"} OR baz{x="y",z="123"})"#,  r#"foo{a="b", x="y"} + (bar{a="b", x="y", z="456"} or baz{a="b", x="y", z="123"})"#);
        f(r#"foo{a="b"} unless bar{c="d"}"#,  r#"foo{a="b"} unless bar{a="b", c="d"}"#);
        f(r#"foo{a="b"} unless on() bar{c="d"}"#,  r#"foo{a="b"} unless on () bar{c="d"}"#);
        f(r#"foo + (bar{x="y"} unless baz{a="b"})"#,  r#"foo{x="y"} + (bar{x="y"} unless baz{a="b", x="y"})"#);
        f(r#"foo + (bar{x="y"} unless on() baz{a="b"})"#,  r#"foo + (bar{x="y"} unless on () baz{a="b"})"#);
        f(r#"foo{a="b"} + (bar UNLESS baz{x="y"})"#,  r#"foo{a="b"} + (bar{a="b"} unless baz{a="b", x="y"})"#);
        f(r#"foo{a="b"} + (bar{x="y"} unLESS baz)"#,  r#"foo{a="b", x="y"} + (bar{a="b", x="y"} unless baz{a="b", x="y"})"#);

        // aggregate funcs
        f(r#"sum(foo{bar="baz"}) / a{b="c"}"#,  r#"sum(foo{bar="baz"}) / a{b="c"}"#);
        f(r#"sum(foo{bar="baz"}) by () / a{b="c"}"#,  r#"sum(foo{bar="baz"}) by () / a{b="c"}"#);
        f(r#"sum(foo{bar="baz"}) by (bar) / a{b="c"}"#,  r#"sum(foo{bar="baz"}) by (bar) / a{b="c", bar="baz"}"#);
        f(r#"sum(foo{bar="baz"}) by (b) / a{b="c"}"#,  r#"sum(foo{b="c", bar="baz"}) by (b) / a{b="c"}"#);
        f(r#"sum(foo{bar="baz"}) by (x) / a{b="c"}"#,  r#"sum(foo{bar="baz"}) by (x) / a{b="c"}"#);
        f(r#"sum(foo{bar="baz"}) by (bar,b) / a{b="c"}"#,  r#"sum(foo{b="c", bar="baz"}) by (bar, b) / a{b="c", bar="baz"}"#);
        f(r#"sum(foo{bar="baz"}) without () / a{b="c"}"#,  r#"sum(foo{b="c", bar="baz"}) without () / a{b="c", bar="baz"}"#);
        f(r#"sum(foo{bar="baz"}) without (bar) / a{b="c"}"#,  r#"sum(foo{b="c", bar="baz"}) without (bar) / a{b="c"}"#);
        f(r#"sum(foo{bar="baz"}) without (b) / a{b="c"}"#,  r#"sum(foo{bar="baz"}) without (b) / a{b="c", bar="baz"}"#);
        f(r#"sum(foo{bar="baz"}) without (x) / a{b="c"}"#,  r#"sum(foo{b="c", bar="baz"}) without (x) / a{b="c", bar="baz"}"#);
        f(r#"sum(foo{bar="baz"}) without (bar,b) / a{b="c"}"#,  r#"sum(foo{bar="baz"}) without (bar, b) / a{b="c"}"#);
        f(r#"sum(foo, bar) by (a) + baz{a="b"}"#,  r#"sum(foo{a="b"}, bar) by (a) + baz{a="b"}"#);
        f(r#"topk(3, foo) by (baz,x) + bar{baz="a"}"#,  r#"topk(3, foo{baz="a"}) by (baz, x) + bar{baz="a"}"#);
        f(r#"topk(a, foo) without (x,y) + bar{baz="a"}"#,  r#"topk(a, foo{baz="a"}) without (x, y) + bar{baz="a"}"#);
        f(r#"a{b="c"} + quantiles("foo", 0.1, 0.2, bar{x="y"}) by (b, x, y)"#,  r#"a{b="c", x="y"} + quantiles("foo", 0.1, 0.2, bar{b="c", x="y"}) by (b, x, y)"#);
        f(r#"count_values("foo", bar{baz="a"}) by (bar,b) + a{b="c"}"#,  r#"count_values("foo", bar{baz="a"}) by (bar, b) + a{b="c"}"#);

        // unknown func
        f(r#"f(foo) + bar{baz="a"}"#,  r#"f(foo) + bar{baz="a"}"#);
        f(r#"f(a,b,foo{a="b"} / bar) + baz{x="y"}"#,  r#"f(a, b, foo{a="b"} / bar{a="b"}) + baz{x="y"}"#);

        // transform funcs
        f(r#"round(foo{bar="baz"}) + sqrt(a{z=~"c"})"#,  r#"round(foo{bar="baz", z=~"c"}) + sqrt(a{bar="baz", z=~"c"})"#);
        f(r#"foo{bar="baz"} + SQRT(a{z=~"c"})"#,  r#"foo{bar="baz", z=~"c"} + SQRT(a{bar="baz", z=~"c"})"#);
        f(r#"round({__name__="foo"}) + bar"#,  r#"round(foo) + bar"#);
        f(r#"round({__name__=~"foo|bar"}) + baz"#,  r#"round({__name__=~"foo|bar"}) + baz"#);
        f(r#"round({__name__=~"foo|bar",a="b"}) + baz"#,  r#"round({__name__=~"foo|bar", a="b"}) + baz{a="b"}"#);
        f(r#"round({__name__=~"foo|bar",a="b"}) + sqrt(baz)"#,  r#"round({__name__=~"foo|bar", a="b"}) + sqrt(baz{a="b"})"#);
        f(r#"round(foo) + {__name__="bar",x="y"}"#,  r#"round(foo{x="y"}) + bar{x="y"}"#);
        f(r#"absent(foo{bar="baz"}) + sqrt(a{z=~"c"})"#,  r#"absent(foo{bar="baz"}) + sqrt(a{z=~"c"})"#);
        f(r#"ABSENT(foo{bar="baz"}) + sqrt(a{z=~"c"})"#,  r#"ABSENT(foo{bar="baz"}) + sqrt(a{z=~"c"})"#);
        f(r#"label_set(foo{bar="baz"}, "xx", "y") + a{x="y"}"#,  r#"label_set(foo{bar="baz"}, "xx", "y") + a{x="y"}"#);
        f(r#"now() + foo{bar="baz"} + x{y="x"}"#,  r#"(now() + foo{bar="baz", y="x"}) + x{bar="baz", y="x"}"#);
        f(r#"limit_offset(5, 10, {x="y"}) if {a="b"}"#,  r#"limit_offset(5, 10, {a="b", x="y"}) if {a="b", x="y"}"#);
        f(r#"buckets_limit(aa, {x="y"}) if {a="b"}"#,  r#"buckets_limit(aa, {a="b", x="y"}) if {a="b", x="y"}"#);
        f(r#"histogram_quantiles("q", 0.1, 0.9, {x="y"}) - {a="b"}"#,  r#"histogram_quantiles("q", 0.1, 0.9, {a="b", x="y"}) - {a="b", x="y"}"#);
        f(r#"histogram_quantiles("q", 0.1, 0.9, sum(rate({x="y"}[5m])) by (le)) - {a="b"}"#,  r#"histogram_quantiles("q", 0.1, 0.9, sum(rate({x="y"}[5m])) by (le)) - {a="b"}"#);
        f(r#"histogram_quantiles("q", 0.1, 0.9, sum(rate({x="y"}[5m])) by (le,x)) - {a="b"}"#,  r#"histogram_quantiles("q", 0.1, 0.9, sum(rate({x="y"}[5m])) by (le, x)) - {a="b", x="y"}"#);
        f(r#"histogram_quantiles("q", 0.1, 0.9, sum(rate({x="y"}[5m])) by (le,x,a)) - {a="b"}"#,  r#"histogram_quantiles("q", 0.1, 0.9, sum(rate({a="b", x="y"}[5m])) by (le, x, a)) - {a="b", x="y"}"#);
        f(r#"vector(foo) + bar{a="b"}"#,  r#"vector(foo) + bar{a="b"}"#);
        f(r#"vector(foo{x="y"} + a) + bar{a="b"}"#,  r#"vector(foo{x="y"} + a{x="y"}) + bar{a="b"}"#);

        // multilevel transform funcs
        f(r#"round(sqrt(foo)) + bar"#,  r#"round(sqrt(foo)) + bar"#);
        f(r#"round(sqrt(foo)) + bar{b="a"}"#,  r#"round(sqrt(foo{b="a"})) + bar{b="a"}"#);
        f(r#"round(sqrt(foo{a="b"})) + bar{x="y"}"#,  r#"round(sqrt(foo{a="b", x="y"})) + bar{a="b", x="y"}"#);

        // rollup funcs
        f(r#"RATE(foo[5m]) / rate(baz{a="b"}) + increase(x{y="z"} offset 5i)"#,  r#"(RATE(foo{a="b", y="z"}[5m]) / rate(baz{a="b", y="z"})) + increase(x{a="b", y="z"} offset 5i)"#);
        f(r#"sum(rate(foo[5m])) / rate(baz{a="b"})"#,  r#"sum(rate(foo[5m])) / rate(baz{a="b"})"#);
        f(r#"sum(rate(foo[5m])) by (a) / rate(baz{a="b"})"#,  r#"sum(rate(foo{a="b"}[5m])) by (a) / rate(baz{a="b"})"#);
        f(r#"rate({__name__="foo"}) + rate({__name__="bar",x="y"}) - rate({__name__=~"baz"})"#,  r#"(rate(foo{x="y"}) + rate(bar{x="y"})) - rate({__name__=~"baz", x="y"})"#);
        f(r#"rate({__name__=~"foo|bar", x="y"}) + rate(baz)"#,  r#"rate({__name__=~"foo|bar", x="y"}) + rate(baz{x="y"})"#);
        f(r#"absent_over_time(foo{x="y"}[5m]) + bar{a="b"}"#,  r#"absent_over_time(foo{x="y"}[5m]) + bar{a="b"}"#);
        f(r#"{x="y"} + quantile_over_time(0.5, {a="b"})"#,  r#"{a="b", x="y"} + quantile_over_time(0.5, {a="b", x="y"})"#);
        f(r#"quantiles_over_time("quantile", 0.1, 0.9, foo{x="y"}[5m] offset 4h) + bar{a!="b"}"#,  r#"quantiles_over_time("quantile", 0.1, 0.9, foo{a!="b", x="y"}[5m] offset 4h) + bar{a!="b", x="y"}"#);

        // @ modifier
        f(r#"foo @ end() + bar{baz="a"}"#,  r#"foo{baz="a"} @ end() + bar{baz="a"}"#);
        f(r#"sum(foo @ end()) + bar{baz="a"}"#,  r#"sum(foo @ end()) + bar{baz="a"}"#);
        f(r#"foo @ (bar{a="b"} + baz{x="y"})"#,  r#"foo @ (bar{a="b", x="y"} + baz{a="b", x="y"})"#);

        // subqueries
        f(r#"rate(avg_over_time(foo[5m:])) + bar{baz="a"}"#,  r#"rate(avg_over_time(foo{baz="a"}[5m:])) + bar{baz="a"}"#);
        f(r#"rate(sum(foo[5m:])) + bar{baz="a"}"#,  r#"rate(sum(foo[5m:])) + bar{baz="a"}"#);
        f(r#"rate(sum(foo[5m:]) by (baz)) + bar{baz="a"}"#,  r#"rate(sum(foo{baz="a"}[5m:]) by (baz)) + bar{baz="a"}"#);

        // binary ops with constants or scalars
        f(r#"100 * foo / bar{baz="a"}"#,  r#"(100 * foo{baz="a"}) / bar{baz="a"}"#);
        f(r#"foo * 100 / bar{baz="a"}"#,  r#"(foo{baz="a"} * 100) / bar{baz="a"}"#);
        f(r#"foo / bar{baz="a"} * 100"#,  r#"(foo{baz="a"} / bar{baz="a"}) * 100"#);
        f(r#"scalar(x) * foo / bar{baz="a"}"#,  r#"(scalar(x) * foo{baz="a"}) / bar{baz="a"}"#);
        f(r#"SCALAR(x) * foo / bar{baz="a"}"#,  r#"(SCALAR(x) * foo{baz="a"}) / bar{baz="a"}"#);
        f(r#"100 * on(foo) bar{baz="z"} + a"#,  r#"(100 * on (foo) bar{baz="z"}) + a"#);
    }
}
