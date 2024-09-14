#[cfg(test)]
mod test {
	use metricsql_runtime::parse_metric_selector;
	use crate::relabel::IfExpression;
	use crate::utils::new_labels_from_string;

	#[test]
	fn test_if_expression_parse_failure() {
		fn f(s: &str) {
			let ie = IfExpression::parse(s);
			assert!(ie.is_err(), "expecting non-nil error when parsing {}", s);
		}

		f("{");
		f("{foo");
		f("foo{");
		f(r#"foo{bar="a" or}"#);
	}

	#[test]
	fn test_if_expression_parse_success() {
		fn check(s: &str) {
			IfExpression::parse(s).unwrap();
		}

		check("foo");
		check(r#"{foo="bar"}"#);
		check(r#"foo{bar=~"baz", x!="y"}"#);
		check(r#"{a="b" or c="d",e="x"}"#);
		check(r#"foo{
        bar="a",x="y" or
        x="a",a="b" or
        a="x"
        }"#);
	}

	#[test]
	fn test_if_expression_marshal_unmarshal_json() {
		fn check(s: &str, json_expected: &str) {
			let ie = IfExpression::parse(s)
				.map_err(|err| format!("cannot parse ifExpression {}: {}", s, err))
				.unwrap();

			let data = match serde_json::to_string(&ie) {
				Ok(data) => data,
				Err(err) => {
					panic!("cannot marshal ifExpression {s}: {:?}", err);
				}
			};

			assert_eq!(data, json_expected,
					   "unexpected value after json marshaling;\ngot\n{}\nwant\n{}", data, json_expected);

			let ie2: IfExpression = match serde_json::from_str(&data) {
				Ok(data) => data,
				Err(err) => {
					panic!("cannot unmarshal ifExpression from json {data}: {:?}", err);
				}
			};

			let data2 = match serde_json::to_string(&ie2) {
				Ok(data) => data,
				Err(err) => {
					panic!("cannot marshal ifExpression {s}: {:?}", err);
				}
			};

			assert_eq!(data2, json_expected,
					   "unexpected data after unmarshal/marshal cycle;\ngot\n{}\nwant\n{}", data2, json_expected);
		}

		check("foo", r#""foo""#);
		check(r#"{foo="bar",baz=~"x.*"}"#, r#""{foo=\"bar\",baz=~\"x.*\"}""#);
		check(r#"{a="b" or c="d",x="z"}"#, r#""{a=\"b\" or c=\"d\",x=\"z\"}""#);
	}


	#[test]
	fn test_if_expression_unmarshal_failure() {
		fn check(s: &str) {
			serde_yaml::from_str(s).expect("expecting non-nil error");
		}

		check("{");
		check("{x:y}");
		check("[1]");
		check(r#""{""#);
		check(r#"'{'"#);
		check("foo{bar");
		check("foo{bar}");
		check("foo{bar=");
		check(r#"foo{bar=""#);
		check(r#"foo{bar='"#);
		check(r#"foo{bar=~"("}"#);
		check(r#"foo{bar!~"("}"#);
		check("foo{bar==aaa}");
		check(r#"foo{bar=="b"}"#);
		check(r#"'foo+bar'"#);
		check(r#"'foo{bar=~"a[b"}'"#);
	}

	#[test]
	fn test_if_expression_unmarshal_success() {
		fn f(s: &str) {
			let ie: IfExpression = serde_yaml::from_str(s).unwrap();
			let mut b = serde_yaml::to_string(&ie).unwrap();
			let b = b.trim();
			assert_eq!(b, s, "unexpected marshaled data;\ngot\n{}\nwant\n{}", b, s);
		}

		f(r#"'{}'"#);
		f("foo");
		f(r#"foo{bar="baz"}"#);
		f(r#"'{a="b", c!="d", e=~"g", h!~"d"}'"#);
		f(r#"foo{bar="zs",a=~"b|c"}"#);
		f(r#"foo{z="y" or bar="zs",a=~"b|c"}"#);
		f(r#"- foo
- bar{baz="abc"}"#)
	}

	#[test]
	fn test_if_expression_match() {
		fn check(if_expr: &str, metric_with_labels: &str) {
			let ie = match serde_yaml::from_str(if_expr) {
				Err(err) => panic!("unexpected error during unmarshal: {:?}", err),
				Ok(val) => val
			};
			let labels = parse_metric_selector(metric_with_labels)
				.unwrap();
			if !ie.Match(labels.GetLabels()) {
				panic!("unexpected mismatch of if_expr={} for {}", if_expr, metric_with_labels)
			}
		}

		check("foo", "foo");
		check("foo", r#"foo{bar="baz",a="b"}"#);
		check(r#"foo{bar="a"}"#, r#"foo{bar="a"}"#);
		check(r#"foo{bar="a" or baz="x"}"#, r#"foo{bar="a"}"#);
		check(r#"foo{baz="x" or bar="a"}"#, r#"foo{bar="a"}"#);
		check(r#"foo{bar="a"}"#, r#"foo{x="y",bar="a",baz="b"}"#);
		check(r#"'{a=~"x|abc",y!="z"}'"#, r#"m{x="aa",a="abc"}"#);
		check(r#"'{a=~"x|abc",y!="z"}'"#, r#"m{x="aa",a="abc",y="qwe"}"#);
		check(r#"'{__name__="foo"}'"#, r#"foo{bar="baz"}"#);
		check(r#"'{__name__=~"foo|bar"}'"#, "bar");
		check(r#"'{__name__!=""}'"#, "foo");
		check(r#"'{__name__!=""}'"#, r#"bar{baz="aa",b="c"}"#);
		check(r#"'{__name__!~"a.+"}'"#, r#"bar{baz="aa",b="c"}"#);
		check(r#"foo{a!~"a.+"}"#, r#"foo{a="baa"}"#);
		check(r#"'{foo=""}'"#, "bar");
		check(r#"'{foo!=""}'"#, r#"aa{foo="b"}"#);
		check(r#"'{foo=~".*"}'"#, "abc");
		check(r#"'{foo=~".*"}'"#, r#"abc{foo="bar"}"#);
		check(r#"'{foo!~".+"}'"#, "abc");
		check(r#"'{foo=~"bar|"}'"#, "abc");
		check(r#"'{foo=~"bar|"}'"#, r#"abc{foo="bar"}"#);
		check(r#"'{foo!~"bar|"}'"#, r#"abc{foo="baz"}"#);
	}

	#[test]
	fn test_if_expression_mismatch() {
		fn check(if_expr: &str, metric_with_labels: &str) {
			let ie: IfExpression = serde_yaml::from_str(if_expr).unwrap();
			let labels = new_labels_from_string(metric_with_labels);
			if ie.is_match(&labels[..]) {
				panic!("unexpected match of if_expr={} for {}", if_expr, metric_with_labels)
			}
		}

		check("foo", "bar");
		check("foo", r#"a{foo="bar"}"#);
		check(r#"foo{bar="a"}"#, "foo");
		check(r#"foo{bar="a" or baz="a"}"#, "foo");
		check(r#"foo{bar="a"}"#, r#"foo{bar="b"}"#);
		check(r#"foo{bar="a"}"#, r#"foo{baz="b",a="b"}"#);
		check(r#"'{a=~"x|abc",y!="z"}'"#, r#"m{x="aa",a="xabc"}"#);
		check(r#"'{a=~"x|abc",y!="z"}'"#, r#"m{x="aa",a="abc",y="z"}"#);
		check(r#"'{__name__!~".+"}'"#, "foo");
		check(r#"'{a!~"a.+"}'"#, r#"foo{a="abc"}"#);
		check(r#"'{foo=""}'"#, r#"bar{foo="aa"}"#);
		check(r#"'{foo!=""}'"#, "aa");
		check(r#"'{foo=~".+"}'"#, "abc");
		check(r#"'{foo!~".+"}'"#, r#"abc{foo="x"}"#);
		check(r#"'{foo=~"bar|"}'"#, r#"abc{foo="baz"}"#);
		check(r#"'{foo!~"bar|"}'"#, "abc");
		check(r#"'{foo!~"bar|"}'"#, r#"abc{foo="bar"}"#);
	}
}