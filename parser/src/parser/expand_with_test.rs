#[cfg(test)]
mod tests {
    use crate::parser::expand_with_exprs;

    #[test]
    fn test_expand_with_exprs_success() {
        let f = |q: &str, expected: &str| {
            for _ in 0..3 {
                let expanded = expand_with_exprs(q)
                    .unwrap_or_else(|_| panic!("unexpected error when expanding {}", q));
                assert_eq!(
                    expanded, expected,
                    "unexpected expanded expression for {};\ngot\n{}\nwant\n{}",
                    q, expanded, expected
                )
            }
        };

        f("1", "1");
        f("foobar", "foobar");
        f("with (x = 1) x+x", "1 + 1");
        f("with (f(x) = x*x) 3+f(2)+2", "9")
    }

    #[test]
    fn test_expand_with_exprs_error() {
        let f = |q: &str| {
            for _ in 0..3 {
                let expanded = expand_with_exprs(q)
                    .unwrap_or_else(|_| panic!("unexpected error when expanding {}", q));
                if !expanded.is_empty() {
                    panic!("unexpected non-empty expanded={}", expanded)
                }
            }
        };

        f("");
        f("  with (");
    }
}
