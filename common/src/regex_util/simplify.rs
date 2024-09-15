use super::hir_utils::{build_hir, hir_to_string, is_empty_regexp, is_literal, literal_to_string};
use regex::{Error as RegexError, Regex};
use regex_syntax::hir::{Hir, HirKind};


/// simplifies the given expr.
///
/// It returns plaintext prefix and the remaining regular expression with dropped '^' and '$' anchors
/// at the beginning and the end of the regular expression.
///
/// The function removes capturing parens from the expr, so it cannot be used when capturing parens
/// are necessary.
pub fn simplify(expr: &str) -> Result<(String, String), RegexError> {
    if expr == ".*" || expr == ".+" || expr == "" {
        return Ok(("".to_string(), expr.to_string()))
    }

    let hir = match build_hir(expr) {
        Ok(hir) => hir,
        Err(_) => {
            return Ok(("".to_string(), "".to_string()))
        }
    };

    let mut sre = simplify_regexp(hir, false)?;

    if is_empty_regexp(&sre) {
        return Ok(("".to_string(), "".to_string()))
    }

    if is_literal(&sre) {
        return Ok((literal_to_string(&sre), "".to_string()))
    }

    let mut prefix: String = "".to_string();
    let mut sre_new: Option<Hir> = None;

    if let HirKind::Concat(concat) = sre.kind() {
        let head = &concat[0];
        let first_literal = is_literal(head);
        if first_literal {
            let lit = literal_to_string(head);
            prefix = lit.clone();
            match concat.len() {
                1 => return Ok((prefix, "".to_string())),
                2 => sre_new = Some( simplify_regexp(concat[1].clone(), true)? ),
                _ => {
                    let sub = Vec::from(&concat[1..]);
                    let temp = Hir::concat(sub);
                    sre_new = Some(temp);
                }
            }
        }
    }

    if sre_new.is_some() {
        sre = sre_new.unwrap();
    }

    if is_empty_regexp(&sre) {
        return Ok((prefix, "".to_string()))
    }

    let mut s = hir_to_string(&sre);

    if let Err(_) = Regex::new(&s) {
        // Cannot compile the regexp. Return it all as prefix.
        return Ok((expr.to_string(), "".to_string()))
    }

    s = s.replace( "(?:)", "");
    s = s.replace( "(?-s:.)", ".");
    s = s.replace("(?-m:$)", "$");
    Ok((prefix, s))
}

fn simplify_regexp(sre: Hir, has_prefix: bool) -> Result<Hir, RegexError> {
    if matches!(sre.kind(), HirKind::Empty) {
        return Ok(sre)
    }
    let mut sre = sre;
    loop {
        let sub = sre.clone();
        let hir_new = simplify_regexp_ext(sub, has_prefix, false);
        if hir_new == sre {
            return Ok(hir_new)
        }

        // build_hir(&s_new)?; // todo: this should panic

        sre = hir_new
    }
}

fn simplify_regexp_ext(sre: Hir, has_prefix: bool, has_suffix: bool) -> Hir {
    use HirKind::*;

    match sre.kind() {
        Alternation(alternate) => {
            // avoid clone if it's all literal
            if alternate.iter().all(|hir| is_literal(hir)) {
                return sre
            }
            let mut sub = Vec::with_capacity(alternate.len());
            for hir in alternate.iter() {
                let simple = simplify_regexp_ext(hir.clone(), has_prefix, has_suffix);
                if !is_empty_regexp(&simple) {
                    sub.push(simple)
                }
            }

            if sub.len() == 1 {
                return sub.remove(0);
            }

            if sub.is_empty() {
                return Hir::empty()
            }

            Hir::alternation(sub)
        }
        Capture(cap) => {
            let sub = simplify_regexp_ext(cap.sub.as_ref().clone(), has_prefix, has_suffix);
            if is_empty_regexp(&sub) {
                return Hir::empty()
            }
            match sub.kind() {
                Concat(concat) => {
                    if concat.len() == 1 {
                        return concat[0].clone();
                    }
                }
                _ => {}
            }
            sub.clone()
        }
        Concat(concat) => {
            let mut sub = Vec::with_capacity(concat.len());
            for hir in concat.iter() {
                let simple = simplify_regexp_ext(hir.clone(), has_prefix, has_suffix);
                if !is_empty_regexp(&simple) {
                    sub.push(simple)
                }
            }

            if sub.len() == 1 {
                return sub.remove(0);
            }

            if sub.is_empty() {
                return Hir::empty()
            }

            Hir::concat(sub)
        }
        _=> {
            sre
        }
    }
}


#[cfg(test)]
mod test {
    use crate::regex_util::simplify::simplify;

    #[test]
    fn test_simplify() {
        fn check(s: &str, expected_prefix: &str, expected_suffix: &str) {
            let (prefix, suffix) = simplify(s).unwrap();
            assert_eq!(
                prefix, expected_prefix,
                "unexpected prefix for s={s}; got {prefix}; want {expected_prefix}");
            assert_eq!(
                suffix, expected_suffix,
                "unexpected suffix for s={s}; got {suffix}; want {expected_suffix}");
        }

        check("", "", "");
 //       check("^", "", "");
        check("$", "", "");
        check("^()$", "", "");
        check("^(?:)$", "", "");
        check("^foo|^bar$|baz", "", "foo|ba[rz]");
        check("^(foo$|^bar)$", "", "foo|bar");
        check("^a(foo$|bar)$", "a", "foo|bar");
        check("^a(^foo|bar$)z$", "a", "(?:\\Afoo|bar$)z");
        check("foobar", "foobar", "");
        check("foo$|^foobar", "foo", "|bar");
        check("^(foo$|^foobar)$", "foo", "|bar");
        check("foobar|foobaz", "fooba", "[rz]");
        check("(fo|(zar|bazz)|x)", "", "fo|zar|bazz|x");
        check("(тестЧЧ|тест)", "тест", "ЧЧ|");
        check("foo(bar|baz|bana)", "fooba", "[rz]|na");
        check("^foobar|foobaz", "fooba", "[rz]");
        check("^foobar|^foobaz$", "fooba", "[rz]");
        check("foobar|foobaz", "fooba", "[rz]");
        check("(?:^foobar|^foobaz)aa.*", "fooba", "[rz]aa.*");
        check("foo[bar]+", "foo", "[a-br]+");
        check("foo[a-z]+", "foo", "[a-z]+");
        check("foo[bar]*", "foo", "[a-br]*");
        check("foo[a-z]*", "foo", "[a-z]*");
        check("foo[x]+", "foo", "x+");
        check("foo[^x]+", "foo", "[^x]+");
        check("foo[x]*", "foo", "x*");
        check("foo[^x]*", "foo", "[^x]*");
        check("foo[x]*bar", "foo", "x*bar");
        check("fo\\Bo[x]*bar?", "fo", "\\Box*bar?");
        check("foo.+bar", "foo", ".+bar");
        check("a(b|c.*).+", "a", "(?:b|c.*).+");
        check("ab|ac", "a", "[b-c]");
        check("(?i)xyz", "", "(?i:XYZ)");
        check("(?i)foo|bar", "", "(?i:FOO)|(?i:BAR)");
        check("(?i)up.+x", "", "(?i:UP).+(?i:X)");
        check("(?smi)xy.*z$", "", "(?i:XY)(?s:.)*(?i:Z)(?m:$)");

        // test invalid regexps
        check("a(", "a(", "");
        check("a[", "a[", "");
        check("a[]", "a[]", "");
        check("a{", "a{", "");
        check("a{}", "a{}", "");
        check("invalid(regexp", "invalid(regexp", "");

        // The transformed regexp mustn't match aba
        check("a?(^ba|c)", "", "a?(?:\\Aba|c)");

        // The transformed regexp mustn't match barx
        check("(foo|bar$)x*", "", "(?:foo|bar$)x*");
    }
}