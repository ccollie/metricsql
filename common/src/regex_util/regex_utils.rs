use super::match_handlers::StringMatchHandler;
use crate::bytes_util::FastRegexMatcher;
use regex::{Error as RegexError, Regex};
use regex_syntax::hir::Class::{Bytes, Unicode};
use regex_syntax::hir::{Capture, Class, Hir, HirKind, Literal, Look};
use regex_syntax::{escape as escape_regex, parse as parse_regex};
use std::borrow::Cow;

// Beyond this, it's better to use regexp.
const MAX_OR_VALUES: usize = 16;

/// remove_start_end_anchors removes '^' at the start of expr and '$' at the end of the expr.
pub fn remove_start_end_anchors(expr: &str) -> &str {
    let mut cursor = expr;
    while let Some(t) = cursor.strip_prefix('^') {
        cursor = t;
    }
    while cursor.ends_with("$") && !cursor.ends_with("\\$") {
        if let Some(t) = cursor.strip_suffix("$") {
            cursor = t;
        } else {
            break;
        }
    }
    cursor
}

/// get_or_values returns "or" values from the given regexp expr.
///
/// It ignores start and end anchors ('^') and ('$') at the start and the end of expr.
/// It returns ["foo", "bar"] for "foo|bar" regexp.
/// It returns ["foo"] for "foo" regexp.
/// It returns [""] for "" regexp.
/// It returns an empty list if it is impossible to extract "or" values from the regexp.
pub fn get_or_values(expr: &str) -> Vec<String> {
    if expr.is_empty() {
        return vec!["".to_string()];
    }
    // cheap check
    if !expr.contains('|') {
        return vec![expr.to_string()];
    }
    let expr = remove_start_end_anchors(expr);
    let simplified = simplify(expr);
    if simplified.is_err() {
        // Cannot simplify the regexp. Use regexp for matching.
        return vec![expr.to_string()];
    }

    let (prefix, tail_expr) = simplified.unwrap();
    if tail_expr.is_empty() {
        return vec![prefix];
    }
    let sre = build_hir(&tail_expr);
    match sre {
        Ok(sre) => {
            let mut or_values = get_or_values_ext(&sre).unwrap_or_default();
            // Sort or_values for faster index seek later
            or_values.sort();
            if !prefix.is_empty() {
                // Add prefix to or_values
                for or_value in or_values.iter_mut() {
                    *or_value = format!("{prefix}{or_value}")
                }
            }
            or_values
        }
        Err(err) => {
            panic!(
                "BUG: unexpected error when parsing verified tail_expr={tail_expr}: {:?}",
                err
            )
        }
    }
}

pub fn get_match_func_for_or_suffixes(or_values: Vec<String>) -> StringMatchHandler {
    if or_values.len() == 1 {
        let mut or_values = or_values;
        let v = or_values.remove(0);
        StringMatchHandler::Literal(v)
    } else {
        // aho-corasick ?
        StringMatchHandler::Alternates(or_values, false)
    }
}

fn get_or_values_ext(sre: &Hir) -> Option<Vec<String>> {
    use HirKind::*;
    match sre.kind() {
        Empty => Some(vec!["".to_string()]),
        Capture(cap) => get_or_values_ext(cap.sub.as_ref()),
        Literal(literal) => {
            if let Ok(s) = String::from_utf8(literal.0.to_vec()) {
                Some(vec![s])
            } else {
                None
            }
        }
        Alternation(alt) => {
            let mut a = Vec::with_capacity(alt.len());
            for sub in alt.iter() {
                let ca = get_or_values_ext(sub).unwrap_or_default();
                if ca.is_empty() {
                    return None;
                }
                a.extend(ca);
                if a.len() > MAX_OR_VALUES {
                    // It is cheaper to use regexp here.
                    return None;
                }
            }
            Some(a)
        }
        Concat(concat) => {
            if concat.is_empty() {
                return Some(vec!["".to_string()]);
            }
            let prefixes = get_or_values_ext(&concat[0]).unwrap_or_default();
            if prefixes.is_empty() {
                return None;
            }
            if concat.len() == 1 {
                return Some(prefixes);
            }
            let subs = Vec::from(&concat[1..]);
            let concat = Hir::concat(subs);
            let suffixes = get_or_values_ext(&concat).unwrap_or_default();
            if suffixes.is_empty() {
                return None;
            }
            let capacity = prefixes.len() * suffixes.len();
            if capacity > MAX_OR_VALUES {
                // It is cheaper to use regexp here.
                return None;
            }
            let mut a = Vec::with_capacity(capacity);
            for prefix in prefixes.iter() {
                for suffix in suffixes.iter() {
                    a.push(format!("{prefix}{suffix}"));
                }
            }
            Some(a)
        }
        Class(class) => {
            if let Some(literal) = class.literal() {
                return if let Ok(s) = String::from_utf8(literal.to_vec()) {
                    Some(vec![s])
                } else {
                    None
                };
            }

            let mut a = Vec::with_capacity(32);
            match class {
                Unicode(uni) => {
                    for urange in uni.iter() {
                        let start = urange.start();
                        let end = urange.end();
                        for c in start..=end {
                            a.push(format!("{c}"));
                            if a.len() > MAX_OR_VALUES {
                                // It is cheaper to use regexp here.
                                return None;
                            }
                        }
                    }
                    Some(a)
                }
                Bytes(bytes) => {
                    for range in bytes.iter() {
                        let start = range.start();
                        let end = range.end();
                        for c in start..=end {
                            a.push(format!("{c}"));
                            if a.len() > MAX_OR_VALUES {
                                // It is cheaper to use regexp here.
                                return None;
                            }
                        }
                    }
                    Some(a)
                }
            }
        }
        Repetition(rep) => {
            // note that the only case that makes sense in this context is when min == 0 and max == 1
            if rep.min > MAX_OR_VALUES as u32 {
                return None;
            }
            None
        }
        _ => None,
    }
}

fn is_literal(sre: &Hir) -> bool {
    match sre.kind() {
        HirKind::Literal(_) => true,
        HirKind::Capture(cap) => is_literal(cap.sub.as_ref()),
        _ => false,
    }
}

/// simplifies the given expr.
///
/// It returns plaintext prefix and the remaining regular expression with dropped '^' and '$' anchors
/// at the beginning and the end of the regular expression.
///
/// The function removes capturing parens from the expr, so it cannot be used when capturing parens
/// are necessary.
pub fn simplify(expr: &str) -> Result<(String, String), RegexError> {
    if expr == ".*" || expr == ".+" {
        return Ok(("".to_string(), expr.to_string()));
    }

    let hir = match build_hir(expr) {
        Ok(hir) => hir,
        // Cannot parse the regexp. Return it all as prefix.
        Err(_) => return Ok((expr.to_string(), "".to_string())),
    };

    let mut sre = simplify_regexp(hir, false)?;

    if is_empty_regexp(&sre) {
        return Ok(("".to_string(), "".to_string()));
    }

    if let Some(literal) = get_literal(&sre) {
        return Ok((literal, "".to_string()));
    }

    let mut prefix: String = "".to_string();
    let mut sre_new: Option<Hir> = None;

    if let HirKind::Concat(concat) = sre.kind() {

        // Drop .* at the end.
        let mut index = concat.len() - 1;
        while index > 0 {
            if !is_dot_star(&concat[index]) {
                break;
            }
            index -= 1;
        }

        let mut subs = Vec::from(&concat[..=index]);
        if prefix == "" {
            // Drop .* at the start
            while !subs.is_empty() && is_dot_star(&subs[0]) {
                subs.remove(0);
            }
        }

        if !subs.is_empty() {
            let head = &subs[0];
            if let Some(literal) = get_literal(head) {
                prefix = literal;
                match subs.len() {
                    1 => return Ok((prefix, "".to_string())),
                    2 => sre_new = Some(subs[1].clone()),
                    _ => {
                        // todo: do we need to simplify ?
                        subs.remove(0);
                        let temp = simplify_regexp(Hir::concat(subs), true)?;
                        sre_new = Some(temp);
                    }
                }
            }
        }
    }

    if sre_new.is_some() {
        sre = sre_new.unwrap();
    }

    if is_empty_regexp(&sre) {
        return Ok((prefix, "".to_string()));
    }

    let mut s = hir_to_string(&sre);

    if !is_valid_regexp(&s) {
        // Cannot compile the regexp. Return it all as prefix.
        return Ok((expr.to_string(), "".to_string()));
    }

    s = s.replace("(?:)", "")
        .replace("(?-s:.)", ".")
        .replace("(?-m:$)", "$");

    Ok((prefix, s))
}

fn is_valid_regexp(expr: &str) -> bool {
    if expr == ".*" || expr == ".+" {
        return true;
    }
    parse_regex(expr).is_ok()
}

fn simplify_regexp(sre: Hir, has_prefix: bool) -> Result<Hir, RegexError> {
    if matches!(sre.kind(), HirKind::Empty) {
        return Ok(sre);
    }
    let mut sre = sre;
    loop {
        let hir_new = simplify_regexp_ext(&sre, has_prefix, false);
        if hir_new == sre {
            return Ok(hir_new);
        }

        // build_hir(&s_new)?; // todo: this should panic
        sre = hir_new
    }
}

fn simplify_regexp_ext(sre: &Hir, has_prefix: bool, has_suffix: bool) -> Hir {
    let simplify_vec = |v: &Vec<Hir>, remove_empty: bool| -> Vec<Hir> {
        let mut sub = Vec::with_capacity(v.len());
        for hir in v.iter() {
            let simple = simplify_regexp_ext(hir, has_prefix, has_suffix);
            if !is_empty_regexp(&simple) || !remove_empty {
                sub.push(simple)
            }
        }
        sub
    };

    match sre.kind() {
        HirKind::Look(Look::Start) | HirKind::Look(Look::End) => Hir::empty(),
        HirKind::Alternation(alternate) => {
            // avoid clone if it's all literal
            if alternate.iter().all(is_literal) {
                return sre.clone();
            }
            // Do not remove empty captures from Alternation, since this may break regexp.
            Hir::alternation(simplify_vec(alternate, false))
        }
        HirKind::Capture(cap) => {
            let sub = simplify_regexp_ext(cap.sub.as_ref(), has_prefix, has_suffix);
            if is_empty_regexp(&sub) {
                return Hir::empty();
            }
            if let HirKind::Concat(concat) = sub.kind() {
                if concat.len() == 1 {
                    return simplify_regexp_ext(&concat[0], has_prefix, has_suffix);
                }
                return Hir::concat(simplify_vec(concat, true));
            }
            sub.clone()
        }
        HirKind::Concat(concat) => {
            let mut values = Vec::with_capacity(concat.len());
            for (i, hir) in concat.iter().enumerate() {
                let simple = simplify_regexp_ext(
                    hir,
                    has_prefix || !values.is_empty(),
                    (i + 1) < concat.len(),
                );
                if !is_empty_regexp(&simple) {
                    if let Some(prev_hir) = values.last_mut() {
                        if let Some(simplified) = coalesce_alternation(prev_hir, &simple) {
                            let alts = simplified
                                .into_iter()
                                .map(|x| Hir::literal(x.into_bytes())).collect();
                            *prev_hir = Hir::alternation(alts);
                            continue;
                        }
                    }
                    values.push(simple)
                }
            }
            if values.is_empty() {
                return Hir::empty();
            }
            // Remove anchors from the beginning and the end of regexp, since they
            // will be added later.
            let mut start: usize = 0;
            if !has_prefix {
                for curr in values.iter() {
                    if matches!(curr.kind(), HirKind::Look(Look::Start)) {
                        start += 1;
                    } else {
                        break;
                    }
                }
                if start > 0 {
                    values.drain(0..start);
                }
            }
            if !has_suffix {
                let mut end = values.len() - 1;
                for curr in values.iter().rev() {
                    if matches!(curr.kind(), HirKind::Look(Look::End)) {
                        end -= 1;
                    } else {
                        break;
                    }
                }
                if end < values.len() - 1 {
                    // todo: truncate() ?
                    values.drain(end + 1..);
                }
            }
            if values.is_empty() {
                return Hir::empty();
            }
            if values.len() == 1 {
                return values.remove(0);
            }

            Hir::concat(values)
        }
        HirKind::Repetition(rep) => {
            let mut repetition = rep.clone();
            let sub = simplify_regexp_ext(rep.sub.as_ref(), has_prefix, has_suffix);
            if is_empty_regexp(&sub) {
                return Hir::empty();
            }
            repetition.sub = Box::new(sub);
            Hir::repetition(repetition)
        }
        _ => sre.clone(),
    }
}

/// These cost values are used for sorting tag filters in ascending order or the required CPU
/// time for execution.
///
/// These values are obtained from BenchmarkOptimizedRematch_cost benchmark.
pub const FULL_MATCH_COST: usize = 1;
pub const PREFIX_MATCH_COST: usize = 2;
pub const LITERAL_MATCH_COST: usize = 3;
pub const SUFFIX_MATCH_COST: usize = 4;
pub const MIDDLE_MATCH_COST: usize = 6;
pub const RE_MATCH_COST: usize = 100;

pub struct OptimizedMatchFunc {
    pub matcher: StringMatchHandler,
    pub cost: usize,
}

impl OptimizedMatchFunc {
    pub fn new(matcher: StringMatchHandler, cost: usize) -> Self {
        Self {
            matcher,
            cost,
        }
    }

    pub fn matches(&self, s: &str) -> bool {
        self.matcher.matches(s)
    }
}

/// get_optimized_re_match_func tries returning optimized function for matching the given expr.
///
///    '.*'
///    '.+'
///    'literal.*'
///    'literal.+'
///    '.*literal'
///    '.+literal
///    '.*literal.*'
///    '.*literal.+'
///    '.+literal.*'
///    '.+literal.+'
///     'foo|bar|baz|quux'
///     '(foo|bar|baz)quux'
///     'foo(bar|baz)'
///
/// It returns re_match if it cannot find optimized function.
///
/// It also returns literal suffix from the expr.
pub fn get_optimized_re_match_func(
    expr: &str,
) -> Result<OptimizedMatchFunc, RegexError> {

    fn create_re_match_fn(expr: &str) -> Result<OptimizedMatchFunc, RegexError> {
        let re = Regex::new(&expr)?;
        let re_match = StringMatchHandler::fast_regex(re);
        Ok(OptimizedMatchFunc::new(re_match, RE_MATCH_COST))
    }

    if expr.is_empty() {
        return create_re_match_fn(expr);
    }
    if expr == ".*" {
        return Ok(OptimizedMatchFunc::new(StringMatchHandler::MatchAll, FULL_MATCH_COST));
    }
    if expr == ".+" {
        return Ok(
            OptimizedMatchFunc::new(StringMatchHandler::NotEmpty, FULL_MATCH_COST)
        );
    }
    let mut sre = match build_hir(expr) {
        Ok(sre) => sre,
        Err(err) => {
            panic!(
                "BUG: unexpected error when parsing verified expr={expr}: {:?}",
                err
            );
        }
    };

    let mut anchor_start = false;
    let mut anchor_end = false;

    if let HirKind::Concat(subs) = sre.kind() {

        let mut concat = &subs[..];

        if !subs.is_empty() {
            if let HirKind::Look(_) = subs[0].kind() {
                concat = &subs[1..];
                anchor_start = true;
            }
            if let HirKind::Look(_) = subs[subs.len() - 1].kind() {
                concat = &concat[..subs.len() - 1];
                anchor_end = true;
            }
        }

        // Drop .* at the start
        while !concat.is_empty() && is_dot_star(&concat[0]) {
            concat = &concat[1..];
        }

        // Drop .* at the end.
        let mut index = concat.len() - 1;
        while index > 0 {
            if !is_dot_star(&concat[index]) {
                break;
            }
            index -= 1;
        }

        if index < concat.len() - 1 {
            concat = &concat[..=index];
        }

        if concat.len() != subs.len() {
            if concat.len() == 1 {
                sre = concat[0].clone();
            } else {
                sre = Hir::concat(Vec::from(concat));
            }
        }
    }

        // Prepare fast string matcher for re_match.
    if let Some(match_func) = get_optimized_re_match_func_ext(expr, &sre, anchor_start, anchor_end)? {
        // Found optimized function for matching the expr.
        return Ok(match_func);
    }

    // Fall back to re_match_fast.
    create_re_match_fn(expr)
}

fn get_regex_matcher(expr: &str) -> Result<StringMatchHandler, RegexError> {
    let expr_str = format!("^(?:{expr})$");
    let re = Regex::new(&expr_str)?;
    Ok(StringMatchHandler::FastRegex(FastRegexMatcher::new(re)))
}


fn get_optimized_re_match_func_ext(
    expr: &str,
    sre: &Hir,
    anchor_start: bool,
    anchor_end: bool,
) -> Result<Option<OptimizedMatchFunc>, RegexError> {
    if is_dot_star(sre) {
        // '.*'
        return Ok(Some(OptimizedMatchFunc::new(
            StringMatchHandler::MatchAll,
            FULL_MATCH_COST,
        )));
    }
    if is_dot_plus(sre) {
        // '.+'
        return Ok(Some(OptimizedMatchFunc::new(StringMatchHandler::NotEmpty, FULL_MATCH_COST)));
    }

    match sre.kind() {
        HirKind::Alternation(alts) => {
            let all_literal = alts.iter().all(is_literal);
            if all_literal {
                let mut or_values = Vec::with_capacity(alts.len());
                for hir in alts.iter() {
                    let s = literal_to_string(hir);
                    or_values.push(s);
                }
                return Ok(Some(OptimizedMatchFunc::new(
                    StringMatchHandler::Alternates(or_values, false),
                    LITERAL_MATCH_COST * alts.len(),
                )));
            }
        }
        HirKind::Capture(cap) => {
            // Remove parenthesis from expr, i.e. '(expr) -> expr'
            return get_optimized_re_match_func_ext(expr, cap.sub.as_ref(), anchor_start, anchor_end);
        }
        HirKind::Literal(_lit) => {
            if let Some(s) = get_literal(sre) {
                // Literal match
                let matcher = match (anchor_start, anchor_end) {
                    (true, true) => StringMatchHandler::Literal(s),
                    (true, false) => StringMatchHandler::StartsWith(s),
                    (false, true) => StringMatchHandler::EndsWith(s),
                    _ => StringMatchHandler::Contains(s),
                };
                return Ok(Some(
                    OptimizedMatchFunc::new(matcher, LITERAL_MATCH_COST)
                ));
            }
            return Ok(None)
        }
        HirKind::Concat(subs) => {
            if subs.len() == 2 {
                let first = &subs[0];
                let second = &subs[1];

                // handle something like foo(bar|baz)
                if let Some(alts) = coalesce_alternation(first, second) {
                    // 'foo|bar|baz|quux'
                    let len = alts.len();
                    if len > MAX_OR_VALUES {
                        // It is cheaper to use regexp here.
                        return Ok(None);
                    }
                    return Ok(Some(OptimizedMatchFunc::new(
                        StringMatchHandler::Alternates(alts, false),
                        LITERAL_MATCH_COST * len,
                    )));
                }

                if let Some(prefix) = get_literal(first) {
                    if is_dot_star(second) {
                        // 'prefix.*'
                        let matcher = if anchor_start {
                            StringMatchHandler::StartsWith(prefix)
                        } else {
                            StringMatchHandler::Contains(prefix)
                        };
                        return Ok(Some(OptimizedMatchFunc::new(matcher, PREFIX_MATCH_COST)));
                    }
                    if is_dot_plus(second) {
                        // 'prefix.+'
                        let matcher = get_prefix_dot_plus_matcher(prefix, anchor_start);
                        return Ok(Some(OptimizedMatchFunc::new(matcher, PREFIX_MATCH_COST)));
                    }
                }
                if let Some(suffix) = get_literal(second) {
                    if is_dot_star(first) {
                        // '.*suffix'
                        let matcher = get_dot_star_suffix_matcher(suffix, anchor_end);
                        return Ok(Some(OptimizedMatchFunc::new(matcher, SUFFIX_MATCH_COST)));
                    }
                    if is_dot_plus(first) {
                        // '.+suffix'
                        let matcher = get_dot_plus_suffix_matcher(suffix, anchor_end);
                        return Ok(Some(OptimizedMatchFunc::new(matcher, SUFFIX_MATCH_COST)));
                    }
                }
            }
            if subs.len() == 3 && is_literal(&subs[1]) {
                let first = &subs[0];
                let third = &subs[2];
                let middle = literal_to_string(&subs[1]);
                if is_dot_star(first) {
                    if is_dot_star(third) {
                        // '.*middle.*'
                        return Ok(Some(OptimizedMatchFunc::new(
                            StringMatchHandler::Contains(middle),
                            MIDDLE_MATCH_COST,
                        )));
                    }
                    if is_dot_plus(third) {
                        // '.*middle.+'
                        return Ok(Some(OptimizedMatchFunc::new(
                            StringMatchHandler::match_fn(middle, |needle, haystack| {
                                let needle_len = needle.len();
                                if needle_len > haystack.len() {
                                    if let Some(pos) = haystack.find(needle) {
                                        pos + needle_len < haystack.len() - 1
                                    } else {
                                        false
                                    }
                                } else {
                                    false
                                }
                            }),
                            MIDDLE_MATCH_COST,
                        )));
                    }
                }
                if is_dot_plus(first) {
                    if is_dot_star(third) {
                        // '.+middle.*'
                        return Ok(Some(OptimizedMatchFunc::new(
                            StringMatchHandler::match_fn(middle, |needle, haystack| {
                                if haystack.len() > needle.len() {
                                    return if let Some(pos) = haystack.find(needle) {
                                        pos > 0
                                    } else {
                                        false
                                    }
                                }
                                false
                            }),
                            MIDDLE_MATCH_COST,
                        )));
                    }
                    if is_dot_plus(third) {
                        // '.+middle.+'
                        return Ok(Some(OptimizedMatchFunc::new(
                            StringMatchHandler::match_fn(middle, |needle, haystack| {
                                let needle_len = needle.len();
                                if haystack.len() > needle_len + 1 {
                                    if let Some(pos) = haystack.find(needle) {
                                        pos > 0 && pos + needle_len < haystack.len() - 1
                                    } else {
                                        false
                                    }
                                } else {
                                    false
                                }
                            }),
                            MIDDLE_MATCH_COST,
                        )));
                    }
                }
            }

            let re = Regex::new(&expr)?;
            let re_match = StringMatchHandler::fast_regex(re);

            // Verify that the string matches all the literals found in the regexp
            // before applying the regexp.
            // This should optimize the case when the regexp doesn't match the string.
            let mut literals = subs
                .iter()
                .filter(|x| is_literal(x))
                .map(literal_to_string)
                .collect::<Vec<_>>();

            if literals.is_empty() {
                return Ok(Some(OptimizedMatchFunc::new(re_match, RE_MATCH_COST)));
            }

            let suffix: String = if is_literal(&subs[subs.len() - 1]) {
                literals.pop().unwrap_or("".to_string())
            } else {
                "".to_string()
            };

            let first = if literals.len() == 1 {
                let literal = literals.pop().unwrap();
                StringMatchHandler::Contains(literal)
            } else {
                StringMatchHandler::OrderedAlternates(literals)
            };

            return if !suffix.is_empty() {
                let ends_with = StringMatchHandler::match_fn(suffix, |needle, haystack| {
                    !needle.is_empty() && haystack.contains(needle)
                });
                let pred = ends_with.and(first).and(re_match);
                Ok(Some(OptimizedMatchFunc::new(pred, RE_MATCH_COST)))
            } else {
                let pred = first.and(re_match);
                Ok(Some(OptimizedMatchFunc::new(pred, RE_MATCH_COST)))
            }
        }
        _ => {
            // todo!()
            return Ok(None);
        }
    }
    Ok(None)
}

// .*suffix
fn get_dot_star_suffix_matcher(suffix: String, anchor_end: bool) -> StringMatchHandler {
    // start anchor does not matter
    if anchor_end {
        StringMatchHandler::EndsWith(suffix)
    } else {
        StringMatchHandler::Contains(suffix)
    }
}

// handle .+suffix
fn get_dot_plus_suffix_matcher(suffix: String, anchor_end: bool) -> StringMatchHandler {
    // start anchor does not matter
    if anchor_end {
        return StringMatchHandler::match_fn(suffix, |needle, haystack| {
            if haystack.len() > needle.len() {
                if let Some(pos) = haystack.find(needle) {
                    pos > 0 && pos + needle.len() == haystack.len() - 1
                } else {
                    false
                }
            } else {
                false
            }
        });
    }

    StringMatchHandler::match_fn(suffix, |needle, haystack| {
        if haystack.len() > needle.len() {
            let haystack = skip_first_char(haystack);
            haystack.contains(needle)
        } else {
            false
        }
    })
}


// prefix.+
fn get_prefix_dot_plus_matcher(prefix: String, anchor_start: bool) -> StringMatchHandler {
    if anchor_start {
        return StringMatchHandler::match_fn(prefix, |needle, haystack| {
            needle.len() < haystack.len() && haystack.starts_with(needle)
        });
    }
    // dot plus
    StringMatchHandler::match_fn(prefix, |needle, haystack| {
        needle.len() < haystack.len() && haystack.contains(needle)
    })
}



// match foo(bar|baz) / foo(bar|baz).* / foo(bar|baz).+
fn get_literal_alternate_matcher(first: &Hir, second: &Hir, match_end: bool) -> Option<(StringMatchHandler, usize)> {
    // handle something like foo(bar|baz)
    if let Some(alts) = coalesce_alternation(first, second) {
        // 'foo|bar|baz|quux'
        let len = alts.len();
        if len > MAX_OR_VALUES {
            // It is cheaper to use regexp here.
            return None;
        }
        let len = alts.len();
        return Some((StringMatchHandler::Alternates(alts, match_end), LITERAL_MATCH_COST * len))
    }
    None
}

fn hir_to_string(sre: &Hir) -> String {
    match sre.kind() {
        HirKind::Literal(lit) => String::from_utf8(lit.0.to_vec()).unwrap_or_default(),
        HirKind::Concat(concat) => {
            let mut s = String::new();
            for hir in concat.iter() {
                s.push_str(&hir_to_string(hir));
            }
            s
        }
        HirKind::Alternation(alternate) => {
            // avoid extra allocation if it's all literal
            if alternate.iter().all(is_literal) {
                return alternate
                    .iter()
                    .map(hir_to_string)
                    .collect::<Vec<_>>()
                    .join("|");
            }
            let mut s = Vec::with_capacity(alternate.len());
            for hir in alternate.iter() {
                s.push(hir_to_string(hir));
            }
            s.join("|")
        }
        HirKind::Repetition(_repetition) => {
            if is_dot_star(sre) {
                return ".*".to_string();
            } else if is_dot_plus(sre) {
                return ".+".to_string();
            }
            sre.to_string()
        }
        _ => sre.to_string(),
    }
}

fn get_literal(sre: &Hir) -> Option<String> {
    match sre.kind() {
        HirKind::Literal(lit) => {
            let s = String::from_utf8(lit.0.to_vec()).unwrap_or_default();
            Some(s)
        }
        _ => None,
    }
}

fn literal_to_string(sre: &Hir) -> String {
    if let HirKind::Literal(lit) = sre.kind() {
        return String::from_utf8(lit.0.to_vec()).unwrap_or_default();
    }
    "".to_string()
}

pub(super) fn get_prefix_matcher(prefix: &str) -> StringMatchHandler {
    if prefix == ".*" {
        return StringMatchHandler::MatchAll;
    }
    if prefix == ".+" {
        return StringMatchHandler::NotEmpty;
    }
    StringMatchHandler::StartsWith(prefix.to_string())
}

pub fn get_suffix_matcher(suffix: &str) -> Result<StringMatchHandler, RegexError> {
    if !suffix.is_empty() {
        if suffix == ".*" {
            return Ok(StringMatchHandler::MatchAll);
        }
        if suffix == ".+" {
            return Ok(StringMatchHandler::NotEmpty);
        }
        if escape_regex(suffix) == suffix {
            // Fast path - literal suffix such as 'foo'
            return Ok(StringMatchHandler::Literal(suffix.to_string()));
        }
        let or_values = get_or_values(suffix);
        if !or_values.is_empty() {
            // Fast path - pr contains only alternate strings such as 'foo|bar|baz'
            return Ok(StringMatchHandler::Alternates(or_values, false));
        }
    }

    // It is expected that optimize returns valid regexp in suffix, so raise error if not.
    // Anchor suffix to the beginning and the end of the matching string.
    get_regex_matcher(suffix)
}

fn is_empty_regexp(sre: &Hir) -> bool {
    matches!(sre.kind(), HirKind::Empty)
}

fn is_dot_star(sre: &Hir) -> bool {
    match sre.kind() {
        HirKind::Capture(cap) => is_dot_star(cap.sub.as_ref()),
        HirKind::Alternation(alternate) => alternate.iter().any(is_dot_star),
        HirKind::Repetition(repetition) => {
            if let HirKind::Class(clazz) = repetition.sub.kind() {
                repetition.min == 0
                    && repetition.max.is_none()
                    && repetition.greedy
                    && is_empty_class(clazz)
            } else {
                false
            }
        }
        _ => false,
    }
}

fn is_dot_plus(sre: &Hir) -> bool {
    match sre.kind() {
        HirKind::Capture(cap) => is_dot_plus(cap.sub.as_ref()),
        HirKind::Repetition(repetition) => {
            if let HirKind::Class(clazz) = repetition.sub.kind() {
                repetition.min == 1
                    && repetition.max.is_none()
                    && repetition.greedy
                    && is_empty_class(clazz)
            } else {
                false
            }
        }
        _ => false,
    }
}

fn is_empty_class(class: &Class) -> bool {
    if class.is_empty() {
        return true;
    }
    match class {
        Unicode(uni) => {
            let ranges = uni.ranges();
            if ranges.len() == 2 {
                let first = ranges.first().unwrap();
                let last = ranges.last().unwrap();
                if first.start() == '\0' && last.end() == '\u{10ffff}' {
                    return true;
                }
            }
        }
        Bytes(bytes) => {
            let ranges = bytes.ranges();
            if ranges.len() == 2 {
                let first = ranges.first().unwrap();
                let last = ranges.last().unwrap();
                if first.start() == 0 && last.end() == 255 {
                    return true;
                }
            }
        }
    }
    false
}

fn get_captured_alternates(v: &Hir) -> Option<Vec<&str>> {
    if let HirKind::Capture(cap, ..) = v.kind() {
        let Capture { sub, .. } = cap;
        if let HirKind::Alternation(alters) = sub.kind() {
            let mut literals = Vec::with_capacity(alters.len());
            for hir in alters {
                if let HirKind::Literal(l) = hir.kind() {
                    if let Some(safe_literal) = str_from_literal(l) {
                        literals.push(safe_literal);
                    } else {
                        return None;
                    }
                }
            }

            return Some(literals);
        } else if let HirKind::Literal(l) = sub.kind() {
            if let Some(safe_literal) = str_from_literal(l) {
                return Some(vec![safe_literal]);
            }
            return None;
        }
    }
    None
}

/// returns a str represented by `Literal` if it contains a valid utf8
fn str_from_literal(l: &Literal) -> Option<&str> {
    // if not utf8, no good
    let s = std::str::from_utf8(&l.0).ok()?;

    Some(s)
}

/// removes start and end anchors.
fn remove_anchors(v: &Vec<Hir>) -> Cow<'_, Vec<Hir>> {
    if v.len() < 2
        || !matches!(
            (v.first().unwrap().kind(), v.last().unwrap().kind()),
            (&HirKind::Look(Look::Start), &HirKind::Look(Look::End))
        )
    {
        return Cow::Borrowed(v);
    }

    if v.len() == 2 {
        return Cow::Owned(Vec::new());
    }

    let arr = v[1..v.len() - 1].to_vec();
    Cow::Owned(arr)
}

// todo: COW
// todo: needs tests
fn coalesce_alternation(first: &Hir, second: &Hir) -> Option<Vec<String>> {
    fn build_alternation(alts: Vec<&str>, prefix: Option<String>, suffix: Option<String>) -> Vec<String> {
        // todo: could be more efficient
        let mut new_alts = Vec::with_capacity(alts.len());
        let suffix = suffix.unwrap_or_default();
        let prefix = prefix.unwrap_or_default();
        let min_len = prefix.len() + suffix.len();
        for alt in alts {
            let mut new_alt = String::with_capacity(min_len);
            if !prefix.is_empty() {
                new_alt.push_str(&prefix);
            }
            new_alt.push_str(alt);
            if !suffix.is_empty() {
                new_alt.push_str(&suffix);
            }
            new_alts.push(new_alt);
        }
        new_alts
    }

    fn reduce_prefixed(alts: Vec<&str>, prefix: String) -> Option<Vec<String>> {
        if alts.len() <= MAX_OR_VALUES {
            return Some(build_alternation(alts, Some(prefix), None));
        }
        None
    }

    fn reduce_suffixed(alts: Vec<&str>, suffix: String) -> Option<Vec<String>> {
        if alts.len() <= MAX_OR_VALUES {
            return Some(build_alternation(alts, None, Some(suffix)));
        }
        None
    }

    match (first.kind(), second.kind()) {
        (HirKind::Literal(_), HirKind::Literal(_)) => {
            // NOTE, This should not happen (the regex crate automatically coalesces literals)
            return Some(vec![hir_to_string(first), hir_to_string(second)]);
        }
        (HirKind::Literal(_), HirKind::Capture(_)) => {
            // we possibly have something like 'foo(bar|baz)'. Convert to Alternation
            // (foobar | foobaz)
            // get alternates from capture, if applicable
            if let Some(alts) = get_captured_alternates(second) {
                return reduce_prefixed(alts, literal_to_string(first));
            }
        }
        (HirKind::Capture(_), HirKind::Literal(_)) => {
            // we possibly have something like '(bar|baz)foo'. Convert to Alternation
            // (barfoo | bazfoo)
            // get alternates from capture, if applicable
            if let Some(alts) = get_captured_alternates(first) {
                return reduce_suffixed(alts, literal_to_string(second));
            }
        }
        (HirKind::Capture(_), HirKind::Capture(_)) => {
            // we possibly have something like '(bar|baz)(foo|qux)'. Convert to Alternation
            // (barfoo | barqux | bazfoo | bazqux)
            // get alternates from capture, if applicable
            if let (Some(left_alts), Some(right_alts)) = (
                get_captured_alternates(first),
                get_captured_alternates(second),
            ) {
                let size = left_alts.len() * right_alts.len();
                if size <= MAX_OR_VALUES {
                    let mut alts = Vec::with_capacity(size);
                    for left_alt in left_alts {
                        for right_alt in right_alts.iter() {
                            let mut alt = String::new();
                            alt.push_str(left_alt);
                            alt.push_str(right_alt);
                            alts.push(alt);
                        }
                    }
                    return Some(alts);
                }
            }
        }
        _ => {}
    }

    None
}

fn build_hir(pattern: &str) -> Result<Hir, RegexError> {
    parse_regex(pattern).map_err(|err| RegexError::Syntax(err.to_string()))
}

pub(super) fn skip_first_char(s: &str) -> &str {
    let mut chars = s.chars();
    chars.next();
    chars.as_str()
}

pub(super) fn skip_last_char(s: &str) -> &str {
    match s.char_indices().next_back() {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

pub(super) fn skip_first_and_last_char(value: &str) -> &str {
    let mut chars = value.chars();
    chars.next();
    chars.next_back();
    chars.as_str()
}

#[cfg(test)]
mod test {
    use super::{get_or_values, remove_start_end_anchors, simplify};
    use crate::prelude::{get_optimized_re_match_func, OptimizedMatchFunc};

    #[test]
    fn test_is_dot_star() {
        fn check(s: &str, expected: bool) {
            let sre = super::build_hir(s).unwrap();
            let got = super::is_dot_star(&sre);
            assert_eq!(
                got, expected,
                "unexpected is_dot_star for s={:?}; got {:?}; want {:?}",
                s, got, expected
            );
        }

        check(".*", true);
        check(".+", false);
        check("foo.*", false);
        check(".*foo", false);
        check("foo.*bar", false);
        check(".*foo.*", false);
        check(".*foo.*bar", false);
        check(".*foo.*bar.*", false);
        check(".*foo.*bar.*baz", false);
        check(".*foo.*bar.*baz.*", false);
        check(".*foo.*bar.*baz.*qux.*", false);
        check(".*foo.*bar.*baz.*qux.*quux.*quuz.*corge.*grault", false);
        check(".*foo.*bar.*baz.*qux.*quux.*quuz.*corge.*grault.*", false);
    }

    #[test]
    fn test_is_dot_plus() {
        fn check(s: &str, expected: bool) {
            let sre = super::build_hir(s).unwrap();
            let got = super::is_dot_plus(&sre);
            assert_eq!(
                got, expected,
                "unexpected is_dot_plus for s={:?}; got {:?}; want {:?}",
                s, got, expected
            );
        }

        check(".*", false);
        check(".+", true);
        check("foo.*", false);
        check(".*foo", false);
        check("foo.*bar", false);
        check(".*foo.*", false);
        check(".*foo.*bar", false);
        check(".*foo.*bar.*", false);
        check(".*foo.*bar.*baz.*qux", false);
        check(".*foo.*bar.*baz.*qux.*", false);
        check(".*foo.*bar.*baz.*qux.*quux.*quuz.*corge.*grault", false);
        check(".*foo.*bar.*baz.*qux.*quux.*quuz.*corge.*grault.*", false);
    }

    #[test]
    fn test_get_or_values() {
        fn check(s: &str, values_expected: Vec<&str>) {
            let values = get_or_values(s);
            assert_eq!(
                values, values_expected,
                "unexpected values for s={:?}; got {:?}; want {:?}",
                s, values, values_expected
            )
        }

        check("", vec![""]);
        check("foo", vec!["foo"]);
        check("^foo$", vec!["foo"]);
        check("|foo", vec!["", "foo"]);
        check("|foo|", vec!["", "", "foo"]);
        check("foo.+", vec![]);
        check("foo.*", vec![]);
        check(".*", vec![]);
        check("foo|.*", vec![]);
        check("(fo((o)))|(bar)", vec!["bar", "foo"]);
        check("foobar", vec!["foobar"]);
        check("z|x|c", vec!["c", "x", "z"]);
        check("foo|bar", vec!["bar", "foo"]);
        check("(foo|bar)", vec!["bar", "foo"]);
        check("(foo|bar)baz", vec!["barbaz", "foobaz"]);
        check("[a-z][a-z]", vec![]);
        check("[a-d]", vec!["a", "b", "c", "d"]);
        check("x[a-d]we", vec!["xawe", "xbwe", "xcwe", "xdwe"]);
        check("foo(bar|baz)", vec!["foobar", "foobaz"]);
        check(
            "foo(ba[rz]|(xx|o))",
            vec!["foobar", "foobaz", "fooo", "fooxx"],
        );
        check("foo(bar||baz)", vec!["foo", "foobar", "foobaz"]);
        check("(a|b|c)(d|e|f|0|1|2)(g|h|k|x|y|z)", vec![]);
        //check("(?i)foo", vec![]);
        check(
            "(?i)(foo|bar)",
            vec![
                "BAR", "BAr", "BaR", "Bar", "FOO", "FOo", "FoO", "Foo", "bAR", "bAr", "baR", "bar",
                "fOO", "fOo", "foO", "foo",
            ],
        );
        check("^foo|bar$", vec!["bar", "foo"]);
        check("^(foo|bar)$", vec!["bar", "foo"]);
        check("^a(foo|b(?:a|r))$", vec!["aba", "abr", "afoo"]);
        check("^a(foo$|b(?:a$|r))$", vec!["aba", "abr", "afoo"]);
        //check("^a(^foo|bar$)z$", vec![]);

        check(
            "foo(?:bar|baz)x(qwe|rt)",
            vec!["foobarxqwe", "foobarxrt", "foobazxqwe", "foobazxrt"],
        );
    }

    #[test]
    fn test_simplify() {
        fn check(s: &str, expected_prefix: &str, expected_suffix: &str) {
            let (prefix, suffix) = simplify(s).unwrap();
            assert_eq!(
                prefix, expected_prefix,
                "unexpected prefix for s={s}; got {prefix}; want {expected_prefix}"
            );
            assert_eq!(
                suffix, expected_suffix,
                "unexpected suffix for s={s}; got {suffix}; want {expected_suffix}"
            );
        }

        // check("a(b|c.*).+", "a", "(?:b|c.*).+");

        check("", "", "");
        check("^", "", "");
        check("$", "", "");
        check("^()$", "", "");
        check("^(?:)$", "", "");
        check("^foo|^bar$|baz", "", "foo|bar|baz");
        check("^(foo$|^bar)$", "", "foo|bar");
        check("^a(foo$|bar)$", "a", "foo|bar");
        //  check("^a(^foo|bar$)z$", "a", "(?:\\Afoo|bar$)z");
        check("foobar", "foobar", "");
        // check("foo$|^foobar", "", "|bar");
        // check("^(foo$|^foobar)$", "foo", "|bar");
        check("(fo|(zar|bazz)|x)", "", "fo|zar|bazz|x");
        // check("(тестЧЧ|тест)", "тест", "ЧЧ|");
        check("foo(bar|baz|bana)", "foo", "bar|baz|bana");
        check("^foobar|^foobaz$", "", "foobar|foobaz");
        check("foobar|foobaz", "", "foobar|foobaz");
        // check("(?:^foobar|^foobaz)aa.*", "", "[rz]aa.*");
        check("foo[bar]+", "foo", "[abr]+");
        check("foo[a-z]+", "foo", "[a-z]+");
        check("foo[bar]*", "foo", "[abr]*");
        check("foo[a-z]*", "foo", "[a-z]*");
        check("foo[x]*", "foo", "x*");
        check("foo[x]+", "foo", "x+");

       // check("foo[^x]+", "foo", "[^x]+");
        // check("foo[^x]*", "foo", "[^x]*");
        check("foo[x]*bar", "foo", "x*bar");
        check("fo\\Bo[x]*bar?", "fo", "\\Box*bar?");
        check("foo.+bar", "foo", ".+bar");
        // check("a(b|c.*).+", "a", "(?:b|c.*).+");
        check("ab|ac", "", "ab|ac");
        check("(?i)xyz", "", "[Xx][Yy][Zz]");
        check("(?i)foo|bar", "", "[Ff][Oo][Oo]|[Bb][Aa][Rr]");
        check("(?i)up.+x", "", "[Uu][Pp].+[Xx]");
        //check("(?smi)xy.*z$", "", "(?i:XY)(?s:.)*(?i:Z)(?m:$)");

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
        //check("(foo|bar$)x*", "", "(?:foo|bar$)x*");

        // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/5297
        check(".+;|;.+", "", ".+;|;.+");
        check("^(.+);|;(.+)$", "", ".+;|;.+");
        check("^(.+);$|^;(.+)$", "", ".+;|;.+");
        check(".*;|;.*", "", ".*;|;.*");
        check("^(.*);|;(.*)$", "", ".*;|;.*");
        check("^(.*);$|^;(.*)$", "", ".*;|;.*")
    }

    #[test]
    fn test_remove_start_end_anchors() {
        fn f(s: &str, result_expected: &str) {
            let result = remove_start_end_anchors(s);
            assert_eq!(
                result, result_expected,
                "unexpected result for remove_start_end_anchors({s}); got {result}; want {}",
                result_expected
            );
        }

        f("", "");
        f("a", "a");
        f("^^abc", "abc");
        f("a^b$c", "a^b$c");
        f("$$abc^", "$$abc^");
        f("^abc|de$", "abc|de");
        f("abc\\$", "abc\\$");
        f("^abc\\$$$", "abc\\$");
        f("^a\\$b\\$$", "a\\$b\\$")
    }

    #[test]
    fn test_regex_failure() {
        let s = "a(";
        let got = super::build_hir(s);
        assert!(got.is_err());
    }

    fn test_optimized_regex(expr: &str, s: &str, result_expected: bool) {
        let OptimizedMatchFunc { matcher: match_func, .. } = get_optimized_re_match_func(expr).unwrap();
        let result = match_func.matches(s);
        assert_eq!(
            result, result_expected,
            "unexpected result when matching {s} against regex={expr}; got {result}; want {result_expected}"
        );
    }

    #[test]
    fn test_simple() {
        let expr = ".+";
        let s = "foobaza";
        let result_expected = true;
        test_optimized_regex(expr, s, result_expected);
    }

    #[test]
    fn test_regex_match() {

        fn f(expr: &str, s: &str, result_expected: bool) {
            test_optimized_regex(expr, s, result_expected);
        }

        f("", "", true);
     //   f("", "foo", true);
        f("foo", "", false);
        f(".*", "", true);
        f(".*", "foo", true);
        f(".+", "", false);
        f(".+", "foo", true);
        f("foo.*", "bar", false);
        f("foo.*", "foo", true);
        f("foo.*", "foobar", true);
        f("foo.*", "a foobar", true);
        f("foo.+", "bar", false);
        f("foo.+", "foo", false);
        f("foo.+", "a foo", true);
        f("foo.+", "foobar", true);
        f("foo.+", "a foobar", true);
        f("foo|bar", "", false);
        f("foo|bar", "a", false);
        f("foo|bar", "foo", true);
        f("foo|bar", "foo a", true);
        f("foo|bar", "a foo a", true);
        f("foo|bar", "bar", true);
        f("foo|bar", "foobar", true);
        f("foo(bar|baz)", "a", false);
        f("foo(bar|baz)", "foobar", true);
        f("foo(bar|baz)", "foobaz", true);
        f("foo(bar|baz)", "foobaza", true);
        f("foo(bar|baz)", "a foobaz a", true);
        f("foo(bar|baz)", "foobal", false);
        f("^foo|b(ar)$", "foo", true);
        f("^foo|b(ar)$", "foo a", true);
        f("^foo|b(ar)$", "a foo", false);
        f("^foo|b(ar)$", "bar", true);
        f("^foo|b(ar)$", "a bar", true);
        f("^foo|b(ar)$", "barz", false);
        f("^foo|b(ar)$", "ar", false);
        f(".*foo.*", "foo", true);
        f(".*foo.*", "afoobar", true);
        f(".*foo.*", "abc", false);
        f("foo.*bar.*", "foobar", true);
        f("foo.*bar.*", "foo_bar_", true);
        f("foo.*bar.*", "a foo bar baz", true);
        f("foo.*bar.*", "foobaz", false);
        f("foo.*bar.*", "baz foo", false);
        f(".+foo.+", "foo", false);
        f(".+foo.+", "afoobar", true);
        f(".+foo.+", "afoo", false);
        f(".+foo.+", "abc", false);
        f("foo.+bar.+", "foobar", false);
        f("foo.+bar.+", "foo_bar_", true);
        f("foo.+bar.+", "a foo_bar_", true);
        f("foo.+bar.+", "foobaz", false);
        f("foo.+bar.+", "abc", false);
        f(".+foo.*", "foo", false);
        f(".+foo.*", "afoo", true);
        f(".+foo.*", "afoobar", true);
        f(".*(a|b).*", "a", true);
        f(".*(a|b).*", "ax", true);
        f(".*(a|b).*", "xa", true);
        f(".*(a|b).*", "xay", true);
        f(".*(a|b).*", "xzy", false);
     //   f("^(?:true);$", "true", true);
    //    f("^(?:true);$", "false", false);

        f(".+;|;.+", ";", false);
        f(".+;|;.+", "foo", false);
        f(".+;|;.+", "foo;bar", true);
        f(".+;|;.+", "foo;", true);
        f(".+;|;.+", ";foo", true);
        f(".+foo|bar|baz.+", "foo", false);
        f(".+foo|bar|baz.+", "afoo", true);
        f(".+foo|bar|baz.+", "fooa", false);
        f(".+foo|bar|baz.+", "afooa", true);
        f(".+foo|bar|baz.+", "bar", true);
        f(".+foo|bar|baz.+", "abar", true);
        f(".+foo|bar|baz.+", "abara", true);
        f(".+foo|bar|baz.+", "bara", true);
        f(".+foo|bar|baz.+", "baz", false);
        f(".+foo|bar|baz.+", "baza", true);
        f(".+foo|bar|baz.+", "abaz", false);
        f(".+foo|bar|baz.+", "abaza", true);
        f(".+foo|bar|baz.+", "afoo|bar|baza", true);
        f(".+(foo|bar|baz).+", "bar", false);
        f(".+(foo|bar|baz).+", "bara", false);
        f(".+(foo|bar|baz).+", "abar", false);
        f(".+(foo|bar|baz).+", "abara", true);
        f(".+(foo|bar|baz).+", "afooa", true);
        f(".+(foo|bar|baz).+", "abaza", true);

        f(".*;|;.*", ";", true);
        f(".*;|;.*", "foo", false);
        f(".*;|;.*", "foo;bar", true);
        f(".*;|;.*", "foo;", true);
        f(".*;|;.*", ";foo", true);

        f("^bar", "foobarbaz", false);
        f("^foo", "foobarbaz", true);
        f("bar$", "foobarbaz", false);
        f("baz$", "foobarbaz", true);
        f("(bar$|^foo)", "foobarbaz", true);
        f("(bar$^boo)", "foobarbaz", false);
        f("foo(bar|baz)", "a fooxfoobaz a", true);
        f("foo(bar|baz)", "a fooxfooban a", false);
        f("foo(bar|baz)", "a fooxfooban foobar a", true);
    }

}
