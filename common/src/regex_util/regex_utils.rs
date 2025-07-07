use super::match_handlers::{get_literal_match_fn, StringMatchHandler};
use crate::prelude::{ConsecutiveLiterals, ContainsMultiStringMatcher, EqualMultiStringMatcher, MatchAnyMatcher, RegexMatcher, RepetitionMatcher};
use crate::regex_util::string_pattern::StringPattern;
use crate::regex_util::{LiteralBracketedMatcher, LiteralMapMatcher, MatchFnHandler, Quantifier, StringMatchOptions};
use regex::{Error as RegexError, Regex};
use regex_syntax::hir::Class::{Bytes, Unicode};
use regex_syntax::hir::{Class, Hir, HirKind, Look, Repetition};
use regex_syntax::{hir, parse as parse_regex};

const MAX_SET_MATCHES: usize = 64;
// Beyond this, it's better to use regexp.
const MAX_OR_VALUES: usize = 64;

/// The minimum number of alternate values a regex should have to trigger
/// the optimization done by `optimize_equal_or_prefix_string_matchers()` to use a map
/// to match values instead of iterating over a list.
const MIN_EQUAL_MULTI_STRING_MATCHER_MAP_THRESHOLD: usize = 16;

const META_CHARS: &str = ".^$*+?{}[]|()\\/%~";
pub fn contains_regex_meta_chars(s: &str) -> bool {
    s.chars().any(|c| META_CHARS.contains(c))
}

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

pub fn is_valid_regexp(expr: &str) -> bool {
    if expr == ".*" || expr == ".+" || expr.is_empty() {
        return true;
    }
    parse_regex(expr).is_ok()
}

pub(super) fn build_hir(pattern: &str) -> Result<Hir, RegexError> {
    let hir = parse_regex(pattern)
        .map_err(|err| RegexError::Syntax(err.to_string()))?;
    Ok(clear_capture(hir))
}

/// `string_matcher_from_regex` tries returning an optimized function for matching the given expr.
///
///    - '.*'
///    - '.+'
///    - 'literal.*'
///    - 'literal.+'
///    - '.*literal'
///    - '.+literal
///    - '.*literal.*'
///    - '.*literal.+'
///    - '.+literal.*'
///    - '.+literal.+'
///    - 'foo|bar|baz|quux'
///    - '(foo|bar|baz)quux'
///    - 'foo(bar|baz)'
///    - 'foo(bar|baz)quux'
///
/// It returns re_match if it cannot find an optimized function.
pub fn string_matcher_from_regex(expr: &str) -> Result<StringMatchHandler, RegexError> {
    fn create_re_match_fn(expr: &str, sre: &Hir) -> Result<StringMatchHandler, RegexError> {
        handle_regex(expr, sre)
    }

    match expr.len() {
        0 => return Ok(StringMatchHandler::MatchAny(MatchAnyMatcher::new(true))),
        2 => {
            if expr == ".*" {
                return Ok(StringMatchHandler::any(false));
            }

            if expr == ".+" {
                return Ok(StringMatchHandler::not_empty(true));
            }
        }
        _ => {}
    }

    let sre = build_hir(expr)?;

    // let debug_str = format!("{:?}, {}", sre, hir_to_string(&sre));
    //
    // println!("expr {}", debug_str);

    // Prepare a fast string matcher for re_match.
    if let Some(match_func) = string_matcher_from_regex_internal(expr, &sre)? {
        // Found optimized function for matching the expr.
        return Ok(match_func);
    }

    // Fall back to re_match_fast.
    create_re_match_fn(expr, &sre)
}

pub(super) fn string_matcher_from_regex_internal(
    expr: &str,
    sre: &Hir,
) -> Result<Option<StringMatchHandler>, RegexError> {

    if sre.properties().is_alternation_literal() {
        let mut matches = Vec::new();
        collect_simple_alternates(sre, "", &mut matches);
        if !matches.is_empty() {
            let matcher = StringMatchHandler::literal_alternates(matches, true);
            return Ok(Some(matcher));
        }
    }

    match sre.kind() {
        HirKind::Empty => Ok(Some(StringMatchHandler::Empty)),
        HirKind::Repetition(rep) => Ok(get_repetition_matcher(rep)),
        HirKind::Alternation(alts) => Ok(get_alternation_matcher(expr, alts)?),
        HirKind::Capture(cap) => {
            // Remove parenthesis from expr, i.e. '(expr) -> expr'
            let hir = cap.sub.as_ref();
            string_matcher_from_regex_internal(expr, hir)
        }
        HirKind::Class(_) => {
            if let Some((alternatives, case_sensitive)) = find_set_matches_internal(sre, "") {
                let matcher = StringMatchHandler::literal_alternates(alternatives, case_sensitive);
                Ok(Some(matcher))
            } else {
                Ok(None)
            }
        }
        HirKind::Literal(_lit) => {
            let literal = literal_to_string(sre);
            Ok(Some(StringMatchHandler::equals(literal)))
        }
        HirKind::Concat(subs) => {
            get_concat_matcher(subs, expr)
        },
        HirKind::Look(_look) => {
            // Lookahead and lookbehind are not supported.
            // We cannot optimize these, so we return None.
            Ok(None)
        }
    }
}

fn get_alternation_matcher(expr: &str, hirs: &[Hir]) -> Result<Option<StringMatchHandler>, RegexError> {
    let mut is_all_literal = true;
    let mut num_values: usize = 0;

    let mut matchers = Vec::new();
    let mut matches_case_sensitive: Option<bool> = None;
    let mut is_mismatch = false;

    for sub_hir in hirs {
        if let Some(matcher) = string_matcher_from_regex_internal(expr, sub_hir)? {
            let case_sensitive = matcher.is_case_sensitive();

            if !is_mismatch {
                if let Some(sensitive) = matches_case_sensitive {
                    if sensitive != case_sensitive {
                        is_mismatch = true;
                    }
                } else {
                    matches_case_sensitive = Some(case_sensitive);
                }
            }

            match &matcher {
                StringMatchHandler::ConsecutiveLiterals(lits) => {
                    // if we have a suffix, then we cannot optimize this
                    is_all_literal = is_all_literal && lits.suffix.is_none();
                    num_values += lits.len();
                }
                StringMatchHandler::Literal(_) => {
                    num_values += 1;
                }
                StringMatchHandler::Alternates(values) => {
                    num_values += values.len();
                }
                _ => is_all_literal = false,
            }

            matchers.push(matcher);
        } else {
            return Ok(None);
        }
    }

    matchers.sort_by_key(|matcher| matcher.cost());

    let case_sensitive = matches_case_sensitive.unwrap_or_default();

    // optimize the case where all the alternatives are literals
    if is_all_literal && !is_mismatch {
        if num_values >= MIN_EQUAL_MULTI_STRING_MATCHER_MAP_THRESHOLD {
            let mut res = LiteralMapMatcher::new();
            res.is_case_sensitive = case_sensitive;

            for matcher in matchers.into_iter() {
                match matcher {
                    StringMatchHandler::Literal(lit) => {
                        res.values.insert(lit.into());
                    }
                    StringMatchHandler::Alternates(matcher) => {
                        for value in matcher.values {
                            res.values.insert(value);
                        }
                    }
                    _ => unreachable!("BUG: unexpected matcher (check is_literal)"),
                }
            }
            return Ok(Some(StringMatchHandler::LiteralMap(res)));
        }

        let mut result = EqualMultiStringMatcher::new(case_sensitive, num_values);
        for matcher in matchers.into_iter() {
            match matcher {
                StringMatchHandler::Literal(lit) => {
                    result.push(lit.into());
                }
                StringMatchHandler::Alternates(matcher) => {
                    for value in matcher.values.into_iter() {
                        result.push(value);
                    }
                }
                _ => unreachable!("BUG: unexpected matcher (check is_literal)"),
            }
        }
        return Ok(Some(StringMatchHandler::Alternates(result)));
    }

    Ok(Some(StringMatchHandler::Or(matchers)))
}

fn get_repetition_matcher(rep: &Repetition) -> Option<StringMatchHandler> {
    fn validate_repetition(rep: &Repetition) -> bool {
        // if re.sub.Op != syntax.OpAnyChar && re.sub.Op != syntax.OpAnyCharNotNL {
        //     return nil
        // }
        matches_any_char(&rep.sub) || matches_any_character_except_newline(&rep.sub)
    }

    if let Some(quantifier) = get_repetition_quantifier(rep) {
        let match_nl = matches_newline(&rep.sub);
        return match quantifier {
            Quantifier::ZeroOrOne => {
                // .? or literal.?
                Some(StringMatchHandler::zero_or_one_chars(match_nl))
            }
            Quantifier::ZeroOrMore => {
                if !validate_repetition(rep) {
                    return None;
                }
                // Any string is fine (including an empty one), as far as it doesn't contain any newline.
                Some(StringMatchHandler::any(match_nl))
            }
            Quantifier::OneOrMore => {
                // .+
                Some(StringMatchHandler::not_empty(match_nl))
            }
        }
    }
    
    if is_literal(&rep.sub) {
        let literal = literal_to_string(&rep.sub);
        let repetition = RepetitionMatcher::new(literal, rep.min, rep.max);
        return Some(StringMatchHandler::Repetition(repetition));
    } 
    None
}

fn get_concat_matcher(hirs: &[Hir], expr: &str) -> Result<Option<StringMatchHandler>, RegexError> {
    if hirs.is_empty() {
        return Ok(Some(StringMatchHandler::Empty));
    }

    if hirs.len() == 1 {
        return string_matcher_from_regex_internal(expr, &hirs[0]);
    }

    fn get_set_matches(hirs: &[Hir]) -> Option<(Vec<String>, bool)> {
        if hirs.len() == 1 {
            find_set_matches_internal(&hirs[0], "")
        } else {
            let hir = Hir::concat(hirs.to_vec());
            find_set_matches_internal(&hir, "")
        }
    }

    fn get_regex_matcher(
        expr: &str,
        matcher: StringMatchHandler,
    ) -> Result<StringMatchHandler, RegexError> {
        let regex = Regex::new(&format!("^(?s:{expr})$"))?;

        let matcher = RegexMatcher {
            regex,
            prefix: "".to_string(),
            suffix: "".to_string(),
            contains: Vec::new(),
            string_matcher: Some(Box::new(matcher)),
            set_matches: vec![],
        };

        Ok(StringMatchHandler::Regex(matcher))
    }

    if let Ok(Some(matcher)) = get_simple_concat_matcher(expr, hirs, true) {
        return Ok(Some(matcher));
    }
    
    let set_matches_result: Option<(Vec<String>, bool)> = get_set_matches(hirs);

    // Ensure we've found some literals to match (optionally with a left and/or right matcher).
    // If not, then this optimization doesn't trigger.
    let Some((matches, case_sensitive)) = set_matches_result else {
        // If we don't have any set matches, then we cannot optimize this.
        return Ok(None);
    };
    if matches.is_empty() {
        // If we have no matches, then we cannot optimize this.
        return Ok(None);
    }

    // We found literals in the middle. We can trigger the fast path only if
    // the matches are case-sensitive because ContainsMultiStringMatcher doesn't
    // support case-insensitive.
    if case_sensitive {
        // partial match
        let matcher = ContainsMultiStringMatcher::new(matches, None, None);
        let regex_matcher = get_regex_matcher(expr, StringMatchHandler::ContainsMulti(matcher))?;
        Ok(Some(regex_matcher))
    } else {
        // No left and right matchers (only fixed set matches).
        let matcher = StringMatchHandler::literal_alternates(matches, case_sensitive);
        // partial match, so fallback to regex
        let matcher = get_regex_matcher(expr, matcher)?;
        Ok(Some(matcher))
    }
}

fn get_simple_concat_matcher(expr: &str, hirs: &[Hir], anchored: bool) -> Result<Option<StringMatchHandler>, RegexError> {
    if hirs.is_empty() {
        return Ok(Some(StringMatchHandler::Empty));
    }

    fn get_fn_match_fn(
        literal: String,
        options: &StringMatchOptions,
    ) -> StringMatchHandler {
        let func = get_literal_match_fn(options);
        StringMatchHandler::MatchFn(MatchFnHandler::new(literal, func))
    }

    fn handle_quantifiers(prefix_quantifier: Option<Quantifier>, 
                          suffix_quantifier: Option<Quantifier>, 
                          lit: String, anchored: bool) -> Option<StringMatchHandler> {
        // Special case for 'literal.+/ literal.* / literal.?'
        let options = StringMatchOptions {
            anchor_start: anchored,
            anchor_end: anchored,
            prefix_quantifier,
            suffix_quantifier,
        };
        Some(get_fn_match_fn(lit, &options))
    }
    
    fn handle_prefix(rep: &Repetition, lit: &hir::Literal, anchored: bool) -> Option<StringMatchHandler> {
        let quantifier = get_repetition_quantifier(rep)?;
        handle_quantifiers(None, Some(quantifier), String::from_utf8(lit.0.to_vec()).unwrap_or_default(), anchored)
    }

    fn handle_suffix(rep: &Repetition, lit: &hir::Literal, anchored: bool) -> Option<StringMatchHandler> {
        let quantifier = get_repetition_quantifier(rep)?;
        handle_quantifiers(Some(quantifier), None, String::from_utf8(lit.0.to_vec()).unwrap_or_default(), anchored)
    }

    fn create_contains_matcher(matcher: StringMatchHandler, left: Option<&Repetition>, right: Option<&Repetition>) -> Option<StringMatchHandler> {
        let case_sensitive = matcher.is_case_sensitive();
        // this branch only handles case_sensitive alternations
        if !case_sensitive {
            return None;
        }

        let mut alternates = Vec::new();
        match matcher {
            StringMatchHandler::Alternates(mut alt) => {
                std::mem::swap(&mut alternates, &mut alt.values);
            }
            StringMatchHandler::Literal(p) => {
                alternates.push(p.into());
            }
            StringMatchHandler::LiteralMap(ref map) => {
                alternates.extend(map.values.iter().cloned());
            }
            // todo: handle other matchers. E.g., if case-insensitive, use regex matcher to reuse the work done
            _ => return None,
        }

        let left_quantifier =  if let Some(l) = left {
            get_repetition_quantifier(l)
        } else {
            None
        };

        let right_quantifier =  if let Some(r) = right {
            get_repetition_quantifier(r)
        } else {
            None
        };

        match (left_quantifier, right_quantifier) {
            (Some(Quantifier::ZeroOrOne), Some(Quantifier::ZeroOrOne)) => {
                // If both are zero or one, we can just return the matcher.
                // Cannot optimize. Return regex matcher.
                //Some(matcher)
                None
            }
            (Some(left_quantifier), Some(right_quantifier)) => {
                let left = quantifier_matcher(left_quantifier)
                    .expect("BUG: Invariant failed. Quantifier is None");
                let right = quantifier_matcher(right_quantifier)
                    .expect("BUG: Invariant failed. Quantifier is None");
                let contains_matcher =
                    ContainsMultiStringMatcher::new(alternates, Some(left), Some(right));
                Some(StringMatchHandler::ContainsMulti(contains_matcher))
            }
            (Some(left_quantifier), None) => {
                let left = quantifier_matcher(left_quantifier)
                    .expect("BUG: Invariant failed. Quantifier is None");
                let contains_matcher =
                    ContainsMultiStringMatcher::new(alternates, Some(left), None);
                Some(StringMatchHandler::ContainsMulti(contains_matcher))
            }
            (None, Some(right_quantifier)) => {
                let right = quantifier_matcher(right_quantifier)
                    .expect("BUG: Invariant failed. Quantifier is None");
                let contains_matcher =
                    ContainsMultiStringMatcher::new(alternates, None, Some(right));
                Some(StringMatchHandler::ContainsMulti(contains_matcher))
            }
            _ => None,
        }
    }

    fn handle_coalesce_literals(expr: &str, hirs: &[Hir]) -> Result<Option<StringMatchHandler>, RegexError> {
        // try to handle mixed literals at the left, followed by any other matcher
        let Some((mut matcher, remainder)) = get_consecutive_literals_matcher(hirs)? else {
            return Ok(None);
        };

        if remainder.len() == 1 {
            let Some(right_matcher) = string_matcher_from_regex_internal(expr, &remainder[0])? else {
                return Ok(None);
            };
            match matcher {
                StringMatchHandler::ConsecutiveLiterals(ref mut m) => {
                    // If we have a suffix, then we cannot optimize this.
                    if m.suffix.is_some() {
                        return Ok(None);
                    }
                    m.suffix = Some(Box::new(right_matcher));
                }
                StringMatchHandler::Literal(ref mut m) => {
                    let prefix = m.to_string();
                    let new_matcher = StringMatchHandler::prefix(prefix, Some(right_matcher), matcher.is_case_sensitive());
                    return Ok(Some(new_matcher));
                }
                _ => {}
            }
            return Ok(Some(matcher));
        } else if !remainder.is_empty() {
            // We have a literal in the middle, but it is not the whole middle.
            // We cannot optimize this.
            return Ok(None);
        }

        Ok(Some(matcher))
    }


    match hirs {
        [h1] => string_matcher_from_regex_internal(expr, h1),
        [h1, h2] => {
            match (h1.kind(), h2.kind()) {
                (HirKind::Literal(lit), HirKind::Repetition(rep)) => {
                    // A literal followed by repetition
                    Ok(handle_prefix(rep, lit, anchored))
                }
                (HirKind::Repetition(rep), HirKind::Literal(lit)) => {
                    // repetition followed by literal
                    Ok(handle_suffix(rep, lit, anchored))
                }
                (HirKind::Literal(_), HirKind::Class(_)) => {
                    handle_coalesce_literals(expr, hirs)
                }
                (HirKind::Literal(lit), _) => {
                    let literal = String::from_utf8(lit.0.to_vec()).unwrap_or_default();
                    if let Some(right_matcher) = string_matcher_from_regex_internal(expr, h2)? {
                        // A literal followed by any matcher
                        Ok(Some(StringMatchHandler::prefix(literal, Some(right_matcher), anchored)))
                    } else {
                        // no matcher found
                        Ok(None)
                    }
                }
                (HirKind::Class(_), HirKind::Literal(_)) => {
                    handle_coalesce_literals(expr, hirs)
                }
                (_, HirKind::Literal(lit)) => {
                    let literal = String::from_utf8(lit.0.to_vec()).unwrap_or_default();
                    if let Some(left_matcher) = string_matcher_from_regex_internal(expr, h1)? {
                        // matcher followed by literal
                        Ok(Some(StringMatchHandler::suffix(Some(left_matcher), literal, anchored)))
                    } else { 
                        Ok(None)
                    }
                }
                (HirKind::Repetition(rep), HirKind::Alternation(hirs)) => {
                    let Some(matcher) = get_alternation_matcher(expr, hirs)? else {
                        return Ok(None);
                    };
                    Ok(create_contains_matcher(matcher, Some(rep), None))
                }
                (HirKind::Alternation(alts), HirKind::Repetition(rep)) => {
                    let Some(matcher) = get_alternation_matcher(expr, alts)? else {
                        return Ok(None);
                    };
                    Ok(create_contains_matcher(matcher, None, Some(rep)))
                }
                (HirKind::Class(_), HirKind::Class(_)) => {
                    handle_coalesce_literals(expr, hirs)
                }
                _ => Ok(None)
            }
        },
        [h1, h2, h3] => {
            let left_kind = h1.kind();
            let right_kind = h3.kind();
            let middle_kind = h2.kind();

            match (left_kind, middle_kind, right_kind) {
                // handle cases like start(b|c)end
                (HirKind::Literal(_left), _, HirKind::Literal(_right)) => {
                    let Some(matcher) = string_matcher_from_regex_internal(expr, h2)? else {
                        return Ok(None);
                    };
                    let left_pattern = StringPattern::case_sensitive(literal_to_string(h1));
                    let right_pattern = StringPattern::case_sensitive(literal_to_string(h3));
                    let bracket = LiteralBracketedMatcher::new(left_pattern, matcher, right_pattern);
                    let matcher = StringMatchHandler::Bracketed(Box::new(bracket));
                    Ok(Some(matcher))
                }
                (HirKind::Repetition(left), HirKind::Literal(lit), HirKind::Repetition(right)) => {
                    let left_quantifier = get_repetition_quantifier(left);
                    let right_quantifier = get_repetition_quantifier(right);
                    match (left_quantifier, right_quantifier) {
                        (Some(_), Some(_)) => {
                            let literal = String::from_utf8(lit.0.to_vec()).unwrap_or_default();
                            Ok(handle_quantifiers(left_quantifier, right_quantifier, literal, anchored))
                        }
                        _ => {
                            let literal = String::from_utf8(lit.0.to_vec()).unwrap_or_default();
                            let matcher = StringMatchHandler::literal(literal, true);
                            Ok(create_contains_matcher(matcher, Some(left), Some(right)))
                        },
                    }
                }
                (HirKind::Repetition(left), HirKind::Alternation(_alts), HirKind::Repetition(right)) => {
                    let Some((alternatives, case_sensitive)) = find_set_matches_internal(h2,"") else {
                        return Ok(None);
                    };
                    let Some(left_matcher) = get_repetition_matcher(left) else {
                        return Ok(None);
                    };
                    let Some(right_matcher) = get_repetition_matcher(right) else {
                        return Ok(None);
                    };
                    if case_sensitive {
                        let contains_matcher =
                            ContainsMultiStringMatcher::new(alternatives, Some(left_matcher), Some(right_matcher));
                        return Ok(Some(StringMatchHandler::ContainsMulti(contains_matcher)))
                    }
                    // todo: handle case-insensitive alternations
                   Ok(None)
                }
                (HirKind::Class(_), _, _) => {
                    // handle cases like [aA][bB][cC].+
                    // We can coalesce these literals together.
                    handle_coalesce_literals(expr, hirs)
                }
                (HirKind::Literal(_), HirKind::Class(_), _) => {
                    // handle cases like literal([aA][bB][cC]).+
                    // We can coalesce these literals together.
                    handle_coalesce_literals(expr, hirs)
                }
                // Concatenated variable length selectors are not supported.
                (HirKind::Repetition(_), HirKind::Repetition(_), _) => {
                    Ok(None)
                }
                (_, HirKind::Repetition(_), HirKind::Repetition(_)) => {
                    Ok(None)
                }
                _ => Ok(None)
            }
        },
        [h1, ..] if matches!(h1.kind(), HirKind::Repetition(_)) => {
            let HirKind::Repetition(rep) = h1.kind() else {
                return Ok(None);
            };
            // Something like '.*(?i)(xyz-abc)'

            let Some(left_matcher) = get_repetition_matcher(rep) else {
                return Ok(None);
            };

            let rest = &hirs[1..];

            let Some(mut right) = handle_coalesce_literals(expr, rest)? else {
                return Ok(None);
            };

            if let StringMatchHandler::ConsecutiveLiterals(ref mut matcher) = right {
                // If we have a suffix, then we cannot optimize this.
                if matcher.suffix.is_some() {
                    return Ok(None);
                }
                matcher.prefix = Some(Box::new(left_matcher));
                return Ok(Some(right));
            };

            Ok(None)
        },
        // literal([aA][bB][cC]).+
        all => {
            handle_coalesce_literals(expr, all)
        }
    }
}

fn handle_literal_alternates(sre: &Hir) -> Option<StringMatchHandler> {
    if sre.properties().is_alternation_literal() {
        match sre.kind() {
            HirKind::Literal(_lit) => {
                // If we have a single literal, we can optimize it.
                let literal = literal_to_string(sre);
                Some(StringMatchHandler::literal(literal, true))
            }
            HirKind::Concat(concat) if concat.len() == 1 => {
                // If we have a single literal, we can optimize it.
                let literal = literal_to_string(&concat[0]);
                Some(StringMatchHandler::literal(literal, true))
            }
            HirKind::Alternation(alternatives) => {
                if alternatives.len() > MAX_OR_VALUES {
                    // Too many alternatives, we cannot optimize this.
                    return None;
                }
                let mut values = Vec::with_capacity(alternatives.len());
                let mut case_sensitive = true;

                for alt in alternatives.iter() {
                    if let Some((alts, cs)) = find_set_matches_internal(alt, "") {
                        values.extend(alts);
                        case_sensitive &= cs;
                    } else {
                        return None; // Not a literal alternation
                    }
                }

                Some(StringMatchHandler::literal_alternates(values, case_sensitive))
            }
            _ => None,
        }
    }
    else {
        None
    }
}

fn is_case_insensitive_class(hir: &Hir) -> Option<char> {
    if let HirKind::Class(class) = hir.kind() {
        match class {
            Unicode(ranges) => {
                if let [first, second] = ranges.ranges() {
                    if first.start() == first.end() && second.start() == second.end() {
                        return Some(first.start());
                    }
                }
            }
            Bytes(ranges) => {
                if let [first, second] = ranges.ranges() {
                    if first.start() == first.end() && second.start() == second.end() {
                        return Some(first.start() as char);
                    }
                }
            }
        }
    }

    None
}

// Try to consume consecutive literals. Because of HIR coalescing, this means alternating case-sensitive/insensitive literals
fn get_consecutive_literals_matcher(hirs: &[Hir]) -> Result<Option<(StringMatchHandler, &[Hir])>, RegexError> {
    if hirs.is_empty() {
        return Ok(None);
    }

    let mut temp: Vec<StringPattern> = Vec::new();
    let mut hirs = hirs;
    while let Some((lit, len)) = consume_literal(hirs) {
        temp.push(lit);
        hirs = &hirs[len..];
    }

    if !temp.is_empty() {
        let matcher = ConsecutiveLiterals::new(None, temp, None);
        return Ok(Some(
            (StringMatchHandler::ConsecutiveLiterals(matcher), hirs)
        ))
    }
    Ok(None)
}

fn consume_literal(hirs: &[Hir]) -> Option<(StringPattern, usize)> {
    if hirs.is_empty() {
        return None;
    }

    fn handle_case_folded_string(hirs: &[Hir]) -> Option<(StringPattern, usize)> {
        // Try to consume consecutive literals. Because of HIR coalescing, this means alternating case-sensitive/insensitive literals
        if let Some((lit, len)) = get_case_folded_string(hirs) {
            let pattern = if lit.is_ascii() {
                StringPattern::ascii_case_insensitive(lit)
            } else {
                StringPattern::case_insensitive(lit)
            };
            return Some((pattern, len));
        }
        None
    }

    let first = uncapture(&hirs[0]);
    match first.kind() {
        HirKind::Literal(_) => {
            let literal = literal_to_string(first);
            Some((StringPattern::new(literal, true), 1))
        }
        HirKind::Class(_) => {
            handle_case_folded_string(hirs)
        }
        _=> None
    }
}

/// In HIR, casing is represented by individual Char classes per Unicode case folding. E.g.
/// `a` is represented by `[aA]` and 'A' is represented by `[aA]`. This function returns the coalesced
/// casing for the given string.
fn get_case_folded_string(hirs: &[Hir]) -> Option<(String, usize)> {
    let mut res: String = String::with_capacity(32); // todo: calculate

    let mut count: usize = 0;
    for hir in hirs.iter() {
        if let Some(ch) = is_case_insensitive_class(hir) {
            res.push(ch);
            count += 1;
        } else if let HirKind::Literal(lit) = hir.kind() {
            // We may have character classes followed by a non-alphanumeric literal (e.g. (?i)xyz-123).
            // Here we'll have classes for xyz, then a literal for '-123'.
            // We coalesce these literal and character classes together. Note that we are not being
            // exhaustive here and only handle common cases.

            // we do this only if we've already seen a case-insensitive class
            if !res.is_empty() {
                let is_alphabetic = lit.0.iter().any(|&b| (b as char).is_alphabetic());
                // if it's alphabetic, it is case-sensitive, so break
                if is_alphabetic {
                    break;
                }
                for ch in lit.0.iter().cloned() {
                    let ch = ch as char;
                    res.push(ch);
                }
                count += 1;
            } else {
                break;
            }
        } else {
            break;
        }
    }

    if res.is_empty() {
        return None;
    }

    Some((res, count))
}

fn is_literal(sre: &Hir) -> bool {
    match sre.kind() {
        HirKind::Literal(_) => true,
        HirKind::Capture(cap) => is_literal(cap.sub.as_ref()),
        _ => false,
    }
}

pub(super) fn is_empty_class(class: &Class) -> bool {
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

pub(super) fn matches_any_char(hir: &Hir) -> bool {
    if let HirKind::Class(class) = hir.kind() {
        return is_empty_class(class);
    }
    false
}

#[cfg(feature = "legacy_newline")]
fn matches_newline(_hir: &Hir) -> bool {
    match _hir.kind() {
        HirKind::Literal(lit) => {
            // Check if the literal contains a newline
            lit.0.contains(&b'\n')
        }
        HirKind::Class(class) => {
            match class {
                // Check if the class includes a newline
                Unicode(class) => class.ranges().iter().any(|range| range.contains(&'\n')),
                Bytes(class) => class.ranges().iter().any(|range| range.contains(&b'\n')),
            }
        }
        HirKind::Repetition(repetition) => {
            // Check the sub-expression of repetition
            matches_newline(&repetition.sub)
        }
        _ => false, // Other node types do not match a newline
    }
}

#[cfg(not(feature = "legacy_newline"))]
fn matches_newline(_hir: &Hir) -> bool {
    // Match Prometheus 3.0 and match newlines by default
    true
}

pub(super) fn matches_any_character_except_newline(hir: &Hir) -> bool {
    match hir.kind() {
        HirKind::Literal(lit) => {
            // Check if the literal is not a newline
            !lit.0.contains(&b'\n')
        }
        HirKind::Class(class) => {
            match class {
                // Check if the class does not include a newline
                Unicode(class) => {
                    let nl = '\n';
                    class
                        .ranges()
                        .iter()
                        .all(|range| !(range.start()..range.end()).contains(&nl))
                }
                Bytes(class) => {
                    let nl = b'\n';
                    class
                        .ranges()
                        .iter()
                        .all(|range| !(range.start()..range.end()).contains(&nl))
                }
            }
        }
        HirKind::Repetition(repetition) => {
            // Check the sub-expression of repetition
            matches_any_character_except_newline(&repetition.sub)
        }
        _ => false, // Other node types do not match any character except newlines
    }
}

pub(super) fn is_anchor(sre: &Hir, look: Look) -> bool {
    matches!(sre.kind(), HirKind::Look(l) if look == *l)
}

pub(super) fn is_start_anchor(sre: &Hir) -> bool {
    is_anchor(sre, Look::Start)
}

pub(super) fn is_end_anchor(sre: &Hir) -> bool {
    is_anchor(sre, Look::End)
}

fn uncapture(hir: &Hir) -> &Hir {
    let HirKind::Capture(cap) = hir.kind() else {
        return hir;
    };
    cap.sub.as_ref()
}

fn clear_capture(sre: Hir) -> Hir {
    
    let mut sre = sre;

    fn is_clearable_variant(hir: &Hir) -> bool {
        matches!(hir.kind(), HirKind::Capture(_) | HirKind::Alternation(_) | HirKind::Repetition(_) |
            HirKind::Concat(_))
    }

    fn should_clear(hir: &Hir) -> bool {
        match hir.kind() {
            HirKind::Capture(_) => true,
            HirKind::Alternation(alternate) => {
                alternate.iter().any(should_clear)
            }
            HirKind::Repetition(rep) => should_clear(&rep.sub),
            HirKind::Concat(concat) => {
                concat.iter().any(should_clear)
            }
            _ => false
        }
    }

    fn clear_item(hir: &mut Hir) {
        match hir.kind() {
            HirKind::Capture(cap) => {
                // If it's a capture, we return the sub-expression without the capture.
                let uncap = clear_capture(*cap.sub.clone());
                *hir = uncap;
            }
            HirKind::Alternation(alternate) => {
                if should_clear(hir) {
                    // If the alternation contains captures, we clear them.
                    let mut new_alternate = Vec::with_capacity(alternate.len());
                    for item in alternate.iter() {
                        let copy = item.clone();
                        if is_clearable_variant(item) {
                            new_alternate.push(clear_capture(copy));
                        } else {
                            new_alternate.push(copy);
                        }
                    }
                    *hir = Hir::alternation(new_alternate);
                }
            }
            HirKind::Repetition(rep) => {
                // If it's a repetition, we clear captures in the sub-expression.
                let sub = rep.sub.as_ref();
                if should_clear(sub) {
                    let clear_sub = clear_capture(sub.clone());
                    let mut new_rep = rep.clone();
                    new_rep.sub = Box::new(clear_sub);
                    *hir = Hir::repetition(new_rep);
                }
            }
            HirKind::Concat(concat) => {
                if should_clear(hir) {
                    // If the concat contains captures, we clear them.
                    let mut new_concat = Vec::with_capacity(concat.len());
                    for item in concat.iter() {
                        let copy = item.clone();
                        if is_clearable_variant(item) {
                            new_concat.push(clear_capture(copy));
                        } else {
                            new_concat.push(copy);
                        }
                    }
                    *hir = Hir::concat(new_concat);
                }
            }
            _ => {
                // For other HIR kinds, we leave them unchanged
                // Could potentially handle more cases if needed
            }
        }
    }
    
    clear_item(&mut sre);
    sre
}

pub(super) fn literal_to_string(sre: &Hir) -> String {
    if let HirKind::Literal(lit) = sre.kind() {
        return String::from_utf8(lit.0.to_vec()).unwrap_or_default();
    }
    "".to_string()
}

pub(super) fn get_literal(sre: &Hir) -> Option<String> {
    match sre.kind() {
        HirKind::Capture(cap) => get_literal(cap.sub.as_ref()),
        HirKind::Literal(lit) => {
            let s = String::from_utf8(lit.0.to_vec()).unwrap_or_default();
            Some(s)
        }
        _ => None,
    }
}

pub(super) fn hir_to_string(sre: &Hir) -> String {
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
            if let Some(quantifier) = get_quantifier(sre) {
                return match quantifier {
                    Quantifier::ZeroOrOne => ".?".to_string(),
                    Quantifier::ZeroOrMore => ".*".to_string(),
                    Quantifier::OneOrMore => ".+".to_string(),
                }
            }
            sre.to_string()
        }
        _ => sre.to_string(),
    }
}

fn get_quantifier(sre: &Hir) -> Option<Quantifier> {
    match sre.kind() {
        HirKind::Capture(cap) => get_quantifier(cap.sub.as_ref()),
        HirKind::Repetition(repetition) => get_repetition_quantifier(repetition),
        _ => None,
    }
}

fn get_repetition_quantifier(repetition: &Repetition) -> Option<Quantifier> {
    if repetition.greedy {
        if let HirKind::Class(clazz) = repetition.sub.kind() {
            if is_empty_class(clazz) {
                return match repetition.min {
                    0 => {
                        if repetition.max == Some(1) {
                            Some(Quantifier::ZeroOrOne)
                        } else {
                            Some(Quantifier::ZeroOrMore)
                        }
                    }
                    1 => Some(Quantifier::OneOrMore),
                    _ => None,
                };
            }
        }
    }
    None
}

fn quantifier_matcher(quantifier: Quantifier) -> Option<StringMatchHandler> {
    if quantifier == Quantifier::ZeroOrMore {
        // '.*'
        Some(StringMatchHandler::any(true))
    } else if quantifier == Quantifier::OneOrMore {
        // '.+'
        Some(StringMatchHandler::not_empty(true))
    } else {
        // .?
        None
    }
}

pub(super) fn optimize_concat_regex(subs: &[Hir]) -> (String, String, Vec<String>, Vec<Hir>) {
    if subs.is_empty() {
        return (String::new(), String::new(), Vec::new(), Vec::new());
    }

    let mut new_subs = subs.to_vec();

    while let Some(first) = new_subs.first() {
        if is_start_anchor(first) {
            new_subs.remove(0);
        } else {
            break;
        }
    }

    while let Some(last) = new_subs.last() {
        if is_end_anchor(last) {
            new_subs.pop();
        } else {
            break;
        }
    }

    let mut prefix = String::new();
    let mut suffix = String::new();
    let mut contains = Vec::new();

    let mut start = 0;
    let mut end = new_subs.len();

    if let Some(first) = new_subs.first() {
        if is_literal(first) {
            prefix = literal_to_string(first);
            start = 1;
        }
    }

    if !prefix.is_empty() && new_subs.len() == 1 {
        return (prefix, suffix, contains, new_subs);
    }

    if let Some(last) = new_subs.last() {
        if is_literal(last) {
            suffix = literal_to_string(last);
            end -= 1;
        }
    }

    for hir in &new_subs[start..end] {
        if is_literal(hir) {
            contains.push(literal_to_string(hir));
        }
    }

    (prefix, suffix, contains, new_subs)
}

/// Extract equality matches from a regexp.
/// Returns an empty vec if we can't replace the regexp by only equality matchers or the regexp contains
/// a mix of case-sensitive and case-insensitive matchers.
pub(super) fn find_set_matches(hir: &mut Hir) -> Option<(Vec<String>, bool)> {
    clear_begin_end_anchor(hir);
    find_set_matches_internal(hir, "")
}

fn find_set_matches_internal(hir: &Hir, base: &str) -> Option<(Vec<String>, bool)> {
    match hir.kind() {
        HirKind::Look(Look::Start) | HirKind::Look(Look::End) => None,
        HirKind::Literal(_) => {
            let literal = if !base.is_empty() {
                format!("{base}{}", hir_to_string(hir))
            } else {
                hir_to_string(hir)
            };
            Some((vec![literal], true))
        }
        HirKind::Empty => {
            if !base.is_empty() {
                Some((vec![base.to_string()], true))
            } else {
                None
            }
        }
        HirKind::Alternation(_) => find_set_matches_from_alternate(hir, base),
        HirKind::Capture(hir) => find_set_matches_internal(&hir.sub, base),
        HirKind::Concat(_) => find_set_matches_from_concat(hir, base),
        HirKind::Class(class) => match class {
            Unicode(ranges) => {
                let total_set = ranges
                    .iter()
                    .map(|r| 1 + (r.end() as usize - r.start() as usize))
                    .sum::<usize>();

                if total_set > MAX_SET_MATCHES {
                    return None;
                }

                let mut matches = Vec::new();
                for range in ranges.iter().flat_map(|r| r.start()..=r.end()) {
                    matches.push(format!("{base}{range}"));
                }

                Some((matches, true))
            }
            Bytes(ranges) => {
                let total_set = ranges
                    .iter()
                    .map(|r| 1 + (r.end() as usize - r.start() as usize))
                    .sum::<usize>();

                if total_set > MAX_SET_MATCHES {
                    return None;
                }

                let mut matches = Vec::new();

                for ch in ranges.iter().flat_map(|r| r.start()..=r.end()) {
                    matches.push(format!("{base}{ch}"));
                }

                Some((matches, true))
            }
        },
        _ => None,
    }
}


fn collect_simple_alternates(
    hir: &Hir,
    base: &str,
    matches: &mut Vec<String>,
) {
    match hir.kind() {
        HirKind::Literal(_) => {
            matches.push(hir_to_string(hir));
        }
        HirKind::Alternation(alternates) => {
            for sub in alternates.iter() {
                let alt = hir_to_string(sub);
                if !base.is_empty() {
                    matches.push(format!("{base}{alt}"));
                } else {
                    matches.push(alt);
                }
            }
        }
        HirKind::Concat(alts) => {
            match &alts.as_slice() {
                [h1, h2] => {
                    let left_kind = h1.kind();
                    let right_kind = h2.kind();
                    match (left_kind, right_kind) {
                        (HirKind::Literal(_), HirKind::Alternation(alts)) => {
                            // If we have two literals, we can optimize it.
                            let prefix = hir_to_string(h1);
                            for sub in alts.iter() {
                                let alt = hir_to_string(sub);
                                let value = format!("{base}{prefix}{alt}");
                                matches.push(value);
                            }
                        }
                        (HirKind::Alternation(alts), HirKind::Literal(_)) => {
                            let suffix = hir_to_string(h2);
                            for sub in alts.iter() {
                                let alt = hir_to_string(sub);
                                let value = format!("{base}{alt}{suffix}");
                                matches.push(value);
                            }
                        }
                        _ => {
                            // If we have something else, we cannot optimize this.
                        }
                    }
                }
                [h1, h2, h3] => {
                    let left_kind = h1.kind();
                    let right_kind = h3.kind();
                    let middle_kind = h2.kind();
                    if let (HirKind::Literal(_), HirKind::Alternation(alts), HirKind::Literal(_)) = (left_kind, middle_kind, right_kind) {
                        let prefix = hir_to_string(h1);
                        let suffix = hir_to_string(h3);
                        for sub in alts.iter() {
                            let mid = hir_to_string(sub);
                            let opt = format!("{base}{prefix}{mid}{suffix}");
                            matches.push(opt);
                        }
                    }
                },
                _=> {}
            }
        }
        _ => (),
    }

}

fn find_set_matches_from_concat(hir: &Hir, base: &str) -> Option<(Vec<String>, bool)> {
    if let HirKind::Concat(hirs) = hir.kind() {
        let mut matches = vec![base.to_string()];
        let mut matches_case_sensitive: Option<bool> = None;

        let cursor = hirs;
        let mut i: usize = 0;

        let len = cursor.len();
        while i < len {
            let mut hir = &cursor[i]; // todo: get_unchecked

            let all_literal = hir.properties().is_alternation_literal();
            if all_literal {
                collect_simple_alternates(hir, base, &mut matches);
                if matches_case_sensitive.is_none() {
                    matches_case_sensitive = Some(true);
                } else if !matches_case_sensitive.unwrap() {
                    return None; // mixed case sensitivity
                }
                i += 1;
                continue;
            }

            if let Some((val, len)) = get_case_folded_string(&cursor[i..]) {
                if let Some(sensitive) = matches_case_sensitive {
                    if sensitive {
                        return None;
                    }
                } else {
                    matches_case_sensitive = Some(false);
                }

                if i == 0 && matches.len() == 1 && matches[0].is_empty() {
                    matches.clear();
                }

                matches.push(format!("{base}{val}"));

                i += len;
                if i < len {
                    hir = &cursor[i];
                } else {
                    break;
                }
            }

            let mut new_matches = Vec::new();

            for b in matches.iter() {
                if let Some((items, sensitive)) = find_set_matches_internal(hir, b) {
                    if matches.len() + items.len() > MAX_SET_MATCHES {
                        return None;
                    }

                    if let Some(new_sensitive) = matches_case_sensitive {
                        if new_sensitive != sensitive {
                            return None;
                        }
                    } else {
                        matches_case_sensitive = Some(sensitive);
                    }

                    new_matches.extend(items);
                } else {
                    return None;
                }
            }

            i += 1;
            matches = new_matches;
        }

        return Some((matches, matches_case_sensitive.unwrap_or(true)));
    }

    None
}

fn find_set_matches_from_alternate(hir: &Hir, base: &str) -> Option<(Vec<String>, bool)> {
    let mut matches = Vec::new();
    let mut matches_case_sensitive = true;

    match hir.kind() {
        HirKind::Alternation(alternates) => {
            for (i, sub) in alternates.iter().enumerate() {
                if let Some((found, sensitive)) = find_set_matches_internal(sub, base) {
                    if found.is_empty() {
                        return None;
                    }
                    if i == 0 {
                        matches_case_sensitive = sensitive;
                    } else if matches_case_sensitive != sensitive {
                        return None;
                    }
                    if matches.len() + found.len() > MAX_SET_MATCHES {
                        return None;
                    }
                    matches.extend(found);
                } else {
                    return None;
                }
            }
        }
        _ => return None,
    }

    Some((matches, matches_case_sensitive))
}

fn clear_begin_end_anchor(hir: &mut Hir) {
    fn handle_vec(items: &[Hir]) -> Option<Vec<Hir>> {
        let mut cursor = &items[0..];

        while !cursor.is_empty() && is_start_anchor(&cursor[0]) {
            cursor = &cursor[1..];
        }

        if !cursor.is_empty() {
            let mut end = cursor.len();
            while end > 0 && is_end_anchor(&cursor[end - 1]) {
                end -= 1;
            }
            cursor = &cursor[..end];
        }

        if cursor.len() != items.len() {
            Some(cursor.to_vec())
        } else {
            None
        }
    }

    fn handle_concat(items: &[Hir]) -> Option<Hir> {
        let items = handle_vec(items)?;
        Some(Hir::concat(items))
    }

    fn handle_alts(items: &[Hir]) -> Option<Hir> {
        let items = handle_vec(items)?;
        Some(Hir::alternation(items))
    }

    match hir.kind() {
        HirKind::Alternation(alts) => {
            if let Some(modified) = handle_alts(alts) {
                *hir = modified;
            }
        }
        HirKind::Concat(hirs) => {
            if let Some(modified) = handle_concat(hirs) {
                *hir = modified;
            }
        }
        HirKind::Capture(capture) => {
            if let HirKind::Concat(hirs) = capture.sub.kind() {
                if let Some(modified) = handle_concat(hirs) {
                    *hir = modified;
                }
            }
        }
        _ => (),
    }
}

fn handle_regex(expr: &str, hir: &Hir) -> Result<StringMatchHandler, RegexError> {
    let expr = format!("^(?s:{expr})$");
    // todo: ensure anchor
    let regex = Regex::new(&expr)?;

    let mut matches = Vec::new();

    if let HirKind::Concat(hirs) = hir.kind() {
        let (prefix, suffix, contains, subs) = optimize_concat_regex(hirs);
        let sub = Hir::concat(subs);
        if let Some((sub_matches, case_sensitive)) = find_set_matches_internal(&sub, "") {
            if case_sensitive {
                matches = sub_matches;
            }
        }
        let matcher = RegexMatcher {
            regex,
            prefix,
            suffix,
            contains,
            string_matcher: None,
            set_matches: matches,
        };
        Ok(StringMatchHandler::Regex(matcher))
    } else {
        if let Some((sub_matches, _)) = find_set_matches_internal(hir, "") {
            matches = sub_matches;
        }
        let matcher = RegexMatcher {
            regex,
            prefix: "".to_string(),
            suffix: "".to_string(),
            contains: Vec::new(),
            string_matcher: None,
            set_matches: matches,
        };
        Ok(StringMatchHandler::Regex(matcher))
    }
}

pub fn get_or_values(pattern: &str) -> Result<Vec<String>, RegexError> {
    let mut values = Vec::new();
    let sre = build_hir(pattern)?;
    if !get_or_values_ext(&sre, &mut values) {
        values.clear();
    } 
    Ok(values)
}

pub fn get_or_values_ext(sre: &Hir, dest: &mut Vec<String>) -> bool {
    use HirKind::*;
    match sre.kind() {
        Empty => {
            dest.push("".to_string());
            true
        }
        Capture(cap) => get_or_values_ext(cap.sub.as_ref(), dest),
        Literal(literal) => {
            if let Ok(s) = String::from_utf8(literal.0.to_vec()) {
                dest.push(s);
                true
            } else {
                false
            }
        }
        Alternation(alt) => {
            dest.reserve(alt.len());
            for sub in alt.iter() {
                let start_count = dest.len();
                if let Some(literal) = get_literal(sub) {
                    dest.push(literal);
                } else if !get_or_values_ext(sub, dest) {
                    return false;
                }
                if dest.len() - start_count > MAX_OR_VALUES {
                    return false;
                }
            }
            true
        }
        Concat(concat) => {
            let mut prefixes = Vec::with_capacity(MAX_OR_VALUES);
            if !get_or_values_ext(&concat[0], &mut prefixes) {
                return false;
            }
            let subs = Vec::from(&concat[1..]);
            let concat = Hir::concat(subs);
            let prefix_count = prefixes.len();
            if !get_or_values_ext(&concat, &mut prefixes) {
                return false;
            }
            let suffix_count = prefixes.len() - prefix_count;
            let additional_capacity = prefix_count * suffix_count;
            if additional_capacity > MAX_OR_VALUES {
                // It is cheaper to use regexp here.
                return false;
            }
            dest.reserve(additional_capacity);
            let (pre, suffixes) = prefixes.split_at(prefix_count);
            for prefix in pre.iter() {
                for suffix in suffixes.iter() {
                    dest.push(format!("{prefix}{suffix}"));
                }
            }
            true
        }
        Class(class) => {
            if let Some(literal) = class.literal() {
                return if let Ok(s) = String::from_utf8(literal.to_vec()) {
                    dest.push(s);
                    true
                } else {
                    false
                };
            }

            match class {
                Unicode(uni) => {
                    for range in uni.iter().flat_map(|r| r.start()..=r.end()) {
                        dest.push(format!("{range}"));
                        if dest.len() > MAX_OR_VALUES {
                            // It is cheaper to use regexp here.
                            return false;
                        }
                    }
                    true
                }
                Bytes(bytes) => {
                    for range in bytes.iter().flat_map(|r| r.start()..=r.end()) {
                        dest.push(format!("{range}"));
                        if dest.len() > MAX_OR_VALUES {
                            return false;
                        }
                    }
                    true
                }
            }
        }
        _ => false,
    }
}

#[cfg(test)]
mod test {
    use super::remove_start_end_anchors;
    use crate::prelude::regex_utils::{find_set_matches, optimize_concat_regex};
    use crate::prelude::string_matcher_from_regex;
    use crate::regex_util::{build_hir, get_or_values};
    use regex_syntax::hir::HirKind;

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
        let got = build_hir(s);
        assert!(got.is_err());
    }

    fn test_optimized_regex(expr: &str, s: &str, result_expected: bool) {
        let matcher = string_matcher_from_regex(expr).unwrap();
        let result = matcher.matches(s);
        assert_eq!(
            result, result_expected,
            "unexpected result when matching {s} against regex={expr}; got {result}; want {result_expected}"
        );
    }

    #[test]
    fn test_optimize_concat_regex() {
        let cases = vec![
            ("foo(hello|bar)", "foo", "", vec![]),
            ("foo(hello|bar)world", "foo", "world", vec![]),
            ("foo.*", "foo", "", vec![]),
            ("foo.*hello.*bar", "foo", "bar", vec!["hello"]),
            (".*foo", "", "foo", vec![]),
            ("^.*foo$", "", "foo", vec![]),
            (".*foo.*", "", "", vec!["foo"]),
            (".*foo.*bar.*", "", "", vec!["foo", "bar"]),
            (".*(foo|bar).*", "", "", vec![]),
            (".*[abc].*", "", "", vec![]),
            (".*((?i)abc).*", "", "", vec![]),
            (".*(?i:abc).*", "", "", vec![]),
            ("(?i:abc).*", "", "", vec![]),
            (".*(?i:abc)", "", "", vec![]),
            (".*(?i:abc)def.*", "", "", vec!["def"]),
            ("(?i).*(?-i:abc)def", "", "", vec!["abc"]),
            (".*(?msU:abc).*", "", "", vec!["abc"]),
            ("[aA]bc.*", "", "", vec!["bc"]),
            ("^5..$", "5", "", vec![]),
            ("^release.*", "release", "", vec![]),
            ("^env-[0-9]+laio[1]?[^0-9].*", "env-", "", vec!["laio"]),
        ];

        for (regex, prefix, suffix, contains) in cases {
            let parsed = build_hir((&format!("^(?s:{})$", regex)).as_ref()).unwrap();
            if let HirKind::Concat(hirs) = &parsed.kind() {
                let (actual_prefix, actual_suffix, actual_contains, _) =
                    optimize_concat_regex(hirs);
                assert_eq!(
                    prefix, actual_prefix,
                    "unexpected prefix for regex={regex}. Expected {prefix}, got {actual_prefix}"
                );
                assert_eq!(
                    suffix, actual_suffix,
                    "unexpected suffix for regex={regex}. Expected {suffix}, got {actual_suffix}"
                );
                assert_eq!(contains, actual_contains);
            } else {
                panic!("Expected HirKind::Concat, got {:?}", parsed.kind());
            }
        }
    }

    // Refer to https://github.com/prometheus/prometheus/issues/2651.
    #[test]
    fn test_find_set_matches() {
        let cases = vec![
            // Single value, coming from a `bar=~"foo"` selector.
            ("foo", vec!["foo"], true),
            ("^foo", vec!["foo"], true),
            ("^foo$", vec!["foo"], true),
            // Simple sets alternates.
            ("foo|bar|zz", vec!["foo", "bar", "zz"], true),
            // Simple sets alternate and concat (bar|baz is parsed as "ba[rz]").
            ("foo|bar|baz", vec!["foo", "bar", "baz"], true),
            // Simple sets alternate and concat and capture
            ("foo|bar|baz|(zz)", vec!["foo", "bar", "baz", "zz"], true),
            // Simple sets alternate and concat and alternates with empty matches
            // parsed as  b(ar|(?:)|uzz) where b(?:) means literal b.
            ("bar|b|buzz", vec!["bar", "b", "buzz"], true),
            // Skip nested capture groups.
            ("^((bar|b|buzz))$", vec!["bar", "b", "buzz"], true),
            // Skip outer anchors (it's enforced anyway at the root).
            ("^(bar|b|buzz)$", vec!["bar", "b", "buzz"], true),
            ("^(?:prod|production)$", vec!["prod", "production"], true),
            // Do not optimize regexp with inner anchors.
            ("(bar|b|b^uz$z)", vec![], false),
            // Do not optimize regexp with an empty string matcher.
            ("^$|Running", vec![], false),
            // Simple sets containing escaped characters.
            ("fo\\.o|bar\\?|\\^baz", vec!["fo.o", "bar?", "^baz"], true),
            // using charclass
            ("[abc]d", vec!["ad", "bd", "cd"], true),
            // high low charset different => A(B[CD]|EF)|BC[XY]
            (
                "ABC|ABD|AEF|BCX|BCY",
                vec!["ABC", "ABD", "AEF", "BCX", "BCY"],
                true,
            ),
            // triple concat
            (
                "api_(v1|prom)_push",
                vec!["api_v1_push", "api_prom_push"],
                true,
            ),
            // triple concat with multiple alternates
            (
                "(api|rpc)_(v1|prom)_push",
                vec![
                    "api_v1_push",
                    "api_prom_push",
                    "rpc_v1_push",
                    "rpc_prom_push",
                ],
                true,
            ),
            (
                "(api|rpc)_(v1|prom)_(push|query)",
                vec![
                    "api_v1_push",
                    "api_v1_query",
                    "api_prom_push",
                    "api_prom_query",
                    "rpc_v1_push",
                    "rpc_v1_query",
                    "rpc_prom_push",
                    "rpc_prom_query",
                ],
                true,
            ),
            // class starting with "-"
            (
                "[-1-2][a-c]",
                vec!["-a", "-b", "-c", "1a", "1b", "1c", "2a", "2b", "2c"],
                true,
            ),
            ("[1^3]", vec!["1", "3", "^"], true),
            // OpPlus with concat
            ("(.+)/(foo|bar)", vec![], false),
            // Simple sets containing special characters without escaping.
            ("fo.o|bar?|^baz", vec![], false),
            // case-sensitive wrapper.
            ("(?i)foo", vec!["FOO"], false),
            // case-sensitive wrapper on alternate.
            //    ("(?i)foo|bar|baz", vec!["FOO", "BAR", "BAZ", "BAr", "BAz"], false),
            // mixed case sensitivity.
            ("(api|rpc)_(v1|prom)_((?i)push|query)", vec![], false),
            // mixed case sensitivity concatenation only without a capture group.
            ("api_v1_(?i)push", vec![], false),
            // mixed case sensitivity alternation only without a capture group.
            ("api|(?i)rpc", vec![], false),
            // case sensitive after unsetting insensitivity.
            ("rpc|(?i)(?-i)api", vec!["rpc", "api"], true),
            // case-sensitive after unsetting insensitivity in all alternation options.
            ("(?i)((?-i)api|(?-i)rpc)", vec!["api", "rpc"], true),
            // mixed case sensitivity after unsetting insensitivity.
            ("(?i)rpc|(?-i)api", vec![], false),
            // too high charset combination
            ("(api|rpc)_[^0-9]", vec![], false),
            // too many combinations
            ("[a-z][a-z]", vec![], false),
        ];

        for (pattern, exp_matches, exp_case_sensitive) in cases {
            let mut parsed = build_hir((&format!("^(?s:{})$", pattern)).as_ref()).unwrap();
            let (matches, case_sensitive) = find_set_matches(&mut parsed).unwrap_or_default();
            assert_eq!(
                exp_matches, matches,
                "parsing {pattern} failed. Expected {:?}, got {:?}",
                exp_matches, matches
            );

            // TODO:
            // if exp_case_sensitive {
            //     // When the regexp is case-sensitive, we want to ensure that the
            //     // set matches are maintained in the final matcher.
            //     let r = FastRegexMatcher::new(pattern).unwrap();
            //     assert_eq!(exp_matches, r.set_matches());
            // }
        }
    }

    #[test]
    fn test_regex_match() {
        fn f(expr: &str, s: &str, result_expected: bool) {
            test_optimized_regex(expr, s, result_expected);
        }

        f("", "foo", true);
        f("", "", true);
        f("foo", "", false);
        f(".*", "", true);
        f(".*", "foo", true);
        f(".+", "", false);
        f(".+", "foo", true);
        f("foo.*", "bar", false);
        f("foo.*", "foo", true);
        f("foo.*", "foobar", true);
        f("foo.+", "bar", false);
        f("foo.+", "foo", false);
        f("foo.+", "foobar", true);
        f("foo|bar", "", false);
        f("foo|bar", "a", false);
        f("foo|bar", "foo", true);
        f("foo|bar", "bar", true);
        f("foo|bar", "foobar", false);
        f("foo(bar|baz)", "a", false);
        f("foo(bar|baz)", "foobar", true);
        f("foo(bar|baz)", "foobaz", true);
        f("foo(bar|baz)", "foobal", false);
        f("^foo|b(ar)$", "foo", true);
        f("^foo|b(ar)$", "bar", true);
        f("^foo|b(ar)$", "barz", false);
        f("^foo|b(ar)$", "ar", false);
        f(".*foo.*", "foo", true);
        f(".*foo.*", "afoobar", true);
        f(".*foo.*", "abc", false);
        f("foo.*bar.*", "foobar", true);
        f("foo.*bar.*", "foo_bar_", true);
        f("foo.*bar.*", "foobaz", false);
        f(".+foo.+", "foo", false);
        f(".+foo.+", "afoobar", true);
        f(".+foo.+", "afoo", false);
        f(".+foo.+", "abc", false);
        f("foo.+bar.+", "foobar", false);
        f("foo.+bar.+", "foo_bar_", true);
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
        f("^(?:true)$", "true", true);
        f("^(?:true)$", "false", false);

        // https://github.com/VictoriaMetrics/VictoriaMetrics/issues/5297
        f(".+;|;.+", ";", false);
        f(".+;|;.+", "foo", false);
        f(".+;|;.+", "foo;bar", false);
        f(".+;|;.+", "foo;", true);
        f(".+;|;.+", ";foo", true);

        f(".+foo|bar|baz.+", "foo", false);
        f(".+foo|bar|baz.+", "afoo", true);
        f(".+foo|bar|baz.+", "fooa", false);
        f(".+foo|bar|baz.+", "afooa", false);
        f(".+foo|bar|baz.+", "bar", true);
        f(".+foo|bar|baz.+", "abar", false);
        f(".+foo|bar|baz.+", "abara", false);
        f(".+foo|bar|baz.+", "bara", false);
        f(".+foo|bar|baz.+", "baz", false);
        f(".+foo|bar|baz.+", "baza", true);
        f(".+foo|bar|baz.+", "abaz", false);
        f(".+foo|bar|baz.+", "abaza", false);
        f(".+foo|bar|baz.+", "afoo|bar|baza", false);
        f(".+(foo|bar|baz).+", "abara", true);
        f(".+(foo|bar|baz).+", "afooa", true);
        f(".+(foo|bar|baz).+", "abaza", true);

        f(".*;|;.*", ";", true);
        f(".*;|;.*", "foo", false);
        f(".*;|;.*", "foo;bar", false);
        f(".*;|;.*", "foo;", true);
        f(".*;|;.*", ";foo", true);

        f(".*foo(bar|baz)", "fooxfoobaz", true);
        f(".*foo(bar|baz)", "fooxfooban", false);
        f(".*foo(bar|baz)", "fooxfooban foobar", true);
    }

    #[test]
    fn  test_get_or_values_regex() {
        let test_cases = vec![
            ("", vec![""]),
            ("foo", vec!["foo"]),
            ("^foo$", vec![]),
            ("|foo", vec!["", "foo"]),
            //            ("|foo|", vec!["", "", "foo"]),
            ("foo.+", vec![]),
            ("foo.*", vec![]),
            (".*", vec![]),
            ("foo|.*", vec![]),
            ("(fo((o)))|(bar)", vec!["bar", "foo"]),
            ("foobar", vec!["foobar"]),
            ("z|x|c", vec!["c", "x", "z"]),
            ("foo|bar", vec!["bar", "foo"]),
            ("(foo|bar)", vec!["bar", "foo"]),
            ("(foo|bar)baz", vec!["barbaz", "foobaz"]),
            ("[a-z][a-z]", vec![]),
            ("[a-d]", vec!["a", "b", "c", "d"]),
            ("x[a-d]we", vec!["xawe", "xbwe", "xcwe", "xdwe"]),
            ("foo(bar|baz)", vec!["foobar", "foobaz"]),
            (
                "foo(ba[rz]|(xx|o))",
                vec!["foobar", "foobaz", "fooo", "fooxx"],
            ),
            (
                "foo(?:bar|baz)x(qwe|rt)",
                vec!["foobarxqwe", "foobarxrt", "foobazxqwe", "foobazxrt"],
            ),
            ("foo(bar||baz)", vec!["foo", "foobar", "foobaz"]),
            ("(a|b|c)(d|e|f|0|1|2)(g|h|k|x|y|z)", vec![]),
            ("(?i)foo", vec!["FOO", "FOo", "FoO", "Foo", "fOO", "fOo", "foO", "foo"]),
            ("(?i)(foo|bar)", vec!["BAR", "BAr", "BaR", "Bar", "FOO", "FOo", "FoO", "Foo", "bAR", "bAr", "baR", "bar", "fOO", "fOo", "foO", "foo"]),
            ("^foo|bar$", vec![]),
            ("^(foo|bar)$", vec![]),
            ("^a(foo|b(?:a|r))$", vec![]),
            ("^a(foo$|b(?:a$|r))$", vec![]),
            ("^a(^foo|bar$)z$", vec![]),
        ];

        for (s, expected) in test_cases {
            let mut result = get_or_values(s).unwrap();
            let mut expected = expected.into_iter().map(|s| s.to_string()).collect::<Vec<_>>();
            result.sort();
            expected.sort();
            
            assert_eq!(
                result, expected,
                "unexpected values for s={}. Got {:?}, want {:?}",
                s, result, expected
            );
        }
    }
}
