use super::index_reader::{IndexReader, IndexReaderError, IndexReaderResult};
use crate::provider::postings::{find_intersecting_postings, without, PostingsEnum, PostingsList};
use crate::SeriesRef;
use futures::future::try_join_all;
use futures::Future;
use metricsql_common::hash::{FastHashSet, HashSetExt};
use metricsql_parser::label::{Label, MatchOp, Matcher, Matchers};
use smallvec::SmallVec;
use std::cmp::Ordering;
use enquote::enquote;

/// `postings_for_matchers` assembles a single postings iterator against the index reader
/// based on the given matchers. The resulting postings are not ordered by series.
pub async fn postings_for_matchers(ix: &impl IndexReader, matchers: &Matchers) -> IndexReaderResult<PostingsEnum> {
    if matchers.is_empty() {
        // ??
        return all_postings(ix).await;
    }

    if !matchers.matchers.is_empty() {
        return postings_for_matchers_slice(ix, &matchers.matchers).await;
    }

    if !matchers.or_matchers.is_empty() {
        run_or_matchers(ix, &matchers.or_matchers).await
    } else {
        Ok(PostingsEnum::empty())
    }
}

pub async fn postings_for_matcher(ix: &impl IndexReader, m: &Matcher) -> IndexReaderResult<PostingsEnum> {
    if m.label.is_empty() && m.value.is_empty() {
        return all_postings(ix).await;
    }
    
    if m.op == MatchOp::Equal {
        // how to avoid clone ???
        return postings(ix, &m.label, &[m.value.clone()]).await;
    }
    
    // Fast-path for set matching.
    if m.op == MatchOp::RegexEqual {
        let set_matches = m.set_matches();
        if let Some(matches) = set_matches {
            if !matches.is_empty() {
                return postings(ix, &m.label, &matches).await;
            }
        }
    }

    let postings = ix.postings_for_label_matching(&m.label, |s| m.matches(s))
        .await?;
    
    let postings = PostingsEnum::wrap(Box::new(postings));
    Ok(postings)
}


pub async fn label_values_with_matchers(r: &impl IndexReader, name: &str, matchers: &[Matcher]) -> IndexReaderResult<Vec<String>> {
    let mut all_values = r.label_values(name, matchers).await?;
    let mut has_matchers_for_other_labels = false;

    for m in matchers {
        if m.label != name {
            has_matchers_for_other_labels = true;
            continue;
        }
        
        all_values = all_values.iter()
            .filter_map(|x| if m.matches(x.as_str()) {
                Some(x.clone())
            } else { 
                None
            }).collect();
    }

    if all_values.is_empty() {
        return Ok(Vec::new());
    }

    if !has_matchers_for_other_labels {
        return Ok(all_values);
    }

    let p = postings_for_matchers_slice(r, matchers).await?;
    let values_postings: Vec<_> = all_values.iter()
        .map(|value| {
            let label_value = value.clone();
            r.postings(name, &[label_value])
        }).collect();

    let postings = try_join_all(values_postings.into_iter()).await?;
    let indexes = find_intersecting_postings(&p, postings)?;
    let mut values = Vec::with_capacity(indexes.len());
    for posting in indexes {
        values.push(all_values[posting.index].clone());
    }

    Ok(values)
}

async fn empty() -> IndexReaderResult<PostingsEnum> {
    Ok(PostingsEnum::empty())
}

type IterResult = dyn Future<Output=IndexReaderResult<dyn Iterator<Item=SeriesRef>>>;

/// `postings_for_matchers` assembles a single postings iterator against the index
/// based on the given matchers. The resulting postings are not ordered by series. 
async fn postings_for_matchers_slice(ix: &impl IndexReader, ms: &[Matcher]) -> IndexReaderResult<PostingsEnum> {
    if ms.len() == 1 {
        let m = &ms[0];
        if m.label.is_empty() && m.label.is_empty() {
            return all_postings(ix).await;
        }
    }

    let mut sorted_matchers: SmallVec<(&Matcher, bool, bool), 4> = SmallVec::new();
    let mut not_its: SmallVec<_, 4> = SmallVec::new();
    let mut its: SmallVec<_, 4> =SmallVec::new(); // todo: smallvec

    let mut has_subtracting_matchers = false;
    let mut has_intersecting_matchers = false;

    // See which label must be non-empty.
    // Optimization for case like {l=~".", l!="1"}.
    let mut label_must_be_set: FastHashSet<&str> = FastHashSet::with_capacity(ms.len());
    for m in ms {
        let matches_empty = m.matches("");
        if !matches_empty {
            label_must_be_set.insert(&m.label);
        }
        let is_subtracting = is_subtracting_matcher(m, &label_must_be_set);

        has_subtracting_matchers |= is_subtracting;
        has_intersecting_matchers |= !is_subtracting;

        sorted_matchers.push((m, matches_empty, is_subtracting))
    }

    if has_subtracting_matchers && !has_intersecting_matchers {
        // If there's nothing to subtract from, add in everything and remove the not_its later.
        // We prefer to get all_postings so that the base of subtraction (i.e. all_postings)
        // doesn't include series that may be added to the index reader during this function call.
        its.push( ix.all_postings() );
    } else {
        its.push(empty());
    };

    // Sort matchers to have the intersecting matchers first.
    // This way the base for subtraction is smaller and there is no chance that the set we subtract
    // from contains postings of series that didn't exist when we constructed the set we subtract by.
    sorted_matchers.sort_by(|i, j| -> Ordering {
        let is_i_subtracting = i.2;
        let is_j_subtracting = j.2;
        if !is_i_subtracting && is_j_subtracting {
            return Ordering::Less;
        }
        // sort by match cost
        let cost_i = i.0.cost();
        let cost_j = j.0.cost();
        cost_i.cmp(&cost_j)
    });

    for (m, matches_empty, _is_subtracting) in sorted_matchers {
        let value = &m.value;
        let name = &m.label;
        let typ = m.op;

        if name.is_empty() && value.is_empty() {
            // We already handled the case at the top of the function,
            // and it is unexpected to get all postings again here.
            return Err(IndexReaderError::MissingMatcher);
        }

        if typ == MatchOp::RegexEqual && value == ".*" {
            // .* regexp matches any string: do nothing.
            continue;
        }

        if typ == MatchOp::RegexNotEqual && value == ".*" {
            return Ok(PostingsEnum::empty());
        }

        if typ == MatchOp::RegexEqual && value == ".+" {
            // .+ regexp matches any non-empty string: get postings for all label values.
            let it = ix.postings_for_all_label_values(&m.label);
            its.push(it);
        } else if typ == MatchOp::RegexNotEqual && value == ".+" {
            // .+ regexp matches any non-empty string: get postings for all label values and remove them.
            let it = ix.postings_for_all_label_values(name);
            not_its.push( it );
        } else if label_must_be_set.contains(&name.as_str()) {
            // If this matcher must be non-empty, we can be smarter.
            let is_not = typ == MatchOp::NotEqual || m.op == MatchOp::RegexNotEqual;

            if is_not {
                // a failure here should probably panic
                let inverse = m.inverse().map_err(|_| {
                    IndexReaderError::InvalidMatcher(m.to_string())
                })?;

                // If the label can't be empty and is a Not, then subtract it out at the end.
                if matches_empty {
                    // l!="foo"
                    // If the label can't be empty and is a Not and the inner matcher
                    // doesn't match empty, then subtract it out at the end.
                    let it = postings_for_matcher(ix, &inverse);
                    not_its.push(it);
                } else {
                    // l!=""
                    // If the label can't be empty and is a Not, but the inner matcher can
                    // be empty we need to use inverse_postings_for_matcher.
                    let it = inverse_postings_for_matcher(ix, &inverse).await?;
                    its.push(it);
                }
            } else {
                // l="a", l=~"a|b", l=~"a.b", etc.
                // Non-Not matcher, use normal `postings_for_matcher`.
                let it = postings_for_matcher(ix, m).await?;
                its.push(it);
            }
        } else {
            // l=""
            // If the matchers for a label name selects an empty value, it selects all
            // the series which don't have the label name set too. See:
            // https://github.com/prometheus/prometheus/issues/3575 and
            // https://github.com/prometheus/prometheus/pull/3578#issuecomment-351653555
            let it = inverse_postings_for_matcher(ix, m).await?;
            not_its.push(it)
        }
    }

    // todo: join the following 2 statements
    let resolved_its = try_join_all(its).await?;
    let resolved_not_its = try_join_all(not_its).await?;

    let mut it = PostingsEnum::intersection(resolved_its);

    for not in resolved_not_its {
        it = without(it, not)
    }

    Ok(it)
}


#[inline]
fn is_subtracting_matcher(m: &Matcher, label_must_be_set: &FastHashSet<&str>) -> bool {
    if !label_must_be_set.contains(&m.label.as_str()) {
        return true;
    }
    matches!(m.op, MatchOp::NotEqual | MatchOp::RegexNotEqual) && m.matches("")
}

async fn inverse_postings_for_matcher<'a>(ix: &impl IndexReader, m: &Matcher) -> IndexReaderResult<PostingsEnum> {
    // Fast-path for RegexNotEqual matching.
    // Inverse of a RegexNotEqual is RegexpEqual (double negation).
    // Fast-path for set matching.
    if m.op == MatchOp::RegexNotEqual {
        if let Some(matches) = m.set_matches() {
            if !matches.is_empty() {
                return postings(ix, &m.label, &matches).await
            }
        }
    }

    // Fast-path for NotEqual matching.
    // Inverse of a NotEqual is Equal (double negation).
    if m.op == MatchOp::NotEqual {
        // todo: figure out how to eliminate this clone
        let value = m.value.clone();
        return postings(ix, &m.label, &[value]).await
    }

    // If the matcher being inverted is =~"" or ="", we just want all the values.
    if m.value.is_empty() && (m.op == MatchOp::RegexEqual || m.op == MatchOp::Equal) {
        let postings = ix.postings_for_all_label_values(&m.label).await?;
        return Ok(PostingsEnum::wrap(Box::new(postings)));
    }

    let postings = ix.postings_for_label_matching(&m.label, |s| !m.matches(s)).await?;
    Ok(PostingsEnum::wrap(Box::new(postings)))
}

async fn run_or_matchers(ix: &impl IndexReader, matchers: &[Vec<Matcher>]) -> IndexReaderResult<PostingsEnum> {
    if matchers.is_empty() {
        Ok(PostingsEnum::empty())
    } else if matchers.len() == 1 {
        let m = matchers.get(0).expect("Out of bounds error running matchers");
        postings_for_matchers_slice(ix, &m).await
    } else {
        let futures = matchers.iter().map(|m| postings_for_matchers_slice(ix, m)).collect();
        let its = try_join_all(futures).await?;
        Ok(PostingsEnum::intersection(its))
    }
}

// Note - assumes that labels is sorted
fn format_metric_name(name: &str, labels: &[Label]) -> String {
    let size_hint = name.len()
        + labels
        .iter()
        .map(|l| l.name.len() + l.value.len() + 3)
        .sum::<usize>();
    let mut full_name: String = String::with_capacity(size_hint);
    format_metric_name_into(&mut full_name, name, labels);
    full_name
}

fn format_metric_name_into(full_name: &mut String, name: &str, labels: &[Label]) {
    full_name.push_str(name);
    if !labels.is_empty() {
        full_name.push('{');
        for (i, label) in labels.iter().enumerate() {
            full_name.push_str(&label.name);
            full_name.push_str("=\"");
            // avoid allocation if possible
            if label.value.contains('"') {
                let quoted_value = enquote('\"', &label.value);
                full_name.push_str(&quoted_value);
            } else {
                full_name.push_str(&label.value);
            }
            full_name.push('"');
            if i < labels.len() - 1 {
                full_name.push(',');
            }
        }
        full_name.push('}');
    }
}

#[inline]
async fn all_postings(ix: &impl IndexReader) -> IndexReaderResult<PostingsEnum> {
    let postings = ix.all_postings().await?;
    Ok(PostingsEnum::Wrapped(Box::new(postings)))
}

async fn postings(ix: &impl IndexReader, name: &str, values: &[String]) -> IndexReaderResult<PostingsEnum> {
    let postings = ix.postings(name, values).await?;
    Ok(PostingsEnum::wrap(Box::new(postings)))
}