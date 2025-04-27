use super::error::{ProviderError, ProviderResult};
use super::index_reader::IndexReader;
use crate::provider::postings::{find_intersecting_postings, PostingsEnum, PostingsIterator};
use futures::future::try_join_all;
use iter_set_ops::{intersect_iters, subtract_iters};
use metricsql_common::hash::{FastHashSet, HashSetExt};
use metricsql_parser::label::{MatchOp, Matcher, Matchers};
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::future::Future;
use std::pin::Pin;

pub struct Querier<T: IndexReader> {
    pub index_reader: T,
}

/// `postings_for_matchers` assembles a single postings iterator against the index reader
/// based on the given matchers. The resulting postings are not ordered by series.
pub async fn postings_for_matchers<'a, Reader>(
    ix: &'a Reader,
    matchers: &'a Matchers,
) -> ProviderResult<PostingsEnum<Reader::Postings<'a>>>
where
    Reader: IndexReader,
    Reader::Postings<'a>: PostingsIterator,
{
    if matchers.is_empty() {
        let all = ix.all_postings().await?;
        return Ok(PostingsEnum::list(all));
    }

    if !matchers.matchers.is_empty() {
        return postings_for_matchers_slice(ix, &matchers.matchers).await;
    }

    if !matchers.or_matchers.is_empty() {
        run_or_matchers(ix, &matchers.or_matchers).await
    } else {
        Ok(PostingsEnum::Empty)
    }
}

pub fn postings_for_matcher<'a, R>(
    index_reader: &'a R,
    m: &'a Matcher,
) -> Pin<Box<dyn Future<Output = Result<R::Postings<'a>, ProviderError>> + Send + 'a>>
where
    R: IndexReader,
{
    postings_for_matcher_internal::<R>(index_reader, m)
}

fn postings_for_matcher_internal<'a, R>(
    ix: &'a R,
    m: &'a Matcher,
) -> Pin<Box<dyn Future<Output = Result<R::Postings<'a>, ProviderError>> + Send + 'a>>
where
    R: IndexReader,
    R::Postings<'a>: PostingsIterator,
{
    if m.label.is_empty() && m.value.is_empty() {
        return ix.all_postings();
    }

    if m.op == MatchOp::Equal {
        // how to avoid clone ???
        return ix.postings(&m.label, &[m.value.as_str()]);
    }

    // Fast-path for set matching.
    if m.op == MatchOp::RegexEqual {
        let set_matches = m.set_matches();
        if let Some(matches) = set_matches {
            if !matches.is_empty() {
                let matches: SmallVec<&str, 6> = matches.iter().map(|s| s.as_str()).collect();
                return ix.postings(&m.label, &matches)
            }
        }
    }

    ix.postings_for_label_matching(&m.label, move |s| m.matches(s))
}

pub async fn label_values_with_matchers<'a, R>(
    ix: &R,
    name: &str,
    matchers: Option<&Matchers>,
) -> ProviderResult<Vec<String>>
where
    R: IndexReader,
{
    let mut all_values = ix.label_values(name, matchers).await?;

    fn process_matchers(matchers: &[Matcher], name: &str, all_values: &mut Vec<String>) -> bool {
        let mut has_matchers_for_other_labels = false;
        for m in matchers {
            if m.label != name {
                has_matchers_for_other_labels = true;
                continue;
            }

            *all_values = all_values
                .iter()
                .filter_map(|x| {
                    if m.matches(x.as_str()) {
                        Some(x.clone())
                    } else {
                        None
                    }
                })
                .collect();
        }
        has_matchers_for_other_labels
    }

    let mut has_matchers_for_other_labels = false;

    if let Some(matchers) = matchers {
        if !matchers.matchers.is_empty() {
            has_matchers_for_other_labels =
                process_matchers(&matchers.matchers, name, &mut all_values);
        } else if !matchers.or_matchers.is_empty() {
            for or_matchers in &matchers.or_matchers {
                has_matchers_for_other_labels |=
                    process_matchers(or_matchers, name, &mut all_values);
            }
        }
    }

    if all_values.is_empty() {
        return Ok(Vec::new());
    }

    if !has_matchers_for_other_labels {
        return Ok(all_values);
    }

    let p = if let Some(matchers) = matchers {
        postings_for_matchers(ix, matchers).await?
    } else {
        PostingsEnum::Wrapped(ix.all_postings().await?)
    };

    let values_postings: Vec<_> = all_values
        .iter()
        .map(|value| ix.postings(name, &[value.as_str()]))
        .collect();

    let postings = try_join_all(values_postings.into_iter()).await?;
    let indexes = find_intersecting_postings(p, postings)?;
    let mut values = Vec::with_capacity(indexes.len());
    for posting in indexes {
        values.push(all_values[posting.index].clone());
    }

    Ok(values)
}

//type PostingsResult<'a> = BoxFuture<'a, ProviderResult<impl PostingsList>>;

/// `postings_for_matchers_slice` assembles a single postings iterator against the index
async fn postings_for_matchers_slice<'a, R>(
    ix: &'a R,
    ms: &'a [Matcher],
) -> ProviderResult<PostingsEnum<R::Postings<'a>>>
where
    R: IndexReader,
    R::Postings<'a>: PostingsIterator,
{
    if ms.len() == 1 {
        let m = &ms[0];
        if m.label.is_empty() && m.value.is_empty() {
            let postings = PostingsEnum::Wrapped(ix.all_postings().await?);
            return Ok(postings);
        }
    }

    let mut not_its_futures: SmallVec<_, 4> = SmallVec::new();
    let mut its_futures: SmallVec<_, 4> = SmallVec::new();

    let mut has_subtracting_matchers = false;
    let mut has_intersecting_matchers = false;

    let mut sorted_matchers: SmallVec<(&Matcher, bool, bool), 4> = SmallVec::new();
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
        its_futures.push(ix.all_postings());
    }

    sorted_matchers.sort_by(|i, j| {
        let is_i_subtracting = i.2;
        let is_j_subtracting = j.2;
        if !is_i_subtracting && is_j_subtracting {
            return Ordering::Less;
        }
        let cost_i = i.0.cost();
        let cost_j = j.0.cost();
        cost_i.cmp(&cost_j)
    });


    // Store owned inverse matchers here
    let mut inverse_matchers_storage: SmallVec<Box<Matcher>, 4> = SmallVec::new();

    for (m, matches_empty, _is_subtracting) in sorted_matchers {
        let value = &m.value;
        let name = &m.label;
        let typ = m.op;

        if name.is_empty() && value.is_empty() {
            return Err(ProviderError::MissingMatcher);
        }

        match (m.op, m.value.as_str()) {
            (MatchOp::RegexEqual, ".*") => continue,
            (MatchOp::RegexNotEqual, ".*") => {
                return Ok(PostingsEnum::Empty);
            }
            (MatchOp::RegexEqual, ".+") => {
                let it = ix.postings_for_all_label_values(&m.label);
                its_futures.push(it);
            }
            (MatchOp::RegexNotEqual, ".+") => {
                let it = ix.postings_for_all_label_values(name);
                not_its_futures.push(it);
            }
            _ if label_must_be_set.contains(&name.as_str()) => {
                let is_not = typ == MatchOp::NotEqual || m.op == MatchOp::RegexNotEqual;

                if is_not {
                    let inverse = m
                        .inverse()
                        .map_err(|_| ProviderError::InvalidMatcher(m.to_string()))?;

                    // Push the owned matcher into storage
                    inverse_matchers_storage.push(Box::new(inverse));

                    // Get a stable reference to the stored matcher
                    let inverse_ref: &Matcher = inverse_matchers_storage.last().unwrap().as_ref();

                    if matches_empty {
                        // Instead of resolving here, push the future to not_its_futures
                        let it = postings_for_matcher_internal(ix, inverse_ref);
                        not_its_futures.push(it);
                    } else {
                        let it = inverse_postings_for_matcher(ix, inverse_ref);
                        its_futures.push(it);
                    }
                } else {
                    let it = postings_for_matcher_internal(ix, m);
                    its_futures.push(it);
                }
            }
            _ => {
                let it = postings_for_matcher_internal(ix, m);
                its_futures.push(it);
            }
        }
    }

    // All futures are now resolved together
    let mut its = try_join_all(its_futures).await?;
    let resolved_not_its = try_join_all(not_its_futures).await?;

    if its.is_empty() {
        return Ok(PostingsEnum::Empty);
    }

    if its.len() == 1 {
        let first = its.pop().unwrap();
        if first.is_empty() {
            return Ok(PostingsEnum::Empty);
        }
        return Ok(PostingsEnum::wrap(first));
    }

    if its.iter().any(|x| x.is_empty()) {
        return Ok(PostingsEnum::Empty);
    }

    let mut it = intersect_iters(&mut its);
    let without = subtract_iters(&mut it, resolved_not_its);

    Ok(PostingsEnum::list(without))
}

#[inline]
fn is_subtracting_matcher(m: &Matcher, label_must_be_set: &FastHashSet<&str>) -> bool {
    if !label_must_be_set.contains(&m.label.as_str()) {
        return true;
    }
    matches!(m.op, MatchOp::NotEqual | MatchOp::RegexNotEqual) && m.matches("")
}

fn inverse_postings_for_matcher<'a, R>(
    ix: &'a R,
    m: &'a Matcher,
) -> Pin<Box<dyn Future<Output = Result<R::Postings<'a>, ProviderError>> + Send + 'a>>
where
    R: IndexReader,
    R::Postings<'a>: PostingsIterator,
{
    // Fast-path for RegexNotEqual matching.
    // Inverse of a RegexNotEqual is RegexpEqual (double negation).
    // Fast-path for set matching.
    if m.op == MatchOp::RegexNotEqual {
        if let Some(matches) = m.set_matches() {
            if !matches.is_empty() {
                let matches: SmallVec<&str, 6> = matches.iter().map(|s| s.as_str()).collect();
                return ix.postings(&m.label, &matches);
            }
        }
    }

    // Fast-path for NotEqual matching.
    // Inverse of a NotEqual is Equal (double negation).
    if m.op == MatchOp::NotEqual {
        return ix.postings(&m.label, &[&m.value]);
    }

    // If the matcher being inverted is =~"" or ="", we just want all the values.
    if m.value.is_empty() && (m.op == MatchOp::RegexEqual || m.op == MatchOp::Equal) {
        return ix.postings_for_all_label_values(&m.label);
    }

    ix.postings_for_label_matching(&m.label, move |s| !m.matches(s))
}

async fn run_or_matchers<'a, R>(
    ix: &'a R,
    matchers: &'a [Vec<Matcher>],
) -> ProviderResult<PostingsEnum<R::Postings<'a>>>
where
    R: IndexReader,
{
    if matchers.is_empty() {
        Ok(PostingsEnum::Empty)
    } else if matchers.len() == 1 {
        let m = matchers
            .get(0)
            .expect("Out of bounds error running matchers");
        postings_for_matchers_slice(ix, &m).await
    } else {
        let futures: Vec<_> = matchers
            .iter()
            .map(|m| postings_for_matchers_slice(ix, m))
            .collect();
        let mut its = try_join_all(futures).await?;
        let it = intersect_iters(&mut its);
        Ok(PostingsEnum::list(it))
    }
}
