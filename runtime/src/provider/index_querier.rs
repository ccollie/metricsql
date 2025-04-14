use futures::Future;
use std::pin::Pin;
use super::error::{ProviderError, ProviderResult};
use super::index_reader::IndexReader;
use crate::provider::postings::{find_intersecting_postings, PostingsEnum, PostingsIterator};
use futures::future::try_join_all;
use iter_set_ops::{intersect_iters, subtract_iters};
use metricsql_common::hash::{FastHashSet, HashSetExt};
use metricsql_parser::label::{MatchOp, Matcher, Matchers};
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::sync::Arc;

pub struct IndexQuerier<T: IndexReader> {
    pub index_reader: Arc<T>,
}

impl <T: IndexReader> IndexQuerier<T> {
    pub fn new(index_reader: Arc<T>) -> Self {
        IndexQuerier { index_reader: index_reader.clone() }
    }

    /// `postings_for_matchers` assembles a single postings iterator against the index reader
    /// based on the given matchers. The resulting postings are not ordered by series.
    pub async fn postings_for_matchers<'a>(
        &'a self,
        matchers: &'a Matchers,
    ) -> ProviderResult<PostingsEnum<T::Postings<'a>>> {
        if matchers.is_empty() {
            let all = self.index_reader.all_postings().await?;
            return Ok(PostingsEnum::list(all));
        }

        if !matchers.matchers.is_empty() {
            return self.postings_for_matchers_slice(&matchers.matchers).await;
        }

        if !matchers.or_matchers.is_empty() {
            self.run_or_matchers(&matchers.or_matchers).await
        } else {
            Ok(PostingsEnum::Empty)
        }
    }

    pub fn postings_for_matcher<'a>(
        &'a self,
        m: &'a Matcher,
    ) -> Pin<Box<dyn Future<Output = Result<T::Postings<'a>, ProviderError>> + Send + 'a>>
    {
        self.postings_for_matcher_internal(m)
    }

    fn postings_for_matcher_internal<'a>(
        &'a self,
        m: &'a Matcher,
    ) -> Pin<Box<dyn Future<Output = Result<T::Postings<'a>, ProviderError>> + Send + 'a>>
    {
        if m.label.is_empty() && m.value.is_empty() {
            return self.index_reader.all_postings();
        }

        if m.op == MatchOp::Equal {
            return self.index_reader.postings(&m.label, &[m.value.as_str()]);
        }

        // Fast-path for set matching.
        if m.op == MatchOp::RegexEqual {
            let set_matches = m.set_matches();
            if let Some(matches) = set_matches {
                if !matches.is_empty() {
                    let matches: SmallVec<&str, 6> = matches.iter().map(|s| s.as_str()).collect();
                    return self.index_reader.postings(&m.label, &matches);
                }
            }
        }

        self.index_reader.postings_for_label_matching(&m.label, |s| m.matches(s))
    }

    /// `postings_for_matchers` assembles a single postings iterator against the index
    /// based on the given matchers. The resulting postings are not ordered by series.
    async fn postings_for_matchers_slice<'a>(
        &'a self,
        ms: &'a [Matcher],
    ) -> ProviderResult<PostingsEnum<T::Postings<'a>>>
    {
        if ms.len() == 1 {
            let m = &ms[0];
            if m.label.is_empty() && m.value.is_empty() {
                let all_postings = self.index_reader.all_postings().await?;
                let postings = PostingsEnum::Wrapped(all_postings);
                return Ok(postings);
            }
        }

        let mut not_its_futures: SmallVec<_, 4> = SmallVec::new();
        let mut its_futures: SmallVec<_, 4> = SmallVec::new();
        let mut not_its: SmallVec<_, 4> = SmallVec::new();

        let mut has_subtracting_matchers = false;
        let mut has_intersecting_matchers = false;

        let mut sorted_matchers: SmallVec<(&Matcher, bool, bool), 4> = SmallVec::new();
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
            let all_postings = self.index_reader.all_postings();
            its_futures.push( all_postings );
        }

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
                return Err(ProviderError::MissingMatcher);
            }

            match (m.op, m.value.as_str()) {
                // .* regexp matches any string: do nothing
                (MatchOp::RegexEqual, ".*") => continue,
                // .* regexp does not match any string: return empty
                (MatchOp::RegexNotEqual, ".*") => {
                    return Ok(PostingsEnum::Empty);
                }
                // .+ regexp matches any non-empty string
                (MatchOp::RegexEqual, ".+") => {
                    // .+ regexp matches any non-empty string: get postings for all label values.
                    let it = self.index_reader.postings_for_all_label_values(&m.label);
                    its_futures.push(it);
                }
                (MatchOp::RegexNotEqual, ".+") => {
                    // .+ regexp matches any non-empty string: get postings for all label values and remove them.
                    let it = self.index_reader.postings_for_all_label_values(name);
                    not_its_futures.push(it);
                }
                _ if label_must_be_set.contains(&name.as_str()) => {
                    // If this matcher must be non-empty, we can be smarter.
                    let is_not = typ == MatchOp::NotEqual || m.op == MatchOp::RegexNotEqual;

                    if is_not {
                        let inverse = m
                            .inverse()
                            .map_err(|_| ProviderError::InvalidMatcher(m.to_string()))?;

                        // If the label can't be empty and is a Not, then subtract it out at the end.
                        if matches_empty {
                            // l!="foo"
                            // If the label can't be empty and is a Not and the inner matcher
                            // doesn't match empty, then subtract it out at the end.
                            // NOTE: we resolve immediately here to avoid borrowing issue with inverse
                            // TODO: Find a way to handle this
                            let it = self.postings_for_matcher_internal(&inverse);
                            not_its.push(it);
                        } else {
                            // l!=""
                            // If the label can't be empty and is a Not, but the inner matcher can
                            // be empty we need to use inverse_postings_for_matcher.
                            let it = self.inverse_postings_for_matcher(&inverse);
                            its_futures.push(it);
                        }
                    } else {
                        // l="a", l=~"a|b", l=~"a.b", etc.
                        // Non-Not matcher, use normal `postings_for_matcher`.
                        let it = self.postings_for_matcher_internal(m);
                        its_futures.push(it);
                    }
                }
                _ => {
                    // l="a", l=~"a|b", l=~"a.b", etc.
                    // Non-Not matcher, use normal `postings_for_matcher`.
                    let it = self.postings_for_matcher_internal(m);
                    its_futures.push(it);
                }
            }
        }

        // todo: join! the following 2 statements
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

    fn inverse_postings_for_matcher<'a>(
        &'a self,
        m: &'a Matcher,
    ) -> Pin<Box<dyn Future<Output = Result<T::Postings<'a>, ProviderError>> + Send + 'a>>
    {
        // Fast-path for RegexNotEqual matching.
        // Inverse of a RegexNotEqual is RegexpEqual (double negation).
        // Fast-path for set matching.
        if m.op == MatchOp::RegexNotEqual {
            if let Some(matches) = m.set_matches() {
                if !matches.is_empty() {
                    let matches: SmallVec<&str, 6> = matches.iter().map(|s| s.as_str()).collect();
                    return self.index_reader.postings(&m.label, &matches);
                }
            }
        }

        // Fast-path for NotEqual matching.
        // Inverse of a NotEqual is Equal (double negation).
        if m.op == MatchOp::NotEqual {
            return self.index_reader.postings(&m.label, &[&m.value]);
        }

        // If the matcher being inverted is =~"" or ="", we just want all the values.
        if m.value.is_empty() && (m.op == MatchOp::RegexEqual || m.op == MatchOp::Equal) {
            return self.index_reader.postings_for_all_label_values(&m.label);
        }

        self.index_reader.postings_for_label_matching(&m.label, |s| !m.matches(s))
    }

    async fn run_or_matchers<'a>(&'a self, matchers: &'a [Vec<Matcher>]) -> ProviderResult<PostingsEnum<T::Postings<'a>>>
    {
        if matchers.is_empty() {
            empty_postings().await
        } else if matchers.len() == 1 {
            let m = matchers
                .get(0)
                .expect("Out of bounds error running matchers");
            self.postings_for_matchers_slice(&m).await
        } else {
            let futures: Vec<_> = matchers
                .iter()
                .map(|m| self.postings_for_matchers_slice(m))
                .collect();
            let mut its = try_join_all(futures).await?;
            let it = intersect_iters(&mut its);
            Ok(PostingsEnum::list(it))
        }
    }

    pub async fn label_values_with_matchers(
        &self,
        name: &str,
        matchers: Option<&Matchers>,
    ) -> ProviderResult<Vec<String>>
    {
        let mut all_values = self.index_reader.label_values(name, matchers).await?;

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
            self.postings_for_matchers(matchers).await?
        } else {
            let all_postings = self.index_reader.all_postings().await?;
            PostingsEnum::Wrapped(all_postings)
        };

        let values_postings: Vec<_> = all_values
            .iter()
            .map(|value| self.index_reader.postings(name, &[value.as_str()]))
            .collect();

        let postings = try_join_all(values_postings.into_iter()).await?;
        let indexes = find_intersecting_postings(p, postings)?;
        let mut values = Vec::with_capacity(indexes.len());
        for posting in indexes {
            values.push(all_values[posting.index].clone());
        }

        Ok(values)
    }

}


#[inline]
async fn empty_postings<P>() -> ProviderResult<PostingsEnum<P>>
where
    P: PostingsIterator,
{
    Ok(PostingsEnum::Empty)
}

//type PostingsResult<'a> = BoxFuture<'a, ProviderResult<impl PostingsList>>;

#[inline]
fn is_subtracting_matcher(m: &Matcher, label_must_be_set: &FastHashSet<&str>) -> bool {
    if !label_must_be_set.contains(&m.label.as_str()) {
        return true;
    }
    matches!(m.op, MatchOp::NotEqual | MatchOp::RegexNotEqual) && m.matches("")
}
