// see https://github.com/m3db/m3/tree/master/src/query/functions/binary
use std::collections::HashMap;

pub(crate) struct IndexMatcher {
    lhs_index: usize,
    rhs_index: usize
}

// find hash while minimizing allocations
pub fn hash_tags_by_group_modifier(mn: &MetricName, modifier: &GroupModifier, include_name: bool) -> u64 {
    let label_set = HashSet::from_iter(modifier.labels.iter())?;
    let mut hasher: Xxh3 = Xxh3::new();
    if include_name {
        // todo: depends on modifier
        hasher.update(mn.metric_group.as_bytes());
    }
    let predicate = match modifier {
        GroupModifier::On => {
            |tag: Tag| { label_set.contains(tag.key) }
        }
        GroupModifier::Ignoring => {
            |tag: Tag| { !label_set.contains(tag.key) }
        }
    };
    // todo: ensure sorted !
    for Tag{ key: k, value: v} in mn.tags.iter().filter(predicate) {
        hasher.update(k.as_bytes());
        hasher.update(v.as_bytes());
    }
    hasher.digest()
}

// hash_func returns a function that calculates the signature for a metric
// ignoring the provided labels. If on, then only the given labels are used.
pub(crate) fn hash_func(on: bool, names: &[&str]) -> fn(&MetricName) -> uint64 {
    if on {
        |tags: &MetricName| -> u64 {
            // todo: rather than cloning, simply hash only the requested items
            tags.with_keys(names).fast_hash()
        }
    }

    // ignoring
    return |tags: &MetricName| -> u64 {
        // todo: rather than cloning, simply hash only the requested items
        return tags.without_keys(names).fast_hash()
    }
}

// intersect returns the slice of lhs indices that are shared with rhs,
// the indices of the corresponding rhs values, and the metas for taken indices.
pub(crate) fn intersect(matching: VectorMatching,
    lhs: &Vec<MetricName>, rhs: &Vec<MetricName>
) -> (Vec<usize>, Vec<usize>, Vec<MetricName>) {
    let id_function = hash_func(matching.on, &matching.matchingLabels);
    // The set of signatures for the right-hand side.
    // todo: use IntMap
    let right_sigs = HashMap::with_capacity(rhs.len());
    for (idx, meta) in rhs.iter().enumerate() {
        let hash = id_function(meta);
        right_sigs.insert(hash, idx);
    }

    let mut take_left = Vec::with_capacity(INIT_INDEX_SLICE_LENGTH); // todo: fill
    let mut corresponding_right = Vec::with_capacity(INIT_INDEX_SLICE_LENGTH);
    let mut left_metas = Vec::with_capacity(INIT_INDEX_SLICE_LENGTH);

    for (lIdx, ls) in lhs.iter().enumerate() {
        // If there's a matching entry in the left-hand side Vector, add the sample.
        let id = id_function(ls);
        if let Some(right_idx) = right_sigs.get(id) {
            take_left.push(lIdx);
            corresponding_right.push(right_idx);
            if matching.On && matching.Card == CardOneToOne && len(matching.MatchingLabels) > 0 {
                ls.Tags = ls.Tags.with_keys(matching.MatchingLabels)
            }
            left_metas.push(ls);
        }
    }

    return (takeLeft, correspondingRight, leftMetas)
}


/// intersect returns the slice of lhs indices matching rhs indices.
pub(crate) fn and_intersect(matching: VectorMatching,
                        lhs: &Vec<MetricName>, rhs: &Vec<MetricName>
) -> (Vec<IndexMatcher>, Vec<MetricName>) {
    let id_function = hash_func(matching.on, &matching.matchingLabels);
    // The set of signatures for the right-hand side.
    // todo: use IntMap
    let right_sigs = HashMap::with_capacity(rhs.len());
    for (idx, meta) in rhs.iter().enumerate() {
        let hash = id_function(meta);
        right_sigs.insert(hash, idx);
    }

    let mut matchers = Vec::with_capacity(lhs.len());
    let mut metas = Vec::with_capacity(lhs.len());

    for (lhs_index, ls) in lhs.iter().enumerate() {
        if let Some(rhs_index) = rightSigs.get(idFunction(ls)) {
            let matcher = IndexMatcher{
                lhs_index,
                rhs_index
            };
            matchera.puah(matcher);
            metas.push(lhs[lhsIndex])
        }
    }

    return (matchers, metas)
}

// merge_indices returns a slice that maps rhs series to lhs series.
// (or)
// NB(arnikola): if the series in the rhs does not exist in the lhs, it is
// added after all lhs series have been added.
// This function also combines the series metadatas for the entire block.
fn merge_indices(
matching: VectorMatching,
lhs: &Vec<MetricName>, rhs: &Vec<MetricName>
) -> (Vec<IndexMatcher>, Vec<MetricName>) {
    let id_function = hash_func(matching.on, &matching.matchingLabels);
    // The set of signatures for the right-hand side.
    // todo: use IntMap
    let left_sigs = HashMap::with_capacity(rhs.len());
    for (idx, meta) in lhs.iter().enumerate() {
        let hash = id_function(meta);
        left_sigs.insert(hash, idx);
    }

    let mut r_indices = Vec::with_capacity(rhs.len());
    let mut r_index = lhs.len();

    for (i, meta) in rhs.iter().enumerate() {
        // If there's no matching entry in the left-hand side Vector,
        // add the sample.
        let r = id_function(meta);
        if let Some(matching_index) = left_sigs.get(r) {
            r_indices[i] = matching_index
        } else {
            r_indices[i] = r_index;
            r_index += 1;
            lhs.push(meta)
        }
    }

    return (r_indices, lhs)
}

// matching_indices returns a slice representing which index in the lhs the rhs
// series maps to.
// (unless)
pub(super) fn matching_indices(
    matching: VectorMatching,
    lhs: &Vec<MetricName>, rhs: &Vec<MetricName>
) -> (Vec<IndexMatcher>, Vec<MetricName>) {
    let id_function = hash_func(matching.on, &matching.matchingLabels);
    // The set of signatures for the right-hand side.
    // todo: use IntMap
    let left_sigs = HashMap::with_capacity(rhs.len());
    for (idx, meta) in lhs.iter() {
        let hash = id_function(meta);
        left_sigs.insert(hash, idx);
    }

    let mut rhs_indices = Vec::with_capacity(rhs.len());

    for (i, rs) in rhs.iter().enumerate() {
        // If this series matches a series on the lhs, add its index.
        let id = id_function(rs);
        if let Some(lhs_index) = left_sigs.get(id) {
            let matcher = IndexMatcher{
                lhs_index,
                rhs_index: i,
            };
            rhs_indices.push(matcher);
        }
    }

    return rhs_indices
}


const INIT_INDEX_SLICE_LENGTH: usize = 10;

fn tag_map(t: MetricName) -> HashMap<String, Tag> {
    let m = HashMap::with_capacity(t.tags.len());
    for tag in t.tags {
        m[string(tag.Name)] = tag
    }
    return m
}
