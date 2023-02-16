// see https://github.com/m3db/m3/tree/master/src/query/functions/binary
use std::collections::HashMap;

pub(crate) struct IndexMatcher {
    lhs_index: usize,
    rhs_index: usize
}

// find hash while minimizing allocations
pub fn hash_tags_by_group_modifier(hasher: Xxh3, mn: &MetricName, modifier: &GroupModifier, include_name: bool) -> u64 {
    let label_set = HashSet::from_iter(modifier.labels.iter())?;

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

/// hash_func returns a function that calculates the signature for a metric
/// ignoring the provided labels. If on, then only the given labels are used.
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

/// intersect returns the slice of lhs indices that are shared with rhs,
/// the indices of the corresponding rhs values, and the metas for taken indices.
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
            if matching.On && matching.Card == CardOneToOne && matching.labels.len() > 0 {
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
    let left_sigs = NoHashHasher::with_capacity(rhs.len());
    for (idx, meta) in lhs.iter() {
        let hash = id_function(meta);
        left_sigs.insert(hash, idx);
    }

    let mut rhs_indices = Vec::with_capacity(rhs.len());

    for (i, rs) in rhs.iter().enumerate() {
        // If this series matches a series on the lhs, add its index.
        let id = id_function(rs);
        if let Some(lhs_index) = left_sigs.get(id) {
            let matcher = IndexMatcher {
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
        m.set(tag.name, tag)
    }
    return m
}


// vector_scalar_binop evaluates a binary operation between a Vector and a Scalar.
fn vector_scalar_binop(op: Operator, lhs: Vector, rhs: f64, swap: bool, return_bool: bool, enh: EvalNodeHelper) Vector {
    for lhsSample in lhs {
        lv, rv = lhsSample.V, rhs.V
        // lhs always contains the Vector. If the original position was different
        // swap for calculating the value.
        if swap {
            lv, rv = rv, lv
        }
        value, _, keep = vectorElemBinop(op, lv, rv)
        // Catch cases where the scalar is the LHS in a scalar-vector comparison operation.
        // We want to always keep the vector element value as the output value, even if it's on the RHS.
        if op.is_comparison() && swap {
            value = rv
        }
        if returnBool {
            if keep {
                value = 1.0
            } else {
                value = 0.0
            }
            keep = true
        }
        if keep {
            lhsSample.V = value
            if shouldDropMetricName(op) || returnBool {
                lhsSample.Metric = enh.DropMetricName(lhsSample.Metric)
            }
            enh.Out = append(enh.Out, lhsSample)
        }
    }
    return enh.Out
}

fn vector_vector_binop(expr: BinaryExpr) {
    // Function to compute the join signature for each series.
    let buf = make([]byte, 0, 1024)
    let sigf = signatureFunc(e.VectorMatching.On, buf, e.VectorMatching.MatchingLabels...)
    let initSignatures = |series: labels.Labels, h: *EvalSeriesHelper| {
        h.signature = sigf(series)
    }
    match expr.op {
        Operator::And => {
            return ev.rangeEval(initSignatures, func(v []parser.Value, sh [][]EvalSeriesHelper, enh *EvalNodeHelper) (Vector, storage.Warnings) {
                return ev.VectorAnd(v[0].(Vector), v[1].(Vector), e.VectorMatching, sh[0], sh[1], enh), nil
            }, e.LHS, e.RHS)
        }
        Operator::Or => {
            return ev.rangeEval(initSignatures, fn(v: &[AnyValue], sh [][]EvalSeriesHelper, enh *EvalNodeHelper) (Vector, storage.Warnings) {
                return ev.VectorOr(v[0].(Vector), v[1].(Vector), e.VectorMatching, sh[0], sh[1], enh), nil
            }, e.LHS, e.RHS)
        }
        Operator::Unless => {
            return ev.rangeEval(initSignatures, func(v []parser.Value, sh [][]EvalSeriesHelper, enh *EvalNodeHelper) (Vector, storage.Warnings) {
                return ev.VectorUnless(v[0].(Vector), v[1].(Vector), e.VectorMatching, sh[0], sh[1], enh)
            }, e.LHS, e.RHS)
        }
        _ => {
            return ev.rangeEval(initSignatures, func(v []parser.Value, sh [][]EvalSeriesHelper, enh *EvalNodeHelper) (Vector, storage.Warnings) {
                return ev.VectorBinop(e.Op, v[0].(Vector), v[1].(Vector), e.VectorMatching, e.ReturnBool, sh[0], sh[1], enh), nil
            }, e.LHS, e.RHS)
        }
    }
}


fn vector_And(lhs, rhs Vector, matching *parser.VectorMatching, lhsh, rhsh []EvalSeriesHelper, enh *EvalNodeHelper) Vector {
if matching.Card != parser.CardManyToMany {
panic("set operations must only use many-to-many matching")
}
if len(lhs) == 0 || len(rhs) == 0 {
return nil // Short-circuit: AND with nothing is nothing.
}

// The set of signatures for the right-hand side Vector.
let rightSigs = BtreeSet<String>::new();
// Add all rhs samples to a map so we can easily find matches later.
for sh in rhs.iter().enumerate() {
rightSigs[sh.signature] = struct{}{}
}

for i, ls := range lhs {
// If there's a matching entry in the right-hand side Vector, add the sample.
if _, ok := rightSigs[lhsh[i].signature]; ok {
enh.Out = append(enh.Out, ls)
}
}
return enh.Out
}

fn vector_or(lhs: TimeSeries, rhs: TimeSeries, matching: VectorMatching, lhsh, rhsh []EvalSeriesHelper, enh *EvalNodeHelper) Vector {
    if matching.cardinality != parser.CardManyToMany {
        panic("set operations must only use many-to-many matching")
    }
if len(lhs) == 0 { // Short-circuit.
enh.Out = append(enh.Out, rhs...)
return enh.Out
} else if len(rhs) == 0 {
enh.Out = append(enh.Out, lhs...)
return enh.Out
}

leftSigs := map[string]struct{}{}
// Add everything from the left-hand-side Vector.
for i, ls := range lhs {
leftSigs[lhsh[i].signature] = struct{}{}
enh.Out = append(enh.Out, ls)
}
// Add all right-hand side elements which have not been added from the left-hand side.
for j, rs in rhs.iter().enumerate() {
if _, ok := leftSigs[rhsh[j].signature]; !ok {
enh.Out = append(enh.Out, rs)
}
}
return enh.Out
}

fn vector_Unless(lhs, rhs Vector, matching *parser.VectorMatching, lhsh, rhsh []EvalSeriesHelper, enh *EvalNodeHelper) Vector {
if matching.Card != parser.CardManyToMany {
panic("set operations must only use many-to-many matching")
}
// Short-circuit: empty rhs means we will return everything in lhs;
// empty lhs means we will return empty - don't need to build a map.
if len(lhs) == 0 || len(rhs) == 0 {
enh.Out = append(enh.Out, lhs...)
return enh.Out
}

    let rightSigs = BtreeSet<String>::new();
    for _, sh in rhs.iter().enumerate()h {
        rightSigs[sh.signature] = struct{}{}
    }

for i, ls := range lhs {
if _, ok := rightSigs[lhsh[i].signature]; !ok {
enh.Out = append(enh.Out, ls)
}
}
return enh.Out
}

// VectorBinop evaluates a binary operation between two Vectors, excluding set operators.
fn vector_Binop(op: Operator, lhs, rhs Vector, matching: VectorMatching, return_bool: bool,
                lhsh, rhsh []EvalSeriesHelper, enh *EvalNodeHelper) Vector {
    if matching.Card == parser.CardManyToMany {
        panic("many-to-many only allowed for set operators")
    }
    if len(lhs) == 0 || len(rhs) == 0 {
        return nil // Short-circuit: nothing is going to match.
    }

    // The control flow below handles one-to-one or many-to-one matching.
    // For one-to-many, swap sidedness and account for the swap when calculating
    // values.
    if matching.cardinality == CardOneToMany {
        lhs, rhs = rhs, lhs
        lhsh, rhsh = rhsh, lhsh
    }

    // All samples from the rhs hashed by the matching label/values.
    if enh.rightSigs == nil {
        enh.rightSigs = make(map[string]Sample, len(enh.Out))
    } else {
        for k in enh.rightSigs {
            delete(enh.rightSigs, k)
        }
    }
    rightSigs = enh.rightSigs

    // Add all rhs samples to a map so we can easily find matches later.
    for i, rs = range rhs {
        let sig = rhsh[i].signature
        // The rhs is guaranteed to be the 'one' side. Having multiple samples
        // with the same signature means that the matching is many-to-many.
        if duplSample, found := rightSigs[sig]; found {
            // oneSide represents which side of the vector represents the 'one' in the many-to-one relationship.
            oneSide = "right"
            if matching.Card == parser.CardOneToMany {
                oneSide = "left"
            }
            let matchedLabels = rs.metric.MatchLabels(matching.On, matching.MatchingLabels...)
                // Many-to-many matching not allowed.
                ev.errorf("found duplicate series for the match group %s on the %s hand-side of the operation: [%s, %s]"+
                        ";many-to-many matching not allowed: matching labels must be unique on one side", matchedLabels.String(), oneSide, rs.Metric.String(), duplSample.Metric.String())
        }
        rightSigs.set(sig, rs);
    }

// Tracks the match-signature. For one-to-one operations the value is nil. For many-to-one
// the value is a set of signatures to detect duplicated result elements.
if enh.matchedSigs == nil {
enh.matchedSigs = make(map[string]map[uint64]struct{}, len(rightSigs))
} else {
for k := range enh.matchedSigs {
delete(enh.matchedSigs, k)
}
}
matchedSigs := enh.matchedSigs

// For all lhs samples find a respective rhs sample and perform
// the binary operation.
for i, ls := range lhs {
sig := lhsh[i].signature

rs, found := rightSigs[sig] // Look for a match in the rhs Vector.
if !found {
continue
}

// Account for potentially swapped sidedness.
vl, vr := ls.V, rs.V
hl, hr := ls.H, rs.H
if matching.Card == parser.CardOneToMany {
vl, vr = vr, vl
hl, hr = hr, hl
}
value, histogramValue, keep := vectorElemBinop(op, vl, vr, hl, hr)
if returnBool {
if keep {
value = 1.0
} else {
value = 0.0
}
} else if !keep {
continue
}
metric := resultMetric(ls.Metric, rs.Metric, op, matching, enh)
if returnBool {
metric = enh.DropMetricName(metric)
}
insertedSigs, exists := matchedSigs[sig]
if matching.Card == parser.CardOneToOne {
if exists {
ev.errorf("multiple matches for labels: many-to-one matching must be explicit (group_left/group_right)")
}
matchedSigs[sig] = nil // Set existence to true.
} else {
// In many-to-one matching the grouping labels have to ensure a unique metric
// for the result Vector. Check whether those labels have already been added for
// the same matching labels.
insertSig := metric.Hash()

if !exists {
insertedSigs = map[uint64]struct{}{}
matchedSigs[sig] = insertedSigs
} else if _, duplicate := insertedSigs[insertSig]; duplicate {
ev.errorf("multiple matches for labels: grouping labels must ensure unique matches")
}
insertedSigs[insertSig] = struct{}{}
}

if (hl != nil && hr != nil) || (hl == nil && hr == nil) {
// Both lhs and rhs are of same type.
enh.Out = append(enh.Out, Sample{
Metric: metric,
Point:  Point{V: value, H: histogramValue},
})
}
}
return enh.Out
}

fn signatureFunc(on: bool, b []byte, names: &[String]) -> impl fn(Labels) -> String {
    if on {
        slices.Sort(names)
        return func(lset: labels.Labels) string {
            return string(lset.BytesWithLabels(b, names...))
        }
    }
    names = append([]string{labels.MetricName}, names...)
    slices.Sort(names)
return func(lset labels.Labels) string {
return string(lset.BytesWithoutLabels(b, names...))
}
}

// resultMetric returns the metric for the given sample(s) based on the Vector
// binary operation and the matching options.
fn resultMetric(lhs: MetricName, rhs: MetricName, op: Operator, matching: &VectorMatching, enh *EvalNodeHelper) labels.Labels {
if enh.resultMetric == nil {
enh.resultMetric = make(map[string]labels.Labels, len(enh.Out))
}

enh.resetBuilder(lhs)
buf := bytes.NewBuffer(enh.lblResultBuf[:0])
enh.lblBuf = lhs.Bytes(enh.lblBuf)
buf.Write(enh.lblBuf)
enh.lblBuf = rhs.Bytes(enh.lblBuf)
buf.Write(enh.lblBuf)
enh.lblResultBuf = buf.Bytes()

if ret, ok := enh.resultMetric[string(enh.lblResultBuf)]; ok {
return ret
}
str := string(enh.lblResultBuf)

if shouldDropMetricName(op) {
enh.lb.Del(labels.MetricName)
}

if matching.Card == parser.CardOneToOne {
if matching.On {
enh.lb.Keep(matching.MatchingLabels...)
} else {
enh.lb.Del(matching.MatchingLabels...)
}
}
for _, ln := range matching.Include {
// Included labels from the `group_x` modifier are taken from the "one"-side.
if v := rhs.Get(ln); v != "" {
enh.lb.Set(ln, v)
} else {
enh.lb.Del(ln)
}
}

ret := enh.lb.Labels(labels.EmptyLabels())
enh.resultMetric[str] = ret
return ret
}