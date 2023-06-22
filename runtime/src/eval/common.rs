// see https://github.com/m3db/m3/tree/master/src/query/functions/binary
use std::collections::{BTreeMap, BTreeSet};
use std::collections::{HashMap, HashSet};

use super::parser::ast::LabelMatching;
use crate::model::{LabelName, Labels, LabelsTrait, SampleValue, Timestamp};

// Every Expr can be evaluated to a value.
#[derive(Debug)]
pub enum QueryValue {
    InstantVector(InstantVector),
    Matrix(RangeVector),
    Scalar(SampleValue),
    String(String)
}

#[derive(Debug, PartialEq)]
pub(super) enum QueryValueKind {
    InstantVector,
    Matrix,
    Scalar,
    String
}

pub(super) trait QueryValueIter: Iterator<Item = QueryValue> {
    fn value_kind(&self) -> QueryValueKind;
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub timestamp: Timestamp,
    pub value: SampleValue,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InstantVector {
    instant: Timestamp,
    // labels, values
    values: Vec<f64>,
    samples: Vec<(Labels, SampleValue)>,
}

impl InstantVector {
    pub fn new(instant: Timestamp, samples: Vec<(Labels, SampleValue)>) -> Self {
        Self { 
            instant,
            values: samples.iter().map(|(_, v)| *v).collect(),
            samples
        }
    }

    #[inline]
    pub fn timestamp(&self) -> Timestamp {
        self.instant
    }

    #[inline]
    pub fn samples(&self) -> &[(Labels, SampleValue)] {
        &self.samples
    }

    pub fn apply_scalar_op(
        &mut self,
        op: impl Fn(SampleValue) -> Option<SampleValue>,
        keep_name: bool,
    ) -> Self {
        let samples = self
            .samples
            .iter()
            .cloned()
            .filter_map(|(mut labels, value)| match op(value) {
                Some(value) => {
                    if !keep_name {
                        labels.drop_name();
                    }
                    Some((labels, value))
                }
                None => None,
            })
            .collect();
        InstantVector::new(self.instant, samples)
    }

    pub fn apply_vector_op_one_to_one(
        &self,
        op: impl Fn(SampleValue, SampleValue) -> Option<SampleValue>,
        other: &InstantVector,
        label_matching: Option<&LabelMatching>,
        keep_name: bool,
    ) -> Self {
        assert!(self.instant == other.instant);

        let mut rhs = HashMap::new();
        for (labels, value) in other.samples.iter() {
            let matched_labels = match label_matching {
                Some(LabelMatching::On(names)) => labels.with(names),
                Some(LabelMatching::Ignoring(names)) => labels.without(names),
                None => labels.without(&HashSet::new()),
            };

            if let Some(duplicate) = rhs.insert(matched_labels.to_vec(), value) {
                // TODO: replace with error
                panic!(
                    "Found series collision for matching labels ({:?}).\nFirst: {:#?}\nSecond: {:#?}",
                    label_matching, duplicate, matched_labels
                );
            }
        }

        let mut samples = Vec::new();
        let mut already_matched = HashSet::new();
        for (labels, lvalue) in self.samples.iter() {
            let mut matched_labels = match label_matching {
                Some(LabelMatching::On(names)) => labels.with(names),
                Some(LabelMatching::Ignoring(names)) => labels.without(names),
                None => labels.without(&HashSet::new()),
            };

            let signature = matched_labels.to_vec();
            let rvalue = match rhs.get(&signature) {
                Some(rvalue) => rvalue,
                None => continue,
            };

            let sample = match op(*lvalue, **rvalue) {
                Some(sample) => sample,
                None => continue,
            };
            if !already_matched.insert(signature) {
                // TODO: replace with error
                return Err(
                    RuntimeError::General(
                    "Many-to-one matching detected! If it's desired, use explicit group_left/group_right modifier".to_string()
                    )
                );
            }

            if keep_name {
                if let Some(name) = labels.name() {
                    matched_labels.set_name(name.to_string());
                }
            }

            samples.push((matched_labels, sample));
        }

        InstantVector::new(self.instant, samples)
    }

    pub fn apply_vector_op_one_to_many(
        &self,
        _op: impl Fn(SampleValue, SampleValue) -> Option<SampleValue>,
        other: &InstantVector,
        _label_matching: Option<&LabelMatching>,
        _include_labels: &[LabelName],
    ) -> Self {
        assert_eq!(self.instant, other.instant);
        unimplemented!();
    }

    pub fn apply_vector_op_many_to_one(
        &self,
        op: impl Fn(SampleValue, SampleValue) -> Option<SampleValue>,
        other: &InstantVector,
        label_matching: Option<&LabelMatching>,
        include_labels: &[LabelName],
    ) -> Self {
        other.apply_vector_op_one_to_many(|l, r| op(r, l), self, label_matching, include_labels)
    }
}

#[derive(Debug)]
pub struct RangeVector {
    instant: Timestamp,
    samples: Vec<(Labels, Vec<(SampleValue, Timestamp)>)>,
}

impl RangeVector {
    pub fn new(instant: Timestamp, samples: Vec<(Labels, Vec<(SampleValue, Timestamp)>)>) -> Self {
        Self { instant, samples }
    }

    #[inline]
    pub fn timestamp(&self) -> Timestamp {
        self.instant
    }

    #[inline]
    pub fn samples(&self) -> &[(Labels, Vec<(SampleValue, Timestamp)>)] {
        &self.samples
    }
}

pub(crate) struct IndexMatcher {
    lhs_index: usize,
    rhs_index: usize,
}

type HashFunction = fn(&MetricName) -> u64;

/// hash_func returns a function that calculates the signature for a metric
/// ignoring the provided labels. If on, then only the given labels are used.
pub(crate) fn hash_func(on: bool, names: &[&str]) -> HashFunction {
    if on {
        |tags: &metric_name| -> u64 {
            // todo: rather than cloning, simply hash only the requested items
            tags.with_keys(names).fast_hash()
        }
    }

    // ignoring
    return |tags: &metric_name| -> u64 {
        // todo: rather than cloning, simply hash only the requested items
        return tags.without_keys(names).fast_hash();
    };
}

/// intersect returns the slice of lhs indices that are shared with rhs,
/// the indices of the corresponding rhs values, and the metas for taken indices.
pub(crate) fn intersect(
    matching: VectorMatching,
    lhs: &Vec<MetricName>,
    rhs: &Vec<MetricName>,
) -> (Vec<usize>, Vec<usize>, Vec<MetricName>) {
    let hasher = Xxh3::new();
    let id_function = hash_func(matching.on, &matching.matchingLabels);
    // The set of signatures for the right-hand side.
    let mut right_sigs = IntMap::with_capacity(rhs.len());
    for (idx, meta) in rhs.iter().enumerate() {
        let hash = meta.get_hash_by_group_modifier(&hasher, &matching.group_modifier);
        right_sigs.insert(hash, idx);
    }

    let mut take_left = Vec::with_capacity(INIT_INDEX_SLICE_LENGTH); // todo: fill
    let mut corresponding_right = Vec::with_capacity(INIT_INDEX_SLICE_LENGTH);
    let mut left_metas = Vec::with_capacity(INIT_INDEX_SLICE_LENGTH);

    for (lIdx, ls) in lhs.iter().enumerate() {
        // If there's a matching entry in the left-hand side Vector, add the sample.
        let id = ls.get_hash_by_group_modifier(&hasher, &matching.groupModifier);
        if let Some(right_idx) = right_sigs.get(id) {
            take_left.push(lIdx);
            corresponding_right.push(right_idx);
            if matching.On && matching.card == VectorMatchingCardinality::OneToOne && matching.labels.len() > 0 {
                ls.tags = ls.tags.with_keys(matching.MatchingLabels)
            }
            left_metas.push(ls);
        }
    }

    return (takeLeft, correspondingRight, leftMetas);
}


/// intersect returns the slice of lhs indices matching rhs indices.
pub(crate) fn and_intersect(matching: VectorMatching,
                            lhs: &Vec<MetricName>, rhs: &Vec<MetricName>,
) -> (Vec<IndexMatcher>, Vec<MetricName>) {
    let hasher = Xxh3::new();

    // The set of signatures for the right-hand side.
    // todo: use IntMap
    let right_sigs = HashMap::with_capacity(rhs.len());
    for (idx, meta) in rhs.iter().enumerate() {
        let hash = meta.get_hash_by_group_modifier(&hasher, &matching.groupModifier);
        right_sigs.insert(hash, idx);
    }

    let mut matchers = Vec::with_capacity(lhs.len());
    let mut metas = Vec::with_capacity(lhs.len());

    for (lhs_index, ls) in lhs.iter().enumerate() {
        let id = ls.get_hash_by_group_modifier(&hasher, &matching.groupModifier);
        if let Some(rhs_index) = right_sigs.get(id) {
            let matcher = IndexMatcher {
                lhs_index,
                rhs_index,
            };
            matchers.push(matcher);
            metas.push(lhs[lhsIndex])
        }
    }

    return (matchers, metas);
}

// merge_indices returns a slice that maps rhs series to lhs series.
// (or)
// NB(arnikola): if the series in the rhs does not exist in the lhs, it is
// added after all lhs series have been added.
// This function also combines the series metadatas for the entire block.
fn merge_indices(
    matching: VectorMatching,
    lhs: &Vec<MetricName>,
    rhs: &Vec<MetricName>,
) -> (Vec<IndexMatcher>, Vec<metric_name>) {
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
            r_indices.push(matching_index);
        } else {
            r_indices.push(r_index);
            r_index += 1;
            lhs.push(meta)
        }
    }

    return (r_indices, lhs);
}

// matching_indices returns a slice representing which index in the lhs the rhs
// series maps to.
// (unless)
pub(super) fn matching_indices(
    matching: VectorMatching,
    lhs: &Vec<MetricName>,
    rhs: &Vec<MetricName>,
) -> (Vec<IndexMatcher>, Vec<MetricName>) {
    let id_function = hash_func(matching.on, &matching.matchingLabels);
    // The set of signatures for the right-hand side.

    let left_sigs = get_hash_map(&lhs, &matching.groupModifier);

    let mut rhs_indices = Vec::with_capacity(rhs.len());

    for (rhs_index, rs) in rhs.iter().enumerate() {
        // If this series matches a series on the lhs, add its index.
        let id = meta.get_hash_by_group_modifier(&hasher, &matching.groupModifier);
        if let Some(lhs_index) = left_sigs.get(id) {
            let matcher = IndexMatcher {
                lhs_index,
                rhs_index,
            };
            rhs_indices.push(matcher);
        }
    }

    return rhs_indices;
}


fn get_hash_map(rhs: &Vec<MetricName>, modifier: &GroupModifier) -> IntMap<u64> {
    let sigs = IntMap::with_capacity(rhs.len());
    for (idx, meta) in rhs.iter().enumerate() {
        let hash = meta.get_hash_by_group_modifier(&hasher, modifier);
        sigs.insert(hash, idx);
    }
    sigs
}

const INIT_INDEX_SLICE_LENGTH: usize = 10;

fn tag_map(t: metric_name) -> HashMap<String, Tag> {
    let m = HashMap::with_capacity(t.tags.len());
    for tag in t.tags {
        m.set(tag.name, tag)
    }
    return m;
}


// vector_scalar_binop evaluates a binary operation between a Vector and a Scalar.
fn vector_scalar_binop(op: Operator,
                       lhs: Vector,
                       rhs: f64,
                       swap: bool,
                       return_bool: bool,
                       enh: EvalNodeHelper) -> Vector {
    let should_swap_value = swap && op.is_comparison();
    for lhsSample in lhs {
        let (mut lv, mut rv) = (lhsSample.v, rhs);
        // lhs always contains the Vector. If the original position was different
        // swap for calculating the value.
        if swap {
            (lv, rv) = (rv, lv);
        }
        let (mut value, keep) = vector_elem_binop(op, lv, rv);
        // Catch cases where the scalar is the LHS in a scalar-vector comparison operation.
        // We want to always keep the vector element value as the output value, even if it's on the RHS.
        if should_swap_value {
            value = rv
        }
        if return_bool {
            if keep {
                value = 1.0
            } else {
                value = 0.0
            }
            keep = true
        }
        if keep {
            lhsSample = value;
            if shouldDropmetric_name(op) || return_bool {
                lhsSample.metric.reset_metric_group()
            }
            enh.out.push(lhsSample)
        }
    }
    return enh.Out;
}

fn vector_vector_binop(expr: BinaryExpr) {
    // Function to compute the join signature for each series.
    let buf = vec![0_u8; 1024];
    let sigf = signature_func(e.VectorMatching.On, buf, e.VectorMatching.MatchingLabels...)
    let init_signatures = |series: Labels, h: &EvalSeriesHelper | {
        h.signature = sigf(series)
    };
    match expr.op {
        Operator::And => {
            return ev.rangeEval(init_signatures, |v: &Value, sh,
                                                  EvalSeriesHelper, enh: &EvalNodeHelper| -> RuntimeResult < Vector > {
                return ev.VectorAnd(v[0], v[1], e.VectorMatching, sh[0], sh[1], enh);
            }, e.LHS, e.RHS)
        }
        Operator::Or => {
            return ev.rangeEval(init_signatures, | v: &[AnyValue], sh
            [], enh: &EvalNodeHelper) -> Vector
            {
                return ev.VectorOr(v[0].(Vector), v[1].(Vector), e.VectorMatching, sh[0], sh[1], enh);
            }, e.LHS, e.RHS)
        }
        Operator::Unless => {
            return ev.rangeEval(init_signatures, |v: &Value, sh, enh: &EvalNodeHelper| -> RuntimeResult < Vector > {
                return vector_unless(v[0], v[1], e.VectorMatching, sh[0], sh[1], enh);
            }, e.LHS, e.RHS);
        }
        _ => {
            return ev.rangeEval(init_signatures, |v: &Value, sh, enh: EvalNodeHelper| -> Vector {
                return ev.VectorBinop(e.Op,
                v[0],
                v[1],
                e.VectorMatching,
                e.return_bool,
                sh[0],
                sh[1],
                enh)
            }, e.LHS, e.RHS);
        }
    }
}


fn vector_and(lhs: Vector, rhs: Vector, matching: VectorMatching, enh: EvalNodeHelper) -> Vector {
    if matching.card != VectorCardinality::ManyToMany {
        panic("set operations must only use many-to-many matching")
    }
    if lhs.len() == 0 || rhs.len() == 0 {
        return nil; // Short-circuit: AND with nothing is nothing.
    }

    // The set of signatures for the right-hand side Vector.
    let right_sigs = IntMap::default();
    // Add all rhs samples to a map so we can easily find matches later.
    for sh in rhs.iter() {
        right_sigs.insert(sh.signature);
    }

    for (ls, lh) in lhs.iter().zip(lhsh.iter()) {
        // If there's a matching entry in the right-hand side Vector, add the sample.
        if right_sigs.has(lh.signature) {
            enh.out.push(ls)
        }
    }

    return enh.Out;
}

fn vector_or(
    lhs: &mut TimeSeries,
    rhs: TimeSeries,
    matching: VectorMatching,
    lhsh, rhsh []EvalSeriesHelper, enh: EvalNodeHelper) -> ReturnResult<Vector> {
    if matching.cardinality != parser.CardManyToMany {
        return Err(RuntimeError::General("set operations must only use many-to-many matching"));
    }
    if lhs.len() == 0 { // Short-circuit.
        enh.Out.push(rhs...)
        return enh.Out;
    } else if rhs.len() == 0 {
        enh.out = append(enh.Out, lhs...)
        return enh.Out;
    }

    let left_sigs = IntSet::default();
    // Add everything from the left-hand-side Vector.
    for (i, ls) in lhs {
        left_sigs.push(lhsh[i].signature);
        enh.Out.push(ls)
    }
    // Add all right-hand side elements which have not been added from the left-hand side.
    for (j, rs) in rhs.iter().enumerate() {
        if !left_sigs.has(rhsh[j].signature) {
            enh.Out = append(enh.Out, rs)
        }
    }

    return enh.Out;
}

fn vector_unless(
    lhs: Vector,
    rhs: Vector,
    matching: &VectorMatching, lhsh, rhsh []EvalSeriesHelper, enh: EvalNodeHelper) -> Vector {
    if matching.card != VectorMatchCardinality::ManyToMany {
        panic("set operations must only use many-to-many matching")
    }
    // Short-circuit: empty rhs means we will return everything in lhs;
    // empty lhs means we will return empty - don't need to build a map.
    if lhs.len() == 0 || rhs.len() == 0 {
        enh.Out = append(enh.Out, lhs...)
        return enh.Out;
    }

    let right_sigs: IntSet<u64> = IntSet::default();
    for sh in rhs.iter().enumerate() {
        right_sigs.push(sh.signature);
    }

    for (ls, lhsh_v) in lhs.iter().zip(lhsh.iter()) {
        if right_sigs.has(lhsh_v.signature) {
            enh.out.push(ls)
        }
    }

    return enh.Out;
}

// VectorBinop evaluates a binary operation between two Vectors, excluding set operators.
fn vector_binop(op: Operator,
                lhs: Vector,
                rhs: Vector,
                matching: VectorMatching,
                return_bool: bool,
                lhsh: Vec<EvalSeriesHelper>,
                rhsh: Vec<EvalSeriesHelper>,
                enh: &mut EvalNodeHelper) -> RuntimeResult<Vector> {
    
    if matching.card == VectorMatchingCardinality::ManyToMany {
        panic("many-to-many only allowed for set operators")
    }
    if lhs.len() == 0 || rhs.len() == 0 {
        return Ok(()); // Short-circuit: nothing is going to match.
    }

    // The control flow below handles one-to-one or many-to-one matching.
    // For one-to-many, swap sidedness and account for the swap when calculating
    // values.
    if matching.cardinality == VectorCardinality::OneToMany {
        (lhs, rhs) = (rhs, lhs);
    }

    // All samples from the rhs hashed by the matching label/values.
    enh.right_sigs.clear();
    right_sigs = enh.right_sigs;

    let mut one_side = "left";

    // Add all rhs samples to a map so we can easily find matches later.
    for (i, rs) in rhs.iter().enumerate() {
        let sig = rhsh[i].signature;
        // The rhs is guaranteed to be the 'one' side. Having multiple samples
        // with the same signature means that the matching is many-to-many.
        if let Some(duplicate_sample) = right_sigs.get(sig) {
            // one_side represents which side of the vector represents the 'one' in the many-to-one relationship.
            one_side = "right";
            if matching.card == VectorCardinality::OneToMany {
                one_side = "left"
            }
            let matched_labels = rs.metric.match_labels(matching.On, matching.MatchingLabels...)
            // Many-to-many matching not allowed.
            ev.errorf("found duplicate series for the match group {} on the {} hand-side of the operation: [{}, {}]" +
                          ";many-to-many matching not allowed: matching labels must be unique on one side",
                      matched_labels.to_string(), one_side, rs.metric,
                      duplicate_sample.metric)
        }
        right_sigs.set(sig, rs);
    }

    // Tracks the match-signature. For one-to-one operations the value is nil. For many-to-one
    // the value is a set of signatures to detect duplicated result elements.
    let mut matched_sigs: HashMap<String, NoHashSet<u64>> = HashMap::with_capacity(right_sigs.len());

    // For all lhs samples find a respective rhs sample and perform
    // the binary operation.
    for (i, ls) in lhs.iter().enumerate() {
        let sig = lhsh[i].signature;
        let rs = right_sigs.get(sig); // Look for a match in the rhs Vector.
        if !rs.is_some() {
            continue;
        }

        // Account for potentially swapped sidedness.
        let (mut vl, mut vr) = (ls.V, rs.V);

        if matching.card == VectorCardinality::OneToMany {
            (vl, vr) = (vr, vl);
        }

        let (value, keep) = vector_elem_binop(op, vl, vr);
        if return_bool {
            if keep {
                value = 1.0
            } else {
                value = 0.0
            }
        } else if !keep {
            continue;
        }

        let metric = result_metric(ls.metric, rs.metric, op, matching, enh);
        if return_bool {
            metric.reset_metric_name();
        }
        let inserted_sigs = matched_sigs.get(sig);
        let exists = inserted_sigs.is_some();
        if matching.card == VectorCardinality::OneToOne {
            if exists {
                ev.errorf("multiple matches for labels: many-to-one matching must be explicit (group_left/group_right)")
            }
            matched_sigs[sig] = true // Set existence to true.
        } else {
            // In many-to-one matching the grouping labels have to ensure a unique metric
            // for the result Vector. Check whether those labels have already been added for
            // the same matching labels.
            let insert_sig = metric.get_hash();

            if !exists {
                inserted_sigs = IntSet::default();
                inserted_sigs.insert(insert_sig)?;
                matched_sigs[sig] = inserted_sigs
            } else if inserted_sigs.has(insert_sig) {
                ev.errorf("multiple matches for labels: grouping labels must ensure unique matches")
            } else {
                inserted_sigs.insert(insert_sig)?;
            }
        }

        if (hl != nil && hr != nil) || (hl == nil && hr == nil) {
            // Both lhs and rhs are of same type.
            enh.out.push(Sample { metric, value });
        }
    }
    return enh.Out;
}

fn signature_func(on: bool, names: &[String]) -> fn(MetricName) -> u64 {
    let mut hasher: Xxh3 = Xxh3::new();
    let buf: Vec<u8> = Vec::with_capacity(512);

    if on {
        return move |lset: MetricName| -> u64 {
            lset.hash_with_labels(&mut hasher, buf, names)
        };
    }
    let mut names = Vec::from(names);
    names.push(METRIC_NAME_LABEL);
    names.sort();

    move |lset: MetricName| -> u64 {
        lset.hash_without_labels(&mut hasher, buf, names)
    }
}

// result_metric returns the metric for the given sample(s) based on the Vector
// binary operation and the matching options.
fn result_metric(
    lhs: MetricName,
    rhs: MetricName,
    op: Operator,
    matching: &VectorMatching,
    enh: EvalNodeHelper) -> MetricName {
    if enh.resultMetric == nil {
        enh.resultMetric = HashMap::with_capacity(enh.out.len());
    }

    enh.resetBuilder(lhs);
    let buf = bytes.NewBuffer(enh.lblResultBuf[: 0])
    enh.lblBuf = lhs.Bytes(enh.lblBuf);
    buf.Write(enh.lblBuf);
    enh.lblBuf = rhs.Bytes(enh.lblBuf);
    buf.Write(enh.lblBuf);
    enh.lblResultBuf = buf.Bytes();

    if ret, ok: = enh.resultMetric[enh.lblResultBuf.toString()];
    ok {
        return ret,
    }
    str = string(enh.lblResultBuf);

    if should_dropmetric_name(op) {
        enh.lb.del(labels.metric_name);
    }

    if matching.card == parser.CardOneToOne {
        if matching.On {
            enh.lb.Keep(matching.MatchingLabels...)
        } else {
            enh.lb.Del(matching.MatchingLabels...)
        }
    }
    for ln in matching.include {
        // Included labels from the `group_x` modifier are taken from the "one"-side.
        if let Some(v) = rhs.get(ln) {
            enh.lb.set(ln, v)
        } else {
            enh.lb.del(ln)
        }
    }

    let ret = enh.lb.Labels(labels.EmptyLabels());
    enh.resultMetric[str] = ret;
    return ret;
}


// rangeEval evaluates the given expressions, and then for each step calls
// the given funcCall with the values computed for each expression at that
// step. The return value is the combination into time series of all the
// function call results.
// The prepSeries function (if provided) can be used to prepare the helper
// for each series, then passed to each call funcCall.
fn rangeEval(ev: &Evaluator,
             prepSeries: |labels: Labels, sh: EvalSeriesHelper| -> EvalSeriesHelper,
            funcCall: Fn(values: &[Value], [][]EvalSeriesHelper, *EvalNodeHelper) -> Vector,
            exprs: &[Expr]) -> RuntimeResult<Matrix> {
    let numSteps = int((ev.endTimestamp-ev.startTimestamp)/ev.interval) + 1
    let matrixes = make([]Matrix, exprs.len())
    let origMatrixes = make([]Matrix, exprs.len())
    let originalNumSamples = ev.currentSamples

    var warnings storage.Warnings
    for (i, e) in exprs.iter().enumerate() {
        // Functions will take string arguments from the expressions, not the values.
        if e != nil && e.Type() != parser.ValueTypeString {
            // ev.currentSamples will be updated to the correct value within the ev.eval call.
            val, ws := ev.eval(e);
            warnings = append(warnings, ws...)
            matrixes[i] = val.(Matrix)

            // Keep a copy of the original point slices so that they
            // can be returned to the pool.
            origMatrixes[i] = make(Matrix, len(matrixes[i]))
            copy(origMatrixes[i], matrixes[i])
        }
    }

    vectors := make([]Vector, exprs.len())    // Input vectors for the function.
    args := make([]parser.Value, exprs.len()) // Argument to function.
    // Create an output vector that is as big as the input matrix with
    // the most time series.
    let mut biggestLen = 1
    for i in 0 .. exprs.len() {
        let len = matrixes[i].len();
        vectors[i] = make(Vector, 0, len)
        if len > biggestLen {
            biggestLen = len
        }
    }
    enh := &EvalNodeHelper{Out: make(Vector, 0, biggestLen)}
    seriess := make(map[uint64]Series, biggestLen) // Output series by series hash.
    tempNumSamples = ev.currentSamples

    var (
    seriesHelpers [][]EvalSeriesHelper
    bufHelpers    [][]EvalSeriesHelper // Buffer updated on each step
    )

    // If the series preparation function is provided, we should run it for
    // every single series in the matrix.
    if let Some(prepSeries) = prepSeries {
        seriesHelpers = make([][]EvalSeriesHelper, exprs.len())
        bufHelpers = make([][]EvalSeriesHelper, exprs.len())

        for i := range exprs {
            let len = maxtrixes[i].len();
            seriesHelpers[i] = make([]EvalSeriesHelper, len)
            bufHelpers[i] = make([]EvalSeriesHelper, len)

            for (si, series) in matrixes[i] {
                let h = seriesHelpers[i][si]
                prepSeries(series.Metric, &h)
                seriesHelpers[i][si] = h
            }
        }
    }

    for ts = ev.startTimestamp; ts <= ev.endTimestamp; ts += ev.interval {
        contextDone(ev.ctx, "expression evaluation")?;
        // Reset number of samples in memory after each timestamp.
        ev.currentSamples = tempNumSamples
        // Gather input vectors for this timestamp.
        for (i, expr) in exprs.iter().enumerate() {
            vectors[i] = vectors[i][:0]
            if prepSeries != nil {
                bufHelpers[i] = bufHelpers[i][:0]
            }

            for (si, series) in matrixes[i].iter().enumerate() {
                for point in series.points {
                    if point.T == ts {
                        if ev.currentSamples < ev.maxSamples {
                            vectors[i].push(Sample{Metric: series.Metric, point: point})
                            if prepSeries != nil {
                                bufHelpers[i].push(seriesHelpers[i][si])
                            }

                            // Move input vectors forward so we don't have to re-scan the same
                            // past points at the next step.
                            matrixes[i][si].points = series.points[1:]
                            ev.currentSamples++
                        } else {
                            ev.error(ErrTooManySamples(env))
                    }
                }
                break
            }
        }
        args[i] = vectors[i]
        ev.samplesStats.UpdatePeak(ev.currentSamples)
    }

    // Make the function call.
    enh.Ts = ts
    result, ws = funcCall(args, bufHelpers, enh);
    if result.ContainsSameLabelset() {
        ev.errorf("vector cannot contain metrics with the same labelset")
    }
    enh.Out = result[:0] // Reuse result vector.
    warnings = append(warnings, ws...)

    ev.currentSamples += result.len();
    // When we reset currentSamples to tempNumSamples during the next iteration of the loop it also
    // needs to include the samples from the result here, as they're still in memory.
    tempNumSamples += result.len()
    ev.samplesStats.UpdatePeak(ev.currentSamples)

    if ev.currentSamples > ev.maxSamples {
        ev.error(ErrTooManySamples(env))
    }
    ev.samplesStats.UpdatePeak(ev.currentSamples)

    // If this could be an instant query, shortcut so as not to change sort order.
    if ev.endTimestamp == ev.startTimestamp {
        let mat = make(Matrix, result.len())
        for i, s := range result {
            s.Point.T = ts
            mat.push( Series{Metric: s.Metric, points: []Point{s.Point}} )
        }
        ev.currentSamples = originalNumSamples + mat.TotalSamples()
        ev.samplesStats.UpdatePeak(ev.currentSamples)
        return mat, warnings
    }

    // Add samples in output vector to output series.
    for sample in result.iter() {
        let h = sample.Metric.hash();
        if let Some(ss) = seriess.get_mut(h) {
            ss.points.push(sample.Point)
        } else {
            seriess.insert(h, Series{Metric: sample.Metric, points: vec![sample.Point]})
        }
    }
    let h = sample.metric.get_hash();
    ss, ok := seriess[h]
    if !ok {
        ss = Series{
            metric: sample.metric.clone(),
            points: getPointSlice(numSteps),
        }
    }
    sample.point.t = ts;
    ss.points = append(ss.points, sample.Point)
    seriess[h] = ss

}
}

    // Reuse the original point slices.
    for m in origMatrixes.iter() {
        for s in m.iter() {
            putPointSlice(s.points)
        }
    }
    // Assemble the output matrix. By the time we get here we know we don't have too many samples.
    let mat = make(Matrix, 0, len(seriess))
    for ss in seriess {
        mat.push(ss)
    }
    ev.currentSamples = originalNumSamples + mat.TotalSamples()
    ev.samplesStats.UpdatePeak(ev.currentSamples)
    return mat, warnings
}

