use std::fmt;
use std::fmt::{Display, Formatter};

use ahash::AHashSet;
use serde::{Deserialize, Serialize};

use crate::common::write_comma_separated;
use crate::common::Labels;
use crate::parser::ParseError;

/// Matching Modifier, for VectorMatching of binary expr.
/// Label lists provided to matching keywords will determine how vectors are combined.
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum VectorMatchModifier {
    On(Labels),
    Ignoring(Labels),
}

impl Display for VectorMatchModifier {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use VectorMatchModifier::*;
        match self {
            On(labels) => write!(f, "on({:?})", labels)?,
            Ignoring(labels) => write!(f, "ignoring({:?})", labels)?,
        }
        Ok(())
    }
}

impl VectorMatchModifier {
    pub fn new(labels: Vec<String>, is_on: bool) -> Self {
        let names = Labels::from_iter(labels);
        if is_on {
            VectorMatchModifier::On(names)
        } else {
            VectorMatchModifier::Ignoring(names)
        }
    }

    pub fn labels(&self) -> &Labels {
        match self {
            VectorMatchModifier::On(l) => l,
            VectorMatchModifier::Ignoring(l) => l,
        }
    }

    pub fn is_on(&self) -> bool {
        matches!(*self, VectorMatchModifier::On(_))
    }
}

/// Binary Expr Modifier
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct BinModifier {
    /// The matching behavior for the operation if both operands are Vectors.
    /// If they are not this field is None.
    pub card: VectorMatchCardinality,

    /// on/ignoring on labels.
    /// like a + b, no match modifier is needed.
    pub matching: Option<VectorMatchModifier>,

    /// If keep_metric_names is set to true, then the operation should keep metric names.
    pub keep_metric_names: bool,

    /// If a comparison operator, return 0/1 rather than filtering.
    /// For example, `foo > bool bar`.
    pub return_bool: bool,
}

impl Default for BinModifier {
    fn default() -> Self {
        Self {
            card: VectorMatchCardinality::OneToOne,
            matching: None,
            keep_metric_names: false,
            return_bool: false,
        }
    }
}

impl BinModifier {
    pub fn with_card(mut self, card: VectorMatchCardinality) -> Self {
        self.card = card;
        self
    }

    pub fn with_matching(mut self, matching: Option<VectorMatchModifier>) -> Self {
        self.matching = matching;
        if self.matching.is_none() {
            self.card = VectorMatchCardinality::OneToOne;
        }
        self
    }

    pub fn with_return_bool(mut self, return_bool: bool) -> Self {
        self.return_bool = return_bool;
        self
    }

    pub fn with_keep_metric_names(mut self, keep_metric_names: bool) -> Self {
        self.keep_metric_names = keep_metric_names;
        self
    }

    pub fn is_labels_joint(&self) -> bool {
        matches!((self.card.labels(), &self.matching),
                 (Some(labels), Some(matching)) if !labels.is_joint(matching.labels()))
    }

    pub fn intersect_labels(&self) -> Option<Vec<String>> {
        if let Some(labels) = self.card.labels() {
            if let Some(matching) = &self.matching {
                let res = labels.intersect(matching.labels());
                return Some(res.0);
            }
        };
        None
    }

    pub fn is_matching_on(&self) -> bool {
        matches!(&self.matching, Some(matching) if matching.is_on())
    }

    pub fn is_matching_labels_not_empty(&self) -> bool {
        matches!(&self.matching, Some(matching) if !matching.labels().is_empty())
    }

    pub fn is_default(&self) -> bool {
        self.card == VectorMatchCardinality::OneToOne
            && self.matching.is_none()
            && !self.keep_metric_names
            && !self.return_bool
    }
}

impl Display for BinModifier {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use VectorMatchCardinality::*;
        if self.return_bool {
            write!(f, "bool")?;
        }
        match &self.card {
            ManyToOne(labels) => {
                write!(f, " group_left")?;
                write_comma_separated(labels.iter(), f, true)?;
            }
            OneToMany(labels) => {
                write!(f, " group_right")?;
                write_comma_separated(labels.iter(), f, true)?;
            }
            _ => {}
        }
        if let Some(matching) = &self.matching {
            match matching {
                VectorMatchModifier::On(labels) => {
                    write!(f, " on")?;
                    write_comma_separated(labels.iter(), f, true)?;
                }
                VectorMatchModifier::Ignoring(labels) => {
                    write!(f, " ignoring")?;
                    write_comma_separated(labels.iter(), f, true)?;
                }
            }
        }
        if self.keep_metric_names {
            write!(f, " keep_metric_names")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Eq, Serialize, Deserialize)]
pub enum AggregateModifier {
    // todo: use BtreeSet<String>, since thee metricsql_engine expects these to be sorted
    By(Vec<String>),
    Without(Vec<String>),
}

impl AggregateModifier {
    /// Creates a new AggregateModifier with the Left op
    pub fn by() -> Self {
        AggregateModifier::By(vec![])
    }

    /// Creates a new AggregateModifier with the Right op
    pub fn without() -> Self {
        AggregateModifier::Without(vec![])
    }

    /// Adds a label key to this AggregateModifier
    pub fn arg<S: Into<String>>(&mut self, arg: S) {
        match self {
            AggregateModifier::By(ref mut args) => {
                args.push(arg.into());
                args.sort();
            }
            AggregateModifier::Without(ref mut args) => {
                args.push(arg.into());
                args.sort();
            }
        }
    }

    pub fn get_args(&self) -> &Vec<String> {
        match self {
            AggregateModifier::By(val) => val,
            AggregateModifier::Without(val) => val,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.get_args().is_empty()
    }
}

impl Display for AggregateModifier {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            AggregateModifier::By(vec) => {
                write!(f, "by ")?;
                write_comma_separated(vec.iter(), f, true)?;
            }
            AggregateModifier::Without(vec) => {
                write!(f, "without ")?;
                write_comma_separated(vec.iter(), f, true)?;
            }
        }
        Ok(())
    }
}

impl PartialEq<Self> for AggregateModifier {
    fn eq(&self, other: &AggregateModifier) -> bool {
        match (self, other) {
            (AggregateModifier::Without(left), AggregateModifier::Without(right)) => {
                string_vecs_equal_unordered(left, right)
            }
            (AggregateModifier::By(left), AggregateModifier::By(right)) => {
                string_vecs_equal_unordered(left, right)
            }
            _ => false,
        }
    }
}

// See https://prometheus.io/docs/prometheus/latest/querying/operators/#vector-matching
#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash, Serialize, Deserialize)]
pub enum GroupModifierOp {
    On,
    Ignoring,
}

impl Display for GroupModifierOp {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use GroupModifierOp::*;
        match self {
            On => write!(f, "on")?,
            Ignoring => write!(f, "ignoring")?,
        }
        Ok(())
    }
}

impl TryFrom<&str> for GroupModifierOp {
    type Error = ParseError;

    fn try_from(op: &str) -> Result<Self, Self::Error> {
        use GroupModifierOp::*;

        match op {
            op if op.eq_ignore_ascii_case("on") => Ok(On),
            op if op.eq_ignore_ascii_case("ignoring") => Ok(Ignoring),
            _ => Err(ParseError::General(format!(
                "Unknown group_modifier op: {op}",
            ))),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum GroupType {
    GroupLeft,
    GroupRight,
}

#[derive(Debug, Clone, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub enum VectorMatchCardinality {
    OneToOne,
    /// on(labels)/ignoring(labels) GROUP_LEFT
    ManyToOne(Labels),
    /// on(labels)/ignoring(labels) GROUP_RIGHT
    OneToMany(Labels),
    /// logical/set binary operators
    ManyToMany,
}

impl VectorMatchCardinality {
    pub fn join_modifier(&self) -> Option<JoinModifierOp> {
        match self {
            VectorMatchCardinality::ManyToOne(_) => Some(JoinModifierOp::GroupLeft),
            VectorMatchCardinality::OneToMany(_) => Some(JoinModifierOp::GroupRight),
            _ => None,
        }
    }

    pub fn group_left(labels: Labels) -> Self {
        VectorMatchCardinality::ManyToOne(labels)
    }

    pub fn group_right(labels: Labels) -> Self {
        VectorMatchCardinality::OneToMany(labels)
    }

    pub fn is_group_left(&self) -> bool {
        matches!(self, VectorMatchCardinality::ManyToOne(_))
    }

    pub fn is_group_right(&self) -> bool {
        matches!(self, VectorMatchCardinality::OneToMany(_))
    }

    pub fn is_grouping(&self) -> bool {
        matches!(
            self,
            VectorMatchCardinality::ManyToOne(_) | VectorMatchCardinality::OneToMany(_)
        )
    }

    pub fn labels(&self) -> Option<&Labels> {
        match self {
            VectorMatchCardinality::ManyToOne(labels) => Some(labels),
            VectorMatchCardinality::OneToMany(labels) => Some(labels),
            _ => None,
        }
    }

    pub fn group_type(&self) -> Option<GroupType> {
        match self {
            VectorMatchCardinality::ManyToOne(_) => Some(GroupType::GroupLeft),
            VectorMatchCardinality::OneToMany(_) => Some(GroupType::GroupRight),
            _ => None,
        }
    }
}

impl Display for VectorMatchCardinality {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        use VectorMatchCardinality::*;
        let str = match self {
            OneToOne => "OneToOne".to_string(),
            OneToMany(labels) => format!("group_right({:?})", labels),
            ManyToOne(labels) => format!("group_left({:?})", labels),
            ManyToMany => "ManyToMany".to_string(),
        };
        write!(f, "{}", str)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Copy, Hash, Serialize, Deserialize)]
pub enum JoinModifierOp {
    GroupLeft,
    GroupRight,
}

impl Display for JoinModifierOp {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use JoinModifierOp::*;
        match self {
            GroupLeft => write!(f, "group_left")?,
            GroupRight => write!(f, "group_right")?,
        }
        Ok(())
    }
}

impl TryFrom<&str> for JoinModifierOp {
    type Error = ParseError;

    fn try_from(op: &str) -> Result<Self, Self::Error> {
        use JoinModifierOp::*;

        match op {
            op if op.eq_ignore_ascii_case("group_left") => Ok(GroupLeft),
            op if op.eq_ignore_ascii_case("group_right") => Ok(GroupRight),
            _ => {
                let msg = format!("Unknown join_modifier op: {}", op);
                Err(ParseError::General(msg))
            }
        }
    }
}

/// A JoinModifier clause's nested grouping clause
#[derive(Debug, PartialEq, Eq, Clone, Hash, Serialize, Deserialize)]
pub struct JoinModifier {
    /// The GroupModifier group's operator type (left or right)
    pub op: JoinModifierOp,

    /// A list of labels to copy to the opposite side of the group operator, i.e.
    /// group_left(foo) copies the label `foo` from the right hand side
    pub labels: Vec<String>,

    /// The cardinality of the two Vectors.
    pub cardinality: VectorMatchCardinality,
}

impl JoinModifier {
    pub fn new(op: JoinModifierOp, labels: Vec<String>) -> Self {
        JoinModifier {
            op,
            labels,
            cardinality: VectorMatchCardinality::OneToOne,
        }
    }

    /// Creates a new JoinModifier with the Left op
    pub fn left() -> Self {
        JoinModifier::new(JoinModifierOp::GroupLeft, vec![])
    }

    /// Creates a new JoinModifier with the Right op
    pub fn right() -> Self {
        JoinModifier::new(JoinModifierOp::GroupRight, vec![])
    }

    /// Replaces this JoinModifier's operator
    pub fn op(mut self, op: JoinModifierOp) -> Self {
        self.op = op;
        self
    }

    /// Adds a label key to this JoinModifier
    pub fn label<S: Into<String>>(mut self, label: S) -> Self {
        self.labels.push(label.into());
        self
    }

    /// Replaces this JoinModifier's labels with the given set
    pub fn set_labels(mut self, labels: &[&str]) -> Self {
        self.labels = labels.iter().map(|l| (*l).to_string()).collect();
        self
    }

    /// Clears this JoinModifier's set of labels
    pub fn clear_labels(mut self) -> Self {
        self.labels.clear();
        self
    }
}

impl Display for JoinModifier {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{} ", self.op)?;
        write_comma_separated(self.labels.iter(), f, true)?;
        Ok(())
    }
}

impl PartialEq<JoinModifier> for &JoinModifier {
    fn eq(&self, other: &JoinModifier) -> bool {
        self.op == other.op
            && string_vecs_equal_unordered(&self.labels, &other.labels)
            && self.cardinality == other.cardinality
    }
}

fn string_vecs_equal_unordered(a: &[String], b: &[String]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let hash_a: AHashSet<_> = a.iter().collect();
    b.iter().all(|x| hash_a.contains(x))
}
