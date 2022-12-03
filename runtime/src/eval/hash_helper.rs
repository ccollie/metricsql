use xxhash_rust::xxh3::Xxh3;
use metricsql::common::{GroupModifier, GroupModifierOp};
use crate::MetricName;

pub(super) struct HashContext<'a> {
    buf: Vec<u8>,
    hasher: Xxh3,
    names: &'a [String],
    op: GroupModifierOp,
}

impl<'a> HashContext<'a> {
    pub fn from_modifier(modifier: &'a GroupModifier) -> Self {
        let GroupModifier{ op, .. } = modifier;
        let names = modifier.labels();
        Self::new(*op, names)
    }

    pub fn new(op: GroupModifierOp, names: &'a [String]) -> Self {
        HashContext {
            buf: Vec::with_capacity(128),
            hasher: Xxh3::new(),
            names,
            op,
        }
    }

    pub fn hash(&mut self, labels: &MetricName) -> u64 {
        match self.op {
            GroupModifierOp::On => labels.hash_with_labels(&mut self.buf, &mut self.hasher, self.names),
            GroupModifierOp::Ignoring => labels.hash_without_labels(&mut self.buf, &mut self.hasher, self.names),
        }
    }
}


/// Helper for hashing metric names and labels.
/// Used mainly to minimize allocations. Actually we can make this completely alloc-free
/// if we use hasher.update() on each tag rather than accumulating a buffer.
pub(super) enum HashHelper<'a> {
    None,
    Group(HashContext<'a>)
}

impl<'a> HashHelper<'a> {
    pub fn new(modifier: &'a Option<GroupModifier>) -> Self {
        match modifier {
            Some(modifier) => {
                let ctx = HashContext::from_modifier(modifier);
                match modifier.op {
                    GroupModifierOp::On | GroupModifierOp::Ignoring => Self::Group(ctx),
                }
            },
            None => {
                return Self::None;
            },
        }
    }

    pub fn hash(&mut self, labels: &mut MetricName) -> u64 {
        match self {
            HashHelper::None => labels.get_hash(),
            HashHelper::Group(ctx) => ctx.hash(labels),
        }
    }
}

