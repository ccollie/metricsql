use std::fmt;
use std::fmt::{Display, Formatter};
use std::str::FromStr;

pub use aggregate::*;
pub use data_type::*;
pub use rollup::*;
pub use signature::*;
pub use transform::*;

use crate::ast::{BExpression, Expression};
use crate::parser::{ParseError, ParseResult};

mod data_type;
mod aggregate;
mod rollup;
mod signature;
mod transform;

/// Maximum number of arguments permitted in a rollup function. This really only applies
/// to variadic functions like `aggr_over_time` and `quantiles_over_time`
const MAX_ARG_COUNT: usize = 32;

#[derive(Debug, Clone, Hash)]
pub struct WithExprFunction {
    name: String,
    //sig: Signature
}

#[derive(Debug, Clone, Hash, PartialEq)]
pub enum BuiltinFunction {
    Aggregate(AggregateFunction),
    Rollup(RollupFunction),
    Transform(TransformFunction)
}

impl BuiltinFunction {
    pub fn new(name: &str) -> ParseResult<Self> {
        let af = AggregateFunction::from_str(name);
        if af.is_ok() {
            return Ok(BuiltinFunction::Aggregate(af.unwrap()))
        }

        let tf = TransformFunction::from_str(name);
        if tf.is_ok() {
            return Ok(BuiltinFunction::Transform(tf.unwrap()))
        }

        let rf = RollupFunction::from_str(name);
        if rf.is_ok() {
            return Ok(BuiltinFunction::Rollup(rf.unwrap()))
        }

        Err(ParseError::InvalidFunction(
            format!("Unknown function: {}", name)
        ))
    }

    pub fn name(&self) -> String {
        use BuiltinFunction::*;
        match self {
            Aggregate(af) => af.to_string(),
            Rollup(rf) => rf.to_string(),
            Transform(tf) => tf.to_string()
        }
    }

    pub fn signature(&self) -> Signature {
        use BuiltinFunction::*;
        match self {
            Aggregate(af) => af.signature(),
            Rollup(rf) => rf.signature(),
            Transform(tf) => tf.signature()
        }
    }

    pub fn type_name(&self) -> &'static str {
        use BuiltinFunction::*;
        match self {
            Aggregate(_) => "aggregate",
            Rollup(_) => "rollup",
            Transform(_) => "transform"
        }
    }

    pub fn is_type(&self, other: BuiltinFunction) -> bool {
        self.type_name() == other.type_name()
    }

    pub fn sorts_results(&self) -> bool {
        use BuiltinFunction::*;
        match self {
            Aggregate(af) => af.sorts_results(),
            Rollup(_) => false,  // todo
            Transform(tf) => tf.sorts_results()
        }
    }

    pub fn get_arg_for_optimization<'a>(&self, args: &'a [BExpression]) -> Option<&'a BExpression> {
        match self.get_arg_idx_for_optimization(args) {
            Some(idx) => Some(&args[idx]),
            None => None
        }
    }

    pub fn get_arg_idx_for_optimization(&self, args: &[Box<Expression>]) -> Option<usize> {
        match self {
            BuiltinFunction::Aggregate(af) => {
                get_aggregate_arg_idx_for_optimization(*af, args.len())
            }
            BuiltinFunction::Rollup(rf) => {
                get_rollup_arg_idx_for_optimization(*rf, args.len())
            }
            BuiltinFunction::Transform(tf) => {
                get_transform_arg_idx_for_optimization(*tf, args.len())
            }
        }
    }
}

impl Display for BuiltinFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        use BuiltinFunction::*;
        match self {
            Aggregate(af) => write!(f, "{}", af),
            Rollup(rf) => write!(f, "{}", rf),
            Transform(tf) => write!(f, "{}", tf)
        }
    }
}

impl FromStr for BuiltinFunction {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::new(s)
    }
}