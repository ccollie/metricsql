use std::fmt;
use std::fmt::{Display, Formatter};
use std::str::FromStr;

use serde::{Deserialize, Serialize};

pub use aggregate::*;
pub use rollup::*;
pub use signature::*;
pub use transform::*;

use crate::ast::Expr;
use crate::common::ValueType;
use crate::parser::{validate_function_args, ParseError, ParseResult};

mod aggregate;
mod rollup;
mod signature;
mod transform;

/// Maximum number of arguments permitted in a rollup function. This really only applies
/// to variadic functions like `aggr_over_time` and `quantiles_over_time`
const MAX_ARG_COUNT: usize = 32;

#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum BuiltinFunction {
    Aggregate(AggregateFunction),
    Rollup(RollupFunction),
    Transform(TransformFunction),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BuiltinFunctionType {
    Aggregate,
    Rollup,
    Transform,
}

impl BuiltinFunctionType {
    pub fn to_str(&self) -> &'static str {
        match self {
            BuiltinFunctionType::Aggregate => "aggregate",
            BuiltinFunctionType::Rollup => "rollup",
            BuiltinFunctionType::Transform => "transform",
        }
    }
}

impl Display for BuiltinFunctionType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_str())?;
        Ok(())
    }
}

impl BuiltinFunction {
    pub fn new(name: &str) -> ParseResult<Self> {
        if let Ok(func) = TransformFunction::from_str(name) {
            return Ok(BuiltinFunction::Transform(func));
        }

        if let Ok(af) = AggregateFunction::from_str(name) {
            return Ok(BuiltinFunction::Aggregate(af));
        }

        if let Ok(rf) = RollupFunction::from_str(name) {
            return Ok(BuiltinFunction::Rollup(rf));
        }

        Err(ParseError::InvalidFunction(format!("built-in::{name}")))
    }

    pub fn is_supported(name: &str) -> bool {
        Self::new(name).is_ok()
    }

    pub fn name(&self) -> &'static str {
        use BuiltinFunction::*;
        match self {
            Aggregate(af) => af.name(),
            Rollup(rf) => rf.name(),
            Transform(tf) => tf.name(),
        }
    }

    pub fn signature(&self) -> Signature {
        use BuiltinFunction::*;
        match self {
            Aggregate(af) => af.signature(),
            Rollup(rf) => rf.signature(),
            Transform(tf) => tf.signature(),
        }
    }

    pub fn validate_arg_count(&self, name: &str, arg_len: usize) -> ParseResult<()> {
        self.signature().validate_arg_count(name, arg_len)
    }

    pub fn validate_args(&self, args: &[Expr]) -> ParseResult<()> {
        validate_function_args(self, args)
    }

    pub fn volatility(&self) -> Volatility {
        self.signature().volatility
    }

    pub fn type_name(&self) -> &'static str {
        self.get_type().to_str()
    }

    pub fn get_type(&self) -> BuiltinFunctionType {
        use BuiltinFunction::*;
        match self {
            Aggregate(_) => BuiltinFunctionType::Aggregate,
            Rollup(_) => BuiltinFunctionType::Rollup,
            Transform(_) => BuiltinFunctionType::Transform,
        }
    }

    pub fn is_type(&self, other: BuiltinFunction) -> bool {
        self.type_name() == other.type_name()
    }

    pub fn is_aggregate_func(name: &str) -> bool {
        AggregateFunction::from_str(name).is_ok()
    }

    pub fn is_aggregation(&self) -> bool {
        matches!(self, BuiltinFunction::Aggregate(_))
    }

    pub fn is_scalar(&self) -> bool {
        match self {
            BuiltinFunction::Transform(func) => func.return_type() == ValueType::Scalar,
            _ => false,
        }
    }

    pub fn is_rollup_function(&self, func: RollupFunction) -> bool {
        match self {
            BuiltinFunction::Rollup(rf) => rf == &func,
            _ => false,
        }
    }

    pub fn may_sort_results(&self) -> bool {
        use BuiltinFunction::*;
        match self {
            Aggregate(af) => af.may_sort_results(),
            Rollup(_) => false, // todo
            Transform(tf) => tf.may_sort_results(),
        }
    }

    pub fn get_arg_for_optimization<'a>(&'a self, args: &'a [Expr]) -> Option<&Expr> {
        match self.get_arg_idx_for_optimization(args.len()) {
            Some(idx) => args.get(idx),
            None => None,
        }
    }

    pub fn get_arg_idx_for_optimization(&self, args_len: usize) -> Option<usize> {
        match self {
            BuiltinFunction::Aggregate(af) => get_aggregate_arg_idx_for_optimization(*af, args_len),
            BuiltinFunction::Rollup(rf) => get_rollup_arg_idx_for_optimization(*rf, args_len),
            BuiltinFunction::Transform(tf) => get_transform_arg_idx_for_optimization(*tf, args_len),
        }
    }

    pub fn return_type(&self, args: &[Expr]) -> ParseResult<ValueType> {
        if self.is_scalar() {
            return Ok(ValueType::Scalar);
        }

        // determine the arg to pass through
        let arg = self.get_arg_for_optimization(args);

        let kind = if let Some(exp) = arg {
            exp.return_type()
        } else {
            // todo: does this depend on the function type (rollup, transform, aggregation)
            ValueType::InstantVector
        };

        match self {
            BuiltinFunction::Rollup(rf) => {
                if is_rollup_aggregation_over_time(*rf) {
                    match kind {
                        ValueType::RangeVector => Ok(ValueType::InstantVector),
                        ValueType::Scalar | ValueType::InstantVector => {
                            Ok(ValueType::InstantVector)
                        }
                        _ => {
                            // invalid arg
                            Err(ParseError::General(format!(
                                "aggregation over time is not valid with Expr returning {:?}",
                                kind
                            )))
                        }
                    }
                } else {
                    Ok(kind)
                }
            }
            _ => Ok(kind),
        }
    }
}

impl Display for BuiltinFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        use BuiltinFunction::*;
        match self {
            Aggregate(af) => write!(f, "{}", af),
            Rollup(rf) => write!(f, "{}", rf),
            Transform(tf) => write!(f, "{}", tf),
        }
    }
}

impl FromStr for BuiltinFunction {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::new(s)
    }
}

impl TryFrom<&str> for BuiltinFunction {
    type Error = ParseError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_is_supported() {
        assert!(super::BuiltinFunction::is_supported("geomean"));
        assert!(super::BuiltinFunction::is_supported("Predict_Linear"));
        assert!(super::BuiltinFunction::is_supported("minute"));
        assert!(!super::BuiltinFunction::is_supported("foo"));
    }
}
