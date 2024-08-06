use std::fmt;
use std::fmt::{Display, Formatter};
use std::str::FromStr;
use std::sync::OnceLock;
use serde::{Deserialize, Serialize};
use strum::IntoEnumIterator;
pub use aggregate::*;
use metricsql_common::hash::FastHashMap;
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
    pub const fn to_str(&self) -> &'static str {
        match self {
            BuiltinFunctionType::Aggregate => "aggregate",
            BuiltinFunctionType::Rollup => "rollup",
            BuiltinFunctionType::Transform => "transform",
        }
    }
}

impl FromStr for BuiltinFunctionType {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, ParseError> {
        match s {
            s if s.eq_ignore_ascii_case("aggregate") => Ok(BuiltinFunctionType::Aggregate),
            s if s.eq_ignore_ascii_case("rollup") => Ok(BuiltinFunctionType::Rollup),
            s if s.eq_ignore_ascii_case("transform") => Ok(BuiltinFunctionType::Transform),
            _ => Err(ParseError::InvalidFunction(s.to_string())),
        }
    }
}

impl Display for BuiltinFunctionType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_str())?;
        Ok(())
    }
}


#[derive(Debug, Clone)]
pub struct FunctionMeta {
    pub name: &'static str,
    pub function: BuiltinFunction,
    pub signature: Signature
}

impl FunctionMeta {
    pub fn lookup(name: &str) -> Option<&'static FunctionMeta> {
        get_registry()
            .get(name)
            .or_else(|| get_registry().get(name.to_lowercase().as_str()))
    }

    pub fn get_rollup_function(name: &str) -> ParseResult<&'static FunctionMeta> {
        if let Some(meta) = FunctionMeta::lookup(name) {
            if let BuiltinFunction::Rollup(_) = &meta.function {
                return Ok(meta);
            }
        }
        Err(ParseError::InvalidFunction(format!("rollup::{name}")))
    }

    pub fn get_aggregate_function(name: &str) -> ParseResult<&'static FunctionMeta> {
        if let Some(meta) = FunctionMeta::lookup(name) {
            if let BuiltinFunction::Aggregate(_) = &meta.function {
                return Ok(meta);
            }
        }
        Err(ParseError::InvalidFunction(format!("aggregate::{name}")))
    }

    pub fn get_type(&self) -> BuiltinFunctionType {
        self.function.get_type()
    }

    pub fn validate_arg_count(&self, name: &str, arg_len: usize) -> ParseResult<()> {
        self.signature.validate_arg_count(name, arg_len)
    }

    pub fn validate_args(&self, args: &[Expr]) -> ParseResult<()> {
        validate_function_args(&self.function, args)
    }

    pub fn is_aggregation(&self) -> bool {
        matches!(self.function, BuiltinFunction::Aggregate(_))
    }

    pub fn is_scalar(&self) -> bool {
        match self.function {
            BuiltinFunction::Transform(func) => func.return_type() == ValueType::Scalar,
            _ => false,
        }
    }

    pub fn is_rollup_function(&self, func: RollupFunction) -> bool {
        match &self.function {
            BuiltinFunction::Rollup(rf) => rf == &func,
            _ => false,
        }
    }

    pub fn is_variadic(&self) -> bool {
        self.signature.is_variadic()
    }
}

impl Display for FunctionMeta {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}(", self.name)?;
        let mut first = true;
        let min = self.signature.min_args();


        write!(f, ")")
    }
}
type FunctionRegistry = FastHashMap<&'static str, FunctionMeta>;
static REGISTRY: OnceLock<FunctionRegistry> = OnceLock::new();

pub fn get_registry() -> &'static FunctionRegistry {
    REGISTRY.get_or_init(init_registry)
}

fn init_registry() -> FunctionRegistry {
    let mut registry = FunctionRegistry::default();

    for af in AggregateFunction::iter() {
        let name = af.name();
        let function = BuiltinFunction::Aggregate(af);
        let signature = af.signature();
        registry.insert(name, FunctionMeta { name, function, signature });
    }

    for rf in RollupFunction::iter() {
        let name = rf.name();
        let function = BuiltinFunction::Rollup(rf);
        let signature = rf.signature();
        registry.insert(name, FunctionMeta { name, function, signature });
    }

    for tf in TransformFunction::iter() {
        let name = tf.name();
        let function = BuiltinFunction::Transform(tf);
        let signature = tf.signature();
        registry.insert(name, FunctionMeta { name, function, signature });
    }

    registry
}

impl BuiltinFunction {
    pub fn new(name: &str) -> ParseResult<Self> {
        if let Some(meta) = FunctionMeta::lookup(name) {
            return Ok(meta.function.clone());
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

    pub fn validate_args(&self, args: &[Expr]) -> ParseResult<()> {
        validate_function_args(self, args)
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
        if let Some(meta) = FunctionMeta::lookup(name) {
            return meta.is_aggregation()
        }
        false
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
