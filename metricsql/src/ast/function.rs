use crate::ast::misc::write_expression_list;
use crate::ast::{BExpression, Expression, ExpressionNode};
use crate::common::ReturnType;
use crate::functions::{BuiltinFunction, RollupFunction};
use crate::parser::{ParseError, ParseResult};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fmt::{Display, Formatter};
use std::str::FromStr;

/// FuncExpr represents a MetricsQL function such as `rate(...)`
#[derive(Debug, Clone, Hash, PartialEq, Serialize, Deserialize)]
pub struct FuncExpr {
    pub name: String,

    pub function: BuiltinFunction,

    /// Args contains function args.
    pub args: Vec<BExpression>,

    /// If keep_metric_names is set to true, then the function should keep metric names.
    pub keep_metric_names: bool,

    pub is_scalar: bool,

    /// internal only name parsed in WITH expression
    #[serde(skip)]
    pub(crate) with_name: String,

    pub return_type: ReturnType,
}

impl FuncExpr {
    pub fn new(name: &str, args: Vec<BExpression>) -> ParseResult<Self> {
        // time() returns scalar in PromQL - see https://prometheus.io/docs/prometheus/latest/querying/functions/#time
        let lower = name.to_lowercase();
        let fname = if name.is_empty() { "union" } else { name };
        let function = BuiltinFunction::new(fname)?;
        let return_type = function.return_type(&args);
        let is_scalar = lower == "time"; // todo: what about now() and pi()

        match return_type {
            ReturnType::Unknown(unknown) => {
                return Err(ParseError::InvalidExpression(unknown.message))
            }
            _ => {}
        }

        Ok(FuncExpr {
            function,
            name: name.to_string(),
            args,
            keep_metric_names: false,
            is_scalar,
            return_type,
            with_name: "".to_string(),
        })
    }

    pub fn default_rollup(arg: Expression) -> ParseResult<Self> {
        FuncExpr::from_single_arg("default_rollup", arg)
    }

    pub fn from_single_arg(name: &str, arg: Expression) -> ParseResult<Self> {
        let args = vec![Box::new(arg)];
        FuncExpr::new(name, args)
    }

    pub fn create(name: &str, args: &[Expression]) -> ParseResult<Self> {
        let params = Vec::from(args).into_iter().map(Box::new).collect();
        FuncExpr::new(name, params)
    }

    pub fn is_rollup(&self) -> bool {
        match RollupFunction::from_str(self.name.as_str()) {
            Ok(_) => true,
            Err(_) => false,
        }
    }

    pub fn type_name(&self) -> &'static str {
        self.function.type_name()
    }

    pub fn return_type(&self) -> ReturnType {
        self.return_type.clone()
    }

    pub fn get_arg_idx_for_optimization(&self) -> Option<usize> {
        self.function.get_arg_idx_for_optimization(self.args.len())
    }

    pub fn get_arg_for_optimization(&self) -> Option<&'_ BExpression> {
        match self.get_arg_idx_for_optimization() {
            None => None,
            Some(idx) => Some(&self.args[idx]),
        }
    }
}

impl Display for FuncExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.name)?;
        write_expression_list(&self.args, f)?;
        if self.keep_metric_names {
            write!(f, " keep_metric_names")?;
        }
        Ok(())
    }
}

impl ExpressionNode for FuncExpr {
    fn cast(self) -> Expression {
        Expression::Function(self)
    }
}
