use crate::ast::{write_expression_list, BExpression, Expression, ExpressionNode};
use crate::common::{AggregateModifier, ReturnType};
use crate::functions::{get_aggregate_arg_idx_for_optimization, AggregateFunction};
use crate::parser::ParseError;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fmt::{Display, Formatter};
use std::str::FromStr;

/// AggrFuncExpr represents aggregate function such as `sum(...) by (...)`
#[derive(Debug, Clone, Hash, PartialEq, Serialize, Deserialize)]
pub struct AggrFuncExpr {
    /// the aggregation function enum
    pub function: AggregateFunction,

    /// name is the function name.
    pub name: String,

    /// function args.
    pub args: Vec<BExpression>,

    /// optional modifier such as `by (...)` or `without (...)`.
    pub modifier: Option<AggregateModifier>,

    /// optional limit for the number of output time series.
    /// This is MetricsQL extension.
    ///
    /// Example: `sum(...) by (...) limit 10` would return maximum 10 time series.
    pub limit: usize,

    pub keep_metric_names: bool,
}

impl AggrFuncExpr {
    pub fn new(function: &AggregateFunction) -> AggrFuncExpr {
        AggrFuncExpr {
            function: *function,
            name: function.to_string(),
            args: vec![],
            modifier: None,
            limit: 0,
            keep_metric_names: false,
        }
    }

    pub fn from_name(name: &str) -> Result<Self, ParseError> {
        let function = AggregateFunction::from_str(name)?;
        Ok(Self::new(&function))
    }

    pub fn with_modifier(mut self, modifier: AggregateModifier) -> Self {
        self.modifier = Some(modifier);
        self
    }

    pub fn with_args(mut self, args: &[BExpression]) -> Self {
        self.args = args.to_vec();
        self.set_keep_metric_names();
        self
    }

    fn set_keep_metric_names(&mut self) {
        // Extract: RollupFunc(...) from aggrFunc(rollupFunc(...)).
        // This case is possible when optimized aggrfn calculations are used
        // such as `sum(rate(...))`
        if self.args.len() != 1 {
            self.keep_metric_names = false;
            return;
        }
        match &*self.args[0] {
            Expression::Function(fe) => {
                self.keep_metric_names = fe.keep_metric_names;
            }
            _ => self.keep_metric_names = false,
        }
    }

    pub fn return_type(&self) -> ReturnType {
        ReturnType::InstantVector
    }

    pub fn get_arg_idx_for_optimization(&self) -> Option<usize> {
        get_aggregate_arg_idx_for_optimization(self.function, self.args.len())
    }

    pub fn get_arg_for_optimization(&self) -> Option<&'_ BExpression> {
        match self.get_arg_idx_for_optimization() {
            None => None,
            Some(idx) => Some(&self.args[idx]),
        }
    }
}

impl Display for AggrFuncExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.function)?;
        let args_len = self.args.len();
        if args_len > 0 {
            write_expression_list(&self.args, f)?;
        }
        if let Some(modifier) = &self.modifier {
            write!(f, " {}", modifier)?;
        }
        if self.limit > 0 {
            write!(f, " limit {}", self.limit)?;
        }
        Ok(())
    }
}

impl ExpressionNode for AggrFuncExpr {
    fn cast(self) -> Expression {
        Expression::Aggregation(self)
    }
}
