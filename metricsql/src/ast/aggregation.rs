use std::fmt;
use std::fmt::{Display, Formatter};
use std::str::FromStr;
use crate::ast::{BExpression, Expression, ExpressionNode, ReturnValue};
use crate::ast::misc::{write_expression_list, write_labels};
use crate::functions::{AggregateFunction, get_aggregate_arg_idx_for_optimization};
use crate::lexer::TextSpan;
use crate::parser::ParseError;
use serde::{Serialize, Deserialize};

#[derive(Default, Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AggregateModifierOp {
    #[default]
    By,
    Without,
}

impl Display for AggregateModifierOp {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use AggregateModifierOp::*;
        match self {
            By => write!(f, "by")?,
            Without => write!(f, "without")?,
        }
        Ok(())
    }
}

impl TryFrom<&str> for AggregateModifierOp {
    type Error = ParseError;

    fn try_from(op: &str) -> Result<Self, Self::Error> {
        use AggregateModifierOp::*;

        match op.to_lowercase().as_str() {
            "by" => Ok(By),
            "without" => Ok(Without),
            _ => Err(ParseError::General(format!(
                "Unknown aggregate modifier op: {}",
                op
            ))),
        }
    }
}

#[derive(Default, Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AggregateModifier {
    /// The modifier operation.
    pub op: AggregateModifierOp,
    /// Modifier args from parens.
    pub args: Vec<String>,
    pub span: Option<TextSpan>,
}

impl AggregateModifier {
    pub fn new(op: AggregateModifierOp, args: Vec<String>) -> Self {
        AggregateModifier {
            op,
            args,
            span: None,
        }
    }

    /// Creates a new AggregateModifier with the Left op
    pub fn by() -> Self {
        AggregateModifier::new(AggregateModifierOp::By, vec![])
    }

    /// Creates a new AggregateModifier with the Right op
    pub fn without() -> Self {
        AggregateModifier::new(AggregateModifierOp::Without, vec![])
    }

    /// Replaces this AggregateModifier's operator
    pub fn op(mut self, op: AggregateModifierOp) -> Self {
        self.op = op;
        self
    }

    /// Adds a label key to this AggregateModifier
    pub fn arg<S: Into<String>>(mut self, arg: S) -> Self {
        self.args.push(arg.into());
        self
    }

    /// Replaces this AggregateModifier's args with the given set
    pub fn args(mut self, args: &[&str]) -> Self {
        self.args = args.iter().map(|l| (*l).to_string()).collect();
        self
    }
}

impl Display for AggregateModifier {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        // Op is the operation itself, i.e. `+`, `-`, `*`, etc.
        write!(f, "{} ", self.op)?;
        write_labels(&self.args, f)?;
        Ok(())
    }
}

/// AggrFuncExpr represents aggregate function such as `sum(...) by (...)`
#[derive(Debug, Clone, Hash, Serialize, Deserialize)]
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

    pub span: TextSpan,
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
            span: TextSpan::default(),
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

    pub fn return_value(&self) -> ReturnValue {
        ReturnValue::InstantVector
    }

    pub fn get_arg_idx_for_optimization(&self) -> Option<usize> {
        get_aggregate_arg_idx_for_optimization(self.function, self.args.len())
    }

    pub fn get_arg_for_optimization(&self) -> Option<&'_ BExpression> {
        match self.get_arg_idx_for_optimization() {
            None => None,
            Some(idx) => {
                Some(&self.args[idx])
            }
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

