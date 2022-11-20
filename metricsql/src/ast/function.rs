use std::fmt;
use std::fmt::{Display, Formatter};
use crate::ast::{BExpression, Expression, ExpressionNode, ReturnValue};
use crate::ast::expression_kind::ExpressionKind;
use crate::ast::misc::write_expression_list;
use crate::functions::{BuiltinFunction, is_rollup_aggregation_over_time};
use crate::lexer::TextSpan;
use crate::parser::{ParseError, ParseResult};
use serde::{Serialize, Deserialize};

/// FuncExpr represents MetricsQL function such as `rate(...)`
#[derive(Debug, Clone, Hash, Serialize, Deserialize)]
pub struct FuncExpr {
    pub function: BuiltinFunction,

    /// Args contains function args.
    pub args: Vec<BExpression>,

    /// If keep_metric_names is set to true, then the function should keep metric names.
    pub keep_metric_names: bool,

    pub is_scalar: bool,

    pub span: TextSpan,

    /// internal only name parsed in WITH expression
    #[serde(skip)]
    pub(crate) with_name: String,
}

impl FuncExpr {
    pub fn new<S: Into<TextSpan>>(name: &str, args: Vec<BExpression>, span: S) -> ParseResult<Self> {
        // time() returns scalar in PromQL - see https://prometheus.io/docs/prometheus/latest/querying/functions/#time
        let lower = name.to_lowercase();
        let function = BuiltinFunction::new(name)?;
        let is_scalar = lower == "time"; // todo: what about now() and pi()

        let expr = FuncExpr {
            function,
            args,
            keep_metric_names: false,
            span: span.into(),
            is_scalar,
            with_name: "".to_string()
        };

        match expr.return_value() {
            // todo: pass span to error
            ReturnValue::Unknown(unknown) => {
                Err(ParseError::InvalidExpression(
                    unknown.message
                ))
            }
            _ => Ok(expr)
        }
    }

    pub fn name(&self) -> String {
        self.function.name()
    }

    pub fn default_rollup(arg: Expression) -> ParseResult<Self> {
        let span = arg.span();
        FuncExpr::from_single_arg("default_rollup", arg, span)
    }

    pub fn from_single_arg<S: Into<TextSpan>>(name: &str, arg: Expression, span: S) -> ParseResult<Self> {
        let args = vec![Box::new(arg)];
        FuncExpr::new(name, args, span)
    }

    pub fn create<S: Into<TextSpan>>(name: &str, args: &[Expression], span: S) -> ParseResult<Self> {
        let params =
            Vec::from(args).into_iter().map(Box::new).collect();
        FuncExpr::new(name, params, span)
    }

    pub fn is_aggregate(&self) -> bool {
        self.type_name() == "aggregate"
    }

    pub fn is_rollup(&self) -> bool {
        self.type_name() == "rollup"
    }

    pub fn type_name(&self) -> &'static str {
        self.function.type_name()
    }

    pub fn return_value(&self) -> ReturnValue {
        if self.is_scalar {
            return ReturnValue::Scalar
        }

        // determine the arg to pass through
        let arg = self.get_arg_for_optimization();

        let kind = if arg.is_none() {
            ReturnValue::RangeVector
        } else {
            arg.unwrap().return_value()
        };

        return match self.function {
            BuiltinFunction::Rollup(rf) => {
                if is_rollup_aggregation_over_time(rf) {
                    match kind {
                        ReturnValue::RangeVector => ReturnValue::InstantVector,
                        // ???
                        ReturnValue::InstantVector => ReturnValue::InstantVector,
                        _ => {
                            // invalid arg
                            ReturnValue::unknown(
                                format!("aggregation over time is not valid with expression returning {:?}", kind),
                                // show the arg as the cause
                                // doesn't follow the usual pattern of showing the parent, but would
                                // otherwise be ambiguous with multiple args
                                *arg.unwrap().clone()
                            )
                        }
                    }
                } else {
                    kind
                }
            },
            _ => kind
        }
    }

    pub fn get_arg_idx_for_optimization(&self) -> Option<usize> {
        self.function.get_arg_idx_for_optimization(&self.args)
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

impl Display for FuncExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.name())?;
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

    fn kind(&self) -> ExpressionKind {
        ExpressionKind::Function
    }
}