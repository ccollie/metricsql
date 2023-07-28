// integrate checks from
// https://github.com/prometheus/prometheus/blob/fa6e05903fd3ce52e374a6e1bf4eb98c9f1f45a7/promql/parser/parse.go#L436

// Original Source: https://github.com/GreptimeTeam/promql-parser/blob/main/src/parser/ast.rs
use crate::ast::{
    AggregationExpr, BExpression, BinaryExpr, Expr, FunctionExpr, InterpolatedSelector, MetricExpr,
    NumberLiteral, ParensExpr, RollupExpr, WithExpr,
};
use crate::common::{
    BinModifier, StringExpr, Value, ValueType, VectorMatchCardinality, NAME_LABEL,
};
use crate::functions::BuiltinFunction;

/// check_ast checks the validity of the provided AST. This includes type checking.
/// Recursively check correct typing for child nodes and raise errors in case of bad typing.
pub fn check_ast(expr: Expr) -> Result<Expr, String> {
    use Expr::*;
    match expr {
        BinaryOperator(ex) => check_ast_for_binary_expr(ex),
        Aggregation(ex) => check_ast_for_aggregate_expr(ex),
        Function(ex) => check_ast_for_call(ex),
        MetricExpression(ex) => check_ast_for_vector_selector(ex),
        Rollup(ex) => check_ast_for_rollup(ex),
        Parens(ex) => check_ast_for_parens(ex),
        StringExpr(ex) => check_ast_for_string_expr(ex),
        With(ex) => check_ast_for_with(ex),
        StringLiteral(_) | Number(_) | Duration(_) => Ok(expr),
        WithSelector(ws) => check_ast_for_interpolated_vector_selector(ws),
    }
}

pub fn validate_func_args(func: &BuiltinFunction, args: &[Expr]) -> Result<(), String> {
    func.validate_args(args).map_err(|e| e.to_string())
}

fn check_ast_for_aggregate_expr(ex: AggregationExpr) -> Result<Expr, String> {
    let func = BuiltinFunction::new(&ex.name);
    if let Ok(func) = func {
        if !func.is_aggregation() {
            return Err(format!(
                "aggregation function expected: got '{}'",
                ex.function
            ));
        }
        validate_func_args(&func, &ex.args)?;
    } else {
        return Err(format!(
            "aggregation function expected: got '{}'",
            ex.function
        ));
    }
    Ok(Expr::Aggregation(ex))
}

fn check_ast_for_call(expr: FunctionExpr) -> Result<Expr, String> {
    validate_func_args(&expr.function, &expr.args)?;
    Ok(Expr::Function(expr))
}

fn check_ast_for_parens(expr: ParensExpr) -> Result<Expr, String> {
    let mut expressions: Vec<Expr> = Vec::with_capacity(expr.len());
    for expr in expr.expressions.into_iter() {
        expressions.push(check_ast(expr)?);
    }
    Ok(Expr::Parens(ParensExpr::new(expressions)))
}

// TODO
fn check_ast_for_with(expr: WithExpr) -> Result<Expr, String> {
    Ok(Expr::With(expr))
}

/// TODO
fn check_ast_for_string_expr(expr: StringExpr) -> Result<Expr, String> {
    Ok(Expr::StringExpr(expr))
}

/// the original logic is redundant in
/// prometheus, and the following coding blocks
/// have been optimized for readability, but all logic SHOULD be covered.
fn check_ast_for_binary_expr(mut ex: BinaryExpr) -> Result<Expr, String> {
    use ValueType::*;

    let operator = ex.op;
    let is_comparison = operator.is_comparison();

    if ex.returns_bool() && !is_comparison {
        return Err("bool modifier can only be used on comparison operators".into());
    }

    let left_type = ex.left.value_type();
    let right_type = ex.right.value_type();

    // we're more lenient than prometheus here
    // if is_comparison {
    //     match (&left_type, &right_type, ex.returns_bool()) {
    //         (ValueType::Scalar, ValueType::Scalar, false) => {
    //             return Err("comparisons between scalars must use BOOL modifier".into());
    //         }
    //         (ValueType::String, ValueType::String, false) => {
    //             return Err("comparisons between strings must use BOOL modifier".into());
    //         }
    //         _ => {}
    //     }
    // }

    // For `on` matching, a label can only appear in one of the lists.
    // Every time series of the result vector must be uniquely identifiable.
    if ex.is_matching_on() && ex.is_labels_joint() {
        if let Some(labels) = ex.intersect_labels() {
            if let Some(label) = labels.first() {
                return Err(format!(
                    "label '{label}' must not occur in ON and GROUP clause at once",
                ));
            }
        };
    }

    if operator.is_set_operator() {
        if left_type == String && right_type == String {
            return Err(format!(
                "operator '{operator}' not allowed in string string operations"
            ));
        }

        if left_type == String || right_type == String {
            return Err(format!(
                "set operator '{operator}' not allowed in binary {left_type}/{right_type} expression",
            ));
        }

        if left_type == InstantVector && right_type == InstantVector {
            if let Some(ref modifier) = ex.modifier {
                if matches!(modifier.card, VectorMatchCardinality::OneToMany(_))
                    || matches!(modifier.card, VectorMatchCardinality::ManyToOne(_))
                {
                    return Err(format!("no grouping allowed for '{operator}' operation"));
                }
            };
        }

        match &mut ex.modifier {
            Some(modifier) => {
                if modifier.card == VectorMatchCardinality::OneToOne {
                    modifier.card = VectorMatchCardinality::ManyToMany;
                }
            }
            None => {
                ex.modifier =
                    Some(BinModifier::default().with_card(VectorMatchCardinality::ManyToMany));
            }
        }
    }

    if left_type == String && right_type == String {
        if !operator.is_valid_string_op() {
            return Err(format!(
                "operator '{operator}' not allowed in string string operations"
            ));
        }
        return Ok(Expr::BinaryOperator(ex));
    }

    let valid_types = [Scalar, InstantVector, RangeVector];

    if !valid_types.contains(&left_type) || !valid_types.contains(&right_type) {
        return Err("mismatched operand types in binary expression".into());
    }

    if (left_type != InstantVector || right_type != InstantVector)
        && ex.is_matching_labels_not_empty()
    {
        return Err("vector matching only allowed between instant vectors".into());
    }

    Ok(Expr::BinaryOperator(ex))
}

fn check_ast_for_rollup(mut ex: RollupExpr) -> Result<Expr, String> {
    ex.expr = BExpression::from(check_ast(*ex.expr)?);
    let value_type = ex.expr.return_type();
    if value_type != ValueType::InstantVector {
        return Err(format!(
            "subquery is only allowed on instant vector, got {value_type} instead"
        ));
    }
    if let Some(at) = &ex.at {
        let at_type = at.return_type();
        if at_type != ValueType::Scalar && value_type != ValueType::InstantVector {
            return Err(format!(
                "subquery @ modifier must be a scalar or expression, got {at_type} instead",
            ));
        }
        match at.as_ref() {
            Expr::Number(NumberLiteral { value, .. }) => {
                if value.is_infinite()
                    || value.is_nan()
                    || value >= &(i64::MAX as f64)
                    || value <= &(i64::MIN as f64)
                {
                    return Err(format!("timestamp out of bounds for @ modifier: {value}"));
                }
            }
            _ => {}
        }
    }

    Ok(Expr::Rollup(ex))
}

fn check_ast_for_vector_selector(ex: MetricExpr) -> Result<Expr, String> {
    // A Vector selector must contain at least one non-empty matcher to prevent
    // implicit selection of all metrics (e.g. by a typo).
    if ex.is_empty_matchers() {
        return Err("vector selector must contain at least one non-empty matcher".into());
    }

    let mut du = ex.find_matchers(NAME_LABEL);
    if du.len() >= 2 {
        // this is to ensure that the err information can be predicted with fixed order
        du.sort();

        return Err(format!(
            "metric name must not be set twice: '{}' or '{}'",
            du[0].name(),
            du[1].name()
        ));
    }

    Ok(Expr::MetricExpression(ex))
}

fn check_ast_for_interpolated_vector_selector(ex: InterpolatedSelector) -> Result<Expr, String> {
    // A Vector selector must contain at least one non-empty matcher to prevent
    // implicit selection of all metrics (e.g. by a typo).
    if ex.is_empty_matchers() {
        return Err("vector selector must contain at least one non-empty matcher".into());
    }

    let mut du = ex.find_matchers(NAME_LABEL);
    if du.len() >= 2 {
        // this is to ensure that the err information can be predicted with fixed order
        du.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        return Err(format!(
            "metric name must not be set twice: '{}' or '{}'",
            du[0].name(),
            du[1].name()
        ));
    }

    Ok(Expr::WithSelector(ex))
}
