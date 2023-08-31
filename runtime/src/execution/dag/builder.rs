use std::collections::HashMap;
use std::default::Default;
use std::ops::Deref;

use topologic::AcyclicDependencyGraph;

use metricsql::ast::{
    AggregationExpr, BinaryExpr, DurationExpr, Expr, FunctionExpr, MetricExpr, NumberLiteral,
    ParensExpr, RollupExpr,
};
use metricsql::functions::{BuiltinFunction, RollupFunction, TransformFunction};
use metricsql::prelude::Operator;

use crate::execution::binary::{
    can_push_down_common_filters, eval_string_string_binop, scalar_binary_operation,
};
use crate::execution::dag::aggregate_node::AggregateNode;
use crate::execution::dag::binop_node::BinopNode;
use crate::execution::dag::dag_evaluator::{DAGEvaluator, Dependency};
use crate::execution::dag::dynamic_node::DynamicNode;
use crate::execution::dag::rollup_node::RollupNode;
use crate::execution::dag::scalar_vector_binop_node::ScalarVectorBinaryNode;
use crate::execution::dag::subquery_node::SubqueryNode;
use crate::execution::dag::transform_node::{AbsentTransformNode, TransformNode};
use crate::execution::dag::vector_scalar_binop_node::VectorScalarBinaryNode;
use crate::execution::dag::vector_vector_binary_node::{
    VectorVectorBinaryNode, VectorVectorPushDownNode,
};
use crate::execution::utils::should_keep_metric_names;
use crate::execution::DAGNode;
use crate::functions::aggregate::IncrementalAggregationHandler;
use crate::functions::rollup::{
    get_rollup_function_handler, rollup_default, rollup_func_requires_config, RollupHandler,
};
use crate::{QueryValue, RuntimeError, RuntimeResult};

pub struct DAGBuilder<'a> {
    expr: &'a Expr,
    node_map: HashMap<usize, DAGNode>,
    graph: AcyclicDependencyGraph<usize>,
}

impl<'a> DAGBuilder<'a> {
    pub fn new(parent: &'a Expr) -> Self {
        DAGBuilder {
            expr: parent,
            graph: AcyclicDependencyGraph::new(),
            node_map: HashMap::with_capacity(16),
        }
    }

    pub(crate) fn build(mut self) -> RuntimeResult<Vec<Vec<Dependency>>> {
        self.create_node(self.expr)?;
        let count = self.node_map.len();
        if count == 1 {
            let root = 0;
            let node = self.node_map.remove(&root).unwrap();
            return Ok(vec![vec![Dependency {
                node,
                result_index: root,
            }]]);
        }
        let dag = self.sort_nodes();
        let mut node_dag: Vec<Vec<Dependency>> = Vec::with_capacity(dag.len());
        let mut max_index: usize = 0;

        for layer in dag.iter() {
            let mut nodes = Vec::with_capacity(layer.len());

            for node_index in layer.iter() {
                let node = self.node_map.remove(node_index).unwrap();
                nodes.push(Dependency::new(node, *node_index));
                max_index = max_index.max(*node_index);
            }
            node_dag.push(nodes);
        }
        let binding = self.graph.get_roots();
        let roots = binding.iter().collect::<Vec<_>>();
        let root_count = roots.len();
        if root_count != 1 {
            let msg = format!(
                "Invalid expression. Expected 1 root node, found {root_count}\n{}",
                self.expr
            );
            return Err(RuntimeError::Internal(msg));
        }

        Ok(node_dag)
    }

    pub(crate) fn compile(expr: &Expr) -> RuntimeResult<DAGNode> {
        let builder = DAGBuilder::new(expr);
        let mut dag = builder.build()?;
        if dag.len() == 1 && dag[0].len() == 1 {
            let node = std::mem::take(&mut dag[0][0].node);
            return Ok(node);
        }
        let dynamic = DynamicNode(DAGEvaluator::new(dag));
        Ok(DAGNode::Dynamic(dynamic))
    }

    fn create_node(&mut self, expr: &Expr) -> RuntimeResult<usize> {
        match expr {
            Expr::Aggregation(ae) => self.create_aggregate_node(expr, ae),
            Expr::BinaryOperator(be) => self.create_binary_node(be),
            Expr::Duration(de) => Ok(self.push_node(DAGNode::from(de))),
            Expr::MetricExpression(_) => self.create_selector_node(expr),
            Expr::Function(fe) => self.create_function_node(expr, fe),
            Expr::Number(n) => Ok(self.push_node(DAGNode::from(n.value))),
            Expr::Parens(parens) => self.create_parens_node(expr, parens),
            Expr::Rollup(re) => self.create_rollup_node(
                expr,
                re,
                RollupFunction::DefaultRollup,
                RollupHandler::Wrapped(rollup_default),
            ),
            Expr::StringLiteral(s) => Ok(self.push_node(DAGNode::from(s.clone()))),
            Expr::StringExpr(_) => {
                panic!("invalid node type (SpringExpr) in expression")
            }
            Expr::With(_) => {
                panic!("invalid node type (With) in expression")
            }
            Expr::WithSelector(_) => {
                panic!("invalid node type (WithSelector) in expression")
            }
        }
    }

    fn push_node(&mut self, node: DAGNode) -> usize {
        let idx = self.node_map.len();
        self.node_map.insert(idx, node);
        idx
    }

    fn create_dependency(&mut self, expr: &Expr, parent_idx: usize) -> RuntimeResult<usize> {
        let idx = self.create_node(expr)?;
        self.add_dependency(parent_idx, idx);
        Ok(idx)
    }

    fn create_parens_node(&mut self, expr: &Expr, parens: &ParensExpr) -> RuntimeResult<usize> {
        if parens.len() == 1 {
            self.create_node(&parens.expressions[0])
        } else {
            // todo(perf):  fix this. It ends up cloning parens twice - once here and once in create_function_node
            let fe = FunctionExpr::new("union", parens.expressions.clone())
                .unwrap_or_else(|_| panic!("BUG: failed to create union function")); // this is  a bug and should crash

            self.create_function_node(expr, &fe)
        }
    }

    fn create_function_node(&mut self, expr: &Expr, fe: &FunctionExpr) -> RuntimeResult<usize> {
        match fe.function {
            BuiltinFunction::Transform(tf) => {
                if tf == TransformFunction::Absent {
                    self.create_absent_node(fe)
                } else {
                    self.create_transform_node(fe, tf)
                }
            }
            BuiltinFunction::Rollup(rf) => self.create_rollup_function_node(expr, fe, rf),
            _ => unreachable!("Invalid function type for node {}", fe.function.name()),
        }
    }

    fn reserve_node(&mut self) -> usize {
        self.push_node(Default::default())
    }

    fn create_transform_node(
        &mut self,
        fe: &FunctionExpr,
        tf: TransformFunction,
    ) -> RuntimeResult<usize> {
        // insert a dummy. we will replace it later
        let idx = self.reserve_node();
        let keep_metric_names = fe.keep_metric_names || tf.keep_metric_name();

        let (args, arg_indexes, args_const) = self.handle_args(&fe.args, idx, None)?;
        let transform_node = TransformNode {
            function: tf,
            args,
            keep_metric_names,
            arg_indexes,
            args_const,
        };

        self.node_map
            .insert(idx, DAGNode::Transform(transform_node));

        Ok(idx)
    }

    fn create_absent_node(&mut self, fe: &FunctionExpr) -> RuntimeResult<usize> {
        // insert a dummy. we will replace it later
        let idx = self.reserve_node();

        // we should only have one arg here.
        // Todo: this should have been checked in the parser
        if fe.args.len() != 1 {
            return Err(RuntimeError::ArgumentError(format!(
                "unexpected number of args; got {}; want 1",
                fe.args.len()
            )));
        }

        let first = &fe.args[0];
        let arg_idx = self.create_dependency(first, idx)?;
        let node = AbsentTransformNode::new(first, arg_idx);

        self.node_map.insert(idx, DAGNode::Absent(node));

        Ok(idx)
    }

    fn create_rollup_function_node(
        &mut self,
        expr: &Expr,
        fe: &FunctionExpr,
        rf: RollupFunction,
    ) -> RuntimeResult<usize> {
        // todo: i dont think we can have a empty arg_idx_for_optimization
        let rollup_arg_idx = fe.arg_idx_for_optimization.expect("rollup_arg_idx is None");

        let arg = &fe.args[rollup_arg_idx];
        let re = get_rollup_expr_arg(arg).map_err(|e| {
            RuntimeError::ArgumentError(format!("Invalid argument for rollup function: {:?}", e))
        })?;

        let get_function_handler =
            |rf: RollupFunction, args: &[QueryValue]| -> RuntimeResult<RollupHandler> {
                get_rollup_function_handler(rf, args).map_err(|e| {
                    RuntimeError::ArgumentError(format!(
                        "Invalid arguments for rollup function: {:?}",
                        e
                    ))
                })
            };

        let mut is_default_handler = false;
        let func_handler = if rollup_func_requires_config(&rf) {
            // Func is parameterized, so will be determined at runtime in dependency setter.
            // for now, set the default handler
            is_default_handler = true;
            RollupHandler::default()
        } else {
            // if a function is not parameterized, we can get the handler now. In fact it's necessary
            // for non DAG nodes, since the set_dependency function will never be called.
            let empty = vec![];
            get_function_handler(rf, &empty)?
        };

        let parent_idx = self.create_rollup_node(expr, &re, rf, func_handler)?;

        let (args, arg_indexes, _all_const) =
            self.handle_args(&fe.args, parent_idx, fe.arg_idx_for_optimization)?;

        if !arg_indexes.is_empty() || !args.is_empty() {
            if let Some(expr) = self.node_map.get_mut(&parent_idx) {
                let handler: Option<RollupHandler> = if !args.is_empty() && is_default_handler {
                    Some(get_function_handler(rf, &args)?)
                } else {
                    None
                };
                match expr {
                    DAGNode::Rollup(ref mut node) => {
                        node.arg_indexes = arg_indexes;
                        if let Some(func_handler) = handler {
                            node.func_handler = func_handler;
                        }
                    }
                    DAGNode::Subquery(ref mut node) => {
                        node.arg_indexes = arg_indexes;
                        if let Some(func_handler) = handler {
                            node.func_handler = func_handler;
                        }
                    }
                    _ => {}
                }
            }
        }

        Ok(parent_idx)
    }

    fn create_rollup_node(
        &mut self,
        expr: &Expr,
        re: &RollupExpr,
        rf: RollupFunction,
        func_handler: RollupHandler,
    ) -> RuntimeResult<usize> {
        let parent_idx = self.reserve_node();

        let at_index = if let Some(at) = &re.at {
            Some(self.create_dependency(at, parent_idx)?)
        } else {
            None
        };

        match re.expr.as_ref() {
            Expr::MetricExpression(me) => {
                let mut rn = RollupNode::default();

                rn.offset = re.offset.clone();
                rn.step = re.step.clone();
                rn.window = re.window.clone();
                rn.func = rf;
                rn.metric_expr = me.clone();
                rn.expr = re.expr.as_ref().clone(); // should this be expr.clone()
                rn.func_handler = func_handler;
                rn.at_index = at_index;

                self.node_map.insert(parent_idx, DAGNode::Rollup(rn));

                Ok(parent_idx)
            }
            _ => {
                let mut node = SubqueryNode::default();

                node.offset = re.offset.clone();
                node.step = re.step.clone();
                node.window = re.window.clone();
                node.func = rf;
                node.expr = expr.clone();
                node.func_handler = func_handler;
                node.at_index = at_index;

                let expr_dag = DAGBuilder::compile(&re.expr)?;
                node.expr_node = Box::new(expr_dag);

                self.node_map.insert(parent_idx, DAGNode::Subquery(node));

                Ok(parent_idx)
            }
        }
    }

    fn create_selector_node(&mut self, expr: &Expr) -> RuntimeResult<usize> {
        debug_assert!(expr.is_metric_expression());
        let re = RollupExpr::new(expr.clone());
        let handler = RollupHandler::Wrapped(rollup_default);
        self.create_rollup_node(expr, &re, RollupFunction::DefaultRollup, handler)
    }

    fn create_aggregate_node(&mut self, expr: &Expr, ae: &AggregationExpr) -> RuntimeResult<usize> {
        // todo: ensure that this is serialized otherwise the contained block will not be executed
        if ae.can_incrementally_eval && IncrementalAggregationHandler::handles(ae.function) {
            if let Ok(Some(fe)) = try_get_arg_rollup_func_with_metric_expr(ae) {
                // There is an optimized path for calculating `AggrFuncExpr` over: RollupFunc
                // over MetricExpr.
                // The optimized path saves RAM for aggregates over big number of time series.
                let rf = match fe.function {
                    BuiltinFunction::Rollup(rf) => rf,
                    _ => {
                        // should not happen
                        unreachable!(
                            "Expected a rollup function in aggregation. Found \"{}\"",
                            fe.function
                        )
                    }
                };

                let rollup_expr = RollupExpr::new(Expr::Aggregation(ae.clone()));
                let handler = RollupHandler::Wrapped(rollup_default);
                let node_idx = self.create_rollup_node(expr, &rollup_expr, rf, handler)?;
                if let Some(DAGNode::Rollup(ref mut node)) = self.node_map.get_mut(&node_idx) {
                    node.keep_metric_names = fe.keep_metric_names;
                    node.is_incr_aggregate = true;
                }

                return Ok(node_idx);
            }
        }

        // add dummy node, we will replace it later
        let idx = self.reserve_node();

        let (args, arg_indexes, args_const) = self.handle_args(&ae.args, idx, None)?;

        let node = AggregateNode {
            function: ae.function,
            args,
            modifier: ae.modifier.clone(),
            limit: ae.limit,
            arg_indexes,
            args_const,
        };

        self.node_map.insert(idx, DAGNode::Aggregate(node));

        Ok(idx)
    }

    fn create_binary_node(&mut self, be: &BinaryExpr) -> RuntimeResult<usize> {
        let idx = self.reserve_node();
        let bool_modifier = be.returns_bool();
        let keep_metric_names = should_keep_metric_names(be);
        let is_left_vector = is_vector_expr(&be.left);
        let is_right_vector = is_vector_expr(&be.right);
        let op = be.op;

        // ops with constant operands have already been handled by the optimizer
        let res = match (&be.left.as_ref(), &be.right.as_ref()) {
            (expr_left, Expr::Number(v)) if is_left_vector => {
                let left_idx = self.create_dependency(expr_left, idx)?;
                // vector_scalar
                let node = VectorScalarBinaryNode {
                    left_idx,
                    left: Default::default(),
                    right: v.value,
                    op: be.op,
                    bool_modifier,
                    keep_metric_names,
                };
                DAGNode::VectorScalarOp(node)
            }
            (Expr::Number(v), expr_right) if is_right_vector => {
                let right_idx = self.create_dependency(expr_right, idx)?;
                // scalar_vector
                let node = ScalarVectorBinaryNode {
                    left: v.value,
                    right_idx,
                    right: Default::default(),
                    op: be.op,
                    bool_modifier,
                    keep_metric_names,
                };
                DAGNode::ScalarVectorOp(node)
            }
            // vector op vector needs special handling where both contain selectors
            (expr_left, expr_right) if is_left_vector && is_right_vector => {
                let can_push_down = can_push_down_common_filters(be);

                // if we can't push down, it means both operands can be evaluated in parallel
                if !can_push_down {
                    // add both as dependents of parent

                    let left_idx = self.create_dependency(expr_left, idx)?;
                    let right_idx = self.create_dependency(expr_right, idx)?;
                    let node = VectorVectorBinaryNode::new(
                        left_idx,
                        right_idx,
                        be.op,
                        be.modifier.clone(),
                    );
                    DAGNode::VectorVectorOp(node)
                } else {
                    // if we can push down, we need to evaluate the left operand first
                    // and then use the result to evaluate the right operand
                    // Note: both sides at this point are aggregations

                    let (left_idx, right_expr) = if be.op == Operator::And || be.op == Operator::If
                    {
                        // Fetch right-side series at first, since it usually contains
                        // lower number of time series for `and` and `if` operator.
                        // This should produce more specific label filters for the left side of the query.
                        // This, in turn, should reduce the time to select series for the left side of the query.
                        let right_idx = self.create_dependency(expr_right, idx)?;
                        (right_idx, be.left.clone())
                    } else {
                        let left_idx = self.create_dependency(expr_left, idx)?;
                        (left_idx, be.right.clone())
                    };

                    let node = VectorVectorPushDownNode {
                        left_idx,
                        right: *right_expr,
                        left: Default::default(),
                        op: be.op,
                        modifier: be.modifier.clone(),
                    };
                    DAGNode::VectorVectorPushDownOp(node)
                }
            }
            (Expr::Number(left), Expr::Number(right)) => {
                // should have been handled in the optimizer, but no harm
                let value = scalar_binary_operation(be.op, left.value, right.value, bool_modifier)?;
                DAGNode::from(value)
            }
            (Expr::Duration(ln), Expr::Duration(rn))
                if op == Operator::Add || op == Operator::Sub =>
            {
                match (ln, rn) {
                    (DurationExpr::Millis(left_val), DurationExpr::Millis(right_val)) => {
                        let n = scalar_binary_operation(
                            op,
                            *left_val as f64,
                            *right_val as f64,
                            false,
                        )? as i64;
                        let dur = DurationExpr::new(n);
                        DAGNode::from(dur)
                    }
                    (DurationExpr::StepValue(left_val), DurationExpr::StepValue(right_val)) => {
                        let n = scalar_binary_operation(op, *left_val, *right_val, bool_modifier)?;
                        let dur = DurationExpr::new_step(n);
                        DAGNode::from(dur)
                    }
                    _ => self.create_binop_node_default(be, idx)?,
                }
            }
            // add/subtract number as secs to duration
            (Expr::Duration(ln), Expr::Number(NumberLiteral { value }))
                if !ln.requires_step() && (op == Operator::Add || op == Operator::Sub) =>
            {
                let secs = *value * 1e3_f64;
                let n = scalar_binary_operation(op, ln.value(1) as f64, secs, false)? as i64;
                let dur = DurationExpr::new(n);
                DAGNode::from(dur)
            }
            (Expr::StringLiteral(left), Expr::StringLiteral(right)) => {
                // should have been handled in the optimizer, but no harm
                let value = eval_string_string_binop(be.op, left, right, bool_modifier)?;
                DAGNode::Value(value)
            }
            _ => self.create_binop_node_default(be, idx)?,
        };

        self.node_map.insert(idx, res);

        Ok(idx)
    }

    fn create_binop_node_default(
        &mut self,
        be: &BinaryExpr,
        parent_idx: usize,
    ) -> RuntimeResult<DAGNode> {
        let left_idx = self.create_dependency(&be.left, parent_idx)?;
        let right_idx = self.create_dependency(&be.right, parent_idx)?;
        let node = BinopNode::new(left_idx, right_idx, be.op, be.modifier.clone());
        Ok(DAGNode::BinOp(node))
    }

    fn handle_args(
        &mut self,
        args: &[Expr],
        parent_idx: usize,
        arg_idx_for_optimization: Option<usize>,
    ) -> RuntimeResult<(Vec<QueryValue>, Vec<usize>, bool)> {
        let ignore_idx = arg_idx_for_optimization.unwrap_or(1000);
        if are_all_args_constant(args, arg_idx_for_optimization) {
            let mut params = Vec::with_capacity(args.len());
            // all args are constant, so we can evaluate the function now
            // copy values to params
            for (i, arg) in args.iter().enumerate() {
                if i == ignore_idx {
                    continue;
                }
                let value = match arg {
                    Expr::Number(n) => QueryValue::Scalar(n.value),
                    Expr::StringLiteral(s) => QueryValue::String(s.clone()),
                    _ => {
                        // this should never happen
                        unreachable!("Invalid arg type")
                    }
                };
                params.push(value);
            }
            return Ok((params, vec![], true));
        }

        let mut arg_indexes = Vec::with_capacity(args.len());
        for (i, arg) in args.iter().enumerate() {
            if i == ignore_idx {
                //
                arg_indexes.push(self.reserve_node());
                continue;
            }
            let idx = self.create_dependency(arg, parent_idx)?;
            arg_indexes.push(idx);
        }

        Ok((vec![], arg_indexes, false))
    }

    fn sort_nodes(&mut self) -> Vec<Vec<usize>> {
        let deps = self.graph.get_forward_dependency_topological_layers();
        deps.iter()
            .map(|layer| layer.iter().copied().collect())
            .collect()
    }

    fn add_dependency(&mut self, parent: usize, child: usize) {
        // NOTE: we don't check for cycles here, because we assume that the
        // expression is valid and doesn't contain cycles. If this crashes,
        // we have problems in the parser that need fixing.
        self.graph.depend_on(parent, child).unwrap();
    }
}

fn is_vector_expr(node: &Expr) -> bool {
    matches!(
        node,
        Expr::MetricExpression(_) | Expr::Rollup(_) | Expr::Aggregation(_) // what about functions returning vectors?
    )
}

fn are_all_args_constant(args: &[Expr], arg_idx_for_optimization: Option<usize>) -> bool {
    if let Some(idx) = arg_idx_for_optimization {
        for (i, arg) in args.iter().enumerate() {
            if i == idx {
                continue;
            }
            if !matches!(arg, Expr::Number(_) | Expr::StringLiteral(_)) {
                return false;
            }
        }
        return true;
    }
    args.iter()
        .all(|x| matches!(x, Expr::Number(_) | Expr::StringLiteral(_)))
}

// todo: COW
/// Normalize the rollup expr to standard form.
fn get_rollup_expr_arg(arg: &Expr) -> RuntimeResult<RollupExpr> {
    return match arg {
        Expr::Rollup(re) => {
            let mut re = re.clone();
            if !re.for_subquery() {
                // Return standard rollup if it doesn't contain subquery.
                return Ok(re);
            }

            match &re.expr.as_ref() {
                Expr::MetricExpression(_) => {
                    // Convert me[w:step] -> default_rollup(me)[w:step]

                    let arg = Expr::Rollup(RollupExpr::new(*re.expr.clone()));

                    match FunctionExpr::default_rollup(arg) {
                        Err(e) => return Err(RuntimeError::General(format!("{:?}", e))),
                        Ok(fe) => {
                            re.expr = Box::new(Expr::Function(fe));
                            Ok(re)
                        }
                    }
                }
                _ => {
                    // arg contains subquery.
                    Ok(re)
                }
            }
        }
        _ => {
            // Wrap non-rollup arg into RollupExpr.
            Ok(RollupExpr::new(arg.clone()))
        }
    };
}

fn try_get_arg_rollup_func_with_metric_expr(
    ae: &AggregationExpr,
) -> RuntimeResult<Option<FunctionExpr>> {
    if !ae.can_incrementally_eval {
        return Ok(None);
    }

    if ae.args.len() != 1 {
        return Ok(None);
    }

    let expr = &ae.args[0];
    // Make sure e contains one of the following:
    // - metricExpr
    // - metricExpr[d]
    // -: RollupFunc(metricExpr)
    // -: RollupFunc(metricExpr[d])

    fn create_func(
        me: &MetricExpr,
        expr: &Expr,
        name: &str,
        for_subquery: bool,
    ) -> RuntimeResult<Option<FunctionExpr>> {
        if me.is_empty() || for_subquery {
            return Ok(None);
        }

        let func_name = if name.is_empty() {
            "default_rollup"
        } else {
            name
        };

        match FunctionExpr::from_single_arg(func_name, expr.clone()) {
            Err(e) => Err(RuntimeError::General(format!(
                "Error creating function {func_name}: {:?}",
                e
            ))),
            Ok(fe) => Ok(Some(fe)),
        }
    }

    return match expr {
        Expr::MetricExpression(me) => return create_func(me, expr, "", false),
        Expr::Rollup(re) => {
            match re.expr.deref() {
                Expr::MetricExpression(me) => {
                    // e = metricExpr[d]
                    create_func(me, expr, "", re.for_subquery())
                }
                _ => Ok(None),
            }
        }
        Expr::Function(fe) => {
            match fe.function {
                BuiltinFunction::Rollup(_) => {
                    return if let Some(arg) = fe.get_arg_for_optimization() {
                        match arg.deref() {
                            Expr::MetricExpression(me) => create_func(me, expr, &fe.name, false),
                            Expr::Rollup(re) => {
                                if let Expr::MetricExpression(me) = &*re.expr {
                                    return if me.is_empty() || re.for_subquery() {
                                        Ok(None)
                                    } else {
                                        // e = RollupFunc(metricExpr[d])
                                        // todo: use COW to avoid clone
                                        Ok(Some(fe.clone()))
                                    };
                                } else {
                                    Ok(None)
                                }
                            }
                            _ => Ok(None),
                        }
                    } else {
                        // Incorrect number of args for rollup func.
                        // TODO: this should be an error
                        // all rollup functions should have a value for this
                        Ok(None)
                    };
                }
                _ => Ok(None),
            }
        }
        _ => Ok(None),
    };
}

#[cfg(test)]
mod tests {
    use metricsql::ast::MetricExpr;
    use metricsql::common::LabelFilter;
    use metricsql::prelude::{DurationExpr, Expr};

    use crate::execution::dag::duration_node::DurationNode;
    use crate::execution::{Context, EvalConfig};

    use super::*;

    const START: i64 = 1000000_i64;
    const END: i64 = 2000000_i64;
    const STEP: i64 = 200000_i64;

    #[test]
    fn test_create_node_from_number() {
        let expr = Expr::from(std::f64::consts::PI);
        let node = DAGBuilder::compile(&expr).unwrap();

        assert_eq!(node, DAGNode::from(std::f64::consts::PI));
    }

    #[test]
    fn test_create_node_from_string() {
        let expr = Expr::from("test");
        let node = DAGBuilder::compile(&expr).unwrap();

        assert!(matches!(node, DAGNode::Value(QueryValue::String(_))));
        assert_eq!(node, DAGNode::from("test"));
    }

    #[test]
    fn test_create_node_from_duration() {
        // a duration with a fixed millisecond value is created to a number
        let millis_expr = Expr::Duration(DurationExpr::Millis(1000));
        let millis_node = DAGBuilder::compile(&millis_expr).unwrap();

        assert!(matches!(millis_node, DAGNode::Value(_)));
        if let DAGNode::Value(QueryValue::Scalar(n)) = millis_node {
            assert_eq!(n, 1.0);
        } else {
            panic!("Expected DAGNode::Value(QueryValue::Number(_))");
        }

        // a duration with a step value millisecond value is created to a DurationNode
        let step_expr = Expr::Duration(DurationExpr::StepValue(1000f64));
        let step_node = DAGBuilder::compile(&step_expr).unwrap();
        assert!(matches!(step_node, DAGNode::Duration(_)));
        if let DAGNode::Duration(DurationNode(n)) = step_node {
            assert_eq!(n, DurationExpr::StepValue(1000f64));
        } else {
            panic!("Expected DAGNode::Duration(DurationNode::StepValue(_))");
        }
    }

    #[test]
    fn test_create_node_from_parens_expr() {
        // a parens expr with a single value resolves to a DAGNode::Value
        let single_node = Expr::Parens(ParensExpr {
            expressions: vec![Expr::from(2.5)],
        });

        let single_node = DAGBuilder::compile(&single_node).unwrap();
        assert_eq!(single_node, DAGNode::from(2.5));

        // create a Union function node for len() > 1
        let expr = Expr::Parens(ParensExpr {
            expressions: vec![Expr::from(2.5), Expr::from("foo"), Expr::from("bar")],
        });

        let node = DAGBuilder::compile(&expr).unwrap();
        assert!(matches!(node, DAGNode::Dynamic(_)));
        match node {
            DAGNode::Dynamic(t) => {
                let deps = &t.0.dag[t.0.dag.len() - 1];
                let func_node = &deps[0].node;
                assert!(matches!(func_node, DAGNode::Transform(_)));
                if let DAGNode::Transform(t) = func_node {
                    assert_eq!(t.function, TransformFunction::Union);
                    assert_eq!(t.args.len(), 3);
                }
            }
            _ => unreachable!(),
        };
    }

    #[test]
    fn test_create_node_from_selector() {
        let filters = vec![
            LabelFilter::equal("foo", "bar").unwrap(),
            LabelFilter::equal("baz", "qux").unwrap(),
        ];
        let expr = Expr::MetricExpression(MetricExpr::with_filters(filters.clone()));
        let node = DAGBuilder::compile(&expr).unwrap();
        if let DAGNode::Rollup(r) = node {
            assert_eq!(r.expr, expr);
            assert_eq!(r.metric_expr, MetricExpr::with_filters(filters.clone()));
            if let Expr::MetricExpression(MetricExpr { label_filters, .. }) = &r.expr {
                assert_eq!(&filters, label_filters);
            } else {
                panic!("Expected Expr::MetricExpression(_)");
            }
        } else {
            panic!("Expected DAGNode::Rollup(_)");
        }
    }

    #[test]
    fn test_create_number_number_binary_node() {
        let expr = Expr::BinaryOperator(BinaryExpr {
            left: Box::new(Expr::from(2.5)),
            right: Box::new(Expr::from(3.5)),
            op: Operator::Add,
            modifier: None,
        });

        let node = DAGBuilder::compile(&expr).unwrap();
        if let DAGNode::Value(b) = node {
            assert_eq!(b, QueryValue::from(6.0));
        } else {
            panic!("Expected DAGNode::Value(_)");
        }
    }

    #[test]
    fn test_create_string_string_binary_node() {
        let expr = Expr::BinaryOperator(BinaryExpr {
            left: Box::new(Expr::from("foo")),
            right: Box::new(Expr::from("bar")),
            op: Operator::Add,
            modifier: None,
        });

        let node = DAGBuilder::compile(&expr).unwrap();
        if let DAGNode::Value(b) = node {
            assert_eq!(b, QueryValue::from("foobar"));
        } else {
            panic!("Expected DAGNode::Value(_)");
        }
    }

    #[test]
    fn test_create_node_from_transform_function() {
        let func = FunctionExpr::new("sgn", vec![Expr::from(-2.5)]).unwrap();
        let expr = Expr::Function(func.clone());

        let mut node = DAGBuilder::compile(&expr).unwrap();
        if !matches!(node, DAGNode::Dynamic(_)) {
            panic!("Expected DAGNode::Dynamic(_)");
        }
        let ec = EvalConfig::new(START, END, STEP);
        let ctx = Context::default();
        let val = node.execute(&ctx, &ec).unwrap();
        if let QueryValue::InstantVector(v) = val {
            let first = &v[0];
            assert!(first.values.iter().all(|v| v.is_sign_negative()));
        } else {
            panic!("Expected QueryValue::InstantVector(_)");
        }
    }
}
