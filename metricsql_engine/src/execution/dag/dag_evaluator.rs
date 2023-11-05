use rayon::iter::IntoParallelRefMutIterator;

use crate::execution::dag::ExecutableNode;
use crate::execution::{Context, EvalConfig};
use crate::rayon::iter::ParallelIterator;
use crate::{QueryValue, RuntimeResult};

use super::DAGNode;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Dependency {
    pub(crate) node: DAGNode,
    pub(crate) result_index: usize,
}

impl Dependency {
    pub(crate) fn new(node: DAGNode, result_index: usize) -> Dependency {
        Dependency { node, result_index }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct DAGEvaluator {
    /// Adjacency list of DAG nodes
    /// The first vec contains the nodes that have no dependencies.
    /// Each subsequent vec contains the nodes that depend on the nodes in the previous layer.
    /// The last vec contains nodes that depend on all other nodes.
    pub(super) dag: Vec<Vec<Dependency>>,

    /// collects the results of the execution of dependencies (1 per dependency). The results are
    /// fed to the nodes that depend on them in a pre-execute phase.
    pub(super) computed: Vec<QueryValue>,
}

impl DAGEvaluator {
    pub(crate) fn new(nodes: Vec<Vec<Dependency>>, node_count: usize) -> DAGEvaluator {
        // todo: tiny vec
        // allocate scratch space for the results of the dependencies. We initialize the entire
        // vector with default values, and then we overwrite the values as we execute the
        // dependencies.
        let mut computed = Vec::with_capacity(node_count + 1);
        computed.resize_with(node_count + 1, Default::default);

        let evaluator = DAGEvaluator {
            dag: nodes,
            computed,
        };
        // evaluator.optimize();
        evaluator
    }

    pub fn evaluate(&mut self, ctx: &Context, ec: &EvalConfig) -> RuntimeResult<QueryValue> {
        for (i, dependencies) in self.dag.iter_mut().enumerate() {
            // update dependencies
            if i > 0 {
                for dep in dependencies.iter_mut() {
                    dep.node.pre_execute(&mut self.computed)?;
                }
            }

            let len = dependencies.len();
            let computed = &mut self.computed[0..];
            match len {
                1 => {
                    let dependency = &mut dependencies[0];
                    let idx = dependency.result_index;
                    let node = &mut dependency.node;
                    computed[idx] = node.execute(ctx, ec)?;
                }
                _ => {
                    // todo: use another iteration method to avoid the allocation (collect)
                    let res = dependencies
                        .par_iter_mut()
                        .map(|dependency| {
                            let value = dependency.node.execute(ctx, ec);
                            (dependency.result_index, value)
                        })
                        .collect::<Vec<_>>();

                    // set values for subsequent nodes
                    for (index, value) in res {
                        computed[index] = value?;
                    }
                }
            }
        }

        let res = std::mem::take(&mut self.computed[0]);
        Ok(res)
    }

    /// Optimizes the DAG by extracting constant nodes and copying them to our output scratch space.
    /// For example, consider the following:
    ///
    ///   sort_desc(2 * (label_set(time(), "foo", "bar", "__name__", "q1")
    ///     or label_set(10, "foo", "qwerty", "__name__", "q2")
    ///     ) keep_metric_names)
    ///
    ///  Here we have 10 constants (2, "foo", "bar", "__name__", "q1", 10, "foo", "qwerty", ...) which
    ///  would otherwise be represented as 10 nodes in the DAG. By extracting these constants,
    ///  there is no need to pass constants to rayon, so there is less work to do. We need to ensure
    ///  however that these values are resolved by the individual nodes. `resolve_value` is used
    ///  for this purpose
    pub(crate) fn optimize(&mut self) {
        let mut result_dag = Vec::with_capacity(self.dag.len());

        let push_node =
            |dep: &mut Dependency, layer: &mut Vec<Dependency>, computed: &mut Vec<QueryValue>| {
                match &dep.node {
                    DAGNode::Value(node) => match node {
                        QueryValue::Scalar(v) => {
                            computed[dep.result_index] = QueryValue::from(*v);
                        }
                        QueryValue::String(s) => {
                            computed[dep.result_index] = QueryValue::from(s.as_str());
                        }
                        _ => {
                            layer.push(std::mem::take(dep));
                        }
                    },
                    _ => {
                        layer.push(std::mem::take(dep));
                    }
                }
            };

        let mut computed_len: usize = 0;

        for dependencies in self.dag.iter() {
            computed_len = std::cmp::max(
                computed_len,
                dependencies
                    .iter()
                    .max_by(|a, b| a.result_index.cmp(&b.result_index))
                    .unwrap()
                    .result_index
                    + 1,
            );
        }

        // todo: tiny vec
        // allocate scratch space for the results of the dependencies. We initialize the entire
        // vector with default values, and then we overwrite the values as we execute the
        // dependencies.
        let mut computed = Vec::with_capacity(computed_len);
        computed.resize_with(computed_len, Default::default);

        for dependencies in self.dag.iter_mut() {
            let mut layer = Vec::with_capacity(dependencies.len());
            for dep in dependencies.iter_mut() {
                push_node(dep, &mut layer, &mut computed);
            }

            if !layer.is_empty() {
                layer.shrink_to_fit();
                result_dag.push(layer);
            }
        }

        self.dag = result_dag;
        self.computed = computed;
    }
}
