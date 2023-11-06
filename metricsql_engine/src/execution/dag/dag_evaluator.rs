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

        DAGEvaluator {
            dag: nodes,
            computed,
        }
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

        Ok(std::mem::take(&mut self.computed[0]))
    }
}
