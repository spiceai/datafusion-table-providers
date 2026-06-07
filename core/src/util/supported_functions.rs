//! Utility functions to enable federation support for scalar functions.
//!
//! Helpful for implementing un-federation when a `SQLExecutor` may not support
//! some DataFusion Scalar/Aggregate/Window UDFs. The federation provider can use
//! [`contains_unsupported_functions`] (typically from a
//! [`datafusion_federation::sql::SQLExecutor::logical_optimizer`] hook) to detect
//! plans that reference denied functions and evaluate them locally instead of
//! pushing them down to the remote engine.

use std::sync::Arc;

use datafusion::{
    catalog::Session,
    common::tree_node::{TreeNode, TreeNodeRecursion},
    error::DataFusionError,
    logical_expr::{
        expr::{AggregateFunction, ScalarFunction},
        AggregateUDF, Expr, LogicalPlan, ScalarUDF, WindowUDF,
    },
};
#[cfg(feature = "federation")]
use datafusion::{error::Result as DataFusionResult, logical_expr::Extension};
#[cfg(feature = "federation")]
use datafusion_federation::FederatedPlanNode;

/// Returns whether any [`ScalarFunction`], [`AggregateFunction`] or window function in the
/// [`LogicalPlan`] are unsupported.
///
/// # Arguments
/// * `plan` - The logical plan to check for functions
/// * `sup` - The support policy (allow-list or deny-list)
///
/// # Returns
/// * `Ok(true)` if there are unsupported functions in the plan
/// * `Ok(false)` if all functions are supported
/// * `Err(DataFusionError)` if an error occurs during traversal
pub fn contains_unsupported_functions(
    plan: &LogicalPlan,
    sup: &FunctionSupport,
) -> Result<bool, DataFusionError> {
    plan.exists(|plan| {
        Ok(plan.expressions().into_iter().any(|expr| {
            let mut found_unsupported = false;
            let _ = expr.apply(|expr| {
                if sup.supports(expr) {
                    Ok(TreeNodeRecursion::Continue)
                } else {
                    found_unsupported = true;
                    Ok(TreeNodeRecursion::Stop)
                }
            });
            found_unsupported
        }))
    })
}

/// Un-federate a plan node if the federated sub-plan references any unsupported
/// function.
///
/// Intended to be returned from a federation `SQLExecutor::logical_optimizer`
/// closure. The federation optimizer wraps a federated sub-plan in a
/// `LogicalPlan::Extension` holding a `FederatedPlanNode` and then calls the
/// executor's `logical_optimizer`. If that sub-plan contains a denied function,
/// returning the inner (unwrapped) plan strips the federation node so
/// `DataFusion` executes the affected expressions locally instead of pushing
/// them down to the remote engine (which would fail with an "unknown function"
/// error).
#[cfg(feature = "federation")]
pub fn unfederate_plan_with_unsupported_functions(
    plan: LogicalPlan,
    function_support: &FunctionSupport,
) -> DataFusionResult<LogicalPlan> {
    if let LogicalPlan::Extension(Extension { node }) = &plan {
        if let Some(federated) = node.as_any().downcast_ref::<FederatedPlanNode>() {
            if contains_unsupported_functions(federated.plan(), function_support)? {
                return Ok(federated.plan().clone());
            }
        }
    }

    Ok(plan)
}

#[derive(Clone, Debug)]
pub struct FunctionSupport {
    scalar: Option<FunctionRestriction>,
    window: Option<FunctionRestriction>,
    aggregate: Option<FunctionRestriction>,
}

impl FunctionSupport {
    #[must_use]
    pub fn new(
        scalar: Option<FunctionRestriction>,
        window: Option<FunctionRestriction>,
        aggregate: Option<FunctionRestriction>,
    ) -> Self {
        Self {
            scalar,
            window,
            aggregate,
        }
    }

    #[must_use]
    pub fn deny_all_from(sess: &dyn Session) -> Self {
        let scalar = Some(FunctionRestriction::Deny(
            sess.scalar_functions().keys().cloned().collect::<Vec<_>>(),
        ));
        let window = Some(FunctionRestriction::Deny(
            sess.window_functions().keys().cloned().collect::<Vec<_>>(),
        ));
        let aggregate = Some(FunctionRestriction::Deny(
            sess.aggregate_functions()
                .keys()
                .cloned()
                .collect::<Vec<_>>(),
        ));

        FunctionSupport {
            scalar,
            window,
            aggregate,
        }
    }

    #[must_use]
    pub fn support_all_from(sess: &dyn Session) -> Self {
        let scalar = Some(FunctionRestriction::Allow(
            sess.scalar_functions().keys().cloned().collect::<Vec<_>>(),
        ));
        let window = Some(FunctionRestriction::Allow(
            sess.window_functions().keys().cloned().collect::<Vec<_>>(),
        ));
        let aggregate = Some(FunctionRestriction::Allow(
            sess.aggregate_functions()
                .keys()
                .cloned()
                .collect::<Vec<_>>(),
        ));

        FunctionSupport {
            scalar,
            window,
            aggregate,
        }
    }

    #[must_use]
    pub fn supports(&self, expr: &Expr) -> bool {
        let mut supports = true;
        let _ = expr.apply(|e| {
            let support_child = match e {
                Expr::ScalarFunction(ScalarFunction { func, .. }) => self.supports_scalar(func),
                Expr::AggregateFunction(AggregateFunction { func, .. }) => {
                    self.supports_aggregate(func)
                }
                Expr::WindowFunction(wind) => self.supports_window_def(&wind.fun),
                _ => true,
            };
            if !support_child {
                supports = false;
                return Ok(TreeNodeRecursion::Stop);
            }
            Ok(TreeNodeRecursion::Continue)
        });
        supports
    }

    /// Resolve support for a window function definition by name. On DataFusion 53
    /// [`datafusion::logical_expr::WindowFunctionDefinition`] exposes `name()`
    /// directly, so a window function is checked against whichever restriction
    /// (window or aggregate) the definition resolves to via its name.
    fn supports_window_def(
        &self,
        fun: &datafusion::logical_expr::WindowFunctionDefinition,
    ) -> bool {
        let name = fun.name().to_string();
        // A window function may be backed by an aggregate UDF; check the window
        // restriction first, then fall back to the aggregate restriction.
        self.window
            .as_ref()
            .map_or(true, |restriction| restriction.supports(&name))
            && self
                .aggregate
                .as_ref()
                .map_or(true, |restriction| restriction.supports(&name))
    }

    #[must_use]
    pub fn supports_window(&self, fnc: &Arc<WindowUDF>) -> bool {
        self.window
            .as_ref()
            .map_or(true, |restriction| {
                restriction.supports(&fnc.name().to_string())
            })
    }
    #[must_use]
    pub fn supports_scalar(&self, fnc: &Arc<ScalarUDF>) -> bool {
        self.scalar
            .as_ref()
            .map_or(true, |restriction| {
                restriction.supports(&fnc.name().to_string())
            })
    }
    #[must_use]
    pub fn supports_aggregate(&self, fnc: &Arc<AggregateUDF>) -> bool {
        self.aggregate
            .as_ref()
            .map_or(true, |restriction| {
                restriction.supports(&fnc.name().to_string())
            })
    }
}

#[derive(Clone, Debug)]
pub enum FunctionRestriction {
    Allow(Vec<String>),
    Deny(Vec<String>),
}

impl FunctionRestriction {
    fn supports(&self, name: &String) -> bool {
        match self {
            Self::Allow(allowed) => allowed.contains(name),
            Self::Deny(denied) => !denied.contains(name),
        }
    }
}
