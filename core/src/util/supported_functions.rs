//! Utility functions to enable federation support for scalar functions.
//!
//! Helpful for implementing [`SQLExecutor::can_execute_plan`], when federating Datafusion Scalar UDFs with a `SQLExecutor` that may not support them.

use std::sync::Arc;

use datafusion::{
    catalog::Session,
    common::tree_node::{TreeNode, TreeNodeRecursion},
    common::DFSchema,
    error::DataFusionError,
    logical_expr::{
        expr::{AggregateFunction, ScalarFunction, WindowFunction},
        AggregateUDF, Expr, LogicalPlan, ScalarUDF, WindowFunctionDefinition, WindowUDF,
    },
};

/// Returns whether any [`ScalarFunction`], [`AggregateFunction`] or [`WindowFunction`]s in the [`LogicalPlan`] are unsupported.
///
/// # Arguments
/// * `plan` - The logical plan to check for scalar functions
/// * `supports` - The support policy (allow-list or deny-list)
///
/// # Returns
/// * `Ok(true)` if there are unsupported scalar functions in the plan
/// * `Ok(false)` if all scalar functions are supported
/// * `Err(DataFusionError)` if an error occurs during traversal
pub fn contains_unsupported_functions(
    plan: &LogicalPlan,
    sup: &FunctionSupport,
) -> Result<bool, DataFusionError> {
    let mut found_unsupported = false;
    plan.apply_with_subqueries(|plan| {
        let schema = expression_scope(plan);
        for expr in plan.expressions() {
            expr.apply(|expr| {
                if sup.supports(expr, schema.as_deref()) {
                    Ok(TreeNodeRecursion::Continue)
                } else {
                    found_unsupported = true;
                    Ok(TreeNodeRecursion::Stop)
                }
            })?;
            if found_unsupported {
                return Ok(TreeNodeRecursion::Stop);
            }
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok(found_unsupported)
}

/// The schema a node's expressions resolve against.
///
/// **Not** `plan.schema()`. That is the node's *output* schema: a `Projection`
/// computing `f(doc)` has only the result in it, and looking `doc` up there
/// fails. An expression resolves against the node's **inputs**, so those are
/// what a per-call check needs to read an operand's type.
///
/// `None` means the type cannot be proven here — a leaf with no inputs, or a
/// join whose sides cannot be merged. A check that depends on the type must
/// treat that as "unknown" and take the safe branch rather than guess; for a
/// rendering that is only correct for one of two column types, the safe branch
/// is to refuse the call and let the local engine evaluate it.
fn expression_scope(plan: &LogicalPlan) -> Option<Arc<DFSchema>> {
    let inputs = plan.inputs();
    match inputs.as_slice() {
        [] => None,
        [only] => Some(Arc::clone(only.schema())),
        many => {
            let mut merged = DFSchema::empty();
            for input in many {
                merged = merged.join(input.schema()).ok()?;
            }
            Some(Arc::new(merged))
        }
    }
}

/// Whether a backend can evaluate *this call* of a scalar function its
/// [`FunctionRestriction`] already allows.
///
/// A deny-list answers by name, which is all it needs until a backend carves a
/// name back out because its unparser dialect rewrites that function into
/// native SQL. But a dialect rewrites a *call*, not a name: it can translate
/// `f(col, 'literal')` and have no equivalent for `f(col, other_col)`. With no
/// way to say so, carving the name out federates the untranslatable call too,
/// and `Dialect::scalar_function_to_sql_overrides` returning `Ok(None)` for it
/// makes the unparser emit the function verbatim into SQL the remote cannot
/// parse.
///
/// A backend supplies this to refuse exactly the call shapes its dialect cannot
/// translate, so those are left for the local engine to evaluate instead. It is
/// consulted for every scalar call the name check allows, so an implementation
/// returns `true` for the functions it has no opinion about.
///
/// The schema is the one the call's arguments resolve against — see
/// [`expression_scope`] — and is `None` where the type cannot be proven. A
/// backend whose rendering depends on an operand's type must refuse on `None`
/// rather than assume one: for a document that is a native JSON column in one
/// engine and a string in another, guessing wrong is a wrong answer, and
/// refusing only costs the pushdown.
pub type ScalarCallSupport = Arc<dyn Fn(&ScalarFunction, Option<&DFSchema>) -> bool + Send + Sync>;

/// Whether a backend can evaluate *this call* of an aggregate its
/// [`FunctionRestriction`] already allows.
///
/// The aggregate counterpart of [`ScalarCallSupport`], and needed for the same
/// reason with one difference that matters: an aggregate call carries more than
/// its arguments. `FILTER (WHERE …)`, `ORDER BY` and `DISTINCT` are part of the
/// call, and a dialect may be able to translate an aggregate in one of those
/// shapes and not another — a `FILTER` an engine has no clause for can move
/// inside the aggregate for `count`, and cannot for an aggregate whose result
/// depends on input order.
///
/// A name-based restriction cannot express that. Denying the *name* is not an
/// option either, because these are names like `count` and `sum` that every
/// query uses; only the untranslatable *shapes* should stay local. A backend
/// supplies this to refuse exactly those, leaving them for the local engine.
///
/// Consulted for every aggregate call the name check allows, so an
/// implementation returns `true` for the shapes it has no opinion about.
pub type AggregateCallSupport = Arc<dyn Fn(&AggregateFunction) -> bool + Send + Sync>;

/// Whether a backend can evaluate *this* window call — the aggregate or window
/// function, together with the `FILTER`, `ORDER BY`, `DISTINCT` and frame the
/// `OVER` clause carries.
///
/// Needed separately from [`AggregateCallSupport`] rather than folded into it,
/// because an aggregate used as a window function is not the same shape: its
/// `ORDER BY` is the window's ordering, not an ordering of the aggregate's
/// input, so a rule about one does not transfer to the other. What *does*
/// transfer is `FILTER`: an engine with no `FILTER` clause refuses
/// `COUNT(x) FILTER (WHERE p) OVER (…)` exactly as it refuses the plain
/// aggregate form.
///
/// Consulted for every window call the name check allows, so an implementation
/// returns `true` for the shapes it has no opinion about.
pub type WindowCallSupport = Arc<dyn Fn(&WindowFunction) -> bool + Send + Sync>;

#[derive(Clone)]
pub struct FunctionSupport {
    scalar: Option<FunctionRestriction>,
    window: Option<FunctionRestriction>,
    aggregate: Option<FunctionRestriction>,
    scalar_call: Option<ScalarCallSupport>,
    aggregate_call: Option<AggregateCallSupport>,
    window_call: Option<WindowCallSupport>,
}

impl std::fmt::Debug for FunctionSupport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FunctionSupport")
            .field("scalar", &self.scalar)
            .field("window", &self.window)
            .field("aggregate", &self.aggregate)
            .field("scalar_call", &self.scalar_call.as_ref().map(|_| "<fn>"))
            .field(
                "aggregate_call",
                &self.aggregate_call.as_ref().map(|_| "<fn>"),
            )
            .field("window_call", &self.window_call.as_ref().map(|_| "<fn>"))
            .finish()
    }
}

impl FunctionSupport {
    pub fn new(
        scalar: Option<FunctionRestriction>,
        window: Option<FunctionRestriction>,
        aggregate: Option<FunctionRestriction>,
    ) -> Self {
        Self {
            scalar,
            window,
            aggregate,
            scalar_call: None,
            aggregate_call: None,
            window_call: None,
        }
    }

    /// Adds a per-call check, consulted for every scalar call the name-based
    /// restriction allows. See [`ScalarCallSupport`].
    #[must_use]
    pub fn with_scalar_call_support(mut self, scalar_call: ScalarCallSupport) -> Self {
        self.scalar_call = Some(scalar_call);
        self
    }

    /// Adds a per-call check, consulted for every aggregate call the name-based
    /// restriction allows. See [`AggregateCallSupport`].
    #[must_use]
    pub fn with_aggregate_call_support(mut self, aggregate_call: AggregateCallSupport) -> Self {
        self.aggregate_call = Some(aggregate_call);
        self
    }

    /// Adds a per-call check, consulted for every window call the name-based
    /// restriction allows. See [`WindowCallSupport`].
    #[must_use]
    pub fn with_window_call_support(mut self, window_call: WindowCallSupport) -> Self {
        self.window_call = Some(window_call);
        self
    }

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
            scalar_call: None,
            aggregate_call: None,
            window_call: None,
        }
    }

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
            scalar_call: None,
            aggregate_call: None,
            window_call: None,
        }
    }

    pub fn supports(&self, expr: &Expr, schema: Option<&DFSchema>) -> bool {
        let mut supports = true;
        let _ = expr.apply(|e| {
            let support_child = match e {
                Expr::ScalarFunction(call) => {
                    self.supports_scalar(&call.func) && self.supports_scalar_call(call, schema)
                }
                Expr::AggregateFunction(call) => {
                    self.supports_aggregate(&call.func) && self.supports_aggregate_call(call)
                }
                Expr::WindowFunction(wind) => {
                    let name_ok = match &wind.fun {
                        WindowFunctionDefinition::AggregateUDF(func) => {
                            self.supports_aggregate(func)
                        }
                        WindowFunctionDefinition::WindowUDF(func) => self.supports_window(func),
                    };
                    name_ok && self.supports_window_call(wind)
                }
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

    pub fn supports_window(&self, fnc: &Arc<WindowUDF>) -> bool {
        self.window
            .as_ref()
            .map(|restriction| restriction.supports(&fnc.name().to_string()))
            .unwrap_or(true)
    }
    /// Whether the backend can evaluate this particular scalar call. `true`
    /// when no per-call check is installed, so a backend that has nothing to
    /// say about call shapes behaves exactly as before.
    pub fn supports_scalar_call(&self, call: &ScalarFunction, schema: Option<&DFSchema>) -> bool {
        self.scalar_call
            .as_ref()
            .map(|supports| supports(call, schema))
            .unwrap_or(true)
    }

    /// Whether the backend can evaluate this particular aggregate call. `true`
    /// when no per-call check is installed, so a backend that has nothing to say
    /// about call shapes behaves exactly as before.
    pub fn supports_aggregate_call(&self, call: &AggregateFunction) -> bool {
        self.aggregate_call
            .as_ref()
            .map(|supports| supports(call))
            .unwrap_or(true)
    }

    /// Whether the backend can evaluate this particular window call. `true` when
    /// no per-call check is installed, so a backend that has nothing to say about
    /// window shapes behaves exactly as before.
    pub fn supports_window_call(&self, call: &WindowFunction) -> bool {
        self.window_call
            .as_ref()
            .map(|supports| supports(call))
            .unwrap_or(true)
    }

    pub fn supports_scalar(&self, fnc: &Arc<ScalarUDF>) -> bool {
        self.scalar
            .as_ref()
            .map(|restriction| restriction.supports(&fnc.name().to_string()))
            .unwrap_or(true)
    }
    pub fn supports_aggregate(&self, fnc: &Arc<AggregateUDF>) -> bool {
        self.aggregate
            .as_ref()
            .map(|restriction| restriction.supports(&fnc.name().to_string()))
            .unwrap_or(true)
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

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::logical_expr::builder::LogicalTableSource;
    use datafusion::logical_expr::expr::ScalarFunction;
    use datafusion::logical_expr::{create_udf, ColumnarValue, LogicalPlanBuilder, Subquery};
    use datafusion::prelude::col;
    use datafusion::scalar::ScalarValue;
    use std::sync::Arc;

    fn stub_udf(name: &str) -> Arc<ScalarUDF> {
        Arc::new(create_udf(
            name,
            vec![DataType::Utf8],
            DataType::Utf8,
            datafusion::logical_expr::Volatility::Immutable,
            Arc::new(|args: &[ColumnarValue]| Ok(args[0].clone())),
        ))
    }

    fn deny_support(names: &[&str]) -> FunctionSupport {
        FunctionSupport::new(
            Some(FunctionRestriction::Deny(
                names.iter().map(|s| s.to_string()).collect(),
            )),
            None,
            None,
        )
    }

    fn scan_plan(table: &str) -> LogicalPlan {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("val", DataType::Utf8, true),
        ]));
        let source = Arc::new(LogicalTableSource::new(schema))
            as Arc<dyn datafusion::logical_expr::TableSource>;
        LogicalPlanBuilder::scan(table, source, None)
            .expect("scan")
            .build()
            .expect("build")
    }

    /// The scope handed to a per-call check is the one the call's arguments
    /// resolve against, so a backend can read an operand's declared type.
    ///
    /// This is the whole point of passing a schema, and it is easy to get
    /// wrong: `plan.schema()` is a `Projection`'s *output*, which holds only the
    /// computed result — looking the argument's column up there fails, every
    /// call looks unprovable, and a check that refuses on an unknown type
    /// silently refuses everything. Asserting the column resolves is what
    /// separates "the schema arrived" from "a schema arrived".
    #[test]
    fn the_scope_resolves_the_arguments_not_the_projected_output() {
        use datafusion::logical_expr::ExprSchemable as _;
        use std::sync::Mutex;

        // What the check saw, recorded so the assertions can be made outside it.
        let seen: Arc<Mutex<Vec<(bool, Option<DataType>)>>> = Arc::new(Mutex::new(Vec::new()));
        let recorder = Arc::clone(&seen);

        let support = FunctionSupport::new(None, None, None).with_scalar_call_support(Arc::new(
            move |call: &ScalarFunction, schema: Option<&DFSchema>| {
                let resolved = schema
                    .and_then(|schema| call.args.first().and_then(|arg| arg.get_type(schema).ok()));
                recorder
                    .lock()
                    .expect("record what the check saw")
                    .push((schema.is_some(), resolved));
                true
            },
        ));

        let plan = projection_of(call("any_fn", vec![col("val")]));
        assert!(
            !contains_unsupported_functions(&plan, &support).expect("walk the plan"),
            "the check allows everything; this is about what it was handed"
        );

        let seen = seen.lock().expect("read what the check saw");
        assert!(!seen.is_empty(), "the per-call check has to be consulted");
        assert!(
            seen.iter().all(|(had_schema, _)| *had_schema),
            "every call must be handed a scope, not None"
        );
        assert!(
            seen.iter().any(|(_, resolved)| resolved.is_some()),
            "the argument's column has to resolve against the scope — if this \
             fails the output schema was passed instead of the input's"
        );
    }

    /// A `FunctionSupport` that allows every name but refuses calls of
    /// `carved_out_fn` whose second argument is not a literal — the shape a
    /// dialect that rewrites a literal path can translate.
    fn literal_argument_support() -> FunctionSupport {
        FunctionSupport::new(None, None, None).with_scalar_call_support(Arc::new(
            |call: &ScalarFunction, _: Option<&DFSchema>| {
                if call.func.name() != "carved_out_fn" {
                    return true;
                }
                call.args
                    .iter()
                    .skip(1)
                    .all(|arg| matches!(arg, Expr::Literal(_, _)))
            },
        ))
    }

    /// A `FunctionSupport` that allows every name but refuses a `count` call
    /// carrying a `FILTER` — the shape a dialect with no `FILTER` clause has to
    /// rewrite, and cannot rewrite for every aggregate.
    fn unfiltered_aggregate_support() -> FunctionSupport {
        FunctionSupport::new(None, None, None).with_aggregate_call_support(Arc::new(
            |call: &AggregateFunction| call.func.name() != "count" || call.params.filter.is_none(),
        ))
    }

    fn projection_of(expr: Expr) -> LogicalPlan {
        LogicalPlanBuilder::from(scan_plan("t"))
            .project(vec![expr])
            .expect("project")
            .build()
            .expect("build")
    }

    /// A stub UDF taking as many `Utf8` arguments as the call passes it, so a
    /// two-argument call plans.
    fn call(name: &str, args: Vec<Expr>) -> Expr {
        let udf = Arc::new(create_udf(
            name,
            vec![DataType::Utf8; args.len()],
            DataType::Utf8,
            datafusion::logical_expr::Volatility::Immutable,
            Arc::new(|args: &[ColumnarValue]| Ok(args[0].clone())),
        ));
        Expr::ScalarFunction(ScalarFunction::new_udf(udf, args))
    }

    #[test]
    fn a_call_shape_the_backend_can_translate_is_supported() {
        let plan = projection_of(call(
            "carved_out_fn",
            vec![col("val"), Expr::Literal(ScalarValue::from("key"), None)],
        ));

        assert!(
            !contains_unsupported_functions(&plan, &literal_argument_support()).expect("check"),
            "a literal argument is the shape the backend declared it can translate"
        );
    }

    #[test]
    fn a_call_shape_the_backend_cannot_translate_is_unsupported() {
        let plan = projection_of(call("carved_out_fn", vec![col("val"), col("val")]));

        assert!(
            contains_unsupported_functions(&plan, &literal_argument_support()).expect("check"),
            "a non-literal argument has no translation, so the plan must not federate"
        );
    }

    #[test]
    fn the_per_call_check_only_speaks_for_the_functions_it_names() {
        let plan = projection_of(call("other_fn", vec![col("val"), col("val")]));

        assert!(
            !contains_unsupported_functions(&plan, &literal_argument_support()).expect("check"),
            "a function the check returns true for is unaffected by it"
        );
    }

    #[test]
    fn a_denied_name_stays_denied_whatever_its_call_shape() {
        let support = FunctionSupport::new(
            Some(FunctionRestriction::Deny(vec!["denied_fn".to_string()])),
            None,
            None,
        )
        .with_scalar_call_support(Arc::new(|_: &ScalarFunction, _: Option<&DFSchema>| true));
        let plan = projection_of(call(
            "denied_fn",
            vec![col("val"), Expr::Literal(ScalarValue::from("key"), None)],
        ));

        assert!(
            contains_unsupported_functions(&plan, &support).expect("check"),
            "the per-call check narrows what the name check allows, it never widens it"
        );
    }

    #[test]
    fn a_call_shape_is_refused_wherever_it_appears_in_the_plan() {
        // The per-call check has to reach as far as the name check does, which
        // includes an expression nested under another function call.
        let plan = projection_of(call(
            "outer_fn",
            vec![call("carved_out_fn", vec![col("val"), col("val")])],
        ));

        assert!(
            contains_unsupported_functions(&plan, &literal_argument_support()).expect("check"),
            "an untranslatable call nested inside another call must still be refused"
        );
    }

    #[test]
    fn detects_denied_function_in_top_level_projection() {
        let udf = stub_udf("denied_fn");
        let plan = LogicalPlanBuilder::from(scan_plan("t"))
            .project(vec![Expr::ScalarFunction(ScalarFunction::new_udf(
                udf,
                vec![col("val")],
            ))])
            .expect("project")
            .build()
            .expect("build");

        let sup = deny_support(&["denied_fn"]);
        assert!(
            contains_unsupported_functions(&plan, &sup).expect("check"),
            "should detect denied function in top-level projection"
        );
    }

    #[test]
    fn allows_plan_without_denied_functions() {
        let udf = stub_udf("allowed_fn");
        let plan = LogicalPlanBuilder::from(scan_plan("t"))
            .project(vec![Expr::ScalarFunction(ScalarFunction::new_udf(
                udf,
                vec![col("val")],
            ))])
            .expect("project")
            .build()
            .expect("build");

        let sup = deny_support(&["denied_fn"]);
        assert!(
            !contains_unsupported_functions(&plan, &sup).expect("check"),
            "should allow plan with only non-denied functions"
        );
    }

    #[test]
    fn detects_denied_function_inside_in_subquery() {
        let udf = stub_udf("denied_fn");

        // Build subquery: SELECT denied_fn(val) FROM inner_t
        let subquery_plan = LogicalPlanBuilder::from(scan_plan("inner_t"))
            .project(vec![Expr::ScalarFunction(ScalarFunction::new_udf(
                udf,
                vec![col("val")],
            ))
            .alias("result")])
            .expect("project")
            .build()
            .expect("build");

        // Build outer: SELECT id FROM t WHERE id IN (subquery)
        let outer = LogicalPlanBuilder::from(scan_plan("t"))
            .filter(Expr::InSubquery(
                datafusion::logical_expr::expr::InSubquery::new(
                    Box::new(col("id")),
                    Subquery {
                        subquery: Arc::new(subquery_plan),
                        outer_ref_columns: vec![],
                        spans: Default::default(),
                    },
                    false,
                ),
            ))
            .expect("filter")
            .build()
            .expect("build");

        let sup = deny_support(&["denied_fn"]);
        assert!(
            contains_unsupported_functions(&outer, &sup).expect("check"),
            "should detect denied function inside IN subquery"
        );
    }

    #[test]
    fn detects_denied_function_inside_scalar_subquery() {
        let udf = stub_udf("denied_fn");

        // Build scalar subquery: SELECT denied_fn(val) FROM inner_t
        let subquery_plan = LogicalPlanBuilder::from(scan_plan("inner_t"))
            .project(vec![Expr::ScalarFunction(ScalarFunction::new_udf(
                udf,
                vec![col("val")],
            ))
            .alias("result")])
            .expect("project")
            .build()
            .expect("build");

        // Build outer: SELECT id FROM t WHERE id = (scalar subquery)
        let outer = LogicalPlanBuilder::from(scan_plan("t"))
            .filter(col("id").eq(Expr::ScalarSubquery(Subquery {
                subquery: Arc::new(subquery_plan),
                outer_ref_columns: vec![],
                spans: Default::default(),
            })))
            .expect("filter")
            .build()
            .expect("build");

        let sup = deny_support(&["denied_fn"]);
        assert!(
            contains_unsupported_functions(&outer, &sup).expect("check"),
            "should detect denied function inside scalar subquery"
        );
    }

    #[test]
    fn detects_denied_function_inside_exists_subquery() {
        let udf = stub_udf("denied_fn");

        // Build subquery: SELECT denied_fn(val) FROM inner_t
        let subquery_plan = LogicalPlanBuilder::from(scan_plan("inner_t"))
            .project(vec![Expr::ScalarFunction(ScalarFunction::new_udf(
                udf,
                vec![col("val")],
            ))
            .alias("result")])
            .expect("project")
            .build()
            .expect("build");

        // Build outer: SELECT id FROM t WHERE EXISTS (subquery)
        let outer = LogicalPlanBuilder::from(scan_plan("t"))
            .filter(Expr::Exists(datafusion::logical_expr::expr::Exists::new(
                Subquery {
                    subquery: Arc::new(subquery_plan),
                    outer_ref_columns: vec![],
                    spans: Default::default(),
                },
                false,
            )))
            .expect("filter")
            .build()
            .expect("build");

        let sup = deny_support(&["denied_fn"]);
        assert!(
            contains_unsupported_functions(&outer, &sup).expect("check"),
            "should detect denied function inside EXISTS subquery"
        );
    }

    /// A per-call aggregate check refuses only the shapes it names, so an
    /// aggregate whose name is allowed and whose shape it has no opinion about
    /// still federates.
    #[test]
    fn an_aggregate_call_check_refuses_only_the_shape_it_names() {
        let bare = projection_of(datafusion::functions_aggregate::expr_fn::count(col("val")));
        assert!(
            !contains_unsupported_functions(&bare, &unfiltered_aggregate_support()).expect("check"),
            "an unfiltered count is a shape the check allows, so it federates"
        );

        let filtered = projection_of({
            use datafusion::logical_expr::ExprFunctionExt as _;
            datafusion::functions_aggregate::expr_fn::count(col("val"))
                .filter(col("val").is_not_null())
                .build()
                .expect("filtered count")
        });
        assert!(
            contains_unsupported_functions(&filtered, &unfiltered_aggregate_support())
                .expect("check"),
            "a filtered count is the shape the check refuses, so it stays local"
        );
    }

    /// With no per-call aggregate check installed, every aggregate the name
    /// restriction allows still federates — the hook is additive.
    #[test]
    fn no_aggregate_call_check_leaves_every_allowed_aggregate_federating() {
        let filtered = projection_of({
            use datafusion::logical_expr::ExprFunctionExt as _;
            datafusion::functions_aggregate::expr_fn::count(col("val"))
                .filter(col("val").is_not_null())
                .build()
                .expect("filtered count")
        });
        assert!(
            !contains_unsupported_functions(&filtered, &FunctionSupport::new(None, None, None))
                .expect("check"),
            "without a per-call check the shape is irrelevant"
        );
    }

    /// A windowed `count`, with or without a `FILTER`.
    fn windowed_count(filter: Option<Expr>) -> Expr {
        Expr::from(WindowFunction {
            fun: WindowFunctionDefinition::AggregateUDF(
                datafusion::functions_aggregate::count::count_udaf(),
            ),
            params: datafusion::logical_expr::expr::WindowFunctionParams {
                args: vec![col("val")],
                partition_by: vec![col("id")],
                order_by: vec![],
                window_frame: datafusion::logical_expr::WindowFrame::new(None),
                null_treatment: None,
                filter: filter.map(Box::new),
                distinct: false,
            },
        })
    }

    /// The window arm consults its own per-call check, so a shape an engine
    /// cannot express is refused there too.
    ///
    /// Without this the aggregate check is trivially bypassed: writing the same
    /// aggregate as `COUNT(x) FILTER (WHERE p) OVER (…)` reaches the window arm,
    /// which only ever checked the *name*. Measured against a real BigQuery
    /// project, that federated as-written and came back
    /// `Syntax error: Expected ")" but got "("` — BigQuery has no `FILTER`
    /// clause in a window either.
    #[test]
    fn a_window_call_check_refuses_the_shape_the_engine_cannot_express() {
        let support = FunctionSupport::new(None, None, None).with_window_call_support(Arc::new(
            |call: &WindowFunction| call.params.filter.is_none(),
        ));

        let filtered = projection_of(windowed_count(Some(col("val").is_not_null())));
        assert!(
            contains_unsupported_functions(&filtered, &support).expect("check"),
            "a filtered window call is the shape the check refuses"
        );

        let unfiltered = projection_of(windowed_count(None));
        assert!(
            !contains_unsupported_functions(&unfiltered, &support).expect("check"),
            "an unfiltered window call is untouched by the check"
        );
    }

    /// With no per-call window check installed, every window call the name
    /// restriction allows still federates — the hook is additive.
    #[test]
    fn no_window_call_check_leaves_every_allowed_window_call_federating() {
        let filtered = projection_of(windowed_count(Some(col("val").is_not_null())));
        assert!(
            !contains_unsupported_functions(&filtered, &FunctionSupport::new(None, None, None))
                .expect("check"),
            "without a per-call check the window shape is irrelevant"
        );
    }
}
