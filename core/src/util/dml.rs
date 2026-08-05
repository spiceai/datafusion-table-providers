use std::{fmt, sync::Arc};

use async_trait::async_trait;
use datafusion::{
    error::DataFusionError,
    execution::{SendableRecordBatchStream, TaskContext},
    logical_expr::Expr,
    physical_plan::{
        stream::RecordBatchStreamAdapter, DisplayAs, DisplayFormatType, ExecutionPlan,
        PlanProperties,
    },
};

use super::count_exec::{count_schema, count_to_record_batch};
use crate::sql::sql_provider_datafusion::expr;

/// Converts filter expressions to a SQL WHERE clause string.
pub fn filters_to_sql(
    filters: &[Expr],
    engine: Option<expr::Engine>,
) -> datafusion::error::Result<String> {
    let sql_parts: Result<Vec<String>, _> = filters
        .iter()
        .map(|f| expr::to_sql_with_engine(f, engine))
        .collect();
    sql_parts
        .map(|parts| match parts.as_slice() {
            [] => String::new(),
            // Nothing is composed with a lone filter, so it needs no parentheses.
            [only] => only.clone(),
            // Every filter is parenthesized before being joined, because `AND` binds tighter
            // than `OR`: a filter that is itself a disjunction would otherwise be re-grouped by
            // the join. `[a OR b, c]` has to mean `(a OR b) AND c`, not `a OR (b AND c)`, which
            // matches different rows.
            _ => parts
                .iter()
                .map(|part| format!("({part})"))
                .collect::<Vec<String>>()
                .join(" AND "),
        })
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))
}

/// Converts assignment expressions to a SQL SET clause string.
pub fn assignments_to_sql(
    assignments: &[(String, Expr)],
    engine: Option<expr::Engine>,
) -> datafusion::error::Result<String> {
    let parts: Result<Vec<String>, _> = assignments
        .iter()
        .map(|(col, val)| {
            expr::to_sql_with_engine(val, engine).map(|sql_val| format!(r#""{col}" = {sql_val}"#))
        })
        .collect();
    parts
        .map(|p| p.join(", "))
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))
}

#[async_trait]
pub trait DeletionSink: Send + Sync {
    async fn delete_from(&self) -> Result<u64, Box<dyn std::error::Error + Send + Sync>>;
}

pub struct DeletionExec {
    deletion_sink: Arc<dyn DeletionSink + 'static>,
    properties: Arc<PlanProperties>,
}

impl DeletionExec {
    pub fn new(deletion_sink: Arc<dyn DeletionSink>) -> Self {
        let properties = PlanProperties::new(
            datafusion::physical_expr::EquivalenceProperties::new(count_schema()),
            datafusion::physical_plan::Partitioning::UnknownPartitioning(1),
            datafusion::physical_plan::execution_plan::EmissionType::Final,
            datafusion::physical_plan::execution_plan::Boundedness::Bounded,
        );
        Self {
            deletion_sink,
            properties: Arc::new(properties),
        }
    }
}

impl fmt::Debug for DeletionExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeletionExec").finish_non_exhaustive()
    }
}

impl DisplayAs for DeletionExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DeletionExec")
    }
}

impl ExecutionPlan for DeletionExec {
    fn name(&self) -> &'static str {
        "DeletionExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> datafusion::error::Result<SendableRecordBatchStream> {
        let schema = count_schema();
        let deletion_sink = Arc::clone(&self.deletion_sink);
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&schema),
            futures::stream::once(async move {
                let count = deletion_sink
                    .delete_from()
                    .await
                    .map_err(DataFusionError::External)?;
                count_to_record_batch(schema, count)
            }),
        )))
    }
}

#[async_trait]
pub trait UpdateSink: Send + Sync {
    async fn execute_update(&self) -> Result<u64, Box<dyn std::error::Error + Send + Sync>>;
}

pub struct UpdateExec {
    update_sink: Arc<dyn UpdateSink + 'static>,
    properties: Arc<PlanProperties>,
}

impl UpdateExec {
    pub fn new(update_sink: Arc<dyn UpdateSink>) -> Self {
        let properties = PlanProperties::new(
            datafusion::physical_expr::EquivalenceProperties::new(count_schema()),
            datafusion::physical_plan::Partitioning::UnknownPartitioning(1),
            datafusion::physical_plan::execution_plan::EmissionType::Final,
            datafusion::physical_plan::execution_plan::Boundedness::Bounded,
        );
        Self {
            update_sink,
            properties: Arc::new(properties),
        }
    }
}

impl fmt::Debug for UpdateExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UpdateExec").finish_non_exhaustive()
    }
}

impl DisplayAs for UpdateExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "UpdateExec")
    }
}

impl ExecutionPlan for UpdateExec {
    fn name(&self) -> &'static str {
        "UpdateExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> datafusion::error::Result<SendableRecordBatchStream> {
        let schema = count_schema();
        let update_sink = Arc::clone(&self.update_sink);
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&schema),
            futures::stream::once(async move {
                let count = update_sink
                    .execute_update()
                    .await
                    .map_err(DataFusionError::External)?;
                count_to_record_batch(schema, count)
            }),
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::prelude::*;

    #[test]
    fn test_filters_to_sql_single() {
        let filter = col("id").eq(lit(1i32));
        let sql = filters_to_sql(&[filter], None).expect("filters_to_sql should succeed");
        assert_eq!(sql, r#""id" = 1"#);
    }

    #[test]
    fn test_filters_to_sql_multiple() {
        let f1 = col("id").eq(lit(1i32));
        let f2 = col("name").eq(lit("foo"));
        let sql = filters_to_sql(&[f1, f2], None).expect("filters_to_sql should succeed");
        assert_eq!(sql, r#"("id" = 1) AND ("name" = 'foo')"#);
    }

    /// A filter that is itself a disjunction must keep its grouping when joined with the others:
    /// `AND` binds tighter than `OR`, so joining the fragments bare turns `(a OR b) AND c` into
    /// `a OR (b AND c)`, which matches a different set of rows.
    #[test]
    fn test_filters_to_sql_parenthesizes_a_disjunction() {
        let disjunction = col("a").eq(lit(1i32)).or(col("b").eq(lit(1i32)));
        let conjunct = col("c").eq(lit(1i32));

        let sql =
            filters_to_sql(&[disjunction, conjunct], None).expect("filters_to_sql should succeed");

        assert_eq!(sql, r#"(("a" = 1) OR ("b" = 1)) AND ("c" = 1)"#);
    }

    #[test]
    fn test_filters_to_sql_empty() {
        let sql = filters_to_sql(&[], None).expect("filters_to_sql should succeed");
        assert_eq!(sql, "");
    }

    #[test]
    fn test_assignments_to_sql_single() {
        let assignments = vec![("name".to_string(), lit("foo"))];
        let sql =
            assignments_to_sql(&assignments, None).expect("assignments_to_sql should succeed");
        assert_eq!(sql, r#""name" = 'foo'"#);
    }

    #[test]
    fn test_assignments_to_sql_multiple() {
        let assignments = vec![
            ("name".to_string(), lit("foo")),
            ("age".to_string(), lit(30i32)),
        ];
        let sql =
            assignments_to_sql(&assignments, None).expect("assignments_to_sql should succeed");
        assert_eq!(sql, r#""name" = 'foo', "age" = 30"#);
    }
}
