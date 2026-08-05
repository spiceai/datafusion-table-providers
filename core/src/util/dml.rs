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
        .map(|parts| parts.join(" AND "))
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
            expr::to_sql_with_engine(val, engine)
                .map(|sql_val| format!("{col} = {sql_val}", col = expr::quoted_identifier(col)))
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
        assert_eq!(sql, r#""id" = 1 AND "name" = 'foo'"#);
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

    /// The WHERE clause a `DELETE`/`UPDATE` is built from must keep a quote-bearing value
    /// inside its literal: rendered bare, `= 'x' OR 1=1 --'` is a tautology and the statement
    /// matches every row.
    #[test]
    fn test_filters_to_sql_escapes_a_quote_bearing_value() {
        let filter = col("name").eq(lit("x' OR 1=1 --"));
        let sql = filters_to_sql(&[filter], None).expect("filters_to_sql should succeed");
        assert_eq!(sql, r#""name" = 'x'' OR 1=1 --'"#);

        let apostrophe = col("name").eq(lit("O'Brien"));
        let sql = filters_to_sql(&[apostrophe], None).expect("filters_to_sql should succeed");
        assert_eq!(sql, r#""name" = 'O''Brien'"#);
    }

    /// A SET clause carries both a value and a column name, and interpolates the name itself.
    #[test]
    fn test_assignments_to_sql_escapes_value_and_column_name() {
        let assignments = vec![("name".to_string(), lit("O'Brien"))];
        let sql =
            assignments_to_sql(&assignments, None).expect("assignments_to_sql should succeed");
        assert_eq!(sql, r#""name" = 'O''Brien'"#);

        let quoted_column = vec![("we\"ird".to_string(), lit(1i32))];
        let sql =
            assignments_to_sql(&quoted_column, None).expect("assignments_to_sql should succeed");
        assert_eq!(sql, r#""we""ird" = 1"#);
    }
}

/// The rendered SQL is only correct if the target engine agrees, so these run the statement the
/// DML paths build and assert on the rows that actually survive it. A rendering assertion alone
/// cannot distinguish "escaped" from "escaped in a form this engine accepts".
#[cfg(all(test, feature = "duckdb"))]
mod duckdb_execution_tests {
    use super::*;
    use crate::sql::sql_provider_datafusion::expr::Engine;
    use datafusion::prelude::*;
    use duckdb::Connection;

    const ROWS: [&str; 4] = ["O'Brien", "plain", "x' OR 1=1 --", r"\' OR 1=1 --"];

    /// Deletes with the WHERE clause `filters_to_sql` renders, and returns the surviving names.
    fn surviving_after_delete(filters: &[Expr]) -> Vec<String> {
        let conn = Connection::open_in_memory().expect("in-memory DuckDB");
        conn.execute_batch("CREATE TABLE t (name VARCHAR)")
            .expect("create");
        for row in ROWS {
            conn.execute("INSERT INTO t VALUES (?)", [row])
                .expect("insert");
        }

        let where_clause =
            filters_to_sql(filters, Some(Engine::DuckDB)).expect("filters_to_sql should succeed");
        conn.execute_batch(&format!("DELETE FROM t WHERE {where_clause}"))
            .expect("the rendered DELETE must be valid SQL for DuckDB");

        let mut stmt = conn
            .prepare("SELECT name FROM t ORDER BY name")
            .expect("prepare");
        let mut names: Vec<String> = stmt
            .query_map([], |row| row.get(0))
            .expect("query")
            .collect::<duckdb::Result<Vec<String>>>()
            .expect("rows");
        names.sort();
        names
    }

    fn all_but(excluded: &str) -> Vec<String> {
        let mut rest: Vec<String> = ROWS
            .iter()
            .filter(|row| **row != excluded)
            .map(|row| (*row).to_string())
            .collect();
        rest.sort();
        rest
    }

    /// An apostrophe rendered bare makes the statement fail to parse, so the delete cannot run.
    #[test]
    fn an_apostrophe_bearing_value_deletes_exactly_its_row() {
        let filters = vec![col("name").eq(lit("O'Brien"))];

        assert_eq!(surviving_after_delete(&filters), all_but("O'Brien"));
    }

    /// The unbounded-deletion shape: rendered bare this binds as a tautology and removes every
    /// row. It must remove exactly the one row whose value it is.
    #[test]
    fn a_tautology_shaped_value_deletes_exactly_its_row() {
        let filters = vec![col("name").eq(lit("x' OR 1=1 --"))];

        assert_eq!(surviving_after_delete(&filters), all_but("x' OR 1=1 --"));
    }

    /// A backslash immediately before the quote is the composition quote doubling cannot handle
    /// on a backslash-escaping engine. DuckDB treats `\` as ordinary, so it must stay contained.
    #[test]
    fn a_backslash_before_a_quote_deletes_exactly_its_row() {
        let filters = vec![col("name").eq(lit(r"\' OR 1=1 --"))];

        assert_eq!(surviving_after_delete(&filters), all_but(r"\' OR 1=1 --"));
    }

    /// Two filters are joined with `AND`, so a quote in either must stay inside its own literal.
    #[test]
    fn a_conjunction_of_quote_bearing_values_matches_nothing() {
        let filters = vec![
            col("name").eq(lit("O'Brien")),
            col("name").eq(lit("x' OR 1=1 --")),
        ];

        let mut expected: Vec<String> = ROWS.iter().map(|row| (*row).to_string()).collect();
        expected.sort();
        assert_eq!(surviving_after_delete(&filters), expected);
    }
}
