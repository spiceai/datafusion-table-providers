use crate::sql::db_connection_pool::dbconnection::{get_schema, Error as DbError};
use crate::sql::sql_provider_datafusion::{get_stream, to_execution_error};
use arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::logical_expr::LogicalPlan;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::sql::sqlparser::ast::{self, VisitMut};
use datafusion::sql::unparser::dialect::Dialect;
use datafusion_federation::sql::{
    ast_analyzer::AstAnalyzer, LogicalOptimizer, MultiPartTableReference, RemoteTable,
    RemoteTableRef, SQLExecutor, SQLFederationProvider, SQLTable, SQLTableSource,
};
use datafusion_federation::{FederatedTableProviderAdaptor, FederatedTableSource};
use futures::TryStreamExt;
use snafu::ResultExt;
use std::sync::Arc;

use super::between::SQLiteBetweenVisitor;
use super::sql_table::SQLiteTable;
use super::sqlite_interval::{
    mark_date_operands, mark_federated_date_operands, SQLiteIntervalVisitor,
};
use datafusion::{
    common::TableReference,
    datasource::TableProvider,
    error::{DataFusionError, Result as DataFusionResult},
    execution::SendableRecordBatchStream,
    physical_plan::stream::RecordBatchStreamAdapter,
};

impl<T, P> SQLiteTable<T, P> {
    fn create_federated_table_source(
        self: Arc<Self>,
    ) -> DataFusionResult<Arc<dyn FederatedTableSource>> {
        let table_reference = self.base_table.table_reference.clone();
        let schema = Arc::clone(&Arc::clone(&self).base_table.schema());
        let fed_provider = Arc::new(SQLFederationProvider::new(self));
        Ok(Arc::new(SQLTableSource::new_with_table(
            fed_provider,
            Arc::new(SQLiteRemoteTable::new(
                RemoteTableRef::from(table_reference),
                schema,
            )),
        )))
    }

    pub fn create_federated_table_provider(
        self: Arc<Self>,
    ) -> DataFusionResult<FederatedTableProviderAdaptor> {
        let table_source = Self::create_federated_table_source(Arc::clone(&self))?;
        Ok(FederatedTableProviderAdaptor::new_with_provider(
            table_source,
            self,
        ))
    }
}

/// The remote table a federated SQLite scan reads: a [`RemoteTable`] whose logical
/// optimizer is [`mark_date_operands`], which runs on the typed plan of each
/// statement sent to SQLite, before it is unparsed, so the interval rewrite in
/// [`sqlite_ast_analyzer`] can tell a `DATE` operand from a timestamp.
#[derive(Debug)]
pub struct SQLiteRemoteTable {
    remote_table: RemoteTable,
}

impl SQLiteRemoteTable {
    pub fn new(table_ref: RemoteTableRef, schema: SchemaRef) -> Self {
        Self {
            remote_table: RemoteTable::new(table_ref, schema),
        }
    }
}

impl SQLTable for SQLiteRemoteTable {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn table_reference(&self) -> MultiPartTableReference {
        SQLTable::table_reference(&self.remote_table)
    }

    fn schema(&self) -> SchemaRef {
        SQLTable::schema(&self.remote_table)
    }

    fn logical_optimizer(&self) -> Option<LogicalOptimizer> {
        Some(Box::new(mark_date_operands))
    }
}

fn sqlite_ast_analyzer(
    decimal_between: bool,
) -> impl Fn(ast::Statement) -> Result<ast::Statement, DataFusionError> {
    move |ast: ast::Statement| -> Result<ast::Statement, DataFusionError> {
        match ast {
            ast::Statement::Query(query) => {
                let mut new_query = query.clone();

                // normalize INTERVAL usage into SQLite-compatible datetime expressions
                let mut interval_visitor = SQLiteIntervalVisitor::default();
                let _ = new_query.visit(&mut interval_visitor);

                // rewrite BETWEEN clauses with numeric operands to decimal comparisons
                if decimal_between {
                    let mut between_visitor = SQLiteBetweenVisitor::default();
                    let _ = new_query.visit(&mut between_visitor);
                }

                Ok(ast::Statement::Query(new_query))
            }
            _ => Ok(ast),
        }
    }
}

#[async_trait]
impl<T, P> SQLExecutor for SQLiteTable<T, P> {
    fn name(&self) -> &str {
        self.base_table.name()
    }

    fn compute_context(&self) -> Option<String> {
        self.base_table.compute_context()
    }

    fn dialect(&self) -> Arc<dyn Dialect> {
        self.base_table.dialect()
    }

    /// Marks the `DATE` operands of interval arithmetic in every federated
    /// statement, including one whose only table is inside a subquery, which the
    /// scan's own optimizer ([`SQLiteRemoteTable`]) is not gathered for.
    fn logical_optimizer(&self) -> Option<LogicalOptimizer> {
        Some(Box::new(mark_federated_date_operands))
    }

    fn ast_analyzer(&self) -> Option<AstAnalyzer> {
        let rule = Box::new(sqlite_ast_analyzer(self.decimal_between));
        Some(AstAnalyzer::new(vec![rule]))
    }

    /// SQLite has no `UNNEST`, so a plan that unnests a column runs it in DataFusion. The
    /// federation then sends SQLite the parts of the plan below the `Unnest` it can run.
    fn can_execute_plan(&self, plan: &LogicalPlan) -> bool {
        !contains_unnest(plan) && self.base_table.can_execute_plan(plan)
    }

    fn execute(
        &self,
        query: &str,
        schema: SchemaRef,
        _filters: &[Arc<dyn PhysicalExpr>],
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let fut = get_stream(
            self.base_table.clone_pool(),
            query.to_string(),
            Arc::clone(&schema),
        );

        let stream = futures::stream::once(fut).try_flatten();
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }

    async fn table_names(&self) -> DataFusionResult<Vec<String>> {
        Err(DataFusionError::NotImplemented(
            "table inference not implemented".to_string(),
        ))
    }

    async fn get_table_schema(&self, table_name: &str) -> DataFusionResult<SchemaRef> {
        let conn = self
            .base_table
            .clone_pool()
            .connect()
            .await
            .map_err(to_execution_error)?;
        get_schema(conn, &TableReference::from(table_name))
            .await
            .boxed()
            .map_err(|e| DbError::UnableToGetSchema { source: e })
            .map_err(to_execution_error)
    }
}

/// Whether `plan`, or a subquery inside one of its expressions, has an `Unnest`. A plan that
/// cannot be walked counts as having one, so it is never sent to SQLite.
fn contains_unnest(plan: &LogicalPlan) -> bool {
    let mut found = false;
    let walked = plan.apply_with_subqueries(|node| {
        found = matches!(node, LogicalPlan::Unnest(_));
        Ok(if found {
            TreeNodeRecursion::Stop
        } else {
            TreeNodeRecursion::Continue
        })
    });
    found || walked.is_err()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Int64Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::common::TableReference;
    use datafusion::execution::context::SessionContext;

    use crate::sql::arrow_sql_gen::statement::{CreateTableBuilder, InsertBuilder};
    use crate::sql::db_connection_pool::sqlitepool::SqliteConnectionPoolFactory;
    use crate::sql::db_connection_pool::{DbConnectionPool, Mode};
    use crate::sqlite::sql_table::SQLiteTable;
    use crate::sqlite::DynSqliteConnectionPool;

    /// A federated session over an in-memory SQLite table holding `batch`.
    async fn federated_session(table: &str, batch: RecordBatch) -> SessionContext {
        let schema = batch.schema();
        let pool = SqliteConnectionPoolFactory::new(
            ":memory:",
            Mode::Memory,
            std::time::Duration::from_millis(5000),
        )
        .build()
        .await
        .expect("pool");
        let conn = pool.connect().await.expect("connection");
        let conn = conn.as_async().expect("async connection");
        conn.execute(
            &CreateTableBuilder::new(Arc::clone(&schema), table).build_sqlite(),
            &[],
        )
        .await
        .expect("table created");
        conn.execute(
            &InsertBuilder::new(&TableReference::from(table), &vec![batch])
                .build_sqlite(None)
                .expect("insert statement"),
            &[],
        )
        .await
        .expect("rows inserted");
        let pool: Arc<DynSqliteConnectionPool> = Arc::new(pool);
        let provider = Arc::new(SQLiteTable::new_with_schema(&pool, schema, table, None))
            .create_federated_table_provider()
            .expect("federated provider");
        let ctx = SessionContext::new_with_state(datafusion_federation::default_session_state());
        ctx.register_table(table, Arc::new(provider))
            .expect("table registered");
        ctx
    }

    /// An error SQLite raises while running a federated statement fails the query
    /// instead of answering with the rows produced before it: an ungrouped `SUM`
    /// that overflows `i64` must not come back as an empty result.
    #[tokio::test]
    async fn an_error_sqlite_raises_mid_query_fails_the_federated_query() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("v", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(Int64Array::from(vec![9_000_000_000_000_000_000_i64; 3])),
            ],
        )
        .expect("batch");
        let ctx = federated_session("ovf", batch).await;
        let df = ctx
            .sql("SELECT sum(v) AS s FROM ovf")
            .await
            .expect("the query plans");
        let plan = df
            .clone()
            .create_physical_plan()
            .await
            .expect("physical plan");
        let display = datafusion::physical_plan::displayable(plan.as_ref())
            .indent(true)
            .to_string();
        assert!(
            display.contains("VirtualExecutionPlan") && display.contains("sum("),
            "the SUM must be pushed down to SQLite for the overflow to be its error:
{display}"
        );
        let err = df
            .collect()
            .await
            .expect_err("SQLite's integer overflow must fail the query");
        assert!(
            err.to_string()
                .contains("Failed to read the next row: integer overflow"),
            "unexpected error: {err}"
        );
    }

    /// Four rows, `id` 1 to 4, in a federated table called `table`.
    async fn four_rows(table: &str) -> SessionContext {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![1, 2, 3, 4]))])
                .expect("batch");
        federated_session(table, batch).await
    }

    async fn count(ctx: &SessionContext, sql: &str) -> datafusion::error::Result<i64> {
        let batches = ctx.sql(sql).await?.collect().await?;
        Ok(batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("count is an Int64")
            .value(0))
    }

    /// SQLite has no `UNNEST`, so a join against one must not be sent to it: unparsed, an
    /// `UNNEST` whose columns the query never reads became `(SELECT 1)`, and the join returned
    /// one row for each of `t`'s instead of one for each pair.
    #[tokio::test]
    async fn a_cross_join_with_an_unnest_returns_every_pair() {
        let ctx = four_rows("pairs").await;
        for sql in [
            "SELECT count(*) FROM pairs a CROSS JOIN (SELECT * FROM unnest(range(0, 100))) b",
            "SELECT count(*) FROM (SELECT a.* FROM pairs a CROSS JOIN unnest(range(0, 100)) b)",
            "SELECT count(*) FROM pairs a CROSS JOIN (SELECT unnest(make_array(1, 2, 3)) AS x) b",
        ] {
            let expected = if sql.contains("make_array") { 12 } else { 400 };
            assert_eq!(count(&ctx, sql).await.expect(sql), expected, "{sql}");
        }
    }

    /// A join that reads the unnested column returns its rows rather than failing with
    /// SQLite's `no such column`.
    #[tokio::test]
    async fn a_join_on_an_unnested_column_returns_its_rows() {
        let ctx = four_rows("matches").await;
        let sql = "SELECT count(*) FROM matches a JOIN (SELECT unnest(make_array(1, 2, 3)) AS x) b ON a.id = b.x";
        assert_eq!(count(&ctx, sql).await.expect(sql), 3);
    }

    /// The table's own scan is still sent to SQLite under a local `UNNEST`.
    #[tokio::test]
    async fn the_scan_under_an_unnest_join_is_still_federated() {
        let ctx = four_rows("scanned").await;
        let plan = ctx
            .sql("SELECT count(*) FROM scanned a CROSS JOIN (SELECT * FROM unnest(range(0, 3))) b")
            .await
            .expect("the query plans")
            .create_physical_plan()
            .await
            .expect("physical plan");
        let display = datafusion::physical_plan::displayable(plan.as_ref())
            .indent(true)
            .to_string();
        assert!(display.contains("VirtualExecutionPlan"), "{display}");
        assert!(display.contains("UnnestExec"), "{display}");
    }
}
