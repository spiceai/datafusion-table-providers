use std::{fmt, sync::Arc};

use arrow::datatypes::Schema;

use async_trait::async_trait;
use datafusion::{
    common::TableReference,
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
    filters_to_sql_with_schema(filters, engine, None)
}

/// [`filters_to_sql`], given the schema the filters' columns belong to.
///
/// The schema is what lets the DuckDB rendering tell a column that needs its timestamp
/// normalization from one the normalization would only truncate — see
/// [`expr::to_sql_with_engine_and_schema`].
pub fn filters_to_sql_with_schema(
    filters: &[Expr],
    engine: Option<expr::Engine>,
    schema: Option<&Schema>,
) -> datafusion::error::Result<String> {
    let sql_parts: Result<Vec<String>, _> = filters
        .iter()
        .map(|f| expr::to_sql_with_engine_and_schema(f, engine, schema))
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
    assignments_to_sql_with_schema(assignments, engine, None)
}

/// [`assignments_to_sql`], given the schema the assignments' columns belong to.
pub fn assignments_to_sql_with_schema(
    assignments: &[(String, Expr)],
    engine: Option<expr::Engine>,
    schema: Option<&Schema>,
) -> datafusion::error::Result<String> {
    let parts: Result<Vec<String>, _> = assignments
        .iter()
        .map(|(col, val)| {
            expr::to_sql_with_engine_and_schema(val, engine, schema)
                .map(|sql_val| format!("{col} = {sql_val}", col = expr::quoted_identifier(col)))
        })
        .collect();
    parts
        .map(|p| p.join(", "))
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))
}

/// Renders `DELETE FROM <table> [WHERE <sql_where>]`, quoting each part of `table` so that a quote
/// in it cannot close the identifier and leave the rest of the name to be read as statement text.
///
/// Doubling is the whole escape here, unlike in a string literal: none of `DuckDB`, `SQLite` and
/// `PostgreSQL` give `\` any meaning inside a delimited identifier.
///
/// The statement addresses the reference in full rather than its bare table part, so that a
/// qualified dataset's `DELETE` reaches the table its truncate path already reaches instead of
/// whatever the session's `search_path` resolves the bare name to.
pub fn delete_statement(table: &TableReference, sql_where: Option<&str>) -> String {
    let table = expr::quoted_table_reference(table);
    match sql_where {
        Some(sql_where) => format!("DELETE FROM {table} WHERE {sql_where}"),
        None => format!("DELETE FROM {table}"),
    }
}

/// Renders [`delete_statement`] wrapped in a CTE that returns the removed rows, so the count comes
/// back as a result row: `WITH deleted AS (DELETE ... RETURNING *) SELECT COUNT(*) FROM deleted`.
///
/// This is for a caller that reads the count from a query rather than from the driver's
/// affected-row tag — the `PostgreSQL` deletion sink runs this with `query_one` inside its
/// transaction, so the count and the delete stay one statement.
pub fn delete_statement_returning_count(table: &TableReference, sql_where: Option<&str>) -> String {
    let delete = delete_statement(table, sql_where);
    format!("WITH deleted AS ({delete} RETURNING *) SELECT COUNT(*) FROM deleted")
}

/// Renders `UPDATE <table> SET <set_clause> [WHERE <sql_where>]`, naming the table as
/// [`delete_statement`] describes.
pub fn update_statement(
    table: &TableReference,
    set_clause: &str,
    sql_where: Option<&str>,
) -> String {
    let table = expr::quoted_table_reference(table);
    match sql_where {
        Some(sql_where) => format!("UPDATE {table} SET {set_clause} WHERE {sql_where}"),
        None => format!("UPDATE {table} SET {set_clause}"),
    }
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

    // Holds no `PhysicalExpr`, so there is nothing to visit.
    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(
            &Arc<dyn datafusion::physical_plan::PhysicalExpr>,
        ) -> datafusion::error::Result<
            datafusion::common::tree_node::TreeNodeRecursion,
        >,
    ) -> datafusion::error::Result<datafusion::common::tree_node::TreeNodeRecursion> {
        Ok(datafusion::common::tree_node::TreeNodeRecursion::Continue)
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

    // Holds no `PhysicalExpr`, so there is nothing to visit.
    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(
            &Arc<dyn datafusion::physical_plan::PhysicalExpr>,
        ) -> datafusion::error::Result<
            datafusion::common::tree_node::TreeNodeRecursion,
        >,
    ) -> datafusion::error::Result<datafusion::common::tree_node::TreeNodeRecursion> {
        Ok(datafusion::common::tree_node::TreeNodeRecursion::Continue)
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

    /// A predicate reaching a DML sink can carry a qualifier, and it must render as the column it
    /// names rather than as one flat identifier.
    #[test]
    fn test_filters_to_sql_renders_a_qualified_column_as_its_own_column() {
        let filter = Expr::Column(datafusion::common::Column::new(Some("t"), "id")).eq(lit(1i32));

        let sql = filters_to_sql(&[filter], None).expect("filters_to_sql should succeed");
        assert_eq!(sql, r#""id" = 1"#);
    }

    /// The same applies to a SET clause, whose value expression can carry a column reference.
    #[test]
    fn test_assignments_to_sql_renders_a_qualified_column_value() {
        let assignments = vec![(
            "name".to_string(),
            Expr::Column(datafusion::common::Column::new(Some("t"), "other")),
        )];

        let sql =
            assignments_to_sql(&assignments, None).expect("assignments_to_sql should succeed");
        assert_eq!(sql, r#""name" = "other""#);
    }

    /// A name needing no escape must render exactly as the interpolated form did, so that quoting
    /// the table name cannot change the statement built for any ordinary dataset.
    #[test]
    fn test_statements_render_an_ordinary_table_name_unchanged() {
        assert_eq!(
            delete_statement(&TableReference::bare("orders"), Some(r#""id" = 1"#)),
            r#"DELETE FROM "orders" WHERE "id" = 1"#
        );
        assert_eq!(
            delete_statement(&TableReference::bare("orders"), None),
            r#"DELETE FROM "orders""#
        );
        assert_eq!(
            update_statement(
                &TableReference::bare("orders"),
                r#""qty" = 2"#,
                Some(r#""id" = 1"#)
            ),
            r#"UPDATE "orders" SET "qty" = 2 WHERE "id" = 1"#
        );
        assert_eq!(
            update_statement(&TableReference::bare("orders"), r#""qty" = 2"#, None),
            r#"UPDATE "orders" SET "qty" = 2"#
        );
        assert_eq!(
            delete_statement_returning_count(&TableReference::bare("orders"), Some(r#""id" = 1"#)),
            r#"WITH deleted AS (DELETE FROM "orders" WHERE "id" = 1 RETURNING *) SELECT COUNT(*) FROM deleted"#
        );
        assert_eq!(
            delete_statement_returning_count(&TableReference::bare("orders"), None),
            r#"WITH deleted AS (DELETE FROM "orders" RETURNING *) SELECT COUNT(*) FROM deleted"#
        );
    }

    /// A quote in the table name closes its identifier early when interpolated, so the rest of the
    /// name becomes statement text — the identifier counterpart of an unescaped string literal.
    #[test]
    fn test_statements_escape_a_quote_bearing_table_name() {
        assert_eq!(
            delete_statement(&TableReference::bare(r#"we"ird"#), Some(r#""id" = 1"#)),
            r#"DELETE FROM "we""ird" WHERE "id" = 1"#
        );
        assert_eq!(
            update_statement(&TableReference::bare(r#"we"ird"#), r#""qty" = 2"#, None),
            r#"UPDATE "we""ird" SET "qty" = 2"#
        );
        assert_eq!(
            delete_statement_returning_count(&TableReference::bare(r#"we"ird"#), None),
            r#"WITH deleted AS (DELETE FROM "we""ird" RETURNING *) SELECT COUNT(*) FROM deleted"#
        );

        // A quote at either end has no ordinary character beside it to make the doubling obvious.
        assert_eq!(
            delete_statement(&TableReference::bare(r#"trailing""#), None),
            r#"DELETE FROM "trailing""""#
        );
        assert_eq!(
            delete_statement(&TableReference::bare(r#""leading"#), None),
            r#"DELETE FROM """leading""#
        );
    }

    /// A name carrying SQL of its own is what the missing escape actually costs: interpolated, the
    /// `WHERE` below escapes the identifier and widens the statement to the whole table. Quoted, the
    /// name stays one identifier and the statement can only ever name that table.
    #[test]
    fn test_a_table_name_carrying_sql_stays_one_identifier() {
        assert_eq!(
            delete_statement(
                &TableReference::bare(r#"t" WHERE 1=1 --"#),
                Some(r#""id" = 1"#)
            ),
            r#"DELETE FROM "t"" WHERE 1=1 --" WHERE "id" = 1"#
        );
    }

    /// Doubling a delimiter is only a complete escape where nothing else escapes: a backslash next
    /// to the quote is the composition that defeats it for a *string literal* on a backslash-reading
    /// engine. Inside a delimited identifier none of the engines here give `\` any meaning, so it
    /// must pass through untouched — asserted so that adding an engine which does has to face it.
    #[test]
    fn test_a_backslash_in_a_table_name_is_not_escaped() {
        assert_eq!(
            delete_statement(&TableReference::bare(r#"back\slash"quote"#), None),
            r#"DELETE FROM "back\slash""quote""#
        );
    }

    /// A qualified dataset must be addressed by the name it was given. Naming only the bare table
    /// part leaves the schema to the session's `search_path`, which is how a `DELETE`/`UPDATE`
    /// came to address a different schema's table than the truncate path on the same provider.
    #[test]
    fn test_statements_address_a_qualified_table_in_full() {
        let qualified = TableReference::partial("myschema", "orders");

        assert_eq!(
            delete_statement(&qualified, Some(r#""id" = 1"#)),
            r#"DELETE FROM "myschema"."orders" WHERE "id" = 1"#
        );
        assert_eq!(
            update_statement(&qualified, r#""qty" = 2"#, None),
            r#"UPDATE "myschema"."orders" SET "qty" = 2"#
        );
        assert_eq!(
            delete_statement_returning_count(&qualified, None),
            r#"WITH deleted AS (DELETE FROM "myschema"."orders" RETURNING *) SELECT COUNT(*) FROM deleted"#
        );

        let full = TableReference::full("mycatalog", "myschema", "orders");
        assert_eq!(
            delete_statement(&full, None),
            r#"DELETE FROM "mycatalog"."myschema"."orders""#
        );
    }

    /// Each part is a separate identifier, so a quote in one may not leak into another and a dot
    /// inside a part may not split it.
    #[test]
    fn test_a_qualified_name_escapes_each_part_separately() {
        assert_eq!(
            delete_statement(&TableReference::partial(r#"we"ird"#, r#"ta"ble"#), None),
            r#"DELETE FROM "we""ird"."ta""ble""#
        );

        // A dot inside a part names a table whose name contains a dot, not two identifiers.
        assert_eq!(
            delete_statement(&TableReference::bare("has.dot"), None),
            r#"DELETE FROM "has.dot""#
        );
    }

    /// Every part is quoted unconditionally, so a table named for a reserved word stays a name.
    /// `TableReference::to_quoted_string` leaves an all-lowercase part bare, which would render
    /// this as syntax.
    #[test]
    fn test_a_reserved_word_table_name_is_still_quoted() {
        assert_eq!(
            delete_statement(&TableReference::partial("public", "order"), None),
            r#"DELETE FROM "public"."order""#
        );
        assert_eq!(
            TableReference::partial("public", "order").to_quoted_string(),
            "public.order",
            "the assertion above is only load-bearing while to_quoted_string leaves these bare"
        );
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
