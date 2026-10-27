use crate::sql::db_connection_pool::dbconnection::{get_schema, Error as DbError};
use crate::sql::sql_provider_datafusion::{get_stream, to_execution_error};
use arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::logical_expr::{Expr, ExprSchemable, LogicalPlan};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::sql::sqlparser::ast::{self, VisitMut};
use datafusion::sql::unparser::dialect::Dialect;
use datafusion_federation::sql::{
    ast_analyzer::AstAnalyzer, RemoteTableRef, SQLExecutor, SQLFederationProvider, SQLTableSource,
};
use datafusion_federation::{FederatedTableProviderAdaptor, FederatedTableSource};
use futures::TryStreamExt;
use snafu::ResultExt;
use std::sync::Arc;

use super::mysql_window::MySQLWindowVisitor;
use super::sql_table::MySQLTable;
use datafusion::{
    common::TableReference,
    datasource::TableProvider,
    error::{DataFusionError, Result as DataFusionResult},
    execution::SendableRecordBatchStream,
    physical_plan::stream::RecordBatchStreamAdapter,
};

impl MySQLTable {
    fn create_federated_table_source(
        self: Arc<Self>,
    ) -> DataFusionResult<Arc<dyn FederatedTableSource>> {
        let table_reference = self.base_table.table_reference.clone();
        let schema = Arc::clone(&Arc::clone(&self).base_table.schema());
        let fed_provider = Arc::new(SQLFederationProvider::new(self));
        Ok(Arc::new(SQLTableSource::new_with_schema(
            fed_provider,
            RemoteTableRef::from(table_reference),
            schema,
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

#[allow(clippy::unnecessary_wraps)]
fn mysql_ast_analyzer(ast: ast::Statement) -> Result<ast::Statement, DataFusionError> {
    match ast {
        ast::Statement::Query(query) => {
            let mut new_query = query.clone();

            let mut window_visitor = MySQLWindowVisitor::default();
            let _ = new_query.visit(&mut window_visitor);

            Ok(ast::Statement::Query(new_query))
        }
        _ => Ok(ast),
    }
}

#[async_trait]
impl SQLExecutor for MySQLTable {
    fn name(&self) -> &str {
        self.base_table.name()
    }

    fn compute_context(&self) -> Option<String> {
        self.base_table.compute_context()
    }

    fn dialect(&self) -> Arc<dyn Dialect> {
        self.base_table.dialect()
    }

    fn ast_analyzer(&self) -> Option<AstAnalyzer> {
        let rule = Box::new(mysql_ast_analyzer);
        Some(AstAnalyzer::new(vec![rule]))
    }

    fn can_execute_plan(&self, plan: &LogicalPlan) -> bool {
        self.base_table.can_execute_plan(plan)
            && requires_local_evaluation(plan).is_ok_and(|computed| !computed)
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

// Keep unsupported joins, grouping references and NULL ordering local. MySQL orders
// NULLs first for ascending keys and last for descending keys.
fn requires_local_evaluation(plan: &LogicalPlan) -> DataFusionResult<bool> {
    let mut found = false;
    plan.apply_with_subqueries(|node| {
        match node {
            LogicalPlan::Join(join) => {
                found |= join.join_type == datafusion::logical_expr::JoinType::Full;
            }
            LogicalPlan::Filter(filter) => {
                if let LogicalPlan::Aggregate(aggregate) = filter.input.as_ref() {
                    found |= filter.predicate.column_refs().iter().any(|column| {
                        aggregate.schema.index_of_column(column).is_ok_and(|index| {
                            index < aggregate.group_expr.len()
                                && !matches!(aggregate.group_expr[index], Expr::Column(_))
                        })
                    });
                }
            }
            LogicalPlan::Sort(sort) => {
                found |= sort
                    .expr
                    .iter()
                    .any(|key| null_order_needs_local(key, sort.input.schema()));
            }
            LogicalPlan::Window(window) => {
                for expression in &window.window_expr {
                    expression.apply(|expression| {
                        if let Expr::WindowFunction(function) = expression {
                            found |= function
                                .params
                                .order_by
                                .iter()
                                .any(|key| null_order_needs_local(key, window.input.schema()));
                        }
                        Ok(if found {
                            TreeNodeRecursion::Stop
                        } else {
                            TreeNodeRecursion::Continue
                        })
                    })?;
                    if found {
                        break;
                    }
                }
            }
            _ => {}
        }
        Ok(if found {
            TreeNodeRecursion::Stop
        } else {
            TreeNodeRecursion::Continue
        })
    })?;
    Ok(found)
}

fn null_order_needs_local(
    sort: &datafusion::logical_expr::expr::Sort,
    schema: &datafusion::common::DFSchema,
) -> bool {
    sort.nulls_first != sort.asc && sort.expr.nullable(schema).unwrap_or(true)
}
