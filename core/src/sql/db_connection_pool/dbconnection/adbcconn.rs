// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use adbc_core::Statement;
use adbc_core::{Connection, Database};
use async_stream::stream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use std::any::Any;
use std::cell::RefCell;

use adbc_core::options::ObjectDepth;
use arrow::array::{AsArray, RecordBatch, RecordBatchIterator, RecordBatchReader};
use arrow_schema::SchemaRef;
use datafusion::error::DataFusionError;
use datafusion::execution::SendableRecordBatchStream;
use datafusion::sql::TableReference;
use r2d2_adbc::AdbcConnectionManager;
use snafu::{prelude::*, ResultExt};
use std::marker::Send;
use std::marker::Sync;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc::Sender;

use crate::sql::db_connection_pool::runtime::run_sync_with_tokio;

use super::DbConnection;
use super::Result;
use super::SyncDbConnection;

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("ADBC Error: {source}"))]
    AdbcError { source: adbc_core::error::Error },

    #[snafu(display(
        "An unexpected error occurred.\n{message}\nVerify the configuration and try again"
    ))]
    ChannelError { message: String },
}

pub struct AdbcDbConnection<D>
where
    D: Database + Send + 'static,
    D::ConnectionType: Send + Sync,
    <D::ConnectionType as Connection>::StatementType: Clone + Send + Unpin + 'static,
{
    pub conn: Arc<Mutex<RefCell<r2d2::PooledConnection<AdbcConnectionManager<D>>>>>,
}

impl<D> DbConnection<r2d2::PooledConnection<AdbcConnectionManager<D>>, RecordBatch>
    for AdbcDbConnection<D>
where
    D: Database + Send + 'static,
    D::ConnectionType: Send + Sync,
    <D::ConnectionType as Connection>::StatementType: Clone + Send + Unpin + 'static,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn as_sync(
        &self,
    ) -> Option<&dyn SyncDbConnection<r2d2::PooledConnection<AdbcConnectionManager<D>>, RecordBatch>>
    {
        Some(self)
    }
}

/// Shared between the stream handed to the caller and the thread running the
/// query, so a caller that goes away can stop a query that has already started.
///
/// The thread inside [`Statement::execute`] holds the only `&mut` to the
/// statement for as long as the query runs, so cancelling needs a second handle
/// to the same statement — which is why the statement type has to be `Clone`.
enum QueryCancellation<S> {
    /// The statement has not been created yet.
    Preparing,
    /// The query is running and can be cancelled through this handle.
    Running(S),
    /// The caller went away. If the query has not started, it must not start.
    Abandoned,
    /// The query ended on its own; there is nothing to cancel.
    Finished,
}

/// A record-batch stream that cancels its query when it is dropped.
///
/// Dropping the stream is how a caller that has gone away — a client deadline,
/// a disconnect, a cancelled plan — reaches this layer. Without this, the
/// blocking thread stays inside `Statement::execute` until the remote query
/// finishes on its own, holding its pooled connection for that whole time, and
/// the remote database keeps doing the work nobody is waiting for.
struct CancelOnDrop<S: Statement> {
    inner: SendableRecordBatchStream,
    cancellation: Arc<Mutex<QueryCancellation<S>>>,
}

impl<S> futures::Stream for CancelOnDrop<S>
where
    S: Statement + Send + Unpin,
{
    type Item = datafusion::common::Result<RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        std::pin::Pin::new(&mut self.inner).poll_next(cx)
    }
}

impl<S> datafusion::execution::RecordBatchStream for CancelOnDrop<S>
where
    S: Statement + Send + Unpin,
{
    fn schema(&self) -> SchemaRef {
        self.inner.schema()
    }
}

impl<S: Statement> Drop for CancelOnDrop<S> {
    fn drop(&mut self) {
        let mut cancellation = match self.cancellation.lock() {
            Ok(cancellation) => cancellation,
            Err(poisoned) => poisoned.into_inner(),
        };
        match std::mem::replace(&mut *cancellation, QueryCancellation::Abandoned) {
            QueryCancellation::Running(mut statement) => {
                // `AdbcStatementCancel` is the one statement call a driver must
                // accept while another is in flight, and it only signals; it does
                // not wait for the query to unwind.
                if let Err(error) = statement.cancel() {
                    tracing::debug!("Failed to cancel abandoned ADBC query: {error}");
                }
            }
            QueryCancellation::Finished => {
                *cancellation = QueryCancellation::Finished;
            }
            QueryCancellation::Preparing | QueryCancellation::Abandoned => {}
        }
    }
}

fn lock_cancellation<S>(
    cancellation: &Arc<Mutex<QueryCancellation<S>>>,
) -> std::sync::MutexGuard<'_, QueryCancellation<S>> {
    match cancellation.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn blocking_channel_send<T>(channel: &Sender<T>, item: T) -> Result<()> {
    match channel.blocking_send(item) {
        Ok(()) => Ok(()),
        Err(e) => Err(Error::ChannelError {
            message: format!("{e}"),
        }
        .into()),
    }
}

impl<D> SyncDbConnection<r2d2::PooledConnection<AdbcConnectionManager<D>>, RecordBatch>
    for AdbcDbConnection<D>
where
    D: Database + Send + 'static,
    D::ConnectionType: Send + Sync,
    <D::ConnectionType as Connection>::StatementType: Clone + Send + Unpin + 'static,
    <D::ConnectionType as Connection>::StatementType: Clone + Send + Unpin + 'static,
{
    fn new(conn: r2d2::PooledConnection<AdbcConnectionManager<D>>) -> Self {
        AdbcDbConnection {
            conn: Arc::new(Mutex::new(RefCell::new(conn))),
        }
    }

    fn tables(&self, schema: &str) -> Result<Vec<String>, super::Error> {
        let conn_mx = self.conn.lock().unwrap();
        let conn = conn_mx.borrow();
        let result = conn
            .get_objects(ObjectDepth::Tables, None, Some(schema), None, None, None)
            .boxed()
            .context(super::UnableToGetTablesSnafu)?;

        let mut tables = vec![];
        for batch in result {
            // Process each batch to extract table names
            //
            // Schema is as follows:
            // 0: CATALOG_NAME
            // 1: list<DB_SCHEMA_SCHEMA>
            //
            // DB_SCHEMA_SCHEMA is as follows:
            // 0: SCHEMA_NAME
            // 1: list<TABLE_INFO>
            //
            // TABLE_INFO is as follows:
            // 0: TABLE_NAME
            // 1: TABLE_TYPE
            // 2: list<COLUMN_SCHEMA>
            // 3: list<CONSTRAINT_SCHEMA>
            //
            // so we need to drill down to the table names
            let b = batch.boxed().context(super::UnableToGetTablesSnafu)?;
            b.column(1).as_list::<i32>().iter().for_each(|value| {
                if let Some(db_schema_schema) = value {
                    db_schema_schema
                        .as_struct()
                        .column(1)
                        .as_list::<i32>()
                        .iter()
                        .for_each(|table_info| {
                            if let Some(table_struct) = table_info {
                                tables.extend(
                                    table_struct
                                        .as_struct()
                                        .column(0)
                                        .as_string::<i32>()
                                        .iter()
                                        .flatten()
                                        .map(|name| name.to_string()),
                                );
                            }
                        })
                }
            })
        }

        Ok(tables)
    }

    fn schemas(&self) -> Result<Vec<String>, super::Error> {
        let conn_mx = self.conn.lock().unwrap();
        let conn = conn_mx.borrow();

        let result = conn
            .get_objects(ObjectDepth::Schemas, None, None, None, None, None)
            .boxed()
            .context(super::UnableToGetSchemaSnafu)?;

        let mut schemas = vec![];
        for batch in result {
            // Process each batch to extract schema names
            //
            // Schema is as follows:
            // 0: CATALOG_NAME
            // 1: list<DB_SCHEMA_SCHEMA>
            //
            // DB_SCHEMA_SCHEMA is as follows:
            // 0: SCHEMA_NAME
            // 1: list<TABLE_INFO>
            //
            // so we need to drill down to the schema names
            let b = batch.boxed().context(super::UnableToGetSchemaSnafu)?;
            b.column(1).as_list::<i32>().iter().for_each(|value| {
                if let Some(db_schema_schema) = value {
                    db_schema_schema
                        .as_struct()
                        .column(0)
                        .as_string::<i32>()
                        .iter()
                        .flatten()
                        .for_each(|name| schemas.push(name.to_string()));
                }
            });
        }
        Ok(schemas)
    }

    fn get_schema(&self, table_reference: &TableReference) -> Result<SchemaRef, super::Error> {
        let conn_mx = self.conn.lock().unwrap();
        let conn = conn_mx.borrow();

        let schema = conn
            .get_table_schema(
                table_reference.catalog(),
                table_reference.schema(),
                table_reference.table(),
            )
            .boxed()
            .context(super::UnableToGetSchemaSnafu)?;

        Ok(Arc::new(schema))
    }

    fn query_arrow(
        &self,
        sql: &str,
        params: &[RecordBatch],
        _projected_schema: Option<SchemaRef>,
    ) -> Result<SendableRecordBatchStream> {
        let (batch_tx, mut batch_rx) = tokio::sync::mpsc::channel::<RecordBatch>(4);

        let create_stream = || -> Result<SendableRecordBatchStream> {
            let schema: SchemaRef;
            {
                let conn_mx = self.conn.lock().unwrap();
                let mut conn = conn_mx.borrow_mut();
                let mut stmt = conn
                    .new_statement()
                    .boxed()
                    .context(super::UnableToQueryArrowSnafu)?;
                stmt.set_sql_query(sql)?;

                match stmt.execute_schema() {
                    Ok(s) => schema = s.into(),
                    // not all drivers implement execute_schema, so fall back to executing
                    // with LIMIT 0 to get the schema.
                    Err(_) => {
                        stmt.set_sql_query(format!(
                            "WITH fetch_schema AS ({sql}) SELECT * FROM fetch_schema LIMIT 0"
                        ))?;
                        let result = stmt
                            .execute()
                            .boxed()
                            .context(super::UnableToQueryArrowSnafu)?;
                        schema = result.schema();
                    }
                }
            }

            let cloned_conn = Arc::clone(&self.conn);

            let sql_owned = sql.to_string();
            let params_owned = params.to_vec();

            let cancellation = Arc::new(Mutex::new(QueryCancellation::Preparing));
            let task_cancellation = Arc::clone(&cancellation);

            let join_handle = tokio::task::spawn_blocking(move || {
                let conn_mx = cloned_conn.lock().unwrap();
                let mut conn = conn_mx.borrow_mut();
                let mut stmt = conn
                    .new_statement()
                    .boxed()
                    .context(super::UnableToQueryArrowSnafu)?;
                stmt.set_sql_query(&sql_owned)?;

                match params_owned.len() {
                    0 => {}
                    1 => stmt.bind(params_owned[0].clone())?,
                    _ => {
                        let param_schema = params_owned[0].schema();
                        let reader = RecordBatchIterator::new(
                            params_owned.into_iter().map(Ok),
                            param_schema,
                        );

                        stmt.bind_stream(Box::new(reader))?;
                    }
                }

                {
                    let mut state = lock_cancellation(&task_cancellation);
                    if matches!(*state, QueryCancellation::Abandoned) {
                        // The caller went away while the statement was being
                        // prepared. Starting the query now would run it for
                        // nobody, and hold this pooled connection while it did.
                        return Ok(());
                    }
                    *state = QueryCancellation::Running(stmt.clone());
                }

                let results = stmt
                    .execute()
                    .boxed()
                    .context(super::UnableToQueryArrowSnafu)?;
                for batch in results {
                    let b = batch.boxed().context(super::UnableToQueryArrowSnafu)?;
                    blocking_channel_send(&batch_tx, b)?;
                }
                *lock_cancellation(&task_cancellation) = QueryCancellation::Finished;
                Ok::<_, Box<dyn std::error::Error + Send + Sync>>(())
            });

            let output_stream = stream! {
                while let Some(batch) = batch_rx.recv().await {
                    yield Ok(batch);
                }

                match join_handle.await {
                    Ok(Ok(())) => {},
                    Ok(Err(task_error)) => {
                        yield Err(DataFusionError::Execution(format!(
                            "Failed to execute ADBC query: {task_error}"
                        )))
                    },
                    Err(join_error) => {
                        yield Err(DataFusionError::Execution(format!(
                            "Failed to execute ADBC query: {join_error}"
                        )))
                    },
                }
            };

            Ok(Box::pin(CancelOnDrop {
                inner: Box::pin(RecordBatchStreamAdapter::new(schema, output_stream)),
                cancellation,
            }))
        };

        run_sync_with_tokio(create_stream)
    }

    fn execute(&self, sql: &str, params: &[RecordBatch]) -> Result<u64> {
        let conn_mx = self.conn.lock().unwrap();
        let mut conn = conn_mx.borrow_mut();
        let mut stmt = conn.new_statement().context(AdbcSnafu)?;
        stmt.set_sql_query(sql)?;

        let params_owned = params.to_vec();
        match params.len() {
            0 => {}
            1 => stmt.bind(params_owned[0].clone())?,
            _ => {
                let param_schema = params_owned[0].schema();
                let reader =
                    RecordBatchIterator::new(params_owned.into_iter().map(Ok), param_schema);

                stmt.bind_stream(Box::new(reader))?;
            }
        }

        let count: Option<i64> = stmt.execute_update().context(AdbcSnafu)?;

        Ok(count.unwrap_or(-1) as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adbc_core::error::{Error as AdbcError, Result as AdbcResult, Status};
    use adbc_core::options::{
        InfoCode, ObjectDepth, OptionConnection, OptionDatabase, OptionStatement, OptionValue,
    };
    use adbc_core::{Optionable, PartitionedResult};
    use arrow_schema::Schema;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Condvar;
    use std::time::{Duration, Instant};

    /// What the fake driver did, so a test can assert on it rather than on
    /// timing alone.
    #[derive(Default)]
    struct DriverActivity {
        executing: Mutex<bool>,
        started: Condvar,
        cancels: AtomicUsize,
        executes: AtomicUsize,
        connections_open: AtomicUsize,
    }

    impl DriverActivity {
        fn wait_until_executing(&self, timeout: Duration) -> bool {
            let deadline = Instant::now() + timeout;
            let mut executing = self.executing.lock().unwrap_or_else(|e| e.into_inner());
            while !*executing {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    return false;
                }
                let (guard, _) = self
                    .started
                    .wait_timeout(executing, remaining)
                    .unwrap_or_else(|e| e.into_inner());
                executing = guard;
            }
            true
        }
    }

    /// A statement whose `execute` blocks like a driver waiting on a remote
    /// query, and returns only when cancelled.
    #[derive(Clone)]
    struct FakeStatement {
        activity: Arc<DriverActivity>,
        cancelled: Arc<(Mutex<bool>, Condvar)>,
    }

    /// Long enough that a test failure is a failure rather than a flake, short
    /// enough that a broken cancel does not hang a suite.
    const EXECUTE_GIVE_UP: Duration = Duration::from_secs(60);

    impl Optionable for FakeStatement {
        type Option = OptionStatement;
        fn set_option(&mut self, _key: Self::Option, _value: OptionValue) -> AdbcResult<()> {
            Ok(())
        }
        fn get_option_string(&self, _key: Self::Option) -> AdbcResult<String> {
            Err(unsupported())
        }
        fn get_option_bytes(&self, _key: Self::Option) -> AdbcResult<Vec<u8>> {
            Err(unsupported())
        }
        fn get_option_int(&self, _key: Self::Option) -> AdbcResult<i64> {
            Err(unsupported())
        }
        fn get_option_double(&self, _key: Self::Option) -> AdbcResult<f64> {
            Err(unsupported())
        }
    }

    fn unsupported() -> AdbcError {
        AdbcError::with_message_and_status(
            "not supported by the fake driver",
            Status::NotImplemented,
        )
    }

    impl Statement for FakeStatement {
        fn bind(&mut self, _batch: RecordBatch) -> AdbcResult<()> {
            Ok(())
        }
        fn bind_stream(&mut self, _reader: Box<dyn RecordBatchReader + Send>) -> AdbcResult<()> {
            Ok(())
        }
        fn cancel(&mut self) -> AdbcResult<()> {
            self.activity.cancels.fetch_add(1, Ordering::SeqCst);
            let (lock, signal) = &*self.cancelled;
            let mut cancelled = lock.lock().unwrap_or_else(|e| e.into_inner());
            *cancelled = true;
            signal.notify_all();
            Ok(())
        }
        fn execute(&mut self) -> AdbcResult<Box<dyn RecordBatchReader + Send + 'static>> {
            self.activity.executes.fetch_add(1, Ordering::SeqCst);
            {
                let mut executing = self
                    .activity
                    .executing
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                *executing = true;
                self.activity.started.notify_all();
            }

            let (lock, signal) = &*self.cancelled;
            let mut cancelled = lock.lock().unwrap_or_else(|e| e.into_inner());
            let deadline = Instant::now() + EXECUTE_GIVE_UP;
            while !*cancelled {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    break;
                }
                let (guard, _) = signal
                    .wait_timeout(cancelled, remaining)
                    .unwrap_or_else(|e| e.into_inner());
                cancelled = guard;
            }
            Err(AdbcError::with_message_and_status(
                "query cancelled",
                Status::Cancelled,
            ))
        }
        fn execute_update(&mut self) -> AdbcResult<Option<i64>> {
            Err(unsupported())
        }
        fn execute_schema(&mut self) -> AdbcResult<Schema> {
            Ok(Schema::empty())
        }
        fn execute_partitions(&mut self) -> AdbcResult<PartitionedResult> {
            Err(unsupported())
        }
        fn get_parameter_schema(&self) -> AdbcResult<Schema> {
            Err(unsupported())
        }
        fn prepare(&mut self) -> AdbcResult<()> {
            Ok(())
        }
        fn set_sql_query(&mut self, _query: impl AsRef<str>) -> AdbcResult<()> {
            Ok(())
        }
        fn set_substrait_plan(&mut self, _plan: impl AsRef<[u8]>) -> AdbcResult<()> {
            Err(unsupported())
        }
    }

    struct FakeConnection {
        activity: Arc<DriverActivity>,
        cancelled: Arc<(Mutex<bool>, Condvar)>,
    }

    impl Drop for FakeConnection {
        fn drop(&mut self) {
            self.activity
                .connections_open
                .fetch_sub(1, Ordering::SeqCst);
        }
    }

    impl Optionable for FakeConnection {
        type Option = OptionConnection;
        fn set_option(&mut self, _key: Self::Option, _value: OptionValue) -> AdbcResult<()> {
            Ok(())
        }
        fn get_option_string(&self, _key: Self::Option) -> AdbcResult<String> {
            Err(unsupported())
        }
        fn get_option_bytes(&self, _key: Self::Option) -> AdbcResult<Vec<u8>> {
            Err(unsupported())
        }
        fn get_option_int(&self, _key: Self::Option) -> AdbcResult<i64> {
            Err(unsupported())
        }
        fn get_option_double(&self, _key: Self::Option) -> AdbcResult<f64> {
            Err(unsupported())
        }
    }

    impl Connection for FakeConnection {
        type StatementType = FakeStatement;

        fn new_statement(&mut self) -> AdbcResult<Self::StatementType> {
            Ok(FakeStatement {
                activity: Arc::clone(&self.activity),
                cancelled: Arc::clone(&self.cancelled),
            })
        }
        fn cancel(&mut self) -> AdbcResult<()> {
            Err(unsupported())
        }
        fn get_info(
            &self,
            _codes: Option<std::collections::HashSet<InfoCode>>,
        ) -> AdbcResult<Box<dyn RecordBatchReader + Send + 'static>> {
            Err(unsupported())
        }
        fn get_objects(
            &self,
            _depth: ObjectDepth,
            _catalog: Option<&str>,
            _db_schema: Option<&str>,
            _table_name: Option<&str>,
            _table_type: Option<Vec<&str>>,
            _column_name: Option<&str>,
        ) -> AdbcResult<Box<dyn RecordBatchReader + Send + 'static>> {
            Err(unsupported())
        }
        fn get_table_schema(
            &self,
            _catalog: Option<&str>,
            _db_schema: Option<&str>,
            _table_name: &str,
        ) -> AdbcResult<Schema> {
            Err(unsupported())
        }
        fn get_table_types(&self) -> AdbcResult<Box<dyn RecordBatchReader + Send + 'static>> {
            Err(unsupported())
        }
        fn get_statistic_names(&self) -> AdbcResult<Box<dyn RecordBatchReader + Send + 'static>> {
            Err(unsupported())
        }
        fn get_statistics(
            &self,
            _catalog: Option<&str>,
            _db_schema: Option<&str>,
            _table_name: Option<&str>,
            _approximate: bool,
        ) -> AdbcResult<Box<dyn RecordBatchReader + Send + 'static>> {
            Err(unsupported())
        }
        fn commit(&mut self) -> AdbcResult<()> {
            Err(unsupported())
        }
        fn rollback(&mut self) -> AdbcResult<()> {
            Err(unsupported())
        }
        fn read_partition(
            &self,
            _partition: impl AsRef<[u8]>,
        ) -> AdbcResult<Box<dyn RecordBatchReader + Send + 'static>> {
            Err(unsupported())
        }
    }

    #[derive(Clone)]
    struct FakeDatabase {
        activity: Arc<DriverActivity>,
        cancelled: Arc<(Mutex<bool>, Condvar)>,
    }

    impl Optionable for FakeDatabase {
        type Option = OptionDatabase;
        fn set_option(&mut self, _key: Self::Option, _value: OptionValue) -> AdbcResult<()> {
            Ok(())
        }
        fn get_option_string(&self, _key: Self::Option) -> AdbcResult<String> {
            Err(unsupported())
        }
        fn get_option_bytes(&self, _key: Self::Option) -> AdbcResult<Vec<u8>> {
            Err(unsupported())
        }
        fn get_option_int(&self, _key: Self::Option) -> AdbcResult<i64> {
            Err(unsupported())
        }
        fn get_option_double(&self, _key: Self::Option) -> AdbcResult<f64> {
            Err(unsupported())
        }
    }

    impl Database for FakeDatabase {
        type ConnectionType = FakeConnection;

        fn new_connection(&self) -> AdbcResult<Self::ConnectionType> {
            self.activity
                .connections_open
                .fetch_add(1, Ordering::SeqCst);
            Ok(FakeConnection {
                activity: Arc::clone(&self.activity),
                cancelled: Arc::clone(&self.cancelled),
            })
        }

        fn new_connection_with_opts(
            &self,
            _opts: impl IntoIterator<Item = (OptionConnection, OptionValue)>,
        ) -> AdbcResult<Self::ConnectionType> {
            self.new_connection()
        }
    }

    fn fake_pool(
        activity: &Arc<DriverActivity>,
    ) -> (
        Arc<crate::sql::db_connection_pool::adbcpool::ADBCPool<FakeDatabase>>,
        Arc<(Mutex<bool>, Condvar)>,
    ) {
        let cancelled = Arc::new((Mutex::new(false), Condvar::new()));
        let database = FakeDatabase {
            activity: Arc::clone(activity),
            cancelled: Arc::clone(&cancelled),
        };
        let pool =
            crate::sql::db_connection_pool::adbcpool::AdbcConnectionPoolBuilder::new(database)
                .with_max_size(Some(1))
                .build()
                .expect("the pool should build");
        (Arc::new(pool), cancelled)
    }

    /// Dropping the stream must cancel the running query and give the pooled
    /// connection back, which is what a client that goes away needs to happen.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn dropping_the_stream_cancels_the_query_and_frees_the_connection() {
        use crate::sql::db_connection_pool::DbConnectionPool;

        let activity = Arc::new(DriverActivity::default());
        let (pool, _cancelled) = fake_pool(&activity);

        let conn = pool
            .connect()
            .await
            .expect("a connection should be available");
        let stream = super::super::query_arrow(conn, "SELECT 1".to_string(), None)
            .await
            .expect("the query should start");

        assert!(
            activity.wait_until_executing(Duration::from_secs(10)),
            "the driver never started executing"
        );

        drop(stream);

        // The pool holds one connection: it can only be handed out again once
        // the cancelled query has released it.
        let waited = Instant::now();
        let second = tokio::time::timeout(Duration::from_secs(20), pool.connect())
            .await
            .expect("the pool connection should come back after cancellation")
            .expect("a connection should be available");
        drop(second);

        assert_eq!(
            activity.cancels.load(Ordering::SeqCst),
            1,
            "the abandoned query was not cancelled"
        );
        assert!(
            waited.elapsed() < Duration::from_secs(20),
            "the pool connection took {:?} to come back",
            waited.elapsed()
        );
    }

    /// A caller that goes away while the statement is still being prepared must
    /// stop the query from starting at all.
    #[test]
    fn abandoning_before_the_query_starts_stops_it_starting() {
        let cancellation: Arc<Mutex<QueryCancellation<FakeStatement>>> =
            Arc::new(Mutex::new(QueryCancellation::Preparing));

        let guard = CancelOnDrop {
            inner: Box::pin(RecordBatchStreamAdapter::new(
                Arc::new(Schema::empty()),
                futures::stream::empty(),
            )),
            cancellation: Arc::clone(&cancellation),
        };
        drop(guard);

        assert!(
            matches!(
                *lock_cancellation(&cancellation),
                QueryCancellation::Abandoned
            ),
            "the query was not marked abandoned, so it would still be started"
        );
    }

    /// A query that finishes normally must not be cancelled when its stream is
    /// dropped.
    #[test]
    fn a_finished_query_is_not_cancelled() {
        let activity = Arc::new(DriverActivity::default());
        let cancellation: Arc<Mutex<QueryCancellation<FakeStatement>>> =
            Arc::new(Mutex::new(QueryCancellation::Finished));

        let guard = CancelOnDrop {
            inner: Box::pin(RecordBatchStreamAdapter::new(
                Arc::new(Schema::empty()),
                futures::stream::empty(),
            )),
            cancellation: Arc::clone(&cancellation),
        };
        drop(guard);

        assert_eq!(
            activity.cancels.load(Ordering::SeqCst),
            0,
            "a completed query was cancelled"
        );
        assert!(matches!(
            *lock_cancellation(&cancellation),
            QueryCancellation::Finished
        ));
    }
}
