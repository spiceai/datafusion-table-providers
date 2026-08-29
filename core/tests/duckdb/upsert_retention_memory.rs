//! Memory-stability stress test for the DuckDB write path an accelerated
//! dataset uses: a primary-key upsert on every refresh, a retention delete
//! trimming rows out from under it, and query load running against the table
//! the whole time.
//!
//! One cycle is one refresh, and it does what a refresh does, in order:
//!
//! 1. **Upsert** — `insert_into(.., InsertOp::Append)` on a writer configured
//!    with `on_conflict: upsert`. Internally this registers the incoming
//!    batches as an FFI arrow scan view (`__scan_<table>_<ts>`), runs
//!    `INSERT INTO <t> SELECT * FROM <scan> ON CONFLICT (<pk>) DO UPDATE SET ..`,
//!    drops the scan, `ANALYZE`s, and commits.
//! 2. **Retention** — a `DELETE FROM <t> WHERE <predicate>` issued as
//!    `delete_from(filters)`, the call a parsed retention SQL statement lowers
//!    to. It runs right after the write, as it does in a refresh.
//!
//! Meanwhile a configurable number of reader tasks query the table
//! continuously, overlapping the writes rather than taking turns with them.
//! That is what a deployment looks like: refreshes on one schedule, queries
//! arriving whenever they arrive. It also puts real pressure on row-version
//! retention — a reader holding a transaction open across an upsert is what
//! keeps old versions alive — which a sequential loop cannot produce.
//!
//! The reader shapes are the ones such deployments issue against an
//! accelerated dataset: the filtered, ordered scan a dependent materialization
//! performs, and a narrower keyed lookup. They aggregate or limit in SQL so
//! the test's own result buffers stay out of the measurement.
//!
//! Deliberately one table and one engine: a second instance would add its own
//! buffer pool to every RSS reading without adding pressure to the path under
//! test.
//!
//! The test asserts that resident memory reaches a plateau and stays there.
//! DuckDB's buffer pool is expected to grow toward its `memory_limit` early,
//! so the baseline is taken after a warmup and the assertion is about growth
//! past that plateau. Every tenth cycle also logs DuckDB's own
//! `duckdb_memory()` accounting, so a failure says which allocator tag grew
//! rather than only that the process did.
//!
//! Every dimension of the workload is tunable from the environment, so a
//! growth configuration can be searched for without recompiling — cycles,
//! rows, key space, table width, payload size, reader count, cadences, and the
//! thresholds; see the constants below for the names.
//!
//! The defaults are the calibrated shape, not arbitrary. The key space has to
//! saturate well before the measurement window, or rising memory is just the
//! table still filling; at 50k rows a cycle against 300k keys that happens
//! around cycle 20, which is where the warmup ends. The width and payload are
//! what made the per-cycle growth large enough to read off in a minute rather
//! than an hour.
//!
//! Run it explicitly, and in release — the debug build of this many rows is
//! far slower than the engine work it is measuring:
//!
//! ```text
//! cargo test --release -p datafusion-table-providers --features duckdb,duckdb-federation \
//!     --test integration upsert_with_retention -- --ignored --nocapture
//! ```
//!
//! It is `#[ignore]`d because it currently fails: DuckDB 1.5.5 grows about
//! 6 MiB per cycle at steady state on this workload and eventually exhausts
//! `memory_limit`, where 1.4.4 plateaus. Remove the attribute once the
//! underlying growth is fixed, so it guards against the regression returning.
//! The failure it is built to produce is the engine's own
//! `Out of Memory Error`: memory that never plateaus reaches the configured
//! `memory_limit` and the refresh write fails, reported with the row count and
//! the memory breakdown at that moment. The RSS comparison at the end of the
//! run is the secondary check, for growth too slow to reach the ceiling.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use datafusion::arrow::array::{
    ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray, TimestampMicrosecondArray,
};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit};
use datafusion::catalog::{TableProvider, TableProviderFactory};
use datafusion::common::{Constraint, Constraints, ScalarValue, ToDFSchema};
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::execution::context::SessionContext;
use datafusion::logical_expr::dml::InsertOp;
use datafusion::logical_expr::{col, lit, CreateExternalTable, Expr};
use datafusion::physical_plan::collect;
use datafusion_table_providers::duckdb::write::DuckDBTableWriter;
use datafusion_table_providers::duckdb::{DuckDB, DuckDBTableProviderFactory};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rstest::rstest;

/// Refresh cycles to run. Each is one upsert, plus a retention delete on
/// its own cadence.
const DEFAULT_CYCLES: usize = 60;
/// Rows written per upsert cycle.
const DEFAULT_ROWS_PER_CYCLE: usize = 50_000;
/// Distinct primary keys the workload cycles through. Smaller than
/// `cycles * rows_per_cycle`, so most rows land on the `DO UPDATE` path
/// rather than inserting.
const DEFAULT_KEY_SPACE: usize = 300_000;
/// Extra columns appended to the base schema, alternating string and float.
/// Raise this to test a wide table.
const DEFAULT_EXTRA_COLUMNS: usize = 24;
/// Bytes in the variable-width payload column of each row.
const DEFAULT_PAYLOAD_BYTES: usize = 128;
/// Run the retention delete every N cycles.
const DEFAULT_RETENTION_EVERY: usize = 4;
/// Reader tasks querying the table concurrently with the writes. Zero runs
/// the refresh loop on its own, which is the comparison for whether
/// concurrency changes the growth curve.
const DEFAULT_READERS: usize = 4;
/// Pause between a reader's queries. Small enough to keep reads overlapping
/// the writes, large enough that readers do not monopolize the CPU.
const DEFAULT_READER_PAUSE_MS: u64 = 25;
/// How long one write or delete may take before the test calls it stuck. The
/// write path can deadlock against concurrent reads; without this a stuck
/// run hangs forever instead of failing.
const DEFAULT_OP_TIMEOUT_SECS: u64 = 300;
/// Cycles to run before the memory baseline is taken. DuckDB's buffer pool
/// and the allocator's arenas both need to reach steady state first.
const DEFAULT_WARMUP_CYCLES: usize = 20;
/// `memory_limit` given to the DuckDB instance.
///
/// Deliberately small. Growth that never plateaus reaches this ceiling within
/// the default cycle count and fails the write with an `Out of Memory Error`,
/// which is the signal this test is built around: a hard error from the
/// engine, with no threshold to tune and no allocator noise to tolerate. A
/// workload that holds steady stays well below it — the table and its open
/// read transactions account for under half of this.
const DEFAULT_MEMORY_LIMIT: &str = "256MB";
/// Rows older than this (seconds) and flagged deleted are trimmed by retention.
const DEFAULT_RETENTION_WINDOW_SECS: i64 = 900;
/// Allowed absolute RSS growth past the post-warmup baseline.
const DEFAULT_MAX_GROWTH_MB: u64 = 256;
/// Allowed relative RSS growth past the post-warmup baseline.
const DEFAULT_MAX_GROWTH_RATIO: f64 = 1.3;

#[derive(Debug, Clone)]
struct StressConfig {
    cycles: usize,
    rows_per_cycle: usize,
    key_space: usize,
    extra_columns: usize,
    payload_bytes: usize,
    retention_every: usize,
    readers: usize,
    reader_pause: Duration,
    warmup_cycles: usize,
    memory_limit: String,
    retention_window_secs: i64,
    op_timeout: Duration,
    max_growth_bytes: u64,
    max_growth_ratio: f64,
}

impl StressConfig {
    fn from_env() -> Self {
        let cycles = env_usize("DUCKDB_MEM_TEST_CYCLES", DEFAULT_CYCLES).max(1);
        let warmup_cycles = env_usize("DUCKDB_MEM_TEST_WARMUP", DEFAULT_WARMUP_CYCLES);

        Self {
            cycles,
            rows_per_cycle: env_usize("DUCKDB_MEM_TEST_ROWS", DEFAULT_ROWS_PER_CYCLE).max(1),
            key_space: env_usize("DUCKDB_MEM_TEST_KEYSPACE", DEFAULT_KEY_SPACE).max(1),
            extra_columns: env_usize("DUCKDB_MEM_TEST_EXTRA_COLUMNS", DEFAULT_EXTRA_COLUMNS),
            payload_bytes: env_usize("DUCKDB_MEM_TEST_PAYLOAD_BYTES", DEFAULT_PAYLOAD_BYTES).max(1),
            retention_every: env_usize("DUCKDB_MEM_TEST_RETENTION_EVERY", DEFAULT_RETENTION_EVERY)
                .max(1),
            readers: env_usize("DUCKDB_MEM_TEST_READERS", DEFAULT_READERS),
            reader_pause: Duration::from_millis(env_u64(
                "DUCKDB_MEM_TEST_READER_PAUSE_MS",
                DEFAULT_READER_PAUSE_MS,
            )),
            // A warmup that swallowed the whole run would leave nothing to
            // compare, so keep at least a few measured cycles.
            warmup_cycles: warmup_cycles.min(cycles.saturating_sub(3)),
            memory_limit: env_string("DUCKDB_MEM_TEST_MEMORY_LIMIT", DEFAULT_MEMORY_LIMIT),
            retention_window_secs: env_i64(
                "DUCKDB_MEM_TEST_RETENTION_WINDOW_SECS",
                DEFAULT_RETENTION_WINDOW_SECS,
            ),
            op_timeout: Duration::from_secs(env_u64(
                "DUCKDB_MEM_TEST_OP_TIMEOUT_SECS",
                DEFAULT_OP_TIMEOUT_SECS,
            )),
            max_growth_bytes: env_u64("DUCKDB_MEM_TEST_MAX_GROWTH_MB", DEFAULT_MAX_GROWTH_MB)
                * 1024
                * 1024,
            max_growth_ratio: env_f64("DUCKDB_MEM_TEST_MAX_GROWTH_RATIO", DEFAULT_MAX_GROWTH_RATIO),
        }
    }
}

fn env_string(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_i64(name: &str, default: i64) -> i64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Current resident set size of this process, in bytes.
///
/// Hand-rolled rather than pulled from a crate: the only consumer is this
/// test, and both platforms that run it expose RSS without a dependency.
/// Linux reads `VmRSS` out of `/proc/self/status` (reported in kB); macOS
/// shells out to `ps`, which is cheap enough at one sample per refresh cycle.
/// Anywhere else the test still runs, reporting no samples, and the memory
/// assertion is skipped rather than guessed at.
///
/// RSS is the symptom that matters — a container is killed on RSS, not on
/// DuckDB's own accounting — and it is the only measure that sees allocations
/// DuckDB does not tag for itself, such as anything retained on the FFI or
/// binding side. `duckdb_memory()` then says which DuckDB subsystem grew.
/// Together they separate an engine problem from a binding problem; neither
/// does that alone.
fn current_rss_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                let kb: u64 = rest
                    .split_whitespace()
                    .next()
                    .and_then(|v| v.parse().ok())?;
                return Some(kb * 1024);
            }
        }
        None
    }

    #[cfg(target_os = "macos")]
    {
        let out = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p"])
            .arg(std::process::id().to_string())
            .output()
            .ok()?;
        let kb: u64 = String::from_utf8_lossy(&out.stdout).trim().parse().ok()?;
        Some(kb * 1024)
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

fn as_mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

/// Which primary key shape the workload upserts on. Both shapes appear in real
/// accelerations, and they take different paths through DuckDB's conflict
/// target handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrimaryKeyShape {
    /// A single surrogate key column.
    SingleColumn,
    /// A composite key over the surrogate key and a grouping column.
    Composite,
}

impl PrimaryKeyShape {
    /// Column indices of the key within [`dataset_schema`].
    fn indices(self) -> Vec<usize> {
        match self {
            Self::SingleColumn => vec![0],
            Self::Composite => vec![0, 1],
        }
    }

    /// The `on_conflict` option string, in the form the table factory parses.
    fn on_conflict_option(self) -> &'static str {
        match self {
            Self::SingleColumn => "upsert:id",
            Self::Composite => "upsert:(id, group_id)",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::SingleColumn => "single_column_pk",
            Self::Composite => "composite_pk",
        }
    }
}

/// Schema of the accelerated dataset: a key, a grouping column, a soft-delete
/// flag, an event timestamp, a variable-width payload, and `extra_columns`
/// filler columns that widen the row.
fn dataset_schema(extra_columns: usize) -> SchemaRef {
    let mut fields = vec![
        Field::new("id", DataType::Int64, false),
        Field::new("group_id", DataType::Int64, false),
        Field::new("deleted", DataType::Utf8, false),
        Field::new(
            "processed_time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("payload", DataType::Utf8, false),
    ];

    for i in 0..extra_columns {
        if i % 2 == 0 {
            fields.push(Field::new(format!("attr_{i}"), DataType::Utf8, true));
        } else {
            fields.push(Field::new(format!("metric_{i}"), DataType::Float64, true));
        }
    }

    Arc::new(Schema::new(fields))
}

fn now_micros() -> i64 {
    i64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time is after the unix epoch")
            .as_micros(),
    )
    .expect("microseconds since the epoch fit in an i64")
}

/// Picks the keys for one refresh batch.
///
/// Keys are unique *within* a batch — the writer rejects a batch that violates
/// its own primary key, and a real refresh batch carries at most one row per
/// key. Repetition that drives the `DO UPDATE` path therefore comes from keys
/// recurring *across* batches: roughly 70% of each batch re-uses keys already
/// in the table, and the rest extend the key space until it is exhausted, after
/// which every row is an update.
fn batch_keys(rng: &mut StdRng, cfg: &StressConfig, high_water: &mut i64) -> Vec<i64> {
    let rows = cfg.rows_per_cycle;
    let remaining_new = cfg.key_space.saturating_sub(*high_water as usize);
    let existing = *high_water as usize;

    // Target 30% fresh keys, bounded by what the key space and the table can
    // actually supply.
    let mut new_count = (rows * 3 / 10).min(remaining_new);
    let mut reused_count = (rows - new_count).min(existing);
    // Whatever the table cannot supply as reuse comes from new keys instead.
    new_count = (rows - reused_count).min(remaining_new);
    reused_count = (rows - new_count).min(existing);

    let mut keys = Vec::with_capacity(new_count + reused_count);
    keys.extend(
        rand::seq::index::sample(rng, existing, reused_count)
            .into_iter()
            .map(|i| i as i64),
    );
    for _ in 0..new_count {
        keys.push(*high_water);
        *high_water += 1;
    }

    keys
}

/// Builds one refresh's worth of rows.
///
/// Keys are drawn from a fixed key space so most rows collide with a row
/// already in the table and take the `DO UPDATE` path; the rest extend the
/// high-water mark and insert. A quarter of the rows are soft-deleted with an
/// aged timestamp, which is what retention later trims.
fn make_batch(
    schema: &SchemaRef,
    rng: &mut StdRng,
    cfg: &StressConfig,
    high_water: &mut i64,
    now_us: i64,
) -> RecordBatch {
    let rows = cfg.rows_per_cycle;
    let window_us = cfg.retention_window_secs * 1_000_000;

    let mut ids = Vec::with_capacity(rows);
    let mut group_ids = Vec::with_capacity(rows);
    let mut deleted = Vec::with_capacity(rows);
    let mut processed = Vec::with_capacity(rows);
    let mut payloads = Vec::with_capacity(rows);

    for id in batch_keys(rng, cfg, high_water) {
        ids.push(id);
        group_ids.push(id % 64);
        deleted.push(if rng.random_bool(0.25) { "true" } else { "false" });
        // Spread timestamps across twice the retention window so each
        // retention pass has something to delete and something to keep.
        processed.push(now_us - rng.random_range(0..(2 * window_us)));
        payloads.push(payload_for(id, cfg.payload_bytes));
    }

    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(ids.clone())),
        Arc::new(Int64Array::from(group_ids)),
        Arc::new(StringArray::from(deleted)),
        Arc::new(
            TimestampMicrosecondArray::from(processed).with_timezone(Arc::from("UTC")),
        ),
        Arc::new(StringArray::from(payloads)),
    ];

    for i in 0..cfg.extra_columns {
        if i % 2 == 0 {
            let values = ids
                .iter()
                .map(|id| format!("attr-{i}-{}", id % 1024))
                .collect::<Vec<_>>();
            columns.push(Arc::new(StringArray::from(values)));
        } else {
            let values = ids
                .iter()
                .map(|id| (*id as f64) * 0.5 + i as f64)
                .collect::<Vec<_>>();
            columns.push(Arc::new(Float64Array::from(values)));
        }
    }

    RecordBatch::try_new(Arc::clone(schema), columns).expect("batch matches the dataset schema")
}

/// A payload of `bytes` bytes whose content varies with the key, so the value
/// genuinely changes on an update rather than rewriting the same string.
fn payload_for(id: i64, bytes: usize) -> String {
    let mut s = format!("{id:016}");
    while s.len() < bytes {
        s.push(char::from(b'a' + (id % 26) as u8));
    }
    s.truncate(bytes);
    s
}

/// Creates a file-backed DuckDB table through the same factory an accelerator
/// uses, with a primary key, an `on_conflict` behaviour, and instance settings.
async fn create_table(
    ctx: &SessionContext,
    name: &str,
    schema: &SchemaRef,
    db_path: &str,
    cfg: &StressConfig,
    pk_indices: Vec<usize>,
    on_conflict: &str,
) -> Arc<dyn TableProvider> {
    let options = HashMap::from([
        ("mode".to_string(), "file".to_string()),
        ("duckdb_open".to_string(), db_path.to_string()),
        ("memory_limit".to_string(), cfg.memory_limit.clone()),
        // Readers and the writer need connections at the same time; a
        // single-connection pool would serialize them, and r2d2's wait has no
        // timeout. Accelerators size the pool by how many components share the
        // instance — this is that sizing, fixed.
        ("connection_pool_size".to_string(), "16".to_string()),
        // What an accelerated dataset configures for write throughput.
        ("preserve_insertion_order".to_string(), "false".to_string()),
        ("on_conflict".to_string(), on_conflict.to_string()),
    ]);

    let cmd = CreateExternalTable {
        schema: Arc::new(
            schema
                .as_ref()
                .clone()
                .to_dfschema()
                .expect("dataset schema converts to a DFSchema"),
        ),
        name: name.into(),
        location: String::new(),
        file_type: String::new(),
        table_partition_cols: vec![],
        if_not_exists: false,
        or_replace: false,
        definition: None,
        order_exprs: vec![],
        unbounded: false,
        options,
        constraints: Constraints::new_unverified(vec![Constraint::PrimaryKey(pk_indices)]),
        column_defaults: HashMap::new(),
        temporary: false,
    };

    DuckDBTableProviderFactory::new(duckdb::AccessMode::ReadWrite)
        .create(&ctx.state(), &cmd)
        .await
        .expect("table provider is created")
}

/// One refresh write: append a batch through the writer, upserting on conflict.
///
/// The error is returned rather than unwrapped because it is the test's
/// primary signal: when the engine's memory grows without bound, this is where
/// it surfaces, as an `Out of Memory Error` against the configured
/// `memory_limit` while the table itself sits at a steady row count.
async fn upsert_cycle(
    ctx: &SessionContext,
    table: &Arc<dyn TableProvider>,
    batch: RecordBatch,
) -> datafusion::error::Result<()> {
    let schema = batch.schema();
    let source = MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None)
        .expect("memory source for the refresh batch");

    let plan = table
        .insert_into(&ctx.state(), source, InsertOp::Append)
        .await?;

    collect(plan, ctx.task_ctx()).await.map(|_| ())
}

/// The retention pass: delete soft-deleted rows older than the window. This is
/// the same call a parsed `retention_sql` DELETE lowers to.
async fn retention_cycle(
    ctx: &SessionContext,
    table: &Arc<dyn TableProvider>,
    cfg: &StressConfig,
    now_us: i64,
) -> datafusion::error::Result<u64> {
    let cutoff = now_us - cfg.retention_window_secs * 1_000_000;
    let filters: Vec<Expr> = vec![col("deleted").eq(lit("true")).and(
        col("processed_time").lt(lit(ScalarValue::TimestampMicrosecond(
            Some(cutoff),
            Some("UTC".into()),
        ))),
    )];

    let plan = table.delete_from(&ctx.state(), filters).await?;
    let batches = collect(plan, ctx.task_ctx()).await?;

    Ok(batches
        .first()
        .and_then(|b| {
            b.column(0)
                .as_any()
                .downcast_ref::<datafusion::arrow::array::UInt64Array>()
                .map(|a| a.value(0))
        })
        .unwrap_or(0))
}

/// What the concurrent readers observed. Reader failures are counted rather
/// than panicked on, so the writer's own error — or a memory limit being hit —
/// is what fails the test, with the reader tally as context.
#[derive(Debug, Default)]
struct ReaderStats {
    queries: AtomicU64,
    errors: AtomicU64,
    first_error: Mutex<Option<String>>,
}

impl ReaderStats {
    fn record_error(&self, err: &str) {
        self.errors.fetch_add(1, Ordering::Relaxed);
        let mut first = self
            .first_error
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if first.is_none() {
            *first = Some(err.to_string());
        }
    }
}

/// Query load against the accelerated table, running for the whole soak.
///
/// Each reader alternates the two shapes a deployment issues: the filtered,
/// ordered scan a dependent materialization performs, and a keyed lookup. Both
/// aggregate or limit in SQL, so what the test itself holds stays negligible
/// next to what the engine holds — otherwise the reader would show up in the
/// very RSS number being asserted on.
fn spawn_readers(
    ctx: &SessionContext,
    table_name: &str,
    cfg: &StressConfig,
    stop: &Arc<AtomicBool>,
    stats: &Arc<ReaderStats>,
) -> Vec<tokio::task::JoinHandle<()>> {
    (0..cfg.readers)
        .map(|reader| {
            let ctx = ctx.clone();
            let table_name = table_name.to_string();
            let stop = Arc::clone(stop);
            let stats = Arc::clone(stats);
            let pause = cfg.reader_pause;

            tokio::spawn(async move {
                let mut round: u64 = 0;
                while !stop.load(Ordering::Relaxed) {
                    let sql = if round % 2 == 0 {
                        format!(
                            "SELECT count(id), sum(group_id) FROM {table_name} \
                             WHERE deleted = 'false'"
                        )
                    } else {
                        let group = (round + reader as u64) % 64;
                        format!(
                            "SELECT id, group_id, payload FROM {table_name} \
                             WHERE deleted = 'false' AND group_id = {group} \
                             ORDER BY id LIMIT 500"
                        )
                    };

                    match ctx.sql(&sql).await {
                        Ok(df) => match df.collect().await {
                            Ok(_) => {
                                stats.queries.fetch_add(1, Ordering::Relaxed);
                            }
                            Err(e) => stats.record_error(&e.to_string()),
                        },
                        Err(e) => stats.record_error(&e.to_string()),
                    }

                    round += 1;
                    tokio::time::sleep(pause).await;
                }
            })
        })
        .collect()
}

/// Runs `op`, failing the test if it does not finish within the configured
/// timeout. The write path can deadlock against a concurrent reader, and a
/// deadlocked run that hangs forever reports nothing at all.
async fn with_watchdog<F, T>(cfg: &StressConfig, what: &str, cycle: usize, op: F) -> T
where
    F: std::future::Future<Output = T>,
{
    match tokio::time::timeout(cfg.op_timeout, op).await {
        Ok(value) => value,
        Err(_) => panic!(
            "{what} did not finish within {:?} on cycle {cycle}; the write path appears stuck",
            cfg.op_timeout
        ),
    }
}

/// What DuckDB itself thinks it is holding, per allocator tag, plus the size
/// of the database file.
///
/// RSS alone cannot say whether growth is DuckDB's buffer manager filling up
/// to its configured limit (expected) or something accumulating that should
/// have been released (not). This reads the engine's own accounting through
/// the writer's pool, so a failing run reports *where* the memory went.
fn duckdb_memory_report(table: &Arc<dyn TableProvider>) -> Option<String> {
    let writer = table.downcast_ref::<DuckDBTableWriter>()?;
    let pool = writer.pool();
    let mut db_conn = pool.connect_sync().ok()?;
    let conn = &DuckDB::duckdb_conn(&mut db_conn).ok()?.conn;

    let mut parts = Vec::new();

    if let Ok(mut stmt) = conn.prepare(
        "SELECT tag, memory_usage_bytes FROM duckdb_memory() \
         WHERE memory_usage_bytes > 0 ORDER BY memory_usage_bytes DESC LIMIT 5",
    ) {
        if let Ok(rows) =
            stmt.query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)))
        {
            for row in rows.flatten() {
                parts.push(format!("{}={:.1}MiB", row.0, row.1 as f64 / (1024.0 * 1024.0)));
            }
        }
    }

    if let Ok(size) = conn.query_row(
        "SELECT database_size FROM pragma_database_size()",
        [],
        |row| row.get::<_, String>(0),
    ) {
        parts.push(format!("file={size}"));
    }

    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" "))
    }
}

async fn count_rows(ctx: &SessionContext, table_name: &str) -> u64 {
    let batches = ctx
        // Counted over a column rather than `COUNT(*)`: the empty projection
        // `COUNT(*)` plans to trips a schema mismatch in the DuckDB scan.
        .sql(&format!("SELECT COUNT(id) FROM {table_name}"))
        .await
        .expect("count query plans")
        .collect()
        .await
        .expect("count query runs");

    batches
        .first()
        .and_then(|b| {
            b.column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .map(|a| a.value(0))
        })
        .unwrap_or(0)
        .try_into()
        .unwrap_or(0)
}

/// Median of a sample window, which rejects the occasional spike from a
/// concurrent allocation better than a mean does.
fn median(values: &mut [u64]) -> u64 {
    if values.is_empty() {
        return 0;
    }
    values.sort_unstable();
    values[values.len() / 2]
}

/// Fails the test on an engine error, with the state that explains it.
///
/// This is the assertion that matters. Unbounded growth inside DuckDB ends as
/// an `Out of Memory Error` against `memory_limit` — a hard, deterministic
/// failure from the engine itself, needing no threshold and tolerating no
/// allocator noise, unlike the RSS comparison at the end of the run. The row
/// count and the memory breakdown are printed with it, because "out of memory
/// at a steady-state row count" is the claim, and neither half makes it alone.
async fn fail_on_engine_error(
    ctx: &SessionContext,
    table: &Arc<dyn TableProvider>,
    table_name: &str,
    cfg: &StressConfig,
    cycle: usize,
    what: &str,
    error: &datafusion::error::DataFusionError,
) -> ! {
    let rows = count_rows(ctx, table_name).await;
    let engine = duckdb_memory_report(table).unwrap_or_else(|| "unavailable".to_string());

    panic!(
        "{what} failed on cycle {cycle} of {} with {rows} rows in the table \
         (key space {}, memory_limit {}).\nEngine memory: {engine}\nError: {error}",
        cfg.cycles, cfg.key_space, cfg.memory_limit,
    );
}

#[rstest]
#[case::single_column(PrimaryKeyShape::SingleColumn)]
#[case::composite(PrimaryKeyShape::Composite)]
#[test_log::test(tokio::test(flavor = "multi_thread"))]
#[ignore = "soak test: minutes to run, and currently red on DuckDB 1.5.5 - see the module docs"]
async fn upsert_with_retention_under_query_load_does_not_grow_memory(
    #[case] pk_shape: PrimaryKeyShape,
) {
    let cfg = StressConfig::from_env();
    let temp_dir = tempfile::tempdir().expect("temp dir is created");
    let dataset_name = format!("dataset_{}", pk_shape.label());
    let dataset_path = temp_dir
        .path()
        .join(format!("{dataset_name}.duckdb"))
        .to_string_lossy()
        .to_string();

    let schema = dataset_schema(cfg.extra_columns);
    let ctx = SessionContext::new();
    let dataset = create_table(
        &ctx,
        &dataset_name,
        &schema,
        &dataset_path,
        &cfg,
        pk_shape.indices(),
        pk_shape.on_conflict_option(),
    )
    .await;

    ctx.register_table(dataset_name.as_str(), Arc::clone(&dataset))
        .expect("dataset is registered");

    tracing::info!(
        "starting {} cycles ({} rows/cycle, {} columns, key space {}, payload {}B, \
         {} readers, memory_limit {})",
        cfg.cycles,
        cfg.rows_per_cycle,
        schema.fields().len(),
        cfg.key_space,
        cfg.payload_bytes,
        cfg.readers,
        cfg.memory_limit,
    );

    let stop = Arc::new(AtomicBool::new(false));
    let reader_stats = Arc::new(ReaderStats::default());
    let readers = spawn_readers(&ctx, &dataset_name, &cfg, &stop, &reader_stats);

    let mut rng = StdRng::seed_from_u64(0x5EED_0000 + pk_shape as u64);
    let mut high_water: i64 = 0;
    let mut samples: Vec<(usize, u64)> = Vec::with_capacity(cfg.cycles);

    for cycle in 0..cfg.cycles {
        let now_us = now_micros();
        let batch = make_batch(&schema, &mut rng, &cfg, &mut high_water, now_us);
        let write = with_watchdog(
            &cfg,
            "refresh write",
            cycle,
            upsert_cycle(&ctx, &dataset, batch),
        )
        .await;
        if let Err(e) = write {
            fail_on_engine_error(&ctx, &dataset, &dataset_name, &cfg, cycle, "refresh write", &e)
                .await;
        }

        // `cycle > 0` so that a cadence set high enough to disable the arm
        // really disables it: cycle 0 is divisible by every interval.
        let deleted = if cycle > 0 && cycle % cfg.retention_every == 0 {
            match with_watchdog(
                &cfg,
                "retention delete",
                cycle,
                retention_cycle(&ctx, &dataset, &cfg, now_us),
            )
            .await
            {
                Ok(deleted) => deleted,
                Err(e) => {
                    fail_on_engine_error(
                        &ctx,
                        &dataset,
                        &dataset_name,
                        &cfg,
                        cycle,
                        "retention delete",
                        &e,
                    )
                    .await
                }
            }
        } else {
            0
        };

        let rows = count_rows(&ctx, &dataset_name).await;
        let reads = reader_stats.queries.load(Ordering::Relaxed);

        if let Some(rss) = current_rss_bytes() {
            samples.push((cycle, rss));
            tracing::info!(
                "cycle {cycle:>3}: rows={rows:>9} retention_deleted={deleted:>8} \
                 reads={reads:>7} rss={:>8.1} MiB",
                as_mib(rss),
            );
            if cycle % 10 == 0 {
                if let Some(report) = duckdb_memory_report(&dataset) {
                    tracing::info!("cycle {cycle:>3}: engine memory: {report}");
                }
            }
        } else {
            tracing::info!(
                "cycle {cycle:>3}: rows={rows:>9} retention_deleted={deleted:>8} reads={reads:>7}"
            );
        }
    }

    stop.store(true, Ordering::Relaxed);
    for reader in readers {
        let _ = reader.await;
    }

    let reads = reader_stats.queries.load(Ordering::Relaxed);
    let read_errors = reader_stats.errors.load(Ordering::Relaxed);
    tracing::info!(
        "{}: {reads} concurrent reads completed, {read_errors} failed{}",
        pk_shape.label(),
        reader_stats
            .first_error
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .map(|e| format!(" (first: {e})"))
            .unwrap_or_default(),
    );

    // With readers configured, a run where none of them completed proves
    // nothing about behaviour under load.
    assert!(
        cfg.readers == 0 || reads > 0,
        "no concurrent read completed, so the writes never overlapped a reader"
    );

    // Upsert is only doing its job if the table never holds more rows than
    // there are distinct keys. A growing row count would make any memory
    // growth unremarkable, so this has to hold before the memory claim means
    // anything.
    let final_rows = count_rows(&ctx, &dataset_name).await;
    assert!(
        final_rows <= cfg.key_space as u64,
        "table holds {final_rows} rows for a key space of {}, so the upsert is appending duplicates rather than updating",
        cfg.key_space
    );

    let Some(baseline_start) = samples.iter().position(|(cycle, _)| *cycle >= cfg.warmup_cycles)
    else {
        tracing::warn!("no RSS samples were collected on this platform; skipping memory assertion");
        return;
    };

    // Baseline: the plateau right after warmup. Final: the tail of the run.
    // Both are medians of a small window so a single spike decides nothing.
    let window = 3.min(samples.len() - baseline_start).max(1);
    let mut baseline_window: Vec<u64> = samples[baseline_start..baseline_start + window]
        .iter()
        .map(|(_, rss)| *rss)
        .collect();
    let mut tail_window: Vec<u64> = samples[samples.len() - window..]
        .iter()
        .map(|(_, rss)| *rss)
        .collect();

    let baseline = median(&mut baseline_window);
    let tail = median(&mut tail_window);
    let peak = samples.iter().map(|(_, rss)| *rss).max().unwrap_or(0);
    let growth = tail.saturating_sub(baseline);

    tracing::info!(
        "{}: baseline {:.1} MiB (after {} warmup cycles), tail {:.1} MiB, peak {:.1} MiB, growth {:.1} MiB",
        pk_shape.label(),
        as_mib(baseline),
        cfg.warmup_cycles,
        as_mib(tail),
        as_mib(peak),
        as_mib(growth),
    );

    assert!(
        growth <= cfg.max_growth_bytes,
        "RSS grew {:.1} MiB past the post-warmup baseline of {:.1} MiB over {} cycles (cap {:.1} MiB); \
         memory is not reaching a plateau under upsert + retention with {} concurrent readers",
        as_mib(growth),
        as_mib(baseline),
        cfg.cycles,
        as_mib(cfg.max_growth_bytes),
        cfg.readers,
    );

    let ratio = if baseline == 0 {
        1.0
    } else {
        tail as f64 / baseline as f64
    };
    assert!(
        ratio <= cfg.max_growth_ratio,
        "RSS ended at {ratio:.2}x the post-warmup baseline ({:.1} MiB -> {:.1} MiB) over {} cycles (cap {:.2}x)",
        as_mib(baseline),
        as_mib(tail),
        cfg.cycles,
        cfg.max_growth_ratio,
    );
}
