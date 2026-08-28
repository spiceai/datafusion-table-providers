//! Memory-stability stress test for the DuckDB write path an accelerated
//! dataset uses when it refreshes by upserting on a primary key while a
//! retention delete trims rows out from under it, and an accelerated view
//! repeatedly re-reads the result.
//!
//! The loop mirrors what a refresh actually does, in the same order:
//!
//! 1. **Upsert** — `insert_into(.., InsertOp::Append)` on a writer configured
//!    with `on_conflict: upsert`. Internally this registers the incoming
//!    batches as an FFI arrow scan view (`__scan_<table>_<ts>`), runs
//!    `INSERT INTO <t> SELECT * FROM <view> ON CONFLICT (<pk>) DO UPDATE SET ..`,
//!    drops the view, `ANALYZE`s, and commits.
//! 2. **Retention** — a `DELETE FROM <t> WHERE <predicate>` issued as
//!    `delete_from(filters)`, the same call a parsed retention SQL statement
//!    lowers to. It runs right after the write, as it does in a refresh.
//! 3. **View refresh** — the dataset is scanned back out and written into a
//!    second, independently file-backed table with `InsertOp::Overwrite`,
//!    which is how an accelerated view over an accelerated dataset refreshes.
//!
//! Each cycle is one refresh. The test asserts that resident memory reaches a
//! plateau and stays there: DuckDB's buffer pool is expected to grow up to its
//! `memory_limit` early, so the baseline is taken *after* a warmup and the
//! assertion is about growth beyond that plateau.
//!
//! Everything about the shape of the workload is tunable from the environment
//! so a failing configuration can be searched for without recompiling — row
//! count, key space, table width, payload size, cadences, and the thresholds
//! themselves. Defaults are sized to run in about a minute; see the constants
//! below.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use datafusion::arrow::array::{
    ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray, TimestampMicrosecondArray,
};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit};
use datafusion::catalog::TableProviderFactory;
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

/// Refresh cycles to run. Each is one upsert, plus a retention delete and a
/// view refresh on their own cadences.
const DEFAULT_CYCLES: usize = 40;
/// Rows written per upsert cycle.
const DEFAULT_ROWS_PER_CYCLE: usize = 20_000;
/// Distinct primary keys the workload cycles through. Smaller than
/// `cycles * rows_per_cycle`, so most rows land on the `DO UPDATE` path
/// rather than inserting.
const DEFAULT_KEY_SPACE: usize = 200_000;
/// Extra columns appended to the base schema, alternating string and float.
/// Raise this to test a wide table.
const DEFAULT_EXTRA_COLUMNS: usize = 8;
/// Bytes in the variable-width payload column of each row.
const DEFAULT_PAYLOAD_BYTES: usize = 64;
/// Run the retention delete every N cycles.
const DEFAULT_RETENTION_EVERY: usize = 4;
/// Run the view refresh every N cycles.
const DEFAULT_VIEW_EVERY: usize = 2;
/// Cycles to run before the memory baseline is taken. DuckDB's buffer pool
/// and the allocator's arenas both need to reach steady state first.
const DEFAULT_WARMUP_CYCLES: usize = 8;
/// `memory_limit` given to the DuckDB instances.
const DEFAULT_MEMORY_LIMIT: &str = "1GB";
/// Rows older than this (seconds) and flagged deleted are trimmed by retention.
const DEFAULT_RETENTION_WINDOW_SECS: i64 = 900;
/// Allowed absolute RSS growth past the post-warmup baseline.
const DEFAULT_MAX_GROWTH_MB: u64 = 512;
/// Allowed relative RSS growth past the post-warmup baseline.
const DEFAULT_MAX_GROWTH_RATIO: f64 = 1.5;

#[derive(Debug, Clone)]
struct StressConfig {
    cycles: usize,
    rows_per_cycle: usize,
    key_space: usize,
    extra_columns: usize,
    payload_bytes: usize,
    retention_every: usize,
    view_every: usize,
    warmup_cycles: usize,
    memory_limit: String,
    retention_window_secs: i64,
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
            view_every: env_usize("DUCKDB_MEM_TEST_VIEW_EVERY", DEFAULT_VIEW_EVERY).max(1),
            // A warmup that swallowed the whole run would leave nothing to
            // compare, so keep at least a few measured cycles.
            warmup_cycles: warmup_cycles.min(cycles.saturating_sub(3)),
            memory_limit: env_string("DUCKDB_MEM_TEST_MEMORY_LIMIT", DEFAULT_MEMORY_LIMIT),
            retention_window_secs: env_i64(
                "DUCKDB_MEM_TEST_RETENTION_WINDOW_SECS",
                DEFAULT_RETENTION_WINDOW_SECS,
            ),
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

/// Columns the view projects out of the dataset — a subset, as an accelerated
/// view's SQL usually is.
const VIEW_COLUMNS: [&str; 4] = ["id", "group_id", "payload", "processed_time"];

fn view_schema(dataset: &SchemaRef) -> SchemaRef {
    let fields = VIEW_COLUMNS
        .iter()
        .map(|name| {
            dataset
                .field_with_name(name)
                .expect("view column exists in the dataset schema")
                .clone()
        })
        .collect::<Vec<Field>>();

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
    memory_limit: &str,
    pk_indices: Vec<usize>,
    on_conflict: Option<&str>,
) -> Arc<dyn datafusion::catalog::TableProvider> {
    let mut options = HashMap::from([
        ("mode".to_string(), "file".to_string()),
        ("duckdb_open".to_string(), db_path.to_string()),
        ("memory_limit".to_string(), memory_limit.to_string()),
        // A view refresh reads one instance while writing another, so a
        // single-connection pool would serialize the two halves of the same
        // refresh against each other. Accelerators size the pool by how many
        // components share the instance; this is that sizing, fixed.
        ("connection_pool_size".to_string(), "8".to_string()),
        // Matches how an accelerated dataset is configured for throughput; the
        // view side keeps insertion order, as an ordered view refresh does.
        (
            "preserve_insertion_order".to_string(),
            if on_conflict.is_some() {
                "false".to_string()
            } else {
                "true".to_string()
            },
        ),
    ]);

    if let Some(on_conflict) = on_conflict {
        options.insert("on_conflict".to_string(), on_conflict.to_string());
    }

    let constraints = if pk_indices.is_empty() {
        Constraints::default()
    } else {
        Constraints::new_unverified(vec![Constraint::PrimaryKey(pk_indices)])
    };

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
        constraints,
        column_defaults: HashMap::new(),
        temporary: false,
    };

    DuckDBTableProviderFactory::new(duckdb::AccessMode::ReadWrite)
        .create(&ctx.state(), &cmd)
        .await
        .expect("table provider is created")
}

/// One refresh write: append a batch through the writer, upserting on conflict.
async fn upsert_cycle(
    ctx: &SessionContext,
    table: &Arc<dyn datafusion::catalog::TableProvider>,
    batch: RecordBatch,
) {
    let schema = batch.schema();
    let source = MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None)
        .expect("memory source for the refresh batch");

    let plan = table
        .insert_into(&ctx.state(), source, InsertOp::Append)
        .await
        .expect("insert plan is built");

    collect(plan, ctx.task_ctx())
        .await
        .expect("refresh write completes");
}

/// The retention pass: delete soft-deleted rows older than the window. This is
/// the same call a parsed `retention_sql` DELETE lowers to.
async fn retention_cycle(
    ctx: &SessionContext,
    table: &Arc<dyn datafusion::catalog::TableProvider>,
    cfg: &StressConfig,
    now_us: i64,
) -> u64 {
    let cutoff = now_us - cfg.retention_window_secs * 1_000_000;
    let filters: Vec<Expr> = vec![col("deleted").eq(lit("true")).and(
        col("processed_time").lt(lit(ScalarValue::TimestampMicrosecond(
            Some(cutoff),
            Some("UTC".into()),
        ))),
    )];

    let plan = table
        .delete_from(&ctx.state(), filters)
        .await
        .expect("delete plan is built");

    let batches = collect(plan, ctx.task_ctx())
        .await
        .expect("retention delete completes");

    batches
        .first()
        .and_then(|b| {
            b.column(0)
                .as_any()
                .downcast_ref::<datafusion::arrow::array::UInt64Array>()
                .map(|a| a.value(0))
        })
        .unwrap_or(0)
}

/// The accelerated view's refresh: scan the dataset and overwrite the view's
/// own table with the result.
async fn view_refresh_cycle(
    ctx: &SessionContext,
    view_table: &Arc<dyn datafusion::catalog::TableProvider>,
    dataset_name: &str,
) {
    let sql = format!(
        "SELECT {} FROM {dataset_name} WHERE deleted = 'false' ORDER BY group_id, id",
        VIEW_COLUMNS.join(", ")
    );

    let scan = ctx
        .sql(&sql)
        .await
        .expect("view query plans")
        .create_physical_plan()
        .await
        .expect("view query builds a physical plan");

    let plan = view_table
        .insert_into(&ctx.state(), scan, InsertOp::Overwrite)
        .await
        .expect("view overwrite plan is built");

    collect(plan, ctx.task_ctx())
        .await
        .expect("view refresh completes");
}

/// What DuckDB itself thinks it is holding, per allocator tag, plus the size
/// of the database file.
///
/// RSS alone cannot say whether growth is DuckDB's buffer manager filling up
/// to its configured limit (expected) or something accumulating that should
/// have been released (not). This reads the engine's own accounting through
/// the writer's pool, so a failing run reports *where* the memory went.
fn duckdb_memory_report(table: &Arc<dyn datafusion::catalog::TableProvider>) -> Option<String> {
    let writer = table.downcast_ref::<DuckDBTableWriter>()?;
    let pool = writer.pool();
    let mut db_conn = pool.connect_sync().ok()?;
    let conn = &DuckDB::duckdb_conn(&mut db_conn).ok()?.conn;

    let mut parts = Vec::new();

    if let Ok(mut stmt) = conn.prepare(
        "SELECT tag, memory_usage_bytes FROM duckdb_memory() \
         WHERE memory_usage_bytes > 0 ORDER BY memory_usage_bytes DESC LIMIT 5",
    ) {
        if let Ok(rows) = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        }) {
            for row in rows.flatten() {
                parts.push(format!("{}={:.1}MiB", row.0, row.1 as f64 / (1024.0 * 1024.0)));
            }
        }
    }

    if let Ok(bytes) = conn.query_row(
        "SELECT database_size FROM pragma_database_size()",
        [],
        |row| row.get::<_, String>(0),
    ) {
        parts.push(format!("file={bytes}"));
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

#[rstest]
#[case::single_column(PrimaryKeyShape::SingleColumn)]
#[case::composite(PrimaryKeyShape::Composite)]
#[test_log::test(tokio::test(flavor = "multi_thread"))]
async fn upsert_with_retention_and_view_refresh_does_not_grow_memory(
    #[case] pk_shape: PrimaryKeyShape,
) {
    let cfg = StressConfig::from_env();
    let temp_dir = tempfile::tempdir().expect("temp dir is created");
    let dataset_name = format!("dataset_{}", pk_shape.label());
    let view_name = format!("view_{}", pk_shape.label());

    let dataset_path = temp_dir
        .path()
        .join(format!("{dataset_name}.duckdb"))
        .to_string_lossy()
        .to_string();
    let view_path = temp_dir
        .path()
        .join(format!("{view_name}.duckdb"))
        .to_string_lossy()
        .to_string();

    let schema = dataset_schema(cfg.extra_columns);
    let view_schema = view_schema(&schema);

    let ctx = SessionContext::new();
    let dataset = create_table(
        &ctx,
        &dataset_name,
        &schema,
        &dataset_path,
        &cfg.memory_limit,
        pk_shape.indices(),
        Some(pk_shape.on_conflict_option()),
    )
    .await;
    let view = create_table(
        &ctx,
        &view_name,
        &view_schema,
        &view_path,
        &cfg.memory_limit,
        vec![],
        None,
    )
    .await;

    ctx.register_table(dataset_name.as_str(), Arc::clone(&dataset))
        .expect("dataset is registered");

    tracing::info!(
        "starting {} cycles ({} rows/cycle, {} columns, key space {}, payload {}B, memory_limit {})",
        cfg.cycles,
        cfg.rows_per_cycle,
        schema.fields().len(),
        cfg.key_space,
        cfg.payload_bytes,
        cfg.memory_limit,
    );

    let mut rng = StdRng::seed_from_u64(0x5EED_0000 + pk_shape as u64);
    let mut high_water: i64 = 0;
    let mut samples: Vec<(usize, u64)> = Vec::with_capacity(cfg.cycles);

    for cycle in 0..cfg.cycles {
        let now_us = now_micros();
        let batch = make_batch(&schema, &mut rng, &cfg, &mut high_water, now_us);
        upsert_cycle(&ctx, &dataset, batch).await;

        let deleted = if cycle % cfg.retention_every == 0 {
            retention_cycle(&ctx, &dataset, &cfg, now_us).await
        } else {
            0
        };

        if cycle % cfg.view_every == 0 {
            view_refresh_cycle(&ctx, &view, &dataset_name).await;
        }

        let rows = count_rows(&ctx, &dataset_name).await;

        if let Some(rss) = current_rss_bytes() {
            samples.push((cycle, rss));
            tracing::info!(
                "cycle {cycle:>3}: rows={rows:>9} retention_deleted={deleted:>8} rss={:>8.1} MiB",
                as_mib(rss),
            );
            if cycle % 10 == 0 {
                if let Some(report) = duckdb_memory_report(&dataset) {
                    tracing::info!("cycle {cycle:>3}: dataset engine memory: {report}");
                }
            }
        } else {
            tracing::info!("cycle {cycle:>3}: rows={rows:>9} retention_deleted={deleted:>8}");
        }
    }

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
         memory is not reaching a plateau under upsert + retention + view refresh",
        as_mib(growth),
        as_mib(baseline),
        cfg.cycles,
        as_mib(cfg.max_growth_bytes),
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
