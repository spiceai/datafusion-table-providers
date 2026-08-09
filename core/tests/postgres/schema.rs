use arrow::datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit};
use datafusion::catalog::TableProviderFactory;
use datafusion::common::Constraints;
use datafusion::common::ToDFSchema;
use datafusion::logical_expr::CreateExternalTable;
use datafusion::prelude::SessionContext;
use datafusion::sql::TableReference;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use crate::postgres::common;
use crate::postgres::PostgresTableProviderFactory;
use datafusion_table_providers::postgres::PostgresTableFactory;
use datafusion_table_providers::sql::db_connection_pool::dbconnection::postgresconn::PostgresVariant;
use datafusion_table_providers::sql::db_connection_pool::postgrespool::PostgresConnectionPool;
use datafusion_table_providers::util::secrets::to_secret_map;

const COMPLEX_TABLE_SQL: &str = include_str!("scripts/complex_table_pg.sql");

fn get_schema() -> SchemaRef {
    let fields = vec![
        Field::new("id", DataType::Int32, false),
        Field::new("large_id", DataType::Int64, true),
        Field::new("name", DataType::Utf8, true),
        Field::new("age", DataType::Int16, true),
        Field::new("height", DataType::Float64, true),
        Field::new("is_active", DataType::Boolean, true),
        Field::new(
            "created_at",
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            true,
        ),
        Field::new("data", DataType::Binary, true),
        Field::new(
            "tags",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            true,
        ),
    ];

    Arc::new(Schema::new(fields))
}

#[tokio::test]
async fn test_postgres_schema_inference() {
    let port = crate::get_random_port();
    let container = common::start_postgres_docker_container("postgres:latest", port, None)
        .await
        .expect("Postgres container to start");

    let factory = PostgresTableProviderFactory::new();
    let ctx = SessionContext::new();
    let table_name = "test_table";
    let schema = get_schema();

    let cmd = CreateExternalTable {
        schema: schema.to_dfschema_ref().expect("to df schema"),
        name: table_name.into(),
        location: "".to_string(),
        file_type: "".to_string(),
        table_partition_cols: vec![],
        if_not_exists: false,
        definition: None,
        order_exprs: vec![],
        unbounded: false,
        options: common::get_pg_params(port),
        constraints: Constraints::default(),
        column_defaults: HashMap::new(),
        temporary: false,
        or_replace: false,
    };
    let _ = factory
        .create(&ctx.state(), &cmd)
        .await
        .expect("table provider created");

    let postgres_pool = Arc::new(
        PostgresConnectionPool::new(to_secret_map(common::get_pg_params(port)))
            .await
            .expect("unable to create Postgres connection pool"),
    );
    let table_factory = PostgresTableFactory::new(postgres_pool);
    let table_provider = table_factory
        .table_provider(TableReference::bare(table_name))
        .await
        .expect("to create table provider");

    assert_eq!(table_provider.schema(), get_schema());

    // Tear down
    container
        .remove()
        .await
        .expect("to stop postgres container");
}

#[tokio::test]
async fn test_postgres_schema_inference_complex_types() {
    let port = crate::get_random_port();
    let container = common::start_postgres_docker_container("postgres:latest", port, None)
        .await
        .expect("Postgres container to start");

    let table_name = "example_table";

    let postgres_pool = Arc::new(
        PostgresConnectionPool::new(to_secret_map(common::get_pg_params(port)))
            .await
            .expect("unable to create Postgres connection pool"),
    );

    let pg_conn = postgres_pool
        .connect_direct()
        .await
        .expect("to connect to postgres");
    for cmd in COMPLEX_TABLE_SQL.split(";") {
        pg_conn
            .conn
            .execute(cmd, &[])
            .await
            .expect("to create table");
    }

    let table_factory = PostgresTableFactory::new(postgres_pool);
    let table_provider = table_factory
        .table_provider(TableReference::bare(table_name))
        .await
        .expect("to create table provider");

    let pretty_schema = format!("{:#?}", table_provider.schema());
    insta::assert_snapshot!(pretty_schema);

    // Tear down
    container
        .remove()
        .await
        .expect("to stop postgres container");
}

#[tokio::test]
async fn test_postgres_view_schema_inference() {
    let port = crate::get_random_port();
    let container = common::start_postgres_docker_container("postgres:latest", port, None)
        .await
        .expect("Postgres container to start");

    let postgres_pool = Arc::new(
        PostgresConnectionPool::new(to_secret_map(common::get_pg_params(port)))
            .await
            .expect("unable to create Postgres connection pool"),
    );
    let pg_conn = postgres_pool
        .connect_direct()
        .await
        .expect("to connect to postgres");

    for cmd in COMPLEX_TABLE_SQL.split(";") {
        if cmd.trim().is_empty() {
            continue;
        }
        pg_conn
            .conn
            .execute(cmd, &[])
            .await
            .expect("executing SQL from complex_table.sql");
    }

    let table_factory = PostgresTableFactory::new(postgres_pool.clone());
    let table_provider = table_factory
        .table_provider(TableReference::bare("example_view"))
        .await
        .expect("to create table provider for view");

    let pretty_schema = format!("{:#?}", table_provider.schema());
    insta::assert_snapshot!(pretty_schema);

    // Tear down
    container
        .remove()
        .await
        .expect("to stop postgres container");
}

#[tokio::test]
async fn test_postgres_materialized_view_schema_inference() {
    let port = crate::get_random_port();
    let container = common::start_postgres_docker_container("postgres:latest", port, None)
        .await
        .expect("Postgres container to start");

    let postgres_pool = Arc::new(
        PostgresConnectionPool::new(to_secret_map(common::get_pg_params(port)))
            .await
            .expect("unable to create Postgres connection pool"),
    );
    let pg_conn = postgres_pool
        .connect_direct()
        .await
        .expect("to connect to postgres");

    for cmd in COMPLEX_TABLE_SQL.split(";") {
        if cmd.trim().is_empty() {
            continue;
        }
        pg_conn
            .conn
            .execute(cmd, &[])
            .await
            .expect("executing SQL from complex_table.sql");
    }

    let table_factory = PostgresTableFactory::new(postgres_pool);
    let table_provider = table_factory
        .table_provider(TableReference::bare("example_materialized_view"))
        .await
        .expect("to create table provider for materialized view");

    let pretty_schema = format!("{:#?}", table_provider.schema());
    insta::assert_snapshot!(pretty_schema);

    // Tear down
    container
        .remove()
        .await
        .expect("to stop postgres container");
}

#[tokio::test]
async fn test_postgres_partitioned_table_schema_inference() {
    let port = crate::get_random_port();
    let container = common::start_postgres_docker_container("postgres:latest", port, None)
        .await
        .expect("Postgres container to start");

    let postgres_pool = Arc::new(
        PostgresConnectionPool::new(to_secret_map(common::get_pg_params(port)))
            .await
            .expect("unable to create Postgres connection pool"),
    );
    let pg_conn = postgres_pool
        .connect_direct()
        .await
        .expect("to connect to postgres");

    let partitioned_table_sql = r#"
        CREATE TABLE partitioned_orders (
            id INTEGER NOT NULL,
            order_date DATE NOT NULL,
            amount NUMERIC(10,2),
            description TEXT
        ) PARTITION BY RANGE (order_date);

        CREATE TABLE partitioned_orders_2024 PARTITION OF partitioned_orders
            FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

        CREATE TABLE partitioned_orders_2025 PARTITION OF partitioned_orders
            FOR VALUES FROM ('2025-01-01') TO ('2026-01-01')
    "#;

    for cmd in partitioned_table_sql.split(";") {
        if cmd.trim().is_empty() {
            continue;
        }
        pg_conn
            .conn
            .execute(cmd, &[])
            .await
            .expect("executing partitioned table SQL");
    }

    let table_factory = PostgresTableFactory::new(postgres_pool);
    let table_provider = table_factory
        .table_provider(TableReference::bare("partitioned_orders"))
        .await
        .expect("to create table provider for partitioned table");

    let expected_fields = vec![
        Field::new("id", DataType::Int32, false),
        Field::new("order_date", DataType::Date32, false),
        Field::new("amount", DataType::Decimal128(10, 2), true),
        Field::new("description", DataType::Utf8, true),
    ];
    let expected_schema = Arc::new(Schema::new(expected_fields));

    assert_eq!(table_provider.schema(), expected_schema);

    // Tear down
    container
        .remove()
        .await
        .expect("to stop postgres container");
}

/// A foreign table's schema comes from its local `pg_attribute` definition, so it
/// resolves without querying the relation's data.
///
/// Three properties are asserted together because they share one setup:
///
/// 1. Types resolve identically to an ordinary table (`relkind = 'f'` is covered by
///    the catalog query).
/// 2. An **empty** foreign table still reports its full column set. Data-based
///    inference yields `Schema::empty()` when the sample query returns no rows, so a
///    foreign table that happens to be empty at registration would otherwise register
///    with no columns at all.
/// 3. A column declared `NOT NULL` on the foreign table is reported **nullable**.
///    `PostgreSQL` never enforces that declaration against the remote data -- here
///    `note` is nullable on the remote and holds a NULL -- so treating it as non-null
///    would let a consumer assume a null-free column and return wrong results.
#[tokio::test]
async fn test_postgres_foreign_table_schema_inference() {
    let port = crate::get_random_port();
    let container = common::start_postgres_docker_container("postgres:latest", port, None)
        .await
        .expect("Postgres container to start");

    let postgres_pool = Arc::new(
        PostgresConnectionPool::new(to_secret_map(common::get_pg_params(port)))
            .await
            .expect("unable to create Postgres connection pool"),
    );
    let pg_conn = postgres_pool
        .connect_direct()
        .await
        .expect("to connect to postgres");

    let mut params = common::get_pg_params(port);
    let password = params.remove("pg_pass").expect("pg_pass should be present");

    // `loopback` points the foreign tables back at this same database, so the
    // remote side is real without needing a second container. The port is the
    // container-internal 5432, not the host-mapped one.
    let foreign_table_sql = format!(
        r#"
        CREATE TABLE remote_orders (
            id INTEGER NOT NULL,
            note TEXT,
            amount NUMERIC(10,2),
            order_date DATE,
            tags TEXT[]
        );

        INSERT INTO remote_orders (id, note, amount, order_date, tags)
            VALUES (1, NULL, 12.34, DATE '2024-01-01', ARRAY['a','b']);

        CREATE TABLE remote_empty (id INTEGER NOT NULL, label TEXT);

        CREATE EXTENSION IF NOT EXISTS postgres_fdw;

        CREATE SERVER loopback FOREIGN DATA WRAPPER postgres_fdw
            OPTIONS (host 'localhost', port '5432', dbname 'postgres');

        CREATE USER MAPPING FOR postgres SERVER loopback
            OPTIONS (user 'postgres', password '{password}');

        CREATE FOREIGN TABLE ft_orders (
            id INTEGER NOT NULL,
            note TEXT NOT NULL,
            amount NUMERIC(10,2),
            order_date DATE,
            tags TEXT[]
        ) SERVER loopback OPTIONS (table_name 'remote_orders');

        CREATE FOREIGN TABLE ft_empty (
            id INTEGER NOT NULL,
            label TEXT
        ) SERVER loopback OPTIONS (table_name 'remote_empty')
    "#
    );

    for cmd in foreign_table_sql.split(";") {
        if cmd.trim().is_empty() {
            continue;
        }
        pg_conn
            .conn
            .execute(cmd, &[])
            .await
            .expect("executing foreign table SQL");
    }

    let table_factory = PostgresTableFactory::new(postgres_pool);

    let orders = table_factory
        .table_provider(TableReference::bare("ft_orders"))
        .await
        .expect("to create table provider for foreign table");

    // Every field is nullable, including `id` and `note`, which the foreign table
    // declares NOT NULL.
    let expected_orders = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, true),
        Field::new("note", DataType::Utf8, true),
        Field::new("amount", DataType::Decimal128(10, 2), true),
        Field::new("order_date", DataType::Date32, true),
        Field::new(
            "tags",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            true,
        ),
    ]));
    assert_eq!(orders.schema(), expected_orders);

    // The remote table is empty; the schema must still be complete.
    let empty = table_factory
        .table_provider(TableReference::bare("ft_empty"))
        .await
        .expect("to create table provider for empty foreign table");

    let expected_empty = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, true),
        Field::new("label", DataType::Utf8, true),
    ]));
    assert_eq!(empty.schema(), expected_empty);

    // Tear down
    container
        .remove()
        .await
        .expect("to stop postgres container");
}

/// The `PostgreSQL` variant describes the server, so it is detected once per pool
/// rather than once per connection.
///
/// Each checkout builds a fresh `PostgresConnection`, and schema inference takes
/// one connection per table, so a per-connection memo would never be read twice:
/// a catalog resolving N tables would run N `SELECT version()` round trips. The
/// memo therefore lives on the pool and is shared with every connection it hands
/// out, including `connect_direct`.
///
/// The absence of a round trip is what has to be proven, and a poisoned memo
/// alone cannot prove it -- `get_or_init` yields the stored value whether or not
/// the query ran, so the assertion would pass either way. The server is stopped
/// instead: after that, any attempt to ask it fails, so returning a variant at
/// all is only possible from the memo.
#[tokio::test]
async fn test_postgres_variant_is_detected_once_per_pool() {
    let port = crate::get_random_port();
    let container = common::start_postgres_docker_container("postgres:latest", port, None)
        .await
        .expect("Postgres container to start");

    let postgres_pool = PostgresConnectionPool::new(to_secret_map(common::get_pg_params(port)))
        .await
        .expect("unable to create Postgres connection pool");

    // An empty memo asks the server, and a vanilla server answers `Default`.
    let detected = postgres_pool
        .connect_direct()
        .await
        .expect("to connect to postgres");
    assert_eq!(
        detected.get_variant().await.expect("variant"),
        PostgresVariant::Default,
        "a vanilla PostgreSQL server must be detected as Default"
    );

    // Take a connection while the server is still up, give it a memo that no
    // query could ever produce, then take the server away.
    let poisoned = Arc::new(OnceLock::new());
    poisoned
        .set(PostgresVariant::Redshift)
        .expect("cell starts empty");
    let memoized = postgres_pool
        .connect_direct()
        .await
        .expect("to connect to postgres")
        .with_variant_cache(Arc::clone(&poisoned));

    container.stop().await.expect("to stop postgres container");

    // With the server gone, `SELECT version()` cannot succeed. Getting an answer
    // proves the memo short-circuited the round trip rather than merely
    // overwriting its result.
    assert_eq!(
        memoized
            .get_variant()
            .await
            .expect("memo answers without the server"),
        PostgresVariant::Redshift,
        "get_variant must read the memo instead of querying the server"
    );

    // Tear down
    container
        .remove()
        .await
        .expect("to remove postgres container");
}
