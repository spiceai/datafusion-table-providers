use std::sync::Arc;

use arrow::array::{Array, Decimal128Array, Float64Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion_table_providers::sql::arrow_sql_gen::postgres::rows_to_arrow;

use crate::postgres::common;

/// A pushed-down aggregate over a `bigint` column comes back from Postgres as an
/// unconstrained `numeric` named after the function — `avg`, `sum` — while the
/// plan the statement was unparsed from expects `avg(hits.UserID)` as `Float64`
/// and `sum(hits.UserID)` as `Int64`. The rows must land in those types
/// directly: read as `Decimal128(38, 20)` first, a 19-digit average needs more
/// than the 18 integer digits that scale leaves, and the whole query fails
/// (spiceai/spiceai#13785).
#[tokio::test]
async fn test_postgres_numeric_aggregate_reads_into_the_plans_type() {
    let port = crate::get_random_port();
    let container = common::start_postgres_docker_container("postgres:latest", port, None)
        .await
        .expect("Postgres container to start");
    let pool = common::get_postgres_connection_pool(port)
        .await
        .expect("connection pool");
    let conn = pool.connect_direct().await.expect("connection");

    conn.conn
        .execute("CREATE TABLE hits (\"UserID\" bigint)", &[])
        .await
        .expect("table created");
    // Postgres keeps 16 significant digits through the division, so the average
    // of these 19-digit values is itself a 19-digit integer: more than the 18
    // integer digits `Decimal128(38, 20)` leaves.
    conn.conn
        .execute(
            "INSERT INTO hits VALUES (2528953029789715792), (2528953029789715793), (2528953029789715796)",
            &[],
        )
        .await
        .expect("rows inserted");

    let plan_schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("avg(hits.UserID)", DataType::Float64, true),
        Field::new("sum(hits.UserID)", DataType::Int64, true),
    ]));
    let aggregate_sql = "SELECT avg(\"UserID\"), sum(\"UserID\") FROM hits";

    // Postgres's own rendering of the exact values is the ground truth.
    let text_rows = conn
        .conn
        .query(
            "SELECT avg(\"UserID\")::text, sum(\"UserID\")::text FROM hits",
            &[],
        )
        .await
        .expect("text query runs");
    let avg_text: String = text_rows[0].get(0);
    let sum_text: String = text_rows[0].get(1);
    assert_eq!(avg_text, "2528953029789715794");

    let rows = conn
        .conn
        .query(aggregate_sql, &[])
        .await
        .expect("aggregate query runs");
    let batch = rows_to_arrow(&rows, &Some(Arc::clone(&plan_schema))).expect(
        "a 19-digit average reads into the Float64 the plan expects instead of overflowing \
         Decimal128(38, 20) (spiceai/spiceai#13785)",
    );
    assert_eq!(batch.schema().field(0).data_type(), &DataType::Float64);
    assert_eq!(batch.schema().field(1).data_type(), &DataType::Int64);
    let avg = batch
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("Float64 column")
        .value(0);
    assert_eq!(
        avg,
        avg_text.parse::<f64>().expect("parses"),
        "the nearest f64 to what Postgres computed ({avg_text})"
    );
    let sum = batch
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("Int64 column")
        .value(0);
    assert_eq!(sum.to_string(), sum_text);

    // A sum past i64 is refused, as the local aggregate would refuse it, rather
    // than wrapped or truncated.
    conn.conn
        .execute(
            "INSERT INTO hits VALUES (2528953029789715792), (2528953029789715793)",
            &[],
        )
        .await
        .expect("rows inserted");
    let rows = conn
        .conn
        .query(aggregate_sql, &[])
        .await
        .expect("aggregate query runs");
    let err = rows_to_arrow(&rows, &Some(Arc::clone(&plan_schema)))
        .expect_err("12644765148948578966 does not fit Int64");
    let message = err.to_string();
    assert!(
        message.contains("column 'sum'")
            && message.contains("12644765148948578966")
            && message.contains("Int64"),
        "{message}"
    );

    // A decimal destination is matched by position too, so a pushed-down
    // aggregate over a declared numeric column lands on the precision and scale
    // the plan chose rather than the undeclared (38, 20).
    conn.conn
        .execute("CREATE TABLE orders (qty NUMERIC(15, 2))", &[])
        .await
        .expect("table created");
    conn.conn
        .execute("INSERT INTO orders VALUES (1.00), (2.00), (2.00)", &[])
        .await
        .expect("rows inserted");
    let decimal_plan_schema: SchemaRef = Arc::new(Schema::new(vec![Field::new(
        "avg(orders.qty)",
        DataType::Decimal128(38, 6),
        true,
    )]));
    let rows = conn
        .conn
        .query("SELECT avg(qty) FROM orders", &[])
        .await
        .expect("aggregate query runs");
    let batch = rows_to_arrow(&rows, &Some(decimal_plan_schema)).expect("decimal average reads");
    let avg = batch
        .column(0)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .expect("Decimal128 column");
    assert_eq!(avg.data_type(), &DataType::Decimal128(38, 6));
    assert_eq!(avg.value_as_string(0), "1.666667");

    // `SELECT avg(x), sum(x) AS avg`: both Postgres columns are named `avg`, and
    // the second plan field is literally called `avg`. Position keeps the
    // fractional average on its Float64 instead of the alias's Int64.
    conn.conn
        .execute("CREATE TABLE small (x bigint)", &[])
        .await
        .expect("table created");
    conn.conn
        .execute("INSERT INTO small VALUES (1), (2)", &[])
        .await
        .expect("rows inserted");
    let rows = conn
        .conn
        .query("SELECT avg(x), sum(x) AS avg FROM small", &[])
        .await
        .expect("aggregate query runs");
    assert_eq!(rows[0].columns()[0].name(), "avg");
    assert_eq!(rows[0].columns()[1].name(), "avg");
    let colliding_plan_schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("avg(small.x)", DataType::Float64, true),
        Field::new("avg", DataType::Int64, true),
    ]));
    let batch = rows_to_arrow(&rows, &Some(colliding_plan_schema))
        .expect("a fractional average is not decoded through the alias's Int64");
    let avg = batch
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("Float64 column")
        .value(0);
    assert_eq!(avg, 1.5);
    let sum = batch
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("Int64 column")
        .value(0);
    assert_eq!(sum, 3);

    // `sum(x) / count(*)` is Int64 to DataFusion (integer division) and an exact
    // `numeric` to Postgres; the quotient truncates toward zero, as the cast
    // this read replaces did, on both sides of zero.
    conn.conn
        .execute("CREATE TABLE neg (x bigint)", &[])
        .await
        .expect("table created");
    conn.conn
        .execute("INSERT INTO neg VALUES (-1), (-2)", &[])
        .await
        .expect("rows inserted");
    let quotient_plan_schema: SchemaRef = Arc::new(Schema::new(vec![Field::new(
        "sum(t.x) / count(*)",
        DataType::Int64,
        true,
    )]));
    for (table, expected) in [("small", 1_i64), ("neg", -1_i64)] {
        let rows = conn
            .conn
            .query(&format!("SELECT sum(x) / count(*) FROM {table}"), &[])
            .await
            .expect("query runs");
        let batch = rows_to_arrow(&rows, &Some(Arc::clone(&quotient_plan_schema)))
            .expect("a fractional quotient truncates into Int64 rather than being refused");
        assert_eq!(
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("Int64 column")
                .value(0),
            expected,
            "{table}"
        );
    }

    // A scale past 1000 is still a value Postgres sends.
    let rows = conn
        .conn
        .query(
            "SELECT round(5::numeric, 1001), round(5::numeric, 16383)",
            &[],
        )
        .await
        .expect("query runs");
    let wide_scale_plan_schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Float64, true),
        Field::new("b", DataType::Int64, true),
    ]));
    let batch = rows_to_arrow(&rows, &Some(wide_scale_plan_schema)).expect("wide scales decode");
    assert_eq!(
        batch
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("Float64 column")
            .value(0),
        5.0
    );
    assert_eq!(
        batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Int64 column")
            .value(0),
        5
    );

    // A value wider than `rust_decimal`'s 28-digit coefficient reaches a float
    // destination with every digit intact, rounded once.
    conn.conn
        .execute("CREATE TABLE wide (v numeric)", &[])
        .await
        .expect("table created");
    conn.conn
        .execute(
            "INSERT INTO wide VALUES (1234567890123456789012345.123456789012), (0.000000000001)",
            &[],
        )
        .await
        .expect("rows inserted");
    let text_rows = conn
        .conn
        .query("SELECT sum(v)::text FROM wide", &[])
        .await
        .expect("text query runs");
    let sum_text: String = text_rows[0].get(0);
    assert_eq!(sum_text, "1234567890123456789012345.123456789013");
    let rows = conn
        .conn
        .query("SELECT sum(v) FROM wide", &[])
        .await
        .expect("aggregate query runs");
    let float_plan_schema: SchemaRef = Arc::new(Schema::new(vec![Field::new(
        "sum(wide.v)",
        DataType::Float64,
        true,
    )]));
    let batch = rows_to_arrow(&rows, &Some(float_plan_schema)).expect("wide sum reads as Float64");
    let sum = batch
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("Float64 column")
        .value(0);
    assert_eq!(sum, sum_text.parse::<f64>().expect("parses"));

    // The same 37-digit value into a Decimal128(38, 12) it fits, every digit
    // intact — read as Decimal128(38, 20) it would not fit, and read through
    // `rust_decimal` it would have been rounded at the 28th digit.
    let decimal_plan_schema: SchemaRef = Arc::new(Schema::new(vec![Field::new(
        "sum(wide.v)",
        DataType::Decimal128(38, 12),
        true,
    )]));
    let batch = rows_to_arrow(&rows, &Some(decimal_plan_schema))
        .expect("wide sum reads as Decimal128(38, 12)");
    let sum = batch
        .column(0)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .expect("Decimal128 column");
    assert_eq!(sum.value_as_string(0), sum_text);

    container
        .remove()
        .await
        .expect("to stop postgres container");
}

/// The same read, through the executor a federated plan calls: the statement
/// the unparser would emit and the plan's own schema, exactly as
/// `VirtualExecutionPlan::execute` hands them over.
#[cfg(feature = "postgres-federation")]
#[tokio::test]
async fn test_postgres_executor_reads_a_numeric_aggregate_into_the_plans_type() {
    use datafusion_federation::sql::SQLExecutor;
    use datafusion_table_providers::postgres::DynPostgresConnectionPool;
    use datafusion_table_providers::sql::sql_provider_datafusion::SqlTable;
    use futures::TryStreamExt;

    let port = crate::get_random_port();
    let container = common::start_postgres_docker_container("postgres:latest", port, None)
        .await
        .expect("Postgres container to start");
    let pool = common::get_postgres_connection_pool(port)
        .await
        .expect("connection pool");
    {
        let conn = pool.connect_direct().await.expect("connection");
        conn.conn
            .execute("CREATE TABLE hits (\"UserID\" bigint)", &[])
            .await
            .expect("table created");
        conn.conn
            .execute(
                "INSERT INTO hits VALUES (2528953029789715792), (2528953029789715793), (2528953029789715796)",
                &[],
            )
            .await
            .expect("rows inserted");
    }
    let pool: Arc<DynPostgresConnectionPool> = Arc::new(pool);
    let table = SqlTable::new("postgres", &pool, "hits", None)
        .await
        .expect("table provider");

    let plan_schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("avg(hits.UserID)", DataType::Float64, true),
        Field::new("sum(hits.UserID)", DataType::Int64, true),
    ]));
    let stream = SQLExecutor::execute(
        &table,
        "SELECT avg(\"hits\".\"UserID\"), sum(\"hits\".\"UserID\") FROM \"hits\"",
        Arc::clone(&plan_schema),
        &[],
    )
    .expect("executor accepts the statement");
    let batches: Vec<_> = stream.try_collect().await.expect(
        "a 19-digit average reads into Float64 through the executor (spiceai/spiceai#13785)",
    );
    let batch = batches.iter().find(|b| b.num_rows() > 0).expect("one row");
    assert_eq!(batch.schema().field(0).data_type(), &DataType::Float64);
    assert_eq!(
        batch
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("Float64 column")
            .value(0),
        "2528953029789715794".parse::<f64>().expect("parses")
    );
    assert_eq!(
        batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Int64 column")
            .value(0),
        7_586_859_089_369_147_381
    );

    container
        .remove()
        .await
        .expect("to stop postgres container");
}
