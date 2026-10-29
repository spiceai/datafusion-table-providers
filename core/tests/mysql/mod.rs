use datafusion::{datasource::memory::MemorySourceConfig, execution::context::SessionContext};
use datafusion_table_providers::sql::db_connection_pool::mysqlpool::{
    MySQLConnectionPool, MySQLServerIdentity,
};
use datafusion_table_providers::sql::{
    db_connection_pool::DbConnectionPool, sql_provider_datafusion::SqlTable,
};
use mysql_async::prelude::{Queryable, ToValue};
use mysql_async::PoolConstraints;
use rstest::{fixture, rstest};
use std::sync::Arc;
use std::time::Duration;

use arrow::{
    array::*,
    datatypes::{i256, DataType, Field, Schema, TimeUnit, UInt16Type},
};

use datafusion_table_providers::sql::db_connection_pool::dbconnection::AsyncDbConnection;

use crate::arrow_record_batch_gen::*;
use crate::docker::RunningContainer;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::catalog::TableProviderFactory;
use datafusion::common::{Constraints, ToDFSchema};
use datafusion::logical_expr::dml::InsertOp;
use datafusion::logical_expr::CreateExternalTable;
use datafusion::physical_plan::collect;
#[cfg(feature = "mysql-federation")]
use datafusion_federation::schema_cast::record_convert::try_cast_to;
use datafusion_table_providers::mysql::MySQLTableProviderFactory;
use secrecy::ExposeSecret;
use tokio::sync::Mutex;

mod common;

async fn test_mysql_timestamp_types(port: usize) {
    let create_table_stmt = "
        CREATE TABLE timestamp_table (
    timestamp_no_fraction TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP, 
    timestamp_one_fraction TIMESTAMP(1),  
    timestamp_two_fraction TIMESTAMP(2), 
    timestamp_three_fraction TIMESTAMP(3),                   
    timestamp_four_fraction TIMESTAMP(4),
    timestamp_five_fraction TIMESTAMP(5),
    timestamp_six_fraction TIMESTAMP(6) 
);
        ";
    let insert_table_stmt = "
INSERT INTO timestamp_table (
    timestamp_no_fraction, 
    timestamp_one_fraction, 
    timestamp_two_fraction, 
    timestamp_three_fraction, 
    timestamp_four_fraction, 
    timestamp_five_fraction, 
    timestamp_six_fraction
) 
VALUES 
(
    '2024-09-12 10:00:00',             
    '2024-09-12 10:00:00.1',
    '2024-09-12 10:00:00.12',
    '2024-09-12 10:00:00.123',
    '2024-09-12 10:00:00.1234',        
    '2024-09-12 10:00:00.12345',       
    '2024-09-12 10:00:00.123456'
);
        ";

    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "timestamp_no_fraction",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        ),
        Field::new(
            "timestamp_one_fraction",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        ),
        Field::new(
            "timestamp_two_fraction",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        ),
        Field::new(
            "timestamp_three_fraction",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        ),
        Field::new(
            "timestamp_four_fraction",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        ),
        Field::new(
            "timestamp_five_fraction",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        ),
        Field::new(
            "timestamp_six_fraction",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        ),
    ]));

    let expected_record = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(TimestampMicrosecondArray::from(vec![1_726_135_200_000_000])),
            Arc::new(TimestampMicrosecondArray::from(vec![1_726_135_200_100_000])),
            Arc::new(TimestampMicrosecondArray::from(vec![1_726_135_200_120_000])),
            Arc::new(TimestampMicrosecondArray::from(vec![1_726_135_200_123_000])),
            Arc::new(TimestampMicrosecondArray::from(vec![1_726_135_200_123_400])),
            Arc::new(TimestampMicrosecondArray::from(vec![1_726_135_200_123_450])),
            Arc::new(TimestampMicrosecondArray::from(vec![1_726_135_200_123_456])),
        ],
    )
    .expect("Failed to created arrow record batch");

    arrow_mysql_one_way(
        port,
        "timestamp_table",
        create_table_stmt,
        insert_table_stmt,
        expected_record,
        None,
    )
    .await;
}

/// Tests the MySQL TIMESTAMP with time zone override.
/// The test verifies that the TIMESTAMP type correctly adjusts to the specified time zone when retrieved from the database.
/// `TIMESTAMP` columns should be automatically converted to the specified time zone,
/// while `DATETIME` columns should remain unchanged.
async fn test_mysql_timestamp_tz_override(port: usize) {
    let create_table_stmt = "
        CREATE TABLE timestamp_tz_table (
            ts TIMESTAMP,
            dt DATETIME
        );
    ";
    // values will be inserted in UTC
    let insert_table_stmt = "
        INSERT INTO timestamp_tz_table (ts, dt)
        VALUES
            ('2024-09-12 10:00:00', '2024-09-12 10:00:00');
    ";

    let schema = Arc::new(Schema::new(vec![
        Field::new("ts", DataType::Timestamp(TimeUnit::Microsecond, None), true),
        Field::new("dt", DataType::Timestamp(TimeUnit::Microsecond, None), true),
    ]));

    // Both columns should remain unchanged as target time zone is UTC (same as insert time zone)
    let expected_utc = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(TimestampMicrosecondArray::from(vec![1_726_135_200_000_000])),
            Arc::new(TimestampMicrosecondArray::from(vec![1_726_135_200_000_000])),
        ],
    )
    .expect("Failed to created arrow record batch");

    arrow_mysql_one_way(
        port,
        "timestamp_tz_table",
        create_table_stmt,
        insert_table_stmt,
        expected_utc,
        Some("UTC"),
    )
    .await;

    // "+02:00"
    let expected_custom_tz = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            // ts: TIMESTAMP column, should be shifted +2 hours (timezone override)
            Arc::new(TimestampMicrosecondArray::from(vec![
                1_726_135_200_000_000 + 2 * 60 * 60 * 1_000_000,
            ])),
            // dt: DATETIME column, should remain unchanged
            Arc::new(TimestampMicrosecondArray::from(vec![1_726_135_200_000_000])),
        ],
    )
    .expect("Failed to created arrow record batch");

    arrow_mysql_one_way(
        port,
        "timestamp_tz_table",
        create_table_stmt,
        insert_table_stmt,
        expected_custom_tz,
        Some("+02:00"), // Override time zone to +02:00
    )
    .await;
}

async fn test_mysql_datetime_types(port: usize) {
    let create_table_stmt = "
CREATE TABLE datetime_table (
    dt0 DATETIME(0),  
    dt1 DATETIME(1), 
    dt2 DATETIME(2), 
    dt3 DATETIME(3), 
    dt4 DATETIME(4), 
    dt5 DATETIME(5), 
    dt6 DATETIME(6)  
);

        ";
    let insert_table_stmt = "
INSERT INTO datetime_table (dt0, dt1, dt2, dt3, dt4, dt5, dt6)
VALUES (
    '2024-09-12 10:00:00',
    '2024-09-12 10:00:00.1',
    '2024-09-12 10:00:00.12',
    '2024-09-12 10:00:00.123',
    '2024-09-12 10:00:00.1234',
    '2024-09-12 10:00:00.12345',
    '2024-09-12 10:00:00.123456'
);
        ";

    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "dt0",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        ),
        Field::new(
            "dt1",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        ),
        Field::new(
            "dt2",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        ),
        Field::new(
            "dt3",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        ),
        Field::new(
            "dt4",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        ),
        Field::new(
            "dt5",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        ),
        Field::new(
            "dt6",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        ),
    ]));

    let expected_record = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(TimestampMicrosecondArray::from(vec![1_726_135_200_000_000])),
            Arc::new(TimestampMicrosecondArray::from(vec![1_726_135_200_100_000])),
            Arc::new(TimestampMicrosecondArray::from(vec![1_726_135_200_120_000])),
            Arc::new(TimestampMicrosecondArray::from(vec![1_726_135_200_123_000])),
            Arc::new(TimestampMicrosecondArray::from(vec![1_726_135_200_123_400])),
            Arc::new(TimestampMicrosecondArray::from(vec![1_726_135_200_123_450])),
            Arc::new(TimestampMicrosecondArray::from(vec![1_726_135_200_123_456])),
        ],
    )
    .expect("Failed to created arrow record batch");

    arrow_mysql_one_way(
        port,
        "datetime_table",
        create_table_stmt,
        insert_table_stmt,
        expected_record,
        None,
    )
    .await;
}

async fn test_mysql_time_types(port: usize) {
    let create_table_stmt = "
CREATE TABLE time_table (
    t0 TIME(0),  
    t1 TIME(1), 
    t2 TIME(2), 
    t3 TIME(3), 
    t4 TIME(4), 
    t5 TIME(5), 
    t6 TIME(6)
);
        ";
    let insert_table_stmt = "
INSERT INTO time_table (t0, t1, t2, t3, t4, t5, t6)
VALUES 
    ('12:30:00', 
     '12:30:00.1', 
     '12:30:00.12', 
     '12:30:00.123', 
     '12:30:00.1234', 
     '12:30:00.12345', 
     '12:30:00.123456');
        ";

    let schema = Arc::new(Schema::new(vec![
        Field::new("t0", DataType::Time64(TimeUnit::Nanosecond), true),
        Field::new("t1", DataType::Time64(TimeUnit::Nanosecond), true),
        Field::new("t2", DataType::Time64(TimeUnit::Nanosecond), true),
        Field::new("t3", DataType::Time64(TimeUnit::Nanosecond), true),
        Field::new("t4", DataType::Time64(TimeUnit::Nanosecond), true),
        Field::new("t5", DataType::Time64(TimeUnit::Nanosecond), true),
        Field::new("t6", DataType::Time64(TimeUnit::Nanosecond), true),
    ]));

    let expected_record = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Time64NanosecondArray::from(vec![
                (12 * 3600 + 30 * 60) * 1_000_000_000,
            ])),
            Arc::new(Time64NanosecondArray::from(vec![
                (12 * 3600 + 30 * 60) * 1_000_000_000 + 100_000_000,
            ])),
            Arc::new(Time64NanosecondArray::from(vec![
                (12 * 3600 + 30 * 60) * 1_000_000_000 + 120_000_000,
            ])),
            Arc::new(Time64NanosecondArray::from(vec![
                (12 * 3600 + 30 * 60) * 1_000_000_000 + 123_000_000,
            ])),
            Arc::new(Time64NanosecondArray::from(vec![
                (12 * 3600 + 30 * 60) * 1_000_000_000 + 123_400_000,
            ])),
            Arc::new(Time64NanosecondArray::from(vec![
                (12 * 3600 + 30 * 60) * 1_000_000_000 + 123_450_000,
            ])),
            Arc::new(Time64NanosecondArray::from(vec![
                (12 * 3600 + 30 * 60) * 1_000_000_000 + 123_456_000,
            ])),
        ],
    )
    .expect("Failed to created arrow record batch");

    arrow_mysql_one_way(
        port,
        "time_table",
        create_table_stmt,
        insert_table_stmt,
        expected_record,
        None,
    )
    .await;
}

async fn test_mysql_enum_types(port: usize) {
    let create_table_stmt = "
CREATE TABLE enum_table (
    status ENUM('active', 'inactive', 'pending', 'suspended')
);
        ";
    let insert_table_stmt = "
INSERT INTO enum_table (status)
VALUES
(NULL),
('active'),
('inactive'),
('pending'),
('suspended'),
('inactive');
        ";

    let mut builder = StringDictionaryBuilder::<UInt16Type>::new();
    builder.append_null();
    builder.append_value("active");
    builder.append_value("inactive");
    builder.append_value("pending");
    builder.append_value("suspended");
    builder.append_value("inactive");

    let array: DictionaryArray<UInt16Type> = builder.finish();

    let schema = Arc::new(Schema::new(vec![Field::new(
        "status",
        DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
        true,
    )]));

    let expected_record = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(array)])
        .expect("Failed to created arrow dictionary array record batch");

    arrow_mysql_one_way(
        port,
        "enum_table",
        create_table_stmt,
        insert_table_stmt,
        expected_record,
        None,
    )
    .await;
}

async fn test_mysql_blob_types(port: usize) {
    let create_table_stmt = "
CREATE TABLE blobs_table (
    tinyblob_col    TINYBLOB,
    tinytext_col    TINYTEXT,
    mediumblob_col  MEDIUMBLOB,
    mediumtext_col  MEDIUMTEXT,
    blob_col        BLOB,
    text_col        TEXT,
    longblob_col    LONGBLOB,
    longtext_col    LONGTEXT
);
        ";
    let insert_table_stmt = "
INSERT INTO blobs_table (
    tinyblob_col, tinytext_col, mediumblob_col, mediumtext_col, blob_col, text_col, longblob_col, longtext_col
)
VALUES
    (
        'small_blob', 'small_text',
        'medium_blob', 'medium_text',
        'larger_blob', 'larger_text',
        'very_large_blob', 'very_large_text'
    );
        ";

    let schema = Arc::new(Schema::new(vec![
        Field::new("tinyblob_col", DataType::Binary, true),
        Field::new("tinytext_col", DataType::Utf8, true),
        Field::new("mediumblob_col", DataType::Binary, true),
        Field::new("mediumtext_col", DataType::Utf8, true),
        Field::new("blob_col", DataType::Binary, true),
        Field::new("text_col", DataType::Utf8, true),
        Field::new("longblob_col", DataType::LargeBinary, true),
        Field::new("longtext_col", DataType::LargeUtf8, true),
    ]));

    let expected_record = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(BinaryArray::from_vec(vec![b"small_blob"])),
            Arc::new(StringArray::from(vec!["small_text"])),
            Arc::new(BinaryArray::from_vec(vec![b"medium_blob"])),
            Arc::new(StringArray::from(vec!["medium_text"])),
            Arc::new(BinaryArray::from_vec(vec![b"larger_blob"])),
            Arc::new(StringArray::from(vec!["larger_text"])),
            Arc::new(LargeBinaryArray::from_vec(vec![b"very_large_blob"])),
            Arc::new(LargeStringArray::from(vec!["very_large_text"])),
        ],
    )
    .expect("Failed to created arrow record batch");

    arrow_mysql_one_way(
        port,
        "blobs_table",
        create_table_stmt,
        insert_table_stmt,
        expected_record,
        None,
    )
    .await;
}

async fn test_mysql_string_types(port: usize) {
    let create_table_stmt = "
CREATE TABLE string_table (
    name VARCHAR(255),
    data VARBINARY(255),
    fixed_name CHAR(10),
    fixed_data BINARY(10)
);
        ";
    let insert_table_stmt = "
INSERT INTO string_table (name, data, fixed_name, fixed_data)
VALUES 
('Alice', 'Alice', 'ALICE', 'abc'),
('Bob', 'Bob', 'BOB', 'bob1234567'),
('Charlie', 'Charlie', 'CHARLIE', '0123456789'),
('Dave', 'Dave', 'DAVE', 'dave000000');
        ";

    let schema = Arc::new(Schema::new(vec![
        Field::new("name", DataType::Utf8, true),
        Field::new("data", DataType::Binary, true),
        Field::new("fixed_name", DataType::Utf8, true),
        Field::new("fixed_data", DataType::Binary, true),
    ]));

    let expected_record = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["Alice", "Bob", "Charlie", "Dave"])),
            Arc::new(BinaryArray::from_vec(vec![
                b"Alice", b"Bob", b"Charlie", b"Dave",
            ])),
            Arc::new(StringArray::from(vec!["ALICE", "BOB", "CHARLIE", "DAVE"])),
            Arc::new(BinaryArray::from_vec(vec![
                b"abc\0\0\0\0\0\0\0",
                b"bob1234567",
                b"0123456789",
                b"dave000000",
            ])),
        ],
    )
    .expect("Failed to created arrow record batch");
    arrow_mysql_one_way(
        port,
        "string_table",
        create_table_stmt,
        insert_table_stmt,
        expected_record,
        None,
    )
    .await;
}

async fn test_mysql_decimal_types_to_decimal256(port: usize) {
    let create_table_stmt = "
CREATE TABLE high_precision_decimal (
    decimal_values DECIMAL(50, 10)
);
        ";
    let insert_table_stmt = "
INSERT INTO high_precision_decimal (decimal_values) VALUES
(NULL),
(1234567890123456789012345678901234567890.1234567890),
(-9876543210987654321098765432109876543210.9876543210),
(0.0000000001),
(-0.000000001),
(0);
        ";

    let schema = Arc::new(Schema::new(vec![Field::new(
        "decimal_values",
        DataType::Decimal256(50, 10),
        true,
    )]));

    let expected_record = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(
            Decimal256Array::from(vec![
                None,
                Some(
                    i256::from_string("12345678901234567890123456789012345678901234567890")
                        .unwrap(),
                ),
                Some(
                    i256::from_string("-98765432109876543210987654321098765432109876543210")
                        .unwrap(),
                ),
                Some(i256::from_string("1").unwrap()),
                Some(i256::from_string("-10").unwrap()),
                Some(i256::from_string("0").unwrap()),
            ])
            .with_precision_and_scale(50, 10)
            .expect("Failed to create decimal256 array"),
        )],
    )
    .expect("Failed to created arrow record batch");

    arrow_mysql_one_way(
        port,
        "high_precision_decimal",
        create_table_stmt,
        insert_table_stmt,
        expected_record,
        None,
    )
    .await;
}

async fn test_mysql_zero_date_type(port: usize) {
    let create_table_stmt = "
        CREATE TABLE zero_datetime_test_table (
            col_date DATE,
            col_time TIME,
            col_datetime DATETIME,
            col_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    ";

    let insert_table_stmt = "
        INSERT INTO zero_datetime_test_table (
            col_date, col_time, col_datetime, col_timestamp
        ) 
        VALUES 
        -- Real Values
        ('2023-05-15', '10:00:00', '2024-09-12 10:00:00', '2024-09-12 10:00:00'),
        -- Null Values
        (NULL, NULL, NULL, NULL),
        -- Zero Values
        ('0000-00-00', '00:00:00', '0000-00-00 00:00:00', '0000-00-00 00:00:00');
    ";

    let schema = Arc::new(Schema::new(vec![
        Field::new("col_date", DataType::Date32, true),
        Field::new("col_time", DataType::Time64(TimeUnit::Nanosecond), true),
        Field::new(
            "col_datetime",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        ),
        Field::new(
            "col_timestamp",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        ),
    ]));

    let expected_record = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Date32Array::from(vec![
                Some(19492), // '2023-05-15'
                None,        // NULL
                None,        // '0000-00-00'
            ])),
            Arc::new(Time64NanosecondArray::from(vec![
                Some(36000000000000), // '10:00:00'
                None,                 // NULL
                Some(0),              // '00:00:00'
            ])),
            Arc::new(TimestampMicrosecondArray::from(vec![
                Some(1_726_135_200_000_000), // '2024-09-12 10:00:00'
                None,                        // NULL
                None,                        // '0000-00-00 00:00:00'
            ])),
            Arc::new(TimestampMicrosecondArray::from(vec![
                Some(1_726_135_200_000_000), // '2024-09-12 10:00:00'
                None,                        // NULL
                None,                        // '0000-00-00 00:00:00'
            ])),
        ],
    )
    .expect("Failed to create expected arrow record batch");

    arrow_mysql_one_way(
        port,
        "zero_datetime_test_table",
        create_table_stmt,
        insert_table_stmt,
        expected_record,
        None,
    )
    .await;
}

async fn test_mysql_decimal_types_to_decimal128(port: usize) {
    let create_table_stmt = "
        CREATE TABLE IF NOT EXISTS decimal_table (decimal_col DECIMAL(10, 2));
        ";
    let insert_table_stmt = "
        INSERT INTO decimal_table (decimal_col) VALUES (NULL), (12);
        ";

    let schema = Arc::new(Schema::new(vec![Field::new(
        "decimal_col",
        DataType::Decimal128(10, 2),
        true,
    )]));

    let expected_record = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(
            Decimal128Array::from(vec![None, Some(i128::from(1200))])
                .with_precision_and_scale(10, 2)
                .unwrap(),
        )],
    )
    .expect("Failed to created arrow record batch");

    let _ = arrow_mysql_one_way(
        port,
        "decimal_table",
        create_table_stmt,
        insert_table_stmt,
        expected_record,
        None,
    )
    .await;
}

async fn test_mysql_nullability_constraints(port: usize) {
    let create_table_stmt = "
        CREATE TABLE nullability_table (
            id INT NOT NULL,
            name VARCHAR(50) NOT NULL,
            age INT,
            email VARCHAR(100),
            score DECIMAL(5, 2) NOT NULL
        );
    ";
    let insert_table_stmt = "
        INSERT INTO nullability_table (id, name, age, email, score)
        VALUES
        (1, 'Alice', 30, 'alice@example.com', 95.50),
        (2, 'Bob', NULL, NULL, 87.25),
        (3, 'Charlie', 25, 'charlie@example.com', 92.00);
    ";

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("age", DataType::Int32, true),
        Field::new("email", DataType::Utf8, true),
        Field::new("score", DataType::Decimal128(5, 2), false),
    ]));

    let expected_record = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["Alice", "Bob", "Charlie"])),
            Arc::new(Int32Array::from(vec![Some(30), None, Some(25)])),
            Arc::new(StringArray::from(vec![
                Some("alice@example.com"),
                None,
                Some("charlie@example.com"),
            ])),
            Arc::new(
                Decimal128Array::from(vec![i128::from(9550), i128::from(8725), i128::from(9200)])
                    .with_precision_and_scale(5, 2)
                    .unwrap(),
            ),
        ],
    )
    .expect("Failed to create expected arrow record batch");

    arrow_mysql_one_way(
        port,
        "nullability_table",
        create_table_stmt,
        insert_table_stmt,
        expected_record,
        None,
    )
    .await;
}

async fn arrow_mysql_one_way(
    port: usize,
    table_name: &str,
    create_table_stmt: &str,
    insert_table_stmt: &str,
    expected_record: RecordBatch,
    test_query_tz: Option<&str>,
) -> Vec<RecordBatch> {
    tracing::debug!("Running tests on {table_name}");

    let ctx = SessionContext::new();
    // For dataset initialization we always use UTC (default) timezone
    let pool = common::get_mysql_connection_pool(port, None)
        .await
        .expect("MySQL connection pool for test table creation should be created");

    let db_conn = pool
        .connect_direct()
        .await
        .expect("Connection should be established");

    // Disable NO_ZERO_DATE and NO_ZERO_IN_DATE (requied for `test_mysql_zero_date_type`)
    let _ = db_conn
        .execute(
            "SET SESSION sql_mode = REPLACE(REPLACE(REPLACE(@@SESSION.sql_mode, 'NO_ZERO_IN_DATE,', ''), 'NO_ZERO_DATE,', ''), 'NO_ZERO_DATE', '')",
            &[]
        )
        .await
        .expect("SQL mode should be adjusted");

    // Drop table if already exists
    let _ = db_conn
        .execute(format!("DROP TABLE IF EXISTS {table_name}").as_str(), &[])
        .await
        .expect("MySQL table should be dropped if exists");

    // Create table and insert data into mysql test_table
    let _ = db_conn
        .execute(create_table_stmt, &[])
        .await
        .expect("MySQL table should be created");

    let _ = db_conn
        .execute(insert_table_stmt, &[])
        .await
        .expect("MySQL table data should be inserted");

    // For the test query, use a new connection pool with optional time zone override
    let pool = common::get_mysql_connection_pool(port, test_query_tz)
        .await
        .expect("MySQL connection pool for test query should be created");

    // Register datafusion table, test mysql row -> arrow conversion
    let sqltable_pool: Arc<
        dyn DbConnectionPool<mysql_async::Conn, &'static (dyn ToValue + Sync)>
            + Send
            + Sync
            + 'static,
    > = Arc::new(pool);
    let table = SqlTable::new("mysql", &sqltable_pool, table_name, None)
        .await
        .expect("Table should be created");

    ctx.register_table(table_name, Arc::new(table))
        .expect("Table should be registered");

    let sql = format!("SELECT * FROM {table_name}");
    let df = ctx
        .sql(&sql)
        .await
        .expect("DataFrame should be created from query");

    let record_batch = df.collect().await.expect("RecordBatch should be collected");
    tracing::debug!(
        "MySQL returned Record Batch: {:?}",
        record_batch[0].columns()
    );

    assert_eq!(record_batch.len(), 1);
    assert_eq!(record_batch[0], expected_record);

    record_batch
}

#[allow(unused_variables)]
async fn arrow_mysql_round_trip(
    port: usize,
    arrow_record: RecordBatch,
    source_schema: SchemaRef,
    table_name: &str,
) {
    let factory = MySQLTableProviderFactory::new();
    let ctx = SessionContext::new();
    let cmd = CreateExternalTable {
        schema: Arc::new(arrow_record.schema().to_dfschema().expect("to df schema")),
        name: table_name.into(),
        locations: vec![],
        file_type: "".to_string(),
        table_partition_cols: vec![],
        if_not_exists: false,
        temporary: false,
        definition: None,
        order_exprs: vec![],
        unbounded: false,
        options: common::get_mysql_params(port, None)
            .into_iter()
            .map(|(k, v)| (k, v.expose_secret().to_string()))
            .collect(),
        constraints: Constraints::default(),
        column_defaults: Default::default(),
        or_replace: false,
    };
    let table_provider = factory
        .create(&ctx.state(), &cmd)
        .await
        .expect("table provider created");

    let ctx = SessionContext::new();
    let mem_exec = MemorySourceConfig::try_new_exec(
        &[vec![arrow_record.clone()]],
        arrow_record.schema(),
        None,
    )
    .expect("memory exec created");
    let insert_plan = table_provider
        .insert_into(&ctx.state(), mem_exec, InsertOp::Overwrite)
        .await
        .expect("insert plan created");

    let _ = collect(insert_plan, ctx.task_ctx())
        .await
        .expect("insert done");
    ctx.register_table(table_name, table_provider)
        .expect("Table should be registered");
    let sql = format!("SELECT * FROM {table_name}");
    let df = ctx
        .sql(&sql)
        .await
        .expect("DataFrame should be created from query");

    let record_batch = df.collect().await.expect("RecordBatch should be collected");
    tracing::debug!("Original Arrow Record Batch: {:?}", arrow_record.columns());
    tracing::debug!(
        "MySQL returned Record Batch: {:?}",
        record_batch[0].columns()
    );

    #[cfg(feature = "mysql-federation")]
    let casted_result =
        try_cast_to(record_batch[0].clone(), source_schema).expect("Failed to cast record batch");

    // Check results
    assert_eq!(record_batch.len(), 1);
    assert_eq!(record_batch[0].num_rows(), arrow_record.num_rows());
    assert_eq!(record_batch[0].num_columns(), arrow_record.num_columns());

    #[cfg(feature = "mysql-federation")]
    assert_eq!(arrow_record, casted_result);
}

#[derive(Debug)]
struct ContainerManager {
    port: usize,
    claimed: bool,
}

#[fixture]
#[once]
fn container_manager() -> Mutex<ContainerManager> {
    Mutex::new(ContainerManager {
        port: crate::get_random_port(),
        claimed: false,
    })
}

async fn start_mysql_container(port: usize) -> RunningContainer {
    let running_container = common::start_mysql_docker_container(port)
        .await
        .expect("MySQL container to start");

    tracing::debug!("Container started");

    running_container
}

#[rstest]
#[case::binary(get_arrow_binary_record_batch(), "binary")]
#[case::int(get_arrow_int_record_batch(), "int")]
#[case::float(get_arrow_float_record_batch(), "float")]
#[case::utf8(get_arrow_utf8_record_batch(), "utf8")]
#[case::time(get_arrow_time_record_batch(), "time")]
#[case::timestamp(get_arrow_timestamp_record_batch_without_timezone(), "timestamp")]
#[case::date(get_arrow_date_record_batch(), "date")]
#[case::struct_type(get_arrow_struct_record_batch(), "struct")]
// MySQL only supports up to 65 precision for decimal through REAL type.
#[case::decimal(get_mysql_arrow_decimal_record(), "decimal")]
#[ignore] // TODO: interval types are broken in MySQL - Interval is not available in MySQL.
#[case::interval(get_arrow_interval_record_batch(), "interval")]
#[case::duration(get_arrow_duration_record_batch(), "duration")]
#[ignore] // TODO: array types are broken in MySQL - array is not available in MySQL.
#[case::list(get_arrow_list_record_batch(), "list")]
#[case::null(get_arrow_null_record_batch(), "null")]
#[ignore]
#[case::bytea_array(get_arrow_bytea_array_record_batch(), "bytea_array")]
#[test_log::test(tokio::test)]
async fn test_arrow_mysql_roundtrip(
    container_manager: &Mutex<ContainerManager>,
    #[case] arrow_result: (RecordBatch, SchemaRef),
    #[case] table_name: &str,
) {
    let mut container_manager = container_manager.lock().await;
    if !container_manager.claimed {
        container_manager.claimed = true;
        start_mysql_container(container_manager.port).await;
    }

    arrow_mysql_round_trip(
        container_manager.port,
        arrow_result.0,
        arrow_result.1,
        table_name,
    )
    .await;
}

async fn mysql_connection_id(pool: &MySQLConnectionPool) -> u64 {
    let connection = pool.connect_direct().await.expect("connection to open");
    let mut conn = connection.conn.lock().await;
    conn.query_first("SELECT CONNECTION_ID()")
        .await
        .expect("CONNECTION_ID() to run")
        .expect("CONNECTION_ID() to return a row")
}

/// The pool options passed to `new_with_pool_opts` apply when the server is given as a
/// `connection_string` URL, where `pool_min` and `pool_max` parameters are not read.
#[test_log::test(tokio::test)]
async fn test_mysql_pool_opts_apply_to_connection_string() {
    let port = crate::get_random_port();
    let mysql_container = start_mysql_container(port).await;
    let pool = MySQLConnectionPool::new_with_pool_opts(
        common::get_mysql_connection_string_params(port),
        |pool_opts| {
            pool_opts
                .with_constraints(PoolConstraints::new(1, 1).expect("1..=1 is a valid pool size"))
                .with_abs_conn_ttl(Some(Duration::from_secs(3)))
        },
    )
    .await
    .expect("pool to be created from the connection string");

    // `pool_max` of 1: a second connection waits while the first is held. The URL alone
    // would allow 100.
    let held = pool.connect_direct().await.expect("connection to open");
    let second = tokio::time::timeout(Duration::from_millis(300), pool.connect_direct()).await;
    assert!(
        second.is_err(),
        "a second connection opened while the pool's only one was held"
    );
    drop(held);

    // Within its lifetime, the pooled connection is reused.
    let first_id = mysql_connection_id(&pool).await;
    assert_eq!(first_id, mysql_connection_id(&pool).await);

    // Past its lifetime, counted from when the pool opened it, it is replaced. The sleep is
    // the lifetime under test.
    tokio::time::sleep(Duration::from_millis(3_500)).await;
    let replaced_id = mysql_connection_id(&pool).await;
    assert_ne!(
        first_id, replaced_id,
        "the connection outlived its 3s lifetime"
    );

    mysql_container.remove().await.expect("container to stop");
}

/// Collects what a pool's new-connection observer reports.
#[derive(Clone, Default)]
struct Reports(Arc<std::sync::Mutex<Vec<MySQLServerIdentity>>>);

impl Reports {
    fn observer(&self) -> impl Fn(&MySQLServerIdentity) + Send + Sync + 'static {
        let reports = Arc::clone(&self.0);
        move |identity| reports.lock().expect("lock").push(identity.clone())
    }

    fn all(&self) -> Vec<MySQLServerIdentity> {
        self.0.lock().expect("lock").clone()
    }

    fn ids(&self) -> Vec<u64> {
        self.all()
            .iter()
            .map(|identity| u64::from(identity.connection_id))
            .collect()
    }
}

/// Creates `user`, password `pw`, with read access to `mysqldb`.
async fn create_user(port: usize, user: &str) {
    let mut admin = mysql_async::Conn::new(common::root_opts(port))
        .await
        .expect("admin connection to open");
    admin
        .query_drop(format!("CREATE USER '{user}'@'%' IDENTIFIED BY 'pw'"))
        .await
        .expect("user to be created");
    admin
        .query_drop(format!("GRANT SELECT ON mysqldb.* TO '{user}'@'%'"))
        .await
        .expect("user to be granted read access");
    admin.disconnect().await.expect("admin connection to close");
}

/// How many connections `user` has opened to the server on `port`, as the server counts them.
async fn connections_opened(port: usize, user: &str) -> u64 {
    let mut admin = mysql_async::Conn::new(common::root_opts(port))
        .await
        .expect("admin connection to open");
    let opened: Option<u64> = admin
        .exec_first(
            "SELECT TOTAL_CONNECTIONS FROM performance_schema.users WHERE USER = ?",
            (user,),
        )
        .await
        .expect("connection totals to be read");
    admin.disconnect().await.expect("admin connection to close");
    opened.unwrap_or(0)
}

/// How many prepared statements the server holds, across every session.
async fn prepared_statements(port: usize) -> u64 {
    let mut admin = mysql_async::Conn::new(common::root_opts(port))
        .await
        .expect("admin connection to open");
    // The text protocol, so that this read prepares nothing itself.
    let count: Option<(String, u64)> = admin
        .query_first("SHOW GLOBAL STATUS LIKE 'Prepared_stmt_count'")
        .await
        .expect("Prepared_stmt_count to be read");
    admin.disconnect().await.expect("admin connection to close");
    count.expect("Prepared_stmt_count to return a row").1
}

/// The server's `@@hostname`.
async fn server_hostname(port: usize) -> String {
    let mut admin = mysql_async::Conn::new(common::root_opts(port))
        .await
        .expect("admin connection to open");
    let hostname: Option<String> = admin
        .query_first("SELECT @@hostname")
        .await
        .expect("@@hostname to be read");
    admin.disconnect().await.expect("admin connection to close");
    hostname.expect("@@hostname to return a row")
}

/// Opens and closes connections to the server on `port` until the next one it opens is given
/// an id of at least `next_id`.
/// Returns the id the next connection is given if nothing else connects first.
///
/// `next_id` is what the caller knows of the server's next id, and nothing is opened when it
/// has already reached `target`: opening a connection to find out would take the target.
async fn advance_connection_ids(port: usize, mut next_id: u64, target: u64) -> u64 {
    while next_id < target {
        let conn = mysql_async::Conn::new(common::root_opts(port))
            .await
            .expect("connection to open");
        next_id = u64::from(conn.id()) + 1;
        conn.disconnect().await.expect("connection to close");
    }
    next_id
}

/// A TCP proxy on a local port whose upstream port can be switched while it runs. Each
/// connection it accepts goes to the upstream set when it is accepted, so connections already
/// open stay on their server, as behind a load-balanced hostname.
struct SwitchingProxy {
    port: u16,
    upstream: Arc<std::sync::atomic::AtomicU16>,
    accept: tokio::task::JoinHandle<()>,
}

impl SwitchingProxy {
    async fn start(upstream: usize) -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("proxy to bind");
        let port = listener.local_addr().expect("proxy address").port();
        let upstream = Arc::new(std::sync::atomic::AtomicU16::new(
            u16::try_from(upstream).expect("port fits"),
        ));
        let accept = tokio::spawn({
            let upstream = Arc::clone(&upstream);
            async move {
                while let Ok((mut client, _)) = listener.accept().await {
                    let to = upstream.load(std::sync::atomic::Ordering::SeqCst);
                    tokio::spawn(async move {
                        if let Ok(mut server) =
                            tokio::net::TcpStream::connect(("127.0.0.1", to)).await
                        {
                            let _ = tokio::io::copy_bidirectional(&mut client, &mut server).await;
                        }
                    });
                }
            }
        });
        Self {
            port,
            upstream,
            accept,
        }
    }

    fn switch_to(&self, upstream: usize) {
        self.upstream.store(
            u16::try_from(upstream).expect("port fits"),
            std::sync::atomic::Ordering::SeqCst,
        );
    }
}

impl Drop for SwitchingProxy {
    fn drop(&mut self) {
        self.accept.abort();
    }
}

/// The observer hears of each connection once, as the pool opens it, with the server it
/// reached: the connection the pool opens to test the server, and the one that replaces it
/// once its lifetime passes. It does not hear again of a connection the pool hands out again.
#[test_log::test(tokio::test)]
async fn test_mysql_new_connection_observer() {
    let port = crate::get_random_port();
    let mysql_container = start_mysql_container(port).await;
    let reports = Reports::default();
    let pool = MySQLConnectionPool::new_with_pool_opts_and_observer(
        common::get_mysql_connection_string_params(port),
        |pool_opts| {
            pool_opts
                .with_constraints(PoolConstraints::new(1, 1).expect("1..=1 is a valid pool size"))
                .with_abs_conn_ttl(Some(Duration::from_secs(3)))
        },
        reports.observer(),
    )
    .await
    .expect("pool to be created");
    // The test connection is reported as the pool opens it, before any checkout.
    assert_eq!(reports.all().len(), 1, "{:?}", reports.all());

    let first_id = mysql_connection_id(&pool).await;
    assert_eq!(first_id, mysql_connection_id(&pool).await);
    // The sleep is the lifetime under test.
    tokio::time::sleep(Duration::from_millis(3_500)).await;
    let replaced_id = mysql_connection_id(&pool).await;
    assert_ne!(first_id, replaced_id);

    let server_hostname = server_hostname(port).await;
    assert_eq!(reports.ids(), vec![first_id, replaced_id]);
    for identity in &reports.all() {
        assert_eq!(identity.configured_host, "localhost");
        assert_eq!(
            identity.configured_port,
            u16::try_from(port).expect("port fits")
        );
        // Inside the container the server listens on 3306, whatever the host port.
        assert_eq!(identity.port, 3306);
        assert_eq!(identity.hostname, server_hostname);
        assert_eq!(identity.aurora_server_id, None);
    }

    mysql_container.remove().await.expect("container to stop");
}

/// Behind one hostname, a pool's connections move to another server, whose connections can
/// take ids that the first server gave the pool's connections still open. Each connection is
/// still reported once, whatever its id: each server's own count of the pool user's
/// connections is the oracle.
#[test_log::test(tokio::test)]
async fn test_mysql_new_connection_observer_across_servers() {
    let (port_a, port_b) = (crate::get_random_port(), crate::get_random_port());
    let server_a = start_mysql_container(port_a).await;
    let server_b = start_mysql_container(port_b).await;
    let user = "observed";
    create_user(port_a, user).await;
    create_user(port_b, user).await;
    // Server A's ids start above B's, so B can be brought up to each id A gives the pool.
    let admin_b = mysql_async::Conn::new(common::root_opts(port_b))
        .await
        .expect("admin connection to open");
    let mut b_next = u64::from(admin_b.id()) + 1;
    admin_b
        .disconnect()
        .await
        .expect("admin connection to close");
    advance_connection_ids(port_a, 0, b_next + 20).await;

    let proxy = SwitchingProxy::start(port_a).await;
    let reports = Reports::default();
    let pool = MySQLConnectionPool::new_with_pool_opts_and_observer(
        common::user_params(proxy.port, user),
        |pool_opts| {
            pool_opts
                .with_constraints(PoolConstraints::new(0, 32).expect("0..=32 is a valid pool size"))
        },
        reports.observer(),
    )
    .await
    .expect("pool to be created");
    // Connections held open, so that each checkout opens a new one.
    let mut held = Vec::new();
    for _ in 0..8 {
        held.push(
            pool.connect_direct()
                .await
                .expect("connection to A to open"),
        );
    }
    let mut ids_on_a = reports.ids();
    ids_on_a.sort_unstable();

    // Bring B up to each id A gave that B has not passed, and open a connection to B. The
    // server's health check can take the id between the two; the next of A's ids is then the
    // next try.
    proxy.switch_to(port_b);
    for id in &ids_on_a {
        if b_next > *id {
            continue;
        }
        advance_connection_ids(port_b, b_next, *id).await;
        let connection = pool
            .connect_direct()
            .await
            .expect("connection to B to open");
        b_next = u64::from(connection.conn.lock().await.id()) + 1;
        held.push(connection);
    }
    drop(held);
    drop(pool);

    let reports = reports.all();
    let mut ids_by_server = Vec::new();
    for port in [port_a, port_b] {
        let hostname = server_hostname(port).await;
        let mut ids: Vec<u64> = reports
            .iter()
            .filter(|identity| identity.hostname == hostname)
            .map(|identity| u64::from(identity.connection_id))
            .collect();
        let reported = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(
            reported,
            ids.len(),
            "a connection was reported twice: {reports:?}"
        );
        let opened = connections_opened(port, user).await;
        assert_eq!(
            u64::try_from(reported).expect("count fits"),
            opened,
            "the server on port {port} saw {opened} connections from '{user}', and {reported} were reported: {reports:?}"
        );
        ids_by_server.push(ids);
    }
    // Non-vacuous: server B gave the pool an id that A had given a connection still open.
    assert!(
        ids_by_server[1]
            .iter()
            .any(|id| ids_by_server[0].contains(id)),
        "no connection to B took an id A had given, so this test did not exercise reused ids: {ids_by_server:?}"
    );

    server_a.remove().await.expect("container to stop");
    server_b.remove().await.expect("container to stop");
}

/// The pool's first checkout gets the connection the pool opened to test the server, however
/// soon after the pool is created it starts: a new pool opens one connection, not two.
#[test_log::test(tokio::test)]
async fn test_mysql_first_checkout_reuses_the_test_connection() {
    let port = crate::get_random_port();
    let mysql_container = start_mysql_container(port).await;
    let user = "first_checkout";
    create_user(port, user).await;
    let reports = Reports::default();
    let pool = MySQLConnectionPool::new_with_pool_opts_and_observer(
        common::user_params(u16::try_from(port).expect("port fits"), user),
        |pool_opts| {
            pool_opts
                .with_constraints(PoolConstraints::new(1, 5).expect("1..=5 is a valid pool size"))
        },
        reports.observer(),
    )
    .await
    .expect("pool to be created");
    let test_connection = reports.ids();
    assert_eq!(test_connection.len(), 1, "{test_connection:?}");
    // The idle test connection, returned to the pool without a reset, holds no prepared
    // statement.
    assert_eq!(prepared_statements(port).await, 0);

    assert_eq!(mysql_connection_id(&pool).await, test_connection[0]);
    assert_eq!(reports.ids(), test_connection);
    assert_eq!(connections_opened(port, user).await, 1);

    mysql_container.remove().await.expect("container to stop");
}

/// Listens on a Unix socket on the host and forwards each connection to the next of
/// `upstreams` in turn, counting what each was sent, as a socket proxy in front of several
/// servers does.
#[cfg(unix)]
struct SocketForwarder {
    path: String,
    forwarded: Arc<Vec<std::sync::atomic::AtomicUsize>>,
    accept: tokio::task::JoinHandle<()>,
}

#[cfg(unix)]
impl SocketForwarder {
    fn start(path: &str, upstreams: Vec<u16>) -> Self {
        let _ = std::fs::remove_file(path);
        let listener = tokio::net::UnixListener::bind(path).expect("socket to bind");
        let forwarded: Arc<Vec<_>> = Arc::new(
            upstreams
                .iter()
                .map(|_| std::sync::atomic::AtomicUsize::new(0))
                .collect(),
        );
        let accept = tokio::spawn({
            let forwarded = Arc::clone(&forwarded);
            async move {
                let mut next = 0;
                while let Ok((mut client, _)) = listener.accept().await {
                    let upstream = next % upstreams.len();
                    next += 1;
                    forwarded[upstream].fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let port = upstreams[upstream];
                    tokio::spawn(async move {
                        if let Ok(mut server) =
                            tokio::net::TcpStream::connect(("127.0.0.1", port)).await
                        {
                            let _ = tokio::io::copy_bidirectional(&mut client, &mut server).await;
                        }
                    });
                }
            }
        });
        Self {
            path: path.to_string(),
            forwarded,
            accept,
        }
    }

    /// How many connections went to the upstream at `index`.
    fn forwarded(&self, index: usize) -> usize {
        self.forwarded[index].load(std::sync::atomic::Ordering::SeqCst)
    }
}

#[cfg(unix)]
impl Drop for SocketForwarder {
    fn drop(&mut self) {
        self.accept.abort();
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Asserts that no id appears twice in `ids`.
fn assert_each_once(ids: &[u64]) {
    let mut unique = ids.to_vec();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(
        unique.len(),
        ids.len(),
        "a connection was reported twice: {ids:?}"
    );
}

/// With `prefer_socket`, on by default, `mysql_async` replaces a TCP connection with one over
/// the Unix socket the server reports, and runs `after_connect` on the replacement twice. The
/// replacement is reported once. The server reports a socket path under `/tmp`, and this test
/// listens on the same path on the host and forwards it to the server's TCP port, so the
/// replacement is opened.
#[cfg(unix)]
#[test_log::test(tokio::test)]
async fn test_mysql_socket_replacement_reported_once() {
    let port = crate::get_random_port();
    let socket_path = format!("/tmp/dftp-mysql-{port}.sock");
    let mysql_container =
        common::start_mysql_docker_container_with_args(port, &[&format!("--socket={socket_path}")])
            .await
            .expect("MySQL container to start");
    let forwarder =
        SocketForwarder::start(&socket_path, vec![u16::try_from(port).expect("port fits")]);

    let reports = Reports::default();
    let pool = MySQLConnectionPool::new_with_pool_opts_and_observer(
        common::get_mysql_connection_string_params(port),
        |pool_opts| {
            pool_opts
                .with_constraints(PoolConstraints::new(1, 1).expect("1..=1 is a valid pool size"))
        },
        reports.observer(),
    )
    .await
    .expect("pool to be created");
    let id = mysql_connection_id(&pool).await;

    // Non-vacuous: the pool's connections were replaced by connections over the socket, and
    // each replacement is reported once.
    let forwarded = forwarder.forwarded(0);
    assert!(
        forwarded >= 1,
        "no connection was replaced by one over the socket"
    );
    let ids = reports.ids();
    assert_eq!(ids.len(), forwarded, "{ids:?}");
    assert!(ids.contains(&id), "{ids:?} does not include {id}");
    assert_each_once(&ids);

    drop(pool);
    drop(forwarder);
    mysql_container.remove().await.expect("container to stop");
}

/// One socket can lead to several servers, as a socket proxy does, and two of them can give
/// replacements the same id. Each replacement is reported once whichever server it reached:
/// both servers report one socket path, and the host socket there forwards connections to
/// them in turn.
#[cfg(unix)]
#[test_log::test(tokio::test)]
async fn test_mysql_socket_replacements_through_one_socket_to_two_servers() {
    let (port_a, port_b) = (crate::get_random_port(), crate::get_random_port());
    let socket_path = format!("/tmp/dftp-mysql-{port_a}.sock");
    let socket_arg = format!("--socket={socket_path}");
    let server_a = common::start_mysql_docker_container_with_args(port_a, &[&socket_arg])
        .await
        .expect("MySQL container to start");
    let server_b = common::start_mysql_docker_container_with_args(port_b, &[&socket_arg])
        .await
        .expect("MySQL container to start");
    let user = "replaced";
    for port in [port_a, port_b] {
        create_user(port, user).await;
        // Over TCP first, so that each server caches the password for the replacements,
        // which the client authenticates as over a socket.
        let warm: mysql_async::Opts = mysql_async::OptsBuilder::default()
            .ip_or_hostname("127.0.0.1")
            .tcp_port(u16::try_from(port).expect("port fits"))
            .user(Some(user))
            .pass(Some("pw"))
            .prefer_socket(false)
            .into();
        mysql_async::Conn::new(warm)
            .await
            .expect("connection to open")
            .disconnect()
            .await
            .expect("connection to close");
    }
    let forwarder = SocketForwarder::start(
        &socket_path,
        [port_a, port_b]
            .map(|port| u16::try_from(port).expect("port fits"))
            .to_vec(),
    );

    let reports = Reports::default();
    let pool = MySQLConnectionPool::new_with_pool_opts_and_observer(
        common::user_params(u16::try_from(port_a).expect("port fits"), user),
        |pool_opts| {
            pool_opts
                .with_constraints(PoolConstraints::new(0, 8).expect("0..=8 is a valid pool size"))
        },
        reports.observer(),
    )
    .await
    .expect("pool to be created");
    let mut held = Vec::new();
    for _ in 0..6 {
        held.push(pool.connect_direct().await.expect("connection to open"));
    }
    drop(held);
    drop(pool);

    let reports = reports.all();
    for (index, port) in [port_a, port_b].into_iter().enumerate() {
        let hostname = server_hostname(port).await;
        let ids: Vec<u64> = reports
            .iter()
            .filter(|identity| identity.hostname == hostname)
            .map(|identity| u64::from(identity.connection_id))
            .collect();
        // Non-vacuous: replacements reached both servers.
        let forwarded = forwarder.forwarded(index);
        assert!(
            forwarded >= 1,
            "no replacement reached the server on port {port}"
        );
        assert_eq!(
            ids.len(),
            forwarded,
            "the server on port {port} was sent {forwarded} replacements, and {} were reported: {reports:?}",
            ids.len()
        );
        assert_each_once(&ids);
    }

    drop(forwarder);
    server_a.remove().await.expect("container to stop");
    server_b.remove().await.expect("container to stop");
}

/// The test connection skips the pool's reset only the first time the pool takes it back.
/// Once used, it is reset like any other connection: a session variable set on the first
/// checkout is gone on the next.
#[test_log::test(tokio::test)]
async fn test_mysql_test_connection_is_reset_after_its_first_use() {
    let port = crate::get_random_port();
    let mysql_container = start_mysql_container(port).await;
    let pool = MySQLConnectionPool::new_with_pool_opts(
        common::get_mysql_connection_string_params(port),
        |pool_opts| {
            pool_opts
                .with_constraints(PoolConstraints::new(1, 1).expect("1..=1 is a valid pool size"))
        },
    )
    .await
    .expect("pool to be created");

    let first_id = {
        let connection = pool.connect_direct().await.expect("connection to open");
        let mut conn = connection.conn.lock().await;
        conn.query_drop("SET @left_behind = 1")
            .await
            .expect("session variable to be set");
        conn.id()
    };
    let connection = pool.connect_direct().await.expect("connection to open");
    let mut conn = connection.conn.lock().await;
    // The pool's only connection, handed out again.
    assert_eq!(conn.id(), first_id);
    let left_behind: Option<Option<u8>> = conn
        .query_first("SELECT @left_behind")
        .await
        .expect("session variable to be read");
    assert_eq!(left_behind, Some(None), "the session was not reset");
    drop(conn);
    drop(connection);

    mysql_container.remove().await.expect("container to stop");
}

#[rstest]
#[test_log::test(tokio::test)]
async fn test_mysql_arrow_oneway() {
    let port = crate::get_random_port();
    let mysql_container = start_mysql_container(port).await;

    test_mysql_timestamp_types(port).await;
    test_mysql_timestamp_tz_override(port).await;
    test_mysql_datetime_types(port).await;
    test_mysql_time_types(port).await;
    test_mysql_enum_types(port).await;
    test_mysql_blob_types(port).await;
    test_mysql_string_types(port).await;
    test_mysql_decimal_types_to_decimal128(port).await;
    test_mysql_decimal_types_to_decimal256(port).await;
    test_mysql_zero_date_type(port).await;
    test_mysql_nullability_constraints(port).await;

    mysql_container.remove().await.expect("container to stop");
}
