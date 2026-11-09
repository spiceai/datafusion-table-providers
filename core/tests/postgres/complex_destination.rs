// Copyright 2024-2026 The Spice.ai OSS Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray};
use arrow::datatypes::{DataType, Field, Int64Type, Schema};
use datafusion_table_providers::sql::arrow_sql_gen::postgres::rows_to_arrow;

use crate::postgres::common;

#[tokio::test]
async fn duplicate_complex_result_names_keep_positional_fields() {
    let port = crate::get_random_port();
    let container = common::start_postgres_docker_container("postgres:latest", port, None)
        .await
        .expect("PostgreSQL container starts");
    let pool = common::get_postgres_connection_pool(port)
        .await
        .expect("connection pool");
    let conn = pool.connect_direct().await.expect("connection");
    let item = Arc::new(Field::new("item", DataType::Int64, true));
    let cases = [
        (
            DataType::Struct(vec![Field::new("n", DataType::Int64, true)].into()),
            r#"{"n":1}"#,
            r#"{"n":2}"#,
        ),
        (DataType::List(Arc::clone(&item)), "[1]", "[2]"),
        (DataType::LargeList(item), "[1]", "[2]"),
        (
            DataType::Map(
                Arc::new(Field::new(
                    "entries",
                    DataType::Struct(
                        vec![
                            Field::new("key", DataType::Utf8, false),
                            Field::new("value", DataType::Int64, true),
                        ]
                        .into(),
                    ),
                    false,
                )),
                false,
            ),
            r#"{"a":1}"#,
            r#"{"a":2}"#,
        ),
    ];
    for (data_type, first, second) in cases {
        for wire_type in ["json", "jsonb", "text"] {
            let schema = Arc::new(Schema::new(vec![
                Field::new("value", data_type.clone(), false)
                    .with_metadata([("side".to_string(), "left".to_string())].into()),
                Field::new("value", data_type.clone(), true)
                    .with_metadata([("side".to_string(), "right".to_string())].into()),
            ]));
            let sql = format!(
                "SELECT payload::{wire_type} AS value, \
                 CASE WHEN n = 1 THEN NULL ELSE payload::{wire_type} END AS value \
                 FROM (VALUES (1, '{first}'), (2, '{second}')) v(n, payload) ORDER BY n"
            );
            let rows = conn.conn.query(&sql, &[]).await.expect("query runs");
            let result = rows_to_arrow(&rows, &Some(Arc::clone(&schema)));
            let batch = match result {
                Ok(batch) => batch,
                Err(error) => {
                    container.remove().await.expect("remove container");
                    panic!("{sql}: {error}");
                }
            };
            assert_eq!(batch.schema(), schema, "{sql}");
            assert_eq!(batch.num_rows(), 2);
            assert_values(batch.column(0), &[Some(1), Some(2)]);
            assert_values(batch.column(1), &[None, Some(2)]);
        }
    }
    // Equal names can also describe different complex destination types.
    for wire_type in ["json", "jsonb", "text"] {
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "value",
                DataType::Struct(vec![Field::new("n", DataType::Int64, true)].into()),
                false,
            ),
            Field::new(
                "value",
                DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
                true,
            ),
        ]));
        let sql = format!(
            r#"SELECT payload::{wire_type} AS value, other::{wire_type} AS value
               FROM (VALUES (1, '{{"n":1}}', NULL), (2, '{{"n":2}}', '[2]'))
                 v(n, payload, other) ORDER BY n"#
        );
        let rows = conn
            .conn
            .query(&sql, &[])
            .await
            .expect("mixed types query runs");
        let batch = rows_to_arrow(&rows, &Some(Arc::clone(&schema)))
            .expect("duplicate names retain distinct complex types");
        assert_eq!(batch.schema(), schema);
        assert_values(batch.column(0), &[Some(1), Some(2)]);
        assert_values(batch.column(1), &[None, Some(2)]);
    }
    container.remove().await.expect("remove container");
}

fn assert_values(array: &ArrayRef, expected: &[Option<i64>]) {
    for (row, expected) in expected.iter().enumerate() {
        assert_eq!(array.is_null(row), expected.is_none());
        let Some(expected) = expected else { continue };
        let value = match array.data_type() {
            DataType::Struct(_) => array
                .as_struct()
                .column(0)
                .as_primitive::<Int64Type>()
                .value(row),
            DataType::List(_) => array
                .as_list::<i32>()
                .value(row)
                .as_primitive::<Int64Type>()
                .value(0),
            DataType::LargeList(_) => array
                .as_list::<i64>()
                .value(row)
                .as_primitive::<Int64Type>()
                .value(0),
            DataType::Map(_, _) => {
                let entries = array.as_map().value(row);
                assert_eq!(entries.column(0).as_string::<i32>().value(0), "a");
                entries.column(1).as_primitive::<Int64Type>().value(0)
            }
            other => panic!("unexpected type: {other:?}"),
        };
        assert_eq!(value, *expected);
    }
}
