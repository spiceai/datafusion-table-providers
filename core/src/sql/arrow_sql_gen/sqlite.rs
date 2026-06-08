/*
Copyright 2024 The Spice.ai OSS Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

use std::sync::Arc;

use crate::sql::arrow_sql_gen::arrow::map_data_type_to_array_builder;
use arrow::{
    array::{
        ArrayBuilder, ArrayRef, BinaryBuilder, BooleanBuilder, Float32Builder, Float64Builder,
        Int16Builder, Int32Builder, Int64Builder, Int8Builder, LargeStringBuilder, NullBuilder,
        RecordBatch, RecordBatchOptions, StringBuilder, UInt16Builder, UInt32Builder,
        UInt64Builder, UInt8Builder,
    },
    datatypes::{DataType, Field, Schema, SchemaRef},
};
use rusqlite::{types::Type, Row, Rows};
use snafu::prelude::*;

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("Failed to build record batch: {source}"))]
    FailedToBuildRecordBatch {
        source: datafusion::arrow::error::ArrowError,
    },

    #[snafu(display("No builder found for index {index}"))]
    NoBuilderForIndex { index: usize },

    #[snafu(display("Failed to downcast builder for {sqlite_type}"))]
    FailedToDowncastBuilder { sqlite_type: String },

    #[snafu(display("Failed to extract row value: {source}"))]
    FailedToExtractRowValue { source: rusqlite::Error },

    #[snafu(display("Failed to extract column name: {source}"))]
    FailedToExtractColumnName { source: rusqlite::Error },

    #[snafu(display("Failed to decode TEXT value as UTF-8: {source}"))]
    InvalidUtf8Text { source: std::str::Utf8Error },
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Converts Sqlite `Row`s to an Arrow `RecordBatch`. Assumes that all rows have the same schema and
/// sets the schema based on the first row.
///
/// # Errors
///
/// Returns an error if there is a failure in converting the rows to a `RecordBatch`.
pub fn rows_to_arrow(
    mut rows: Rows,
    num_cols: usize,
    projected_schema: Option<SchemaRef>,
) -> Result<RecordBatch> {
    let mut arrow_fields: Vec<Field> = Vec::new();
    let mut arrow_columns_builders: Vec<Box<dyn ArrayBuilder>> = Vec::new();
    let mut arrow_types: Vec<DataType> = Vec::new();
    let mut row_count = 0;

    if let Ok(Some(row)) = rows.next() {
        for i in 0..num_cols {
            let mut column_type = row
                .get_ref(i)
                .context(FailedToExtractRowValueSnafu)?
                .data_type();
            let column_name = row
                .as_ref()
                .column_name(i)
                .context(FailedToExtractColumnNameSnafu)?
                .to_string();

            // SQLite can store floating point values without a fractional component as integers.
            // Therefore, we need to verify if the column is actually a floating point type
            // by examining the projected schema.
            // Note: The same column may contain both integer and floating point values.
            // Reading values as Float is safe even if the value is stored as an integer.
            // Refer to the rusqlite type handling documentation for more details:
            // https://github.com/rusqlite/rusqlite/blob/95680270eca6f405fb51f5fbe6a214aac5fdce58/src/types/mod.rs#L21C1-L22C75
            //
            // `REAL` to integer: always returns an [`Error::InvalidColumnType`](crate::Error::InvalidColumnType) error.
            // `INTEGER` to float: casts using `as` operator. Never fails.
            // `REAL` to float: casts using `as` operator. Never fails.

            // Decimal columns use Utf8 decoding regardless of the first row's
            // storage class (see `to_sqlite_decoding_type`), so skip the
            // Integer→Real promotion for them — it's only needed for float targets.
            if column_type == Type::Integer {
                if let Some(projected_schema) = projected_schema.as_ref() {
                    match projected_schema.fields[i].data_type() {
                        DataType::Float16 | DataType::Float32 | DataType::Float64 => {
                            column_type = Type::Real;
                        }
                        _ => {}
                    }
                }
            }

            let data_type = match &projected_schema {
                Some(schema) => {
                    to_sqlite_decoding_type(schema.fields()[i].data_type(), &column_type)
                }
                None => map_column_type_to_data_type(column_type),
            };

            arrow_types.push(data_type.clone());
            arrow_columns_builders.push(map_data_type_to_array_builder(&data_type));
            arrow_fields.push(Field::new(column_name, data_type, true));
        }

        add_row_to_builders(row, &arrow_types, &mut arrow_columns_builders)?;
        row_count += 1;
    };

    while let Ok(Some(row)) = rows.next() {
        add_row_to_builders(row, &arrow_types, &mut arrow_columns_builders)?;
        row_count += 1;
    }

    let mut columns = arrow_columns_builders
        .into_iter()
        .map(|mut b| b.finish())
        .collect::<Vec<ArrayRef>>();

    // Cast columns whose decode type differs from the projected schema type
    // (e.g. Utf8 → Decimal128 for decimal columns read as text).
    if let Some(ref projected_schema) = projected_schema {
        for (i, target_field) in projected_schema.fields().iter().enumerate() {
            if arrow_fields[i].data_type() != target_field.data_type() {
                columns[i] = arrow::compute::cast(&columns[i], target_field.data_type())
                    .context(FailedToBuildRecordBatchSnafu)?;
                arrow_fields[i] = Field::new(
                    arrow_fields[i].name().clone(),
                    target_field.data_type().clone(),
                    arrow_fields[i].is_nullable(),
                );
            }
        }
    }

    let options = &RecordBatchOptions::new().with_row_count(Some(row_count));
    match RecordBatch::try_new_with_options(Arc::new(Schema::new(arrow_fields)), columns, options) {
        Ok(record_batch) => Ok(record_batch),
        Err(e) => Err(e).context(FailedToBuildRecordBatchSnafu),
    }
}

fn to_sqlite_decoding_type(data_type: &DataType, sqlite_type: &Type) -> DataType {
    if *sqlite_type == Type::Text {
        // Text is a special case as it can represent different types while correctly decoded to
        // desired Arrow type during additional type casting step.
        return DataType::Utf8;
    }
    // Other SQLite types are Integer, Real, Blob, Null are safe to decode based on target Arrow type
    match data_type {
        DataType::Null => DataType::Null,
        DataType::Int8 => DataType::Int8,
        DataType::Int16 => DataType::Int16,
        DataType::Int32 => DataType::Int32,
        DataType::Int64 => DataType::Int64,
        DataType::UInt8 => DataType::UInt8,
        DataType::UInt16 => DataType::UInt16,
        DataType::UInt32 => DataType::UInt32,
        DataType::UInt64 => DataType::UInt64,
        DataType::Boolean => DataType::Boolean,
        DataType::Float16 => DataType::Float16,
        DataType::Float32 => DataType::Float32,
        DataType::Float64 => DataType::Float64,
        DataType::Utf8 => DataType::Utf8,
        DataType::LargeUtf8 => DataType::LargeUtf8,
        DataType::Binary | DataType::LargeBinary | DataType::FixedSizeBinary(_) => DataType::Binary,
        // Decode decimals as Utf8 so that every SQLite storage class
        // (INTEGER, REAL, TEXT) is accepted — `sqlite3_column_text()` handles
        // them all.  The caller casts Utf8 → Decimal after all rows are read.
        DataType::Decimal128(_, _) | DataType::Decimal256(_, _) => DataType::Utf8,
        DataType::Duration(_) => DataType::Int64,

        // Timestamp, Date32, Date64, Time32, Time64, List, Struct, Union, Dictionary, Map
        _ => DataType::Utf8,
    }
}

macro_rules! append_value {
    ($builder:expr, $row:expr, $index:expr, $type:ty, $builder_type:ty, $sqlite_type:expr) => {{
        let Some(builder) = $builder.as_any_mut().downcast_mut::<$builder_type>() else {
            FailedToDowncastBuilderSnafu {
                sqlite_type: format!("{}", $sqlite_type),
            }
            .fail()?
        };
        let value: Option<$type> = $row.get($index).context(FailedToExtractRowValueSnafu)?;
        match value {
            Some(value) => builder.append_value(value),
            None => builder.append_null(),
        }
    }};
}

fn add_row_to_builders(
    row: &Row,
    arrow_types: &[DataType],
    arrow_columns_builders: &mut [Box<dyn ArrayBuilder>],
) -> Result<()> {
    for (i, arrow_type) in arrow_types.iter().enumerate() {
        let Some(builder) = arrow_columns_builders.get_mut(i) else {
            return NoBuilderForIndexSnafu { index: i }.fail();
        };

        match *arrow_type {
            DataType::Null => {
                let Some(builder) = builder.as_any_mut().downcast_mut::<NullBuilder>() else {
                    return FailedToDowncastBuilderSnafu {
                        sqlite_type: format!("{}", Type::Null),
                    }
                    .fail();
                };
                builder.append_null();
            }
            DataType::Int8 => append_value!(builder, row, i, i8, Int8Builder, Type::Integer),
            DataType::Int16 => append_value!(builder, row, i, i16, Int16Builder, Type::Integer),
            DataType::Int32 => append_value!(builder, row, i, i32, Int32Builder, Type::Integer),
            DataType::Int64 => append_value!(builder, row, i, i64, Int64Builder, Type::Integer),
            DataType::UInt8 => append_value!(builder, row, i, u8, UInt8Builder, Type::Integer),
            DataType::UInt16 => append_value!(builder, row, i, u16, UInt16Builder, Type::Integer),
            DataType::UInt32 => append_value!(builder, row, i, u32, UInt32Builder, Type::Integer),
            DataType::UInt64 => {
                // rusqlite 0.40 dropped the `u64` `FromSql` impl (SQLite integers are
                // i64); read as i64 and reinterpret the bits as u64.
                let Some(builder) = builder.as_any_mut().downcast_mut::<UInt64Builder>() else {
                    return FailedToDowncastBuilderSnafu {
                        sqlite_type: format!("{}", Type::Integer),
                    }
                    .fail();
                };
                // rusqlite 0.40 removed FromSql for u64 (SQLite INTEGER is signed 64-bit).
                // Read as i64 and convert; negative values are out of range for u64.
                let value: Option<i64> = row.get(i).context(FailedToExtractRowValueSnafu)?;
                match value {
                    Some(v) => {
                        let u = u64::try_from(v)
                            .map_err(|_| rusqlite::Error::IntegralValueOutOfRange(i, v))
                            .context(FailedToExtractRowValueSnafu)?;
                        builder.append_value(u);
                    }
                    None => builder.append_null(),
                }
            }

            DataType::Boolean => {
                append_value!(builder, row, i, bool, BooleanBuilder, Type::Integer)
            }

            DataType::Float32 => append_value!(builder, row, i, f32, Float32Builder, Type::Real),
            DataType::Float64 => append_value!(builder, row, i, f64, Float64Builder, Type::Real),

            // Use get_ref() instead of get::<String>() so that any SQLite
            // storage class (INTEGER, REAL, TEXT) is accepted — rusqlite's
            // FromSql<String> rejects non-TEXT cells in 0.40+.
            DataType::Utf8 => {
                let Some(builder) = builder.as_any_mut().downcast_mut::<StringBuilder>() else {
                    return FailedToDowncastBuilderSnafu {
                        sqlite_type: Type::Text.to_string(),
                    }
                    .fail();
                };
                match row.get_ref(i).context(FailedToExtractRowValueSnafu)? {
                    rusqlite::types::ValueRef::Null => builder.append_null(),
                    rusqlite::types::ValueRef::Integer(v) => builder.append_value(v.to_string()),
                    rusqlite::types::ValueRef::Real(v) => builder.append_value(v.to_string()),
                    rusqlite::types::ValueRef::Text(v) => builder.append_value(
                        std::str::from_utf8(v).context(InvalidUtf8TextSnafu)?,
                    ),
                    rusqlite::types::ValueRef::Blob(_) => builder.append_null(),
                }
            }
            DataType::LargeUtf8 => {
                let Some(builder) = builder.as_any_mut().downcast_mut::<LargeStringBuilder>()
                else {
                    return FailedToDowncastBuilderSnafu {
                        sqlite_type: Type::Text.to_string(),
                    }
                    .fail();
                };
                match row.get_ref(i).context(FailedToExtractRowValueSnafu)? {
                    rusqlite::types::ValueRef::Null => builder.append_null(),
                    rusqlite::types::ValueRef::Integer(v) => builder.append_value(v.to_string()),
                    rusqlite::types::ValueRef::Real(v) => builder.append_value(v.to_string()),
                    rusqlite::types::ValueRef::Text(v) => builder.append_value(
                        std::str::from_utf8(v).context(InvalidUtf8TextSnafu)?,
                    ),
                    rusqlite::types::ValueRef::Blob(_) => builder.append_null(),
                }
            }

            DataType::Binary => append_value!(builder, row, i, Vec<u8>, BinaryBuilder, Type::Blob),
            _ => {
                unimplemented!("Unsupported data type {arrow_type} for column index {i}")
            }
        }
    }

    Ok(())
}

fn map_column_type_to_data_type(column_type: Type) -> DataType {
    match column_type {
        Type::Null => DataType::Null,
        Type::Integer => DataType::Int64,
        Type::Real => DataType::Float64,
        Type::Text => DataType::Utf8,
        Type::Blob => DataType::Binary,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, AsArray, Decimal128Array};
    use rusqlite::Connection;

    /// SQLite NUMERIC affinity stores values with per-cell storage classes:
    /// integer-valued decimals as INTEGER, fractional ones as REAL, and
    /// high-precision values as TEXT.  `rows_to_arrow` must handle all three
    /// in the same column without "Invalid column type Text" errors.
    #[test]
    fn test_decimal_mixed_storage_classes() {
        let conn = Connection::open_in_memory().expect("open in-memory db");

        // decimal(10,2) gets NUMERIC affinity — SQLite picks storage class per cell.
        conn.execute_batch(
            "CREATE TABLE dec_test (id INTEGER PRIMARY KEY, val decimal(10,2));
             INSERT INTO dec_test VALUES (1, 1.11);   -- stored as REAL
             INSERT INTO dec_test VALUES (2, NULL);    -- NULL
             INSERT INTO dec_test VALUES (3, 99.99);   -- stored as REAL
             INSERT INTO dec_test VALUES (4, 0);       -- stored as INTEGER
             INSERT INTO dec_test VALUES (5, 2);       -- stored as INTEGER
             INSERT INTO dec_test VALUES (6, '12345678.99'); -- stored as TEXT",
        )
        .expect("setup table");

        let projected_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("val", DataType::Decimal128(10, 2), true),
        ]));

        let mut stmt = conn
            .prepare("SELECT id, val FROM dec_test ORDER BY id")
            .expect("prepare");
        let column_count = stmt.column_count();
        let rows = stmt.query([]).expect("query");

        let batch =
            rows_to_arrow(rows, column_count, Some(projected_schema)).expect("rows_to_arrow");

        assert_eq!(batch.num_rows(), 6);

        let dec_col = batch
            .column(1)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("should be Decimal128Array");

        assert_eq!(dec_col.precision(), 10);
        assert_eq!(dec_col.scale(), 2);
        assert_eq!(dec_col.value(0), 111); // 1.11
        assert!(dec_col.is_null(1)); // NULL
        assert_eq!(dec_col.value(2), 9999); // 99.99
        assert_eq!(dec_col.value(3), 0); // 0
        assert_eq!(dec_col.value(4), 200); // 2.00
        assert_eq!(dec_col.value(5), 1234567899); // 12345678.99
    }
}
