//! Checks that decimal values fit the precision their Arrow type declares.
//!
//! Arrow does not check this when it imports data through the C data interface,
//! and every kernel and formatter after the import trusts the declared
//! precision: a value with more digits is printed with its trailing digits cut
//! off. DuckDB exports its `HUGEINT`, which holds up to 39 digits, as
//! `Decimal128(38, 0)`, so a 39-digit value crosses the boundary without either
//! side failing. A connector that receives Arrow from a driver checks each
//! batch with [`find_decimal_overflow`] and refuses one that fails, because a
//! wrong number is worse than an error.

use std::fmt;

use datafusion::arrow::array::{Array, AsArray, PrimitiveArray, RecordBatch};
use datafusion::arrow::datatypes::{
    ArrowNativeTypeOp, DataType, Decimal128Type, Decimal256Type, Decimal32Type, Decimal64Type,
    DecimalType,
};

/// A decimal value with more digits than its column's type allows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecimalOverflow {
    /// The column holding the value; a nested field is written as `column.field`.
    pub column: String,
    /// The value, with every digit.
    pub value: String,
    /// The type the source declared for the column, such as `Decimal128(38, 0)`.
    pub data_type: DataType,
}

impl fmt::Display for DecimalOverflow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "column '{}' holds the value {}, which has more digits than its type {} allows, so it cannot be returned exactly",
            self.column, self.value, self.data_type
        )
    }
}

impl std::error::Error for DecimalOverflow {}

/// Returns the first decimal value in `batch` with more digits than its column's
/// type allows, looking inside struct, list, map, and dictionary columns too.
///
/// A NULL slot is never reported, whatever bytes it holds.
#[must_use]
pub fn find_decimal_overflow(batch: &RecordBatch) -> Option<DecimalOverflow> {
    batch
        .schema_ref()
        .fields()
        .iter()
        .zip(batch.columns())
        .find_map(|(field, column)| find_in_array(field.name(), column.as_ref()))
}

fn find_in_array(name: &str, array: &dyn Array) -> Option<DecimalOverflow> {
    match array.data_type() {
        DataType::Decimal32(precision, scale) => find_in_decimals(
            name,
            array.as_primitive::<Decimal32Type>(),
            *precision,
            *scale,
        ),
        DataType::Decimal64(precision, scale) => find_in_decimals(
            name,
            array.as_primitive::<Decimal64Type>(),
            *precision,
            *scale,
        ),
        DataType::Decimal128(precision, scale) => find_in_decimals(
            name,
            array.as_primitive::<Decimal128Type>(),
            *precision,
            *scale,
        ),
        DataType::Decimal256(precision, scale) => find_in_decimals(
            name,
            array.as_primitive::<Decimal256Type>(),
            *precision,
            *scale,
        ),
        DataType::Struct(_) => {
            let array = array.as_struct();
            array
                .fields()
                .iter()
                .zip(array.columns())
                .find_map(|(field, child)| {
                    find_in_array(&format!("{name}.{}", field.name()), child.as_ref())
                })
        }
        DataType::List(_) => find_in_array(name, array.as_list::<i32>().values().as_ref()),
        DataType::LargeList(_) => find_in_array(name, array.as_list::<i64>().values().as_ref()),
        DataType::ListView(_) => find_in_array(name, array.as_list_view::<i32>().values().as_ref()),
        DataType::LargeListView(_) => {
            find_in_array(name, array.as_list_view::<i64>().values().as_ref())
        }
        DataType::FixedSizeList(..) => {
            find_in_array(name, array.as_fixed_size_list().values().as_ref())
        }
        DataType::Map(..) => {
            let array = array.as_map();
            find_in_array(&format!("{name}.key"), array.keys().as_ref())
                .or_else(|| find_in_array(&format!("{name}.value"), array.values().as_ref()))
        }
        DataType::Dictionary(..) => {
            find_in_array(name, array.as_any_dictionary().values().as_ref())
        }
        _ => None,
    }
}

fn find_in_decimals<T: DecimalType>(
    name: &str,
    array: &PrimitiveArray<T>,
    precision: u8,
    scale: i8,
) -> Option<DecimalOverflow> {
    // Arrow refuses a type whose precision is past the maximum, so the bound exists.
    let max = *T::MAX_FOR_EACH_PRECISION.get(usize::from(precision))?;
    let min = max.neg_wrapping();
    let outside = |value: T::Native| value.is_gt(max) || value.is_lt(min);

    // One pass over every slot, NULL or not, with no branch to break vectorization:
    // almost every batch holds only values that fit, and this is all it costs.
    let any_outside = array
        .values()
        .iter()
        .fold(false, |any, value| any | outside(*value));
    if !any_outside {
        return None;
    }

    // Something is out of range, but it may sit in a NULL slot, which holds no value.
    let value = array.iter().flatten().find(|value| outside(*value))?;
    Some(DecimalOverflow {
        column: name.to_string(),
        // Arrow's formatter keeps only as many digits as the precision it is
        // given, which is the very truncation being reported; `u8::MAX` keeps all.
        value: T::format_decimal(value, u8::MAX, scale),
        data_type: array.data_type().clone(),
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::arrow::array::{
        ArrayRef, Decimal128Array, Decimal256Array, Decimal32Array, Decimal64Array,
        DictionaryArray, FixedSizeListArray, Int32Array, ListArray, MapArray, StringArray,
        StructArray,
    };
    use datafusion::arrow::buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};
    use datafusion::arrow::datatypes::{i256, Field, Fields, Int32Type, Schema};

    use super::*;

    /// `HUGEINT`'s largest value, 2^127 - 1: 39 digits.
    const HUGEINT_MAX: i128 = i128::MAX;
    /// The largest value with 38 digits.
    const DIGITS_38: i128 = 99_999_999_999_999_999_999_999_999_999_999_999_999;

    fn batch(columns: Vec<(&str, ArrayRef)>) -> RecordBatch {
        let fields: Vec<Field> = columns
            .iter()
            .map(|(name, array)| Field::new(*name, array.data_type().clone(), true))
            .collect();
        RecordBatch::try_new(
            Arc::new(Schema::new(fields)),
            columns.into_iter().map(|(_, array)| array).collect(),
        )
        .expect("valid batch")
    }

    fn hugeint(values: Vec<Option<i128>>) -> ArrayRef {
        Arc::new(
            Decimal128Array::from(values)
                .with_precision_and_scale(38, 0)
                .expect("valid type"),
        )
    }

    #[test]
    fn a_hugeint_past_38_digits_is_reported_with_every_digit() {
        let batch = batch(vec![
            ("id", Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef),
            (
                "h",
                hugeint(vec![Some(12_345), Some(DIGITS_38), Some(HUGEINT_MAX)]),
            ),
        ]);
        let overflow = find_decimal_overflow(&batch).expect("an overflow");
        assert_eq!(
            overflow,
            DecimalOverflow {
                column: "h".to_string(),
                value: "170141183460469231731687303715884105727".to_string(),
                data_type: DataType::Decimal128(38, 0),
            }
        );
        assert_eq!(
            overflow.to_string(),
            "column 'h' holds the value 170141183460469231731687303715884105727, which has more \
             digits than its type Decimal128(38, 0) allows, so it cannot be returned exactly"
        );
    }

    #[test]
    fn the_smallest_hugeint_is_reported() {
        let batch = batch(vec![("h", hugeint(vec![Some(i128::MIN)]))]);
        assert_eq!(
            find_decimal_overflow(&batch).map(|o| o.value),
            Some("-170141183460469231731687303715884105728".to_string())
        );
    }

    #[test]
    fn values_up_to_the_precision_pass() {
        let batch = batch(vec![(
            "h",
            hugeint(vec![Some(DIGITS_38), Some(-DIGITS_38), Some(0), None]),
        )]);
        assert_eq!(find_decimal_overflow(&batch), None);
    }

    #[test]
    fn an_empty_batch_passes() {
        let batch = batch(vec![("h", hugeint(vec![]))]);
        assert_eq!(find_decimal_overflow(&batch), None);
    }

    #[test]
    fn a_null_slot_is_not_read() {
        // The slot's bytes are out of range, but its validity bit says NULL.
        let array = Decimal128Array::new(
            ScalarBuffer::from(vec![1, HUGEINT_MAX, 2]),
            Some(NullBuffer::from(vec![true, false, true])),
        )
        .with_precision_and_scale(38, 0)
        .expect("valid type");
        let batch = batch(vec![("h", Arc::new(array) as ArrayRef)]);
        assert_eq!(find_decimal_overflow(&batch), None);
    }

    #[test]
    fn a_sliced_array_reads_only_its_own_rows() {
        let array = hugeint(vec![Some(HUGEINT_MAX), Some(1), Some(2)]).slice(1, 2);
        let batch = batch(vec![("h", array)]);
        assert_eq!(find_decimal_overflow(&batch), None);
    }

    #[test]
    fn the_scale_is_kept_in_the_reported_value() {
        let array = Decimal128Array::from(vec![123_456_i128])
            .with_precision_and_scale(5, 2)
            .expect("valid type");
        let batch = batch(vec![("d", Arc::new(array) as ArrayRef)]);
        let overflow = find_decimal_overflow(&batch).expect("an overflow");
        assert_eq!(overflow.value, "1234.56");
        assert_eq!(overflow.data_type, DataType::Decimal128(5, 2));
    }

    #[test]
    fn every_decimal_width_is_checked() {
        let d32 = Decimal32Array::from(vec![1_000]).with_precision_and_scale(3, 0);
        let d64 = Decimal64Array::from(vec![1_000]).with_precision_and_scale(3, 0);
        let d256 =
            Decimal256Array::from(vec![i256::from_i128(1_000)]).with_precision_and_scale(3, 0);
        for (name, array) in [
            ("d32", Arc::new(d32.expect("valid type")) as ArrayRef),
            ("d64", Arc::new(d64.expect("valid type")) as ArrayRef),
            ("d256", Arc::new(d256.expect("valid type")) as ArrayRef),
        ] {
            let found = find_decimal_overflow(&batch(vec![(name, array)]));
            assert_eq!(
                found.map(|o| (o.column, o.value)),
                Some((name.to_string(), "1000".to_string()))
            );
        }
    }

    #[test]
    fn a_decimal_inside_a_struct_is_named_by_its_path() {
        let inner = hugeint(vec![Some(1), Some(HUGEINT_MAX)]);
        let array = StructArray::new(
            Fields::from(vec![Field::new("h", inner.data_type().clone(), true)]),
            vec![inner],
            None,
        );
        let batch = batch(vec![("s", Arc::new(array) as ArrayRef)]);
        assert_eq!(
            find_decimal_overflow(&batch).map(|o| o.column),
            Some("s.h".to_string())
        );
    }

    #[test]
    fn a_decimal_inside_a_list_is_found() {
        let values = hugeint(vec![Some(1), Some(HUGEINT_MAX)]);
        let field = Arc::new(Field::new("item", values.data_type().clone(), true));
        let list = ListArray::new(
            Arc::clone(&field),
            OffsetBuffer::from_lengths([2]),
            values.clone(),
            None,
        );
        let fixed = FixedSizeListArray::new(field, 2, values, None);
        for array in [Arc::new(list) as ArrayRef, Arc::new(fixed) as ArrayRef] {
            let batch = batch(vec![("l", array)]);
            assert_eq!(
                find_decimal_overflow(&batch).map(|o| o.value),
                Some("170141183460469231731687303715884105727".to_string())
            );
        }
    }

    #[test]
    fn a_decimal_map_value_is_found() {
        let keys: ArrayRef = Arc::new(StringArray::from(vec!["a", "b"]));
        let values = hugeint(vec![Some(1), Some(HUGEINT_MAX)]);
        let entries = StructArray::new(
            Fields::from(vec![
                Field::new("key", DataType::Utf8, false),
                Field::new("value", values.data_type().clone(), true),
            ]),
            vec![keys, values],
            None,
        );
        let map = MapArray::new(
            Arc::new(Field::new("entries", entries.data_type().clone(), false)),
            OffsetBuffer::from_lengths([2]),
            entries,
            None,
            false,
        );
        let batch = batch(vec![("m", Arc::new(map) as ArrayRef)]);
        assert_eq!(
            find_decimal_overflow(&batch).map(|o| o.column),
            Some("m.value".to_string())
        );
    }

    #[test]
    fn a_dictionary_of_decimals_is_checked() {
        let values = hugeint(vec![Some(1), Some(HUGEINT_MAX)]);
        let dictionary =
            DictionaryArray::<Int32Type>::try_new(Int32Array::from(vec![0, 1, 0]), values)
                .expect("valid dictionary");
        let batch = batch(vec![("d", Arc::new(dictionary) as ArrayRef)]);
        assert_eq!(
            find_decimal_overflow(&batch).map(|o| o.column),
            Some("d".to_string())
        );
    }
}
