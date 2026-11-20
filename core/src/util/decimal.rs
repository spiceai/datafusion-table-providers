//! Checks that decimal values fit the precision their Arrow type declares.
//!
//! Arrow does not check this when it imports data through the C data interface,
//! and every kernel and formatter after the import trusts the declared
//! precision: a value with more digits is printed with its trailing digits cut
//! off. DuckDB exports its `HUGEINT`, which holds up to 39 digits, as
//! `Decimal128(38, 0)`, so a 39-digit value crosses the boundary without either
//! side failing. A connection that receives Arrow from a driver checks each
//! batch with [`find_decimal_overflow`] and refuses one that fails, because a
//! wrong number is worse than an error.
//!
//! Only a decimal declared at its width's maximum precision is checked:
//! `Decimal32(9, _)`, `Decimal64(18, _)`, `Decimal128(38, _)` or
//! `Decimal256(76, _)`. A source exceeds a precision when it maps a type wider
//! than Arrow can hold onto the widest one this width offers, as DuckDB does
//! with `HUGEINT` and with a `SUM` it computes as one. A column a source
//! declares narrower, such as `DECIMAL(15, 2)`, holds what it says unless the
//! source sends invalid Arrow, and skipping it keeps the check off ordinary
//! decimal columns.

use std::fmt;
use std::ops::Range;

use datafusion::arrow::array::{Array, AsArray, PrimitiveArray, RecordBatch, RunArray};
use datafusion::arrow::buffer::NullBuffer;
use datafusion::arrow::datatypes::{
    i256, ArrowNativeTypeOp, DataType, Decimal128Type, Decimal256Type, Decimal32Type,
    Decimal64Type, DecimalType, Int16Type, Int32Type, Int64Type, RunEndIndexType,
};

/// A decimal value with more digits than its column's type allows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecimalOverflow {
    /// The column holding the value; a nested field is written as `column.field`.
    pub column: String,
    /// The value, with every digit.
    pub value: String,
    /// The type the source declared for the value, such as `Decimal128(38, 0)`.
    pub data_type: DataType,
}

impl fmt::Display for DecimalOverflow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "column '{}' holds the value {}, which has more digits than its type {} allows, so it cannot be returned exactly. Cast the value to VARCHAR at the source to read it",
            self.column, self.value, self.data_type
        )?;
        if self.data_type == DataType::Decimal128(38, 0) {
            write!(
                f,
                ". DuckDB exports its HUGEINT type, which holds up to 39 digits, as Decimal128(38, 0)"
            )?;
        }
        Ok(())
    }
}

impl std::error::Error for DecimalOverflow {}

impl DecimalOverflow {
    /// Names the field this value was found under, outermost last.
    fn under(mut self, field: &str) -> Self {
        self.column = if self.column.is_empty() {
            field.to_string()
        } else {
            format!("{field}.{}", self.column)
        };
        self
    }
}

/// Returns the first decimal value in `batch` with more digits than its type
/// allows, looking inside struct, list, map, dictionary, and union columns too.
///
/// Only values a reader can reach count: a NULL slot, a child under a NULL
/// parent, a list element outside every list, an unused dictionary entry, and
/// an inactive union member are never reported, whatever bytes they hold. The
/// search costs what the batch stores: a run-end-encoded array is walked by its
/// runs, never by the logical length it claims.
#[must_use]
pub fn find_decimal_overflow(batch: &RecordBatch) -> Option<DecimalOverflow> {
    batch
        .schema_ref()
        .fields()
        .iter()
        .zip(batch.columns())
        .find_map(|(field, column)| {
            // A scan of every decimal buffer, reachable or not, is all a column of
            // values that fit costs. Only a column it flags is walked row by row.
            find_in_array(column.as_ref(), Scope::AllSlots)?;
            find_in_array(column.as_ref(), Scope::Reachable(None))
                .map(|overflow| overflow.under(field.name()))
        })
}

/// Which slots of an array the search reads.
#[derive(Clone, Copy)]
enum Scope<'a> {
    /// Every slot, NULL or unreachable included: a cheap superset of the answer.
    AllSlots,
    /// Only the values a reader can reach: the slots in these sorted, disjoint
    /// ranges (every slot when `None`) that the array itself does not mark NULL.
    Reachable(Option<&'a [Range<usize>]>),
}

fn find_in_array(array: &dyn Array, scope: Scope<'_>) -> Option<DecimalOverflow> {
    match array.data_type() {
        DataType::Decimal32(..) => find_in_decimals(array.as_primitive::<Decimal32Type>(), scope),
        DataType::Decimal64(..) => find_in_decimals(array.as_primitive::<Decimal64Type>(), scope),
        DataType::Decimal128(..) => find_in_decimals(array.as_primitive::<Decimal128Type>(), scope),
        DataType::Decimal256(..) => find_in_decimals(array.as_primitive::<Decimal256Type>(), scope),
        DataType::Struct(_) => {
            let array = array.as_struct();
            // A struct row maps to the same row of each child. Without a validity
            // bitmap the struct stores nothing per row, so its reach passes through
            // unchanged rather than being walked row by row.
            let mask = array
                .nulls()
                .map(|nulls| child_scope(scope, array.len(), Some(nulls), |row| row..row + 1));
            let child_scope = mask.as_ref().map_or(scope, ChildScope::scope);
            array
                .fields()
                .iter()
                .zip(array.columns())
                .find_map(|(field, child)| {
                    find_in_array(child.as_ref(), child_scope)
                        .map(|overflow| overflow.under(field.name()))
                })
        }
        DataType::List(_) => {
            let array = array.as_list::<i32>();
            let offsets = array.value_offsets();
            let mask = child_scope(scope, array.len(), array.nulls(), |row| {
                as_index(offsets[row])..as_index(offsets[row + 1])
            });
            find_in_array(array.values().as_ref(), mask.scope())
        }
        DataType::LargeList(_) => {
            let array = array.as_list::<i64>();
            let offsets = array.value_offsets();
            let mask = child_scope(scope, array.len(), array.nulls(), |row| {
                as_index(offsets[row])..as_index(offsets[row + 1])
            });
            find_in_array(array.values().as_ref(), mask.scope())
        }
        DataType::ListView(_) => {
            let array = array.as_list_view::<i32>();
            let (offsets, sizes) = (array.offsets(), array.sizes());
            let mask = child_scope(scope, array.len(), array.nulls(), |row| {
                as_index(offsets[row])..as_index(offsets[row]).saturating_add(as_index(sizes[row]))
            });
            find_in_array(array.values().as_ref(), mask.scope())
        }
        DataType::LargeListView(_) => {
            let array = array.as_list_view::<i64>();
            let (offsets, sizes) = (array.offsets(), array.sizes());
            let mask = child_scope(scope, array.len(), array.nulls(), |row| {
                as_index(offsets[row])..as_index(offsets[row]).saturating_add(as_index(sizes[row]))
            });
            find_in_array(array.values().as_ref(), mask.scope())
        }
        DataType::FixedSizeList(_, size) => {
            let array = array.as_fixed_size_list();
            let size = as_index(*size);
            let mask = match (scope, array.nulls()) {
                // Row `r` holds child slots `r * size..(r + 1) * size`. Without a
                // validity bitmap the list stores nothing per row, so each range of
                // rows maps to its range of slots without visiting a row.
                (Scope::Reachable(reach), None) => {
                    let len = array.len();
                    let every_row = 0..len;
                    ChildScope::Reachable(merged(
                        reach
                            .unwrap_or(std::slice::from_ref(&every_row))
                            .iter()
                            .map(|rows| {
                                rows.start.min(len).saturating_mul(size)
                                    ..rows.end.min(len).saturating_mul(size)
                            }),
                    ))
                }
                (scope, nulls) => child_scope(scope, array.len(), nulls, |row| {
                    row * size..(row + 1) * size
                }),
            };
            find_in_array(array.values().as_ref(), mask.scope())
        }
        DataType::Map(..) => {
            let array = array.as_map();
            let offsets = array.value_offsets();
            let mask = child_scope(scope, array.len(), array.nulls(), |row| {
                as_index(offsets[row])..as_index(offsets[row + 1])
            });
            find_in_array(array.entries(), mask.scope())
        }
        DataType::Dictionary(..) => {
            let array = array.as_any_dictionary();
            // No values, no decimal; and Arrow's `normalized_keys` asserts there are some.
            if array.values().is_empty() {
                return None;
            }
            let keys = match scope {
                Scope::AllSlots => Vec::new(),
                Scope::Reachable(_) => array.normalized_keys(),
            };
            let mask = child_scope(scope, array.len(), array.nulls(), |row| {
                keys[row]..keys[row] + 1
            });
            find_in_array(array.values().as_ref(), mask.scope())
        }
        DataType::Union(fields, _) => {
            let array = array.as_union();
            fields.iter().find_map(|(type_id, field)| {
                let child = array.child(type_id);
                // A row reaches the member its type id names, and no other.
                let mask = child_scope(scope, array.len(), None, |row| {
                    if array.type_id(row) != type_id {
                        return 0..0;
                    }
                    let slot = array
                        .offsets()
                        .map_or(row, |offsets| as_index(offsets[row]));
                    slot..slot + 1
                });
                find_in_array(child.as_ref(), mask.scope())
                    .map(|overflow| overflow.under(field.name()))
            })
        }
        DataType::RunEndEncoded(run_ends, _) => match run_ends.data_type() {
            DataType::Int16 => find_in_runs(array.as_run::<Int16Type>(), scope),
            DataType::Int32 => find_in_runs(array.as_run::<Int32Type>(), scope),
            DataType::Int64 => find_in_runs(array.as_run::<Int64Type>(), scope),
            _ => None,
        },
        _ => None,
    }
}

/// Each row of a run-end-encoded array reads the value of the run it falls in.
fn find_in_runs<R: RunEndIndexType>(
    array: &RunArray<R>,
    scope: Scope<'_>,
) -> Option<DecimalOverflow> {
    let mask = match scope {
        Scope::AllSlots => ChildScope::AllSlots,
        // A range of rows reads the runs from the one holding its first row to the
        // one holding its last; a binary search finds each, so no row is visited.
        Scope::Reachable(reach) => {
            let len = array.len();
            let every_row = 0..len;
            ChildScope::Reachable(merged(
                reach
                    .unwrap_or(std::slice::from_ref(&every_row))
                    .iter()
                    .map(|rows| rows.start.min(len)..rows.end.min(len))
                    .filter(|rows| !rows.is_empty())
                    .map(|rows| {
                        array.get_physical_index(rows.start)
                            ..array.get_physical_index(rows.end - 1) + 1
                    }),
            ))
        }
    };
    find_in_array(array.values().as_ref(), mask.scope())
}

fn find_in_decimals<T: DecimalType>(
    array: &PrimitiveArray<T>,
    scope: Scope<'_>,
) -> Option<DecimalOverflow>
where
    T::Native: Magnitude,
{
    // Only the widest precision can be the target of a lossy mapping (module docs).
    if array.precision() < T::MAX_PRECISION {
        return None;
    }
    // Arrow refuses a type whose precision is past the maximum, so the bound exists.
    let max = *T::MAX_FOR_EACH_PRECISION.get(usize::from(array.precision()))?;
    let values = array.values();
    let index = match scope {
        // One pass with no branch to break vectorization; almost every batch ends here.
        Scope::AllSlots => {
            if !values
                .iter()
                .fold(false, |any, value| any | value.exceeds(max))
            {
                return None;
            }
            values.iter().position(|value| value.exceeds(max))?
        }
        Scope::Reachable(reach) => reachable_rows(reach, array.len())
            .find(|&row| array.is_valid(row) && values[row].exceeds(max))?,
    };
    Some(DecimalOverflow {
        column: String::new(),
        // Arrow's formatter keeps only as many digits as the precision it is
        // given, which is the very truncation being reported; `u8::MAX` keeps all.
        value: T::format_decimal(values[index], u8::MAX, array.scale()),
        data_type: array.data_type().clone(),
    })
}

/// Whether a value has more digits than the bound `max` of its precision allows.
/// A decimal's bounds are `-max..=max`, so one unsigned comparison of the
/// magnitude decides it where the integer type has one.
trait Magnitude: ArrowNativeTypeOp {
    fn exceeds(self, max: Self) -> bool;
}

macro_rules! magnitude_by_unsigned_abs {
    ($($native:ty),*) => {$(
        impl Magnitude for $native {
            #[inline]
            fn exceeds(self, max: Self) -> bool {
                self.unsigned_abs() > max.unsigned_abs()
            }
        }
    )*};
}
magnitude_by_unsigned_abs!(i32, i64, i128);

impl Magnitude for i256 {
    #[inline]
    fn exceeds(self, max: Self) -> bool {
        self.is_gt(max) || self.is_lt(max.neg_wrapping())
    }
}

/// The scope a child array is searched in, owning the ranges it may need.
enum ChildScope {
    AllSlots,
    Reachable(Vec<Range<usize>>),
}

impl ChildScope {
    fn scope(&self) -> Scope<'_> {
        match self {
            Self::AllSlots => Scope::AllSlots,
            Self::Reachable(ranges) => Scope::Reachable(Some(ranges)),
        }
    }
}

/// Every caller's parent stores something per row (offsets, sizes, keys, type
/// ids, or a validity bitmap), so walking its reachable rows costs what the
/// batch stores. A struct or fixed-size list with no bitmap stores nothing per
/// row and passes ranges through instead of calling this.
///
/// The scope for a child of a parent of `parent_len` rows with `parent_nulls`:
/// each parent row a reader reaches makes the child slots in `slots(row)`
/// reachable. Only a reachable search, which runs after the cheap scan flagged
/// the column, visits the parent's rows, and only the reachable ones.
fn child_scope(
    scope: Scope<'_>,
    parent_len: usize,
    parent_nulls: Option<&NullBuffer>,
    slots: impl Fn(usize) -> Range<usize>,
) -> ChildScope {
    match scope {
        Scope::AllSlots => ChildScope::AllSlots,
        Scope::Reachable(reach) => ChildScope::Reachable(merged(
            reachable_rows(reach, parent_len)
                .filter(|&row| parent_nulls.is_none_or(|nulls| nulls.is_valid(row)))
                .map(slots),
        )),
    }
}

/// The rows of an array of `len` rows that `reach` covers, in order.
fn reachable_rows(reach: Option<&[Range<usize>]>, len: usize) -> impl Iterator<Item = usize> + '_ {
    let every_row = reach.is_none().then_some(0..len);
    every_row
        .into_iter()
        .chain(
            reach
                .unwrap_or_default()
                .iter()
                .map(move |rows| rows.start.min(len)..rows.end.min(len)),
        )
        .flatten()
}

/// `ranges` as sorted, disjoint ranges, with empty ones dropped.
///
/// Ranges are coalesced as they arrive. Most callers produce them in order, so
/// this holds one range per run of reachable slots rather than one per row; a
/// list view's or a dictionary's may arrive out of order and are sorted at the end.
fn merged(ranges: impl Iterator<Item = Range<usize>>) -> Vec<Range<usize>> {
    let mut disjoint: Vec<Range<usize>> = Vec::new();
    let mut in_order = true;
    for range in ranges.filter(|range| !range.is_empty()) {
        match disjoint.last_mut() {
            Some(last) if range.start >= last.start && range.start <= last.end => {
                last.end = last.end.max(range.end);
            }
            Some(last) if range.start < last.start => {
                in_order = false;
                disjoint.push(range);
            }
            _ => disjoint.push(range),
        }
    }
    if in_order {
        return disjoint;
    }
    disjoint.sort_unstable_by_key(|range| range.start);
    let mut sorted: Vec<Range<usize>> = Vec::new();
    for range in disjoint {
        match sorted.last_mut() {
            Some(last) if range.start <= last.end => last.end = last.end.max(range.end),
            _ => sorted.push(range),
        }
    }
    sorted
}

/// An Arrow offset or size, which a valid array never makes negative.
fn as_index<O: TryInto<usize>>(offset: O) -> usize {
    offset.try_into().unwrap_or(0)
}
#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::arrow::array::{
        ArrayRef, Decimal128Array, Decimal256Array, Decimal32Array, Decimal64Array,
        DictionaryArray, FixedSizeListArray, Int32Array, Int64Array, LargeListArray, ListArray,
        ListViewArray, MapArray, StringArray, StructArray, UnionArray,
    };
    use datafusion::arrow::buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};
    use datafusion::arrow::datatypes::{
        i256, Field, Fields, Int32Type, Int64Type, Schema, UnionFields,
    };

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
             digits than its type Decimal128(38, 0) allows, so it cannot be returned exactly. \
             Cast the value to VARCHAR at the source to read it. DuckDB exports its HUGEINT \
             type, which holds up to 39 digits, as Decimal128(38, 0)"
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
        let array = Decimal128Array::from(vec![10_i128.pow(38)])
            .with_precision_and_scale(38, 2)
            .expect("valid type");
        let batch = batch(vec![("d", Arc::new(array) as ArrayRef)]);
        let overflow = find_decimal_overflow(&batch).expect("an overflow");
        assert_eq!(overflow.value, format!("1{}.00", "0".repeat(36)));
        assert_eq!(overflow.data_type, DataType::Decimal128(38, 2));
    }

    #[test]
    fn a_column_declared_below_its_widths_maximum_precision_is_not_checked() {
        // Only a source mapping a wider type onto the widest precision exceeds it;
        // a narrower declared precision is the source's own promise.
        let d128 = Decimal128Array::from(vec![123_456_i128])
            .with_precision_and_scale(5, 2)
            .expect("valid type");
        let d32 = Decimal32Array::from(vec![1_000])
            .with_precision_and_scale(3, 0)
            .expect("valid type");
        let batch = batch(vec![
            ("d128", Arc::new(d128) as ArrayRef),
            ("d32", Arc::new(d32) as ArrayRef),
        ]);
        assert_eq!(find_decimal_overflow(&batch), None);
    }

    #[test]
    fn every_decimal_width_is_checked() {
        let ten_to_the_76 = i256::from_i128(10).pow_wrapping(76);
        let columns: Vec<(&str, ArrayRef, String)> = vec![
            (
                "d32",
                Arc::new(
                    Decimal32Array::from(vec![1_000_000_000])
                        .with_precision_and_scale(9, 0)
                        .expect("valid type"),
                ),
                format!("1{}", "0".repeat(9)),
            ),
            (
                "d64",
                Arc::new(
                    Decimal64Array::from(vec![1_000_000_000_000_000_000])
                        .with_precision_and_scale(18, 0)
                        .expect("valid type"),
                ),
                format!("1{}", "0".repeat(18)),
            ),
            (
                "d256",
                Arc::new(
                    Decimal256Array::from(vec![ten_to_the_76.neg_wrapping()])
                        .with_precision_and_scale(76, 0)
                        .expect("valid type"),
                ),
                format!("-1{}", "0".repeat(76)),
            ),
        ];
        for (name, array, expected) in columns {
            let found = find_decimal_overflow(&batch(vec![(name, array)]));
            assert_eq!(
                found.map(|o| (o.column, o.value)),
                Some((name.to_string(), expected))
            );
        }
        // The message names HUGEINT only for the type DuckDB exports it as.
        let d32 = find_decimal_overflow(&batch(vec![(
            "d32",
            Arc::new(
                Decimal32Array::from(vec![1_000_000_000])
                    .with_precision_and_scale(9, 0)
                    .expect("valid type"),
            ) as ArrayRef,
        )]))
        .expect("an overflow");
        assert!(!d32.to_string().contains("HUGEINT"), "{d32}");
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
            Arc::clone(&values),
            None,
        );
        let large = LargeListArray::new(
            Arc::clone(&field),
            OffsetBuffer::from_lengths([2]),
            Arc::clone(&values),
            None,
        );
        let view = ListViewArray::new(
            Arc::clone(&field),
            ScalarBuffer::from(vec![0]),
            ScalarBuffer::from(vec![2]),
            Arc::clone(&values),
            None,
        );
        let fixed = FixedSizeListArray::new(field, 2, values, None);
        for array in [
            Arc::new(list) as ArrayRef,
            Arc::new(large),
            Arc::new(view),
            Arc::new(fixed),
        ] {
            let batch = batch(vec![("l", array)]);
            assert_eq!(
                find_decimal_overflow(&batch).map(|o| o.value),
                Some("170141183460469231731687303715884105727".to_string())
            );
        }
    }

    #[test]
    fn a_list_element_outside_every_list_is_not_read() {
        // [[HUGEINT_MAX], [1]] sliced to its second list: the first list's element
        // stays in the child buffer, but no row reaches it.
        let values = hugeint(vec![Some(HUGEINT_MAX), Some(1)]);
        let field = Arc::new(Field::new("item", values.data_type().clone(), true));
        let list = ListArray::new(field, OffsetBuffer::from_lengths([1, 1]), values, None);
        let batch = batch(vec![("l", Arc::new(list.slice(1, 1)) as ArrayRef)]);
        assert_eq!(find_decimal_overflow(&batch), None);
    }

    #[test]
    fn a_child_under_a_null_struct_is_not_read() {
        let inner = hugeint(vec![Some(HUGEINT_MAX), Some(1)]);
        let array = StructArray::new(
            Fields::from(vec![Field::new("h", inner.data_type().clone(), true)]),
            vec![inner],
            Some(NullBuffer::from(vec![false, true])),
        );
        let batch = batch(vec![("s", Arc::new(array) as ArrayRef)]);
        assert_eq!(find_decimal_overflow(&batch), None);
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
    fn a_run_end_encoded_decimal_is_checked() {
        let values = hugeint(vec![Some(1), Some(HUGEINT_MAX)]);
        let runs = RunArray::<Int32Type>::try_new(&Int32Array::from(vec![2, 3]), &values)
            .expect("valid run array");
        let batch1 = batch(vec![("r", Arc::new(runs.clone()) as ArrayRef)]);
        assert_eq!(
            find_decimal_overflow(&batch1).map(|o| o.value),
            Some("170141183460469231731687303715884105727".to_string())
        );
        // Rows 0 and 1 are both in the first run, so the second value is unreachable.
        let batch2 = batch(vec![("r", Arc::new(runs.slice(0, 2)) as ArrayRef)]);
        assert_eq!(find_decimal_overflow(&batch2), None);
    }

    #[test]
    fn an_encoded_child_is_walked_by_its_storage_not_its_logical_length() {
        // One empty list over a run-end-encoded child of 10^12 NULL decimals stored
        // as one run. The NULL slot's bytes are out of range, so the list is walked
        // exactly; that walk must cost what the batch stores, not 10^12 rows.
        let value = Decimal128Array::new(
            ScalarBuffer::from(vec![HUGEINT_MAX]),
            Some(NullBuffer::from(vec![false])),
        )
        .with_precision_and_scale(38, 0)
        .expect("valid type");
        let runs =
            RunArray::<Int64Type>::try_new(&Int64Array::from(vec![1_000_000_000_000_i64]), &value)
                .expect("valid run array");
        let field = Arc::new(Field::new("item", runs.data_type().clone(), true));
        let list = LargeListArray::new(
            field,
            OffsetBuffer::new(ScalarBuffer::from(vec![0_i64, 0])),
            Arc::new(runs),
            None,
        );
        let batch = batch(vec![("l", Arc::new(list) as ArrayRef)]);
        assert_eq!(find_decimal_overflow(&batch), None);
    }

    /// 10^12 NULL decimals stored as one run whose unused payload is out of range,
    /// so the cheap scan flags it and the exact walk must not visit 10^12 rows.
    fn long_null_run() -> RunArray<Int64Type> {
        let value = Decimal128Array::new(
            ScalarBuffer::from(vec![HUGEINT_MAX]),
            Some(NullBuffer::from(vec![false])),
        )
        .with_precision_and_scale(38, 0)
        .expect("valid type");
        RunArray::<Int64Type>::try_new(&Int64Array::from(vec![1_000_000_000_000_i64]), &value)
            .expect("valid run array")
    }

    #[test]
    fn a_struct_over_an_encoded_child_is_walked_by_its_storage() {
        let runs: ArrayRef = Arc::new(long_null_run());
        let inner = StructArray::new(
            Fields::from(vec![Field::new("h", runs.data_type().clone(), true)]),
            vec![runs],
            None,
        );
        // One list entry spanning all 10^12 struct rows.
        let field = Arc::new(Field::new("item", inner.data_type().clone(), true));
        let list = LargeListArray::new(
            field,
            OffsetBuffer::new(ScalarBuffer::from(vec![0_i64, 1_000_000_000_000])),
            Arc::new(inner),
            None,
        );
        let batch = batch(vec![("l", Arc::new(list) as ArrayRef)]);
        assert_eq!(find_decimal_overflow(&batch), None);
    }

    #[test]
    fn a_fixed_size_list_over_an_encoded_child_is_walked_by_its_storage() {
        let runs: ArrayRef = Arc::new(long_null_run());
        let field = Arc::new(Field::new("item", runs.data_type().clone(), true));
        // `try_new` expands the child's NULLs into a 10^12-bit buffer to validate
        // them, which is the very cost under test, so the parts are checked here.
        // SAFETY: size 2 >= 0; values.len() == 5 * 10^11 * 2; no NULL buffer; the
        // field's type is the values' type.
        let fixed =
            unsafe { FixedSizeListArray::new_unchecked(field, 2, runs, None, 500_000_000_000) };
        let list_field = Arc::new(Field::new("item", fixed.data_type().clone(), true));
        let list = LargeListArray::new(
            list_field,
            OffsetBuffer::new(ScalarBuffer::from(vec![0_i64, 500_000_000_000])),
            Arc::new(fixed),
            None,
        );
        let batch = batch(vec![("l", Arc::new(list) as ArrayRef)]);
        assert_eq!(find_decimal_overflow(&batch), None);
    }

    #[test]
    fn adjacent_reachable_slots_coalesce_as_they_arrive() {
        // A nullable struct's reachable rows arrive one at a time. The walk holds
        // one range per run of them, not one per row.
        let ranges = merged((0..1_000_000).map(|row| row..row + 1));
        assert_eq!(ranges, vec![0..1_000_000]);
        assert!(ranges.capacity() < 1_000, "capacity {}", ranges.capacity());
        // A list view's or a dictionary's slots can arrive out of order.
        assert_eq!(
            merged([5..6, 0..2, 1..3, 4..4].into_iter()),
            vec![0..3, 5..6]
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

    #[test]
    fn an_empty_dictionary_beside_a_flagged_null_slot_is_walked_without_panicking() {
        // The NULL decimal's bytes are out of range, so the struct is walked row by
        // row, and the walk reaches a dictionary with no values at all.
        let empty_values = hugeint(vec![]);
        let keys = Int32Array::from(vec![None::<i32>, None]);
        let dictionary: ArrayRef = Arc::new(
            DictionaryArray::<Int32Type>::try_new(keys, empty_values).expect("valid dictionary"),
        );
        let decimal: ArrayRef = Arc::new(
            Decimal128Array::new(
                ScalarBuffer::from(vec![i128::MIN, 1]),
                Some(NullBuffer::from(vec![false, true])),
            )
            .with_precision_and_scale(38, 0)
            .expect("valid type"),
        );
        let array = StructArray::new(
            Fields::from(vec![
                Field::new("d", dictionary.data_type().clone(), true),
                Field::new("h", decimal.data_type().clone(), true),
            ]),
            vec![dictionary, decimal],
            None,
        );
        let batch = batch(vec![("s", Arc::new(array) as ArrayRef)]);
        assert_eq!(find_decimal_overflow(&batch), None);
    }

    #[test]
    fn an_unused_dictionary_entry_is_not_read() {
        let values = hugeint(vec![Some(1), Some(HUGEINT_MAX)]);
        let keys = Int32Array::from(vec![Some(0), None, Some(0)]);
        let dictionary =
            DictionaryArray::<Int32Type>::try_new(keys, values).expect("valid dictionary");
        let batch = batch(vec![("d", Arc::new(dictionary) as ArrayRef)]);
        assert_eq!(find_decimal_overflow(&batch), None);
    }

    fn union_fields(member: &ArrayRef) -> UnionFields {
        UnionFields::try_new(
            [0, 1],
            [
                Field::new("h", member.data_type().clone(), true),
                Field::new("s", DataType::Utf8, true),
            ],
        )
        .expect("valid union fields")
    }

    #[test]
    fn the_active_member_of_a_union_is_checked() {
        // DuckDB exports UNION(h HUGEINT, s VARCHAR) as a sparse union.
        let h = hugeint(vec![Some(HUGEINT_MAX), None, Some(5)]);
        let s: ArrayRef = Arc::new(StringArray::from(vec![None, Some("x"), None]));
        let sparse = UnionArray::try_new(
            union_fields(&h),
            ScalarBuffer::from(vec![0_i8, 1, 0]),
            None,
            vec![Arc::clone(&h), s],
        )
        .expect("valid sparse union");
        let dense_h = hugeint(vec![Some(5), Some(HUGEINT_MAX)]);
        let dense_s: ArrayRef = Arc::new(StringArray::from(vec!["x"]));
        let dense = UnionArray::try_new(
            union_fields(&dense_h),
            ScalarBuffer::from(vec![0_i8, 1, 0]),
            Some(ScalarBuffer::from(vec![0, 0, 1])),
            vec![dense_h, dense_s],
        )
        .expect("valid dense union");
        for array in [Arc::new(sparse) as ArrayRef, Arc::new(dense)] {
            let overflow = find_decimal_overflow(&batch(vec![("u", array)])).expect("an overflow");
            assert_eq!(
                (overflow.column.as_str(), overflow.value.as_str()),
                ("u.h", "170141183460469231731687303715884105727")
            );
        }
    }

    #[test]
    fn an_inactive_union_member_is_not_read() {
        // Row 0 selects `s`, so the HUGEINT member's slot at row 0 is unreachable.
        let h = hugeint(vec![Some(HUGEINT_MAX), Some(5)]);
        let s: ArrayRef = Arc::new(StringArray::from(vec![Some("x"), None]));
        let sparse = UnionArray::try_new(
            union_fields(&h),
            ScalarBuffer::from(vec![1_i8, 0]),
            None,
            vec![h, s],
        )
        .expect("valid sparse union");
        let batch = batch(vec![("u", Arc::new(sparse) as ArrayRef)]);
        assert_eq!(find_decimal_overflow(&batch), None);
    }
}
