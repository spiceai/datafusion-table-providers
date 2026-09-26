use super::*;
use datafusion::arrow::datatypes::Field as ArrowField;
use datafusion::logical_expr::{col, lit, Expr};
use std::sync::Arc;

fn schema() -> Schema {
    Schema::new(vec![
        ArrowField::new("_id", DataType::Utf8, false),
        ArrowField::new("name", DataType::Utf8, true),
        ArrowField::new("age", DataType::Int32, true),
        ArrowField::new("big", DataType::Int64, true),
        ArrowField::new("score", DataType::Float64, true),
        ArrowField::new("vip", DataType::Boolean, true),
        ArrowField::new(
            "created",
            DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())),
            true,
        ),
        ArrowField::new("day", DataType::Date32, true),
        ArrowField::new(
            "tags",
            DataType::List(Arc::new(ArrowField::new("item", DataType::Utf8, true))),
            true,
        ),
        ArrowField::new("price", DataType::Decimal128(18, 6), true),
        ArrowField::new("address.city", DataType::Utf8, true),
        ArrowField::new("$where", DataType::Utf8, true),
        ArrowField::new("data", DataType::Utf8, true),
    ])
}

fn scope() -> FilterScope {
    FilterScope {
        unpushable_columns: HashSet::from(["data".to_string()]),
        unnest_depth: Some(0),
    }
}

fn translate(expr: &Expr) -> Option<MongoFilter> {
    translate_filter(expr, &schema(), &scope())
}

fn translate_unnested(expr: &Expr, depth: usize) -> Option<MongoFilter> {
    let scope = FilterScope {
        unnest_depth: Some(depth),
        ..scope()
    };
    translate_filter(expr, &schema(), &scope)
}

fn exact(expr: &Expr) -> Document {
    let filter = translate(expr).expect("translatable");
    assert!(filter.exact, "expected an exact translation of {expr}: {filter:?}");
    filter.document
}

fn inexact(expr: &Expr) -> Document {
    let filter = translate(expr).expect("translatable");
    assert!(!filter.exact, "expected an inexact translation of {expr}: {filter:?}");
    filter.document
}

fn not_array() -> Document {
    doc! { "$type": "array" }
}

fn oid(hex: &str) -> ObjectId {
    ObjectId::parse_str(hex).expect("valid object id")
}

/// The `$type` list `rendered` selects: every non-string type, optionally
/// without `ObjectId`s.
fn rendered_types(with_object_ids: bool) -> Bson {
    let list: Vec<&str> = NON_STRING_TYPES
        .iter()
        .copied()
        .filter(|t| with_object_ids || *t != "objectId")
        .collect();
    types(&list)
}

/// Relaxed extended JSON, which compares NaN by value where `Bson` cannot.
fn json(document: Document) -> serde_json::Value {
    Bson::Document(document).into_relaxed_extjson()
}

// --- three-valued logic ---

#[test]
fn not_equal_excludes_null_and_missing_integers() {
    // `$ne` alone matches a null or missing `age`, which SQL evaluates to NULL.
    assert_eq!(
        exact(&col("age").not_eq(lit(30))),
        doc! { "age": {
            "$type": ["int", "long"],
            "$ne": 30,
            "$gte": i32::MIN,
            "$lte": i32::MAX,
            "$not": not_array(),
        } }
    );
}

#[test]
fn not_equal_on_a_string_column_is_a_superset_without_nulls() {
    assert_eq!(
        inexact(&col("name").not_eq(lit("alice"))),
        doc! { "$or": [
            { "name": { "$nin": ["alice", Bson::Null] } },
            { "name": { "$type": "array" } },
        ] }
    );
}

#[test]
fn a_negated_comparison_is_its_complement_not_a_nor() {
    // NOT (age > 30) keeps rows where age <= 30; `$nor` would also keep NULLs.
    assert_eq!(
        exact(&Expr::Not(Box::new(col("age").gt(lit(30))))),
        doc! { "age": { "$type": ["int", "long"], "$lte": 30, "$gte": i32::MIN, "$not": not_array() } }
    );
}

#[test]
fn a_negated_conjunction_applies_de_morgan() {
    let expr = Expr::Not(Box::new(col("age").gt(lit(30)).and(col("vip").eq(lit(true)))));
    assert_eq!(
        exact(&expr),
        doc! { "$or": [
            { "age": { "$type": ["int", "long"], "$lte": 30, "$gte": i32::MIN, "$not": not_array() } },
            { "vip": { "$type": "bool", "$eq": false, "$not": not_array() } },
        ] }
    );
}

#[test]
fn not_in_excludes_nulls_and_a_null_element_refuses() {
    assert_eq!(
        exact(&col("age").in_list(vec![lit(1), lit(2)], true)),
        doc! { "age": {
            "$type": ["int", "long"],
            "$nin": [1, 2],
            "$gte": i32::MIN,
            "$lte": i32::MAX,
            "$not": not_array(),
        } }
    );
    // `age NOT IN (1, NULL)` is never true, which DataFusion evaluates itself.
    assert!(translate(&col("age").in_list(vec![lit(1), lit(ScalarValue::Int32(None))], true)).is_none());
}

#[test]
fn in_drops_null_elements() {
    assert_eq!(
        exact(&col("age").in_list(vec![lit(30), lit(ScalarValue::Int32(None))], false)),
        doc! { "age": { "$type": ["int", "long"], "$in": [30], "$not": not_array() } }
    );
    assert_eq!(
        exact(&col("age").in_list(vec![lit(ScalarValue::Int32(None))], false)),
        nothing()
    );
}

#[test]
fn a_comparison_with_null_is_left_to_datafusion() {
    assert!(translate(&col("age").eq(lit(ScalarValue::Int32(None)))).is_none());
    assert!(translate(&col("name").not_eq(lit(ScalarValue::Utf8(None)))).is_none());
}

#[test]
fn not_like_keeps_only_non_null_strings() {
    let regex = BsonRegex {
        pattern: "\\Aa.*\\z".to_string(),
        options: "s".to_string(),
    };
    assert_eq!(
        inexact(&col("name").not_like(lit("a%"))),
        doc! { "$or": [
            { "name": { "$type": "string", "$not": Bson::RegularExpression(regex) } },
            { "name": { "$type": rendered_types(true) } },
        ] }
    );
}

// --- types ---

#[test]
fn an_object_id_column_matches_its_hex_rendering() {
    let hex = "65f000000000000000000001";
    assert_eq!(
        inexact(&col("_id").eq(lit(hex))),
        doc! { "$or": [
            { "_id": { "$in": [hex, oid(hex)] } },
            { "_id": { "$type": rendered_types(false) } },
        ] }
    );
    // Uppercase hex is not how the conversion renders an ObjectId.
    let upper = "65F000000000000000000001";
    assert_eq!(
        inexact(&col("_id").eq(lit(upper))),
        doc! { "$or": [
            { "_id": { "$in": [upper] } },
            { "_id": { "$type": rendered_types(false) } },
        ] }
    );
}

#[test]
fn a_string_range_takes_in_every_rendered_type() {
    assert_eq!(
        inexact(&col("name").gt(lit("6"))),
        doc! { "$or": [
            { "name": { "$gt": "6" } },
            { "name": { "$type": rendered_types(true) } },
        ] }
    );
}

#[test]
fn an_integer_column_compared_with_a_fraction_rounds_the_bound() {
    // CAST(age AS DOUBLE) > 2.5 keeps ages of 3 and above.
    let cast = Expr::Cast(Cast::new(Box::new(col("age")), DataType::Float64));
    assert_eq!(
        exact(&cast.clone().gt(lit(2.5))),
        doc! { "age": { "$type": ["int", "long"], "$gte": 3, "$lte": i32::MAX, "$not": not_array() } }
    );
    assert_eq!(exact(&cast.eq(lit(2.5))), nothing());
}

#[test]
fn an_integer_bound_outside_the_column_range_is_decided_locally() {
    let cast = Expr::Cast(Cast::new(Box::new(col("age")), DataType::Int64));
    assert_eq!(exact(&cast.clone().gt(lit(5_000_000_000_i64))), nothing());
    assert_eq!(
        exact(&cast.lt(lit(5_000_000_000_i64))),
        doc! { "age": { "$type": ["int", "long"], "$gte": i32::MIN, "$lte": i32::MAX, "$not": not_array() } }
    );
}

#[test]
fn a_float_range_includes_nan() {
    assert_eq!(
        json(inexact(&col("score").gt(lit(2.0)))),
        json(doc! { "$or": [
            { "score": { "$type": ["double", "int", "long"], "$gt": 2.0, "$not": not_array() } },
            { "score": { "$type": "double", "$eq": f64::NAN, "$not": not_array() } },
        ] })
    );
    // Arrow orders a NaN with its sign bit set below every number.
    assert_eq!(
        json(inexact(&col("score").lt(lit(2.0)))),
        json(doc! { "$or": [
            { "score": { "$type": ["double", "int", "long"], "$lt": 2.0, "$not": not_array() } },
            { "score": { "$type": "double", "$eq": f64::NAN, "$not": not_array() } },
        ] })
    );
}

#[test]
fn float_equality_is_exact_except_at_zero_and_nan() {
    assert_eq!(
        exact(&col("score").eq(lit(2.5))),
        doc! { "score": { "$type": ["double", "int", "long"], "$eq": 2.5, "$not": not_array() } }
    );
    inexact(&col("score").eq(lit(0.0)));
    inexact(&col("score").not_eq(lit(-0.0)));
    inexact(&col("score").lt(lit(f64::NAN)));
}

#[test]
fn a_float_literal_beyond_two_to_the_fifty_third_is_refused() {
    assert!(translate(&col("score").eq(lit(9_007_199_254_740_992.0))).is_none());
    assert!(translate(&col("score").eq(lit(-9_007_199_254_740_993_i64))).is_none());
}

#[test]
fn a_decimal_column_is_not_compared_remotely() {
    // The conversion rounds a Decimal128 to the column's scale.
    let value = lit(ScalarValue::Decimal128(Some(1_000_001), 18, 6));
    assert!(translate(&col("price").gt(value)).is_none());
}

#[test]
fn a_date_column_compares_whole_utc_days() {
    let day = 19_723; // 2024-01-01
    let start = i64::from(day) * 86_400_000;
    assert_eq!(
        exact(&col("day").eq(lit(ScalarValue::Date32(Some(day))))),
        doc! { "day": {
            "$type": "date",
            "$gte": BsonDateTime::from_millis(start),
            "$lte": BsonDateTime::from_millis(start + 86_400_000 - 1),
            "$not": not_array(),
        } }
    );
}

#[test]
fn a_cast_of_a_timestamp_to_a_date_is_not_unwrapped() {
    let cast = Expr::Cast(Cast::new(Box::new(col("created")), DataType::Date32));
    assert!(translate(&cast.eq(lit(ScalarValue::Date32(Some(19_723))))).is_none());
    let cast = Expr::Cast(Cast::new(Box::new(col("age")), DataType::Utf8));
    assert!(translate(&cast.eq(lit("30"))).is_none());
}

#[test]
fn a_sub_millisecond_timestamp_bound_rounds_toward_the_kept_rows() {
    // 2024-01-01T00:00:00.000500Z
    let ns = lit(ScalarValue::TimestampNanosecond(Some(1_704_067_200_000_500_000), None));
    assert_eq!(
        exact(&col("created").gt_eq(ns.clone())),
        doc! { "$or": [
            { "created": { "$type": "date", "$gte": BsonDateTime::from_millis(1_704_067_200_001), "$not": not_array() } },
            { "created": {
                "$type": "timestamp",
                "$gte": Bson::Timestamp(BsonTimestamp { time: 1_704_067_201, increment: 0 }),
                "$not": not_array(),
            } },
        ] }
    );
    assert_eq!(
        exact(&col("created").lt(ns.clone())),
        doc! { "$or": [
            { "created": { "$type": "date", "$lte": BsonDateTime::from_millis(1_704_067_200_000), "$not": not_array() } },
            { "created": {
                "$type": "timestamp",
                "$lte": Bson::Timestamp(BsonTimestamp { time: 1_704_067_200, increment: u32::MAX }),
                "$not": not_array(),
            } },
        ] }
    );
    // No stored millisecond equals a bound that falls between two of them.
    assert_eq!(exact(&col("created").eq(ns)), nothing());
}

#[test]
fn a_bson_timestamp_is_compared_by_its_seconds() {
    // A BSON timestamp converts to its seconds, so 1s after the epoch matches
    // every increment of second 1 and no instant between whole seconds.
    let ms = lit(ScalarValue::TimestampMillisecond(Some(1_000), None));
    assert_eq!(
        exact(&col("created").eq(ms)),
        doc! { "$or": [
            { "created": { "$type": "date", "$eq": BsonDateTime::from_millis(1_000), "$not": not_array() } },
            { "created": {
                "$type": "timestamp",
                "$gte": Bson::Timestamp(BsonTimestamp { time: 1, increment: 0 }),
                "$lte": Bson::Timestamp(BsonTimestamp { time: 1, increment: u32::MAX }),
                "$not": not_array(),
            } },
        ] }
    );
}

#[test]
fn a_pre_epoch_timestamp_bound_floors() {
    // -1.5ms: the rows at or before it are those at -2ms or earlier.
    let us = lit(ScalarValue::TimestampMicrosecond(Some(-1_500), None));
    assert_eq!(
        exact(&col("created").lt_eq(us)),
        doc! { "created": { "$type": "date", "$lte": BsonDateTime::from_millis(-2), "$not": not_array() } }
    );
}

// --- boolean columns ---

#[test]
fn a_bare_boolean_column_is_pushed_down() {
    assert_eq!(
        exact(&col("vip")),
        doc! { "vip": { "$type": "bool", "$eq": true, "$not": not_array() } }
    );
    assert_eq!(
        exact(&Expr::Not(Box::new(col("vip")))),
        doc! { "vip": { "$type": "bool", "$eq": false, "$not": not_array() } }
    );
}

#[test]
fn is_not_true_takes_in_nulls() {
    assert_eq!(
        exact(&col("vip").is_not_true()),
        doc! { "$nor": [{ "vip": { "$type": "bool", "$eq": true, "$not": not_array() } }] }
    );
}

// --- null tests ---

#[test]
fn is_null_on_a_string_column_counts_arrays_as_values() {
    assert_eq!(
        exact(&col("name").is_not_null()),
        doc! { "$or": [
            { "name": { "$type": "array" } },
            { "name": { "$exists": true, "$not": { "$type": "null" } } },
        ] }
    );
    assert_eq!(
        exact(&col("name").is_null()),
        doc! { "$nor": [{ "$or": [
            { "name": { "$type": "array" } },
            { "name": { "$exists": true, "$not": { "$type": "null" } } },
        ] }] }
    );
}

#[test]
fn is_null_on_an_integer_column_counts_other_types_as_null() {
    assert_eq!(
        exact(&col("age").is_null()),
        doc! { "$nor": [{ "age": {
            "$type": ["int", "long"],
            "$gte": i32::MIN,
            "$lte": i32::MAX,
            "$not": not_array(),
        } }] }
    );
}

#[test]
fn is_distinct_from() {
    let equal = doc! { "age": { "$type": ["int", "long"], "$eq": 30, "$not": not_array() } };
    assert_eq!(
        exact(&Expr::BinaryExpr(BinaryExpr::new(Box::new(col("age")), Operator::IsNotDistinctFrom, Box::new(lit(30))))),
        equal
    );
    assert_eq!(
        exact(&Expr::BinaryExpr(BinaryExpr::new(Box::new(col("age")), Operator::IsDistinctFrom, Box::new(lit(30))))),
        doc! { "$nor": [equal] }
    );
    inexact(&Expr::BinaryExpr(BinaryExpr::new(
        Box::new(col("name")),
        Operator::IsDistinctFrom,
        Box::new(lit("x")),
    )));
}

// --- LIKE ---

#[test]
fn like_matches_newlines_and_anchors_absolutely() {
    let filter = translate(&col("name").like(lit("line1%"))).expect("translatable");
    let Some(Bson::RegularExpression(regex)) = filter
        .document
        .get_array("$or")
        .ok()
        .and_then(|branches| branches.first())
        .and_then(Bson::as_document)
        .and_then(|branch| branch.get("name"))
    else {
        panic!("expected a regex branch: {filter:?}");
    };
    assert_eq!(regex.pattern, "\\Aline1.*\\z");
    assert_eq!(regex.options, "s");
}

#[test]
fn like_treats_backslash_as_the_escape() {
    assert_eq!(like_regex("a\\%b", false).map(|r| r.pattern), Some("\\Aa\\%b\\z".to_string()));
    assert_eq!(like_regex("a\\_", false).map(|r| r.pattern), Some("\\Aa\\_\\z".to_string()));
    // A trailing backslash is a literal one.
    assert_eq!(like_regex("a\\", false).map(|r| r.pattern), Some("\\Aa\\\\\\z".to_string()));
    assert_eq!(like_regex("a.b$", false).map(|r| r.pattern), Some("\\Aa\\.b\\$\\z".to_string()));
    assert_eq!(like_regex("a\nb\0", false).map(|r| r.pattern), Some("\\Aa\\x{A}b\\x{0}\\z".to_string()));
}

#[test]
fn like_with_another_escape_character_is_refused() {
    let expr = Expr::Like(Like::new(false, Box::new(col("name")), Box::new(lit("a#%")), Some('#'), false));
    assert!(translate(&expr).is_none());
}

#[test]
fn ilike_folds_ascii_letters_to_their_unicode_partners() {
    assert_eq!(
        like_regex("Ks_%", true).map(|r| r.pattern),
        Some("\\A[kK\\x{212A}][sS\\x{17F}]..*\\z".to_string())
    );
    assert_eq!(like_regex("é%", true), None);
}

// --- what cannot be pushed down ---

#[test]
fn unaddressable_columns_are_refused() {
    // A catch-all is assembled from undeclared fields; `$where` would be read as an operator.
    assert!(translate(&col("data").eq(lit("x"))).is_none());
    assert!(translate(&col("$where").eq(lit("x"))).is_none());
    // Without unnesting a dotted column is a field named with a dot.
    assert!(translate(&col(r#""address.city""#).eq(lit("x"))).is_none());
}

#[test]
fn an_unnested_path_requires_document_parents() {
    let parent = doc! { "address": { "$not": { "$type": "array" } } };
    let not_null = |absent: Bson| {
        doc! { "$or": [
            { "$and": [parent.clone(), { "address.city": { "$type": "array" } }] },
            { "$and": [parent.clone(), { "address.city": { "$exists": true, "$not": { "$type": absent } } }] },
        ] }
    };
    // At the unnest depth an embedded document is rendered as JSON...
    let at_depth = translate_unnested(&col(r#""address.city""#).is_not_null(), 1).expect("translatable");
    assert!(at_depth.exact);
    assert_eq!(at_depth.document, not_null(types(&["null"])));
    // ...above it, unnesting flattens it away and leaves the column NULL.
    let above_depth = translate_unnested(&col(r#""address.city""#).is_not_null(), 2).expect("translatable");
    assert_eq!(above_depth.document, not_null(types(&["null", "object"])));
}

#[test]
fn column_to_column_comparisons_are_refused() {
    assert!(translate(&col("age").eq(col("big"))).is_none());
}

#[test]
fn a_conjunction_with_an_untranslatable_side_keeps_the_other_as_a_superset() {
    let filter = translate(&col("age").eq(lit(30)).and(col("age").eq(col("big")))).expect("translatable");
    assert!(!filter.exact);
    assert_eq!(filter.document, exact(&col("age").eq(lit(30))));
    assert!(translate(&col("age").eq(lit(30)).or(col("age").eq(col("big")))).is_none());
}

#[test]
fn functions_are_not_pushed_down() {
    use datafusion::logical_expr::{create_udf, ColumnarValue, Volatility};
    // A user function may reuse a built-in's name, so no call is translated.
    let udf = create_udf(
        "starts_with",
        vec![DataType::Utf8, DataType::Utf8],
        DataType::Boolean,
        Volatility::Immutable,
        Arc::new(|args: &[ColumnarValue]| Ok(args[0].clone())),
    );
    assert!(translate(&udf.call(vec![col("name"), lit("a")])).is_none());
    assert!(translate(&udf.call(vec![col("name"), lit("a")]).is_true()).is_none());
}
