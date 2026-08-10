use std::{borrow::Cow, sync::Arc};

use bigdecimal::{num_bigint::BigInt, BigDecimal};
use datafusion::{
    logical_expr::{Cast, Expr, Operator},
    scalar::ScalarValue,
    sql::unparser::dialect::{
        DefaultDialect, Dialect, DuckDBDialect, MySqlDialect, PostgreSqlDialect, SqliteDialect,
    },
    sql::TableReference,
};

pub const SECONDS_IN_DAY: i32 = 86_400;

#[derive(Debug, snafu::Snafu)]
pub enum Error {
    #[snafu(display("Expression not supported {expr}"))]
    UnsupportedFilterExpr { expr: String },

    #[snafu(display("Engine {engine} not supported for expression {expr}"))]
    EngineNotSupportedForExpression { engine: String, expr: String },
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Copy, Debug)]
pub enum Engine {
    Spark,
    SQLite,
    DuckDB,
    ODBC,
    Postgres,
    MySQL,
}

impl Engine {
    /// Get the corresponding `Dialect` to use for unparsing
    pub fn dialect(&self) -> Arc<dyn Dialect + Send + Sync> {
        match self {
            Engine::SQLite => Arc::new(SqliteDialect {}),
            Engine::Postgres => Arc::new(PostgreSqlDialect {}),
            Engine::MySQL => Arc::new(MySqlDialect {}),
            Engine::DuckDB => Arc::new(DuckDBDialect::new()),
            Engine::Spark | Engine::ODBC => Arc::new(DefaultDialect {}),
        }
    }
}

/// Renders `value` as a SQL string literal for `engine`, escaping the characters that engine
/// reads as significant inside one.
///
/// Interpolating the value bare lets a single quote close the literal early: an ordinary
/// apostrophe then makes the statement fail to parse, and a value shaped like `x' OR 1=1 --`
/// binds as a wider predicate than the caller wrote, so a `DELETE` can match rows the filter
/// excluded. Every engine here ends a literal on `'` and accepts a doubled `''` for an embedded
/// one, which is also what `DataFusion`'s own unparser emits.
///
/// `Engine::MySQL` and `Engine::Spark` additionally read `\` as an escape character inside a
/// literal, so for them a backslash must be doubled as well — otherwise `\'` consumes the first
/// quote of a doubled pair and the second one closes the literal, which is worse than not
/// escaping at all. `Engine::DuckDB`, `Engine::SQLite`, `Engine::ODBC` and Postgres (under
/// `standard_conforming_strings`, on by default since 9.1) take `\` literally and must not gain
/// an extra one.
///
/// Two escape-mode residues remain, both of them server settings this function cannot see:
/// MySQL's `NO_BACKSLASH_ESCAPES` and Postgres's `standard_conforming_strings = off`. Binding the
/// value instead of rendering it is the only fix that removes them.
///
/// The match is exhaustive over [`Engine`] on purpose: a new engine must state which escaping it
/// needs rather than inherit whichever arm happens to be the catch-all.
pub(crate) fn string_literal(value: &str, engine: Option<Engine>) -> String {
    let escaped = match engine {
        Some(Engine::MySQL | Engine::Spark) => {
            if value.contains(['\\', '\'']) {
                Cow::Owned(value.replace('\\', r"\\").replace('\'', "''"))
            } else {
                Cow::Borrowed(value)
            }
        }
        Some(Engine::DuckDB | Engine::SQLite | Engine::Postgres | Engine::ODBC) | None => {
            if value.contains('\'') {
                Cow::Owned(value.replace('\'', "''"))
            } else {
                Cow::Borrowed(value)
            }
        }
    };
    format!("'{escaped}'")
}

/// Renders `name` as a double-quoted SQL identifier, doubling any embedded quote so that a
/// quote in a column name cannot close the identifier early.
pub(crate) fn quoted_identifier(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

/// Renders `table` as a qualified SQL identifier chain, quoting each part with
/// [`quoted_identifier`] so that a statement addresses the same table the reference names.
///
/// Every part is quoted unconditionally, unlike [`TableReference::to_quoted_string`], which leaves
/// an all-lowercase part bare and so renders a reserved word (`order`, `user`) as syntax rather
/// than as a name.
pub(crate) fn quoted_table_reference(table: &TableReference) -> String {
    match table {
        TableReference::Bare { table } => quoted_identifier(table),
        TableReference::Partial { schema, table } => {
            format!("{}.{}", quoted_identifier(schema), quoted_identifier(table))
        }
        TableReference::Full {
            catalog,
            schema,
            table,
        } => format!(
            "{}.{}.{}",
            quoted_identifier(catalog),
            quoted_identifier(schema),
            quoted_identifier(table)
        ),
    }
}

/// Renders `expr` as a SQL fragment for `engine`.
///
/// The fragment goes into a statement that names exactly one table, so every qualified column in
/// `expr` must carry the same relation. A predicate spanning two relations has no meaning in such a
/// statement, and rendering it by column name alone would silently *widen* it: `t1.id = t2.id` would
/// become `"id" = "id"`, a tautology that a `DELETE` applies to every row. `DataFusion` never offers
/// a cross-relation predicate to a single-table provider — it pushes a filter into a scan only when
/// that scan covers it, and it validates ownership before unqualifying a DML filter — but this
/// function is public and its callers have no such guarantee, so the check belongs here.
///
/// References are compared for exact equality rather than with `TableReference::resolved_eq`, which
/// treats a component the other side omits as matching: a bare `t` is `resolved_eq` to both
/// `a.schema.t` and `b.schema.t`, so it is not transitive, and comparing every reference against one
/// arbitrarily chosen pivot could then admit two genuinely different tables. Without the session's
/// default catalog and schema this function cannot canonicalise a partial reference, so mixed
/// qualification of one table (`t` alongside `public.t`) is refused too — no caller produces it, and
/// refusing is the direction that cannot widen a statement.
pub fn to_sql_with_engine(expr: &Expr, engine: Option<Engine>) -> Result<String> {
    let mut relations = expr
        .column_refs()
        .into_iter()
        .filter_map(|column| column.relation.as_ref());
    if let Some(first) = relations.next() {
        if relations.any(|other| other != first) {
            return Err(Error::UnsupportedFilterExpr {
                expr: format!("{expr}"),
            });
        }
    }

    render_expr(expr, engine)
}

#[allow(clippy::too_many_lines)]
#[allow(clippy::cast_precision_loss)]
fn render_expr(expr: &Expr, engine: Option<Engine>) -> Result<String> {
    match expr {
        Expr::BinaryExpr(binary_expr) => {
            let left = render_expr(&binary_expr.left, engine)?;
            let right = render_expr(&binary_expr.right, engine)?;

            if let Some(Engine::DuckDB) = engine {
                // TODO: DuckDB doesn't support comparison between timestamp_s /timestamp_ms with timestampz as of v1
                // Revisit in future DuckDB versions
                //
                // Which operand is a timestamp is decided by its `Expr` shape, never by whether
                // its rendered SQL starts with `TO_TIMESTAMP`. A nested comparison is itself
                // normalized, so it renders as `TO_TIMESTAMP(..)` while being a boolean, and
                // normalizing it emits `EPOCH_MS(<predicate>)`, for which DuckDB has no overload.
                //
                // `Minus` is included because subtraction is the only arithmetic DuckDB defines
                // between two timestamps. `+`, `*` and `/` have no timestamp/timestamp overload
                // at all, so normalizing their operands could not make them bind.
                let normalizes_timestamp_operands = matches!(
                    binary_expr.op,
                    Operator::Eq
                        | Operator::NotEq
                        | Operator::Lt
                        | Operator::LtEq
                        | Operator::Gt
                        | Operator::GtEq
                        | Operator::IsDistinctFrom
                        | Operator::IsNotDistinctFrom
                        | Operator::Minus
                );
                // Only the left operand is normalized, and only when the right is a timestamp
                // and the left is not. `EPOCH_MS` yields whole milliseconds, so normalizing an
                // operand costs sub-millisecond precision: that is worth it for the
                // `TIMESTAMP_S`/`TIMESTAMP_MS` columns the rewrite exists for, which carry no
                // finer precision to lose, but not for a microsecond `TIMESTAMPTZ` column,
                // which compares exactly without it. Normalizing a bare *right* operand would
                // extend that truncation to comparisons that bind correctly today, trading a
                // loud binder error for a silently shifted result, so the mismatch is left to
                // fail loudly in that direction.
                if normalizes_timestamp_operands
                    && is_duckdb_timestamp_operand(&binary_expr.right)
                    && !is_duckdb_timestamp_operand(&binary_expr.left)
                {
                    // `EPOCH_MS(..)` parenthesizes the left operand on its own.
                    return Ok(format!(
                        "TO_TIMESTAMP(EPOCH_MS({}) / 1000) {} {}",
                        left,
                        binary_expr.op,
                        group_if_binary(&binary_expr.right, &right)
                    ));
                }
            }

            match binary_expr.op {
                Operator::And | Operator::Or => {
                    Ok(format!("({}) {} ({})", left, binary_expr.op, right))
                }
                // The `AND`/`OR` arm above parenthesizes its own operands; every other
                // operator needs a nested operand grouped so precedence cannot regroup it.
                _ => Ok(format!(
                    "{} {} {}",
                    group_if_binary(&binary_expr.left, &left),
                    binary_expr.op,
                    group_if_binary(&binary_expr.right, &right)
                )),
            }
        }
        Expr::Column(column) => {
            // Render the column's own name, dropping any qualifier. Rendering `Column`'s
            // `Display` would emit its *flat* name, making a qualified `t.a` the single identifier
            // `"t.a"` — a column that usually does not exist, and where it does (dotted names are
            // legal in Arrow and in every engine here) is not the one the caller named.
            //
            // Dropping it is safe because `to_sql_with_engine` has already established that the
            // whole fragment names at most one relation, and the statement it goes into names
            // exactly one table. `DataFusion` unqualifies at the same boundary for the same reason
            // (`strip_column_qualifiers` for DML filters, `unnormalize_cols` for pushed-down ones).
            //
            // Rejecting a lone qualifier instead would break the callers that use this function as
            // a *capability probe*: `supports_filters_pushdown` is asked about the qualified
            // predicate while `scan` receives the unqualified one, and `ClickHouseTable` (plus the
            // runtime's DuckDB vector index) answers it by rendering here and mapping `Err(_)` to
            // `Unsupported`. An error would silently keep those predicates — `LIKE` among them —
            // out of the remote statement.
            match engine {
                Some(Engine::Spark | Engine::ODBC) => Ok(column.name.clone()),
                _ => Ok(quoted_identifier(&column.name)),
            }
        }
        Expr::Cast(cast) => handle_cast(cast, engine, expr),
        Expr::Literal(value, _) => match value {
            ScalarValue::Date32(Some(value)) => match engine {
                Some(Engine::SQLite) => {
                    Ok(format!("date({}, 'unixepoch')", value * SECONDS_IN_DAY))
                }
                _ => Ok(format!("TO_TIMESTAMP({})", value * SECONDS_IN_DAY)),
            },
            ScalarValue::Date64(Some(value)) => match engine {
                Some(Engine::SQLite) => Ok(format!(
                    "date({}, 'unixepoch')",
                    value * i64::from(SECONDS_IN_DAY)
                )),
                _ => Ok(format!(
                    "TO_TIMESTAMP({})",
                    value * i64::from(SECONDS_IN_DAY)
                )),
            },
            ScalarValue::Null => Ok(value.to_string()),
            ScalarValue::Int16(Some(value)) => Ok(value.to_string()),
            ScalarValue::Int32(Some(value)) => Ok(value.to_string()),
            ScalarValue::Int64(Some(value)) => Ok(value.to_string()),
            ScalarValue::Boolean(Some(value)) => Ok(value.to_string()),
            ScalarValue::Utf8(Some(value))
            | ScalarValue::LargeUtf8(Some(value))
            | ScalarValue::Utf8View(Some(value)) => Ok(string_literal(value, engine)),
            ScalarValue::Float32(Some(value)) => Ok(value.to_string()),
            ScalarValue::Float64(Some(value)) => Ok(value.to_string()),
            ScalarValue::Int8(Some(value)) => Ok(value.to_string()),
            ScalarValue::UInt8(Some(value)) => Ok(value.to_string()),
            ScalarValue::UInt16(Some(value)) => Ok(value.to_string()),
            ScalarValue::UInt32(Some(value)) => Ok(value.to_string()),
            ScalarValue::UInt64(Some(value)) => Ok(value.to_string()),
            ScalarValue::TimestampNanosecond(Some(value), timezone) => match engine {
                Some(Engine::SQLite) => {
                    Ok(format!("datetime({}, 'unixepoch')", value / 1_000_000_000))
                }
                Some(Engine::Postgres) => {
                    if timezone.is_none() {
                        Ok(format!(
                            "TO_TIMESTAMP({}) AT TIME ZONE 'UTC'",
                            *value as f64 / 1_000_000_000.0
                        ))
                    } else {
                        Ok(format!("TO_TIMESTAMP({})", *value as f64 / 1_000_000_000.0))
                    }
                }
                _ => Ok(format!("TO_TIMESTAMP({})", value / 1_000_000_000)),
            },
            ScalarValue::TimestampMicrosecond(Some(value), timezone) => match engine {
                Some(Engine::SQLite) => Ok(format!("datetime({}, 'unixepoch')", value / 1_000_000)),
                Some(Engine::Postgres) => {
                    if timezone.is_none() {
                        Ok(format!(
                            "TO_TIMESTAMP({}) AT TIME ZONE 'UTC'",
                            *value as f64 / 1_000_000.0
                        ))
                    } else {
                        Ok(format!("TO_TIMESTAMP({})", *value as f64 / 1_000_000.0))
                    }
                }
                Some(Engine::MySQL) => {
                    Ok(format!("FROM_UNIXTIME({})", *value as f64 / 1_000_000.0))
                }
                _ => Ok(format!("TO_TIMESTAMP({})", value / 1_000_000)),
            },
            ScalarValue::TimestampMillisecond(Some(value), timezone) => match engine {
                Some(Engine::SQLite) => Ok(format!("datetime({}, 'unixepoch')", value / 1000)),
                Some(Engine::Postgres) => {
                    if timezone.is_none() {
                        Ok(format!(
                            "TO_TIMESTAMP({}) AT TIME ZONE 'UTC'",
                            *value as f64 / 1000.0
                        ))
                    } else {
                        Ok(format!("TO_TIMESTAMP({})", *value as f64 / 1000.0))
                    }
                }
                _ => Ok(format!("TO_TIMESTAMP({})", value / 1000)),
            },
            ScalarValue::TimestampSecond(Some(value), timezone) => match engine {
                Some(Engine::SQLite) => Ok(format!("datetime({value}, 'unixepoch')")),
                Some(Engine::Postgres) => {
                    if timezone.is_none() {
                        Ok(format!("TO_TIMESTAMP({value}) AT TIME ZONE 'UTC'"))
                    } else {
                        Ok(format!("TO_TIMESTAMP({value})"))
                    }
                }
                _ => Ok(format!("TO_TIMESTAMP({value})")),
            },
            ScalarValue::Decimal128(Some(v), _, s) => {
                let decimal = BigDecimal::new(BigInt::from(*v), *s as i64);
                Ok(decimal.to_string())
            }
            _ => Err(Error::UnsupportedFilterExpr {
                expr: format!("{expr}"),
            }),
        },
        Expr::Like(like_expr) => {
            if like_expr.escape_char.is_some() {
                // Escape char is not supported
                return Err(Error::UnsupportedFilterExpr {
                    expr: format!("{expr}"),
                });
            }
            let expr = render_expr(&like_expr.expr, engine)?;
            let pattern = render_expr(&like_expr.pattern, engine)?;

            let mut op_and_pattern = match (engine, like_expr.case_insensitive) {
                (Some(Engine::Postgres | Engine::DuckDB), true) => format!("ILIKE {pattern}"),
                (Some(Engine::Postgres | Engine::DuckDB), false) => format!("LIKE {pattern}"),
                (Some(Engine::SQLite), true) => format!("LIKE {pattern}"),
                (Some(Engine::SQLite), false) => format!("LIKE {pattern} COLLATE BINARY"),
                // Standard SQL LIKE for unknown/unspecified engines (e.g. FlightSQL federation)
                (None | Some(Engine::MySQL | Engine::Spark | Engine::ODBC), false) => {
                    format!("LIKE {pattern}")
                }
                // Case-insensitive LIKE via UPPER() for engines without ILIKE
                (None | Some(Engine::MySQL | Engine::Spark | Engine::ODBC), true) => {
                    return Ok(format!("UPPER({expr}) LIKE UPPER({pattern})"));
                }
            };

            if like_expr.negated {
                op_and_pattern = format!("NOT {}", op_and_pattern)
            };

            Ok(format!("{expr} {op_and_pattern}"))
        }
        Expr::InList(in_list) => {
            let expr = render_expr(&in_list.expr, engine)?;
            let list = in_list
                .list
                .iter()
                .map(|expr| render_expr(expr, engine))
                .collect::<Result<Vec<String>>>()?;

            let op = if in_list.negated { "NOT IN" } else { "IN" };

            Ok(format!("{expr} {op} ({list})", list = list.join(", ")))
        }
        Expr::ScalarFunction(scalar_function) => {
            let args = scalar_function
                .args
                .iter()
                .map(|expr| render_expr(expr, engine))
                .collect::<Result<Vec<String>>>()?;

            Ok(format!("{}({})", scalar_function.name(), args.join(", ")))
        }
        _ => Err(Error::UnsupportedFilterExpr {
            expr: format!("{expr}"),
        }),
    }
}

pub fn to_sql(expr: &Expr) -> Result<String> {
    to_sql_with_engine(expr, None)
}

/// Parenthesize `rendered` when `expr` is itself a binary expression, so that composing it into a
/// larger expression cannot lose the grouping the `Expr` tree carries.
///
/// Left to SQL operator precedence, the grouping is re-derived and can differ: `a * (b + c)`
/// rendered as `a * b + c` is a different value that binds and returns silently; `x - (t + d)`
/// rendered as `x - t + d` means `(x - t) + d`, since `-` and `+` share a precedence level and
/// associate left; and `flag = (ts < t)` rendered as `flag = ts < t` is a chained comparison,
/// which DuckDB rejects outright.
fn group_if_binary(expr: &Expr, rendered: &str) -> String {
    if matches!(expr, Expr::BinaryExpr(_)) {
        format!("({rendered})")
    } else {
        rendered.to_string()
    }
}

/// Whether `expr` is a timestamp-valued operand under [`Engine::DuckDB`] — that is, whether it
/// renders as a term DuckDB types as `TIMESTAMPTZ`.
///
/// This is deliberately a test of what the expression *is*, not of what its SQL looks like. A
/// nested comparison the normalization has already rewritten also renders with a `TO_TIMESTAMP`
/// prefix, yet it is a boolean; conflating the two is what made the normalization emit
/// `EPOCH_MS(<predicate>)`.
///
/// The leaf shapes mirror the arms that produce a `TIMESTAMPTZ` rendering — the date and
/// timestamp [`ScalarValue`] literals, and [`handle_cast`]'s DuckDB arm for a cast to a
/// timestamp. Keep them in step with those arms.
///
/// Timestamp arithmetic is timestamp-valued in turn, following DuckDB's own typing: adding an
/// interval to a timestamp yields a timestamp either way round, and subtracting one does too,
/// while subtracting a timestamp *from* a timestamp yields an interval instead. Both operands
/// can be columns, whose type is not visible here, so this is decided from the operator and the
/// operands' own shapes.
fn is_duckdb_timestamp_operand(expr: &Expr) -> bool {
    match expr {
        Expr::Literal(value, _) => matches!(
            value,
            ScalarValue::Date32(Some(_))
                | ScalarValue::Date64(Some(_))
                | ScalarValue::TimestampSecond(Some(_), _)
                | ScalarValue::TimestampMillisecond(Some(_), _)
                | ScalarValue::TimestampMicrosecond(Some(_), _)
                | ScalarValue::TimestampNanosecond(Some(_), _)
        ),
        Expr::Cast(cast) => matches!(
            cast.field.data_type(),
            arrow::datatypes::DataType::Timestamp(_, _)
        ),
        Expr::BinaryExpr(binary_expr) => match binary_expr.op {
            Operator::Plus => {
                is_duckdb_timestamp_operand(&binary_expr.left)
                    || is_duckdb_timestamp_operand(&binary_expr.right)
            }
            Operator::Minus => {
                is_duckdb_timestamp_operand(&binary_expr.left)
                    && !is_duckdb_timestamp_operand(&binary_expr.right)
            }
            _ => false,
        },
        _ => false,
    }
}

fn handle_cast(cast: &Cast, engine: Option<Engine>, expr: &Expr) -> Result<String> {
    match cast.field.data_type() {
        arrow::datatypes::DataType::Timestamp(_, Some(_) | None) => match engine {
            Some(Engine::ODBC) => Ok(format!(
                "CAST({} AS TIMESTAMP)",
                render_expr(&cast.expr, engine)?,
            )),
            // This needs to match the timestamp conversion below
            Some(Engine::DuckDB) => Ok(format!(
                "TO_TIMESTAMP(EPOCH(CAST({} AS TIMESTAMP)))",
                render_expr(&cast.expr, engine)?,
            )),
            Some(Engine::SQLite) => Ok(format!(
                "datetime({}, 'subsec', 'utc')",
                render_expr(&cast.expr, engine)?,
            )),
            Some(Engine::Spark) => EngineNotSupportedForExpressionSnafu {
                engine: "Spark".to_string(),
                expr: format!("{expr}"),
            }
            .fail()?,
            _ => Ok(format!(
                "CAST({} AS TIMESTAMPTZ)",
                render_expr(&cast.expr, engine)?,
            )),
        },
        arrow::datatypes::DataType::Int64 => Ok(format!(
            "CAST({} AS BIGINT)",
            render_expr(&cast.expr, engine)?,
        )),
        _ => Err(Error::UnsupportedFilterExpr {
            expr: format!("{expr}"),
        }),
    }
}

// Helper function to check if expression contains subquery or outer reference
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};

pub(super) fn expr_contains_subquery_or_outer_ref(expr: &Expr) -> datafusion::error::Result<bool> {
    let mut found = false;
    expr.apply(|expr| match expr {
        Expr::ScalarSubquery(_)
        | Expr::InSubquery(_)
        | Expr::Exists(_)
        | Expr::OuterReferenceColumn(_, _) => {
            found = true;
            Ok(TreeNodeRecursion::Stop)
        }
        _ => Ok(TreeNodeRecursion::Continue),
    })?;
    Ok(found)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use arrow::datatypes::DataType;
    use datafusion::{
        logical_expr::{
            expr::{InList, ScalarFunction},
            ColumnarValue, Expr, Like, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
            Volatility,
        },
        prelude::col,
        scalar::ScalarValue,
    };

    #[test]
    fn test_like_expr_to_sql() -> Result<()> {
        for (engine, case_insensitive, negated, expected) in [
            (Engine::Postgres, false, false, "\"name\" LIKE '%John%'"),
            (Engine::Postgres, true, true, "\"name\" NOT ILIKE '%John%'"),
            (Engine::DuckDB, false, false, "\"name\" LIKE '%John%'"),
            (Engine::DuckDB, true, true, "\"name\" NOT ILIKE '%John%'"),
            (
                Engine::SQLite,
                false,
                false,
                "\"name\" LIKE '%John%' COLLATE BINARY",
            ),
            (Engine::SQLite, true, true, "\"name\" NOT LIKE '%John%'"),
        ] {
            let expr = Expr::Like(Like {
                expr: Box::new(col("name")),
                pattern: Box::new(Expr::Literal(
                    ScalarValue::Utf8(Some("%John%".to_string())),
                    None,
                )),
                case_insensitive,
                negated,
                escape_char: None,
            });

            let sql = to_sql_with_engine(&expr, Some(engine))?;
            assert_eq!(sql, expected);
        }
        Ok(())
    }

    #[test]
    fn test_utf8view_literal_to_sql() -> Result<()> {
        let expr = Expr::Literal(
            ScalarValue::Utf8View(Some("ECONOMY ANODIZED STEEL".to_string())),
            None,
        );
        assert_eq!(to_sql_with_engine(&expr, None)?, "'ECONOMY ANODIZED STEEL'");

        // Utf8View equality filter (common after DataFusion type coercion)
        let eq_expr = Expr::BinaryExpr(datafusion::logical_expr::BinaryExpr {
            left: Box::new(col("p_type")),
            op: datafusion::logical_expr::Operator::Eq,
            right: Box::new(Expr::Literal(
                ScalarValue::Utf8View(Some("ECONOMY ANODIZED STEEL".to_string())),
                None,
            )),
        });
        assert_eq!(to_sql(&eq_expr)?, "\"p_type\" = 'ECONOMY ANODIZED STEEL'");

        Ok(())
    }

    #[test]
    fn test_utf8view_inlist_to_sql() -> Result<()> {
        let expr = Expr::InList(InList {
            expr: Box::new(col("l_shipmode")),
            list: vec![
                Expr::Literal(ScalarValue::Utf8View(Some("MAIL".to_string())), None),
                Expr::Literal(ScalarValue::Utf8View(Some("SHIP".to_string())), None),
            ],
            negated: false,
        });
        assert_eq!(
            to_sql_with_engine(&expr, None)?,
            "\"l_shipmode\" IN ('MAIL', 'SHIP')"
        );

        Ok(())
    }

    #[test]
    fn test_like_expr_no_engine_to_sql() -> Result<()> {
        // LIKE with no engine should produce standard SQL LIKE
        let expr = Expr::Like(Like {
            expr: Box::new(col("name")),
            pattern: Box::new(Expr::Literal(
                ScalarValue::Utf8(Some("%John%".to_string())),
                None,
            )),
            case_insensitive: false,
            negated: false,
            escape_char: None,
        });
        assert_eq!(to_sql(&expr)?, "\"name\" LIKE '%John%'");

        // NOT LIKE with no engine
        let expr_negated = Expr::Like(Like {
            expr: Box::new(col("name")),
            pattern: Box::new(Expr::Literal(
                ScalarValue::Utf8(Some("MEDIUM POLISHED%".to_string())),
                None,
            )),
            case_insensitive: false,
            negated: true,
            escape_char: None,
        });
        assert_eq!(
            to_sql(&expr_negated)?,
            "\"name\" NOT LIKE 'MEDIUM POLISHED%'"
        );

        // Case-insensitive LIKE with no engine uses UPPER()
        let expr_ci = Expr::Like(Like {
            expr: Box::new(col("name")),
            pattern: Box::new(Expr::Literal(
                ScalarValue::Utf8(Some("%john%".to_string())),
                None,
            )),
            case_insensitive: true,
            negated: false,
            escape_char: None,
        });
        assert_eq!(to_sql(&expr_ci)?, "UPPER(\"name\") LIKE UPPER('%john%')");

        Ok(())
    }

    #[test]
    fn test_like_expr_utf8view_no_engine_to_sql() -> Result<()> {
        // LIKE with Utf8View pattern and no engine
        let expr = Expr::Like(Like {
            expr: Box::new(col("p_name")),
            pattern: Box::new(Expr::Literal(
                ScalarValue::Utf8View(Some("forest%".to_string())),
                None,
            )),
            case_insensitive: false,
            negated: false,
            escape_char: None,
        });
        assert_eq!(to_sql(&expr)?, "\"p_name\" LIKE 'forest%'");

        Ok(())
    }

    #[test]
    fn test_decimal128_literal_to_sql() -> Result<()> {
        let expr = Expr::Literal(ScalarValue::Decimal128(Some(1234567890), 38, 2), None);
        assert_eq!(to_sql_with_engine(&expr, None)?, "12345678.90");

        let expr_negative = Expr::Literal(ScalarValue::Decimal128(Some(-1234567890), 38, 4), None);
        assert_eq!(to_sql_with_engine(&expr_negative, None)?, "-123456.7890");

        let expr_int = Expr::Literal(ScalarValue::Decimal128(Some(1234567890), 38, 0), None);
        assert_eq!(to_sql_with_engine(&expr_int, None)?, "1234567890");

        Ok(())
    }

    #[test]
    fn test_expr_inlist_to_sql() -> Result<()> {
        let expr = Expr::InList(InList {
            expr: Box::new(col("a")),
            list: vec![
                Expr::Literal(ScalarValue::Int32(Some(1)), None),
                Expr::Literal(ScalarValue::Int32(Some(2)), None),
                Expr::Literal(ScalarValue::Int32(Some(3)), None),
            ],
            negated: false,
        });
        assert_eq!(to_sql_with_engine(&expr, None)?, "\"a\" IN (1, 2, 3)");

        let expr_negated = Expr::InList(InList {
            expr: Box::new(col("a")),
            list: vec![
                Expr::Literal(ScalarValue::Int32(Some(4)), None),
                Expr::Literal(ScalarValue::Int32(Some(5)), None),
                Expr::Literal(ScalarValue::Int32(Some(6)), None),
            ],
            negated: true,
        });

        assert_eq!(
            to_sql_with_engine(&expr_negated, None)?,
            "\"a\" NOT IN (4, 5, 6)"
        );

        Ok(())
    }

    #[test]
    fn test_expr_scalar_function_to_sql() -> Result<()> {
        #[derive(Debug, Hash, Eq, PartialEq)]
        struct TestScalarUDF {
            signature: Signature,
        }
        impl ScalarUDFImpl for TestScalarUDF {
            fn name(&self) -> &str {
                "substring"
            }

            fn signature(&self) -> &Signature {
                &self.signature
            }

            fn return_type(&self, _arg_types: &[DataType]) -> datafusion::error::Result<DataType> {
                Ok(DataType::Utf8)
            }

            fn invoke_with_args(
                &self,
                _args: ScalarFunctionArgs,
            ) -> datafusion::error::Result<ColumnarValue> {
                Ok(ColumnarValue::Scalar(ScalarValue::from("a")))
            }
        }
        let substring_udf = Arc::new(ScalarUDF::from(TestScalarUDF {
            signature: Signature::uniform(
                3,
                vec![DataType::Utf8, DataType::Int32, DataType::Int32],
                Volatility::Stable,
            ),
        }));

        let expr = Expr::ScalarFunction(ScalarFunction {
            func: substring_udf,
            args: vec![
                Expr::Literal(ScalarValue::Utf8(Some("hello world".to_string())), None),
                Expr::Literal(ScalarValue::Int32(Some(1)), None),
                Expr::Literal(ScalarValue::Int32(Some(5)), None),
            ],
        });
        assert_eq!(
            to_sql_with_engine(&expr, None)?,
            "substring('hello world', 1, 5)"
        );

        Ok(())
    }

    fn binary(left: Expr, op: Operator, right: Expr) -> Expr {
        Expr::BinaryExpr(datafusion::logical_expr::BinaryExpr {
            left: Box::new(left),
            op,
            right: Box::new(right),
        })
    }

    fn int(v: i32) -> Expr {
        Expr::Literal(ScalarValue::Int32(Some(v)), None)
    }

    #[test]
    fn test_and_or_binary_exprs_are_parenthesized() -> Result<()> {
        // AND operands are wrapped so the tree shape is preserved in SQL
        let and_expr = binary(col("a"), Operator::And, col("b"));
        assert_eq!(to_sql(&and_expr)?, "(\"a\") AND (\"b\")");

        // OR operands are wrapped for the same reason
        let or_expr = binary(col("a"), Operator::Or, col("b"));
        assert_eq!(to_sql(&or_expr)?, "(\"a\") OR (\"b\")");

        Ok(())
    }

    #[test]
    fn test_non_boolean_binary_exprs_not_parenthesized() -> Result<()> {
        // Comparison and arithmetic operators must NOT add extra parens
        let eq_expr = binary(col("k1"), Operator::Eq, int(1));
        assert_eq!(to_sql(&eq_expr)?, "\"k1\" = 1");

        let lt_expr = binary(col("v"), Operator::Lt, int(42));
        assert_eq!(to_sql(&lt_expr)?, "\"v\" < 42");

        let plus_expr = binary(col("x"), Operator::Plus, int(5));
        assert_eq!(to_sql(&plus_expr)?, "\"x\" + 5");

        Ok(())
    }

    #[test]
    fn test_composite_key_or_chain_parenthesized() -> Result<()> {
        // Simulates a two-row composite-key DELETE:
        //   (k1 = 1 AND k2 = 2) OR (k1 = 2 AND k2 = 4)
        //
        // Without parens this serialises as a flat chain that DuckDB's optimizer
        // reconstructs as a left-recursive tree of depth N, causing a stack overflow
        // for large N. With parens the structure is explicit.
        let row1 = binary(
            binary(col("k1"), Operator::Eq, int(1)),
            Operator::And,
            binary(col("k2"), Operator::Eq, int(2)),
        );
        let row2 = binary(
            binary(col("k1"), Operator::Eq, int(2)),
            Operator::And,
            binary(col("k2"), Operator::Eq, int(4)),
        );
        let or_chain = binary(row1, Operator::Or, row2);

        assert_eq!(
            to_sql(&or_chain)?,
            "((\"k1\" = 1) AND (\"k2\" = 2)) OR ((\"k1\" = 2) AND (\"k2\" = 4))"
        );

        Ok(())
    }

    #[test]
    fn test_expr_timestamp_scalar_value_to_sql() -> Result<()> {
        let expr = Expr::Literal(
            ScalarValue::TimestampNanosecond(Some(1_693_219_803_001_000_000), None),
            None,
        );
        assert_eq!(
            to_sql_with_engine(&expr, Some(Engine::Postgres))?,
            "TO_TIMESTAMP(1693219803.001) AT TIME ZONE 'UTC'"
        );
        let expr = Expr::Literal(
            ScalarValue::TimestampNanosecond(
                Some(1_693_219_803_001_000_000),
                Some(Arc::from("+10:00")),
            ),
            None,
        );
        assert_eq!(
            to_sql_with_engine(&expr, Some(Engine::Postgres))?,
            "TO_TIMESTAMP(1693219803.001)"
        );

        let expr = Expr::Literal(
            ScalarValue::TimestampMicrosecond(Some(1_693_219_803_001_000), None),
            None,
        );
        assert_eq!(
            to_sql_with_engine(&expr, Some(Engine::Postgres))?,
            "TO_TIMESTAMP(1693219803.001) AT TIME ZONE 'UTC'"
        );
        let expr = Expr::Literal(
            ScalarValue::TimestampMicrosecond(
                Some(1_693_219_803_001_000),
                Some(Arc::from("+10:00")),
            ),
            None,
        );
        assert_eq!(
            to_sql_with_engine(&expr, Some(Engine::Postgres))?,
            "TO_TIMESTAMP(1693219803.001)"
        );

        let expr = Expr::Literal(
            ScalarValue::TimestampMillisecond(Some(1_693_219_803_001), None),
            None,
        );
        assert_eq!(
            to_sql_with_engine(&expr, Some(Engine::Postgres))?,
            "TO_TIMESTAMP(1693219803.001) AT TIME ZONE 'UTC'"
        );
        let expr = Expr::Literal(
            ScalarValue::TimestampMillisecond(Some(1_693_219_803_001), Some(Arc::from("+10:00"))),
            None,
        );
        assert_eq!(
            to_sql_with_engine(&expr, Some(Engine::Postgres))?,
            "TO_TIMESTAMP(1693219803.001)"
        );

        let expr = Expr::Literal(
            ScalarValue::TimestampSecond(Some(1_693_219_803), None),
            None,
        );
        assert_eq!(
            to_sql_with_engine(&expr, Some(Engine::Postgres))?,
            "TO_TIMESTAMP(1693219803) AT TIME ZONE 'UTC'"
        );
        let expr = Expr::Literal(
            ScalarValue::TimestampSecond(Some(1_693_219_803), Some(Arc::from("+10:00"))),
            None,
        );
        assert_eq!(
            to_sql_with_engine(&expr, Some(Engine::Postgres))?,
            "TO_TIMESTAMP(1693219803)"
        );

        Ok(())
    }

    fn bool_lit(v: bool) -> Expr {
        Expr::Literal(ScalarValue::Boolean(Some(v)), None)
    }

    fn int_lit(v: i64) -> Expr {
        Expr::Literal(ScalarValue::Int64(Some(v)), None)
    }

    fn ts_lit() -> Expr {
        Expr::Literal(
            ScalarValue::TimestampNanosecond(Some(1_767_225_600_000_000_000), None),
            None,
        )
    }

    /// A logical connective is not a timestamp comparison. The DuckDB timestamp
    /// normalization must not fire on `AND`/`OR`: the left operand of a connective is a
    /// predicate, and wrapping a predicate in `EPOCH_MS` makes DuckDB's binder reject the
    /// whole statement with `Binder Error: epoch_ms(BOOLEAN)`.
    #[test]
    fn test_duckdb_or_with_timestamp_right_operand_is_not_rewritten() {
        let expr = col("flag").eq(bool_lit(true)).or(col("ts").lt(ts_lit()));

        let sql = to_sql_with_engine(&expr, Some(Engine::DuckDB)).expect("to unparse");

        assert!(
            !sql.contains("EPOCH_MS(\"flag\""),
            "the boolean operand of OR must not be wrapped in EPOCH_MS: {sql}"
        );
        assert_eq!(
            sql,
            "(\"flag\" = true) OR (TO_TIMESTAMP(EPOCH_MS(\"ts\") / 1000) < TO_TIMESTAMP(1767225600))"
        );
    }

    #[test]
    fn test_duckdb_and_with_timestamp_right_operand_is_not_rewritten() {
        let expr = col("flag").eq(bool_lit(true)).and(col("ts").lt(ts_lit()));

        let sql = to_sql_with_engine(&expr, Some(Engine::DuckDB)).expect("to unparse");

        assert!(
            !sql.contains("EPOCH_MS(\"flag\""),
            "the boolean operand of AND must not be wrapped in EPOCH_MS: {sql}"
        );
        assert_eq!(
            sql,
            "(\"flag\" = true) AND (TO_TIMESTAMP(EPOCH_MS(\"ts\") / 1000) < TO_TIMESTAMP(1767225600))"
        );
    }

    /// The left operand of the outer connective is itself a connective, so it renders
    /// parenthesised rather than as a bare predicate — it must still not be wrapped.
    #[test]
    fn test_duckdb_nested_connective_operand_is_not_rewritten() {
        let inner = col("x").eq(int_lit(1)).and(col("y").eq(int_lit(2)));
        let expr = inner.or(col("ts").lt(ts_lit()));

        let sql = to_sql_with_engine(&expr, Some(Engine::DuckDB)).expect("to unparse");

        assert!(
            !sql.contains("EPOCH_MS((") && !sql.contains("EPOCH_MS(\"x\""),
            "a predicate operand must not be wrapped in EPOCH_MS: {sql}"
        );
    }

    /// The normalization must be preserved for a genuine timestamp comparison — this is
    /// the case the rewrite exists to serve.
    #[test]
    fn test_duckdb_timestamp_comparison_is_still_rewritten() {
        let expr = col("ts").lt(ts_lit());

        let sql = to_sql_with_engine(&expr, Some(Engine::DuckDB)).expect("to unparse");

        assert_eq!(
            sql,
            "TO_TIMESTAMP(EPOCH_MS(\"ts\") / 1000) < TO_TIMESTAMP(1767225600)"
        );
    }

    fn utf8(value: &str) -> Expr {
        Expr::Literal(ScalarValue::Utf8(Some(value.to_string())), None)
    }

    /// An apostrophe is ordinary in real data (names, addresses, free text). Rendered bare it
    /// closes the literal early and the statement fails to parse, so a `DELETE`/`UPDATE`
    /// filtering on such a value cannot run at all.
    #[test]
    fn test_apostrophe_in_string_literal_is_doubled() -> Result<()> {
        assert_eq!(to_sql(&utf8("O'Brien"))?, "'O''Brien'");

        // Every engine ends a literal on `'` and accepts the doubled form for an embedded one.
        for engine in [
            Engine::DuckDB,
            Engine::SQLite,
            Engine::Postgres,
            Engine::ODBC,
            Engine::MySQL,
            Engine::Spark,
        ] {
            assert_eq!(
                to_sql_with_engine(&utf8("O'Brien"), Some(engine))?,
                "'O''Brien'",
                "engine {engine:?} must escape the quote"
            );
        }

        Ok(())
    }

    /// A backslash is only significant inside a literal for the engines that read it as an escape
    /// character; the others must not gain an extra one, since doubling a literal backslash
    /// silently changes the value.
    #[test]
    fn test_backslash_is_escaped_only_where_it_is_significant() -> Result<()> {
        for engine in [Engine::MySQL, Engine::Spark] {
            assert_eq!(
                to_sql_with_engine(&utf8(r"C:\temp\"), Some(engine))?,
                r"'C:\\temp\\'",
                "engine {engine:?} must double the backslash"
            );
        }

        for engine in [
            None,
            Some(Engine::DuckDB),
            Some(Engine::SQLite),
            Some(Engine::Postgres),
            Some(Engine::ODBC),
        ] {
            assert_eq!(
                to_sql_with_engine(&utf8(r"C:\temp\"), engine)?,
                r"'C:\temp\'",
                "engine {engine:?} must leave the backslash alone"
            );
        }

        Ok(())
    }

    /// A backslash immediately before a quote is the composition quote doubling alone cannot
    /// handle: on a backslash-escaping engine `'\''` would be `\'` followed by a closing quote,
    /// which is worse than not escaping. Both directions are pinned here.
    #[test]
    fn test_backslash_before_a_quote_is_escaped_per_engine() -> Result<()> {
        // `\` ordinary: `''` is the escaped quote and the whole value stays inside the literal.
        assert_eq!(to_sql(&utf8(r"\' OR 1=1 --"))?, r"'\'' OR 1=1 --'");

        // `\` significant: it is doubled first, so it can no longer consume the escaped quote.
        assert_eq!(
            to_sql_with_engine(&utf8(r"\' OR 1=1 --"), Some(Engine::MySQL))?,
            r"'\\'' OR 1=1 --'"
        );

        Ok(())
    }

    /// `LargeUtf8`/`Utf8View` share the arm — DataFusion's type coercion picks between them, so
    /// escaping only one variant would leave the defect reachable.
    #[test]
    fn test_every_string_scalar_variant_is_escaped() -> Result<()> {
        for value in [
            ScalarValue::Utf8(Some("it's".to_string())),
            ScalarValue::LargeUtf8(Some("it's".to_string())),
            ScalarValue::Utf8View(Some("it's".to_string())),
        ] {
            assert_eq!(to_sql(&Expr::Literal(value, None))?, "'it''s'");
        }

        Ok(())
    }

    /// The unbounded-deletion shape: rendered bare, `WHERE "name" = 'x' OR 1=1 --'` binds as a
    /// tautology and a `DELETE` matches every row instead of none.
    #[test]
    fn test_quote_bearing_value_stays_a_single_literal() -> Result<()> {
        let expr = col("name").eq(utf8("x' OR 1=1 --"));

        assert_eq!(to_sql(&expr)?, "\"name\" = 'x'' OR 1=1 --'");

        Ok(())
    }

    /// Every construct that carries a string value routes through the same literal arm.
    #[test]
    fn test_quotes_are_escaped_in_like_and_in_list() -> Result<()> {
        let like = Expr::Like(Like {
            expr: Box::new(col("name")),
            pattern: Box::new(utf8("%O'B%")),
            case_insensitive: false,
            negated: false,
            escape_char: None,
        });
        assert_eq!(to_sql(&like)?, "\"name\" LIKE '%O''B%'");

        let in_list = Expr::InList(InList {
            expr: Box::new(col("name")),
            list: vec![utf8("O'Brien"), utf8("plain")],
            negated: false,
        });
        assert_eq!(to_sql(&in_list)?, "\"name\" IN ('O''Brien', 'plain')");

        Ok(())
    }

    #[test]
    fn test_string_literal_edge_cases() -> Result<()> {
        assert_eq!(to_sql(&utf8(""))?, "''");
        assert_eq!(to_sql(&utf8("'"))?, "''''");
        assert_eq!(to_sql(&utf8("''"))?, "''''''");
        assert_eq!(to_sql(&utf8("a'b'c"))?, "'a''b''c'");

        // A backslash is an ordinary character for the engines reached through this function,
        // so it must pass through unchanged rather than be doubled.
        assert_eq!(to_sql(&utf8(r"a\b"))?, r"'a\b'");

        Ok(())
    }

    /// An Arrow field name may contain a double quote, which would otherwise close the
    /// identifier early and leave the rest of the name as SQL text.
    #[test]
    fn test_quote_in_column_name_is_doubled() -> Result<()> {
        let column = Expr::Column(datafusion::common::Column::new_unqualified("we\"ird"));
        let expr = column.eq(utf8("v"));

        assert_eq!(to_sql(&expr)?, "\"we\"\"ird\" = 'v'");

        Ok(())
    }

    /// A value with no quote in it must render exactly as before — the escape is a no-op for
    /// the overwhelmingly common case.
    #[test]
    fn test_ordinary_values_are_unchanged() -> Result<()> {
        assert_eq!(to_sql(&col("a").eq(utf8("plain")))?, "\"a\" = 'plain'");
        assert_eq!(to_sql(&col("a").eq(int(1)))?, "\"a\" = 1");

        Ok(())
    }

    /// A qualified column must render as the column it names. Rendering `Column`'s flat name made
    /// `t.a` one identifier, which either fails to bind or — where a column of that literal name
    /// exists — binds the wrong one.
    #[test]
    fn a_qualified_column_renders_as_its_own_column() -> Result<()> {
        let qualified = Expr::Column(datafusion::common::Column::new(Some("t"), "a"));

        for (engine, expected) in [
            (None, "\"a\""),
            (Some(Engine::DuckDB), "\"a\""),
            (Some(Engine::SQLite), "\"a\""),
            (Some(Engine::Postgres), "\"a\""),
            (Some(Engine::MySQL), "\"a\""),
            (Some(Engine::ODBC), "a"),
            (Some(Engine::Spark), "a"),
        ] {
            assert_eq!(
                to_sql_with_engine(&qualified, engine)?,
                expected,
                "engine {engine:?}"
            );
        }

        Ok(())
    }

    /// A predicate spanning two relations cannot be rendered for a single-table statement, and
    /// rendering it by column name alone would *widen* it into `"id" = "id"` — a tautology that a
    /// `DELETE` applies to every row. It has to be refused, not erased.
    #[test]
    fn a_cross_relation_predicate_is_refused() {
        let left = Expr::Column(datafusion::common::Column::new(Some("t1"), "id"));
        let right = Expr::Column(datafusion::common::Column::new(Some("t2"), "id"));

        let err = to_sql(&left.eq(right)).expect_err("a cross-relation predicate must be refused");
        assert!(
            matches!(err, Error::UnsupportedFilterExpr { .. }),
            "unexpected error: {err}"
        );
    }

    /// One relation used by two columns is not a cross-relation predicate — this is the shape a
    /// planner-produced filter over two columns of the same table has.
    #[test]
    fn one_relation_used_twice_renders() -> Result<()> {
        let left = Expr::Column(datafusion::common::Column::new(Some("t"), "a"));
        let right = Expr::Column(datafusion::common::Column::new(Some("t"), "b"));

        assert_eq!(to_sql(&left.eq(right))?, "\"a\" = \"b\"");

        Ok(())
    }

    /// Mixed qualification of one table cannot be told apart from two tables without the session's
    /// default catalog and schema, so it is refused rather than resolved.
    #[test]
    fn mixed_qualification_of_one_relation_is_refused() {
        let bare = Expr::Column(datafusion::common::Column::new(Some("t"), "a"));
        let partial = Expr::Column(datafusion::common::Column::new(
            Some(datafusion::sql::TableReference::partial("public", "t")),
            "b",
        ));

        assert!(matches!(
            to_sql(&bare.eq(partial)),
            Err(Error::UnsupportedFilterExpr { .. })
        ));
    }

    /// `TableReference::resolved_eq` treats an omitted component as matching, so a bare reference is
    /// `resolved_eq` to two different fully qualified tables. Comparing against one arbitrarily
    /// chosen pivot under those semantics would admit this predicate — over columns of two distinct
    /// catalogs — and then erase both qualifiers into `"id" = "id"`.
    #[test]
    fn two_catalogs_bridged_by_a_bare_reference_are_refused() {
        let left = Expr::Column(datafusion::common::Column::new(
            Some(datafusion::sql::TableReference::full("a", "public", "t")),
            "id",
        ));
        let bridge = Expr::Column(datafusion::common::Column::new(Some("t"), "id"));
        let right = Expr::Column(datafusion::common::Column::new(
            Some(datafusion::sql::TableReference::full("b", "public", "t")),
            "id",
        ));

        // Every ordering of the three references must refuse, whatever order `column_refs` yields.
        for predicate in [
            left.clone()
                .eq(bridge.clone())
                .and(right.clone().eq(bridge.clone())),
            bridge
                .clone()
                .eq(left.clone())
                .and(bridge.clone().eq(right.clone())),
            right.eq(bridge.clone()).and(bridge.eq(left)),
        ] {
            assert!(
                matches!(to_sql(&predicate), Err(Error::UnsupportedFilterExpr { .. })),
                "accepted a predicate spanning two catalogs: {predicate}"
            );
        }
    }

    /// The qualifier is what gets dropped, not every dot: a column whose own name contains a dot is
    /// a legal Arrow field name and must still render as itself. This is also the column a
    /// qualified `t.a` would have bound to had it kept rendering flat.
    #[test]
    fn an_unqualified_dotted_column_name_is_not_a_qualifier() -> Result<()> {
        let dotted = Expr::Column(datafusion::common::Column::new_unqualified("t.a"));

        assert_eq!(to_sql(&dotted.clone().eq(int(1)))?, "\"t.a\" = 1");
        assert_eq!(
            to_sql_with_engine(&dotted, Some(Engine::DuckDB))?,
            "\"t.a\""
        );

        Ok(())
    }

    /// The qualifier has to be dropped wherever the column sits, not only at the top of the tree.
    #[test]
    fn a_qualified_column_inside_a_predicate_renders_as_its_own_column() -> Result<()> {
        let predicate = Expr::Column(datafusion::common::Column::new(Some("t"), "a"))
            .eq(int(1))
            .and(col("b").eq(utf8("x")));

        assert_eq!(to_sql(&predicate)?, "(\"a\" = 1) AND (\"b\" = 'x')");

        Ok(())
    }

    /// An unqualified column keeps rendering exactly as before, for both the quoted and the bare
    /// engine arms.
    #[test]
    fn an_unqualified_column_renders_unchanged() -> Result<()> {
        assert_eq!(to_sql(&col("a"))?, "\"a\"");
        assert_eq!(
            to_sql_with_engine(&col("a"), Some(Engine::DuckDB))?,
            "\"a\""
        );
        assert_eq!(to_sql_with_engine(&col("a"), Some(Engine::Spark))?, "a");
        assert_eq!(to_sql_with_engine(&col("a"), Some(Engine::ODBC))?, "a");

        Ok(())
    }

    /// `supports_filters_pushdown` is asked about the **qualified** predicate while `scan` receives
    /// the unqualified one, and `ClickHouseTable` answers that question by rendering here and
    /// mapping `Err(_)` to `Unsupported` (as does the runtime's DuckDB vector index). So this
    /// function must render the shape the probe passes it: refusing a qualifier would silently stop
    /// those predicates from reaching the remote statement, and the physical-pushdown fallback does
    /// not cover `LIKE`.
    ///
    /// The predicate is captured from a real `SessionContext` rather than hand-built, so the test
    /// keeps holding if `DataFusion` changes when it normalizes.
    #[tokio::test]
    async fn a_predicate_as_the_pushdown_probe_receives_it_still_renders() {
        use datafusion::catalog::{Session, TableProvider};
        use datafusion::logical_expr::{TableProviderFilterPushDown, TableType};
        use datafusion::physical_plan::{empty::EmptyExec, ExecutionPlan};
        use datafusion::prelude::SessionContext;
        use std::sync::Mutex;

        #[derive(Debug)]
        struct ProbeRecorder {
            schema: arrow::datatypes::SchemaRef,
            probed: Arc<Mutex<Vec<Expr>>>,
        }

        #[async_trait::async_trait]
        impl TableProvider for ProbeRecorder {
            fn schema(&self) -> arrow::datatypes::SchemaRef {
                Arc::clone(&self.schema)
            }
            fn table_type(&self) -> TableType {
                TableType::Base
            }
            fn supports_filters_pushdown(
                &self,
                filters: &[&Expr],
            ) -> datafusion::error::Result<Vec<TableProviderFilterPushDown>> {
                self.probed
                    .lock()
                    .expect("lock")
                    .extend(filters.iter().map(|f| (*f).clone()));
                Ok(vec![TableProviderFilterPushDown::Exact; filters.len()])
            }
            async fn scan(
                &self,
                _state: &dyn Session,
                _projection: Option<&Vec<usize>>,
                _filters: &[Expr],
                _limit: Option<usize>,
            ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
                Ok(Arc::new(EmptyExec::new(Arc::clone(&self.schema))))
            }
        }

        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("name", DataType::Utf8, true),
        ]));
        let probed = Arc::new(Mutex::new(Vec::new()));
        let ctx = SessionContext::new();
        ctx.register_table(
            "docs",
            Arc::new(ProbeRecorder {
                schema,
                probed: Arc::clone(&probed),
            }),
        )
        .expect("register");

        ctx.sql("SELECT * FROM docs WHERE name LIKE '%x%'")
            .await
            .expect("plan")
            .collect()
            .await
            .expect("collect");

        let filters = probed.lock().expect("lock").clone();
        assert!(!filters.is_empty(), "the pushdown probe saw no filter");
        for filter in &filters {
            assert!(
                filter
                    .column_refs()
                    .iter()
                    .any(|column| column.relation.is_some()),
                "the probe is expected to receive a qualified column: {filter:?}"
            );
            assert_eq!(
                to_sql_with_engine(filter, None).expect("the probed predicate must render"),
                "\"name\" LIKE '%x%'"
            );
        }
    }

    fn cast_to_timestamp(expr: Expr) -> Expr {
        Expr::Cast(Cast::new(
            Box::new(expr),
            DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, None),
        ))
    }

    /// A comparison whose right operand is itself a normalized comparison renders as
    /// `TO_TIMESTAMP(..)` without being a timestamp. Deciding by rendered text wrapped that
    /// boolean in `EPOCH_MS`, which has no `BOOLEAN` overload; deciding by `Expr` shape does
    /// not. Verified against DuckDB v1.5.5: the asserted SQL evaluates, and the text-sniffed
    /// form is rejected at parse time as a chained comparison, before the binder is reached.
    #[test]
    fn test_duckdb_predicate_compared_to_a_timestamp_comparison_is_not_normalized() {
        let expr = col("flag").eq(col("ts").lt(ts_lit()));

        let sql = to_sql_with_engine(&expr, Some(Engine::DuckDB)).expect("to unparse");

        assert!(
            !sql.contains("EPOCH_MS(\"flag\""),
            "a boolean operand must not be wrapped in EPOCH_MS: {sql}"
        );
        assert_eq!(
            sql,
            "\"flag\" = (TO_TIMESTAMP(EPOCH_MS(\"ts\") / 1000) < TO_TIMESTAMP(1767225600))"
        );
    }

    /// Subtraction between a timestamp column and a timestamp literal is normalized: DuckDB
    /// v1.5.5 rejects `"ts" - TO_TIMESTAMP(..)` with
    /// `No function matches '-(TIMESTAMP_MS, TIMESTAMP WITH TIME ZONE)'`, and accepts the
    /// normalized form, returning the interval between the two instants.
    #[test]
    fn test_duckdb_timestamp_subtraction_is_normalized() {
        let expr = col("ts") - ts_lit();

        let sql = to_sql_with_engine(&expr, Some(Engine::DuckDB)).expect("to unparse");

        assert_eq!(
            sql,
            "TO_TIMESTAMP(EPOCH_MS(\"ts\") / 1000) - TO_TIMESTAMP(1767225600)"
        );
    }

    /// A bare *right* operand is deliberately not normalized. Doing so would apply `EPOCH_MS`'s
    /// whole-millisecond truncation to a microsecond `TIMESTAMPTZ` column that compares exactly
    /// without it, turning a loud binder error into a silently shifted comparison. DuckDB
    /// rejects this shape, which is the safe outcome.
    #[test]
    fn test_duckdb_timestamp_literal_on_the_left_is_not_normalized() {
        let expr = ts_lit().gt(col("ts"));

        let sql = to_sql_with_engine(&expr, Some(Engine::DuckDB)).expect("to unparse");

        assert_eq!(sql, "TO_TIMESTAMP(1767225600) > \"ts\"");
    }

    /// Two timestamp operands are already comparable and must be left alone. This is the shape
    /// retention emits — a cast to a timestamp against a timestamp literal.
    #[test]
    fn test_duckdb_two_timestamp_operands_are_not_normalized() {
        let expr = cast_to_timestamp(col("ts")).lt(ts_lit());

        let sql = to_sql_with_engine(&expr, Some(Engine::DuckDB)).expect("to unparse");

        assert_eq!(
            sql,
            "TO_TIMESTAMP(EPOCH(CAST(\"ts\" AS TIMESTAMP))) < TO_TIMESTAMP(1767225600)"
        );
    }

    /// Adding an interval to a timestamp is itself timestamp-valued, so a comparison against it
    /// still normalizes the bare side. `"delta"` is an interval column, whose type is invisible
    /// here — the operand is recognised from the operator and the literal's shape. Verified
    /// against DuckDB v1.5.5: the asserted SQL evaluates, while leaving `"ts"` bare fails with
    /// `Cannot compare values of type TIMESTAMP_MS and type TIMESTAMP WITH TIME ZONE`.
    #[test]
    fn test_duckdb_comparison_against_timestamp_plus_interval_is_normalized() {
        let expr = col("ts").lt(ts_lit() + col("delta"));

        let sql = to_sql_with_engine(&expr, Some(Engine::DuckDB)).expect("to unparse");

        assert_eq!(
            sql,
            "TO_TIMESTAMP(EPOCH_MS(\"ts\") / 1000) < (TO_TIMESTAMP(1767225600) + \"delta\")"
        );
    }

    /// Subtracting an interval from a timestamp is timestamp-valued too, and the operands may be
    /// the other way round for `+`, since DuckDB accepts `INTERVAL + TIMESTAMPTZ` as well.
    #[test]
    fn test_duckdb_timestamp_interval_arithmetic_is_timestamp_valued() {
        for (expr, expected) in [
            (
                col("ts").lt(ts_lit() - col("delta")),
                "TO_TIMESTAMP(EPOCH_MS(\"ts\") / 1000) < (TO_TIMESTAMP(1767225600) - \"delta\")",
            ),
            (
                col("ts").lt(col("delta") + ts_lit()),
                "TO_TIMESTAMP(EPOCH_MS(\"ts\") / 1000) < (\"delta\" + TO_TIMESTAMP(1767225600))",
            ),
        ] {
            let sql = to_sql_with_engine(&expr, Some(Engine::DuckDB)).expect("to unparse");

            assert_eq!(sql, expected);
        }
    }

    /// The normalized rendering must group its right operand too. `-` and `+` share a precedence
    /// level and associate left, so `x - (t + d)` rendered flat as `x - t + d` means
    /// `(x - t) + d` — a different value, silently.
    #[test]
    fn test_duckdb_normalized_subtraction_groups_its_right_operand() {
        let expr = col("x") - (ts_lit() + col("delta"));

        let sql = to_sql_with_engine(&expr, Some(Engine::DuckDB)).expect("to unparse");

        assert_eq!(
            sql,
            "TO_TIMESTAMP(EPOCH_MS(\"x\") / 1000) - (TO_TIMESTAMP(1767225600) + \"delta\")"
        );
    }

    /// Subtracting a timestamp *from* a timestamp yields an interval, not a timestamp, so the
    /// bare operand opposite it must not be normalized — DuckDB would then be asked to compare a
    /// timestamp against an interval.
    #[test]
    fn test_duckdb_difference_of_two_timestamps_is_not_timestamp_valued() {
        let expr = col("ts").lt(ts_lit() - ts_lit());

        let sql = to_sql_with_engine(&expr, Some(Engine::DuckDB)).expect("to unparse");

        assert_eq!(
            sql,
            "\"ts\" < (TO_TIMESTAMP(1767225600) - TO_TIMESTAMP(1767225600))"
        );
    }

    /// `Minus` is in the normalization's operator set, but neither operand here is a timestamp,
    /// so ordinary arithmetic must not be pulled into the rewrite.
    #[test]
    fn test_duckdb_subtraction_without_a_timestamp_operand_is_not_normalized() {
        let expr = col("a") - col("b");

        let sql = to_sql_with_engine(&expr, Some(Engine::DuckDB)).expect("to unparse");

        assert_eq!(sql, "\"a\" - \"b\"");
    }

    /// A nested binary operand keeps its parentheses so the SQL means what the `Expr` meant.
    /// Rendered flat, `a * (b + c)` becomes `a * b + c` — a different value that binds and
    /// returns silently (10 rather than 14 for a=2, b=3, c=4).
    #[test]
    fn test_nested_arithmetic_keeps_its_grouping() {
        let expr = col("a") * (col("b") + col("c"));

        for engine in [
            None,
            Some(Engine::DuckDB),
            Some(Engine::Postgres),
            Some(Engine::SQLite),
            Some(Engine::MySQL),
        ] {
            let sql = to_sql_with_engine(&expr, engine).expect("to unparse");

            assert_eq!(sql, "\"a\" * (\"b\" + \"c\")", "engine {engine:?}");
        }
    }

    /// Subtraction is not associative, so grouping is load-bearing for `-` as well:
    /// `a - (b - c)` rendered flat is `(a - b) - c`.
    #[test]
    fn test_nested_subtraction_keeps_its_grouping() {
        let expr = col("a") - (col("b") - col("c"));

        let sql = to_sql(&expr).expect("to unparse");

        assert_eq!(sql, "\"a\" - (\"b\" - \"c\")");
    }

    /// The normalization is DuckDB-specific and must not leak into other engines.
    #[test]
    fn test_non_duckdb_engines_do_not_normalize_timestamps() {
        let expr = col("ts").lt(ts_lit());

        for engine in [None, Some(Engine::Postgres), Some(Engine::MySQL)] {
            let sql = to_sql_with_engine(&expr, engine).expect("to unparse");

            assert!(
                !sql.contains("EPOCH_MS"),
                "engine {engine:?} must not normalize: {sql}"
            );
        }
    }
}
