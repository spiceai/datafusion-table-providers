use std::sync::Arc;

use bigdecimal::{num_bigint::BigInt, BigDecimal};
use datafusion::{
    logical_expr::{Cast, Expr, Operator},
    scalar::ScalarValue,
    sql::unparser::dialect::{
        DefaultDialect, Dialect, DuckDBDialect, MySqlDialect, PostgreSqlDialect, SqliteDialect,
    },
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

/// Renders `value` as a SQL string literal, doubling any embedded single quote.
///
/// Interpolating the value bare lets a single quote close the literal early: an ordinary
/// apostrophe then makes the statement fail to parse, and a value shaped like `x' OR 1=1 --`
/// binds as a wider predicate than the caller wrote, so a `DELETE` can match rows the filter
/// excluded. Quote doubling is accepted by every engine this function renders for, and matches
/// what `DataFusion`'s own unparser emits.
///
/// `Engine::MySQL` and `Engine::Spark` additionally treat `\` as an escape character (MySQL
/// unless `NO_BACKSLASH_ESCAPES` is set), so a backslash-bearing value needs engine-aware
/// escaping on top of this. No caller renders either engine through this function today — both
/// reach their target through `SqlTable`, which unparses with a `Dialect`.
pub(crate) fn string_literal(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

/// Renders `name` as a double-quoted SQL identifier, doubling any embedded quote so that a
/// quote in a column name cannot close the identifier early.
pub(crate) fn quoted_identifier(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

#[allow(clippy::too_many_lines)]
#[allow(clippy::cast_precision_loss)]
pub fn to_sql_with_engine(expr: &Expr, engine: Option<Engine>) -> Result<String> {
    match expr {
        Expr::BinaryExpr(binary_expr) => {
            let left = to_sql_with_engine(&binary_expr.left, engine)?;
            let right = to_sql_with_engine(&binary_expr.right, engine)?;

            if let Some(Engine::DuckDB) = engine {
                // TODO: DuckDB doesn't support comparison between timestamp_s /timestamp_ms with timestampz as of v1
                // Revisit in future DuckDB versions
                //
                // Only a comparison can have a timestamp operand to normalize. Without this
                // guard the rewrite also fires on `AND`/`OR`, whose left operand is a
                // predicate, not a timestamp: it then emits `EPOCH_MS(<predicate>)` and
                // DuckDB rejects the statement with `Binder Error: epoch_ms(BOOLEAN)`.
                let is_timestamp_comparison = matches!(
                    binary_expr.op,
                    Operator::Eq
                        | Operator::NotEq
                        | Operator::Lt
                        | Operator::LtEq
                        | Operator::Gt
                        | Operator::GtEq
                        | Operator::IsDistinctFrom
                        | Operator::IsNotDistinctFrom
                );
                if is_timestamp_comparison
                    && right.starts_with("TO_TIMESTAMP")
                    && !left.starts_with("TO_TIMESTAMP")
                {
                    return Ok(format!(
                        "TO_TIMESTAMP(EPOCH_MS({}) / 1000) {} {}",
                        left, binary_expr.op, right
                    ));
                }
            }

            match binary_expr.op {
                Operator::And | Operator::Or => {
                    Ok(format!("({}) {} ({})", left, binary_expr.op, right))
                }
                _ => Ok(format!("{} {} {}", left, binary_expr.op, right)),
            }
        }
        Expr::Column(name) => match engine {
            Some(Engine::Spark | Engine::ODBC) => Ok(format!("{name}")),
            _ => Ok(quoted_identifier(&name.to_string())),
        },
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
            | ScalarValue::Utf8View(Some(value)) => Ok(string_literal(value)),
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
            let expr = to_sql_with_engine(&like_expr.expr, engine)?;
            let pattern = to_sql_with_engine(&like_expr.pattern, engine)?;

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
            let expr = to_sql_with_engine(&in_list.expr, engine)?;
            let list = in_list
                .list
                .iter()
                .map(|expr| to_sql_with_engine(expr, engine))
                .collect::<Result<Vec<String>>>()?;

            let op = if in_list.negated { "NOT IN" } else { "IN" };

            Ok(format!("{expr} {op} ({list})", list = list.join(", ")))
        }
        Expr::ScalarFunction(scalar_function) => {
            let args = scalar_function
                .args
                .iter()
                .map(|expr| to_sql_with_engine(expr, engine))
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

fn handle_cast(cast: &Cast, engine: Option<Engine>, expr: &Expr) -> Result<String> {
    match cast.field.data_type() {
        arrow::datatypes::DataType::Timestamp(_, Some(_) | None) => match engine {
            Some(Engine::ODBC) => Ok(format!(
                "CAST({} AS TIMESTAMP)",
                to_sql_with_engine(&cast.expr, engine)?,
            )),
            // This needs to match the timestamp conversion below
            Some(Engine::DuckDB) => Ok(format!(
                "TO_TIMESTAMP(EPOCH(CAST({} AS TIMESTAMP)))",
                to_sql_with_engine(&cast.expr, engine)?,
            )),
            Some(Engine::SQLite) => Ok(format!(
                "datetime({}, 'subsec', 'utc')",
                to_sql_with_engine(&cast.expr, engine)?,
            )),
            Some(Engine::Spark) => EngineNotSupportedForExpressionSnafu {
                engine: "Spark".to_string(),
                expr: format!("{expr}"),
            }
            .fail()?,
            _ => Ok(format!(
                "CAST({} AS TIMESTAMPTZ)",
                to_sql_with_engine(&cast.expr, engine)?,
            )),
        },
        arrow::datatypes::DataType::Int64 => Ok(format!(
            "CAST({} AS BIGINT)",
            to_sql_with_engine(&cast.expr, engine)?,
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

        // Every engine renders through the same arm, and quote doubling is correct for each.
        for engine in [
            Engine::DuckDB,
            Engine::SQLite,
            Engine::Postgres,
            Engine::MySQL,
            Engine::Spark,
            Engine::ODBC,
        ] {
            assert_eq!(
                to_sql_with_engine(&utf8("O'Brien"), Some(engine))?,
                "'O''Brien'",
                "engine {engine:?} must escape the quote"
            );
        }

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
}
