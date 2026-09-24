//! Runs the SQL that the shared DML rendering builds against an in-process SQLite and asserts on
//! the rows that survive it.

use datafusion::common::TableReference;

/// `SQLite` is the second engine these statements are built for, and it reads a delimited
/// identifier by its own rules. Postgres, the third, needs a running server and so has no
/// in-process equivalent to assert against.
mod sqlite_execution_tests {
    use super::*;
    use datafusion_table_providers_common::util::dml::*;
    use rusqlite::Connection;

    #[test]
    fn a_quote_bearing_table_name_deletes_by_the_table_it_names() {
        let conn = Connection::open_in_memory().expect("in-memory SQLite");
        conn.execute_batch(r#"CREATE TABLE "we""ird" (id INTEGER)"#)
            .expect("create");
        conn.execute_batch(r#"INSERT INTO "we""ird" VALUES (1), (2), (3)"#)
            .expect("insert");

        conn.execute_batch(&delete_statement(
            &TableReference::bare(r#"we"ird"#),
            Some(r#""id" = 2"#),
        ))
        .expect("the rendered DELETE must be valid SQL for SQLite");

        let mut stmt = conn
            .prepare(r#"SELECT id FROM "we""ird" ORDER BY id"#)
            .expect("prepare");
        let surviving: Vec<i32> = stmt
            .query_map([], |row| row.get(0))
            .expect("query")
            .collect::<rusqlite::Result<Vec<i32>>>()
            .expect("rows");
        assert_eq!(surviving, vec![1, 3]);
    }

    #[test]
    fn a_quote_bearing_table_name_updates_by_the_table_it_names() {
        let conn = Connection::open_in_memory().expect("in-memory SQLite");
        conn.execute_batch(r#"CREATE TABLE "we""ird" (id INTEGER, qty INTEGER)"#)
            .expect("create");
        conn.execute_batch(r#"INSERT INTO "we""ird" VALUES (1, 10), (2, 20)"#)
            .expect("insert");

        conn.execute_batch(&update_statement(
            &TableReference::bare(r#"we"ird"#),
            r#""qty" = 99"#,
            Some(r#""id" = 2"#),
        ))
        .expect("the rendered UPDATE must be valid SQL for SQLite");

        let mut stmt = conn
            .prepare(r#"SELECT qty FROM "we""ird" ORDER BY id"#)
            .expect("prepare");
        let quantities: Vec<i32> = stmt
            .query_map([], |row| row.get(0))
            .expect("query")
            .collect::<rusqlite::Result<Vec<i32>>>()
            .expect("rows");
        assert_eq!(quantities, vec![10, 99]);
    }
}
