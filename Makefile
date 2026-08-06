all:
	cargo build --all-features

.PHONY: test
test:
	cargo test --features clickhouse-federation,duckdb-federation,flight,mysql-federation,postgres-federation,sqlite-federation,adbc-federation -p datafusion-table-providers --lib

.PHONY: lint
lint:
	cargo clippy --all-features

.PHONY: test-integration
test-integration:
	RUST_LOG=debug cargo test --test integration --no-default-features --features postgres,sqlite,mysql,flight,clickhouse,mongodb,adbc -- --nocapture

# Type-checks the integration suite without needing Docker, so `make test`
# running only `--lib` can't let the suite rot uncompiled.
.PHONY: check-integration
check-integration:
	cargo test --test integration --no-default-features --features postgres,sqlite,mysql,flight,clickhouse,mongodb,adbc --no-run
