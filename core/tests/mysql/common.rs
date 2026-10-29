use bollard::models::HealthConfig;
use datafusion_table_providers::sql::db_connection_pool::mysqlpool::MySQLConnectionPool;
use secrecy::SecretString;
use std::collections::HashMap;
use tracing::instrument;

use crate::{
    container_registry,
    docker::{ContainerRunnerBuilder, RunningContainer},
};

const MYSQL_ROOT_PASSWORD: &str = "integration-test-pw";
const MYSQL_DOCKER_CONTAINER: &str = "runtime-integration-test-mysql";

pub(super) fn get_mysql_params(
    port: usize,
    time_zone: Option<&str>,
) -> HashMap<String, SecretString> {
    let mut params = HashMap::new();
    params.insert(
        "mysql_host".to_string(),
        SecretString::from("localhost".to_string()),
    );
    params.insert(
        "mysql_tcp_port".to_string(),
        SecretString::from(port.to_string()),
    );
    params.insert(
        "mysql_user".to_string(),
        SecretString::from("root".to_string()),
    );
    params.insert(
        "mysql_pass".to_string(),
        SecretString::from(MYSQL_ROOT_PASSWORD.to_string()),
    );
    params.insert(
        "mysql_db".to_string(),
        SecretString::from("mysqldb".to_string()),
    );
    params.insert(
        "mysql_sslmode".to_string(),
        SecretString::from("disabled".to_string()),
    );
    if let Some(tz) = time_zone {
        params.insert(
            "mysql_connection_time_zone".to_string(),
            SecretString::from(tz.to_string()),
        );
    }
    params.insert(
        "mysql_pool_min".to_string(),
        SecretString::from("1".to_string()),
    );
    params.insert(
        "mysql_pool_max".to_string(),
        SecretString::from("10".to_string()),
    );
    params
}

#[instrument]
pub async fn start_mysql_docker_container(port: usize) -> Result<RunningContainer, anyhow::Error> {
    start_mysql_docker_container_with_args(port, &[]).await
}

/// Starts a MySQL container whose server takes `mysqld_args`, such as `--socket=…`. The
/// image's entrypoint passes a command that starts with `-` to `mysqld`.
#[instrument]
pub async fn start_mysql_docker_container_with_args(
    port: usize,
    mysqld_args: &[&str],
) -> Result<RunningContainer, anyhow::Error> {
    let container_name = format!("{MYSQL_DOCKER_CONTAINER}-{port}");

    let port = port.try_into().unwrap_or(15432);

    let mysql_docker_image = std::env::var("MYSQL_DOCKER_IMAGE")
        .unwrap_or_else(|_| format!("{}mysql:latest", container_registry()));

    let mut builder = ContainerRunnerBuilder::new(container_name)
        .image(mysql_docker_image)
        .add_port_binding(3306, port)
        .add_env_var("MYSQL_ROOT_PASSWORD", MYSQL_ROOT_PASSWORD)
        .add_env_var("MYSQL_DATABASE", "mysqldb")
        .healthcheck(HealthConfig {
            test: Some(vec![
                "CMD-SHELL".to_string(),
                format!(
                    "mysqladmin ping --host=127.0.0.1 --port=3306 --password={MYSQL_ROOT_PASSWORD}"
                ),
            ]),
            interval: Some(500_000_000), // 250ms
            timeout: Some(100_000_000),  // 100ms
            retries: Some(5),
            start_period: Some(500_000_000), // 100ms
            start_interval: None,
        });
    for arg in mysqld_args {
        builder = builder.add_cmd_arg(arg);
    }
    let running_container = builder.build()?.run().await?;

    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
    Ok(running_container)
}

/// Connection parameters that give the server as a `connection_string` URL rather than as
/// individual parameters.
pub(super) fn get_mysql_connection_string_params(port: usize) -> HashMap<String, SecretString> {
    let mut params = HashMap::new();
    params.insert(
        "mysql_connection_string".to_string(),
        SecretString::from(format!(
            "mysql://root:{MYSQL_ROOT_PASSWORD}@localhost:{port}/mysqldb"
        )),
    );
    params.insert(
        "mysql_sslmode".to_string(),
        SecretString::from("disabled".to_string()),
    );
    params
}

/// Options for an administrative connection as `root`, over TCP only.
pub(super) fn root_opts(port: usize) -> mysql_async::Opts {
    mysql_async::OptsBuilder::default()
        .ip_or_hostname("127.0.0.1")
        .tcp_port(u16::try_from(port).expect("port fits"))
        .user(Some("root"))
        .pass(Some(MYSQL_ROOT_PASSWORD))
        .db_name(Some("mysqldb"))
        .prefer_socket(false)
        .into()
}

/// Connection parameters for `user` (password `pw`) on `host_port`, which can be a proxy in
/// front of the server.
pub(super) fn user_params(host_port: u16, user: &str) -> HashMap<String, SecretString> {
    [
        ("mysql_host", "127.0.0.1".to_string()),
        ("mysql_tcp_port", host_port.to_string()),
        ("mysql_user", user.to_string()),
        ("mysql_pass", "pw".to_string()),
        ("mysql_db", "mysqldb".to_string()),
        ("mysql_sslmode", "disabled".to_string()),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_string(), SecretString::from(value)))
    .collect()
}

#[instrument]
pub(super) async fn get_mysql_connection_pool(
    port: usize,
    time_zone: Option<&str>,
) -> Result<MySQLConnectionPool, anyhow::Error> {
    let mysql_pool = MySQLConnectionPool::new(get_mysql_params(port, time_zone))
        .await
        .expect("Failed to create MySQL Connection Pool");

    Ok(mysql_pool)
}
