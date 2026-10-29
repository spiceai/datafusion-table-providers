use std::{
    collections::{HashMap, HashSet},
    fmt,
    path::PathBuf,
    sync::{
        atomic::{AtomicU8, AtomicUsize, Ordering},
        Arc, Mutex, PoisonError,
    },
    time::{Duration, Instant},
};

use async_trait::async_trait;
use mysql_async::{
    prelude::{Queryable, ToValue},
    Conn, DriverError, Metrics, Opts, OptsBuilder, Params, PoolConstraints, PoolOpts, Row, SslOpts,
    DEFAULT_POOL_CONSTRAINTS,
};
use secrecy::{ExposeSecret, SecretBox, SecretString};
use snafu::{ResultExt, Snafu};

use crate::{
    sql::{
        arrow_sql_gen::mysql::MysqlZeroDateBehavior,
        db_connection_pool::{
            dbconnection::{mysqlconn::MySQLConnection, AsyncDbConnection, DbConnection},
            JoinPushDown,
        },
    },
    util::{self, ns_lookup::verify_ns_lookup_and_tcp_connect},
};

use super::DbConnectionPool;

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("MySQL connection failed.\n{source}\nFor further information, refer to the MySQL manual: https://dev.mysql.com/doc/mysql-errors/9.1/en/error-reference-introduction.html"))]
    MySQLConnectionError { source: mysql_async::Error },

    #[snafu(display(
        "Invalid MySQL connection string.\n{source}\nEnsure the MySQL connection string is valid"
    ))]
    InvalidConnectionString { source: mysql_async::UrlError },

    #[snafu(display("Invalid value for parameter {parameter_name}\nEnsure the value is valid for parameter {parameter_name}"))]
    InvalidParameterError { parameter_name: String },

    #[snafu(display("Invalid root cert path: {path}\nEnsure the root cert path is valid"))]
    InvalidRootCertPathError { path: String },

    #[snafu(display("Cannot connect to MySQL on {host}:{port}. Ensure the host and port are correct and reachable."))]
    InvalidHostOrPortError {
        source: crate::util::ns_lookup::Error,
        host: String,
        port: u16,
    },

    #[snafu(display("Authentication failed. Verify username and password."))]
    InvalidUsernameOrPassword,

    #[snafu(display("{message}\nEnsure the given MySQL database exists"))]
    UnknownMySQLDatabase { message: String },
}

#[derive(Debug, Clone)]
pub struct MySQLConnectionPool {
    pool: Arc<mysql_async::Pool>,
    join_push_down: JoinPushDown,
    zero_date_behavior: MysqlZeroDateBehavior,
}

/// The server a connection reached, as the server reports itself. A load-balanced
/// hostname (a cluster reader endpoint, a proxy, DNS round-robin) can reach a different
/// server on every connection, and this names which one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MySQLServerIdentity {
    /// The host the pool connects to, as configured.
    pub configured_host: String,
    /// The port the pool connects to, as configured.
    pub configured_port: u16,
    /// The connection's id on the server, as `CONNECTION_ID()` and the process list show it.
    pub connection_id: u32,
    /// The server's `@@hostname`.
    pub hostname: String,
    /// The server's `@@port`.
    pub port: u16,
    /// The DB instance identifier, on Aurora MySQL (`aurora_server_id`).
    pub aurora_server_id: Option<String>,
}

/// Called with the server a new connection reached.
type ObserveFn = Arc<dyn Fn(&MySQLServerIdentity) + Send + Sync>;

/// Reports the server each new connection reached, once per connection, as it is opened.
///
/// It runs as the connection's `after_connect` callback, which `mysql_async` calls once on
/// each connection it opens, before the connection's setup statements, and not again when the
/// pool resets the connection (`COM_RESET_CONNECTION` or `COM_CHANGE_USER`) to hand it out
/// again. A connection is therefore reported whatever its id, including one whose id a
/// connection to another server behind the same hostname had before it.
#[derive(Clone)]
struct NewConnectionObserver {
    observe: ObserveFn,
    server_kind: Arc<AtomicU8>,
    /// Whether the pool is configured to connect over a Unix socket.
    configured_socket: bool,
    /// The ids of the socket replacements reported once, whose second callback is still due
    /// (see [`NewConnectionObserver::is_second_callback`]).
    socket_replacements: Arc<Mutex<HashSet<u32>>>,
}

impl fmt::Debug for NewConnectionObserver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NewConnectionObserver")
            .finish_non_exhaustive()
    }
}

impl NewConnectionObserver {
    fn new(observe: ObserveFn, opts: &Opts) -> Self {
        Self {
            observe,
            server_kind: Arc::new(AtomicU8::new(SERVER_KIND_UNKNOWN)),
            configured_socket: opts.socket().is_some(),
            socket_replacements: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    /// Reports the server `conn` reached. It never fails: a connection whose server does not
    /// answer is still used, and is only missing from the reports.
    async fn observe(&self, conn: &mut Conn) {
        if self.is_second_callback(conn) {
            return;
        }
        let connection_id = conn.id();
        match server_identity(conn, &self.server_kind).await {
            Ok(Some(identity)) => (self.observe)(&identity),
            Ok(None) => tracing::debug!(
                "MySQL connection {connection_id} gave no usable server hostname and port"
            ),
            Err(error) => tracing::debug!(
                "Could not ask MySQL connection {connection_id} which server it reached: {error}"
            ),
        }
    }

    /// Whether `conn` was already reported. With `prefer_socket`, on by default outside
    /// Windows, `mysql_async` replaces a TCP connection to a server whose Unix socket the client
    /// can open with a connection over that socket, and runs `after_connect` on the replacement
    /// twice: as it opens it, and again where the TCP connection would have run it. A
    /// connection over a socket the pool was not configured with is such a replacement; the
    /// first callback reports it and records its id, and the second finds the id and removes
    /// it. A replacement that fails after its first callback, so has no second, leaves its id
    /// behind, and the next replacement given the same id is reported on its second callback
    /// instead: still once.
    fn is_second_callback(&self, conn: &Conn) -> bool {
        if self.configured_socket || conn.opts().socket().is_none() {
            return false;
        }
        let mut reported = self
            .socket_replacements
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if reported.remove(&conn.id()) {
            true
        } else {
            reported.insert(conn.id());
            false
        }
    }
}

/// What the pool's servers answer, learned from the first connection asked.
const SERVER_KIND_UNKNOWN: u8 = 0;
/// `@@aurora_server_id` is readable: Aurora MySQL.
const SERVER_KIND_AURORA: u8 = 1;
/// `@@hostname` and `@@port` are readable, `@@aurora_server_id` is not: MySQL, MariaDB, TiDB,
/// Vitess and the other MySQL-compatible servers.
const SERVER_KIND_MYSQL: u8 = 2;
/// Neither query is accepted, as by a proxy that rejects system variables: stop asking.
const SERVER_KIND_UNANSWERED: u8 = 3;

const WITH_AURORA: &str = "SELECT @@hostname, @@port, @@aurora_server_id";
const WITHOUT_AURORA: &str = "SELECT @@hostname, @@port";

/// Asks the server which server it is, in one query. `@@aurora_server_id` exists only on
/// Aurora MySQL; MySQL 5.7, 8.0 and 9.7, MariaDB, TiDB and Vitess reject it with
/// `ER_UNKNOWN_SYSTEM_VARIABLE` (1193). The first connection learns which form the pool's
/// servers answer, from any server error rather than only that code, and later connections
/// ask only that form, or nothing when neither is answered.
/// Returns `None` when the server answers neither form, or with a value that is NULL or does
/// not decode.
async fn server_identity(
    conn: &mut Conn,
    server_kind: &AtomicU8,
) -> mysql_async::Result<Option<MySQLServerIdentity>> {
    let row: Option<Row> = match server_kind.load(Ordering::Relaxed) {
        SERVER_KIND_UNANSWERED => return Ok(None),
        SERVER_KIND_MYSQL => without_aurora(conn, server_kind).await?,
        _ => match conn.query_first(WITH_AURORA).await {
            Ok(row) => {
                server_kind.store(SERVER_KIND_AURORA, Ordering::Relaxed);
                row
            }
            Err(mysql_async::Error::Server(_)) => {
                server_kind.store(SERVER_KIND_MYSQL, Ordering::Relaxed);
                without_aurora(conn, server_kind).await?
            }
            Err(error) => return Err(error),
        },
    };
    let Some(mut row) = row else {
        return Ok(None);
    };
    // `take_opt` reports a value that does not decode instead of panicking, as decoding the
    // row into a tuple would.
    let hostname = row.take_opt::<String, _>(0).and_then(Result::ok);
    let port = row.take_opt::<u16, _>(1).and_then(Result::ok);
    let aurora_server_id = row
        .take_opt::<Option<String>, _>(2)
        .and_then(Result::ok)
        .flatten()
        .filter(|value| !value.is_empty());
    Ok(hostname
        .zip(port)
        .map(|(hostname, port)| MySQLServerIdentity {
            configured_host: conn.opts().ip_or_hostname().to_string(),
            configured_port: conn.opts().tcp_port(),
            connection_id: conn.id(),
            hostname,
            port,
            aurora_server_id,
        }))
}

/// Asks `@@hostname` and `@@port`. A server error means the pool's servers do not answer it,
/// and the pool stops asking.
async fn without_aurora(
    conn: &mut Conn,
    server_kind: &AtomicU8,
) -> mysql_async::Result<Option<Row>> {
    match conn.query_first(WITHOUT_AURORA).await {
        Ok(row) => Ok(row),
        Err(mysql_async::Error::Server(_)) => {
            server_kind.store(SERVER_KIND_UNANSWERED, Ordering::Relaxed);
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

#[allow(dead_code)]
const SETUP_QUERIES: [&str; 4] = [
    "SET time_zone = '+00:00'",
    "SET character_set_results = 'utf8mb4'",
    "SET character_set_client = 'utf8mb4'",
    "SET character_set_connection = 'utf8mb4'",
];

/// Returns the setup queries for the MySQL connection, optionally overriding default time zone (UTC).
fn get_setup_queries(time_zone: Option<&str>) -> Vec<String> {
    let tz = time_zone.unwrap_or("+00:00");
    vec![
        format!("SET time_zone = '{tz}'"),
        "SET character_set_results = 'utf8mb4'".to_string(),
        "SET character_set_client = 'utf8mb4'".to_string(),
        "SET character_set_connection = 'utf8mb4'".to_string(),
    ]
}

impl MySQLConnectionPool {
    /// Creates a new instance of `MySQLConnectionPool`.
    ///
    /// # Arguments
    ///
    /// * `params` - A map of parameters to create the connection pool.
    ///   * `connection_string` - The connection string to use to connect to the MySQL database, or can be specified with the below individual parameters.
    ///   * `host` - The host of the MySQL database.
    ///   * `user` - The user to use when connecting to the MySQL database.
    ///   * `db` - The database to connect to.
    ///   * `pass` - The password to use when connecting to the MySQL database.
    ///   * `tcp_port` - The TCP port to use when connecting to the MySQL database.
    ///   * `sslmode` - The SSL mode to use when connecting to the MySQL database. Can be "disabled", "required", or "preferred".
    ///   * `sslrootcert` - The path to the root certificate to use when connecting to the MySQL database.
    ///   * `pool_min` - The minimum number of connections to keep open in the pool, lazily created when requested.
    ///   * `pool_max` - The maximum number of connections to allow in the pool.
    ///   * `time_zone` - The time zone to use for the MySQL connection (e.g., "+2:00", "UTC", etc.). Default is "+00:00" (UTC).
    ///
    /// # Errors
    ///
    /// Returns an error if there is a problem creating the connection pool.
    pub async fn new(params: HashMap<String, SecretString>) -> Result<Self> {
        Self::new_with_pool_opts(params, |pool_opts| pool_opts).await
    }

    /// Creates a new instance of `MySQLConnectionPool`, passing the pool options through
    /// `configure_pool` before the pool is built.
    ///
    /// `configure_pool` receives the pool options read from `params`: from the
    /// `connection_string` URL when one is given, otherwise from `pool_min` and `pool_max`.
    /// What it returns is what the pool uses on both paths, so a caller can apply pool size
    /// limits or a connection lifetime whether or not the connection is given as a URL.
    ///
    /// # Errors
    ///
    /// Returns an error if there is a problem creating the connection pool.
    pub async fn new_with_pool_opts(
        params: HashMap<String, SecretString>,
        configure_pool: impl FnOnce(PoolOpts) -> PoolOpts + Send,
    ) -> Result<Self> {
        Self::build(params, configure_pool, None).await
    }

    /// Creates a new instance of `MySQLConnectionPool` as [`Self::new_with_pool_opts`] does,
    /// and calls `observe` with the server each connection the pool opens reached.
    ///
    /// `observe` is called once for every connection the pool opens, the pool's own test
    /// connection included, as the connection is opened and before its first use. Finding the
    /// server costs one query on each new connection (two on the pool's first) and none on a
    /// connection the pool reuses. A connection whose server does not answer is opened and
    /// used as any other, and is not reported.
    ///
    /// # Errors
    ///
    /// Returns an error if there is a problem creating the connection pool.
    pub async fn new_with_pool_opts_and_observer(
        params: HashMap<String, SecretString>,
        configure_pool: impl FnOnce(PoolOpts) -> PoolOpts + Send,
        observe: impl Fn(&MySQLServerIdentity) + Send + Sync + 'static,
    ) -> Result<Self> {
        Self::build(params, configure_pool, Some(Arc::new(observe))).await
    }

    async fn build(
        params: HashMap<String, SecretString>,
        configure_pool: impl FnOnce(PoolOpts) -> PoolOpts + Send,
        observe: Option<ObserveFn>,
    ) -> Result<Self> {
        // Remove the "mysql_" prefix from the keys to keep backward compatibility
        let params = util::remove_prefix_from_hashmap_keys(params, "mysql_");

        let mut connection_string = mysql_async::OptsBuilder::default();
        let mut ssl_mode = "required";
        let mut ssl_rootcert_path: Option<PathBuf> = None;

        if let Some(mysql_connection_string) = params
            .get("connection_string")
            .map(SecretBox::expose_secret)
        {
            connection_string = mysql_async::OptsBuilder::from_opts(
                mysql_async::Opts::from_url(mysql_connection_string)
                    .context(InvalidConnectionStringSnafu)?,
            );
        } else {
            if let Some(mysql_host) = params.get("host").map(SecretBox::expose_secret) {
                connection_string = connection_string.ip_or_hostname(mysql_host);
            }
            if let Some(mysql_user) = params.get("user").map(SecretBox::expose_secret) {
                connection_string = connection_string.user(Some(mysql_user));
            }
            if let Some(mysql_db) = params.get("db").map(SecretBox::expose_secret) {
                connection_string = connection_string.db_name(Some(mysql_db));
            }
            if let Some(mysql_pass) = params.get("pass").map(SecretBox::expose_secret) {
                connection_string = connection_string.pass(Some(mysql_pass));
            }
            if let Some(mysql_tcp_port) = params.get("tcp_port").map(SecretBox::expose_secret) {
                connection_string =
                    connection_string.tcp_port(mysql_tcp_port.parse::<u16>().unwrap_or(3306));
            }

            let pool_min = params
                .get("pool_min")
                .map(SecretBox::expose_secret)
                .unwrap_or_default()
                .parse::<usize>()
                .unwrap_or(DEFAULT_POOL_CONSTRAINTS.min());

            let pool_max = params
                .get("pool_max")
                .map(SecretBox::expose_secret)
                .unwrap_or_default()
                .parse::<usize>()
                .unwrap_or(DEFAULT_POOL_CONSTRAINTS.max());

            connection_string = connection_string
                .pool_opts(PoolOpts::default().with_constraints(
                    PoolConstraints::new(pool_min, pool_max).unwrap_or_default(),
                ));
        }

        if let Some(mysql_sslmode) = params.get("sslmode").map(SecretBox::expose_secret) {
            match mysql_sslmode.to_lowercase().as_str() {
                "disabled" | "required" | "preferred" => {
                    ssl_mode = mysql_sslmode;
                }
                _ => {
                    InvalidParameterSnafu {
                        parameter_name: "sslmode".to_string(),
                    }
                    .fail()?;
                }
            }
        }
        if let Some(mysql_sslrootcert) = params.get("sslrootcert").map(SecretBox::expose_secret) {
            if !std::path::Path::new(mysql_sslrootcert).exists() {
                InvalidRootCertPathSnafu {
                    path: mysql_sslrootcert,
                }
                .fail()?;
            }

            ssl_rootcert_path = Some(PathBuf::from(mysql_sslrootcert));
        }

        let ssl_opts = get_ssl_opts(ssl_mode, ssl_rootcert_path);

        connection_string = connection_string.ssl_opts(ssl_opts);

        connection_string = connection_string.setup(get_setup_queries(
            params.get("time_zone").map(SecretBox::expose_secret),
        ));

        let opts = mysql_async::Opts::from(connection_string);
        let pool_opts = configure_pool(opts.pool_opts().clone());
        let mut opts_builder = OptsBuilder::from_opts(opts.clone()).pool_opts(pool_opts);
        if let Some(observe) = observe {
            let observer = NewConnectionObserver::new(observe, &opts);
            opts_builder = opts_builder.after_connect(move |conn| {
                let observer = observer.clone();
                Box::pin(async move {
                    observer.observe(conn).await;
                    Ok(())
                })
            });
        }
        let opts = mysql_async::Opts::from(opts_builder);

        verify_mysql_opts(&opts).await?;

        let join_push_down = get_join_context(&opts);

        let pool = mysql_async::Pool::new(opts);

        // Test the connection
        let mut conn = pool.get_conn().await.map_err(|err| match err {
            // In case of an incorrect user name, the error `Unknown authentication plugin 'sha256_password'` is returned.
            // We override it with a more user-friendly error message.
            mysql_async::Error::Driver(DriverError::UnknownAuthPlugin { .. }) => {
                Error::InvalidUsernameOrPassword
            }
            mysql_async::Error::Server(server_error) => {
                match server_error.code {
                    // Code 1049: Server error: `ERROR 42000 (1049): Unknown database <database>
                    1049 => Error::UnknownMySQLDatabase {
                        message: server_error.message,
                    },
                    // Code 1045: Server error: ERROR 1045 (28000): Access denied for user <user> (using password: YES / NO)
                    1045 => Error::InvalidUsernameOrPassword,
                    _ => Error::MySQLConnectionError {
                        source: mysql_async::Error::Server(server_error),
                    },
                }
            }
            _ => Error::MySQLConnectionError { source: err },
        })?;

        let _rows: Vec<Row> = conn
            .exec("SELECT 1", Params::Empty)
            .await
            .context(MySQLConnectionSnafu)?;

        // Hand the test connection to the pool's first checkout rather than open another.
        // `SELECT 1` changes no session state, so the connection skips the reset `mysql_async`
        // gives a returned connection; its prepared `SELECT 1` stays in the connection's
        // statement cache, as statements do on a pool that does not reset connections. The
        // pool takes a returned connection back on a background task, and a checkout that
        // started before then would open a second connection. Waiting for that task keeps the
        // connection in the pool's accounting and under its lifetime and idle limits.
        conn.reset_connection(false);
        drop(conn);
        wait_for_returned_connection(&pool.metrics()).await;

        Ok(Self {
            pool: Arc::new(pool),
            join_push_down,
            zero_date_behavior: MysqlZeroDateBehavior::default(),
        })
    }

    /// Override the [`MysqlZeroDateBehavior`] used by connections created from this pool.
    /// Connections created before this call are unaffected.
    #[must_use]
    pub fn with_zero_date_behavior(mut self, behavior: MysqlZeroDateBehavior) -> Self {
        self.zero_date_behavior = behavior;
        self
    }

    /// Returns the [`MysqlZeroDateBehavior`] that will be applied to connections created from
    /// this pool.
    #[must_use]
    pub fn zero_date_behavior(&self) -> MysqlZeroDateBehavior {
        self.zero_date_behavior
    }

    /// Returns a direct connection to the underlying database.
    ///
    /// # Errors
    ///
    /// Returns an error if there is a problem creating the connection pool.
    pub async fn connect_direct(&self) -> super::Result<MySQLConnection> {
        let pool = Arc::clone(&self.pool);
        let conn = pool.get_conn().await.context(MySQLConnectionSnafu)?;

        Ok(MySQLConnection::new(conn).with_zero_date_behavior(self.zero_date_behavior))
    }

    pub fn metrics(&self) -> Arc<Metrics> {
        self.pool.metrics()
    }
}

/// The longest pool creation waits for its test connection to be back in the pool.
const RETURNED_CONNECTION_TIMEOUT: Duration = Duration::from_secs(5);

/// Waits until the pool has finished with the one connection returned to it: put it among its
/// idle connections, or closed it, as a pool whose limits keep no idle connection does. Gives
/// up after [`RETURNED_CONNECTION_TIMEOUT`], when the next checkout opens a new connection.
async fn wait_for_returned_connection(metrics: &Metrics) {
    let recycled = |metrics: &Metrics| {
        [
            &metrics.connection_returned_to_pool,
            &metrics.discarded_superfluous_connection,
            &metrics.discarded_unestablished_connection,
            &metrics.discarded_expired_connection,
            &metrics.discarded_error_during_cleanup,
        ]
        .into_iter()
        .map(|counter: &AtomicUsize| counter.load(Ordering::Acquire))
        .sum::<usize>()
    };
    let deadline = Instant::now() + RETURNED_CONNECTION_TIMEOUT;
    while recycled(metrics) == 0 && Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
}

async fn verify_mysql_opts(opts: &Opts) -> Result<()> {
    // Verify the host and port are correct
    let host = opts.ip_or_hostname();
    let port = opts.tcp_port();

    verify_ns_lookup_and_tcp_connect(host, port)
        .await
        .context(InvalidHostOrPortSnafu { host, port })?;

    Ok(())
}

fn get_join_context(opts: &mysql_async::Opts) -> JoinPushDown {
    let mut join_context = format!("host={},port={}", opts.ip_or_hostname(), opts.tcp_port());
    if let Some(db_name) = opts.db_name() {
        join_context.push_str(&format!(",db={db_name}"));
    }
    if let Some(user) = opts.user() {
        join_context.push_str(&format!(",user={user}"));
    }

    JoinPushDown::AllowedFor(join_context)
}

fn get_ssl_opts(ssl_mode: &str, rootcert_path: Option<PathBuf>) -> Option<SslOpts> {
    if ssl_mode == "disabled" {
        return None;
    }

    let mut opts = SslOpts::default();

    if let Some(rootcert_path) = rootcert_path {
        let path = rootcert_path;
        opts = opts.with_root_certs(vec![path.into()]);
    }

    // If ssl_mode is "preferred", we will accept invalid certs and skip domain validation
    // mysql_async does not have a "ssl_mode" https://github.com/blackbeam/mysql_async/issues/225#issuecomment-1409922237
    if ssl_mode == "preferred" {
        opts = opts
            .with_danger_accept_invalid_certs(true)
            .with_danger_skip_domain_validation(true);
    }

    Some(opts)
}

#[async_trait]
impl DbConnectionPool<mysql_async::Conn, &'static (dyn ToValue + Sync)> for MySQLConnectionPool {
    async fn connect(
        &self,
    ) -> super::Result<Box<dyn DbConnection<mysql_async::Conn, &'static (dyn ToValue + Sync)>>>
    {
        let pool = Arc::clone(&self.pool);
        let conn = pool.get_conn().await.context(MySQLConnectionSnafu)?;

        Ok(Box::new(
            MySQLConnection::new(conn).with_zero_date_behavior(self.zero_date_behavior),
        ))
    }

    fn join_push_down(&self) -> JoinPushDown {
        self.join_push_down.clone()
    }
}
