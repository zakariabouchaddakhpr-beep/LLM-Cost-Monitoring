"""
Connection factory for the ClickHouse trace store.

Replaces the DuckDB engine builder. There is no `read_only` flag any more:
ClickHouse is a server, so readers and writers coexist and there is no file lock
to negotiate. That removed the whole `read_only=` parameter chain and the
lock-retry loop the repository used to need.

The only new failure mode is that the server can be unreachable — the price of
the architecture — which this module turns into a readable message rather than a
driver traceback.
"""

import logging

import clickhouse_connect

from llm_observability.config.settings import ClickHouseConfig, get_clickhouse_config
from llm_observability.core.infrastructure.persistence import schema

logger = logging.getLogger(__name__)


class ClickHouseUnavailable(RuntimeError):
    """The server could not be reached.

    Subclasses RuntimeError so callers that only care that storage is down can
    catch the broader type.
    """


def _connect(config: ClickHouseConfig, database: str | None):
    try:
        return clickhouse_connect.get_client(
            host=config.host,
            port=config.port,
            username=config.username,
            password=config.password,
            **({"database": database} if database else {}),
        )
    except Exception as exc:
        raise ClickHouseUnavailable(
            f"Cannot reach ClickHouse at {config.host}:{config.port}. "
            f"Is it running? Try: docker compose up -d clickhouse\n"
            f"Underlying error: {exc}"
        ) from exc


def get_client(config: ClickHouseConfig | None = None, *,
               database: str | None = None,
               create_database: bool = False):
    """
    Return a client bound to this project's database.

    With create_database=True the database is created if absent — the writer's
    path. Readers pass False so a missing database surfaces as "nothing has
    written traces yet" rather than an empty store that looks like data loss.
    That distinction is inherited from the DuckDB version, which refused to
    conjure an empty file for a reader.

    Raises:
        ConfigError: if CLICKHOUSE_DATABASE is unset.
        ClickHouseUnavailable: if the server cannot be reached.
    """
    config = config or get_clickhouse_config(database)

    if create_database:
        # Connect without a database first: you cannot bind to one that does not
        # exist yet. Separate short-lived client, then bind properly below.
        admin = _connect(config, None)
        try:
            admin.command(schema.database_ddl(config.database))
        finally:
            admin.close()

    logger.debug("Connecting to ClickHouse %s:%s database=%s",
                 config.host, config.port, config.database)
    return _connect(config, config.database)


def database_exists(config: ClickHouseConfig) -> bool:
    """True when this project's database is present on the server."""
    client = _connect(config, None)
    try:
        rows = client.query(
            "SELECT count() FROM system.databases WHERE name = %(name)s",
            parameters={"name": config.database},
        ).result_rows
        return bool(rows and rows[0][0])
    finally:
        client.close()
