import logging

import clickhouse_connect

from llm_observability.config.settings import (
    ClickHouseConfig,
    get_clickhouse_config,
)
from llm_observability.core.infrastructure.persistence import schema

logger = logging.getLogger(__name__)


class ClickHouseUnavailable(RuntimeError):
    pass


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
    config = config or get_clickhouse_config(database)

    if create_database:
        admin = _connect(config, None)
        try:
            admin.command(schema.database_ddl(config.database))
        finally:
            admin.close()

    logger.debug("Connecting to ClickHouse %s:%s database=%s",
                 config.host, config.port, config.database)
    return _connect(config, config.database)


def database_exists(config: ClickHouseConfig) -> bool:
    client = _connect(config, None)
    try:
        rows = client.query(
            "SELECT count() FROM system.databases WHERE name = %(name)s",
            parameters={"name": config.database},
        ).result_rows
        return bool(rows and rows[0][0])
    finally:
        client.close()
