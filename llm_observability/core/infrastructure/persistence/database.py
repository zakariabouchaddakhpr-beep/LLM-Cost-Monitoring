"""
Connecting to the ClickHouse database.

ClickHouse is a database SERVER - a separate program, usually running in
Docker, that our code talks to over the network. That's different from a
database that's just a file on disk, and it has one big advantage: lots of
programs can read and write at the same time without taking turns.

The trade-off is a new way things can go wrong: the server might not be
running at all. Turning that into a message you can actually act on - instead
of a wall of red text from the database driver - is this file's main job.
"""

import logging

import clickhouse_connect

from llm_observability.config.settings import (
    ClickHouseConfig,
    get_clickhouse_config,
)
from llm_observability.core.infrastructure.persistence import schema

logger = logging.getLogger(__name__)


class ClickHouseUnavailable(RuntimeError):
    """
    Our own error, for when the database server can't be reached.

    It's built on top of RuntimeError, which is one of Python's built-in error
    types. That means code elsewhere that only cares "storage is broken" can
    catch RuntimeError and will catch this too, without ever needing to know
    this class exists.
    """


def _connect(config: ClickHouseConfig, database: str | None):
    """
    Open one connection, or fail with a message a person can read.

    Pass database=None to connect to the server without choosing a database.
    You need that when the database doesn't exist yet - you can't select
    something that hasn't been created.
    """
    try:
        return clickhouse_connect.get_client(
            host=config.host,
            port=config.port,
            username=config.username,
            password=config.password,
            # Only include the database setting when we actually have one.
            # The ** unpacks the dictionary into named arguments, and an empty
            # dictionary adds nothing at all.
            **({"database": database} if database else {}),
        )
    except Exception as exc:
        # Replace the driver's error with one that says what to DO about it.
        #
        # `raise ... from exc` keeps the original error attached underneath.
        # So you see the helpful message first, and the technical detail is
        # still there below it if you need to dig.
        raise ClickHouseUnavailable(
            f"Cannot reach ClickHouse at {config.host}:{config.port}. "
            f"Is it running? Try: docker compose up -d clickhouse\n"
            f"Underlying error: {exc}"
        ) from exc


def get_client(config: ClickHouseConfig | None = None, *,
               database: str | None = None,
               create_database: bool = False):
    """
    Open a connection to this project's database.

    create_database=True  -> create the database if it's missing. This is what
                             code that WRITES passes. Something with data to
                             save and nowhere to put it should just make
                             somewhere.

    create_database=False -> don't create it. This is what code that READS
                             passes, so that a missing database shows up
                             honestly as "nothing has ever been saved here"
                             rather than as an empty database, which looks
                             like your data disappeared.

    Raises:
        ConfigError: if nobody said which database to use.
        ClickHouseUnavailable: if the server can't be reached.
    """
    # `X or Y` means "use X if we were given one, otherwise work out Y".
    config = config or get_clickhouse_config(database)

    if create_database:
        # Connect without choosing a database, create it, then disconnect.
        # This has to be a separate short-lived connection, because the real
        # one below needs to select a database that already exists.
        admin = _connect(config, None)
        try:
            admin.command(schema.database_ddl(config.database))
        finally:
            # `finally` runs no matter what - so the connection gets closed
            # even if the command above failed.
            admin.close()

    # logger.debug is quieter than logger.info. It only shows up if you ask
    # for extra detail, which is right for a message on every connection.
    logger.debug("Connecting to ClickHouse %s:%s database=%s",
                 config.host, config.port, config.database)
    return _connect(config, config.database)


def database_exists(config: ClickHouseConfig) -> bool:
    """
    True if this project's database is already on the server.

    ClickHouse keeps information about itself in a built-in database called
    `system`, so we can simply ask it which databases exist.
    """
    client = _connect(config, None)
    try:
        rows = client.query(
            # %(name)s is a placeholder. We pass the real value separately in
            # `parameters` and let the driver put it in safely.
            #
            # NEVER build a query by gluing values into the text yourself. If
            # a value came from a user, they could type SQL into it and have
            # the database run it - that's the classic "SQL injection" attack.
            "SELECT count() FROM system.databases WHERE name = %(name)s",
            parameters={"name": config.database},
        ).result_rows
        # rows comes back as [(1,)] when found, or [(0,)] when not.
        # rows[0][0] means: the first row, then the first value in that row.
        # bool(...) turns 1 into True and 0 into False.
        return bool(rows and rows[0][0])
    finally:
        client.close()
