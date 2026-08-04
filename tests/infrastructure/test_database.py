"""Client factory behaviour that needs no running server."""
import pytest

from llm_observability.config.settings import ClickHouseConfig, ConfigError
from llm_observability.core.infrastructure.persistence import database


def _cfg(**over):
    base = dict(host="127.0.0.1", port=59999, database="nope",
                username="default", password="")
    base.update(over)
    return ClickHouseConfig(**base)


def test_unreachable_server_raises_a_readable_error():
    """The one new failure mode a server introduces deserves better than a
    driver traceback."""
    with pytest.raises(database.ClickHouseUnavailable) as exc:
        database.get_client(_cfg())
    message = str(exc.value)
    assert "127.0.0.1:59999" in message
    assert "docker compose up" in message


def test_missing_database_configuration_raises_config_error(monkeypatch):
    monkeypatch.delenv("CLICKHOUSE_DATABASE", raising=False)
    with pytest.raises(ConfigError):
        database.get_client()


def test_unavailable_is_a_runtime_error():
    """Callers that only care 'storage is down' can catch RuntimeError."""
    assert issubclass(database.ClickHouseUnavailable, RuntimeError)
