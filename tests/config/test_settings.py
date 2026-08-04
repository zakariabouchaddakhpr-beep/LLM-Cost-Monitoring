"""Configuration resolves ClickHouse connection details from the consuming project."""
import pytest

from llm_observability.config.settings import (
    ConfigError,
    ClickHouseConfig,
    get_clickhouse_config,
    resolve_database,
    slug,
)

CH_VARS = ("CLICKHOUSE_HOST", "CLICKHOUSE_PORT", "CLICKHOUSE_DATABASE",
           "CLICKHOUSE_USER", "CLICKHOUSE_PASSWORD")


@pytest.fixture
def clean_env(monkeypatch):
    for var in CH_VARS:
        monkeypatch.delenv(var, raising=False)


class TestSlug:
    def test_uses_underscores_not_hyphens(self):
        # ClickHouse identifiers reject hyphens without backtick quoting.
        assert slug("My App!") == "my_app"

    def test_collapses_runs(self):
        assert slug("a---b") == "a_b"

    def test_rejects_leading_digit(self):
        assert slug("2024 report").startswith("p_")

    def test_empty_falls_back(self):
        assert slug("   ") == "project"


class TestGetClickHouseConfig:
    def test_database_has_no_default(self, clean_env):
        with pytest.raises(ConfigError):
            get_clickhouse_config()

    def test_blank_database_is_an_error(self, clean_env, monkeypatch):
        monkeypatch.setenv("CLICKHOUSE_DATABASE", "   ")
        with pytest.raises(ConfigError):
            get_clickhouse_config()

    def test_explicit_database_wins(self, clean_env, monkeypatch):
        monkeypatch.setenv("CLICKHOUSE_DATABASE", "from_env")
        assert get_clickhouse_config("explicit").database == "explicit"

    def test_host_port_user_have_defaults(self, clean_env, monkeypatch):
        monkeypatch.setenv("CLICKHOUSE_DATABASE", "proj")
        cfg = get_clickhouse_config()
        assert isinstance(cfg, ClickHouseConfig)
        assert cfg.host == "localhost"
        assert cfg.port == 8123
        assert cfg.username == "default"
        assert cfg.password == ""

    def test_environment_overrides_defaults(self, clean_env, monkeypatch):
        monkeypatch.setenv("CLICKHOUSE_DATABASE", "proj")
        monkeypatch.setenv("CLICKHOUSE_HOST", "clickhouse")
        monkeypatch.setenv("CLICKHOUSE_PORT", "9999")
        monkeypatch.setenv("CLICKHOUSE_USER", "llmobs")
        monkeypatch.setenv("CLICKHOUSE_PASSWORD", "secret")
        cfg = get_clickhouse_config()
        assert (cfg.host, cfg.port, cfg.username, cfg.password) == (
            "clickhouse", 9999, "llmobs", "secret")

    def test_config_is_frozen(self, clean_env, monkeypatch):
        monkeypatch.setenv("CLICKHOUSE_DATABASE", "proj")
        cfg = get_clickhouse_config()
        with pytest.raises(Exception):
            cfg.database = "other"


class TestResolveDatabase:
    def test_explicit_wins_over_everything(self, clean_env, monkeypatch):
        monkeypatch.setenv("CLICKHOUSE_DATABASE", "from_env")
        assert resolve_database(name="my-app", explicit_database="explicit") == "explicit"

    def test_env_wins_over_name(self, clean_env, monkeypatch):
        monkeypatch.setenv("CLICKHOUSE_DATABASE", "from_env")
        assert resolve_database(name="my-app") == "from_env"

    def test_name_is_slugified_when_nothing_else_is_set(self, clean_env):
        assert resolve_database(name="My App") == "my_app"

    def test_nothing_configured_is_an_error(self, clean_env):
        with pytest.raises(ConfigError):
            resolve_database()
