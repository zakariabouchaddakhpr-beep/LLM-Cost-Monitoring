"""Tests for the public API: llm_observability.init / call / dashboard.

litellm is patched throughout — a real call would cost money, need a key, and
return different text every run.

Note on CLICKHOUSE_DATABASE: importing litellm runs its own `load_dotenv()`,
which walks up from its install directory and can inject this repo's own
environment into the process. Every test here sets or deletes the variable
explicitly, so it never depends on what leaked in. The leak matters more now
than it did under DuckDB: a stray value points a project at another project's
*database* on a shared server, not merely at another file.
"""

import os
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Imported here, at module scope, purely for its side effect: litellm calls
# load_dotenv() on first import. Doing it now means it happens before any fixture
# runs, so the fixtures' delenv is not undone by a later import.
import litellm  # noqa: F401

from llm_observability import api
from llm_observability.config import settings
from llm_observability.config.settings import ConfigError

# Falls back to the project's own CLICKHOUSE_* settings, already loaded from
# .env by settings.py. TEST_CLICKHOUSE_* overrides for a separate server.
CH_HOST = os.getenv("TEST_CLICKHOUSE_HOST") or os.getenv("CLICKHOUSE_HOST", "localhost")
CH_PORT = os.getenv("TEST_CLICKHOUSE_PORT") or os.getenv("CLICKHOUSE_PORT", "8123")
CH_USER = os.getenv("TEST_CLICKHOUSE_USER") or os.getenv("CLICKHOUSE_USER", "default")
CH_PASS = os.getenv("TEST_CLICKHOUSE_PASSWORD") or os.getenv("CLICKHOUSE_PASSWORD", "")


class FakeUsage:
    prompt_tokens = 10
    completion_tokens = 20
    total_tokens = 30


class FakeChoice:
    finish_reason = "stop"
    message = MagicMock(content="Hello world")


class FakeResponse:
    usage = FakeUsage()
    choices = [FakeChoice()]
    id = "resp-1"


@pytest.fixture(scope="session")
def server_available():
    import clickhouse_connect
    try:
        c = clickhouse_connect.get_client(host=CH_HOST, port=int(CH_PORT),
                                          username=CH_USER, password=CH_PASS)
        c.command("SELECT 1")
        c.close()
    except Exception as exc:
        pytest.skip(f"No ClickHouse at {CH_HOST}:{CH_PORT} ({exc}). "
                    "Start one with: docker compose up -d clickhouse")


@pytest.fixture
def project(server_available, monkeypatch):
    """A clean stand-in for a consuming project: its own database, nothing
    inherited from the environment."""
    monkeypatch.delenv("CLICKHOUSE_DATABASE", raising=False)
    monkeypatch.setenv("CLICKHOUSE_HOST", CH_HOST)
    monkeypatch.setenv("CLICKHOUSE_PORT", CH_PORT)
    monkeypatch.setenv("CLICKHOUSE_USER", CH_USER)
    monkeypatch.setenv("CLICKHOUSE_PASSWORD", CH_PASS)
    monkeypatch.setattr(api, "_default", None)
    created: list[str] = []
    yield created
    # Drop whatever the test created, so runs do not accumulate databases.
    import clickhouse_connect
    c = clickhouse_connect.get_client(host=CH_HOST, port=int(CH_PORT),
                                      username=CH_USER, password=CH_PASS)
    try:
        for name in created:
            c.command(f"DROP DATABASE IF EXISTS {name}")
    finally:
        c.close()


@pytest.fixture
def fake_llm():
    """Patch the provider layer so no network call happens."""
    with patch("litellm.completion", return_value=FakeResponse()) as completion, \
         patch("litellm.completion_cost", return_value=0.00042):
        yield completion


def _unique(prefix="t"):
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


class TestDatabaseResolution:
    def test_name_becomes_the_database(self, project):
        assert settings.resolve_database("my-app") == "my_app"

    def test_name_is_slugified_with_underscores(self, project):
        # Hyphens are not legal in a ClickHouse identifier without quoting.
        assert settings.resolve_database("My App!") == "my_app"
        assert settings.resolve_database("A  //  B") == "a_b"

    def test_env_var_wins_over_name(self, project, monkeypatch):
        monkeypatch.setenv("CLICKHOUSE_DATABASE", "from_env")
        assert settings.resolve_database("my-app") == "from_env"

    def test_explicit_wins_over_both(self, project, monkeypatch):
        monkeypatch.setenv("CLICKHOUSE_DATABASE", "from_env")
        assert settings.resolve_database(
            "my-app", explicit_database="chosen") == "chosen"

    def test_nothing_configured_is_an_error(self, project):
        with pytest.raises(ConfigError):
            settings.resolve_database(None)

    def test_litellm_cannot_leak_a_database_name(self, project):
        """A leaked value would now point at another project's database."""
        import litellm  # noqa: F811  - re-import must not repopulate it
        assert os.getenv("CLICKHOUSE_DATABASE") is None


class TestInit:
    def test_creates_the_database_for_the_project(self, project, fake_llm):
        name = _unique("init")
        project.append(name)
        obs = api.init(name=name, model="cerebras/gpt-oss-120b", api_key="sk-x")
        assert obs.database == name
        # Created at init, so a bad configuration fails before any paid call.
        assert obs.summary()["total_calls"] == 0

    def test_provider_derived_from_the_model_prefix(self, project, fake_llm):
        project.append(_unique("a"))
        obs = api.init(name=project[-1], model="cerebras/gpt-oss-120b")
        assert obs.provider == "cerebras"

    def test_model_without_a_prefix_keeps_its_name(self, project, fake_llm):
        project.append(_unique("a"))
        obs = api.init(name=project[-1], model="gpt-4o")
        assert obs.provider == "gpt-4o"

    def test_reuses_an_existing_database(self, project, fake_llm):
        name = _unique("reuse")
        project.append(name)
        first = api.init(name=name, model="m/x")
        api.call("hello")
        second = api.init(name=name, model="m/x")
        assert first.database == second.database
        # A second init must not start a fresh database.
        assert second.summary()["total_calls"] == 1

    def test_name_and_model_are_required(self, project):
        with pytest.raises(ValueError):
            api.init(name="  ", model="m/x")
        with pytest.raises(ValueError):
            api.init(name="a", model="")


class TestCall:
    def test_saves_the_trace(self, project, fake_llm):
        project.append(_unique("save"))
        api.init(name=project[-1], model="cerebras/gpt-oss-120b")
        result = api.call("hello", tenant_id="acme", use_case="triage")
        assert result.succeeded
        assert api.summary()["total_calls"] == 1

    def test_api_key_reaches_litellm(self, project, fake_llm):
        project.append(_unique("key"))
        api.init(name=project[-1], model="cerebras/gpt-oss-120b",
                 api_key="sk-secret")
        api.call("hello")
        assert fake_llm.call_args.kwargs["api_key"] == "sk-secret"

    def test_no_api_key_means_litellm_finds_its_own(self, project, fake_llm):
        # Passing api_key=None must not send api_key=None to litellm; that would
        # override the key it would otherwise pick up from the environment.
        project.append(_unique("nokey"))
        api.init(name=project[-1], model="cerebras/gpt-oss-120b")
        api.call("hello")
        assert "api_key" not in fake_llm.call_args.kwargs

    def test_quality_scoring_is_off_by_default(self, project, fake_llm):
        # It costs a second call to a different provider, so it must be opt-in.
        project.append(_unique("q"))
        with patch("llm_observability.core.application.monitor.evaluate") as judge:
            api.init(name=project[-1], model="cerebras/gpt-oss-120b")
            result = api.call("hello")
        judge.assert_not_called()
        assert result.quality_score is None

    def test_quality_scoring_can_be_enabled(self, project, fake_llm):
        project.append(_unique("q2"))
        with patch("llm_observability.core.application.monitor.evaluate",
                   return_value={"quality_score": 8, "quality_reasoning": "ok",
                                 "query_type": None}) as judge:
            api.init(name=project[-1], model="m/x", score_quality=True,
                     judge_api_key="gsk-judge")
            result = api.call("hello")
        assert result.quality_score == 8
        assert judge.call_args.kwargs["api_key"] == "gsk-judge"

    def test_a_reader_sees_the_trace_while_the_writer_is_open(self, project, fake_llm):
        """Replaces the old lock-release test. Under DuckDB the writer had to
        close before a reader could open; now they simply coexist."""
        from llm_observability.core.infrastructure.persistence import TraceRepository

        project.append(_unique("live"))
        obs = api.init(name=project[-1], model="m/x")
        api.call("hello")
        reader = TraceRepository(obs.database, create_schema=False)
        try:
            assert reader.query("SELECT count() FROM llm_traces")[0][0] == 1
        finally:
            reader.close()

    def test_two_projects_stay_separate(self, project, fake_llm):
        a_name, b_name = _unique("alpha"), _unique("beta")
        project.extend([a_name, b_name])
        a = api.init(name=a_name, model="m/x")
        api.call("one")
        b = api.init(name=b_name, model="m/x")
        api.call("two")
        assert a.database != b.database
        assert a.summary()["total_calls"] == 1
        assert b.summary()["total_calls"] == 1


class TestModuleLevelHelpers:
    def test_call_before_init_explains_itself(self, project):
        with pytest.raises(RuntimeError, match="not initialised"):
            api.call("hello")

    def test_dashboard_before_init_explains_itself(self, project):
        with pytest.raises(RuntimeError, match="not initialised"):
            api.dashboard()


class TestDashboardLaunch:
    def test_passes_the_connection_and_package_cwd(self, project, fake_llm):
        name = _unique("dash")
        project.append(name)
        obs = api.init(name=name, model="m/x")
        with patch("subprocess.run") as run:
            obs.dashboard(port=8599, address="0.0.0.0")
        cmd = run.call_args.args[0]
        kwargs = run.call_args.kwargs

        assert cmd[1:4] == ["-m", "streamlit", "run"]
        assert cmd[4].endswith("app.py")
        assert "8599" in cmd and "0.0.0.0" in cmd
        # Passed explicitly so the child never rediscovers a .env from whatever
        # working directory it inherited.
        assert kwargs["env"]["CLICKHOUSE_DATABASE"] == name
        assert kwargs["env"]["CLICKHOUSE_HOST"] == CH_HOST
        assert kwargs["env"]["CLICKHOUSE_PORT"] == CH_PORT
        assert kwargs["env"]["PROJECT_NAME"] == name
        # cwd must be the package dir, which is where .streamlit/config.toml
        # lives — otherwise the theme silently does not load.
        assert (Path(kwargs["cwd"]) / ".streamlit" / "config.toml").exists()

    def test_no_address_leaves_streamlit_default(self, project, fake_llm):
        project.append(_unique("dash2"))
        obs = api.init(name=project[-1], model="m/x")
        with patch("subprocess.run") as run:
            obs.dashboard()
        assert "--server.address" not in run.call_args.args[0]
