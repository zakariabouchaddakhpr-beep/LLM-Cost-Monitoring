"""
The public API: three calls to instrument a project.

    import llm_observability as obs

    obs.init(name="my-app", model="cerebras/gpt-oss-120b", api_key=...)
    result = obs.call("Summarise this ticket", tenant_id="acme", use_case="support")
    obs.dashboard()

Everything here is a thin layer over LLMMonitor and TraceRepository. litellm is
imported lazily inside the call path so `import llm_observability` stays cheap.
"""

import os
import subprocess
import sys
from pathlib import Path

from llm_observability.config.settings import resolve_database

_PACKAGE_DIR = Path(__file__).resolve().parent
_DASHBOARD_APP = _PACKAGE_DIR / "dashboard" / "app.py"


def _provider_from_model(model: str) -> str:
    """
    "cerebras/gpt-oss-120b" -> "cerebras".

    LiteLLM already routes on this prefix, so the caller does not need to repeat
    it. Models without a prefix keep their full name as the label.
    """
    return model.split("/", 1)[0] if "/" in model else model


class Observer:
    """One project's instrumentation: a model, a key, and a database."""

    def __init__(self, name: str, model: str, api_key: str | None = None, *,
                 score_quality: bool = False, classify_query: bool = False,
                 judge_api_key: str | None = None,
                 database: str | None = None):
        if not name or not name.strip():
            raise ValueError('name is required, e.g. name="my-app"')
        if not model or not model.strip():
            raise ValueError('model is required, e.g. model="cerebras/gpt-oss-120b"')

        self.name = name.strip()
        self.model = model.strip()
        self.provider = _provider_from_model(self.model)
        self.score_quality = score_quality
        self.classify_query = classify_query
        self._api_key = api_key
        self._judge_api_key = judge_api_key
        # Resolved once so every save and the dashboard agree on the database.
        self._database = resolve_database(self.name, explicit_database=database)

        # Imported here rather than at module level: litellm is a heavy import
        # and `import llm_observability` should not pay for it.
        from llm_observability.core.application.monitor import LLMMonitor

        self._monitor = LLMMonitor(provider_name=self.provider, model=self.model,
                                   api_key=api_key)

        # One client for the Observer's lifetime, opened now so a misconfigured
        # database or an unreachable server fails at init rather than after the
        # first paid API call. The previous version reconnected per call to
        # avoid holding DuckDB's write lock; a server has no such lock, so that
        # would now be pure overhead. Creating the schema is idempotent.
        from llm_observability.core.infrastructure.persistence import TraceRepository

        self._repo = TraceRepository(self._database)

    @property
    def database(self) -> str:
        """The ClickHouse database this project's traces go to."""
        return self._database

    def call(self, prompt: str, *, tenant_id: str | None = None,
             use_case: str | None = None):
        """
        Make an instrumented LLM call and save the trace.

        Returns the LLMCallResult. A failed call returns a result with .error
        set rather than raising, so instrumentation never takes down the caller.
        """
        result = self._monitor.call(
            prompt,
            score_response_quality=self.score_quality,
            classify_query=self.classify_query,
            tenant_id=tenant_id,
            use_case=use_case,
            judge_api_key=self._judge_api_key,
        )
        # Logs and swallows on failure, for the same reason the call above
        # returns an errored result instead of raising: instrumentation must
        # never be the reason the calling application goes down.
        self._repo.save(result)
        return result

    def embed(self, inputs, *, tenant_id: str | None = None,
              use_case: str | None = None):
        """
        Make an instrumented embedding call and save the trace.

        Accepts one string or a list. A batch is one billed request, so it
        produces one trace with input_count set - not one row per input.

        The vectors come back on result.embeddings; only the metadata is
        persisted, because the vectors are large and the caller needs them in
        hand rather than in a database.
        """
        result = self._monitor.embed(inputs, tenant_id=tenant_id,
                                     use_case=use_case)
        self._repo.save(result)
        return result

    def summary(self) -> dict:
        """Aggregate totals for this project."""
        return self._repo.get_summary()

    def close(self) -> None:
        """Release the ClickHouse connection."""
        self._repo.close()

    def dashboard(self, *, port: int = 8501, address: str | None = None,
                  block: bool = True):
        """
        Launch the Streamlit dashboard for this project's database.

        Blocks until you stop it (Ctrl-C). Pass block=False to get the Popen
        back and keep going.
        """
        return launch_dashboard(database=self._database, name=self.name,
                                port=port, address=address, block=block)


def launch_dashboard(database: str | None = None, name: str | None = None, *,
                     port: int = 8501, address: str | None = None,
                     block: bool = True):
    """
    Start the dashboard for one database. Shared by `Observer.dashboard()` and
    `python -m llm_observability.dashboard`.

    address defaults to Streamlit's own (localhost). Containers must pass
    "0.0.0.0", or the published port looks dead from the host.
    """
    from llm_observability.config.settings import get_clickhouse_config

    database = database or resolve_database(name)
    cfg = get_clickhouse_config(database)

    env = {
        **os.environ,
        # Passed explicitly so the dashboard never has to discover a .env from
        # whatever working directory it inherited — the same reason the previous
        # version passed a resolved absolute path.
        "CLICKHOUSE_DATABASE": cfg.database,
        "CLICKHOUSE_HOST": cfg.host,
        "CLICKHOUSE_PORT": str(cfg.port),
        "CLICKHOUSE_USER": cfg.username,
        "CLICKHOUSE_PASSWORD": cfg.password,
    }
    if name:
        env["PROJECT_NAME"] = name

    cmd = [sys.executable, "-m", "streamlit", "run", str(_DASHBOARD_APP),
           "--server.port", str(port), "--server.headless", "true"]
    if address:
        cmd += ["--server.address", address]
    # cwd is the package directory because Streamlit reads .streamlit/config.toml
    # from the working directory, and that file ships inside the package. Running
    # from the caller's folder would silently drop the light/dark theme. Safe to
    # override cwd only because the connection is passed through env above.
    if block:
        return subprocess.run(cmd, cwd=str(_PACKAGE_DIR), env=env)
    return subprocess.Popen(cmd, cwd=str(_PACKAGE_DIR), env=env)


_default: Observer | None = None


def init(name: str, model: str, api_key: str | None = None, *,
         score_quality: bool = False, classify_query: bool = False,
         judge_api_key: str | None = None,
         database: str | None = None) -> Observer:
    """
    Set up instrumentation for this project and return the Observer.

    Also stored as the module default, so `obs.init(...)` followed by
    `obs.call(...)` works without holding on to the return value.

    score_quality and classify_query are both off by default: either one makes a
    second API call to a different provider (a Groq model), which would
    otherwise be a surprise on both the bill and the latency.

    They share that one call, so turning BOTH on costs the same as turning on
    either alone:

        neither          1 API call per obs.call()
        scoring only     2
        classifying only 2
        both             2

    classify_query fills in query_type, the dimension that makes
    v_model_by_query_type - which model is better for which kind of request -
    answerable.
    """
    global _default
    _default = Observer(name=name, model=model, api_key=api_key,
                        score_quality=score_quality,
                        classify_query=classify_query,
                        judge_api_key=judge_api_key, database=database)
    return _default


def _require_default() -> Observer:
    if _default is None:
        raise RuntimeError(
            "llm_observability is not initialised. Call "
            'obs.init(name="my-app", model="cerebras/gpt-oss-120b", api_key=...) first.'
        )
    return _default


def call(prompt: str, *, tenant_id: str | None = None,
         use_case: str | None = None):
    """Instrumented call on the Observer created by init()."""
    return _require_default().call(prompt, tenant_id=tenant_id, use_case=use_case)


def embed(inputs, *, tenant_id: str | None = None,
          use_case: str | None = None):
    """Instrumented embedding call on the Observer created by init()."""
    return _require_default().embed(inputs, tenant_id=tenant_id,
                                    use_case=use_case)


def summary() -> dict:
    """Aggregate totals for the Observer created by init()."""
    return _require_default().summary()


def dashboard(*, port: int = 8501, address: str | None = None,
              block: bool = True):
    """Launch the dashboard for the Observer created by init()."""
    return _require_default().dashboard(port=port, address=address,
                                        block=block)


def database() -> str:
    """The ClickHouse database the Observer created by init() writes to."""
    return _require_default().database
