"""
The  front door - the three calls most people will ever need.
    import llm_observability as obs
    obs.init(name="my-app", model="cerebras/gpt-oss-120b", api_key=...)
    result = obs.call("Summarise this ticket", tenant_id="acme", use_case="support")
    obs.dashboard()
"""

import os
import subprocess
import sys
from pathlib import Path

from llm_observability.config.settings import resolve_database

# Where this package lives on disk, and where the dashboard script sits inside
# it. __file__ is a variable Python fills in with the path of the current
# file, so these are worked out fresh at runtime and stay correct no matter
# where the package ends up installed.
_PACKAGE_DIR = Path(__file__).resolve().parent
_DASHBOARD_APP = _PACKAGE_DIR / "dashboard" / "app.py"


def _provider_from_model(model: str) -> str:
    """
    Take the company name off the front of a model name.
    """
    return model.split("/", 1)[0] if "/" in model else model


class Observer:
    """
    One project's monitoring setup: a model, an API key, and a database.
    """

    def __init__(self, name: str, model: str, api_key: str | None = None, *,
                 score_quality: bool = False, classify_query: bool = False,
                 judge_api_key: str | None = None,
                 database: str | None = None):
        # Check the two required values immediately, with messages that show
        # what a correct answer looks like. Failing here, at setup, is far
        # kinder than failing later during a real call.
        if not name or not name.strip():
            raise ValueError('name is required, e.g. name="my-app"')
        if not model or not model.strip():
            raise ValueError('model is required, e.g. model="cerebras/gpt-oss-120b"')

        # .strip() removes accidental spaces at either end.
        self.name = name.strip()
        self.model = model.strip()
        self.provider = _provider_from_model(self.model)
        self.score_quality = score_quality
        self.classify_query = classify_query
        self._api_key = api_key
        self._judge_api_key = judge_api_key

        # Worked out ONCE, here. That way every save and the dashboard all
        # agree on which database this project uses. Working it out separately
        # in several places is how they end up disagreeing.
        self._database = resolve_database(self.name, explicit_database=database)

        # This import sits inside __init__ rather than at the top of the file
        # on purpose. Loading litellm is slow, and simply typing
        # `import llm_observability` shouldn't pay that cost for someone who
        # never ends up making a call.
        from llm_observability.core.application.monitor import LLMMonitor

        self._monitor = LLMMonitor(provider_name=self.provider,
                                   model=self.model, api_key=api_key)

        # Open the database connection NOW, rather than waiting until the
        # first save. If the settings are wrong or the server is down, you
        # find out here - not after you've already paid for an API call whose
        # result you then can't record.
        from llm_observability.core.infrastructure.persistence import TraceRepository

        self._repo = TraceRepository(self._database)

    @property
    def database(self) -> str:
        """The name of the database this project's records go into."""
        return self._database

    def call(self, prompt: str, *, tenant_id: str | None = None,
             use_case: str | None = None):
        """
        Make one measured AI call and save the record.
        """
        result = self._monitor.call(
            prompt,
            score_response_quality=self.score_quality,
            classify_query=self.classify_query,
            tenant_id=tenant_id,
            use_case=use_case,
            judge_api_key=self._judge_api_key,
        )
        # save() writes a warning and carries on if it fails, for exactly the
        self._repo.save(result)
        return result

    def embed(self, inputs, *, tenant_id: str | None = None,
              use_case: str | None = None):
        """
        Make one measured embedding call and save the record.
        """
        result = self._monitor.embed(inputs, tenant_id=tenant_id,
                                     use_case=use_case)
        self._repo.save(result)
        return result

    def summary(self) -> dict:
        """Totals for this project: calls, cost, average speed and quality."""
        return self._repo.get_summary()

    def close(self) -> None:
        """Close the database connection."""
        self._repo.close()

    def dashboard(self, *, port: int = 8501, address: str | None = None,
                  block: bool = True):
        """
        Open the web dashboard for this project.
        """
        return launch_dashboard(database=self._database, name=self.name,
                                port=port, address=address, block=block)


def launch_dashboard(database: str | None = None, name: str | None = None, *,
                     port: int = 8501, address: str | None = None,
                     block: bool = True):
    """
    Start the dashboard as a separate program.
    """
    from llm_observability.config.settings import get_clickhouse_config

    database = database or resolve_database(name)
    cfg = get_clickhouse_config(database)

    env = {
        **os.environ,
        "CLICKHOUSE_DATABASE": cfg.database,
        "CLICKHOUSE_HOST": cfg.host,
        "CLICKHOUSE_PORT": str(cfg.port),
        "CLICKHOUSE_USER": cfg.username,
        "CLICKHOUSE_PASSWORD": cfg.password,
    }
    if name:
        env["PROJECT_NAME"] = name

    # sys.executable is the exact Python program currently running.
    cmd = [sys.executable, "-m", "streamlit", "run", str(_DASHBOARD_APP),
           "--server.port", str(port), "--server.headless", "true"]
    if address:
        cmd += ["--server.address", address]

    if block:
        # subprocess.run starts the program and WAITS for it to finish.
        return subprocess.run(cmd, cwd=str(_PACKAGE_DIR), env=env)
    # subprocess.Popen starts it and comes straight back.
    return subprocess.Popen(cmd, cwd=str(_PACKAGE_DIR), env=env)


_default: Observer | None = None


def init(name: str, model: str, api_key: str | None = None, *,
         score_quality: bool = False, classify_query: bool = False,
         judge_api_key: str | None = None,
         database: str | None = None) -> Observer:
    """
    Set up monitoring for this project. Call this once, when your app starts.

    """
    # `global` means: change the _default defined at the top of this file,
    # rather than quietly creating a new variable that only exists in here.
    global _default
    _default = Observer(name=name, model=model, api_key=api_key,
                        score_quality=score_quality,
                        classify_query=classify_query,
                        judge_api_key=judge_api_key, database=database)
    return _default


def _require_default() -> Observer:
    """
    Fetch the Observer that init() made, or explain that it's missing.

    """
    if _default is None:
        raise RuntimeError(
            "llm_observability is not initialised. Call "
            'obs.init(name="my-app", model="cerebras/gpt-oss-120b", api_key=...) first.'
        )
    return _default


# Everything below is a shortcut. Each one just passes your request along to
# the Observer that init() made, so you never have to hold on to it yourself.
# This is why `obs.call(...)` works after a single `obs.init(...)`.

def call(prompt: str, *, tenant_id: str | None = None,
         use_case: str | None = None):
    """Make a measured AI call, using the Observer that init() made."""
    return _require_default().call(prompt, tenant_id=tenant_id,
                                   use_case=use_case)


def embed(inputs, *, tenant_id: str | None = None,
          use_case: str | None = None):
    """Make a measured embedding call, using the Observer that init() made."""
    return _require_default().embed(inputs, tenant_id=tenant_id,
                                    use_case=use_case)


def summary() -> dict:
    """Totals for the project that init() set up."""
    return _require_default().summary()


def dashboard(*, port: int = 8501, address: str | None = None,
              block: bool = True):
    """Open the dashboard for the project that init() set up."""
    return _require_default().dashboard(port=port, address=address, block=block)


def database() -> str:
    """The database name the project that init() set up writes to."""
    return _require_default().database
