import subprocess
import sys
from pathlib import Path

from llm_observability.config.settings import resolve_database
from llm_observability.core.domain.routing import choose, resolve

_PACKAGE_DIR = Path(__file__).resolve().parent
_DASHBOARD_APP = _PACKAGE_DIR / "dashboard" / "app.py"


def _provider_from_model(model: str) -> str:
    return model.split("/", 1)[0] if "/" in model else model


class Observer:

    def __init__(self, name: str, model: str | dict, api_key: str | None = None, *,
                 score_quality: bool = False, classify_query: bool = False,
                 judge_api_key: str | None = None,
                 database: str | None = None):
        if not name or not name.strip():
            raise ValueError('name is required, e.g. name="my-app"')
        if not model:
            raise ValueError('model is required, e.g. model="cerebras/gpt-oss-120b"')

        self.name = name.strip()
        self._models = (model if isinstance(model, dict)
                        else {"weak": model.strip()})
        self._default_tier = next(iter(self._models))
        self.score_quality = score_quality
        self.classify_query = classify_query
        self._api_key = api_key
        self._judge_api_key = judge_api_key

        self._database = resolve_database(self.name, explicit_database=database)

        from llm_observability.core.application.monitor import LLMMonitor

        self._monitors = {
            tier: LLMMonitor(provider_name=_provider_from_model(model_string),
                             model=model_string, api_key=api_key)
            for tier, model_string in self._models.items()
        }

        self._classifier = None
        if isinstance(model, dict):
            from llm_observability.core.infrastructure.classifier import (
                PromptClassifier,
            )

            self._classifier = PromptClassifier()

        from llm_observability.core.infrastructure.persistence import TraceRepository

        self._repo = TraceRepository(self._database)

    @property
    def database(self) -> str:
        return self._database

    def _pick(self, prompt: str):
        if self._classifier is None:
            return self._default_tier, None

        profile = self._classifier.classify(prompt)
        tier = resolve(choose(profile.task, profile.complexity), self._models)
        return tier, profile

    def call(self, prompt: str, *, tenant_id: str | None = None,
             use_case: str | None = None):
        tier, profile = self._pick(prompt)

        result = self._monitors[tier].call(
            prompt,
            score_response_quality=self.score_quality,
            classify_query=self.classify_query,
            tenant_id=tenant_id,
            use_case=use_case,
            judge_api_key=self._judge_api_key,
        )

        result.tier = tier
        if profile is not None:
            result.task_type = profile.task
            result.complexity = profile.complexity

        self._repo.save(result)
        return result

    def embed(self, inputs, *, tenant_id: str | None = None,
              use_case: str | None = None):
        result = self._monitors[self._default_tier].embed(
            inputs, tenant_id=tenant_id, use_case=use_case)
        self._repo.save(result)
        return result

    def summary(self) -> dict:
        return self._repo.get_summary()

    def close(self) -> None:
        self._repo.close()

    def dashboard(self, *, port: int = 8501, address: str | None = None,
                  block: bool = True):
        return launch_dashboard(database=self._database, name=self.name,
                                port=port, address=address, block=block)


def launch_dashboard(database: str | None = None, name: str | None = None, *,
                     port: int = 8501, address: str | None = None,
                     block: bool = True):
    from llm_observability.config.settings import (
        dashboard_env,
        get_clickhouse_config,
    )

    database = database or resolve_database(name)
    cfg = get_clickhouse_config(database)

    env = dashboard_env(cfg, name)

    cmd = [sys.executable, "-m", "streamlit", "run", str(_DASHBOARD_APP),
           "--server.port", str(port), "--server.headless", "true"]
    if address:
        cmd += ["--server.address", address]

    if block:
        return subprocess.run(cmd, cwd=str(_PACKAGE_DIR), env=env)
    return subprocess.Popen(cmd, cwd=str(_PACKAGE_DIR), env=env)


_default: Observer | None = None


def init(name: str, model: str | dict, api_key: str | None = None, *,
         score_quality: bool = False, classify_query: bool = False,
         judge_api_key: str | None = None,
         database: str | None = None) -> Observer:
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
    return _require_default().call(prompt, tenant_id=tenant_id,
                                   use_case=use_case)


def embed(inputs, *, tenant_id: str | None = None,
          use_case: str | None = None):
    return _require_default().embed(inputs, tenant_id=tenant_id,
                                    use_case=use_case)


def summary() -> dict:
    return _require_default().summary()


def dashboard(*, port: int = 8501, address: str | None = None,
              block: bool = True):
    return _require_default().dashboard(port=port, address=address, block=block)


def database() -> str:
    return _require_default().database
