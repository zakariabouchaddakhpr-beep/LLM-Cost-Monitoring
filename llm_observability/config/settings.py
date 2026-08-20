import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

DATABASE_ENV_VAR = "CLICKHOUSE_DATABASE"


class ConfigError(RuntimeError):
    pass


def _load_project_env() -> Path | None:
    try:
        dotenv_path = find_dotenv(usecwd=True)
        if not dotenv_path:
            return None
        load_dotenv(dotenv_path, override=False)
    except OSError:
        return None
    return Path(dotenv_path).parent


PROJECT_ROOT = _load_project_env()

PROJECT_NAME = os.getenv("PROJECT_NAME", "")

JUDGE_MODEL = "groq/llama-3.3-70b-versatile"

MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 1.5

OTEL_EXPORTER_OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
OTEL_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "llm-monitor")


_MISSING_DATABASE_HELP = (
    f"{DATABASE_ENV_VAR} is not set, so there is no database to write to. "
    f"Add it to your project's .env file, for example:\n"
    f"    {DATABASE_ENV_VAR}=my_project\n"
    "Each project should name its own database: that is what keeps its traces "
    "separate from every other project's. There is deliberately no default - a "
    "fallback is what causes unrelated applications to share one store."
)


@dataclass(frozen=True)
class ClickHouseConfig:

    host: str
    port: int
    database: str
    username: str
    password: str


def get_clickhouse_config(database: str | None = None) -> ClickHouseConfig:
    chosen = database if database is not None else os.getenv(DATABASE_ENV_VAR)

    if chosen is None or not str(chosen).strip():
        raise ConfigError(_MISSING_DATABASE_HELP)

    return ClickHouseConfig(
        host=os.getenv("CLICKHOUSE_HOST", "localhost").strip() or "localhost",
        port=int(os.getenv("CLICKHOUSE_PORT", "8123")),
        database=str(chosen).strip(),
        username=os.getenv("CLICKHOUSE_USER", "default").strip() or "default",
        password=os.getenv("CLICKHOUSE_PASSWORD", ""),
    )


def slug(name: str) -> str:
    cleaned = "".join(
        character if (character.isalnum() or character == "_") else "_"
        for character in name.strip().lower()
    )

    cleaned = cleaned.strip("_")

    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")

    if not cleaned:
        return "project"
    if cleaned[0].isdigit():
        return f"p_{cleaned}"
    return cleaned


def resolve_database(name: str | None = None,
                     explicit_database: str | None = None) -> str:
    if explicit_database is not None and str(explicit_database).strip():
        return str(explicit_database).strip()

    from_env = os.getenv(DATABASE_ENV_VAR)
    if from_env is not None and from_env.strip():
        return from_env.strip()

    if name and name.strip():
        return slug(name)

    raise ConfigError(
        "Nothing identifies this project's database. Pass name= (for example "
        f'name="my-app") or set the {DATABASE_ENV_VAR} environment variable.'
    )
