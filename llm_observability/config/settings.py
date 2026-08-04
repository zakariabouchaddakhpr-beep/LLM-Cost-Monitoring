"""
Configuration for the library: environment variables and constants.

Everything here is resolved from the environment of the *consuming* project,
so the same library code can be imported by any number of applications and
each one keeps its own observability data.

Provider/model are NOT set here — they're set directly where LLMMonitor is
instantiated, since that's the one place that actually decides which LLM is
being called for a given run.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

DATABASE_ENV_VAR = "CLICKHOUSE_DATABASE"


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or unusable."""


def _load_project_env() -> Path | None:
    """
    Load the `.env` belonging to the application that imported this library.

    `usecwd=True` searches upward from the working directory rather than from
    this file, so an external project's own `.env` is the one that wins.
    Returns the directory containing it, or None if there isn't one.
    """
    try:
        dotenv_path = find_dotenv(usecwd=True)
        if not dotenv_path:
            return None
        load_dotenv(dotenv_path, override=False)
    except OSError:
        # A local .env that cannot be read — for example a OneDrive placeholder
        # file that is not downloaded. Continue with environment variables.
        return None
    return Path(dotenv_path).parent


# Anchor for relative paths: the .env's directory rather than the raw working
# directory, so the same value resolves identically from a subdirectory.
PROJECT_ROOT = _load_project_env()

# Optional label, used for display and docker-compose interpolation only.
PROJECT_NAME = os.getenv("PROJECT_NAME", "")

# Model used as the "judge" to score response quality (optional feature).
JUDGE_MODEL = "groq/llama-3.3-70b-versatile"

MAX_RETRIES = 2              # number of retries AFTER the first attempt
RETRY_BACKOFF_SECONDS = 1.5  # base delay, doubled on each subsequent retry

# OpenTelemetry configuration
OTEL_EXPORTER_OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
OTEL_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "llm-monitor")

_MISSING_DATABASE_HELP = (
    f"{DATABASE_ENV_VAR} is not set, so there is no database to write to. "
    f"Add it to your project's .env file, for example:\n"
    f"    {DATABASE_ENV_VAR}=my_project\n"
    "Each project should name its own database: that is what keeps its traces "
    "separate from every other project's. There is deliberately no default — a "
    "fallback is what causes unrelated applications to share one store."
)


@dataclass(frozen=True)
class ClickHouseConfig:
    """Everything needed to open a connection, resolved from the environment."""

    host: str
    port: int
    database: str
    username: str
    password: str


def get_clickhouse_config(database: str | None = None) -> ClickHouseConfig:
    """
    Resolve ClickHouse connection settings for the current project.

    `database` is the isolation boundary and has NO default: an explicit
    argument, then `CLICKHOUSE_DATABASE`, then ConfigError. Host, port, username
    and password describe infrastructure rather than identity, so they default.

    Read at call time rather than import time, so a host application can set the
    variables before its first LLM call and importing never fails.

    Raises:
        ConfigError: if no database is configured.
    """
    raw = database if database is not None else os.getenv(DATABASE_ENV_VAR)
    if raw is None or not str(raw).strip():
        raise ConfigError(_MISSING_DATABASE_HELP)

    return ClickHouseConfig(
        host=os.getenv("CLICKHOUSE_HOST", "localhost").strip() or "localhost",
        # 8123 is the HTTP interface. The native protocol on 9000 is a different
        # wire format that clickhouse-connect does not speak.
        port=int(os.getenv("CLICKHOUSE_PORT", "8123")),
        database=str(raw).strip(),
        username=os.getenv("CLICKHOUSE_USER", "default").strip() or "default",
        # No .strip(): a password may legitimately contain leading whitespace.
        password=os.getenv("CLICKHOUSE_PASSWORD", ""),
    )


def slug(name: str) -> str:
    """
    Reduce a project name to a legal ClickHouse identifier.

    Underscores rather than hyphens: `my-app` would need backtick quoting in
    every statement, `my_app` does not. A leading digit is also illegal
    unquoted, so it gets a `p_` prefix.
    """
    kept = [c if (c.isalnum() or c == "_") else "_" for c in name.strip().lower()]
    out = "".join(kept).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    if not out:
        return "project"
    return f"p_{out}" if out[0].isdigit() else out


def resolve_database(name: str | None = None,
                     explicit_database: str | None = None) -> str:
    """
    Which ClickHouse database this project's traces go to.

    Precedence, unchanged from the DuckDB version — only the third step
    changes shape, yielding a database name rather than a file path:
      1. `explicit_database`, when a caller passes one directly
      2. the `CLICKHOUSE_DATABASE` environment variable
      3. `slug(name)`

    Raises:
        ConfigError: if none of the three is available.
    """
    if explicit_database is not None and str(explicit_database).strip():
        return str(explicit_database).strip()

    env_value = os.getenv(DATABASE_ENV_VAR)
    if env_value is not None and env_value.strip():
        return env_value.strip()

    if name and name.strip():
        return slug(name)

    raise ConfigError(
        "Nothing identifies this project's database. Pass name= (for example "
        f'name="my-app") or set the {DATABASE_ENV_VAR} environment variable.'
    )
