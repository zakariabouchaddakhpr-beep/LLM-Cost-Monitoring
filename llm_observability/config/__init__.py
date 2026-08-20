"""
The config folder.

Everything real lives in settings.py. This file just lists those names again,
so both of these work and mean exactly the same thing:

    from llm_observability.config import JUDGE_MODEL
    from llm_observability.config.settings import JUDGE_MODEL

If you're looking for the actual settings, open settings.py.
"""

from llm_observability.config.settings import (
    DATABASE_ENV_VAR,
    JUDGE_MODEL,
    MAX_RETRIES,
    OTEL_EXPORTER_OTLP_ENDPOINT,
    OTEL_SERVICE_NAME,
    PROJECT_NAME,
    PROJECT_ROOT,
    RETRY_BACKOFF_SECONDS,
    ClickHouseConfig,
    ConfigError,
    get_clickhouse_config,
    resolve_database,
    slug,
)

__all__ = [
    "DATABASE_ENV_VAR",
    "JUDGE_MODEL",
    "MAX_RETRIES",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_SERVICE_NAME",
    "PROJECT_NAME",
    "PROJECT_ROOT",
    "RETRY_BACKOFF_SECONDS",
    "ClickHouseConfig",
    "ConfigError",
    "get_clickhouse_config",
    "resolve_database",
    "slug",
]
