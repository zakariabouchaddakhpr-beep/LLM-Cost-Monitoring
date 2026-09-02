import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DATABASE_ENV_VAR = "CLICKHOUSE_DATABASE"

os.environ.setdefault("LITELLM_MODE", "PRODUCTION")


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
    return Path(dotenv_path)


_ENV_FILE = _load_project_env()

PROJECT_ROOT = _ENV_FILE.parent if _ENV_FILE else None

JUDGE_MODEL = "groq/llama-3.3-70b-versatile"

MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 1.5

_ENV_CONFIG = SettingsConfigDict(
    env_file=_ENV_FILE,
    env_file_encoding="utf-8",
    case_sensitive=False,
    extra="ignore",
)


class AppSettings(BaseSettings):
    model_config = _ENV_CONFIG

    project_name: str = ""
    otel_exporter_otlp_endpoint: str | None = None
    otel_service_name: str = "llm-monitor"


class ClickHouseSettings(BaseSettings):
    model_config = _ENV_CONFIG

    clickhouse_host: str = "localhost"
    clickhouse_port: int = 8123
    clickhouse_user: str = "default"
    clickhouse_password: str = ""
    clickhouse_database: str | None = None

    @field_validator("clickhouse_host")
    @classmethod
    def _host_or_default(cls, value: str) -> str:
        return value.strip() or "localhost"

    @field_validator("clickhouse_user")
    @classmethod
    def _user_or_default(cls, value: str) -> str:
        return value.strip() or "default"

    @field_validator("clickhouse_database")
    @classmethod
    def _database_or_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None


_app_settings = AppSettings()

PROJECT_NAME = _app_settings.project_name

OTEL_EXPORTER_OTLP_ENDPOINT = _app_settings.otel_exporter_otlp_endpoint
OTEL_SERVICE_NAME = _app_settings.otel_service_name


_MISSING_DATABASE_HELP = (
    f"{DATABASE_ENV_VAR} is not set, so there is no database to write to. "
    f"Add it to your project's .env file, for example:\n"
    f"    {DATABASE_ENV_VAR}=my_project\n"
    "Each project should name its own database: that is what keeps its traces "
    "separate from every other project's. There is deliberately no default - a "
    "fallback is what causes unrelated applications to share one store."
)


def _readable(exc: ValidationError) -> str:
    lines = []
    for error in exc.errors():
        field = ".".join(str(part) for part in error["loc"]) or "setting"
        lines.append(f"    {field.upper()}: {error['msg']}")
    return ("Some settings could not be read from your environment:\n"
            + "\n".join(lines))


def get_settings() -> ClickHouseSettings:
    try:
        return ClickHouseSettings()
    except ValidationError as exc:
        raise ConfigError(_readable(exc)) from exc


class ClickHouseConfig(BaseModel):
    model_config = {"frozen": True}

    host: str
    port: int
    database: str
    username: str
    password: str


def get_clickhouse_config(database: str | None = None) -> ClickHouseConfig:
    settings = get_settings()

    chosen = database if database is not None else settings.clickhouse_database

    if chosen is None or not str(chosen).strip():
        raise ConfigError(_MISSING_DATABASE_HELP)

    return ClickHouseConfig(
        host=settings.clickhouse_host,
        port=settings.clickhouse_port,
        database=str(chosen).strip(),
        username=settings.clickhouse_user,
        password=settings.clickhouse_password,
    )


def dashboard_env(config: ClickHouseConfig,
                  project_name: str | None = None) -> dict[str, str]:
    env = dict(os.environ)

    env.update({
        "CLICKHOUSE_DATABASE": config.database,
        "CLICKHOUSE_HOST": config.host,
        "CLICKHOUSE_PORT": str(config.port),
        "CLICKHOUSE_USER": config.username,
        "CLICKHOUSE_PASSWORD": config.password,
    })

    if project_name:
        env["PROJECT_NAME"] = project_name

    return env


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

    from_env = get_settings().clickhouse_database
    if from_env:
        return from_env

    if name and name.strip():
        return slug(name)

    raise ConfigError(
        "Nothing identifies this project's database. Pass name= (for example "
        f'name="my-app") or set the {DATABASE_ENV_VAR} environment variable.'
    )
