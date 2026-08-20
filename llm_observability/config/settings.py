"""
All the settings for this library, in one place.

WHAT IS AN ENVIRONMENT VARIABLE?
It is a setting that lives outside your code, attached to the program while
it runs. You set them in a file called `.env`, like this:

    CLICKHOUSE_DATABASE=my_project
    CLICKHOUSE_HOST=localhost

They exist so you don't have to type passwords and addresses into your source
code, where they would end up shared with everyone who reads it.

Nothing here is fixed to one project. Every value is read from the settings
of whichever app is using this library, so two different apps can use it and
keep two completely separate sets of data.

Notice what is NOT in this file: which AI company and which model to use.
Those get passed in directly when you create an Observer, because that is the
one place that actually decides which model is being called.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

# The name of the setting that says which database this project uses.
# Written out once here so a typo can only happen in one place.
DATABASE_ENV_VAR = "CLICKHOUSE_DATABASE"


class ConfigError(RuntimeError):
    """
    Our own error type, for when a setting is missing or unusable.

    Having our own type means code elsewhere can catch specifically this
    ("the settings are wrong") and not accidentally catch unrelated crashes
    at the same time.
    """


def _load_project_env() -> Path | None:
    """
    Find the app's `.env` file and load the settings out of it.

    `usecwd=True` tells the search to start from the folder you ran the
    program in, and work upwards - NOT from the folder this file is in.
    That matters. We want the `.env` belonging to the app being monitored,
    not one sitting next to our library's source code.

    Gives back the folder the `.env` was found in, or None if there isn't one.
    """
    try:
        dotenv_path = find_dotenv(usecwd=True)
        if not dotenv_path:
            return None
        # override=False means: if a setting is ALREADY set on the computer,
        # leave it alone and don't replace it with the one from the file.
        # This is what lets Docker and build servers override these values.
        load_dotenv(dotenv_path, override=False)
    except OSError:
        # OSError means the file is there but we couldn't read it - for
        # example a OneDrive file that shows in the folder but hasn't actually
        # been downloaded yet. Not worth crashing over: carry on with whatever
        # settings the computer already has.
        return None
    return Path(dotenv_path).parent


# This runs once, the moment this file is first loaded, so the settings are
# ready before anything below tries to read them.
PROJECT_ROOT = _load_project_env()

# A friendly name shown at the top of the dashboard. Just for looks.
PROJECT_NAME = os.getenv("PROJECT_NAME", "")

# The model used as a "judge" to mark how good an answer was. See
# core/application/quality_scoring.py. This feature is switched off by default.
JUDGE_MODEL = "groq/llama-3.3-70b-versatile"

# What to do when a call to the AI fails. Used by monitor.py.
MAX_RETRIES = 2              # extra tries AFTER the first one, so 3 tries in total
RETRY_BACKOFF_SECONDS = 1.5  # wait 1.5s before try 2, then 3s before try 3

# OpenTelemetry settings. Leave the address empty and traces are just printed
# to your terminal instead of being sent anywhere.
OTEL_EXPORTER_OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
OTEL_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "llm-monitor")


# The message shown when nobody said which database to use. Written out here,
# rather than inline below, because it is long and would bury the code.
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
    """
    Everything needed to connect to the database, in one small object.

    `frozen=True` means that once you make one of these, nobody can change
    what's inside it. If some code tried `config.database = "something_else"`,
    Python would refuse. Once we have worked out where to connect, quietly
    changing it later would be a nasty bug to track down.
    """

    host: str       # the machine the database is on, e.g. "localhost"
    port: int       # the numbered door on that machine, e.g. 8123
    database: str   # which database on that server
    username: str
    password: str


def get_clickhouse_config(database: str | None = None) -> ClickHouseConfig:
    """
    Work out how to connect to the database.

    The database NAME is what keeps one project's data separate from another
    project's, so we refuse to guess it. We look in two places, in order:

        1. the `database` value passed into this function
        2. the CLICKHOUSE_DATABASE setting

    ...and raise ConfigError if neither of them has anything.

    The host, port, username and password DO have sensible defaults, because
    they only describe where the server is - not whose data lives in it.
    Getting those wrong gives you a connection error you'll notice
    immediately. Getting the database name wrong silently mixes two projects'
    data together, which you might not notice for months.

    All of this is read when the function runs, not when the file is loaded.
    That way an app can set these values while starting up, and simply
    importing this library never fails.

    Raises:
        ConfigError: if nobody said which database to use.
    """
    # `if X is not None else Y` picks the passed-in value when there is one,
    # and falls back to the setting when there isn't.
    chosen = database if database is not None else os.getenv(DATABASE_ENV_VAR)

    # Catches both "nothing at all" and "just spaces", which is what you get
    # from a line like `CLICKHOUSE_DATABASE=   ` in a .env file.
    if chosen is None or not str(chosen).strip():
        raise ConfigError(_MISSING_DATABASE_HELP)

    return ClickHouseConfig(
        # os.getenv(name, default) reads a setting, or uses the default if it
        # isn't there. The extra `or "localhost"` covers a different case: the
        # setting exists but is blank, which .strip() would turn into "".
        host=os.getenv("CLICKHOUSE_HOST", "localhost").strip() or "localhost",
        # 8123 is the door ClickHouse uses for ordinary web-style requests.
        # Don't use 9000 - the database also listens there, but it speaks a
        # different language on that door, which our driver doesn't know.
        port=int(os.getenv("CLICKHOUSE_PORT", "8123")),
        database=str(chosen).strip(),
        username=os.getenv("CLICKHOUSE_USER", "default").strip() or "default",
        # Deliberately no .strip() here. A password is allowed to start or end
        # with a space, and trimming it would break the login in a way that
        # looks impossible to explain.
        password=os.getenv("CLICKHOUSE_PASSWORD", ""),
    )


def slug(name: str) -> str:
    """
    Turn a project name into something the database will accept as a name.

        "My App!"      ->  "my_app"
        "a---b"        ->  "a_b"
        "2024 report"  ->  "p_2024_report"
        "   "          ->  "project"

    Databases are fussy about names. Two rules matter here:

      - Only letters, numbers and underscores are allowed. Anything else has
        to be wrapped in backticks every single time you mention it in SQL.
        Rather than do that forever, we swap the awkward characters out.
        Underscore rather than hyphen, because "my-app" would need those
        backticks but "my_app" doesn't.

      - A name can't start with a number, so those get "p_" stuck on the
        front. (p for project.)
    """
    # Step 1: make it lowercase, then go through it one character at a time.
    # Keep letters, numbers and underscores. Turn everything else into "_".
    #
    # .isalnum() is True for letters and numbers. This is a "generator
    # expression" - it does the same job as a for-loop that builds a list,
    # just written on one line. "".join(...) then glues the characters
    # back together into a single piece of text.
    cleaned = "".join(
        character if (character.isalnum() or character == "_") else "_"
        for character in name.strip().lower()
    )

    # Step 2: chop off any underscores left at the very start or end.
    # "_my_app_" becomes "my_app".
    cleaned = cleaned.strip("_")

    # Step 3: squash runs of underscores down to one. "a---b" became "a___b"
    # in step 1, and this loop turns it into "a_b". It repeats because one
    # pass over "a____b" leaves "a__b" behind.
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")

    # Step 4: the two awkward cases.
    # Nothing left at all (the name was just punctuation, or blank).
    if not cleaned:
        return "project"
    # Starts with a number, which databases won't accept.
    if cleaned[0].isdigit():
        return f"p_{cleaned}"
    return cleaned


def resolve_database(name: str | None = None,
                     explicit_database: str | None = None) -> str:
    """
    Decide which database this project's records go into.

    Looks in three places and takes the first one that has an answer:

        1. explicit_database   - a name handed straight to this function
        2. CLICKHOUSE_DATABASE - the setting
        3. slug(name)          - the project name, tidied up

    The order goes from most specific to least. Something typed directly into
    the code beats a general setting, which beats a name we had to guess from.

    Raises:
        ConfigError: if all three are empty. We refuse to invent a name here,
                     because a made-up default is exactly how two unrelated
                     projects end up quietly writing into the same database.
    """
    # 1. A name passed straight in beats everything else.
    if explicit_database is not None and str(explicit_database).strip():
        return str(explicit_database).strip()

    # 2. Then the setting.
    from_env = os.getenv(DATABASE_ENV_VAR)
    if from_env is not None and from_env.strip():
        return from_env.strip()

    # 3. Then the project's name, tidied into something a database accepts.
    if name and name.strip():
        return slug(name)

    raise ConfigError(
        "Nothing identifies this project's database. Pass name= (for example "
        f'name="my-app") or set the {DATABASE_ENV_VAR} environment variable.'
    )
