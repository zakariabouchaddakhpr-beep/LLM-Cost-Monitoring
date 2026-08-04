"""
Run the dashboard without going through init():

    python -m llm_observability.dashboard --port 8501

The database comes from CLICKHOUSE_DATABASE, or from --name. This is what the
Docker `dashboard` service runs, so the container and `obs.dashboard()` take
exactly the same path — including the working directory that makes the theme
load.
"""

import argparse
import sys

from llm_observability.api import launch_dashboard
from llm_observability.config.settings import ConfigError
from llm_observability.core.infrastructure.persistence.database import (
    ClickHouseUnavailable,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m llm_observability.dashboard")
    parser.add_argument("--name", default=None,
                        help="Project name, used when CLICKHOUSE_DATABASE is not set.")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--address", default=None,
                        help="Bind address. Containers need 0.0.0.0.")
    args = parser.parse_args(argv)

    try:
        completed = launch_dashboard(name=args.name, port=args.port,
                                     address=args.address, block=True)
    except (ConfigError, ClickHouseUnavailable) as exc:
        # An unreachable server is the one new failure mode a server introduces,
        # and deserves the same clean exit as a missing configuration rather
        # than a driver traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
