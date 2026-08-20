"""
Start the dashboard from a terminal, without writing any code:

    python -m llm_observability.dashboard --port 8501

WHY IS THIS FILE CALLED __main__.py?
That exact name is what makes `python -m <package>` work. When you run a
package with -m, Python looks inside it for a file named __main__.py and runs
that. Any other name and the command does nothing.

Which database it opens comes from the CLICKHOUSE_DATABASE setting, or from
the --name option. This is also what the Docker `dashboard` service runs, so
the container and obs.dashboard() end up doing exactly the same thing.
"""

import argparse
import sys

from llm_observability.api import launch_dashboard
from llm_observability.config.settings import ConfigError
from llm_observability.core.infrastructure.persistence.database import (
    ClickHouseUnavailable,
)


def main(argv=None) -> int:
    """
    Read the options typed on the command line, then start the dashboard.

    Hands back an "exit code", which is how a program tells whatever started
    it whether things went well. 0 means success; anything else means failure.
    Build servers and shell scripts check this number.
    """
    # argparse handles reading command-line options for us. It also builds a
    # --help message automatically out of the `help` text below.
    parser = argparse.ArgumentParser(
        prog="python -m llm_observability.dashboard")
    parser.add_argument(
        "--name", default=None,
        help="Project name, used when CLICKHOUSE_DATABASE is not set.")
    # type=int makes argparse convert what you typed into a number, and
    # complain clearly if you typed something that isn't one.
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument(
        "--address", default=None,
        help="Bind address. Containers need 0.0.0.0.")
    args = parser.parse_args(argv)

    try:
        completed = launch_dashboard(name=args.name, port=args.port,
                                     address=args.address, block=True)
    except (ConfigError, ClickHouseUnavailable) as exc:
        # These two are the failures we EXPECT and can explain: a missing
        # setting, or a database server that isn't running. Both deserve one
        # clear line, not a page of red technical text that buries the point.
        #
        # sys.stderr is the error channel, kept separate from normal output so
        # that piping the output somewhere doesn't swallow the error message.
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Pass along whatever the dashboard itself finished with.
    return completed.returncode


if __name__ == "__main__":
    # SystemExit hands the number up to the terminal, so scripts can tell
    # whether this worked. Just calling main() would throw that away.
    raise SystemExit(main())
