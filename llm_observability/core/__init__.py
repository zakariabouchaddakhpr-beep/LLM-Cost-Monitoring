"""
Shortcut imports, so older code that used shorter paths keeps working.

WHAT IS AN __init__.py FILE?
It's what turns a folder into something Python can import from. It also runs
when you import that folder, which makes it a handy place to pull a few
commonly-used names up to the top level.

So these two lines do exactly the same thing:

    from llm_observability.core import LLMCallResult                    # short
    from llm_observability.core.domain.result import LLMCallResult      # full

Use the full path in new code. It's longer, but it tells the reader which
part of the project the thing comes from:

    core.domain          plain data and maths
    core.application     the logic that uses them
    core.infrastructure  databases, AI providers, monitoring

DELIBERATELY NOT LISTED HERE: LLMMonitor and the marking functions.
Importing either one pulls in litellm, which is slow to load. If they were
listed here, then simply importing `core` would become slow - including for
the dashboard, which only ever needs the database part. Import those two from
llm_observability.core.application.monitor instead.
"""

from llm_observability.core.domain.result import LLMCallResult
from llm_observability.core.domain.metrics import (
    compute_throughput,
    compute_cost_efficiency,
)
from llm_observability.core.infrastructure.persistence.repository import (
    TraceRepository,
)

__all__ = [
    "LLMCallResult",
    "compute_throughput",
    "compute_cost_efficiency",
    "TraceRepository",
]
