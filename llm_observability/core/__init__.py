"""Shortcut imports for the names used most often.

LLMMonitor and the marking functions are deliberately NOT re-exported here:
importing either pulls in litellm, which is slow, and would make a bare
`import core` slow for the dashboard too. Import those from
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
