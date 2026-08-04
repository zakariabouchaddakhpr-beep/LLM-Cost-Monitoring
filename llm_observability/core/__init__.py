"""
Backward-compatibility re-exports so existing imports don't break.

Preferred imports go through their layer:
  from llm_observability.core.domain.result import LLMCallResult
  from llm_observability.core.application.monitor import LLMMonitor
  from llm_observability.core.infrastructure.persistence import TraceRepository

LLMMonitor and score_quality are deliberately not re-exported here: importing
them pulls in litellm, which would make `import core` (and therefore the
dashboard, which only needs persistence) pay for it. Import them from
core.application.monitor instead.
"""

from llm_observability.core.domain.result import LLMCallResult
from llm_observability.core.domain.metrics import compute_throughput, compute_cost_efficiency
from llm_observability.core.infrastructure.persistence.repository import TraceRepository

__all__ = [
    "LLMCallResult",
    "compute_throughput",
    "compute_cost_efficiency",
    "TraceRepository",
]
