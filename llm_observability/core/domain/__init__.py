"""Plain data and maths. Imports no database, no web framework, no AI library."""

from llm_observability.core.domain.result import LLMCallResult
from llm_observability.core.domain.metrics import (
    compute_throughput,
    compute_cost_efficiency,
)

__all__ = ["LLMCallResult", "compute_throughput", "compute_cost_efficiency"]
