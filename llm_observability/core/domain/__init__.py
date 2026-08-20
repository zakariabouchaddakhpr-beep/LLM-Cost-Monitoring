"""
The "domain" folder: plain data and simple maths.

Nothing in here imports a database, a web framework, or an AI library. It is
the part of the project everything else agrees on and builds around.

That also makes it the easiest part to test. No server, no internet, no API
key - you can create these objects and check them straight from a Python
prompt.

    result.py   LLMCallResult, the object describing one AI call
    metrics.py  two small sums done on its numbers
"""

from llm_observability.core.domain.result import LLMCallResult
from llm_observability.core.domain.metrics import (
    compute_throughput,
    compute_cost_efficiency,
)

__all__ = ["LLMCallResult", "compute_throughput", "compute_cost_efficiency"]
