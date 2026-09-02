"""The logic: make a call, measure it, mark it."""

from llm_observability.core.application.monitor import LLMMonitor
from llm_observability.core.application.quality_scoring import QUERY_TYPES, evaluate

__all__ = ["LLMMonitor", "QUERY_TYPES", "evaluate"]
