"""
The "application" folder: the actual logic of the tool.

This is where the plain data objects from `domain` and the outside-world
connections from `infrastructure` get put to work together. The domain folder
knows what a call result looks like; the infrastructure folder knows how to
reach an AI company. This folder is what actually makes the call, times it,
and fills the result in.

    monitor.py          makes a call and measures everything about it
    quality_scoring.py  optionally asks a second model to mark the answer
"""

from llm_observability.core.application.monitor import LLMMonitor
from llm_observability.core.application.quality_scoring import QUERY_TYPES, evaluate

__all__ = ["LLMMonitor", "QUERY_TYPES", "evaluate"]
