"""
Structured result object for a single instrumented LLM call.

Why a class instead of a plain dict:
- The full set of fields is defined ONCE here, instead of being
  copy-pasted as a dict literal in every place a result gets built
  (previously: once for success, once for errors — easy to let them
  drift apart).
- Serialization (to_dict / to_json / save) lives with the data itself,
  instead of ad-hoc json.dump() calls scattered in whichever script
  happens to use the result.
- Attribute access (result.total_tokens) instead of dict access
  (result["total_tokens"]) catches typos at development time in most
  editors/IDEs, instead of failing silently or crashing at runtime.
"""

from dataclasses import dataclass, asdict
from typing import Optional
import json


@dataclass
class LLMCallResult:
    provider: str
    trace_id: str
    timestamp: str
    model: str
    prompt_preview: str
    response_preview: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    latency_ms: Optional[float] = None
    tokens_per_second: Optional[float] = None
    cost_usd: Optional[float] = None
    quality_score: Optional[int] = None
    quality_reasoning: Optional[str] = None
    cost_efficiency: Optional[float] = None
    request_id: Optional[str] = None
    attempts_used: Optional[int] = None
    error_type: Optional[str] = None
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, path: str) -> None:
        """Writes this result to disk as JSON. Owns its own file format,
        so callers never need to know it's JSON under the hood."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())