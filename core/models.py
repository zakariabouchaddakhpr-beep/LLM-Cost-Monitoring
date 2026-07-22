"""
Structured result object for a single instrumented LLM call.
"""

from dataclasses import dataclass, asdict
from typing import Optional
import json


@dataclass
class LLMCallResult:
    # Core identity
    provider: str
    trace_id: str
    timestamp: str
    model: str
    prompt_preview: str

    # Response content
    response_preview: Optional[str] = None

    # Token usage
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    # Lifecycle
    finish_reason: Optional[str] = None
    latency_ms: Optional[float] = None
    tokens_per_second: Optional[float] = None
    attempts_used: Optional[int] = None

    # Cost & quality
    cost_usd: Optional[float] = None
    quality_score: Optional[int] = None
    quality_reasoning: Optional[str] = None
    cost_efficiency: Optional[float] = None

    # Provider correlation
    request_id: Optional[str] = None
    error_type: Optional[str] = None
    error: Optional[str] = None

    # OTel correlation (filled by the instrumentation layer)
    span_id: Optional[str] = None
    trace_otel_id: Optional[str] = None

    # Cost attribution (filled by caller for multi-tenant / multi-use-case tracking)
    tenant_id: Optional[str] = None
    use_case: Optional[str] = None

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
