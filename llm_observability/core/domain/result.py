"""
Domain entities — pure business objects with no external dependencies.

These are the central types that every layer (application, infrastructure)
agrees on. They know nothing about databases, HTTP, or observability.
"""

from dataclasses import dataclass, asdict
from typing import Optional
import json


@dataclass
class LLMCallResult:
    """Structured result object for a single instrumented LLM call."""

    # Core identity
    provider: str
    trace_id: str
    timestamp: str
    model: str
    prompt: str
    response: Optional[str] = None
    call_type: str = "chat"

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None

    # Lifecycle & performance
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

    # OTel correlation 
    span_id: Optional[str] = None
    trace_otel_id: Optional[str] = None

    # Cost attribution
    tenant_id: Optional[str] = None
    use_case: Optional[str] = None

    query_type: Optional[str] = None

    # Embeddings only
    vector_dimensions: Optional[int] = None
    input_count: Optional[int] = None
    embeddings: Optional[list] = None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, path: str) -> None:
        """Writes this result to disk as JSON."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())
