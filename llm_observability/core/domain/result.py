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
    # Stored whole. This was prompt_preview, truncated at 100 characters; a
    # field named "preview" holding complete text is a lie someone eventually
    # trusts.
    prompt: str

    # Response content. Never truncated - it was only ever named "preview".
    response: Optional[str] = None

    # What kind of call this was. NOT Optional: it always has a value, which is
    # why the column is non-nullable despite carrying a default.
    call_type: str = "chat"

    # Token usage
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    # Billed at a fraction of the normal rate. Absent on providers that do not
    # report it, so None means "not reported" and 0 means "reported as zero" -
    # a distinction that matters when judging whether caching is working.
    cached_tokens: Optional[int] = None
    # Billed but invisible in the input/output split on thinking models.
    reasoning_tokens: Optional[int] = None

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

    # Classification, filled by the judge when classify_query is on. The third
    # attribution dimension the proposal asks for, alongside tenant and use case.
    query_type: Optional[str] = None

    # Embeddings only
    vector_dimensions: Optional[int] = None
    input_count: Optional[int] = None

    # The vectors themselves. Deliberately NOT persisted - far too large, and
    # the caller needs them in hand rather than in a database. schema.COLUMNS
    # omits it and _to_row builds rows by name, so it is ignored automatically.
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
