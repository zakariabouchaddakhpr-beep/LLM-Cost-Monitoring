from dataclasses import dataclass, asdict
from typing import Optional
import json


@dataclass
class LLMCallResult:

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

    finish_reason: Optional[str] = None
    latency_ms: Optional[float] = None
    tokens_per_second: Optional[float] = None
    attempts_used: Optional[int] = None

    cost_usd: Optional[float] = None
    quality_score: Optional[int] = None
    quality_reasoning: Optional[str] = None
    cost_efficiency: Optional[float] = None

    request_id: Optional[str] = None
    error_type: Optional[str] = None
    error: Optional[str] = None

    span_id: Optional[str] = None
    trace_otel_id: Optional[str] = None

    tenant_id: Optional[str] = None
    use_case: Optional[str] = None
    query_type: Optional[str] = None

    task_type: Optional[str] = None
    complexity: Optional[float] = None
    tier: Optional[str] = None

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
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())
