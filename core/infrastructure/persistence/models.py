"""
SQLAlchemy ORM models for LLM trace persistence.

Uses declarative base (SQLAlchemy 2.0 style) mapped to DuckDB.
"""

from datetime import datetime

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Integer, Float, DateTime, Text


class Base(DeclarativeBase):
    pass


class LLMTrace(Base):
    """ORM model for a single instrumented LLM call."""

    __tablename__ = "llm_traces"

    trace_id:        Mapped[str] = mapped_column(String, primary_key=True)
    provider:        Mapped[str] = mapped_column(String, nullable=False)
    timestamp:       Mapped[datetime] = mapped_column(DateTime, nullable=False)
    model:           Mapped[str] = mapped_column(String, nullable=False)

    prompt_preview:  Mapped[str | None] = mapped_column(Text, nullable=True)
    response_preview: Mapped[str | None] = mapped_column(Text, nullable=True)

    input_tokens:    Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens:   Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens:    Mapped[int | None] = mapped_column(Integer, nullable=True)

    finish_reason:   Mapped[str | None] = mapped_column(String, nullable=True)
    latency_ms:      Mapped[float | None] = mapped_column(Float, nullable=True)
    tokens_per_second: Mapped[float | None] = mapped_column(Float, nullable=True)

    cost_usd:        Mapped[float | None] = mapped_column(Float, nullable=True)
    quality_score:   Mapped[int | None] = mapped_column(Integer, nullable=True)
    quality_reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    cost_efficiency: Mapped[float | None] = mapped_column(Float, nullable=True)

    request_id:      Mapped[str | None] = mapped_column(String, nullable=True)
    attempts_used:   Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_type:      Mapped[str | None] = mapped_column(String, nullable=True)
    error:           Mapped[str | None] = mapped_column(Text, nullable=True)

    span_id:         Mapped[str | None] = mapped_column(String, nullable=True)
    trace_otel_id:   Mapped[str | None] = mapped_column(String, nullable=True)

    tenant_id:       Mapped[str | None] = mapped_column(String, nullable=True)
    use_case:        Mapped[str | None] = mapped_column(String, nullable=True)
