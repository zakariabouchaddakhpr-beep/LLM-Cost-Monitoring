"""
Example usage of the LLM monitoring library.

This demonstrates:
1. Calling an LLM through LLMMonitor with OTel tracing
2. Saving results to DuckDB for persistence and analytics
3. Querying aggregated cost and latency views

To switch provider/model, edit the PROVIDER_NAME and MODEL constants below.

Run with: python quickstart.py
"""

import os

from core.monitor import LLMMonitor
from core.trace_store import TraceStore
from core.logging_config import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)


def log_result(result) -> None:
    if not result.succeeded:
        logger.error("Call failed (%s): %s", result.error_type, result.error)
        return

    cost_display = f"{result.cost_usd} $" if result.cost_usd is not None else "N/A"
    tps_display = f"{result.tokens_per_second} tok/s" if result.tokens_per_second is not None else "N/A"
    quality_display = f"{result.quality_score}/10" if result.quality_score is not None else "N/A"

    logger.info("%s (%s)", result.provider, result.model)
    logger.info(
        "tokens=%d latency_ms=%d throughput=%s cost=%s quality=%s finish=%s attempts=%d",
        result.total_tokens, round(result.latency_ms), tps_display,
        cost_display, quality_display, result.finish_reason, result.attempts_used
    )


def main():
    monitor = LLMMonitor(provider_name="gemini", model="gemini/gemini-3.5-flash")
    store = TraceStore()

    prompts = [
        ("Explain the Factory design pattern in 3 sentences.", "docs", "tutorial"),
        ("Summarize the key benefits of microservices.", "docs", "summarization"),
        ("Write a Python function to reverse a string.", "engineering", "code-gen"),
    ]

    results = []
    for prompt, tenant, use_case in prompts:
        logger.info("Calling LLM for tenant=%s use_case=%s", tenant, use_case)
        result = monitor.call(prompt, tenant_id=tenant, use_case=use_case)
        results.append(result)
        log_result(result)

    # Persist all traces to DuckDB
    store.save_many(results)
    logger.info("Saved %d traces to DuckDB", len(results))

    # Show aggregated views
    logger.info("--- Cost by Tenant ---")
    for row in store.query("SELECT * FROM v_cost_by_tenant"):
        logger.info("tenant=%s calls=%s total_cost=%.6f", row[0], row[1], row[2])

    logger.info("--- Cost by Use Case ---")
    for row in store.query("SELECT * FROM v_cost_by_use_case"):
        logger.info("use_case=%s calls=%s total_cost=%.6f avg_quality=%.1f", row[0], row[1], row[2], row[4])

    logger.info("--- Latency Percentiles ---")
    for row in store.query("SELECT * FROM v_latency_percentiles"):
        logger.info("provider=%s model=%s p50=%.1fms p90=%.1fms p99=%.1fms", row[0], row[1], row[3], row[4], row[5])

    # Summary
    summary = store.get_summary()
    logger.info("Overall summary: calls=%d cost=$%.6f avg_latency=%.1fms",
                summary["total_calls"], summary["total_cost_usd"], summary["avg_latency_ms"])

    # Also save to JSON for compatibility
    os.makedirs("results", exist_ok=True)
    for i, result in enumerate(results):
        result.save(os.path.join("results", f"call_log_{i}.json"))

    store.close()


if __name__ == "__main__":
    main()
