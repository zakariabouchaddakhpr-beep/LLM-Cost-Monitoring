"""
A worked example of using this library. Run it with:  python main.py

I
"""

from pathlib import Path

from llm_observability.core.application.monitor import LLMMonitor
from llm_observability.core.infrastructure.persistence import TraceRepository
from llm_observability.core.logging_config import configure_logging, get_logger


def log_result(result) -> None:
    """Print one call's measurements as a readable line."""
    # result.succeeded is False when the call failed. There is nothing to
    # report but the error, so say that and stop.
    if not result.succeeded:
        logger.error("Call failed (%s): %s", result.error_type, result.error)
        return

    # Any of these three can be missing - the provider didn't report it, or
    # we couldn't work out the price, or marking was switched off. Show "N/A"
    # rather than printing the word "None", which looks like a bug.
    cost_display = f"{result.cost_usd} $" if result.cost_usd is not None else "N/A"
    tps_display = (f"{result.tokens_per_second} tok/s"
                   if result.tokens_per_second is not None else "N/A")
    quality_display = (f"{result.quality_score}/10"
                       if result.quality_score is not None else "N/A")

    logger.info("%s (%s)", result.provider, result.model)
    # The %s and %d markers get replaced by the values listed after the text.
    # We let the logger do the swapping instead of building the string
    # ourselves, so that nothing is assembled if this message ends up hidden.
    logger.info(
        "tokens=%d latency_ms=%d throughput=%s cost=%s quality=%s "
        "finish=%s attempts=%d",
        result.total_tokens, round(result.latency_ms), tps_display,
        cost_display, quality_display, result.finish_reason,
        result.attempts_used,
    )


def main():
    # Switch logging on FIRST, so everything after this point is visible.
    configure_logging()
    global logger
    logger = get_logger(__name__)

    monitor = LLMMonitor(provider_name="cerebras", model="cerebras/gpt-oss-120b")
    repo = TraceRepository()

    # Each entry here is (the question, who it's for, which feature).
    # Those last two labels are what make the cost report worth reading.
    # Without them you know what you spent but not who or what spent it.
    prompts = [
        ("Explain the Factory design pattern in 3 sentences.", "docs", "tutorial"),
        ("Summarize the key benefits of microservices.", "docs", "summarization"),
        ("Write a Python function to reverse a string.", "engineering", "code-gen"),
    ]

    results = []
    # Each entry has three parts, and this unpacks them into three variables
    # in one go.
    for prompt, tenant, use_case in prompts:
        logger.info("Calling LLM for tenant=%s use_case=%s", tenant, use_case)
        result = monitor.call(prompt, tenant_id=tenant, use_case=use_case)
        results.append(result)
        log_result(result)

    # save_many sends all three in ONE trip to the database, instead of three
    # separate ones. Fewer round trips, same result.
    repo.save_many(results)
    logger.info("Saved %d traces to ClickHouse", len(results))

    # Now read the numbers back out, using the saved questions ("views")
    # defined in schema.py. Each row comes back as a tuple, and the values are
    # in the same order the view lists them - so row[0] is the first column,
    # row[1] the second, and so on.
    logger.info("--- Cost by Tenant ---")
    for row in repo.query("SELECT * FROM v_cost_by_tenant"):
        logger.info("tenant=%s calls=%s total_cost=%.6f", row[0], row[1], row[2])

    logger.info("--- Cost by Use Case ---")
    for row in repo.query("SELECT * FROM v_cost_by_use_case"):
        logger.info("use_case=%s calls=%s total_cost=%.6f avg_quality=%.1f",
                    row[0], row[1], row[2], row[4])

    logger.info("--- Latency Percentiles ---")
    # p50 is the middle: half the calls were faster than this. p99 is the slow
    # tail: only 1 call in 100 was worse than this.
    for row in repo.query("SELECT * FROM v_latency_percentiles"):
        logger.info("provider=%s model=%s p50=%.1fms p90=%.1fms p99=%.1fms",
                    row[0], row[1], row[3], row[4], row[5])

    summary = repo.get_summary()
    logger.info("Overall summary: calls=%d cost=$%.6f avg_latency=%.1fms",
                summary["total_calls"], summary["total_cost_usd"],
                summary["avg_latency_ms"])

    # Also write each call to its own JSON file. Handy for looking closely at
    # one single call without opening the database.
    log_dir = Path("results")
    # parents=True creates any missing folders above it too.
    # exist_ok=True stops it complaining if the folder is already there.
    log_dir.mkdir(parents=True, exist_ok=True)
    # enumerate() gives you a counter alongside each item, so we get 0, 1, 2
    # for the file names.
    for i, result in enumerate(results):
        result.save(str(log_dir / f"call_log_{i}.json"))

    repo.close()
    logger.info("Done.")


# This line means main() runs when you type `python main.py`, but NOT when
# some other file imports this one. Without it, importing anything from here
# would immediately start making paid API calls.
if __name__ == "__main__":
    main()
