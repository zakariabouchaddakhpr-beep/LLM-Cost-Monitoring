"""
Example usage of the LLM monitoring library.

This is NOT a benchmark comparing providers against each other — it
demonstrates how a developer would use LLMMonitor as a middleman
around whichever single LLM their application actually uses.

To switch provider/model, just change PROVIDER_NAME and MODEL below.
Nothing else in the library needs to change.

Run with: python quickstart.py
"""

import os

from core.monitor import LLMMonitor
from core.logging_config import configure_logging, get_logger

configure_logging()  # set up logging once, at the entry point
logger = get_logger(__name__)

# Change these two lines to use a different provider/model.
PROVIDER_NAME = "gemini"
MODEL = "gemini/gemini-3.5-flash"


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
    monitor = LLMMonitor(provider_name=PROVIDER_NAME, model=MODEL)

    prompt = "Explain the Factory design pattern in 3 sentences."
    result = monitor.call(prompt)

    os.makedirs("results", exist_ok=True)
    result.save(os.path.join("results", "call_log.json"))

    log_result(result)


if __name__ == "__main__":
    main()