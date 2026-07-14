"""
Pure metric calculations. No API calls, no side effects — just math
on numbers you already have. Kept separate so they're trivial to
unit test (e.g. compute_throughput(100, 1000) == 100.0) without
needing to mock any LLM calls.
"""


def compute_throughput(output_tokens, latency_ms):
    """Metric 1: tokens generated per second (output only, not input)."""
    if not output_tokens or latency_ms is None or latency_ms <= 0:
        return None
    latency_s = latency_ms / 1000
    return round(output_tokens / latency_s, 2)


def compute_cost_efficiency(quality_score, cost_usd):
    """
    Metric 12: composite score = quality per dollar spent.
    Higher is better. Undefined if cost is missing/zero or quality is missing.
    """
    if quality_score is None or cost_usd is None or cost_usd <= 0:
        return None
    return round(quality_score / cost_usd, 4)