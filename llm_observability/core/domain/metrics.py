def compute_throughput(output_tokens, latency_ms):
    if not output_tokens or latency_ms is None or latency_ms <= 0:
        return None

    latency_seconds = latency_ms / 1000

    return round(output_tokens / latency_seconds, 2)


def compute_cost_efficiency(quality_score, cost_usd):
    if quality_score is None or cost_usd is None or cost_usd <= 0:
        return None

    return round(quality_score / cost_usd, 4)
