"""
Two small maths helpers.

They are ordinary functions on purpose. No classes, no database, no internet.
You give them numbers, they give you a number back. That makes them very easy
to test - you could try them out in a Python prompt right now.

Both give back None when the numbers don't make sense. None means "we can't
work this out". That is not the same as 0, which would mean "we worked it out
and the answer really was zero".
"""


def compute_throughput(output_tokens, latency_ms):
    """
    How fast the model wrote its answer, in tokens per second.

    Example: 300 tokens written in 2000 milliseconds.
             2000 ms is 2 seconds, and 300 / 2 = 150 tokens per second.

    We only count OUTPUT tokens (the answer), not input tokens (your question).
    Your question was already written before the model started, so counting it
    would make a slow model look fast just because you sent a long prompt.
    """
    # Stop if we have nothing to divide, or nothing to divide by.
    # A time of zero or less is impossible, and dividing by zero crashes.
    if not output_tokens or latency_ms is None or latency_ms <= 0:
        return None

    # The timing is in milliseconds but we want an answer per second,
    # so convert first: 2000 milliseconds -> 2.0 seconds.
    latency_seconds = latency_ms / 1000

    # round(x, 2) keeps 2 numbers after the dot, so we get 150.25
    # instead of 150.24999999999997.
    return round(output_tokens / latency_seconds, 2)


def compute_cost_efficiency(quality_score, cost_usd):
    """
    How much answer-quality you got for each dollar. Bigger is better.

    Example: the answer scored 8 out of 10 and cost $0.0004.
             8 / 0.0004 = 20000

    That number means nothing on its own. It is useful for COMPARING two
    models: if a cheap model scores 20000 here and an expensive one scores
    5000, the cheap model gives you four times more quality per dollar.
    Making that comparison possible is the whole point of this tool.
    """
    # Stop if either number is missing, or if the cost is zero.
    # 8 divided by 0 has no answer, and Python would crash trying.
    if quality_score is None or cost_usd is None or cost_usd <= 0:
        return None

    return round(quality_score / cost_usd, 4)
