TIERS = ("weak", "mid", "strong")

THRESHOLDS = {
    "Summarization": (0.45, 0.80),
    "Text Generation": (0.40, 0.75),
    "Code Generation": (0.30, 0.65),
    "Open QA": (0.40, 0.75),
    "Closed QA": (0.50, 0.85),
    "Extraction": (0.50, 0.85),
    "Classification": (0.55, 0.90),
    "Rewrite": (0.50, 0.85),
    "Brainstorming": (0.45, 0.80),
    "Chatbot": (0.45, 0.80),
    "Other": (0.35, 0.70),
}


def choose(task, complexity):
    if task not in THRESHOLDS or complexity is None:
        return "strong"

    mid_at, strong_at = THRESHOLDS[task]

    if complexity >= strong_at:
        return "strong"
    if complexity >= mid_at:
        return "mid"
    return "weak"


def resolve(tier, available):
    position = TIERS.index(tier)
    ranked = TIERS[position:] + tuple(reversed(TIERS[:position]))

    for candidate in ranked:
        if candidate in available:
            return candidate

    raise ValueError(f"no usable tier in {available!r}")
