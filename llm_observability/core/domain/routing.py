TIERS = ("weak", "mid", "strong")

THRESHOLDS = {
    "Summarization":   (0.14,  0.24),
    "Text Generation": (0.350, 0.550),   
    "Code Generation": (0.175, 0.290),   
    "Open QA":         (0.070, 0.330),   
    "Closed QA":       (0.040, 0.385),   
    "Extraction":      (0.16,  0.28),
    "Classification":  (0.18,  0.30),
    "Rewrite":         (0.18,  0.30),
    "Brainstorming":   (0.35,  0.50),
    "Chatbot":         (0.20,  0.32),
    "Other":           (0.18,  0.30),
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
