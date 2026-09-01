import pandas as pd

from llm_observability.recommender import neighbours, sprout

QUALITY_BAR = "relative"
FIXED_THRESHOLD = 0.7
RELATIVE_DELTA = 0.05

PRICES = {
    "weak": {"input": 0.05, "output": 0.08},
    "mid": {"input": 0.10, "output": 0.50},
    "strong": {"input": 0.15, "output": 0.75},
}

OPTIMAL = "OPTIMAL"
OVERPAID = "OVERPAID"
UNDERSERVED = "UNDERSERVED"
NO_VERDICT = "NO_VERDICT"


def load_reference(stand_ins: dict) -> pd.DataFrame:
    models = sorted(set(stand_ins.values()))
    fields = ["score", "num_input_tokens", "num_output_tokens"]

    columns = ["key"] + [f"{name}.{field}" for name in models for field in fields]
    frame = sprout.read_columns(columns)
    frame.columns = (["key"]
                     + [f"{name}|{field}" for name in models for field in fields]
                     + ["split"])
    return frame.set_index("key")


def tier_scores(near: pd.DataFrame, stand_ins: dict,
                reference: pd.DataFrame) -> dict:
    rows = reference.loc[reference.index.intersection(near["key"])]

    return {tier: float(rows[f"{model}|score"].mean())
            for tier, model in stand_ins.items()}


def tier_costs(near: pd.DataFrame, stand_ins: dict, reference: pd.DataFrame,
               prices: dict = None) -> dict:
    prices = prices or PRICES
    rows = reference.loc[reference.index.intersection(near["key"])]

    costs = {}
    for tier, model in stand_ins.items():
        price = prices[tier]
        spend = (rows[f"{model}|num_input_tokens"] * price["input"]
                 + rows[f"{model}|num_output_tokens"] * price["output"])
        costs[tier] = float(spend.mean()) / 1_000_000
    return costs


def clears_bar(score: float, best: float) -> bool:
    if QUALITY_BAR == "fixed":
        return score >= FIXED_THRESHOLD
    return score >= best - RELATIVE_DELTA


def choose_ideal(scores: dict, costs: dict):
    best = max(scores.values())
    passing = [tier for tier in scores if clears_bar(scores[tier], best)]
    if not passing:
        return None

    return min(passing, key=lambda tier: costs[tier])


def classify_verdict(actual: str, ideal: str, scores: dict) -> str:
    best = max(scores.values())

    if not clears_bar(scores[actual], best):
        return UNDERSERVED
    if actual == ideal:
        return OPTIMAL
    return OVERPAID


def regret(label: str, actual: str, ideal: str,
           scores: dict, costs: dict) -> float:
    if label == OVERPAID:
        return costs[actual] - costs[ideal]
    if label == UNDERSERVED:
        return max(scores.values()) - scores[actual]
    return 0.0


def judge(task_type: str, complexity: float, actual_tier: str,
          stand_ins: dict, pool: pd.DataFrame, reference: pd.DataFrame,
          prices: dict = None) -> dict:
    near = neighbours.find_neighbours(task_type, complexity, pool)

    if near.empty:
        return {"verdict": NO_VERDICT, "reason": "no similar prompt found",
                "neighbours": 0, "actual": actual_tier}

    scores = tier_scores(near, stand_ins, reference)
    costs = tier_costs(near, stand_ins, reference, prices)
    ideal = choose_ideal(scores, costs)

    if ideal is None:
        return {"verdict": NO_VERDICT, "reason": "no tier clears the bar",
                "neighbours": len(near), "actual": actual_tier}

    label = classify_verdict(actual_tier, ideal, scores)

    return {
        "verdict": label,
        "actual": actual_tier,
        "ideal": ideal,
        "regret": regret(label, actual_tier, ideal, scores, costs),
        "scores": scores,
        "costs": costs,
        "neighbours": len(near),
    }
