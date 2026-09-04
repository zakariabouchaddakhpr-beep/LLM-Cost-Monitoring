from pathlib import Path

import numpy as np
import pandas as pd

from llm_observability.recommender import sprout

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "threshold_study"

RELATIVE_DELTA = 0.05

GRID_STEP = 0.005
MIN_SUPPORT = 200
BOOTSTRAP_DRAWS = 200

PRICES_PER_MTOK = {
    "wxai-llama-3-2-1b-instruct": (0.04, 0.04),
    "wxai-llama-3-2-3b-instruct": (0.06, 0.06),
    "wxai-granite-3-2b-instruct-8k-max-tokens": (0.10, 0.10),
    "wxai-granite-3-8b-instruct-8k-max-tokens": (0.20, 0.20),
    "wxai-llama-3-1-8b-instruct": (0.10, 0.10),
    "wxai-mixtral-8x7b-instruct-v01": (0.24, 0.24),
    "aws-titan-text-premier-v1": (0.50, 1.50),
    "wxai-llama-3-3-70b-instruct": (0.60, 0.60),
    "wxai-llama-3-1-70b-instruct": (0.60, 0.60),
    "wxai-llama-3-405b-instruct": (3.50, 3.50),
    "openai-gpt-4o-mini": (0.15, 0.60),
    "openai-gpt-4o": (2.50, 10.00),
    "aws-claude-3-5-sonnet-v1": (3.00, 15.00),
}

TRIPLES = {
    "primary": {
        "weak": "wxai-llama-3-2-1b-instruct",
        "mid": "wxai-llama-3-1-8b-instruct",
        "strong": "openai-gpt-4o",
    },
    "granite_claude": {
        "weak": "wxai-granite-3-2b-instruct-8k-max-tokens",
        "mid": "wxai-granite-3-8b-instruct-8k-max-tokens",
        "strong": "aws-claude-3-5-sonnet-v1",
    },
    "llama_ladder": {
        "weak": "wxai-llama-3-2-3b-instruct",
        "mid": "wxai-llama-3-3-70b-instruct",
        "strong": "openai-gpt-4o",
    },
}

TIER_ORDER = ("weak", "mid", "strong")


def load_panel(classified: pd.DataFrame, triple: dict,
               prices: dict = None) -> pd.DataFrame:
    prices = prices or PRICES_PER_MTOK
    models = [triple[tier] for tier in TIER_ORDER]

    fields = ["score", "num_input_tokens", "num_output_tokens"]
    columns = ["key"] + [f"{m}.{f}" for m in models for f in fields]

    raw = sprout.read_columns(columns)
    raw.columns = (["key"] + [f"{m}|{f}" for m in models for f in fields]
                   + ["split"])

    panel = classified.merge(raw.drop(columns=["split"]), on="key", how="inner")

    for tier in TIER_ORDER:
        model = triple[tier]
        price_in, price_out = prices[model]
        panel[f"score_{tier}"] = panel[f"{model}|score"]
        panel[f"cost_{tier}"] = (
            panel[f"{model}|num_input_tokens"] * price_in
            + panel[f"{model}|num_output_tokens"] * price_out
        ) / 1_000_000

    best = panel[[f"score_{t}" for t in TIER_ORDER]].max(axis=1)
    for tier in TIER_ORDER:
        panel[f"ok_{tier}"] = panel[f"score_{tier}"] >= best - RELATIVE_DELTA

    panel["best_score"] = best
    return panel


def _prefix(values: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(values)])


def _grid(complexity: np.ndarray, step: float = GRID_STEP) -> np.ndarray:
    low = np.floor(complexity.min() / step) * step
    high = np.ceil(complexity.max() / step) * step
    return np.round(np.arange(low, high + step, step), 6)


def sweep_frontier(panel: pd.DataFrame, step: float = GRID_STEP) -> pd.DataFrame:
    frame = panel.sort_values("complexity")
    complexity = frame["complexity"].to_numpy()
    n = len(frame)

    cost = {t: _prefix(frame[f"cost_{t}"].to_numpy()) for t in TIER_ORDER}
    bad = {t: _prefix((~frame[f"ok_{t}"]).to_numpy().astype(float))
           for t in TIER_ORDER}

    grid = _grid(complexity, step)
    cuts = np.searchsorted(complexity, grid, side="left")
    size = len(grid)

    idx_i = np.repeat(np.arange(size), size)
    idx_j = np.tile(np.arange(size), size)
    keep = idx_i <= idx_j
    idx_i, idx_j = idx_i[keep], idx_j[keep]

    ci, cj = cuts[idx_i], cuts[idx_j]

    spend = (cost["weak"][ci]
             + cost["mid"][cj] - cost["mid"][ci]
             + cost["strong"][n] - cost["strong"][cj])
    missed = (bad["weak"][ci]
              + bad["mid"][cj] - bad["mid"][ci]
              + bad["strong"][n] - bad["strong"][cj])

    return pd.DataFrame({
        "mid_at": grid[idx_i],
        "strong_at": grid[idx_j],
        "spend": spend,
        "underserved": missed / n,
    })


def best_under_ceiling(frontier: pd.DataFrame, ceiling: float):
    feasible = frontier[frontier["underserved"] <= ceiling]
    if feasible.empty:
        return None

    cheapest = feasible["spend"].min()
    tied = feasible[np.isclose(feasible["spend"], cheapest, rtol=1e-12)]

    return {
        "mid_at": float(tied["mid_at"].median()),
        "strong_at": float(tied["strong_at"].median()),
        "spend": float(cheapest),
        "underserved": float(tied["underserved"].min()),
        "tied_options": int(len(tied)),
    }


def pareto(frontier: pd.DataFrame, ceilings) -> pd.DataFrame:
    rows = []
    for ceiling in ceilings:
        found = best_under_ceiling(frontier, ceiling)
        if found is None:
            continue
        found["ceiling"] = ceiling
        rows.append(found)
    return pd.DataFrame(rows)


def always_tier_baseline(panel: pd.DataFrame) -> pd.DataFrame:
    n = len(panel)
    rows = []
    for tier in TIER_ORDER:
        rows.append({
            "policy": f"always {tier}",
            "spend": float(panel[f"cost_{tier}"].sum()),
            "underserved": float((~panel[f"ok_{tier}"]).sum()) / n,
        })
    return pd.DataFrame(rows)


def oracle(panel: pd.DataFrame) -> dict:
    n = len(panel)
    costs = panel[[f"cost_{t}" for t in TIER_ORDER]].to_numpy()
    oks = panel[[f"ok_{t}" for t in TIER_ORDER]].to_numpy()

    masked = np.where(oks, costs, np.inf)
    chosen = masked.min(axis=1)
    unreachable = ~np.isfinite(chosen)
    chosen = np.where(unreachable, costs.min(axis=1), chosen)

    return {"policy": "oracle", "spend": float(chosen.sum()),
            "underserved": float(unreachable.sum()) / n}


def evaluate(panel: pd.DataFrame, mid_at: float, strong_at: float) -> dict:
    complexity = panel["complexity"].to_numpy()
    tier = np.where(complexity >= strong_at, 2,
                    np.where(complexity >= mid_at, 1, 0))

    costs = panel[[f"cost_{t}" for t in TIER_ORDER]].to_numpy()
    oks = panel[[f"ok_{t}" for t in TIER_ORDER]].to_numpy()
    rows = np.arange(len(panel))

    return {
        "spend": float(costs[rows, tier].sum()),
        "underserved": float((~oks[rows, tier]).sum()) / len(panel),
        "share_weak": float((tier == 0).mean()),
        "share_mid": float((tier == 1).mean()),
        "share_strong": float((tier == 2).mean()),
    }


def bootstrap(panel: pd.DataFrame, ceiling: float,
              draws: int = BOOTSTRAP_DRAWS, seed: int = 0,
              step: float = GRID_STEP) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(panel)

    rows = []
    for _ in range(draws):
        take = rng.integers(0, n, size=n)
        drawn = panel.iloc[take]
        found = best_under_ceiling(sweep_frontier(drawn, step), ceiling)
        if found is not None:
            rows.append({"mid_at": found["mid_at"],
                         "strong_at": found["strong_at"]})

    return pd.DataFrame(rows)


def ideal_tier(panel: pd.DataFrame) -> pd.Series:
    costs = panel[[f"cost_{t}" for t in TIER_ORDER]].to_numpy()
    oks = panel[[f"ok_{t}" for t in TIER_ORDER]].to_numpy()

    masked = np.where(oks, costs, np.inf)
    pick = masked.argmin(axis=1)
    none_pass = ~oks.any(axis=1)
    pick = np.where(none_pass, costs.argmin(axis=1), pick)

    return pd.Series([TIER_ORDER[i] for i in pick], index=panel.index)


def assign_tier(complexity: pd.Series, mid_at: pd.Series,
                strong_at: pd.Series) -> pd.Series:
    out = np.where(complexity >= strong_at, "strong",
                   np.where(complexity >= mid_at, "mid", "weak"))
    return pd.Series(out, index=complexity.index)


def verdict_for(panel: pd.DataFrame, chosen: pd.Series) -> pd.Series:
    rows = np.arange(len(panel))
    index = {t: i for i, t in enumerate(TIER_ORDER)}
    taken = chosen.map(index).to_numpy()

    oks = panel[[f"ok_{t}" for t in TIER_ORDER]].to_numpy()
    costs = panel[[f"cost_{t}" for t in TIER_ORDER]].to_numpy()
    best = panel["_ideal_index"].to_numpy()

    passed = oks[rows, taken]
    same = taken == best
    dearer = costs[rows, taken] > costs[rows, best]

    labels = np.where(~passed, "UNDERSERVED",
                      np.where(same, "OPTIMAL",
                               np.where(dearer, "OVERPAID", "OPTIMAL")))
    return pd.Series(labels, index=panel.index)


def export_panel(classified: pd.DataFrame, triple: dict,
                 fitted: dict = None, prompts: pd.DataFrame = None,
                 prices: dict = None) -> pd.DataFrame:
    from llm_observability.core.domain import routing

    panel = load_panel(classified, triple, prices)

    panel["ideal_tier"] = ideal_tier(panel)
    index = {t: i for i, t in enumerate(TIER_ORDER)}
    panel["_ideal_index"] = panel["ideal_tier"].map(index)
    panel["ideal_cost"] = [panel.iloc[i][f"cost_{panel.iloc[i]['ideal_tier']}"]
                           for i in range(len(panel))]

    shipped = panel["task_type"].map(
        lambda t: routing.THRESHOLDS.get(t, (np.nan, np.nan)))
    panel["shipped_mid_at"] = [pair[0] for pair in shipped]
    panel["shipped_strong_at"] = [pair[1] for pair in shipped]
    panel["shipped_tier"] = np.where(
        panel["shipped_mid_at"].isna(), "strong",
        assign_tier(panel["complexity"], panel["shipped_mid_at"].fillna(0),
                    panel["shipped_strong_at"].fillna(0)))

    fitted = fitted or {}
    panel["fitted_mid_at"] = panel["task_type"].map(
        lambda t: fitted.get(t, (np.nan, np.nan))[0])
    panel["fitted_strong_at"] = panel["task_type"].map(
        lambda t: fitted.get(t, (np.nan, np.nan))[1])
    panel["fitted_tier"] = np.where(
        panel["fitted_mid_at"].isna(), panel["shipped_tier"],
        assign_tier(panel["complexity"], panel["fitted_mid_at"].fillna(0),
                    panel["fitted_strong_at"].fillna(0)))

    for label in ("shipped", "fitted"):
        chosen = panel[f"{label}_tier"]
        panel[f"verdict_{label}"] = verdict_for(panel, chosen)
        taken_cost = [panel.iloc[i][f"cost_{chosen.iloc[i]}"]
                      for i in range(len(panel))]
        taken_score = [panel.iloc[i][f"score_{chosen.iloc[i]}"]
                       for i in range(len(panel))]
        panel[f"wasted_usd_{label}"] = np.where(
            panel[f"verdict_{label}"] == "OVERPAID",
            np.array(taken_cost) - panel["ideal_cost"], 0.0).round(10)
        panel[f"quality_lost_{label}"] = np.where(
            panel[f"verdict_{label}"] == "UNDERSERVED",
            panel["best_score"] - np.array(taken_score), 0.0).round(6)

    if prompts is not None:
        excerpt = prompts.set_index("key")["prompt"].str.slice(0, 160)
        panel["prompt_excerpt"] = panel["key"].map(excerpt)

    panel = panel.drop(columns=["_ideal_index"])
    keep = [c for c in panel.columns
            if not c.startswith(("wxai-", "openai-", "aws-"))]
    return panel[keep]


BAND_WIDTH = 0.05


def band_table(panel: pd.DataFrame, fitted: dict = None,
               width: float = BAND_WIDTH) -> pd.DataFrame:
    from llm_observability.core.domain import routing

    fitted = fitted or {}
    lower = np.floor(panel["complexity"] / width) * width
    work = panel.assign(band_lo=lower.round(4))

    rows = []
    for task in routing.THRESHOLDS:
        subset = work[work["task_type"] == task]
        shipped_mid, shipped_strong = routing.THRESHOLDS[task]
        fit_mid, fit_strong = fitted.get(task, (np.nan, np.nan))

        if subset.empty:
            rows.append({
                "task_type": task, "band_lo": None, "band_hi": None,
                "n": 0, "shipped_mid_at": shipped_mid,
                "shipped_strong_at": shipped_strong,
                "fitted_mid_at": fit_mid, "fitted_strong_at": fit_strong,
                "note": "no prompts of this type in the sample",
            })
            continue

        for band_lo, group in subset.groupby("band_lo"):
            middle = band_lo + width / 2
            passes = {t: float(group[f"ok_{t}"].mean()) for t in TIER_ORDER}
            best = max(passes.values())
            costs = {t: float(group[f"cost_{t}"].mean()) for t in TIER_ORDER}

            def cheapest_within(margin):
                clears = [t for t in TIER_ORDER
                          if passes[t] >= best - margin]
                return min(clears, key=lambda t: costs[t])

            cheapest = cheapest_within(RELATIVE_DELTA)
            cheapest_loose = cheapest_within(0.20)

            shipped = ("strong" if middle >= shipped_strong
                       else "mid" if middle >= shipped_mid else "weak")
            if np.isnan(fit_mid):
                fit_choice = ""
            else:
                fit_choice = ("strong" if middle >= fit_strong
                              else "mid" if middle >= fit_mid else "weak")

            rows.append({
                "task_type": task,
                "band_lo": round(band_lo, 3),
                "band_hi": round(band_lo + width, 3),
                "n": len(group),
                "weak_pass_rate": round(passes["weak"], 3),
                "mid_pass_rate": round(passes["mid"], 3),
                "strong_pass_rate": round(passes["strong"], 3),
                "avg_cost_weak": round(costs["weak"], 8),
                "avg_cost_mid": round(costs["mid"], 8),
                "avg_cost_strong": round(costs["strong"], 8),
                "cheapest_tier_strict_bar": cheapest,
                "cheapest_tier_loose_bar": cheapest_loose,
                "weak_gap_to_best": round(best - passes["weak"], 3),
                "mid_gap_to_best": round(best - passes["mid"], 3),
                "shipped_mid_at": shipped_mid,
                "shipped_strong_at": shipped_strong,
                "shipped_sends_to": shipped,
                "fitted_mid_at": fit_mid,
                "fitted_strong_at": fit_strong,
                "fitted_sends_to": fit_choice,
                "note": "",
            })

    return pd.DataFrame(rows)


def fit_by_task(panel: pd.DataFrame, ceiling: float,
                min_support: int = MIN_SUPPORT,
                step: float = GRID_STEP) -> pd.DataFrame:
    rows = []
    for task, group in panel.groupby("task_type"):
        entry = {"task_type": task, "n": len(group)}

        if len(group) < min_support:
            entry["status"] = "insufficient_support"
            rows.append(entry)
            continue

        found = best_under_ceiling(sweep_frontier(group, step), ceiling)
        if found is None:
            entry["status"] = "no_feasible_pair"
            rows.append(entry)
            continue

        entry.update(found)
        entry.update(evaluate(group, found["mid_at"], found["strong_at"]))
        entry["status"] = "fitted"
        rows.append(entry)

    return pd.DataFrame(rows).sort_values("n", ascending=False)
