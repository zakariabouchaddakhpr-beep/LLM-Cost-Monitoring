import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import litellm
import pandas as pd

from llm_observability.core.domain.routing import TIERS
from llm_observability.recommender import grading, probes

INSTRUCTION = (
    "\n\nAnswer the question above. Put your final answer on the last line, "
    "in exactly this form:\nANSWER: <the answer only>"
)

MAX_TOKENS = 1024
WORKERS = 1
RETRIES = 5
BACKOFF_SECONDS = 2.0

TOKENS_PER_MINUTE = 8000
CHARS_PER_TOKEN = 4

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "recommender_output"


def pace(prompt: str) -> None:
    reserved = MAX_TOKENS + len(prompt) // CHARS_PER_TOKEN
    time.sleep(60.0 * reserved / TOKENS_PER_MINUTE)


def ask(model: str, prompt: str) -> tuple:
    last_error = None

    for attempt in range(RETRIES):
        pace(prompt)
        try:
            reply = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt + INSTRUCTION}],
                max_tokens=MAX_TOKENS,
                temperature=0,
            )
            return reply.choices[0].message.content or "", None
        except Exception as exc:
            last_error = type(exc).__name__
            time.sleep(BACKOFF_SECONDS * (attempt + 1))

    return "", last_error


def score_model(model: str, rebuild: bool = False) -> pd.DataFrame:
    safe_name = model.replace("/", "_")
    path = OUTPUT_DIR / f"scores_{safe_name}.csv"

    if path.exists() and not rebuild:
        return pd.read_csv(path)

    frame = probes.get_probes()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        answers = list(pool.map(lambda text: ask(model, text), frame["prompt"]))

    rows = []
    for (_, probe), (response, error) in zip(frame.iterrows(), answers):
        result = grading.grade(response, probe["golden_answer"], probe["family"])
        rows.append({
            "key": probe["key"],
            "family": probe["family"],
            "correct": int(result["correct"]),
            "parsed": int(result["parsed"]),
            "extracted": result["extracted"],
            "error": error,
        })

    scored = pd.DataFrame(rows)
    failed = scored["error"].notna().mean()
    if failed > 0.02:
        raise RuntimeError(
            f"{failed:.0%} of calls to {model} failed after {RETRIES} attempts "
            f"({scored['error'].dropna().value_counts().to_dict()}). "
            "Scores would be meaningless, so nothing was saved.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scored.to_csv(path, index=False)
    return scored


def is_scored(model: str) -> bool:
    return (OUTPUT_DIR / f"scores_{model.replace('/', '_')}.csv").exists()


def stand_in(model: str) -> str:
    scored = score_model(model)
    ranking = compare(scored, reference_profiles())
    return ranking.iloc[0]["reference"]


def reference_matrix() -> pd.DataFrame:
    frame = probes.get_probes()
    names = probes.model_names()
    matrix = frame.set_index("key")[names]
    return (matrix >= 0.5).astype(int)


def ability_profile(scored: pd.DataFrame) -> pd.Series:
    return scored.groupby("family")["correct"].mean()


def reference_profiles() -> pd.DataFrame:
    frame = probes.get_probes()
    matrix = reference_matrix()
    return matrix.groupby(frame.set_index("key")["family"]).mean()


def compare(scored: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    mine = ability_profile(scored)

    rows = []
    for name in profiles.columns:
        theirs = profiles[name]
        difference = theirs - mine
        rows.append({
            "reference": name,
            "distance": float((difference ** 2).sum() ** 0.5),
            "their_score": theirs.mean(),
            "worst_family": difference.abs().idxmax(),
            "worst_gap": difference.abs().max(),
        })

    return pd.DataFrame(rows).sort_values("distance").reset_index(drop=True)


def tier_names(count: int) -> list:
    if count <= 1:
        return ["weak"]
    if count == 2:
        return ["weak", "strong"]

    names = []
    for position in range(count):
        names.append(TIERS[min(position * len(TIERS) // count, len(TIERS) - 1)])
    return names


def suggest(models: list, rebuild: bool = False) -> pd.DataFrame:
    profiles = reference_profiles()

    rows = []
    for model in models:
        if not is_scored(model) and not rebuild:
            print(f"  measuring {model} on {len(probes.get_probes())} probe "
                  f"prompts (one-time, about 17 minutes)", flush=True)

        scored = score_model(model, rebuild=rebuild)
        ranking = compare(scored, profiles)
        row = {"model": model,
               "score": scored["correct"].mean(),
               "behaves_like": ranking.iloc[0]["reference"]}
        row.update(ability_profile(scored).to_dict())
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values("score").reset_index(drop=True)
    frame["suggested_tier"] = tier_names(len(frame))
    return frame


def recommend(models: list, rebuild: bool = False) -> dict:
    frame = suggest(models, rebuild=rebuild)
    families = [name for name in frame.columns
                if name not in ("model", "score", "behaves_like", "suggested_tier")]

    print()
    print("=" * 78)
    print("TIER RECOMMENDATION")
    print("=" * 78)
    header = f"  {'model':<30}{'score':>6}  " + "".join(
        f"{name[:9]:>10}" for name in families) + "   behaves like"
    print(header)
    for row in frame.itertuples():
        scores = "".join(f"{getattr(row, name):>10.2f}" for name in families)
        print(f"  {row.model:<30}{row.score:>6.2f}  {scores}   {row.behaves_like}")

    print()
    print("  Suggested pool - paste this into obs.init():")
    print()
    print("    model={")
    for row in frame.itertuples():
        print(f'        "{row.suggested_tier}": "{row.model}",')
    print("    }")
    print()

    return dict(zip(frame["suggested_tier"], frame["model"]))
