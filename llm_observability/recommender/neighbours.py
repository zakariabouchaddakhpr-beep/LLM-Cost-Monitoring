from pathlib import Path

import pandas as pd

from llm_observability.recommender import sprout

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "recommender_output"
CACHE_FILE = OUTPUT_DIR / "sprout_classified.parquet"

SAMPLE_SIZE = 1000
NEIGHBOURS = 5
COMPLEXITY_FLOOR = 0.15


def build_sample(size: int = SAMPLE_SIZE) -> pd.DataFrame:
    frame = sprout.read_columns(["key", "dataset", "prompt"])
    per_dataset = max(1, size // frame["dataset"].nunique())

    parts = []
    for _, group in frame.groupby("dataset"):
        parts.append(group.head(per_dataset))

    return pd.concat(parts, ignore_index=True)


def classify_sample(rebuild: bool = False, report_every: int = 100) -> pd.DataFrame:
    if CACHE_FILE.exists() and not rebuild:
        return pd.read_parquet(CACHE_FILE)

    from llm_observability.core.infrastructure.classifier import PromptClassifier

    sample = build_sample()
    classifier = PromptClassifier()

    rows = []
    for position, row in enumerate(sample.itertuples(), start=1):
        profile = classifier.classify(row.prompt)
        rows.append({
            "key": row.key,
            "dataset": row.dataset,
            "task_type": profile.task,
            "complexity": profile.complexity,
        })
        if position % report_every == 0:
            print(f"  classified {position} of {len(sample)}", flush=True)

    classified = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    classified.to_parquet(CACHE_FILE)
    return classified


def find_neighbours(task_type: str, complexity: float,
                    pool: pd.DataFrame, k: int = NEIGHBOURS) -> pd.DataFrame:
    same_task = pool[pool["task_type"] == task_type]
    if same_task.empty:
        return same_task

    gap = (same_task["complexity"] - complexity).abs()
    nearest = same_task.assign(gap=gap).nsmallest(k, "gap")
    return nearest[nearest["gap"] <= COMPLEXITY_FLOOR]
