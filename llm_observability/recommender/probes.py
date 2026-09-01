from pathlib import Path

import pandas as pd

from llm_observability.recommender import sprout

FAMILIES = {
    "maths": ["lighteval/MATH/all/test"],
    "multiple_choice": ["TIGER-Lab/MMLU-Pro"],
    "reasoning": [
        "TAUR-Lab/MuSR",
        "TAUR-Lab/MuSR/object_placements",
        "TAUR-Lab/MuSR/team_allocation",
    ],
    "science": ["Idavidrein/gpqa/gpqa_extended"],
}

SPLIT = None

BANDS = [2, 4, 6, 8, 10, 12]
PER_BAND = 4
PER_FAMILY = PER_BAND * len(BANDS)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "recommender_output"
CACHE_FILE = OUTPUT_DIR / "probe_set.csv"


def model_names() -> list:
    return list(sprout.schema().names[6:])


def load_candidates() -> pd.DataFrame:
    models = model_names()
    plain = ["key", "dataset", "prompt", "golden_answer"]

    frame = sprout.read_columns(plain + [name + ".score" for name in models])
    frame.columns = plain + models + ["split"]

    if SPLIT is None:
        return frame
    return frame[frame["split"] == SPLIT]


def add_family(frame: pd.DataFrame) -> pd.DataFrame:
    lookup = {}
    for family, datasets in FAMILIES.items():
        for name in datasets:
            lookup[name] = family

    frame = frame[frame["dataset"].isin(lookup)].copy()
    frame["family"] = frame["dataset"].map(lookup)
    return frame


def add_variance(frame: pd.DataFrame) -> pd.DataFrame:
    models = model_names()

    frame = frame.copy()
    frame["variance"] = frame[models].var(axis=1).round(6)
    return frame


def model_strength(frame: pd.DataFrame) -> pd.Series:
    return frame[model_names()].mean()


def add_alignment(frame: pd.DataFrame) -> pd.DataFrame:
    models = model_names()
    strength = model_strength(frame)

    frame = frame.copy()
    frame["alignment"] = 0.0

    varied = frame["variance"] > 0
    frame.loc[varied, "alignment"] = frame.loc[varied, models].corrwith(
        strength, axis=1)
    return frame


def add_difficulty(frame: pd.DataFrame) -> pd.DataFrame:
    models = model_names()

    frame = frame.copy()
    frame["n_correct"] = (frame[models] >= 0.5).astype(int).sum(axis=1)
    return frame


def choose(frame: pd.DataFrame) -> pd.DataFrame:
    picked = []
    for family in FAMILIES:
        for band in BANDS:
            rows = frame[(frame["family"] == family)
                         & (frame["n_correct"] == band)]
            rows = rows.sort_values("alignment", ascending=False)
            picked.append(rows.head(PER_BAND))

    return pd.concat(picked, ignore_index=True)


def build() -> pd.DataFrame:
    frame = load_candidates()
    frame = add_family(frame)
    frame = add_variance(frame)
    frame = add_alignment(frame)
    frame = add_difficulty(frame)
    return choose(frame)


def get_probes(rebuild: bool = False) -> pd.DataFrame:
    if CACHE_FILE.exists() and not rebuild:
        return pd.read_csv(CACHE_FILE)

    frame = build()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(CACHE_FILE, index=False)
    return frame
