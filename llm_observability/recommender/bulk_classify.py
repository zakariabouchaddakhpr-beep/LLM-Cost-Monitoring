import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from llm_observability.recommender import sprout

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "threshold_study"

SAMPLE_SIZE = 3000
SEED = 20260902
CHECKPOINT_EVERY = 100

FIELDS = ["key", "dataset", "task_type", "complexity", "confidence",
          "prompt_chars"]


def paths(size: int):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return (OUTPUT_DIR / f"classified_{size}.csv",
            OUTPUT_DIR / f"sample_{size}.csv",
            OUTPUT_DIR / f"classify_{size}.log")


def build_sample(size: int = SAMPLE_SIZE, seed: int = SEED,
                 keep_keys=None) -> pd.DataFrame:
    frame = sprout.read_columns(["key", "dataset", "prompt"])

    if not keep_keys:
        return frame.sample(n=size, random_state=seed).reset_index(drop=True)

    marked = frame["key"].astype(str).isin(set(keep_keys))
    held = frame[marked]
    remaining = frame[~marked]
    topup = remaining.sample(n=max(0, size - len(held)), random_state=seed)

    return pd.concat([held, topup], ignore_index=True)


def load_done(checkpoint: Path) -> set:
    if not checkpoint.exists():
        return set()
    try:
        return set(pd.read_csv(checkpoint)["key"].astype(str))
    except Exception:
        return set()


def flush(rows: list, checkpoint: Path) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows, columns=FIELDS)
    header = not checkpoint.exists()
    frame.to_csv(checkpoint, mode="a", header=header, index=False)


def run(size: int = SAMPLE_SIZE, seed: int = SEED,
        extend_from: str = None) -> None:
    checkpoint, sample_path, log_path = paths(size)
    log = open(log_path, "a", encoding="utf-8", buffering=1)

    def say(message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        log.write(f"[{stamp}] {message}\n")
        print(f"[{stamp}] {message}", flush=True)

    say(f"run start size={size} seed={seed}")

    prior = None
    keep_keys = set()
    if extend_from and Path(extend_from).exists():
        prior = pd.read_csv(extend_from)
        keep_keys = set(prior["key"].astype(str))
        say(f"extending from {extend_from}: {len(keep_keys)} rows carried over")

    sample = build_sample(size, seed, keep_keys)
    sample[["key", "dataset"]].to_csv(sample_path, index=False)
    say(f"sample drawn: {len(sample)} rows, {sample['dataset'].nunique()} datasets")

    if prior is not None and not checkpoint.exists():
        inside = prior[prior["key"].astype(str).isin(
            set(sample["key"].astype(str)))]
        inside.to_csv(checkpoint, index=False)
        say(f"checkpoint seeded with {len(inside)} previously classified rows")

    done = load_done(checkpoint)
    if done:
        say(f"resuming: {len(done)} already classified")
    todo = sample[~sample["key"].astype(str).isin(done)]
    say(f"to classify: {len(todo)}")

    if todo.empty:
        say("nothing to do")
        return

    from llm_observability.core.infrastructure.classifier import PromptClassifier

    loading = time.perf_counter()
    classifier = PromptClassifier()
    classifier._ensure_loaded()
    say(f"model loaded in {time.perf_counter() - loading:.1f}s")

    rows = []
    started = time.perf_counter()
    total = len(todo)

    for position, record in enumerate(todo.itertuples(), start=1):
        profile = classifier.classify(record.prompt)
        rows.append([record.key, record.dataset, profile.task,
                     profile.complexity, profile.confidence,
                     len(record.prompt)])

        if position % CHECKPOINT_EVERY == 0 or position == total:
            flush(rows, checkpoint)
            rows = []
            elapsed = time.perf_counter() - started
            rate = elapsed / position
            left = timedelta(seconds=int(rate * (total - position)))
            say(f"{position}/{total}  {rate:.2f}s/prompt  eta {left}")

    say("run complete")
    log.close()


if __name__ == "__main__":
    argument = int(sys.argv[1]) if len(sys.argv) > 1 else SAMPLE_SIZE
    source = sys.argv[2] if len(sys.argv) > 2 else None
    run(argument, extend_from=source)
