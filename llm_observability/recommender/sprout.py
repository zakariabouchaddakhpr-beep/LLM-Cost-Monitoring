from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

DATA_DIR = Path(__file__).resolve().parents[2] / "sproutdb" / "data"

SPLIT_FILES = {
    "train": [
        "train-00000-of-00003.parquet",
        "train-00001-of-00003.parquet",
        "train-00002-of-00003.parquet",
    ],
    "validation": ["validation-00000-of-00001.parquet"],
    "test": ["test-00000-of-00001.parquet"],
}


def schema():
    one_file = DATA_DIR / SPLIT_FILES["validation"][0]
    return pq.read_schema(one_file)


def row_counts() -> dict:
    counts = {}
    for split, filenames in SPLIT_FILES.items():
        total = 0
        for name in filenames:
            total = total + pq.ParquetFile(DATA_DIR / name).metadata.num_rows
        counts[split] = total
    return counts


def read_columns(columns: list) -> pd.DataFrame:
    frames = []
    for split, filenames in SPLIT_FILES.items():
        for name in filenames:
            table = pq.read_table(DATA_DIR / name, columns=columns)
            frame = table.to_pandas()
            frame["split"] = split
            frames.append(frame)

    return pd.concat(frames, ignore_index=True)
