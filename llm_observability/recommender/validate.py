import pandas as pd

from llm_observability.recommender import grading, probes, sprout

GATE_MODELS = ["openai-gpt-4o-mini", "aws-claude-3-5-sonnet-v1"]


def grade_model(model: str) -> pd.DataFrame:
    chosen = probes.get_probes()
    family_of = dict(zip(chosen["key"], chosen["family"]))
    golden_of = dict(zip(chosen["key"], chosen["golden_answer"]))

    frame = sprout.read_columns(["key", model + ".response", model + ".score"])
    frame.columns = ["key", "response", "score", "split"]
    frame = frame[frame["key"].isin(family_of)].copy()

    frame["family"] = frame["key"].map(family_of)
    frame["golden"] = frame["key"].map(golden_of)

    graded = frame.apply(
        lambda row: grading.grade(row["response"], row["golden"], row["family"]),
        axis=1)

    frame["ours"] = [item["correct"] for item in graded]
    frame["parsed"] = [item["parsed"] for item in graded]
    frame["theirs"] = frame["score"] >= 0.5
    frame["agree"] = frame["ours"] == frame["theirs"]
    return frame


def report(models: list = None) -> None:
    for model in models or GATE_MODELS:
        frame = grade_model(model)

        print()
        print("=" * 70)
        print(f"{model}")
        print(f"  agreement    {frame['agree'].mean() * 100:.0f}%")
        print(f"  unparseable  {(~frame['parsed']).mean() * 100:.0f}%")
        print("=" * 70)
        print(frame.groupby("family").agg(
            n=("key", "count"),
            agree=("agree", "mean"),
            unparsed=("parsed", lambda column: 1 - column.mean()),
            ours=("ours", "mean"),
            theirs=("theirs", "mean"),
        ).to_string(float_format=lambda value: f"{value:.2f}"))


if __name__ == "__main__":
    report()
