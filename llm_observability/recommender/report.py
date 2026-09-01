from llm_observability.recommender import sprout

CHECKABLE = [
    "TIGER-Lab/MMLU-Pro",
    "lighteval/MATH/all",
    "lighteval/MATH/all/test",
    "Idavidrein/gpqa/gpqa_extended",
    "TAUR-Lab/MuSR",
    "TAUR-Lab/MuSR/object_placements",
    "TAUR-Lab/MuSR/team_allocation",
]


def title(text):
    print()
    print("=" * 74)
    print(text)
    print("=" * 74)


def print_schema():
    columns = sprout.schema()

    title(f"ALL {len(columns.names)} COLUMNS")
    for name in columns.names:
        print(f"  {name}")

    title("THE 6 PLAIN COLUMNS")
    for name in columns.names[:6]:
        print(f"  {name:<16} {columns.field(name).type}")

    title("INSIDE EACH MODEL COLUMN")
    model_column = columns.field(columns.names[6]).type
    for i in range(model_column.num_fields):
        field = model_column.field(i)
        print(f"  {field.name:<20} {field.type}")


def print_row_counts():
    counts = sprout.row_counts()

    title("ROWS PER SPLIT")
    for split, number in counts.items():
        print(f"  {split:<12} {number:,}")
    print()
    print(f"  TOTAL        {sum(counts.values()):,}")


def print_distribution(frame):
    table = frame.pivot_table(index="dataset", columns="split",
                              aggfunc="size", fill_value=0)
    table = table[["train", "validation", "test"]]
    table["TOTAL"] = table.sum(axis=1)
    table = table.sort_values("TOTAL", ascending=False)

    usable = table[table.index.isin(CHECKABLE)]
    prose = table[~table.index.isin(CHECKABLE)]

    title("CHECKABLE - maths and multiple choice")
    print(usable.to_string())
    print(f"\n  usable rows: {usable['TOTAL'].sum():,}")

    title("FREE-FORM - answers are prose, cannot be matched")
    print(prose.to_string())
    print(f"\n  unusable rows: {prose['TOTAL'].sum():,}")


def print_levels(frame):
    filled = frame[frame["dataset_level"] != ""]

    title("dataset_level - only filled in for MATH")
    print(filled.groupby(["dataset", "dataset_level"]).size().to_string())


def main():
    print_schema()
    print_row_counts()

    frame = sprout.read_columns(["dataset", "dataset_level"])
    print_distribution(frame)
    print_levels(frame)


if __name__ == "__main__":
    main()
