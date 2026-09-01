import pandas as pd

from llm_observability.recommender import neighbours, tiering, verdict

TRACE_COLUMNS = ["timestamp", "tier", "task_type", "complexity",
                 "tenant_id", "use_case", "cost_usd"]


def load_traces(database: str = None, days: int = None) -> pd.DataFrame:
    from llm_observability.config.settings import resolve_database
    from llm_observability.core.infrastructure.persistence import TraceRepository

    repo = TraceRepository(database or resolve_database(), create_schema=False)
    try:
        where = "WHERE tier IS NOT NULL AND call_type = 'chat'"
        if days:
            where += f" AND timestamp >= now() - INTERVAL {int(days)} DAY"
        return repo.query_df(
            f"SELECT {', '.join(TRACE_COLUMNS)} FROM llm_traces {where}")
    finally:
        repo.close()


def judge_traces(traces: pd.DataFrame, stand_ins: dict,
                 pool: pd.DataFrame = None,
                 reference: pd.DataFrame = None) -> pd.DataFrame:
    if pool is None:
        pool = neighbours.classify_sample()
    if reference is None:
        reference = verdict.load_reference(stand_ins)

    seen = {}
    rows = []
    for trace in traces.itertuples():
        key = (trace.task_type, round(float(trace.complexity or 0), 3), trace.tier)
        if key not in seen:
            seen[key] = verdict.judge(trace.task_type, trace.complexity,
                                      trace.tier, stand_ins, pool, reference)
        result = seen[key]

        rows.append({
            "tier": trace.tier,
            "task_type": trace.task_type,
            "tenant_id": getattr(trace, "tenant_id", None),
            "use_case": getattr(trace, "use_case", None),
            "cost_usd": getattr(trace, "cost_usd", None),
            "verdict": result["verdict"],
            "ideal": result.get("ideal"),
            "wasted": result["regret"] if result["verdict"] == verdict.OVERPAID else 0.0,
            "quality_lost": (result["regret"]
                             if result["verdict"] == verdict.UNDERSERVED else 0.0),
        })

    return pd.DataFrame(rows)


def summarise(judged: pd.DataFrame) -> dict:
    counts = judged["verdict"].value_counts()

    return {
        "calls": len(judged),
        "optimal": int(counts.get(verdict.OPTIMAL, 0)),
        "overpaid": int(counts.get(verdict.OVERPAID, 0)),
        "underserved": int(counts.get(verdict.UNDERSERVED, 0)),
        "no_verdict": int(counts.get(verdict.NO_VERDICT, 0)),
        "wasted_usd": float(judged["wasted"].sum()),
        "quality_lost": float(judged["quality_lost"].sum()),
        "actual_spend": float(judged["cost_usd"].fillna(0).sum()),
    }


def breakdown(judged: pd.DataFrame, by: str) -> pd.DataFrame:
    decided = judged[judged["verdict"] != verdict.NO_VERDICT]
    if decided.empty or by not in decided.columns:
        return pd.DataFrame()

    table = decided.groupby(by).agg(
        calls=("verdict", "count"),
        optimal=("verdict", lambda c: (c == verdict.OPTIMAL).mean()),
        overpaid=("verdict", lambda c: (c == verdict.OVERPAID).mean()),
        underserved=("verdict", lambda c: (c == verdict.UNDERSERVED).mean()),
        wasted_usd=("wasted", "sum"),
    )
    return table.sort_values("wasted_usd", ascending=False)


def stand_ins_for(models: dict, quiet: bool = False) -> dict:
    chosen = {}
    for tier, model in models.items():
        if not quiet and not tiering.is_scored(model):
            print(f"  measuring {model} against the probe set "
                  f"(one-time, about 17 minutes)", flush=True)
        chosen[tier] = tiering.stand_in(model)
    return chosen


def run(models: dict, database: str = None, days: int = None,
        traces: pd.DataFrame = None) -> pd.DataFrame:
    stand_ins = stand_ins_for(models)

    if traces is None:
        traces = load_traces(database, days)

    if traces.empty:
        print("No routed calls found. Pass a model pool to obs.init() and make "
              "some calls first.")
        return traces

    judged = judge_traces(traces, stand_ins)
    report(judged)
    return judged


def report(judged: pd.DataFrame) -> None:
    figures = summarise(judged)
    total = figures["calls"]

    def share(number):
        return f"{number / total:.0%}" if total else "n/a"

    print("=" * 70)
    print(f"ROUTING REVIEW - {total:,} calls")
    print("=" * 70)
    print(f"  OPTIMAL       {figures['optimal']:>6,}  {share(figures['optimal']):>5}")
    print(f"  OVERPAID      {figures['overpaid']:>6,}  {share(figures['overpaid']):>5}"
          f"   ${figures['wasted_usd']:.6f} wasted")
    print(f"  UNDERSERVED   {figures['underserved']:>6,}  "
          f"{share(figures['underserved']):>5}"
          f"   {figures['quality_lost']:.2f} quality lost")
    print(f"  NO_VERDICT    {figures['no_verdict']:>6,}  "
          f"{share(figures['no_verdict']):>5}")
    print()
    print(f"  actual spend  ${figures['actual_spend']:.6f}")

    for field in ["tier", "task_type", "use_case", "tenant_id"]:
        table = breakdown(judged, field)
        if table.empty:
            continue
        print()
        print(f"-- by {field} " + "-" * (56 - len(field)))
        print(table.to_string(float_format=lambda value: f"{value:.2f}"))
