"""
Streamlit dashboard for the traces of whichever project launched it.

The database is located through the same configuration the library itself uses,
so running `streamlit run streamlit_app.py` from inside a project shows that
project's traces and nothing else.
"""

import html
import sys
from datetime import datetime
from pathlib import Path

# Streamlit does not put the script's own folder on sys.path, and the working
# directory belongs to the project being monitored, so add the directory that
# CONTAINS the llm_observability package. parents[2] because this file sits at
# llm_observability/dashboard/app.py. Must run before the package imports below.
_PKG_PARENT = str(Path(__file__).resolve().parents[2])
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

import altair as alt
import pandas as pd
import streamlit as st

from llm_observability.config.settings import (
    PROJECT_NAME,
    ConfigError,
    get_clickhouse_config,
    resolve_database,
)
from llm_observability.core.infrastructure.persistence import TraceRepository
from llm_observability.core.infrastructure.persistence.database import (
    ClickHouseUnavailable,
    database_exists,
)

# Two selected palettes, not one flipped. Each was validated against its own
# surface for lightness band, chroma, colour-vision separation and contrast.
# In light mode the third slot sits under 3:1, so charts using it also ship a
# table view.
LIGHT = {
    "surface": "#fcfcfb", "ink": "#0b0b0b", "ink_soft": "#52514e",
    "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
    "s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a",
    "border": "rgba(11,11,11,.10)", "chip": "rgba(42,120,214,.10)",
    "chip_ink": "#2a78d6", "good": "#006300", "bad": "#d03b3b",
}
DARK = {
    "surface": "#1a1a19", "ink": "#ffffff", "ink_soft": "#c3c2b7",
    "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
    "s1": "#3987e5", "s2": "#d95926", "s3": "#199e70",
    "border": "rgba(255,255,255,.12)", "chip": "rgba(57,135,229,.18)",
    "chip_ink": "#86b6ef", "good": "#0ca30c", "bad": "#e66767",
}
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"


def _tokens() -> tuple[dict, str]:
    """Match the chart palette to whichever theme Streamlit is rendering."""
    mode = "light"
    try:
        theme = getattr(st.context, "theme", None)
        if theme is not None and getattr(theme, "type", None) == "dark":
            mode = "dark"
    except Exception:
        pass
    return (DARK if mode == "dark" else LIGHT), mode


def _css(T: dict) -> str:
    return f"""
<style>
  .block-container, [data-testid="stMainBlockContainer"] {{
      padding-top: 2.1rem; padding-bottom: 3rem; max-width: 1560px;
  }}
  .hdr {{ display:flex; align-items:baseline; gap:.7rem; flex-wrap:wrap;
         margin-bottom:.15rem; }}
  .hdr h1 {{ font-family:{FONT}; font-size:1.55rem; font-weight:600;
             color:{T['ink']}; margin:0; letter-spacing:-.01em; }}
  .chip {{ font-family:{FONT}; font-size:.72rem; font-weight:600;
           color:{T['chip_ink']}; background:{T['chip']};
           border-radius:999px; padding:.16rem .6rem; letter-spacing:.02em; }}
  .lede {{ font-family:{FONT}; font-size:.85rem; color:{T['ink_soft']};
           margin:0 0 1.1rem 0; }}
  .tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(148px,1fr));
            gap:.55rem; margin:.2rem 0 .5rem 0; }}
  .tile {{ background:{T['surface']}; border:1px solid {T['border']};
           border-radius:10px; padding:.7rem .85rem; }}
  .tile .lbl {{ font-family:{FONT}; font-size:.68rem; font-weight:600;
                letter-spacing:.05em; text-transform:uppercase;
                color:{T['muted']}; margin-bottom:.28rem; }}
  .tile .val {{ font-family:{FONT}; font-size:1.45rem; font-weight:600;
                color:{T['ink']}; line-height:1.12; }}
  .tile .sub {{ font-family:{FONT}; font-size:.71rem; color:{T['muted']};
                margin:.2rem 0 0 0; }}
  .tile .sub.good {{ color:{T['good']}; }}
  .tile .sub.bad {{ color:{T['bad']}; }}
  .sec {{ font-family:{FONT}; font-size:.95rem; font-weight:600;
          color:{T['ink']}; margin:.35rem 0 .1rem 0; }}
  .sec-sub {{ font-family:{FONT}; font-size:.78rem; color:{T['muted']};
              margin:0 0 .5rem 0; }}
</style>
"""


@st.cache_data(ttl=15, show_spinner="Loading traces from ClickHouse...")
def load_data(database: str, start=None, end=None, providers=None,
              tenants=None, use_cases=None, query_types=None):
    """
    Traces for the current filter, plus the three analytics views.

    Filters are pushed into SQL rather than applied in pandas afterwards.
    Against a local file an unfiltered SELECT * was merely wasteful; against a
    server it is an unbounded transfer on every cache expiry. Everything below
    this function — the aggregation and every chart — is unchanged.
    """
    conditions, params = [], {}
    if start is not None and end is not None:
        conditions.append("timestamp >= %(start)s AND timestamp < %(end)s")
        params["start"], params["end"] = start, end
    for field, values in (("provider", providers), ("tenant_id", tenants),
                          ("use_case", use_cases), ("query_type", query_types)):
        if values:
            conditions.append(f"{field} IN %({field}_list)s")
            params[f"{field}_list"] = list(values)
    where = "WHERE " + " AND ".join(conditions) if conditions else ""

    repo = TraceRepository(database, create_schema=False)
    try:
        traces = repo.query_df(f"SELECT * FROM llm_traces {where}", params)
        cost_by_tenant = repo.query_df("SELECT * FROM v_cost_by_tenant")
        cost_by_use_case = repo.query_df("SELECT * FROM v_cost_by_use_case")
        latency_percentiles = repo.query_df("SELECT * FROM v_latency_percentiles")
    finally:
        repo.close()

    traces["timestamp"] = pd.to_datetime(traces["timestamp"], errors="coerce")
    return traces, cost_by_tenant, cost_by_use_case, latency_percentiles


@st.cache_data(ttl=15)
def get_filter_options(database: str):
    """Distinct values and the date bounds, read cheaply so the filter widgets
    can be built before the traces are fetched."""
    repo = TraceRepository(database, create_schema=False)
    try:
        providers = [r[0] for r in repo.query(
            "SELECT DISTINCT provider FROM llm_traces ORDER BY provider")]
        tenants = [r[0] for r in repo.query(
            "SELECT DISTINCT tenant_id FROM llm_traces "
            "WHERE tenant_id IS NOT NULL ORDER BY tenant_id")]
        use_cases = [r[0] for r in repo.query(
            "SELECT DISTINCT use_case FROM llm_traces "
            "WHERE use_case IS NOT NULL ORDER BY use_case")]
        query_types = [r[0] for r in repo.query(
            "SELECT DISTINCT query_type FROM llm_traces "
            "WHERE query_type IS NOT NULL ORDER BY query_type")]
        bounds = repo.query("SELECT min(timestamp), max(timestamp) FROM llm_traces")[0]
    finally:
        repo.close()
    return providers, tenants, use_cases, query_types, bounds


@st.cache_data(ttl=15)
def load_model_by_query_type(database: str):
    """Which model is better for which kind of request — the routing argument."""
    repo = TraceRepository(database, create_schema=False)
    try:
        return repo.query_df("SELECT * FROM v_model_by_query_type")
    finally:
        repo.close()


@st.cache_data(ttl=15)
def get_db_info(database: str):
    """None when this project has never written traces — deliberately not an
    empty result, which would look like the data was lost rather than never
    written. The DuckDB version made the same distinction with a missing file."""
    cfg = get_clickhouse_config(database)
    if not database_exists(cfg):
        return None
    repo = TraceRepository(database, create_schema=False)
    try:
        row = repo.query("SELECT count(), max(timestamp) FROM llm_traces")[0]
    finally:
        repo.close()
    return {
        "database": database,
        "server": f"{cfg.host}:{cfg.port}",
        "rows": row[0],
        "last_trace": row[1],
    }


def format_currency(value: float) -> str:
    return f"${value:,.4f}" if value is not None else "N/A"


def _compact(n: float) -> str:
    """1284 -> 1,284 · 12900 -> 12.9K · 3400000 -> 3.4M"""
    if n is None or pd.isna(n):
        return "N/A"
    n = float(n)
    for limit, suffix in ((1_000_000, "M"), (1_000, "K")):
        if abs(n) >= limit:
            return f"{n / limit:,.1f}{suffix}"
    return f"{n:,.0f}"


def _tile(label: str, value: str, sub: str = "", tone: str = "") -> str:
    cls = f"sub {tone}".strip()
    sub_html = f'<p class="{cls}">{html.escape(sub)}</p>' if sub else ""
    return (
        f'<div class="tile"><div class="lbl">{html.escape(label)}</div>'
        f'<div class="val">{html.escape(value)}</div>{sub_html}</div>'
    )


def _tiles(items: list[str]) -> None:
    st.markdown(f'<div class="tiles">{"".join(items)}</div>',
                unsafe_allow_html=True)


def _section(title: str, subtitle: str = "") -> None:
    st.markdown(f'<p class="sec">{html.escape(title)}</p>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<p class="sec-sub">{html.escape(subtitle)}</p>',
                    unsafe_allow_html=True)


def _chrome(chart: alt.Chart, height: int, T: dict) -> alt.Chart:
    """Recessive hairline axes, muted labels, no view border."""
    return (
        chart.properties(height=height, background=T["surface"])
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor=T["muted"], titleColor=T["ink_soft"],
            labelFont=FONT, titleFont=FONT, labelFontSize=11, titleFontSize=11,
            gridColor=T["grid"], gridWidth=1,
            domainColor=T["axis"], domainWidth=1,
            tickColor=T["axis"], tickSize=4, labelPadding=4,
        )
        .configure_legend(
            labelColor=T["ink_soft"], titleColor=T["ink_soft"],
            labelFont=FONT, titleFont=FONT, labelFontSize=11, titleFontSize=11,
            orient="top", direction="horizontal", offset=6,
        )
    )


def _ranked_bar(df, cat, val, cat_title, val_title, value_format,
                tooltip_title, T) -> alt.Chart:
    """One measure across nominal categories: one hue, labels at the tip."""
    # Headroom on the value axis so tip labels are not clipped at the edge.
    top = float(df[val].max() or 0)
    scale = alt.Scale(domain=[0, top * 1.28 if top > 0 else 1.0], nice=False)
    extra = ([alt.Tooltip("call_count:Q", title="Calls", format=",")]
             if "call_count" in df.columns else [])

    base = alt.Chart(df).encode(
        y=alt.Y(f"{cat}:N", sort="-x", title=None,
                axis=alt.Axis(labelLimit=190, labelPadding=8)),
        x=alt.X(f"{val}:Q", title=val_title, scale=scale,
                axis=alt.Axis(format=value_format, grid=True, tickCount=5,
                              labelOverlap=True)),
        tooltip=[alt.Tooltip(f"{cat}:N", title=cat_title),
                 alt.Tooltip(f"{val}:Q", title=tooltip_title,
                             format=value_format)] + extra,
    )
    bars = base.mark_bar(size=16, cornerRadiusEnd=4, color=T["s1"])
    labels = base.mark_text(align="left", dx=8, baseline="middle",
                            color=T["ink_soft"], font=FONT, fontSize=11
                            ).encode(text=alt.Text(f"{val}:Q", format=value_format))
    return _chrome(bars + labels, max(120, 46 * len(df)), T)


def _percentile_range(lp: pd.DataFrame, T: dict) -> alt.Chart:
    """
    p50 -> p99 spread per model. A range, so the form is a dumbbell: a track
    from p50 to p99 with a marker per percentile. Reads the same with one model
    or ten, where grouped bars collapse into stripes at low model counts.
    """
    long = lp.melt(id_vars=["provider", "model", "call_count"],
                   value_vars=["p50_ms", "p90_ms", "p99_ms"],
                   var_name="percentile", value_name="ms")
    long["percentile"] = long["percentile"].str.replace("_ms", "", regex=False)

    top = float(long["ms"].max() or 0)
    scale = alt.Scale(domain=[0, top * 1.12 if top > 0 else 1.0], nice=False)
    y = alt.Y("model:N", title=None, axis=alt.Axis(labelLimit=200, labelPadding=8))

    track = alt.Chart(lp).mark_rule(color=T["grid"], strokeWidth=3).encode(
        y=y,
        x=alt.X("p50_ms:Q", title="Latency (ms)", scale=scale,
                axis=alt.Axis(format=",.0f", grid=True, tickCount=6)),
        x2="p99_ms:Q",
    )
    dots = alt.Chart(long).mark_point(
        size=115, filled=True, stroke=T["surface"], strokeWidth=2,
    ).encode(
        y=y, x=alt.X("ms:Q", scale=scale),
        color=alt.Color("percentile:N",
                        # Fixed domain, so filtering never repaints survivors.
                        scale=alt.Scale(domain=["p50", "p90", "p99"],
                                        range=[T["s1"], T["s2"], T["s3"]]),
                        legend=alt.Legend(title=None, symbolType="circle")),
        tooltip=[alt.Tooltip("model:N", title="Model"),
                 alt.Tooltip("percentile:N", title="Percentile"),
                 alt.Tooltip("ms:Q", title="Latency (ms)", format=",.1f"),
                 alt.Tooltip("call_count:Q", title="Calls", format=",")],
    )
    return _chrome(track + dots, max(150, 58 * lp["model"].nunique()), T)


def _by_day(daily, field, y_title, value_format, T):
    """
    One measure per day. Columns while the span is short — discrete buckets are
    easier to compare and a 2-point line barely reads as a trend — switching to
    a line once there are enough days for a trend to mean something.
    """
    d = daily.copy()
    top = float(d[field].max() or 0)
    y = alt.Y(f"{field}:Q", title=y_title,
              scale=alt.Scale(domain=[0, top * 1.18 if top > 0 else 1.0],
                              nice=False),
              axis=alt.Axis(format=value_format, grid=True, tickCount=5))
    tooltip = [alt.Tooltip("date:T", title="Date", format="%d %b %Y"),
               alt.Tooltip("cost_usd:Q", title="Cost", format="$,.6f"),
               alt.Tooltip("calls:Q", title="Calls", format=",")]

    if len(d) <= 14:
        # Ordinal axis: evenly spaced bars and one label per day, rather than a
        # temporal scale inventing ticks between them.
        d["day"] = d["date"].dt.strftime("%d %b")
        x = alt.X("day:O", sort=None, title=None,
                  axis=alt.Axis(grid=False, labelAngle=0))
        bars = alt.Chart(d).mark_bar(size=30, cornerRadiusEnd=4,
                                     color=T["s1"]).encode(x=x, y=y,
                                                           tooltip=tooltip)
        labels = alt.Chart(d).mark_text(
            dy=-9, color=T["ink_soft"], font=FONT, fontSize=11,
        ).encode(x=x, y=f"{field}:Q",
                 text=alt.Text(f"{field}:Q", format=value_format))
        return _chrome(bars + labels, 250, T)

    base = alt.Chart(d).encode(
        x=alt.X("date:T", title=None,
                axis=alt.Axis(format="%d %b", grid=False, labelOverlap=True,
                              tickCount=8)),
        y=y, tooltip=tooltip)
    line = base.mark_line(color=T["s1"], strokeWidth=2, interpolate="monotone")
    dots = base.mark_point(size=64, filled=True, color=T["s1"],
                           stroke=T["surface"], strokeWidth=2)
    end = (alt.Chart(d.tail(1))
           .mark_text(align="right", dy=-14, color=T["ink_soft"], font=FONT,
                      fontSize=11, fontWeight=600)
           .encode(x="date:T", y=f"{field}:Q",
                   text=alt.Text(f"{field}:Q", format=value_format)))
    return _chrome(line + dots + end, 250, T)


def _histogram(df, field, x_title, value_format, T, bins=18) -> alt.Chart:
    """Distribution of one measure. One hue; binSpacing gives the surface gap."""
    chart = alt.Chart(df.dropna(subset=[field])).mark_bar(
        binSpacing=2, cornerRadiusEnd=3, color=T["s1"],
    ).encode(
        x=alt.X(f"{field}:Q", bin=alt.Bin(maxbins=bins), title=x_title,
                axis=alt.Axis(format=value_format, grid=False)),
        y=alt.Y("count():Q", title="Calls",
                axis=alt.Axis(grid=True, tickMinStep=1, format=",.0f")),
        tooltip=[alt.Tooltip(f"{field}:Q", bin=alt.Bin(maxbins=bins),
                             title=x_title, format=value_format),
                 alt.Tooltip("count():Q", title="Calls", format=",")],
    )
    return _chrome(chart, 230, T)


def _score_counts(df: pd.DataFrame, T: dict) -> alt.Chart:
    """
    Quality is a whole number 1-10, so count per score on an ordinal axis.
    Binning it as a continuous measure produces fractional bins whose rounded
    labels repeat ("9 9 9 9 10 10 10 10").
    """
    d = df.dropna(subset=["quality_score"]).copy()
    d["quality_score"] = d["quality_score"].astype(int)
    agg = d.groupby("quality_score").size().reset_index(name="calls")
    chart = alt.Chart(agg).mark_bar(size=22, cornerRadiusEnd=3,
                                    color=T["s1"]).encode(
        x=alt.X("quality_score:O", title="Quality (1-10)",
                scale=alt.Scale(domain=list(range(1, 11))),
                axis=alt.Axis(labelAngle=0, grid=False)),
        y=alt.Y("calls:Q", title="Calls",
                axis=alt.Axis(grid=True, tickMinStep=1, format=",.0f")),
        tooltip=[alt.Tooltip("quality_score:O", title="Score"),
                 alt.Tooltip("calls:Q", title="Calls", format=",")],
    )
    return _chrome(chart, 230, T)


def _grouped_tokens(df: pd.DataFrame, T: dict) -> alt.Chart:
    """Input vs output tokens per use case: two series, grouped (not stacked)
    so the surface gap comes from the band rather than a stroke."""
    long = df.melt(id_vars=["use_case"],
                   value_vars=["input_tokens", "output_tokens"],
                   var_name="kind", value_name="tokens")
    long["kind"] = long["kind"].map({"input_tokens": "Input",
                                     "output_tokens": "Output"})
    chart = alt.Chart(long).mark_bar(size=13, cornerRadiusEnd=3).encode(
        y=alt.Y("use_case:N", title=None,
                axis=alt.Axis(labelLimit=190, labelPadding=8)),
        yOffset=alt.YOffset("kind:N", sort=["Input", "Output"]),
        x=alt.X("tokens:Q", title="Tokens",
                axis=alt.Axis(format=",.0f", grid=True, tickCount=5)),
        color=alt.Color("kind:N",
                        scale=alt.Scale(domain=["Input", "Output"],
                                        range=[T["s1"], T["s2"]]),
                        legend=alt.Legend(title=None)),
        tooltip=[alt.Tooltip("use_case:N", title="Use case"),
                 alt.Tooltip("kind:N", title="Direction"),
                 alt.Tooltip("tokens:Q", title="Tokens", format=",")],
    )
    return _chrome(chart, max(170, 70 * df["use_case"].nunique()), T)


def _cost_quality(df: pd.DataFrame, T: dict) -> alt.Chart:
    """Cost against quality, one dot per call. Single hue: colouring by
    category here would need the all-pairs palette cap."""
    d = df.dropna(subset=["cost_usd", "quality_score"])
    chart = alt.Chart(d).mark_point(
        size=110, filled=True, color=T["s1"], opacity=0.85,
        stroke=T["surface"], strokeWidth=2,
    ).encode(
        x=alt.X("cost_usd:Q", title="Cost (USD)",
                axis=alt.Axis(format="$,.5f", grid=True, tickCount=5)),
        y=alt.Y("quality_score:Q", title="Quality (1-10)",
                scale=alt.Scale(domain=[0, 10.5], nice=False),
                axis=alt.Axis(grid=True, tickCount=6)),
        tooltip=[alt.Tooltip("model:N", title="Model"),
                 alt.Tooltip("use_case:N", title="Use case"),
                 alt.Tooltip("cost_usd:Q", title="Cost", format="$,.6f"),
                 alt.Tooltip("quality_score:Q", title="Quality"),
                 alt.Tooltip("latency_ms:Q", title="Latency (ms)", format=",.0f")],
    )
    return _chrome(chart, 260, T)


def _hourly(df: pd.DataFrame, T: dict) -> alt.Chart:
    """Calls by hour of day. Column form, one hue."""
    d = df.dropna(subset=["timestamp"]).copy()
    d["hour"] = d["timestamp"].dt.hour
    agg = (d.groupby("hour")
           .agg(calls=("trace_id", "count"), cost_usd=("cost_usd", "sum"))
           .reindex(range(24), fill_value=0).reset_index())
    chart = alt.Chart(agg).mark_bar(size=14, cornerRadiusEnd=3,
                                    color=T["s1"]).encode(
        x=alt.X("hour:O", title="Hour of day (UTC)",
                axis=alt.Axis(grid=False, labelAngle=0, values=list(range(0, 24, 2)))),
        y=alt.Y("calls:Q", title="Calls",
                axis=alt.Axis(grid=True, tickMinStep=1, format=",.0f")),
        tooltip=[alt.Tooltip("hour:O", title="Hour"),
                 alt.Tooltip("calls:Q", title="Calls", format=","),
                 alt.Tooltip("cost_usd:Q", title="Cost", format="$,.6f")],
    )
    return _chrome(chart, 210, T)


def main() -> None:
    st.set_page_config(page_title="LLM Cost Monitor", page_icon="📊",
                       layout="wide", initial_sidebar_state="collapsed")
    T, mode = _tokens()
    st.markdown(_css(T), unsafe_allow_html=True)

    chip = f'<span class="chip">{html.escape(PROJECT_NAME)}</span>' if PROJECT_NAME else ""
    st.markdown(f'<div class="hdr"><h1>LLM Cost Monitor</h1>{chip}</div>',
                unsafe_allow_html=True)
    st.markdown('<p class="lede">Cost, latency and quality for every LLM call '
                'this project has recorded.</p>', unsafe_allow_html=True)

    try:
        database = resolve_database()
    except ConfigError as exc:
        st.error(str(exc))
        return

    # Sidebar: Data source and actions only. Filters live above the charts.
    with st.sidebar:
        st.markdown("### Data source")
        st.code(database, language=None)
        if st.button("Refresh", width="stretch"):
            st.cache_data.clear()
            st.rerun()
        st.markdown("### Appearance")
        st.caption(
            f"Currently **{mode}**. Switch it from the ⋮ menu, top right → "
            "Settings → Appearance. Charts follow whichever you pick."
        )

    try:
        db_info = get_db_info(database)
    except ClickHouseUnavailable as exc:
        st.error(str(exc))
        return

    if db_info is None:
        st.warning(
            "This project has no trace database yet. Make at least one call through "
            "`LLMMonitor` and save it with `TraceRepository`, then refresh."
        )
        return

    with st.sidebar:
        st.caption(f"Server {db_info['server']}")
        st.caption(f"{db_info['rows']:,} traces")
        if db_info["last_trace"] is not None:
            st.caption(f"Latest {db_info['last_trace']:%d %b %Y, %H:%M:%S}")

    if not db_info["rows"]:
        st.info("The database exists but holds no traces yet. Save a call with "
                "`TraceRepository.save(...)`, then refresh.")
        return

    # Filters: one row, above everything they scope. Read from the server rather
    # than from a loaded frame, because the frame is now the *result* of them.
    provider_options, tenant_options, use_case_options, query_type_options, bounds = \
        get_filter_options(database)
    min_date, max_date = bounds

    with st.container(border=True):
        f1, f2 = st.columns([1, 2], gap="medium")
        with f1:
            # Date range filter
            date_range = ()
            if min_date is not None and max_date is not None:
                date_range = st.date_input(
                    "Date range", value=(min_date.date(), max_date.date()),
                    min_value=min_date.date(), max_value=max_date.date())
        with f2:
            providers = st.pills("Provider", provider_options,
                                 selection_mode="multi", default=provider_options)
        f3, f4 = st.columns(2, gap="medium")
        with f3:
            tenants = st.pills("Tenant", tenant_options,
                               selection_mode="multi", default=tenant_options)
        with f4:
            use_cases = st.pills("Use case", use_case_options,
                                 selection_mode="multi", default=use_case_options)
        if query_type_options:
            # Only shown once something has been classified — an empty row of
            # pills would just be noise for projects with classification off.
            query_types = st.pills("Query type", query_type_options,
                                   selection_mode="multi",
                                   default=query_type_options)
        else:
            query_types = None

    start_ts = end_ts = None
    if len(date_range) == 2:
        start_ts = pd.Timestamp(date_range[0]).to_pydatetime()
        end_ts = (pd.Timestamp(date_range[1])
                  + pd.Timedelta(days=1)).to_pydatetime()

    # One query, already filtered. No client-side re-filtering below: the server
    # has applied all four filters.
    try:
        data = load_data(database, start_ts, end_ts, providers, tenants,
                         use_cases, query_types)
    except ClickHouseUnavailable as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"Could not read the trace database: {exc}")
        return

    traces, cost_by_tenant, cost_by_use_case, latency_percentiles = data
    filtered = traces

    if filtered.empty:
        st.warning("No data matches the selected filters.")
        return

    # KPI Row
    total_calls = len(filtered)
    total_cost = filtered["cost_usd"].sum(skipna=True)
    avg_latency = filtered["latency_ms"].mean(skipna=True)
    avg_quality = filtered["quality_score"].mean(skipna=True)
    total_in = filtered["input_tokens"].sum(skipna=True)
    total_out = filtered["output_tokens"].sum(skipna=True)
    avg_throughput = filtered["tokens_per_second"].mean(skipna=True)
    failures = int(filtered["error"].notna().sum())
    retried = int((filtered["attempts_used"].fillna(1) > 1).sum())
    truncated = int((filtered["finish_reason"] == "length").sum())
    span_days = filtered["timestamp"].dt.normalize().nunique()

    _tiles([
        _tile("Calls", f"{total_calls:,}",
              f"{failures} failed" if failures else "all succeeded",
              "bad" if failures else "good"),
        _tile("Total cost", format_currency(total_cost),
              f"{format_currency(total_cost / total_calls)} per call"),
        _tile("Avg latency", f"{avg_latency:,.0f} ms",
              f"{filtered['latency_ms'].max():,.0f} ms slowest"),
        _tile("Avg quality", f"{avg_quality:,.1f}" if pd.notna(avg_quality) else "N/A",
              "judge score, 1-10"),
        _tile("Tokens", _compact(total_in + total_out),
              f"{_compact(total_in)} in · {_compact(total_out)} out"),
        _tile("Throughput",
              f"{avg_throughput:,.0f} tok/s" if pd.notna(avg_throughput) else "N/A",
              "output only"),
    ])
    _tiles([
        _tile("Retries", f"{retried:,}",
              "none needed" if not retried else "calls retried",
              "good" if not retried else "bad"),
        _tile("Truncated", f"{truncated:,}",
              "hit the token cap" if truncated else "none cut off",
              "bad" if truncated else "good"),
        _tile("Models", f"{filtered['model'].nunique():,}",
              f"{filtered['provider'].nunique()} provider(s)"),
        _tile("Tenants", f"{filtered['tenant_id'].nunique():,}",
              f"{filtered['use_case'].nunique()} use case(s)"),
        _tile("Active days", f"{span_days:,}",
              f"{total_calls / max(span_days, 1):,.1f} calls/day"),
        _tile("Cost efficiency",
              f"{filtered['cost_efficiency'].mean(skipna=True):,.0f}"
              if filtered["cost_efficiency"].notna().any() else "N/A",
              "quality per $"),
    ])

    # Daily rollup, shared by the Cost and Activity tabs so both agree.
    ts = filtered.dropna(subset=["timestamp"]).copy()
    daily = pd.DataFrame(columns=["date", "cost_usd", "calls"])
    if not ts.empty:
        ts["date"] = ts["timestamp"].dt.floor("D")
        daily = (ts.groupby("date")
                 .agg(cost_usd=("cost_usd", "sum"), calls=("trace_id", "count"))
                 .reset_index().sort_values("date"))

    tab_cost, tab_lat, tab_qual, tab_tok, tab_time, tab_calls, tab_sql = st.tabs(
        ["Cost", "Latency", "Quality", "Tokens", "Activity", "Calls", "SQL"]
    )

    # Analytics Views
    with tab_cost:
        _section("Cost by day", "Total spend per day for the current filter.")
        if daily.empty:
            st.info("No timestamped calls in this selection.")
        else:
            st.altair_chart(_by_day(daily, "cost_usd", "Cost (USD)", "$,.4f", T),
                            width="stretch")

        c1, c2 = st.columns(2, gap="large")
        with c1:
            _section("Cost by tenant", "Which customer the spend belongs to.")
            # Filter cost_by_tenant to match current filters
            tenant_costs = (cost_by_tenant[cost_by_tenant["tenant_id"].isin(tenants)]
                            if tenants else cost_by_tenant)
            if tenant_costs.empty:
                st.info("No tenant data. Pass `tenant_id=` to `monitor.call()`.")
            else:
                st.altair_chart(_ranked_bar(tenant_costs, "tenant_id",
                                            "total_cost_usd", "Tenant",
                                            "Total cost (USD)", "$,.4f",
                                            "Total cost", T), width="stretch")
                st.dataframe(
                    tenant_costs.sort_values("total_cost_usd", ascending=False),
                    width="stretch", hide_index=True,
                    column_config={
                        "tenant_id": st.column_config.TextColumn("Tenant"),
                        "call_count": st.column_config.NumberColumn("Calls", format="%d"),
                        "total_cost_usd": st.column_config.NumberColumn("Total cost", format="$%.6f"),
                        "avg_cost_usd": st.column_config.NumberColumn("Avg / call", format="$%.6f"),
                        "total_input_tokens": st.column_config.NumberColumn("Input tokens", format="%d"),
                        "total_output_tokens": st.column_config.NumberColumn("Output tokens", format="%d"),
                    },
                )
        with c2:
            _section("Cost by use case", "Which feature the spend belongs to.")
            uc_costs = (cost_by_use_case[cost_by_use_case["use_case"].isin(use_cases)]
                        if use_cases else cost_by_use_case)
            if uc_costs.empty:
                st.info("No use-case data. Pass `use_case=` to `monitor.call()`.")
            else:
                st.altair_chart(_ranked_bar(uc_costs, "use_case",
                                            "total_cost_usd", "Use case",
                                            "Total cost (USD)", "$,.4f",
                                            "Total cost", T), width="stretch")
                st.dataframe(
                    uc_costs.sort_values("total_cost_usd", ascending=False),
                    width="stretch", hide_index=True,
                    column_config={
                        "use_case": st.column_config.TextColumn("Use case"),
                        "call_count": st.column_config.NumberColumn("Calls", format="%d"),
                        "total_cost_usd": st.column_config.NumberColumn("Total cost", format="$%.6f"),
                        "avg_cost_usd": st.column_config.NumberColumn("Avg / call", format="$%.6f"),
                        "avg_quality_score": st.column_config.NumberColumn("Avg quality", format="%.1f"),
                    },
                )

        by_model = (filtered.groupby("model", dropna=True)
                    .agg(total_cost_usd=("cost_usd", "sum"),
                         call_count=("trace_id", "count"))
                    .reset_index())
        # A single model is a number, not a chart — a one-bar bar chart says
        # nothing the tiles above do not already say.
        if len(by_model) < 2:
            row = by_model.iloc[0]
            _section("Cost by model",
                     "Only one model in this selection, so there is nothing "
                     "to compare.")
            _tiles([_tile(str(row["model"]),
                          format_currency(row["total_cost_usd"]),
                          f"{int(row['call_count']):,} calls")])
        else:
            _section("Cost by model", "Same spend, grouped by which model answered.")
            st.altair_chart(_ranked_bar(by_model, "model", "total_cost_usd",
                                        "Model", "Total cost (USD)", "$,.4f",
                                        "Total cost", T), width="stretch")

        _section("Most expensive calls", "The ten calls that cost the most.")
        st.dataframe(
            filtered.nlargest(10, "cost_usd")[
                ["timestamp", "tenant_id", "use_case", "model", "cost_usd",
                 "total_tokens", "latency_ms", "quality_score", "prompt"]],
            width="stretch", hide_index=True,
            column_config={
                "timestamp": st.column_config.DatetimeColumn("Time", format="YYYY-MM-DD HH:mm"),
                "tenant_id": st.column_config.TextColumn("Tenant"),
                "use_case": st.column_config.TextColumn("Use case"),
                "model": st.column_config.TextColumn("Model"),
                "cost_usd": st.column_config.NumberColumn("Cost", format="$%.6f"),
                "total_tokens": st.column_config.NumberColumn("Tokens", format="%d"),
                "latency_ms": st.column_config.NumberColumn("Latency (ms)", format="%.0f"),
                "quality_score": st.column_config.NumberColumn("Quality", format="%d"),
                "prompt": st.column_config.TextColumn("Prompt", width="large"),
            },
        )

    # Latency Percentiles
    with tab_lat:
        _section("Latency percentiles by model",
                 "p50 is the typical call; p99 is the tail users complain about.")
        if latency_percentiles.empty:
            st.info("Latency percentile analytics are not available yet.")
        else:
            lp = (latency_percentiles[latency_percentiles["provider"].isin(providers)]
                  if providers else latency_percentiles)
            if lp.empty:
                st.info("No latency data for the selected providers.")
            else:
                st.altair_chart(_percentile_range(lp, T), width="stretch")
                st.dataframe(
                    lp.sort_values(["provider", "model"]),
                    width="stretch", hide_index=True,
                    column_config={
                        "provider": st.column_config.TextColumn("Provider"),
                        "model": st.column_config.TextColumn("Model"),
                        "call_count": st.column_config.NumberColumn("Calls", format="%d"),
                        "p50_ms": st.column_config.NumberColumn("p50 (ms)", format="%.1f"),
                        "p90_ms": st.column_config.NumberColumn("p90 (ms)", format="%.1f"),
                        "p99_ms": st.column_config.NumberColumn("p99 (ms)", format="%.1f"),
                        "avg_ms": st.column_config.NumberColumn("avg (ms)", format="%.1f"),
                    },
                )
                if lp["call_count"].max() < 20:
                    st.caption("Fewer than 20 calls per model — these percentiles "
                               "show the mechanism, not a reliable distribution.")

        l1, l2 = st.columns(2, gap="large")
        with l1:
            _section("Latency distribution", "How the individual calls spread out.")
            st.altair_chart(_histogram(filtered, "latency_ms", "Latency (ms)",
                                       ",.0f", T), width="stretch")
        with l2:
            _section("Throughput by model", "Output tokens generated per second.")
            tp = (filtered.groupby("model", dropna=True)
                  .agg(tokens_per_second=("tokens_per_second", "mean"),
                       call_count=("trace_id", "count")).reset_index())
            st.altair_chart(_ranked_bar(tp, "model", "tokens_per_second",
                                        "Model", "Tokens / second", ",.0f",
                                        "Avg throughput", T), width="stretch")

        _section("Slowest calls", "The ten calls that took the longest.")
        st.dataframe(
            filtered.nlargest(10, "latency_ms")[
                ["timestamp", "model", "latency_ms", "attempts_used",
                 "total_tokens", "tokens_per_second", "finish_reason"]],
            width="stretch", hide_index=True,
            column_config={
                "timestamp": st.column_config.DatetimeColumn("Time", format="YYYY-MM-DD HH:mm"),
                "model": st.column_config.TextColumn("Model"),
                "latency_ms": st.column_config.NumberColumn("Latency (ms)", format="%.0f"),
                "attempts_used": st.column_config.NumberColumn("Attempts", format="%d"),
                "total_tokens": st.column_config.NumberColumn("Tokens", format="%d"),
                "tokens_per_second": st.column_config.NumberColumn("tok/s", format="%.1f"),
                "finish_reason": st.column_config.TextColumn("Finish"),
            },
        )

    with tab_qual:
        if filtered["quality_score"].notna().sum() == 0:
            st.info("No quality scores in this selection. Quality scoring is on "
                    "by default; pass `score_response_quality=False` to skip it.")
        else:
            q1, q2 = st.columns(2, gap="large")
            with q1:
                _section("Quality distribution",
                         "How the judge scored these responses.")
                st.altair_chart(_score_counts(filtered, T), width="stretch")
            with q2:
                _section("Cost against quality",
                         "One dot per call. Up and to the left is better value.")
                st.altair_chart(_cost_quality(filtered, T), width="stretch")

            _section("Quality by use case", "Average judge score per feature.")
            qbu = (filtered.dropna(subset=["quality_score"])
                   .groupby("use_case", dropna=True)
                   .agg(quality_score=("quality_score", "mean"),
                        call_count=("trace_id", "count")).reset_index())
            if not qbu.empty:
                st.altair_chart(_ranked_bar(qbu, "use_case", "quality_score",
                                            "Use case", "Avg quality (1-10)",
                                            ",.1f", "Avg quality", T),
                                width="stretch")

        _section("Models by query type",
                 "Same job, different models. High quality per dollar is the "
                 "cheaper model doing comparable work.")
        mbq = load_model_by_query_type(database)
        if mbq.empty:
            st.info("No classified calls yet. Turn classification on with "
                    "`obs.init(..., classify_query=True)` — it shares the judge "
                    "call with quality scoring, so it adds no API calls if "
                    "scoring is already enabled.")
        else:
            st.dataframe(
                mbq.sort_values(["query_type", "quality_per_dollar"],
                                ascending=[True, False]),
                width="stretch", hide_index=True,
                column_config={
                    "query_type": st.column_config.TextColumn("Query type"),
                    "provider": st.column_config.TextColumn("Provider"),
                    "model": st.column_config.TextColumn("Model"),
                    "call_count": st.column_config.NumberColumn("Calls", format="%d"),
                    "avg_cost_usd": st.column_config.NumberColumn("Avg cost", format="$%.6f"),
                    "avg_quality": st.column_config.NumberColumn("Avg quality", format="%.1f"),
                    "p90_ms": st.column_config.NumberColumn("p90 (ms)", format="%.0f"),
                    "quality_per_dollar": st.column_config.NumberColumn("Quality / $", format="%.0f"),
                },
            )

        _section("Finish reasons", "`stop` finished naturally; `length` was cut off.")
        fr = (filtered.groupby("finish_reason", dropna=False)
              .agg(call_count=("trace_id", "count")).reset_index())
        fr["finish_reason"] = fr["finish_reason"].fillna("(none — call failed)")
        st.altair_chart(_ranked_bar(fr, "finish_reason", "call_count",
                                    "Finish reason", "Calls", ",.0f", "Calls", T),
                        width="stretch")

    with tab_tok:
        _section("Input vs output tokens by use case",
                 "Long prompts and long answers cost differently.")
        tok = (filtered.groupby("use_case", dropna=True)
               .agg(input_tokens=("input_tokens", "sum"),
                    output_tokens=("output_tokens", "sum")).reset_index())
        if tok.empty:
            st.info("No use-case data to group tokens by.")
        else:
            st.altair_chart(_grouped_tokens(tok, T), width="stretch")

        t1, t2 = st.columns(2, gap="large")
        with t1:
            _section("Tokens per call", "Distribution across the selection.")
            st.altair_chart(_histogram(filtered, "total_tokens", "Tokens per call",
                                       ",.0f", T), width="stretch")
        with t2:
            _section("Avg tokens per call by model", "")
            tpm = (filtered.groupby("model", dropna=True)
                   .agg(total_tokens=("total_tokens", "mean"),
                        call_count=("trace_id", "count")).reset_index())
            st.altair_chart(_ranked_bar(tpm, "model", "total_tokens", "Model",
                                        "Avg tokens", ",.0f", "Avg tokens", T),
                            width="stretch")

    # Time Series: call volume. Spend per day lives in the Cost tab.
    with tab_time:
        if daily.empty:
            st.info("No timestamped calls in this selection.")
        else:
            _section("Calls by day", "How much traffic this project sent.")
            st.altair_chart(_by_day(daily, "calls", "Calls", ",.0f", T),
                            width="stretch")

            _section("Calls by hour of day", "When this project talks to the LLM.")
            st.altair_chart(_hourly(filtered, T), width="stretch")

            _section("Daily totals", "")
            st.dataframe(
                daily, width="stretch", hide_index=True,
                column_config={
                    "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                    "cost_usd": st.column_config.NumberColumn("Cost (USD)", format="$%.6f"),
                    "calls": st.column_config.NumberColumn("Calls", format="%d"),
                },
            )

    # Recent Calls Table
    with tab_calls:
        _section("Recent calls", "Newest first, capped at 200 rows.")
        recent = filtered.sort_values("timestamp", ascending=False).head(200)
        st.dataframe(
            recent[["timestamp", "tenant_id", "use_case", "provider", "model",
                    "cost_usd", "latency_ms", "quality_score", "finish_reason",
                    "total_tokens", "attempts_used"]],
            width="stretch", hide_index=True,
            column_config={
                "timestamp": st.column_config.DatetimeColumn("Time", format="YYYY-MM-DD HH:mm:ss"),
                "tenant_id": st.column_config.TextColumn("Tenant"),
                "use_case": st.column_config.TextColumn("Use case"),
                "provider": st.column_config.TextColumn("Provider"),
                "model": st.column_config.TextColumn("Model"),
                "cost_usd": st.column_config.NumberColumn("Cost (USD)", format="$%.6f"),
                "latency_ms": st.column_config.NumberColumn("Latency (ms)", format="%.1f"),
                "quality_score": st.column_config.NumberColumn("Quality", format="%d"),
                "finish_reason": st.column_config.TextColumn("Finish"),
                "total_tokens": st.column_config.NumberColumn("Tokens", format="%d"),
                "attempts_used": st.column_config.NumberColumn("Attempts", format="%d"),
            },
        )

        # Export Section
        st.download_button(
            "Download these rows as CSV",
            filtered.to_csv(index=False).encode("utf-8"),
            f"llm_traces_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv")

    with tab_sql:
        _section("SQL explorer",
                 "Table: llm_traces · Views: v_cost_by_tenant, "
                 "v_cost_by_use_case, v_latency_percentiles, v_summary")
        query = st.text_area(
            "Query", height=110,
            value="SELECT * FROM llm_traces ORDER BY timestamp DESC LIMIT 50;")
        if st.button("Run query"):
            if query.strip():
                repo = TraceRepository(database, create_schema=False)
                try:
                    result = repo.query_df(query)
                    st.dataframe(result, width="stretch", hide_index=True)
                    st.success(f"Returned {len(result):,} rows")
                except Exception as e:
                    st.error(f"Query failed: {e}")
                finally:
                    repo.close()


if __name__ == "__main__":
    main()
