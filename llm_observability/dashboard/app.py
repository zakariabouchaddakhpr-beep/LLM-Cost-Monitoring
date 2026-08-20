"""
The web dashboard: charts showing one project's AI call records.

HOW STREAMLIT WORKS - read this first, it explains a lot
A Streamlit app is an ordinary Python script that runs from top to bottom.
The surprising part: every time you click a button or change a filter,
Streamlit runs the ENTIRE script again, from line one.

That one fact explains most of the odd-looking things below:

  - Slow database reads are wrapped in @st.cache_data, which remembers the
    answer for a few seconds. Without it, every click would hit the database
    again for data that hasn't changed.

  - There are no click handlers or callbacks anywhere. You don't react to a
    click; you just check what the filters currently say and draw the page to
    match. The click already caused a full rerun.

HOW THIS FILE IS LAID OUT
  1. colours and CSS      - what the page looks like
  2. loading the data     - the cached functions that read the database
  3. small formatters     - turning numbers into readable text
  4. chart builders       - one function per chart, each returns a chart
  5. main()               - lays the page out and calls everything above

Which database it opens is worked out the same way the library does it, so
starting this from inside a project shows that project's records and nobody
else's.
"""

import html
import sys
from datetime import datetime
from pathlib import Path

# Streamlit doesn't add this script's own folder to the list of places Python
# searches for imports, and the folder you started from belongs to the project
# being monitored. So without this, Python can't find llm_observability at all.
#
# parents[] walks up the folder tree. This file is at
# llm_observability/dashboard/app.py, so:
#   parents[0] = dashboard/   parents[1] = llm_observability/   parents[2] = the root
# We want parents[2], the folder CONTAINING the package.
#
# sys.path is that list of search folders, and insert(0, ...) puts ours first.
# This has to happen BEFORE the llm_observability imports further down, which
# is why it sits oddly in the middle of the import block.
_PACKAGE_PARENT_DIR = str(Path(__file__).resolve().parents[2])
if _PACKAGE_PARENT_DIR not in sys.path:
    sys.path.insert(0, _PACKAGE_PARENT_DIR)

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

# ---------------------------------------------------------------------------
# 1. Colours and CSS
# ---------------------------------------------------------------------------

# Two separate sets of colours - NOT one set with the light and dark values
# swapped. Each was checked against its own background for brightness, for
# telling the colours apart (including for colour-blind readers), and for
# whether text stays readable on it.
#
# What the names mean:
#   surface   the page background
#   ink       main text      ink_soft  less important text   muted  labels
#   grid      the faint lines across a chart
#   axis      the solid line along the edge of a chart
#   s1/s2/s3  the three colours used for data in charts
#   chip      the little rounded pill behind the project name
#   good/bad  green and red, for "all fine" and "something failed"
#
# One honest caveat: in light mode, s3 is slightly too pale to meet the usual
# readability guideline. So any chart using it also has a plain data table
# underneath, which anyone can read regardless.
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

# A list of fonts, not one font. The browser works down the list and uses the
# first one the computer actually has. That way the page looks like it belongs
# on Windows, Mac or Linux without downloading anything.
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"


def theme_colors() -> tuple[dict, str]:
    """
    Pick the colour set matching whichever theme Streamlit is currently using.

    Hands back two things: the colours, and the name of the mode.
    For example (DARK, "dark").

    The whole thing is wrapped in try/except because st.context.theme is a
    newer Streamlit feature. On an older version, asking for it would crash
    the entire page - over a colour choice. Falling back to light is fine.
    """
    mode = "light"
    try:
        # getattr(x, "name", None) means "give me x.name if it exists,
        # otherwise give me None" - so this can't crash on an older version.
        theme = getattr(st.context, "theme", None)
        if theme is not None and getattr(theme, "type", None) == "dark":
            mode = "dark"
    except Exception:
        pass
    return (DARK if mode == "dark" else LIGHT), mode


def page_css(colors: dict) -> str:
    """
    Build the page's styling, with the chosen colours filled in.

    CSS is the language that decides how a web page looks - sizes, spacing,
    colours. The class names used here:

        .hdr    the header row at the top   .chip   the project-name pill
        .lede   the sentence beneath it     .tiles  the grid of number boxes
        .tile   one number box              .sec    a section heading

    NOTE THE DOUBLED BRACES {{ }}.
    This is an f-string, where a single { means "put a Python value here". CSS
    also uses braces for its own purposes. Writing {{ tells Python "this is a
    real brace, leave it alone" - so {{ in the code becomes { on the page.
    """
    return f"""
<style>
  .block-container, [data-testid="stMainBlockContainer"] {{
      padding-top: 2.1rem; padding-bottom: 3rem; max-width: 1560px;
  }}
  .hdr {{ display:flex; align-items:baseline; gap:.7rem; flex-wrap:wrap;
         margin-bottom:.15rem; }}
  .hdr h1 {{ font-family:{FONT}; font-size:1.55rem; font-weight:600;
             color:{colors['ink']}; margin:0; letter-spacing:-.01em; }}
  .chip {{ font-family:{FONT}; font-size:.72rem; font-weight:600;
           color:{colors['chip_ink']}; background:{colors['chip']};
           border-radius:999px; padding:.16rem .6rem; letter-spacing:.02em; }}
  .lede {{ font-family:{FONT}; font-size:.85rem; color:{colors['ink_soft']};
           margin:0 0 1.1rem 0; }}
  .tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(148px,1fr));
            gap:.55rem; margin:.2rem 0 .5rem 0; }}
  .tile {{ background:{colors['surface']}; border:1px solid {colors['border']};
           border-radius:10px; padding:.7rem .85rem; }}
  .tile .lbl {{ font-family:{FONT}; font-size:.68rem; font-weight:600;
                letter-spacing:.05em; text-transform:uppercase;
                color:{colors['muted']}; margin-bottom:.28rem; }}
  .tile .val {{ font-family:{FONT}; font-size:1.45rem; font-weight:600;
                color:{colors['ink']}; line-height:1.12; }}
  .tile .sub {{ font-family:{FONT}; font-size:.71rem; color:{colors['muted']};
                margin:.2rem 0 0 0; }}
  .tile .sub.good {{ color:{colors['good']}; }}
  .tile .sub.bad {{ color:{colors['bad']}; }}
  .sec {{ font-family:{FONT}; font-size:.95rem; font-weight:600;
          color:{colors['ink']}; margin:.35rem 0 .1rem 0; }}
  .sec-sub {{ font-family:{FONT}; font-size:.78rem; color:{colors['muted']};
              margin:0 0 .5rem 0; }}
</style>
"""


# ---------------------------------------------------------------------------
# 2. Loading the data
#
# Every function down here has @st.cache_data written above it. That is a
# "decorator" - a line starting with @ that wraps the function in extra
# behaviour without changing the function itself.
#
# This particular one remembers the answer for `ttl` seconds (ttl = "time to
# live"), so repeated calls skip the database entirely. What it remembers is
# keyed on the ARGUMENTS, so changing a filter correctly counts as a new
# question and does fetch fresh data.
# ---------------------------------------------------------------------------

@st.cache_data(ttl=15, show_spinner="Loading traces from ClickHouse...")
def load_data(database: str, start=None, end=None, providers=None,
              tenants=None, use_cases=None, query_types=None):
    """
    Fetch the records matching the current filters, plus three summary tables.

    Hands back four tables:
        (traces, cost_by_tenant, cost_by_use_case, latency_percentiles)

    The filters get turned into SQL and applied by the DATABASE, rather than
    fetching everything and throwing most of it away here. On a table that
    keeps growing, that is the difference between pulling over a few hundred
    rows and pulling over all of them, every fifteen seconds, for ever.
    """
    conditions, params = [], {}

    # Date range: from `start` up to but not including `end`.
    if start is not None and end is not None:
        conditions.append("timestamp >= %(start)s AND timestamp < %(end)s")
        params["start"], params["end"] = start, end

    # These four "pick from a list" filters all work identically, so we build
    # them in a loop instead of writing the same four blocks out by hand.
    # Each loop turn gives us a column name and the values chosen for it.
    for field, values in (("provider", providers), ("tenant_id", tenants),
                          ("use_case", use_cases), ("query_type", query_types)):
        # Skip filters where nothing was selected.
        if values:
            # IN means "is any one of these". The %(...)s is a placeholder the
            # driver fills in safely - we never glue values into SQL text.
            conditions.append(f"{field} IN %({field}_list)s")
            params[f"{field}_list"] = list(values)

    # Glue the conditions together with AND, or use an empty string if there
    # are none - an empty string simply vanishes when dropped into the query.
    where = "WHERE " + " AND ".join(conditions) if conditions else ""

    # create_schema=False because this only READS. It must never create a
    # database that isn't there - an empty one looks like your data vanished,
    # while a missing one honestly says nothing was ever saved.
    repo = TraceRepository(database, create_schema=False)
    try:
        traces = repo.query_df(f"SELECT * FROM llm_traces {where}", params)
        cost_by_tenant = repo.query_df("SELECT * FROM v_cost_by_tenant")
        cost_by_use_case = repo.query_df("SELECT * FROM v_cost_by_use_case")
        latency_percentiles = repo.query_df("SELECT * FROM v_latency_percentiles")
    finally:
        # `finally` runs even if a query above failed, so the connection
        # always gets closed.
        repo.close()

    # errors="coerce" means: if a date can't be understood, put an empty value
    # there instead of crashing. One malformed row shouldn't blank the entire
    # dashboard.
    traces["timestamp"] = pd.to_datetime(traces["timestamp"], errors="coerce")
    return traces, cost_by_tenant, cost_by_use_case, latency_percentiles


@st.cache_data(ttl=15)
def get_filter_options(database: str):
    """
    Fetch what values you're allowed to filter by, and the earliest and latest
    dates in the data.

    These are cheap questions, and they run BEFORE the records themselves.
    They have to: we need to draw the filter buttons first, because the
    records we then load are the RESULT of whatever those filters say.
    """
    repo = TraceRepository(database, create_schema=False)
    try:
        # DISTINCT means "one of each, no duplicates".
        #
        # Each answer comes back as rows like [("groq",), ("openai",)] - a
        # list of one-item tuples. The `r[0] for r in ...` part pulls the
        # single value out of each row, giving a plain list: ["groq", "openai"].
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
        # min and max together in one question, so we only ask once. [0]
        # because this always returns exactly one row.
        bounds = repo.query(
            "SELECT min(timestamp), max(timestamp) FROM llm_traces")[0]
    finally:
        repo.close()
    return providers, tenants, use_cases, query_types, bounds


@st.cache_data(ttl=15)
def load_model_by_query_type(database: str):
    """
    Fetch the "which model is best for which kind of job" comparison.

    This is the table that answers the routing question. If a cheap model
    scores nearly as well as an expensive one at summarising, you should be
    sending your summaries to the cheap one.
    """
    repo = TraceRepository(database, create_schema=False)
    try:
        return repo.query_df("SELECT * FROM v_model_by_query_type")
    finally:
        repo.close()


@st.cache_data(ttl=15)
def get_db_info(database: str):
    """
    Basic facts about the database: which server, how many rows, newest record.

    Hands back None when the database doesn't exist at all.

    That is deliberately different from handing back an empty result. "No
    database" means nothing has ever been saved here, which is normal for a
    new project. "Empty database" would suggest data was there and has gone
    missing, which is alarming. The dashboard shows a different message for
    each, so you never get frightened by a brand-new project.
    """
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


# ---------------------------------------------------------------------------
# 3. Small formatters and layout helpers
# ---------------------------------------------------------------------------

def format_currency(value: float) -> str:
    """
    Turn a number into a dollar amount: 0.01234 becomes "$0.0123".

    Four decimal places, because individual AI calls are often fractions of a
    cent and "$0.01" would round most of them away to nothing.
    """
    return f"${value:,.4f}" if value is not None else "N/A"


def compact_number(n: float) -> str:
    """
    Shorten a big number so it fits inside a small box.

        1284     ->  "1,284"
        12900    ->  "12.9K"
        3400000  ->  "3.4M"
    """
    # pd.isna catches pandas' version of "empty", which plain `is None` misses.
    if n is None or pd.isna(n):
        return "N/A"

    n = float(n)
    # Check millions BEFORE thousands. The other way round, 3,400,000 would
    # match the thousands rule first and come out as the useless "3400.0K".
    #
    # abs() so that negative numbers get shortened too.
    for limit, suffix in ((1_000_000, "M"), (1_000, "K")):
        if abs(n) >= limit:
            return f"{n / limit:,.1f}{suffix}"
    # Small enough to show in full. The comma in the format adds thousands
    # separators, and .0f means no decimal places.
    return f"{n:,.0f}"


def metric_tile(label: str, value: str, sub: str = "", tone: str = "") -> str:
    """
    Build the HTML for one number box.

        label  the small grey caption on top, e.g. "TOTAL COST"
        value  the big number, e.g. "$0.0421"
        sub    an optional line underneath, e.g. "$0.0014 per call"
        tone   "", "good" (green) or "bad" (red), which colours `sub`

    NOTE html.escape() ON EVERY VALUE.
    These strings can contain data straight from the database. If a customer
    were named "<script>", writing it into the page unescaped would let the
    browser RUN it instead of showing it. escape() turns the dangerous
    characters into harmless ones, so it appears as text - which is what a
    customer name should do.
    """
    # .strip() removes the trailing space when tone is empty.
    css_class = f"sub {tone}".strip()
    # Leave the whole sub-line out entirely when there's nothing to put in it.
    sub_html = f'<p class="{css_class}">{html.escape(sub)}</p>' if sub else ""
    return (
        f'<div class="tile"><div class="lbl">{html.escape(label)}</div>'
        f'<div class="val">{html.escape(value)}</div>{sub_html}</div>'
    )


def render_tiles(items: list[str]) -> None:
    """
    Draw a row of boxes built by metric_tile().

    "".join(items) glues the boxes together into one piece of HTML.
    unsafe_allow_html=True tells Streamlit to treat it as real HTML rather
    than showing the tags as text. That's safe here only because metric_tile
    escaped everything that came from the database.
    """
    st.markdown(f'<div class="tiles">{"".join(items)}</div>',
                unsafe_allow_html=True)


def section_heading(title: str, subtitle: str = "") -> None:
    """Draw a section heading, with an optional line of explanation under it."""
    st.markdown(f'<p class="sec">{html.escape(title)}</p>',
                unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<p class="sec-sub">{html.escape(subtitle)}</p>',
                    unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 4. Chart builders
#
# These use a library called Altair. With Altair you don't draw a chart step
# by step - you DESCRIBE it, by saying which column of data controls which
# part of the picture:
#
#     x=...      position left to right    y=...      position up and down
#     color=...  the colour                tooltip=   the box shown on hover
#
# The :Q, :N, :O and :T suffixes on column names tell Altair what KIND of data
# a column holds, so it can pick sensible axes and sorting:
#
#     Q  a number you can do maths on          (cost, latency)
#     N  a name, with no natural order         (provider, model)
#     O  categories that DO have an order      (a quality score of 1 to 10)
#     T  a date or a time
#
# Every builder takes `colors` (the palette) and hands back a chart. None of
# them decide where on the page it goes - that's main()'s job. Keeping those
# two things separate is why the same chart function gets reused in several
# different tabs.
# ---------------------------------------------------------------------------

def style_chart(chart: alt.Chart, height: int, colors: dict) -> alt.Chart:
    """
    Apply the shared house style to any chart: thin axis lines, soft grey
    labels, no border boxed around the plotting area.

    Every chart below finishes by calling this, which is what makes them all
    look like they belong to the same dashboard. Change it here and every
    chart changes together.
    """
    return (
        chart.properties(height=height, background=colors["surface"])
        # strokeWidth=0 removes the rectangle Altair would otherwise draw
        # around the plot area.
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor=colors["muted"], titleColor=colors["ink_soft"],
            labelFont=FONT, titleFont=FONT, labelFontSize=11, titleFontSize=11,
            gridColor=colors["grid"], gridWidth=1,
            domainColor=colors["axis"], domainWidth=1,
            tickColor=colors["axis"], tickSize=4, labelPadding=4,
        )
        .configure_legend(
            labelColor=colors["ink_soft"], titleColor=colors["ink_soft"],
            labelFont=FONT, titleFont=FONT, labelFontSize=11, titleFontSize=11,
            orient="top", direction="horizontal", offset=6,
        )
    )


def ranked_bar_chart(df, cat, val, cat_title, val_title, value_format,
                     tooltip_title, colors) -> alt.Chart:
    """
    A bar chart lying on its side: one measurement compared across categories.

        df            the data
        cat           the column with the category name, e.g. "tenant_id"
        val           the column with the number, e.g. "total_cost_usd"
        value_format  how to print that number, e.g. "$,.4f"

    Bars are sorted longest-first, and each one has its value written at the
    tip. That means you can read the actual number without tracing the bar
    back down to the axis and squinting.

    Sideways rather than upright because category names are words, and words
    fit far better along the side than crammed under vertical bars.
    """
    # Stretch the axis 28% past the longest bar. Without that headroom, the
    # number written at the tip of the longest bar gets cut off by the edge.
    top = float(df[val].max() or 0)
    scale = alt.Scale(domain=[0, top * 1.28 if top > 0 else 1.0], nice=False)

    # Add the call count to the hover box, but only when the data has one.
    extra = ([alt.Tooltip("call_count:Q", title="Calls", format=",")]
             if "call_count" in df.columns else [])

    # `base` holds the shared description. The bars and the text labels are
    # then both built FROM it and layered together with `+`. Doing it this way
    # guarantees the labels sit exactly where the bars end - they're using the
    # same positioning rules, not a copy that could drift out of step.
    base = alt.Chart(df).encode(
        # sort="-x" means sort by the x value, descending - longest at the top.
        y=alt.Y(f"{cat}:N", sort="-x", title=None,
                # labelLimit cuts off very long names instead of letting them
                # squash the chart into nothing.
                axis=alt.Axis(labelLimit=190, labelPadding=8)),
        x=alt.X(f"{val}:Q", title=val_title, scale=scale,
                axis=alt.Axis(format=value_format, grid=True, tickCount=5,
                              labelOverlap=True)),
        tooltip=[alt.Tooltip(f"{cat}:N", title=cat_title),
                 alt.Tooltip(f"{val}:Q", title=tooltip_title,
                             format=value_format)] + extra,
    )
    bars = base.mark_bar(size=16, cornerRadiusEnd=4, color=colors["s1"])
    labels = base.mark_text(align="left", dx=8, baseline="middle",
                            color=colors["ink_soft"], font=FONT, fontSize=11
                            ).encode(text=alt.Text(f"{val}:Q",
                                                   format=value_format))

    # Grow the height with the number of bars, so rows never squash together.
    # max(120, ...) stops a one-bar chart from being comically short.
    return style_chart(bars + labels, max(120, 46 * len(df)), colors)


def latency_dumbbell_chart(lp: pd.DataFrame, colors: dict) -> alt.Chart:
    """
    How spread out each model's response times are, from p50 to p99.

    WHAT THOSE NUMBERS MEAN
    p50 = 400ms means half your calls finished in under 400ms.
    p99 = 1800ms means 99 out of 100 finished in under 1800ms, and the
    unlucky 1 in 100 took even longer than that.

    The gap between them is the point. A small gap means consistent
    performance; a huge gap means most calls are fine but some users are
    occasionally left waiting - and those are the ones who complain.

    This is drawn as a "dumbbell": a grey bar running from p50 to p99, with a
    coloured dot marking each of the three percentiles. Chosen over grouped
    bars because it reads just as well with one model as with ten, whereas
    grouped bars collapse into thin unreadable stripes when there are few.
    """
    # melt() reshapes the table. It goes from WIDE (one row per model, with
    # three separate percentile columns) to LONG (one row per model AND
    # percentile). Altair needs the long shape, because you can only map a
    # COLUMN to colour - and we want colour to mean "which percentile".
    long = lp.melt(id_vars=["provider", "model", "call_count"],
                   value_vars=["p50_ms", "p90_ms", "p99_ms"],
                   var_name="percentile", value_name="ms")
    # Tidy the labels: "p50_ms" becomes just "p50".
    long["percentile"] = long["percentile"].str.replace("_ms", "", regex=False)

    top = float(long["ms"].max() or 0)
    scale = alt.Scale(domain=[0, top * 1.12 if top > 0 else 1.0], nice=False)
    # Defined once and reused by both layers, so they line up exactly.
    y = alt.Y("model:N", title=None,
              axis=alt.Axis(labelLimit=200, labelPadding=8))

    # mark_rule with both x AND x2 draws a line FROM one value TO another.
    # That's the grey bar connecting p50 to p99.
    track = alt.Chart(lp).mark_rule(color=colors["grid"], strokeWidth=3).encode(
        y=y,
        x=alt.X("p50_ms:Q", title="Latency (ms)", scale=scale,
                axis=alt.Axis(format=",.0f", grid=True, tickCount=6)),
        x2="p99_ms:Q",
    )
    dots = alt.Chart(long).mark_point(
        # A ring of background colour around each dot, so overlapping dots
        # still read as separate.
        size=115, filled=True, stroke=colors["surface"], strokeWidth=2,
    ).encode(
        y=y, x=alt.X("ms:Q", scale=scale),
        color=alt.Color("percentile:N",
                        # A FIXED list of the three names. Without this, if a
                        # filter removed one percentile, the other two would
                        # be re-coloured - so p50 would silently change colour
                        # between two views of the same dashboard.
                        scale=alt.Scale(domain=["p50", "p90", "p99"],
                                        range=[colors["s1"], colors["s2"],
                                               colors["s3"]]),
                        legend=alt.Legend(title=None, symbolType="circle")),
        tooltip=[alt.Tooltip("model:N", title="Model"),
                 alt.Tooltip("percentile:N", title="Percentile"),
                 alt.Tooltip("ms:Q", title="Latency (ms)", format=",.1f"),
                 alt.Tooltip("call_count:Q", title="Calls", format=",")],
    )
    # nunique() counts how many DIFFERENT models there are, which sets the height.
    return style_chart(track + dots,
                       max(150, 58 * lp["model"].nunique()), colors)


def daily_chart(daily, field, y_title, value_format, colors):
    """
    One measurement per day. Used for both daily spend and daily call count.

    The SHAPE changes depending on how much data there is:

      14 days or fewer -> upright bars. Separate blocks are easier to compare
                          one against another, and a line drawn through two
                          points doesn't look like a trend, it looks like a
                          mistake.
      more than that   -> a line, which is what a trend actually looks like.
    """
    # .copy() because we're about to add a column, and changing the caller's
    # data behind their back causes very confusing bugs.
    d = daily.copy()
    top = float(d[field].max() or 0)

    # The up-and-down axis and the hover box are the same either way, so
    # they're defined once here and used by both versions below.
    y = alt.Y(f"{field}:Q", title=y_title,
              scale=alt.Scale(domain=[0, top * 1.18 if top > 0 else 1.0],
                              nice=False),
              axis=alt.Axis(format=value_format, grid=True, tickCount=5))
    tooltip = [alt.Tooltip("date:T", title="Date", format="%d %b %Y"),
               alt.Tooltip("cost_usd:Q", title="Cost", format="$,.6f"),
               alt.Tooltip("calls:Q", title="Calls", format=",")]

    if len(d) <= 14:
        # :O treats the days as separate labelled categories, so they sit
        # evenly spaced with exactly one label each. Using a real date axis
        # here would invent extra tick marks in between the bars.
        d["day"] = d["date"].dt.strftime("%d %b")
        x = alt.X("day:O", sort=None, title=None,
                  axis=alt.Axis(grid=False, labelAngle=0))
        bars = alt.Chart(d).mark_bar(
            size=30, cornerRadiusEnd=4, color=colors["s1"],
        ).encode(x=x, y=y, tooltip=tooltip)
        # dy=-9 nudges each label 9 pixels up, so it floats just above its bar
        # instead of sitting on top of it.
        labels = alt.Chart(d).mark_text(
            dy=-9, color=colors["ink_soft"], font=FONT, fontSize=11,
        ).encode(x=x, y=f"{field}:Q",
                 text=alt.Text(f"{field}:Q", format=value_format))
        return style_chart(bars + labels, 250, colors)

    # Enough days for a trend. Draw a line, put a dot on each day, and label
    # only the LAST value - so the current number is readable at a glance
    # without labelling all thirty days into an unreadable mess.
    base = alt.Chart(d).encode(
        x=alt.X("date:T", title=None,
                # labelOverlap=True automatically hides labels that would
                # collide with each other.
                axis=alt.Axis(format="%d %b", grid=False, labelOverlap=True,
                              tickCount=8)),
        y=y, tooltip=tooltip)
    # interpolate="monotone" gives a gently curved line that never wiggles
    # above or below the real data points.
    line = base.mark_line(color=colors["s1"], strokeWidth=2,
                          interpolate="monotone")
    dots = base.mark_point(size=64, filled=True, color=colors["s1"],
                           stroke=colors["surface"], strokeWidth=2)
    # d.tail(1) is just the final row, which is the only one we label.
    end = (alt.Chart(d.tail(1))
           .mark_text(align="right", dy=-14, color=colors["ink_soft"],
                      font=FONT, fontSize=11, fontWeight=600)
           .encode(x="date:T", y=f"{field}:Q",
                   text=alt.Text(f"{field}:Q", format=value_format)))
    return style_chart(line + dots + end, 250, colors)


def histogram_chart(df, field, x_title, value_format, colors,
                    bins=18) -> alt.Chart:
    """
    A histogram: how one measurement is spread across all the calls.

    Each bar covers a RANGE of values, and its height is how many calls landed
    in that range. So "12 calls took between 400 and 500 milliseconds".

    This answers a question an average simply cannot: are most of my calls
    fast with a few disasters, or are they all equally mediocre? Both of those
    can produce exactly the same average.

    Altair does the grouping itself via bin=alt.Bin(maxbins=...), and count()
    on the up-and-down axis counts how many landed in each group.
    """
    chart = alt.Chart(df.dropna(subset=[field])).mark_bar(
        # binSpacing=2 leaves a small gap between bars, so you can see where
        # one range ends and the next begins.
        binSpacing=2, cornerRadiusEnd=3, color=colors["s1"],
    ).encode(
        x=alt.X(f"{field}:Q", bin=alt.Bin(maxbins=bins), title=x_title,
                axis=alt.Axis(format=value_format, grid=False)),
        y=alt.Y("count():Q", title="Calls",
                # tickMinStep=1 stops the axis offering "2.5 calls", which is
                # not a thing.
                axis=alt.Axis(grid=True, tickMinStep=1, format=",.0f")),
        tooltip=[alt.Tooltip(f"{field}:Q", bin=alt.Bin(maxbins=bins),
                             title=x_title, format=value_format),
                 alt.Tooltip("count():Q", title="Calls", format=",")],
    )
    return style_chart(chart, 230, colors)


def quality_score_chart(df: pd.DataFrame, colors: dict) -> alt.Chart:
    """
    How many calls got each quality score.

    Deliberately NOT a histogram, even though it looks like one.

    A quality score is a whole number from 1 to 10 - there is no such thing as
    8.5. Letting the chart library group these into ranges produces fractional
    boundaries whose rounded-off labels repeat, so the axis ends up reading
    "9 9 9 9 10 10 10 10". Counting each score as its own category is the
    honest way to draw data that only comes in whole numbers.
    """
    d = df.dropna(subset=["quality_score"]).copy()
    # Force to whole numbers. They arrive from the database as decimals
    # because the column allows empty values.
    d["quality_score"] = d["quality_score"].astype(int)
    # Group by score and count how many are in each group.
    agg = d.groupby("quality_score").size().reset_index(name="calls")

    chart = alt.Chart(agg).mark_bar(
        size=22, cornerRadiusEnd=3, color=colors["s1"],
    ).encode(
        x=alt.X("quality_score:O", title="Quality (1-10)",
                # A fixed 1-to-10 range, so a score nobody got shows up as a
                # visible gap rather than quietly vanishing off the axis. That
                # gap is information: it tells you nothing scored a 3.
                scale=alt.Scale(domain=list(range(1, 11))),
                axis=alt.Axis(labelAngle=0, grid=False)),
        y=alt.Y("calls:Q", title="Calls",
                axis=alt.Axis(grid=True, tickMinStep=1, format=",.0f")),
        tooltip=[alt.Tooltip("quality_score:O", title="Score"),
                 alt.Tooltip("calls:Q", title="Calls", format=",")],
    )
    return style_chart(chart, 230, colors)


def tokens_by_use_case_chart(df: pd.DataFrame, colors: dict) -> alt.Chart:
    """
    Text going IN versus text coming OUT, per feature.

    Two bars side by side, NOT stacked one on top of the other. Side by side
    lets you compare the two directly; stacking would only show their total
    and hide the split.

    That split matters because AI companies usually charge different rates for
    text you send and text they generate - so two features with identical
    totals can cost quite different amounts.
    """
    long = df.melt(id_vars=["use_case"],
                   value_vars=["input_tokens", "output_tokens"],
                   var_name="kind", value_name="tokens")
    # Swap the raw database column names for words a person would want to read.
    long["kind"] = long["kind"].map({"input_tokens": "Input",
                                     "output_tokens": "Output"})

    chart = alt.Chart(long).mark_bar(size=13, cornerRadiusEnd=3).encode(
        y=alt.Y("use_case:N", title=None,
                axis=alt.Axis(labelLimit=190, labelPadding=8)),
        # yOffset is the bit that splits one bar into a pair sitting side by
        # side, rather than one drawn on top of the other.
        yOffset=alt.YOffset("kind:N", sort=["Input", "Output"]),
        x=alt.X("tokens:Q", title="Tokens",
                axis=alt.Axis(format=",.0f", grid=True, tickCount=5)),
        color=alt.Color("kind:N",
                        scale=alt.Scale(domain=["Input", "Output"],
                                        range=[colors["s1"], colors["s2"]]),
                        legend=alt.Legend(title=None)),
        tooltip=[alt.Tooltip("use_case:N", title="Use case"),
                 alt.Tooltip("kind:N", title="Direction"),
                 alt.Tooltip("tokens:Q", title="Tokens", format=",")],
    )
    return style_chart(chart, max(170, 70 * df["use_case"].nunique()), colors)


def cost_vs_quality_chart(df: pd.DataFrame, colors: dict) -> alt.Chart:
    """
    A scatter plot: cost across the bottom, quality up the side, one dot per call.

    Read it as "up and to the LEFT is better" - high quality for little money.
    A dot in the bottom right cost a lot AND wasn't even good, and is worth
    clicking on to find out what happened.

    One colour for everything. Colouring the dots by model or customer would
    need a much bigger set of colours to stay tellable-apart, and with enough
    dots they'd blur together anyway.
    """
    # Drop rows missing either value - a dot needs both to have a position.
    d = df.dropna(subset=["cost_usd", "quality_score"])
    chart = alt.Chart(d).mark_point(
        # Slightly see-through, so a cluster of dots in one spot still shows
        # as denser rather than as one solid blob.
        size=110, filled=True, color=colors["s1"], opacity=0.85,
        stroke=colors["surface"], strokeWidth=2,
    ).encode(
        x=alt.X("cost_usd:Q", title="Cost (USD)",
                axis=alt.Axis(format="$,.5f", grid=True, tickCount=5)),
        y=alt.Y("quality_score:Q", title="Quality (1-10)",
                # Up to 10.5 rather than 10, so a dot at a perfect 10 isn't
                # sliced in half by the top edge of the chart.
                scale=alt.Scale(domain=[0, 10.5], nice=False),
                axis=alt.Axis(grid=True, tickCount=6)),
        tooltip=[alt.Tooltip("model:N", title="Model"),
                 alt.Tooltip("use_case:N", title="Use case"),
                 alt.Tooltip("cost_usd:Q", title="Cost", format="$,.6f"),
                 alt.Tooltip("quality_score:Q", title="Quality"),
                 alt.Tooltip("latency_ms:Q", title="Latency (ms)",
                             format=",.0f")],
    )
    return style_chart(chart, 260, colors)


def calls_by_hour_chart(df: pd.DataFrame, colors: dict) -> alt.Chart:
    """
    How many calls happen in each hour of the day, 0 to 23 (in UTC).

    Useful for spotting when your traffic peaks - which is exactly when rate
    limits and slow responses tend to bite.
    """
    d = df.dropna(subset=["timestamp"]).copy()
    # .dt.hour pulls just the hour number out of each full date and time.
    d["hour"] = d["timestamp"].dt.hour

    # reindex(range(24)) forces all 24 hours onto the chart, filling quiet
    # hours with 0. Without it, an hour with no calls would simply be absent,
    # and the hours either side would sit next to each other as though nothing
    # were missing - which reads as a completely different pattern.
    agg = (d.groupby("hour")
           .agg(calls=("trace_id", "count"), cost_usd=("cost_usd", "sum"))
           .reindex(range(24), fill_value=0).reset_index())

    chart = alt.Chart(agg).mark_bar(
        size=14, cornerRadiusEnd=3, color=colors["s1"],
    ).encode(
        x=alt.X("hour:O", title="Hour of day (UTC)",
                # range(0, 24, 2) counts 0, 2, 4... so we label every other
                # hour. All 24 labels would overlap into a smudge.
                axis=alt.Axis(grid=False, labelAngle=0,
                              values=list(range(0, 24, 2)))),
        y=alt.Y("calls:Q", title="Calls",
                axis=alt.Axis(grid=True, tickMinStep=1, format=",.0f")),
        tooltip=[alt.Tooltip("hour:O", title="Hour"),
                 alt.Tooltip("calls:Q", title="Calls", format=","),
                 alt.Tooltip("cost_usd:Q", title="Cost", format="$,.6f")],
    )
    return style_chart(chart, 210, colors)


# ---------------------------------------------------------------------------
# 5. The page itself
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Draw the whole dashboard, top to bottom.

    The order of events:
        set the page up  ->  find the database  ->  check it has data
        ->  draw the filters  ->  load the matching records
        ->  draw the number boxes  ->  draw the seven tabs

    Notice how many times this function does a check and then `return`s early
    with a helpful message. That's on purpose. Each `return` stops the page
    at the exact point something is missing and says what to do about it,
    instead of pressing on and failing later with a confusing error about a
    column that doesn't exist.
    """
    st.set_page_config(page_title="LLM Cost Monitor", page_icon="📊",
                       layout="wide", initial_sidebar_state="collapsed")
    colors, mode = theme_colors()
    st.markdown(page_css(colors), unsafe_allow_html=True)

    # --- The header ---
    # Only show the little name pill if a project name was actually set.
    chip = (f'<span class="chip">{html.escape(PROJECT_NAME)}</span>'
            if PROJECT_NAME else "")
    st.markdown(f'<div class="hdr"><h1>LLM Cost Monitor</h1>{chip}</div>',
                unsafe_allow_html=True)
    st.markdown('<p class="lede">Cost, latency and quality for every LLM call '
                'this project has recorded.</p>', unsafe_allow_html=True)

    # --- Which database are we looking at? ---
    try:
        database = resolve_database()
    except ConfigError as exc:
        # str(exc) is the helpful message settings.py wrote, which already
        # explains exactly which setting is missing and how to add it.
        st.error(str(exc))
        return

    # The sidebar holds the data source and the buttons. The FILTERS live above
    # the charts instead, right next to the things they change - hiding them in
    # a sidebar makes people forget a filter is on and misread the numbers.
    with st.sidebar:
        st.markdown("### Data source")
        st.code(database, language=None)
        if st.button("Refresh", width="stretch"):
            # Throw away the remembered answers and redraw, so the next draw
            # goes to the database instead of returning data up to 15 seconds
            # old. This is the escape hatch from that caching.
            st.cache_data.clear()
            st.rerun()
        st.markdown("### Appearance")
        st.caption(
            f"Currently **{mode}**. Switch it from the ⋮ menu, top right → "
            "Settings → Appearance. Charts follow whichever you pick."
        )

    # --- Is there anything to show? Three checks, three different messages ---
    try:
        db_info = get_db_info(database)
    except ClickHouseUnavailable as exc:
        st.error(str(exc))
        return

    # Check 1: the database has never been created at all.
    if db_info is None:
        st.warning(
            "This project has no trace database yet. Make at least one call through "
            "`LLMMonitor` and save it with `TraceRepository`, then refresh."
        )
        return

    with st.sidebar:
        st.caption(f"Server {db_info['server']}")
        # The :, adds thousands separators, so 12345 shows as "12,345".
        st.caption(f"{db_info['rows']:,} traces")
        if db_info["last_trace"] is not None:
            st.caption(f"Latest {db_info['last_trace']:%d %b %Y, %H:%M:%S}")

    # Check 2: the database is there, but nothing has been saved into it.
    if not db_info["rows"]:
        st.info("The database exists but holds no traces yet. Save a call with "
                "`TraceRepository.save(...)`, then refresh.")
        return

    # --- The filters ---
    # The options come from the DATABASE, not from data we've already loaded.
    # They have to: the data we load is the RESULT of these filters, so it
    # can't also be the source of their options.
    (provider_options, tenant_options, use_case_options,
     query_type_options, bounds) = get_filter_options(database)
    min_date, max_date = bounds

    # st.columns splits a row into side-by-side sections. [1, 2] means the
    # second one gets twice the width of the first.
    with st.container(border=True):
        f1, f2 = st.columns([1, 2], gap="medium")
        with f1:
            date_range = ()
            if min_date is not None and max_date is not None:
                date_range = st.date_input(
                    "Date range", value=(min_date.date(), max_date.date()),
                    min_value=min_date.date(), max_value=max_date.date())
        with f2:
            # default=all the options means nothing is hidden until you
            # actively choose to hide it. A dashboard that silently starts up
            # already filtered is how people misread their own numbers.
            providers = st.pills("Provider", provider_options,
                                 selection_mode="multi",
                                 default=provider_options)
        f3, f4 = st.columns(2, gap="medium")
        with f3:
            tenants = st.pills("Tenant", tenant_options,
                               selection_mode="multi", default=tenant_options)
        with f4:
            use_cases = st.pills("Use case", use_case_options,
                                 selection_mode="multi",
                                 default=use_case_options)
        if query_type_options:
            # Only shown once something has actually been classified. An empty
            # row of buttons would just be clutter for anyone who hasn't
            # switched the classification feature on.
            query_types = st.pills("Query type", query_type_options,
                                   selection_mode="multi",
                                   default=query_type_options)
        else:
            query_types = None

    # Turn the chosen dates into timestamps for the query.
    #
    # The end date gets a whole day ADDED to it. That's because the query uses
    # "less than end", so without the extra day, picking the 20th would stop
    # at midnight on the morning of the 20th and silently miss that whole day.
    start_ts = end_ts = None
    if len(date_range) == 2:
        start_ts = pd.Timestamp(date_range[0]).to_pydatetime()
        end_ts = (pd.Timestamp(date_range[1])
                  + pd.Timedelta(days=1)).to_pydatetime()

    # --- Load the data, already narrowed down by the database ---
    try:
        data = load_data(database, start_ts, end_ts, providers, tenants,
                         use_cases, query_types)
    except ClickHouseUnavailable as exc:
        # The specific, explainable failure first...
        st.error(str(exc))
        return
    except Exception as exc:
        # ...then a catch-all, so an unexpected problem still shows as a
        # message on the page rather than a blank screen.
        st.error(f"Could not read the trace database: {exc}")
        return

    traces, cost_by_tenant, cost_by_use_case, latency_percentiles = data
    # `filtered` is just a clearer name to use for the rest of this function.
    # The database already applied all four filters, so nothing below narrows
    # it down any further.
    filtered = traces

    # Check 3: the filters matched nothing at all.
    if filtered.empty:
        st.warning("No data matches the selected filters.")
        return

    # --- The number boxes across the top ---
    # skipna=True means missing values are ignored. Without it, a single
    # missing cost would turn the whole total into "not a number".
    total_calls = len(filtered)
    total_cost = filtered["cost_usd"].sum(skipna=True)
    avg_latency = filtered["latency_ms"].mean(skipna=True)
    avg_quality = filtered["quality_score"].mean(skipna=True)
    total_in = filtered["input_tokens"].sum(skipna=True)
    total_out = filtered["output_tokens"].sum(skipna=True)
    avg_throughput = filtered["tokens_per_second"].mean(skipna=True)
    # notna() is True where there IS an error message, and a row only has one
    # if that call failed. .sum() then counts the Trues.
    failures = int(filtered["error"].notna().sum())
    # More than one attempt means it had to be retried. fillna(1) covers old
    # rows saved before we started recording the attempt count.
    retried = int((filtered["attempts_used"].fillna(1) > 1).sum())
    # "length" means the answer hit the size limit and got cut off mid-sentence.
    truncated = int((filtered["finish_reason"] == "length").sum())
    # normalize() throws away the time and keeps only the date, so nunique()
    # then counts distinct DAYS rather than distinct moments.
    span_days = filtered["timestamp"].dt.normalize().nunique()

    render_tiles([
        metric_tile("Calls", f"{total_calls:,}",
                    f"{failures} failed" if failures else "all succeeded",
                    "bad" if failures else "good"),
        metric_tile("Total cost", format_currency(total_cost),
                    f"{format_currency(total_cost / total_calls)} per call"),
        metric_tile("Avg latency", f"{avg_latency:,.0f} ms",
                    f"{filtered['latency_ms'].max():,.0f} ms slowest"),
        # pd.notna checks there IS an average. There won't be if quality
        # marking was switched off for every one of these calls.
        metric_tile("Avg quality",
                    f"{avg_quality:,.1f}" if pd.notna(avg_quality) else "N/A",
                    "judge score, 1-10"),
        metric_tile("Tokens", compact_number(total_in + total_out),
                    f"{compact_number(total_in)} in · "
                    f"{compact_number(total_out)} out"),
        metric_tile("Throughput",
                    f"{avg_throughput:,.0f} tok/s"
                    if pd.notna(avg_throughput) else "N/A",
                    "output only"),
    ])
    render_tiles([
        metric_tile("Retries", f"{retried:,}",
                    "none needed" if not retried else "calls retried",
                    "good" if not retried else "bad"),
        metric_tile("Truncated", f"{truncated:,}",
                    "hit the token cap" if truncated else "none cut off",
                    "bad" if truncated else "good"),
        metric_tile("Models", f"{filtered['model'].nunique():,}",
                    f"{filtered['provider'].nunique()} provider(s)"),
        metric_tile("Tenants", f"{filtered['tenant_id'].nunique():,}",
                    f"{filtered['use_case'].nunique()} use case(s)"),
        # max(span_days, 1) guards the division - it can never be 0, because
        # we already checked there's at least one row.
        metric_tile("Active days", f"{span_days:,}",
                    f"{total_calls / max(span_days, 1):,.1f} calls/day"),
        metric_tile("Cost efficiency",
                    f"{filtered['cost_efficiency'].mean(skipna=True):,.0f}"
                    if filtered["cost_efficiency"].notna().any() else "N/A",
                    "quality per $"),
    ])

    # Group everything by day ONCE, here, because both the Cost tab and the
    # Activity tab need it. Working it out twice would risk the two tabs
    # quietly disagreeing with each other.
    ts = filtered.dropna(subset=["timestamp"]).copy()
    # Start with an empty table, so the tabs below always have something with
    # the right columns to work with even if there are no dated rows.
    daily = pd.DataFrame(columns=["date", "cost_usd", "calls"])
    if not ts.empty:
        # floor("D") rounds every timestamp DOWN to midnight of its own day,
        # which is what turns individual moments into daily groups.
        ts["date"] = ts["timestamp"].dt.floor("D")
        daily = (ts.groupby("date")
                 .agg(cost_usd=("cost_usd", "sum"), calls=("trace_id", "count"))
                 .reset_index().sort_values("date"))

    tab_cost, tab_lat, tab_qual, tab_tok, tab_time, tab_calls, tab_sql = st.tabs(
        ["Cost", "Latency", "Quality", "Tokens", "Activity", "Calls", "SQL"]
    )

    # ---- Cost tab: where the money went ----
    with tab_cost:
        section_heading("Cost by day",
                        "Total spend per day for the current filter.")
        if daily.empty:
            st.info("No timestamped calls in this selection.")
        else:
            st.altair_chart(
                daily_chart(daily, "cost_usd", "Cost (USD)", "$,.4f", colors),
                width="stretch")

        c1, c2 = st.columns(2, gap="large")
        with c1:
            section_heading("Cost by tenant",
                            "Which customer the spend belongs to.")
            # This summary table covers ALL rows, because it comes from a
            # saved view rather than the filtered query. So narrow it here to
            # match whichever customers are currently selected.
            tenant_costs = (cost_by_tenant[cost_by_tenant["tenant_id"].isin(tenants)]
                            if tenants else cost_by_tenant)
            if tenant_costs.empty:
                st.info("No tenant data. Pass `tenant_id=` to `monitor.call()`.")
            else:
                st.altair_chart(
                    ranked_bar_chart(tenant_costs, "tenant_id",
                                     "total_cost_usd", "Tenant",
                                     "Total cost (USD)", "$,.4f",
                                     "Total cost", colors), width="stretch")
                st.dataframe(
                    tenant_costs.sort_values("total_cost_usd", ascending=False),
                    width="stretch", hide_index=True,
                    # column_config renames the raw database columns into
                    # readable headings and formats the numbers. Without it
                    # you'd see "total_cost_usd" and "0.004213000000001".
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
            section_heading("Cost by use case",
                            "Which feature the spend belongs to.")
            uc_costs = (cost_by_use_case[cost_by_use_case["use_case"].isin(use_cases)]
                        if use_cases else cost_by_use_case)
            if uc_costs.empty:
                st.info("No use-case data. Pass `use_case=` to `monitor.call()`.")
            else:
                st.altair_chart(
                    ranked_bar_chart(uc_costs, "use_case", "total_cost_usd",
                                     "Use case", "Total cost (USD)", "$,.4f",
                                     "Total cost", colors), width="stretch")
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
        # With only one model there's nothing to compare it against, and a bar
        # chart with a single bar tells you nothing the boxes at the top
        # haven't already said. Show the number on its own instead.
        if len(by_model) < 2:
            row = by_model.iloc[0]
            section_heading("Cost by model",
                            "Only one model in this selection, so there is "
                            "nothing to compare.")
            render_tiles([metric_tile(str(row["model"]),
                                      format_currency(row["total_cost_usd"]),
                                      f"{int(row['call_count']):,} calls")])
        else:
            section_heading("Cost by model",
                            "Same spend, grouped by which model answered.")
            st.altair_chart(
                ranked_bar_chart(by_model, "model", "total_cost_usd",
                                 "Model", "Total cost (USD)", "$,.4f",
                                 "Total cost", colors), width="stretch")

        section_heading("Most expensive calls", "The ten calls that cost the most.")
        st.dataframe(
            # nlargest(10, "cost_usd") picks the 10 rows with the highest cost.
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

    # ---- Latency tab: how fast the calls were ----
    with tab_lat:
        section_heading("Latency percentiles by model",
                        "p50 is the typical call; p99 is the tail users "
                        "complain about.")
        if latency_percentiles.empty:
            st.info("Latency percentile analytics are not available yet.")
        else:
            lp = (latency_percentiles[latency_percentiles["provider"].isin(providers)]
                  if providers else latency_percentiles)
            if lp.empty:
                st.info("No latency data for the selected providers.")
            else:
                st.altair_chart(latency_dumbbell_chart(lp, colors),
                                width="stretch")
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
                # An honesty warning. "99 out of 100 calls" computed from only
                # 15 calls is not really a measurement, and quoting it in a
                # meeting as though it were would be misleading.
                if lp["call_count"].max() < 20:
                    st.caption("Fewer than 20 calls per model — these percentiles "
                               "show the mechanism, not a reliable distribution.")

        l1, l2 = st.columns(2, gap="large")
        with l1:
            section_heading("Latency distribution",
                            "How the individual calls spread out.")
            st.altair_chart(
                histogram_chart(filtered, "latency_ms", "Latency (ms)",
                                ",.0f", colors), width="stretch")
        with l2:
            section_heading("Throughput by model",
                            "Output tokens generated per second.")
            tp = (filtered.groupby("model", dropna=True)
                  .agg(tokens_per_second=("tokens_per_second", "mean"),
                       call_count=("trace_id", "count")).reset_index())
            st.altair_chart(
                ranked_bar_chart(tp, "model", "tokens_per_second",
                                 "Model", "Tokens / second", ",.0f",
                                 "Avg throughput", colors), width="stretch")

        section_heading("Slowest calls", "The ten calls that took the longest.")
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

    # ---- Quality tab: how good the answers were ----
    with tab_qual:
        # notna().sum() counts how many rows have a score. Zero means quality
        # marking was switched off for everything in this selection.
        if filtered["quality_score"].notna().sum() == 0:
            st.info("No quality scores in this selection. Quality scoring is on "
                    "by default; pass `score_response_quality=False` to skip it.")
        else:
            q1, q2 = st.columns(2, gap="large")
            with q1:
                section_heading("Quality distribution",
                                "How the judge scored these responses.")
                st.altair_chart(quality_score_chart(filtered, colors),
                                width="stretch")
            with q2:
                section_heading("Cost against quality",
                                "One dot per call. Up and to the left is better.")
                st.altair_chart(cost_vs_quality_chart(filtered, colors),
                                width="stretch")

            section_heading("Quality by use case",
                            "Average judge score per feature.")
            qbu = (filtered.dropna(subset=["quality_score"])
                   .groupby("use_case", dropna=True)
                   .agg(quality_score=("quality_score", "mean"),
                        call_count=("trace_id", "count")).reset_index())
            if not qbu.empty:
                st.altair_chart(
                    ranked_bar_chart(qbu, "use_case", "quality_score",
                                     "Use case", "Avg quality (1-10)",
                                     ",.1f", "Avg quality", colors),
                    width="stretch")

        section_heading("Models by query type",
                        "Same job, different models. High quality per dollar "
                        "is the cheaper model doing comparable work.")
        mbq = load_model_by_query_type(database)
        if mbq.empty:
            st.info("No classified calls yet. Turn classification on with "
                    "`obs.init(..., classify_query=True)` — it shares the judge "
                    "call with quality scoring, so it adds no API calls if "
                    "scoring is already enabled.")
        else:
            st.dataframe(
                # Sort by job type, and within each job put the best value
                # first. ascending=[True, False] applies a different direction
                # to each of the two columns.
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

        section_heading("Finish reasons",
                        "`stop` finished naturally; `length` was cut off.")
        fr = (filtered.groupby("finish_reason", dropna=False)
              .agg(call_count=("trace_id", "count")).reset_index())
        # A failed call never got a finish reason at all. Give that case a
        # readable label instead of showing a mysterious blank row.
        fr["finish_reason"] = fr["finish_reason"].fillna("(none — call failed)")
        st.altair_chart(
            ranked_bar_chart(fr, "finish_reason", "call_count",
                             "Finish reason", "Calls", ",.0f", "Calls", colors),
            width="stretch")

    # ---- Tokens tab: what the calls were made of ----
    with tab_tok:
        section_heading("Input vs output tokens by use case",
                        "Long prompts and long answers cost differently.")
        tok = (filtered.groupby("use_case", dropna=True)
               .agg(input_tokens=("input_tokens", "sum"),
                    output_tokens=("output_tokens", "sum")).reset_index())
        if tok.empty:
            st.info("No use-case data to group tokens by.")
        else:
            st.altair_chart(tokens_by_use_case_chart(tok, colors),
                            width="stretch")

        t1, t2 = st.columns(2, gap="large")
        with t1:
            section_heading("Tokens per call", "Distribution across the selection.")
            st.altair_chart(
                histogram_chart(filtered, "total_tokens", "Tokens per call",
                                ",.0f", colors), width="stretch")
        with t2:
            section_heading("Avg tokens per call by model", "")
            tpm = (filtered.groupby("model", dropna=True)
                   .agg(total_tokens=("total_tokens", "mean"),
                        call_count=("trace_id", "count")).reset_index())
            st.altair_chart(
                ranked_bar_chart(tpm, "model", "total_tokens", "Model",
                                 "Avg tokens", ",.0f", "Avg tokens", colors),
                width="stretch")

    # ---- Activity tab: how much traffic, and when. Money is in the Cost tab ----
    with tab_time:
        if daily.empty:
            st.info("No timestamped calls in this selection.")
        else:
            section_heading("Calls by day", "How much traffic this project sent.")
            # Same chart function as the Cost tab, just pointed at a different
            # column. That's the payoff for keeping the chart builders general.
            st.altair_chart(
                daily_chart(daily, "calls", "Calls", ",.0f", colors),
                width="stretch")

            section_heading("Calls by hour of day",
                            "When this project talks to the LLM.")
            st.altair_chart(calls_by_hour_chart(filtered, colors),
                            width="stretch")

            section_heading("Daily totals", "")
            st.dataframe(
                daily, width="stretch", hide_index=True,
                column_config={
                    "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                    "cost_usd": st.column_config.NumberColumn("Cost (USD)", format="$%.6f"),
                    "calls": st.column_config.NumberColumn("Calls", format="%d"),
                },
            )

    # ---- Calls tab: the raw rows, plus a download button ----
    with tab_calls:
        section_heading("Recent calls", "Newest first, capped at 200 rows.")
        # Capped at 200 because a web table holding tens of thousands of rows
        # is slow to draw and impossible to read anyway. The download below is
        # NOT capped, so nothing is actually being kept from you.
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

        # Downloads ALL the filtered rows, not just the 200 shown above.
        # .encode("utf-8") turns the text into the raw bytes a file needs.
        st.download_button(
            "Download these rows as CSV",
            filtered.to_csv(index=False).encode("utf-8"),
            f"llm_traces_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv")

    # ---- SQL tab: write your own query ----
    with tab_sql:
        section_heading("SQL explorer",
                        "Table: llm_traces · Views: v_cost_by_tenant, "
                        "v_cost_by_use_case, v_latency_percentiles, v_summary")
        query = st.text_area(
            "Query", height=110,
            value="SELECT * FROM llm_traces ORDER BY timestamp DESC LIMIT 50;")
        if st.button("Run query"):
            # Ignore an empty box, so clicking the button by accident does
            # nothing rather than throwing an error at you.
            if query.strip():
                repo = TraceRepository(database, create_schema=False)
                try:
                    result = repo.query_df(query)
                    st.dataframe(result, width="stretch", hide_index=True)
                    st.success(f"Returned {len(result):,} rows")
                except Exception as e:
                    # A typo in the SQL shows up as a message on the page
                    # rather than crashing the whole dashboard. This box is
                    # for experimenting, and experiments go wrong.
                    st.error(f"Query failed: {e}")
                finally:
                    repo.close()


if __name__ == "__main__":
    main()
