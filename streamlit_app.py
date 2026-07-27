import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from core.infrastructure.persistence import TraceRepository

DEFAULT_DB_PATHS = [
    os.environ.get("DUCKDB_PATH"),
    "/app/results/llm_traces.duckdb",
    "results/llm_traces.duckdb",
]


def get_default_db_path() -> str:
    for candidate in DEFAULT_DB_PATHS:
        if candidate and Path(candidate).exists():
            return candidate
    return os.environ.get("DUCKDB_PATH", "/app/results/llm_traces.duckdb")


@st.cache_data(ttl=15, show_spinner="Loading traces from DuckDB...")
def load_data(db_path: str):
    path = Path(db_path)
    if not path.exists():
        return None

    repo = TraceRepository(db_path)
    try:
        traces = repo.query_df("SELECT * FROM llm_traces")
        cost_by_tenant = repo.query_df("SELECT * FROM v_cost_by_tenant")
        cost_by_use_case = repo.query_df("SELECT * FROM v_cost_by_use_case")
        latency_percentiles = repo.query_df("SELECT * FROM v_latency_percentiles")
    finally:
        repo.close()

    traces["timestamp"] = pd.to_datetime(traces["timestamp"], errors="coerce")
    return traces, cost_by_tenant, cost_by_use_case, latency_percentiles


@st.cache_data(ttl=15)
def get_db_info(db_path: str):
    path = Path(db_path)
    if not path.exists():
        return None
    return {
        "path": str(path.resolve()),
        "last_modified": datetime.fromtimestamp(path.stat().st_mtime),
        "size_bytes": path.stat().st_size,
    }


def format_currency(value: float) -> str:
    return f"${value:,.4f}" if value is not None else "N/A"


def main() -> None:
    st.set_page_config(
        page_title="LLM Cost Monitor",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🤖 LLM Cost Monitor")
    st.markdown(
        "Explore persisted LLM telemetry and cost analytics from DuckDB. "
        "Run the monitor (`python main.py`) to generate data."
    )

    # Sidebar: Data source
    default_path = get_default_db_path()
    db_path = st.sidebar.text_input("DuckDB file path", value=default_path)

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with col2:
        if st.button("🗑️ Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.success("Cache cleared!")
            st.rerun()

    db_info = get_db_info(db_path)
    if db_info is None:
        st.warning(
            "No DuckDB file found at the configured path. Run the monitor first to generate `llm_traces.duckdb`, then refresh."
        )
        st.info(f"Configured path: `{db_path}`")
        return

    st.sidebar.markdown("### Data Source")
    st.sidebar.write(db_info["path"])
    st.sidebar.write(f"Last updated: {db_info['last_modified']:%Y-%m-%d %H:%M:%S}")
    st.sidebar.write(f"File size: {db_info['size_bytes']:,} bytes")

    # Load data
    data = load_data(db_path)
    if data is None:
        st.error("Unable to load dashboard data from DuckDB.")
        return

    traces, cost_by_tenant, cost_by_use_case, latency_percentiles = data

    # Sidebar: Filters
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Filters")

    # Date range filter
    min_date = traces["timestamp"].min()
    max_date = traces["timestamp"].max()
    if pd.notna(min_date) and pd.notna(max_date):
        date_range = st.sidebar.date_input(
            "Date range",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
        )
        if len(date_range) == 2:
            start_ts = pd.Timestamp(date_range[0])
            end_ts = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
            traces = traces[
                (traces["timestamp"] >= start_ts) & (traces["timestamp"] < end_ts)
            ]

    provider_options = sorted(traces["provider"].dropna().unique())
    use_case_options = sorted(traces["use_case"].dropna().unique())
    tenant_options = sorted(traces["tenant_id"].dropna().unique())

    providers = st.sidebar.multiselect("Providers", provider_options, default=provider_options)
    tenants = st.sidebar.multiselect("Tenants", tenant_options, default=tenant_options)
    use_cases = st.sidebar.multiselect("Use cases", use_case_options, default=use_case_options)

    # Apply filters
    filtered = traces.copy()
    if providers:
        filtered = filtered[filtered["provider"].isin(providers)]
    if tenants:
        filtered = filtered[filtered["tenant_id"].isin(tenants)]
    if use_cases:
        filtered = filtered[filtered["use_case"].isin(use_cases)]

    if filtered.empty:
        st.warning("No data matches the selected filters.")
        return

    # KPI Row
    total_calls = len(filtered)
    total_cost = filtered["cost_usd"].sum(skipna=True)
    avg_latency = filtered["latency_ms"].mean(skipna=True)
    avg_quality = filtered["quality_score"].mean(skipna=True)
    total_input = filtered["input_tokens"].sum(skipna=True)
    total_output = filtered["output_tokens"].sum(skipna=True)
    total_tokens = int(total_input + total_output)
    avg_throughput = filtered["tokens_per_second"].mean(skipna=True)

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total Calls", f"{total_calls:,}")
    col2.metric("Total Cost", format_currency(total_cost))
    col3.metric("Avg Latency", f"{avg_latency:,.1f} ms")
    col4.metric("Avg Quality", f"{avg_quality:,.1f}" if pd.notna(avg_quality) else "N/A")
    col5.metric("Total Tokens", f"{total_tokens:,}")
    col6.metric("Avg Throughput", f"{avg_throughput:,.1f} tok/s" if pd.notna(avg_throughput) else "N/A")

    st.markdown("---")

    # Analytics Views
    with st.expander("📊 Analytics Views", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Cost by Tenant")
            if not cost_by_tenant.empty:
                # Filter cost_by_tenant to match current filters
                tenant_costs = cost_by_tenant[cost_by_tenant["tenant_id"].isin(tenants)]
                if not tenant_costs.empty:
                    st.bar_chart(
                        tenant_costs.set_index("tenant_id")["total_cost_usd"].sort_values(ascending=False)
                    )
                    st.dataframe(
                        tenant_costs.sort_values("total_cost_usd", ascending=False),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No tenant data for selected filters.")
            else:
                st.info("No tenant-level data available.")

        with col2:
            st.subheader("Cost by Use Case")
            if not cost_by_use_case.empty:
                uc_costs = cost_by_use_case[cost_by_use_case["use_case"].isin(use_cases)]
                if not uc_costs.empty:
                    st.bar_chart(
                        uc_costs.set_index("use_case")["total_cost_usd"].sort_values(ascending=False)
                    )
                    st.dataframe(
                        uc_costs.sort_values("total_cost_usd", ascending=False),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No use case data for selected filters.")
            else:
                st.info("No use case analytics available.")

    st.markdown("---")

    # Latency Percentiles
    st.subheader("⏱️ Latency Percentiles")
    if not latency_percentiles.empty:
        lp = latency_percentiles[
            latency_percentiles["provider"].isin(providers)
        ].sort_values(["provider", "model"])
        st.dataframe(lp, use_container_width=True, hide_index=True)

        # Visualize p50/p90/p99
        if len(lp) > 0:
            chart_data = lp.set_index("model")[["p50_ms", "p90_ms", "p99_ms"]]
            st.bar_chart(chart_data)
    else:
        st.info("Latency percentile analytics are not available yet.")

    st.markdown("---")

    # Time Series: Cost over time
    st.subheader("📈 Cost Over Time")
    if not filtered.empty:
        ts = filtered.copy()
        ts["date"] = ts["timestamp"].dt.date
        daily_cost = ts.groupby("date")["cost_usd"].sum().reset_index()
        st.line_chart(daily_cost.set_index("date")["cost_usd"])
    else:
        st.info("No data for time series.")

    st.markdown("---")

    # Recent Calls Table
    st.subheader("📋 Recent Calls")
    recent = filtered.sort_values("timestamp", ascending=False).head(50)
    display_cols = [
        "timestamp",
        "tenant_id",
        "use_case",
        "provider",
        "model",
        "cost_usd",
        "latency_ms",
        "quality_score",
        "finish_reason",
        "total_tokens",
    ]
    st.dataframe(
        recent[display_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "timestamp": st.column_config.DatetimeColumn("Time", format="YYYY-MM-DD HH:mm:ss"),
            "cost_usd": st.column_config.NumberColumn("Cost (USD)", format="$%.6f"),
            "latency_ms": st.column_config.NumberColumn("Latency (ms)", format="%.1f"),
            "quality_score": st.column_config.NumberColumn("Quality", format="%d"),
            "total_tokens": st.column_config.NumberColumn("Tokens", format="%d"),
        },
    )

    st.markdown("---")

    # Export Section
    st.subheader("💾 Export Data")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Export Filtered Traces (CSV)"):
            csv = filtered.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                csv,
                f"llm_traces_{datetime.now():%Y%m%d_%H%M%S}.csv",
                "text/csv",
            )

    with col2:
        if st.button("Export Cost by Tenant (CSV)"):
            csv = cost_by_tenant.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                csv,
                f"cost_by_tenant_{datetime.now():%Y%m%d_%H%M%S}.csv",
                "text/csv",
            )

    with col3:
        if st.button("Export Cost by Use Case (CSV)"):
            csv = cost_by_use_case.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                csv,
                f"cost_by_use_case_{datetime.now():%Y%m%d_%H%M%S}.csv",
                "text/csv",
            )

    st.markdown("---")
    st.subheader("🔍 SQL Explorer")
    st.write(
        "Run custom SQL queries against the DuckDB database. "
        "Table: `llm_traces` | Views: `v_cost_by_tenant`, `v_cost_by_use_case`, `v_latency_percentiles`"
    )

    default_query = "SELECT * FROM llm_traces ORDER BY timestamp DESC LIMIT 50;"
    query = st.text_area("SQL Query", value=default_query, height=100)

    if st.button("Run Query"):
        if query.strip():
            repo = TraceRepository(db_path)
            try:
                result = repo.query_df(query)
                st.dataframe(result, use_container_width=True, hide_index=True)
                st.success(f"Returned {len(result)} rows")
            except Exception as e:
                st.error(f"Query failed: {e}")
            finally:
                repo.close()


if __name__ == "__main__":
    main()