"""
Streamlit dashboard: KPI tree + experiment readout.

Run:  streamlit run dashboard/app.py
On first startup, the app creates the synthetic DuckDB data and mart tables if
they are missing. This keeps Streamlit Cloud deploys self-contained.
"""
from __future__ import annotations

from pathlib import Path
import sys

import duckdb
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.effect import run as run_effect  # noqa: E402
from data_gen.simulate import simulate  # noqa: E402

DB = str(ROOT / "auctionlift.duckdb")

st.set_page_config(page_title="AuctionLift", layout="wide")
st.title("AuctionLift — Soft-close auction experiment")
st.caption("Should we extend the auction timer when a bid lands in the final seconds?")


def table_exists(con: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    return bool(
        con.execute(
            """
            select count(*)
            from information_schema.tables
            where table_schema = 'main' and table_name = ?
            """,
            [table_name],
        ).fetchone()[0]
    )


def write_raw_tables(con: duckdb.DuckDBPyConnection) -> None:
    tables = simulate(n_streams=2000, seed=42)
    for name, df in tables.items():
        con.execute(f"create or replace table {name} as select * from df")


def build_marts(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        create or replace view stg_auctions as
        select auction_id, stream_id, sale_price, duration_s, n_bids, was_sniped
        from raw_auctions
        """
    )
    con.execute(
        """
        create or replace view stg_experiment_assignment as
        select
            stream_id,
            variant,
            case when variant = 1 then 'soft_close' else 'hard_close' end as variant_name
        from raw_experiment_assignment
        """
    )
    con.execute(
        """
        create or replace view stg_streams as
        select stream_id, seller_size, time_budget_s, planned_auctions, pre_period_gmv
        from raw_streams
        """
    )
    con.execute(
        """
        create or replace table fct_auctions as
        select
            a.auction_id,
            a.stream_id,
            x.variant,
            x.variant_name,
            a.sale_price,
            a.duration_s,
            a.n_bids,
            a.was_sniped,
            s.pre_period_gmv
        from stg_auctions a
        join stg_experiment_assignment x using (stream_id)
        join stg_streams s using (stream_id)
        """
    )
    con.execute(
        """
        create or replace table stream_kpis as
        select
            stream_id,
            variant,
            variant_name,
            any_value(pre_period_gmv) as pre_period_gmv,
            sum(sale_price) as gmv,
            count(*) as items_sold,
            avg(sale_price) as avg_sale_price,
            avg(duration_s) as avg_duration_s,
            avg(n_bids) as avg_bids_per_auction,
            sum(case when was_sniped = 1 then 1 else 0 end) as sniped_auctions
        from fct_auctions
        group by stream_id, variant, variant_name
        """
    )


def ensure_data() -> None:
    con = duckdb.connect(DB)
    try:
        if not table_exists(con, "raw_auctions"):
            write_raw_tables(con)
        if not table_exists(con, "fct_auctions") or not table_exists(con, "stream_kpis"):
            build_marts(con)
    finally:
        con.close()


@st.cache_data(show_spinner="Preparing synthetic auction data...")
def load() -> pd.DataFrame:
    ensure_data()
    con = duckdb.connect(DB, read_only=True)
    kpis = con.execute("SELECT * FROM stream_kpis").fetchdf()
    con.close()
    return kpis


try:
    kpis = load()
except Exception as exc:
    st.error("Unable to prepare the dashboard data.")
    st.exception(exc)
    st.stop()

by_variant = (
    kpis.groupby("variant_name")
    .agg(streams=("stream_id", "count"), gmv_per_stream=("gmv", "mean"),
         items_per_stream=("items_sold", "mean"),
         avg_sale_price=("avg_sale_price", "mean"),
         avg_duration=("avg_duration_s", "mean"))
    .reset_index()
)

st.subheader("KPI tree by variant")
st.dataframe(
    by_variant,
    column_config={
        "variant_name": "Variant",
        "streams": st.column_config.NumberColumn("Streams", format="%d"),
        "gmv_per_stream": st.column_config.NumberColumn("GMV / stream", format="$%d"),
        "items_per_stream": st.column_config.NumberColumn("Items / stream", format="%.1f"),
        "avg_sale_price": st.column_config.NumberColumn("Avg sale price", format="$%.2f"),
        "avg_duration": st.column_config.NumberColumn("Avg duration", format="%d s"),
    },
    hide_index=True,
    width="stretch",
)

st.subheader("Experiment result")
res = run_effect(DB)  # prints to stdout; we surface the numbers here too

c1, c2, c3 = st.columns(3)
c1.metric("Net GMV / stream", f"{res['gmv_lift_pct']:+.2f}%",
          help=f"95% CI [{res['gmv_ci_pct'][0]:+.2f}%, {res['gmv_ci_pct'][1]:+.2f}%]")
c2.metric("Price / auction", f"{res['price_lift_pct']:+.2f}%")
c3.metric("Items / stream (throughput)", f"{res['items_lift_pct']:+.2f}%")

st.markdown(
    f"**Recommendation:** {'Ship' if res['ship'] else 'Do not ship a global rollout.'} "
    "Higher winning prices are real and significant, but the longer auctions cost "
    "more in lost throughput than they add in price — net GMV per stream falls. "
    "Worth a follow-up test targeting only high-demand streams, where the time "
    "budget isn't the binding constraint."
)

st.subheader("GMV per stream — distribution by variant")
st.bar_chart(by_variant.set_index("variant_name")["gmv_per_stream"])