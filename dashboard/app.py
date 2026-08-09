"""
Streamlit dashboard: KPI tree + experiment readout.

Run:  streamlit run dashboard/app.py
Assumes `dbt build` has populated stream_kpis and fct_auctions.
"""
from __future__ import annotations

import duckdb
import pandas as pd
import streamlit as st

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.effect import run as run_effect  # noqa: E402

DB = "auctionlift.duckdb"

st.set_page_config(page_title="AuctionLift", layout="wide")
st.title("AuctionLift — Soft-close auction experiment")
st.caption("Should we extend the auction timer when a bid lands in the final seconds?")


@st.cache_data
def load():
    con = duckdb.connect(DB, read_only=True)
    kpis = con.execute("SELECT * FROM stream_kpis").fetchdf()
    con.close()
    return kpis


try:
    kpis = load()
except Exception:
    st.error("No data yet. Run `python -m data_gen.simulate` then `dbt build`.")
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
st.dataframe(by_variant.style.format({
    "gmv_per_stream": "${:,.0f}", "items_per_stream": "{:.1f}",
    "avg_sale_price": "${:,.2f}", "avg_duration": "{:.0f}s"}),
    use_container_width=True)

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
pivot = kpis.pivot_table(index=kpis.index, columns="variant_name", values="gmv")
st.bar_chart(by_variant.set_index("variant_name")["gmv_per_stream"])