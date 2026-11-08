"""
Streamlit dashboard: KPI tree + experiment readout.

Console: global sidebar filters that live-update every view,
KPI cards with soft-vs-hard deltas, Plotly animated charts
(animation_frame over sequential auction cohorts) and a genuine 3D
auction space (Scatter3d + surface).

Run:  streamlit run dashboard/app.py
On first startup, the app creates the synthetic DuckDB data and mart tables if
they are missing. This keeps Streamlit Cloud deploys self-contained.
"""
from __future__ import annotations

from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.effect import run as run_effect  # noqa: E402
from data_gen.simulate import simulate  # noqa: E402

DB = str(ROOT / "auctionlift.duckdb")

st.set_page_config(page_title="AuctionLift", page_icon="🔨", layout="wide")

# ----------------------------------------------------------------------------
# Custom CSS — dark glass / gradient console styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(1200px 600px at 15% -10%, #5e2b14 0%, transparent 55%),
                    radial-gradient(1000px 500px at 90% 0%, #113a5e 0%, transparent 50%),
                    #0e1117;
    }
    .al-hero {
        padding: 1.1rem 1.4rem; border-radius: 16px; margin-bottom: 0.6rem;
        background: linear-gradient(120deg, rgba(190,90,20,.50), rgba(30,80,160,.45));
        border: 1px solid rgba(255,180,120,.35);
        box-shadow: 0 8px 32px rgba(0,0,0,.45);
    }
    .al-hero h1 {
        margin: 0; font-size: 2.1rem; letter-spacing: .5px;
        background: linear-gradient(90deg,#ffd09f,#ffb3c8,#9fd0ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .al-hero p { margin: .25rem 0 0 0; color: #cfd8e3; font-size: .95rem; }
    div[data-testid="stMetric"] {
        background: rgba(22,27,34,.85); border: 1px solid rgba(255,170,100,.35);
        border-radius: 12px; padding: .6rem .8rem;
        box-shadow: 0 4px 18px rgba(0,0,0,.35);
    }
    div[data-testid="stMetric"]:hover { border-color: rgba(159,208,255,.7); }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background: rgba(22,27,34,.8); border-radius: 10px 10px 0 0; padding: 6px 14px;
    }
    section[data-testid="stSidebar"] { background: rgba(14,17,23,.92); }
    </style>
    <div class="al-hero">
      <h1>🔨 AuctionLift — 3D Soft-Close Experiment Console</h1>
      <p>Should Whatnot extend the auction timer when a bid lands in the final seconds?
         Drag the 3D auction space, press play on the animated cohort charts, and slice
         every view live from the sidebar.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


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
def load():
    ensure_data()
    con = duckdb.connect(DB, read_only=True)
    kpis = con.execute("SELECT * FROM stream_kpis").fetchdf()
    auctions = con.execute("SELECT * FROM fct_auctions").fetchdf()
    streams = con.execute(
        "SELECT stream_id, seller_size, time_budget_s, planned_auctions FROM raw_streams"
    ).fetchdf()
    con.close()
    kpis = kpis.merge(streams, on="stream_id", how="left")
    auctions = auctions.merge(streams, on="stream_id", how="left")
    # Sequential cohort (auction order in 20 batches) — the time axis for animation.
    auctions = auctions.sort_values("auction_id").reset_index(drop=True)
    n = max(len(auctions), 1)
    cohort = (np.arange(n) * 20) // n
    auctions["cohort"] = cohort
    auctions["cohort_label"] = [f"Batch {int(c) + 1:02d}" for c in cohort]
    return kpis, auctions


@st.cache_data(show_spinner="Running the experiment analysis...")
def load_effect() -> dict:
    return run_effect(DB)


try:
    kpis, auctions = load()
except Exception as exc:
    st.error("Unable to prepare the dashboard data.")
    st.exception(exc)
    st.stop()

res = load_effect()

# ----------------------------------------------------------------------------
# Sidebar — global dynamic filters (every chart below reads the filtered data)
# ----------------------------------------------------------------------------
st.sidebar.header("🎛️ Live filters")
all_variants = sorted(kpis["variant_name"].dropna().unique())
sel_variants = st.sidebar.multiselect("Variant", all_variants, default=all_variants)
snipe_mode = st.sidebar.selectbox(
    "Sniping", ["All auctions", "Sniped only", "Not sniped"],
    help="Was a bid placed in the final seconds?",
)
seller_lo, seller_hi = float(kpis["seller_size"].min()), float(kpis["seller_size"].max())
sel_seller = st.sidebar.slider("Seller size", seller_lo, seller_hi, (seller_lo, seller_hi))
gmv_lo, gmv_hi = float(kpis["pre_period_gmv"].min()), float(kpis["pre_period_gmv"].max())
sel_gmv = st.sidebar.slider("Pre-period GMV ($)", gmv_lo, gmv_hi, (gmv_lo, gmv_hi))
price_lo, price_hi = float(auctions["sale_price"].min()), float(auctions["sale_price"].max())
sel_price = st.sidebar.slider("Sale price ($)", price_lo, price_hi, (price_lo, price_hi))
min_bids = st.sidebar.slider("Min bids per auction", 0, int(auctions["n_bids"].max()), 0)
sample_3d = st.sidebar.select_slider("3D sample size", [1500, 3000, 5000], value=3000)

f_kpis = kpis[
    kpis["variant_name"].isin(sel_variants or all_variants)
    & kpis["seller_size"].between(sel_seller[0], sel_seller[1])
    & kpis["pre_period_gmv"].between(sel_gmv[0], sel_gmv[1])
]
f_auc = auctions[
    auctions["variant_name"].isin(sel_variants or all_variants)
    & auctions["seller_size"].between(sel_seller[0], sel_seller[1])
    & auctions["pre_period_gmv"].between(sel_gmv[0], sel_gmv[1])
    & auctions["sale_price"].between(sel_price[0], sel_price[1])
    & (auctions["n_bids"] >= min_bids)
]
if snipe_mode == "Sniped only":
    f_auc = f_auc[f_auc["was_sniped"] == 1]
elif snipe_mode == "Not sniped":
    f_auc = f_auc[f_auc["was_sniped"] == 0]
# Keep stream KPIs consistent with the auction-level filters that can apply.
if not f_auc.empty:
    f_kpis = f_kpis[f_kpis["stream_id"].isin(f_auc["stream_id"].unique())]

by_variant = (
    f_kpis.groupby("variant_name")
    .agg(streams=("stream_id", "count"), gmv_per_stream=("gmv", "mean"),
         items_per_stream=("items_sold", "mean"),
         avg_sale_price=("avg_sale_price", "mean"),
         avg_duration=("avg_duration_s", "mean"))
    .reset_index()
) if not f_kpis.empty else pd.DataFrame()


def _soft_hard_delta(metric: str) -> str | None:
    if by_variant.empty or len(by_variant) < 2:
        return None
    d = by_variant.set_index("variant_name")
    if "soft_close" not in d.index or "hard_close" not in d.index:
        return None
    hard = float(d.loc["hard_close", metric])
    if hard == 0:
        return None
    return f"{100 * (float(d.loc['soft_close', metric]) - hard) / hard:+.1f}% soft vs hard"


# ----------------------------------------------------------------------------
# KPI row — with deltas (soft close vs hard close, on the filtered data)
# ----------------------------------------------------------------------------
if not f_kpis.empty:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("GMV / stream", f"${f_kpis['gmv'].mean():,.0f}",
              delta=_soft_hard_delta("gmv_per_stream"))
    c2.metric("Price / auction", f"${f_auc['sale_price'].mean():,.2f}" if not f_auc.empty else "—",
              delta=_soft_hard_delta("avg_sale_price"))
    c3.metric("Items / stream", f"{f_kpis['items_sold'].mean():.1f}",
              delta=_soft_hard_delta("items_per_stream"))
    c4.metric("Avg duration", f"{f_kpis['avg_duration_s'].mean():.0f} s",
              delta=_soft_hard_delta("avg_duration"))
    c5.metric("Streams in view", f"{f_kpis['stream_id'].nunique():,}",
              delta=f"{len(f_auc):,} auctions")

tab_over, tab_anim, tab_3d, tab_mech, tab_explore = st.tabs(
    ["📊 KPI tree & verdict", "🎬 Animated cohorts", "🧊 3D auction space",
     "⚖️ Mechanism & decomposition", "🔎 Explorer"]
)

with tab_over:
    st.subheader("KPI tree by variant")
    if by_variant.empty:
        st.info("No streams match the current filters.")
    else:
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
        fig = px.bar(
            by_variant, x="variant_name", y="gmv_per_stream", color="variant_name",
            labels={"gmv_per_stream": "GMV / stream ($)", "variant_name": "Variant"},
            template="plotly_dark", title="GMV per stream by variant (filtered)",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Experiment result")
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
    if not by_variant.empty:
        st.bar_chart(by_variant.set_index("variant_name")["gmv_per_stream"])

with tab_anim:
    st.subheader("Press play — GMV evolving across sequential auction batches")
    if f_auc.empty:
        st.info("No auctions match the current filters.")
    else:
        daily = (
            f_auc.groupby(["cohort", "cohort_label", "variant_name"], as_index=False)
            .agg(gmv=("sale_price", "sum"), auctions=("auction_id", "count"),
                 mean_price=("sale_price", "mean"))
            .sort_values("cohort")
        )
        lim = float(daily["gmv"].max()) or 1.0
        fig_a = px.bar(
            daily, x="variant_name", y="gmv", color="variant_name",
            animation_frame="cohort_label", range_y=[0, lim * 1.15],
            labels={"gmv": "GMV in batch ($)", "variant_name": "Variant"},
            template="plotly_dark", title="GMV per auction batch — animated",
        )
        fig_a.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        if fig_a.layout.updatemenus:
            fig_a.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 350
        st.plotly_chart(fig_a, width="stretch")

        cum = daily.copy()
        cum["cum_gmv"] = cum.groupby("variant_name")["gmv"].cumsum()
        fig_c = px.line(
            cum, x="cohort_label", y="cum_gmv", color="variant_name",
            animation_frame="cohort_label",
            labels={"cum_gmv": "Cumulative GMV ($)", "cohort_label": "Auction batch"},
            template="plotly_dark", title="Cumulative GMV race — animated",
        )
        fig_c.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        if fig_c.layout.updatemenus:
            fig_c.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 350
        st.plotly_chart(fig_c, width="stretch")

        fig_p = px.bar(
            daily, x="variant_name", y="mean_price", color="variant_name",
            animation_frame="cohort_label",
            labels={"mean_price": "Mean winning price ($)", "variant_name": "Variant"},
            template="plotly_dark", title="Mean winning price per batch — animated",
        )
        fig_p.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        if fig_p.layout.updatemenus:
            fig_p.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 350
        st.plotly_chart(fig_p, width="stretch")
    st.caption("Animations are native Plotly `animation_frame` charts over 20 sequential auction batches — hit ▶ Play on each chart.")

with tab_3d:
    st.subheader("🧊 3D auction space — drag to rotate, scroll to zoom")
    if f_auc.empty:
        st.info("No auctions match the current filters.")
    else:
        d3 = f_auc.sample(n=min(sample_3d, len(f_auc)), random_state=42)
        fig3 = px.scatter_3d(
            d3.sort_values("auction_id"),
            x="sale_price", y="duration_s", z="n_bids",
            color="variant_name", size="sale_price", size_max=14, opacity=0.85,
            hover_data=["auction_id", "stream_id", "was_sniped"],
            labels={"sale_price": "Winning price ($)", "duration_s": "Duration (s)",
                    "n_bids": "Bids", "variant_name": "Variant"},
            template="plotly_dark",
            title="Every auction as a point in price × duration × bids space",
        )
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            scene=dict(
                xaxis=dict(backgroundcolor="rgba(20,24,32,.9)"),
                yaxis=dict(backgroundcolor="rgba(20,24,32,.9)"),
                zaxis=dict(backgroundcolor="rgba(20,24,32,.9)"),
            ),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig3, width="stretch")

        st.subheader("Price surface — duration × bids terrain")
        surf_src = f_auc.copy()
        surf_src["duration_bin"] = pd.cut(surf_src["duration_s"], bins=10).apply(lambda iv: round(iv.mid, 1))
        pivot = (
            surf_src.pivot_table(index="n_bids", columns="duration_bin",
                                 values="sale_price", aggfunc="mean")
            .sort_index().sort_index(axis=1)
        )
        if pivot.shape[0] > 2 and pivot.shape[1] > 2:
            z = pivot.fillna(pivot.stack().mean()).to_numpy()
            fig_s = go.Figure(data=[go.Surface(
                z=z, x=[str(c) for c in pivot.columns], y=list(pivot.index),
                colorscale="Turbo", colorbar=dict(title="Mean price ($)"),
            )])
            fig_s.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                scene=dict(xaxis_title="Duration midpoint (s)", yaxis_title="Bids",
                           zaxis_title="Mean winning price ($)",
                           camera=dict(eye=dict(x=1.5, y=-1.4, z=0.9))),
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_s, width="stretch")
        else:
            st.info("Not enough duration/bid coverage in the current filter for a surface.")

with tab_mech:
    st.subheader("Decomposition — why the net effect is negative")
    decomp = pd.DataFrame({
        "component": ["Price effect", "Throughput effect", "Net GMV / stream"],
        "lift_pct": [res["price_lift_pct"], res["items_lift_pct"], res["gmv_lift_pct"]],
    })
    fig_d = px.bar(
        decomp, x="component", y="lift_pct", color="lift_pct",
        color_continuous_scale="RdBu", color_continuous_midpoint=0,
        labels={"lift_pct": "Lift (%)", "component": "Component"},
        template="plotly_dark",
        title=f"Price {res['price_lift_pct']:+.2f}% · Throughput {res['items_lift_pct']:+.2f}% · Net {res['gmv_lift_pct']:+.2f}% (CUPED cut SE² by {res['cuped_variance_reduction_pct']:.1f}%)",
    )
    fig_d.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_d, width="stretch")

    if not f_auc.empty:
        st.subheader("Winning-price distribution by variant (filtered)")
        fig_h = px.histogram(
            f_auc, x="sale_price", color="variant_name", barmode="overlay", opacity=0.65,
            nbins=60, labels={"sale_price": "Winning price ($)"},
            template="plotly_dark",
        )
        fig_h.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_h, width="stretch")

        st.subheader("Price vs duration — the soft-close tradeoff in one picture")
        samp = f_auc.sample(n=min(4000, len(f_auc)), random_state=7)
        fig_sc = px.scatter(
            samp, x="duration_s", y="sale_price", color="variant_name", opacity=0.55,
            hover_data=["n_bids", "was_sniped"],
            labels={"duration_s": "Duration (s)", "sale_price": "Winning price ($)"},
            template="plotly_dark",
        )
        fig_sc.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_sc, width="stretch")

with tab_explore:
    st.subheader("Filtered stream KPIs")
    st.dataframe(f_kpis.sort_values("gmv", ascending=False), width="stretch", hide_index=True)
    st.subheader("Filtered auctions (first 500)")
    st.dataframe(
        f_auc.head(500)[["auction_id", "stream_id", "variant_name", "sale_price",
                         "duration_s", "n_bids", "was_sniped", "seller_size",
                         "pre_period_gmv", "cohort_label"]],
        width="stretch", hide_index=True,
    )
    st.caption(f"{len(f_auc):,} auctions · {f_kpis['stream_id'].nunique():,} streams in the current filter.")
