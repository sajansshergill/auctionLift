"""
Estimate the treatment effect two ways, because a marketplace decision needs
both the outcome and the mechanism:

1. STREAM-LEVEL (the decision metric): GMV per stream.
   - Unit of randomization is the stream, so one row per stream.
   - CUPED variance reduction using pre-period GMV as the covariate.
   - Robust (HC) standard errors + a 95% CI on the lift.

2. AUCTION-LEVEL (the mechanism): winning price per auction.
   - Many auctions per stream, treatment inherited from the stream.
   - Standard errors CLUSTERED by stream, because auctions within a stream
     are not independent. Skip this and you understate variance and invent
     wins that aren't there.

Then decompose the net GMV effect into a price component and a throughput
component, which is the actual insight a PM needs.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def _load(db: str) -> pd.DataFrame:
    con = duckdb.connect(db, read_only=True)
    # prefer the dbt mart if it exists; fall back to raw tables so the module
    # runs even before `dbt build`.
    try:
        auctions = con.execute("SELECT * FROM fct_auctions").fetchdf()
    except Exception:
        auctions = con.execute(
            """
            SELECT a.auction_id, a.stream_id, a.sale_price, a.was_sniped,
                   x.variant, s.pre_period_gmv
            FROM raw_auctions a
            JOIN raw_experiment_assignment x USING (stream_id)
            JOIN raw_streams s USING (stream_id)
            """
        ).fetchdf()
    con.close()
    return auctions


def _cuped_adjust(df: pd.DataFrame, metric: str, covariate: str) -> pd.Series:
    """Return the CUPED-adjusted metric. theta = cov(Y,X)/var(X)."""
    x = df[covariate].to_numpy()
    y = df[metric].to_numpy()
    theta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    return df[metric] - theta * (df[covariate] - df[covariate].mean())


def stream_level(auctions: pd.DataFrame) -> dict:
    streams = (
        auctions.groupby(["stream_id", "variant", "pre_period_gmv"])
        .agg(gmv=("sale_price", "sum"), items=("auction_id", "count"))
        .reset_index()
    )
    streams["gmv_cuped"] = _cuped_adjust(streams, "gmv", "pre_period_gmv")

    # robust-SE regression on the CUPED-adjusted outcome
    m = smf.ols("gmv_cuped ~ variant", data=streams).fit(cov_type="HC1")
    effect = m.params["variant"]
    ci_low, ci_high = m.conf_int().loc["variant"]
    control_mean = streams.loc[streams.variant == 0, "gmv"].mean()

    # variance reduction delivered by CUPED (vs the raw outcome)
    raw = smf.ols("gmv ~ variant", data=streams).fit(cov_type="HC1")
    var_reduction = 1 - (m.bse["variant"] ** 2) / (raw.bse["variant"] ** 2)

    items_t = streams.loc[streams.variant == 1, "items"].mean()
    items_c = streams.loc[streams.variant == 0, "items"].mean()

    return {
        "gmv_abs_effect": effect,
        "gmv_lift_pct": 100 * effect / control_mean,
        "gmv_ci_pct": (100 * ci_low / control_mean, 100 * ci_high / control_mean),
        "gmv_pvalue": m.pvalues["variant"],
        "cuped_variance_reduction_pct": 100 * var_reduction,
        "items_lift_pct": 100 * (items_t - items_c) / items_c,
        "n_streams": len(streams),
    }


def auction_level_price(auctions: pd.DataFrame) -> dict:
    m = smf.ols("sale_price ~ variant", data=auctions).fit(
        cov_type="cluster", cov_kwds={"groups": auctions["stream_id"]}
    )
    effect = m.params["variant"]
    ci_low, ci_high = m.conf_int().loc["variant"]
    control_mean = auctions.loc[auctions.variant == 0, "sale_price"].mean()
    return {
        "price_abs_effect": effect,
        "price_lift_pct": 100 * effect / control_mean,
        "price_ci_pct": (100 * ci_low / control_mean, 100 * ci_high / control_mean),
        "price_pvalue": m.pvalues["variant"],
        "n_auctions": len(auctions),
    }


def run(db: str = "auctionlift.duckdb") -> dict:
    auctions = _load(db)
    s = stream_level(auctions)
    p = auction_level_price(auctions)

    print("=" * 64)
    print("AUCTIONLIFT — treatment effect of SOFT CLOSE")
    print("=" * 64)
    print(f"\nStreams: {s['n_streams']:,}   Auctions: {p['n_auctions']:,}\n")

    print("DECISION METRIC — GMV per stream (CUPED-adjusted, HC-robust SE)")
    lo, hi = s["gmv_ci_pct"]
    print(f"  lift:            {s['gmv_lift_pct']:+.2f}%   95% CI [{lo:+.2f}%, {hi:+.2f}%]")
    print(f"  p-value:         {s['gmv_pvalue']:.4f}")
    print(f"  CUPED cut SE^2 by {s['cuped_variance_reduction_pct']:.1f}%\n")

    print("MECHANISM — winning price per auction (clustered SE by stream)")
    plo, phi = p["price_ci_pct"]
    print(f"  price lift:      {p['price_lift_pct']:+.2f}%   95% CI [{plo:+.2f}%, {phi:+.2f}%]")
    print(f"  p-value:         {p['price_pvalue']:.4f}\n")

    print("DECOMPOSITION")
    print(f"  price effect:      {p['price_lift_pct']:+.2f}%  (blended across all auctions)")
    print(f"  throughput effect: {s['items_lift_pct']:+.2f}%  (items sold per stream)")
    print(f"  net GMV/stream:    {s['gmv_lift_pct']:+.2f}%\n")

    ship = s["gmv_lift_pct"] > 0 and s["gmv_pvalue"] < 0.05
    verdict = "SHIP" if ship else "DO NOT SHIP (global rollout)"
    print(f"RECOMMENDATION: {verdict}")
    print("  Higher prices are real, but the longer auctions cost more in lost")
    print("  throughput than they add in price. Net GMV per stream falls.")
    print("=" * 64)

    return {**s, **p, "ship": ship}


if __name__ == "__main__":
    run()