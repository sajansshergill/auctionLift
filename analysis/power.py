"""
Power analysis for the decision metric (GMV per stream).

Answers two questions a PM asks before launch:
  - "How many streams do we need to detect a 3% lift?"
  - "Given the streams we'll get, what's the smallest lift we can call?"

Uses the observed GMV-per-stream distribution to get a realistic CV, and
credits CUPED with the variance reduction it delivers.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
from statsmodels.stats.power import TTestIndPower


def _stream_gmv(db: str) -> pd.DataFrame:
    con = duckdb.connect(db, read_only=True)
    try:
        df = con.execute(
            "SELECT stream_id, variant, gmv, pre_period_gmv FROM stream_kpis"
        ).fetchdf()
    except Exception:
        df = con.execute(
            """
            SELECT a.stream_id, x.variant, SUM(a.sale_price) AS gmv,
                   ANY_VALUE(s.pre_period_gmv) AS pre_period_gmv
            FROM raw_auctions a
            JOIN raw_experiment_assignment x USING (stream_id)
            JOIN raw_streams s USING (stream_id)
            GROUP BY 1, 2
            """
        ).fetchdf()
    con.close()
    return df


def run(db: str = "auctionlift.duckdb", alpha: float = 0.05, power: float = 0.8) -> dict:
    df = _stream_gmv(db)
    mean, sd = df["gmv"].mean(), df["gmv"].std(ddof=1)
    cv = sd / mean

    # CUPED variance reduction from the correlation with the pre-period covariate
    rho = np.corrcoef(df["gmv"], df["pre_period_gmv"])[0, 1]
    cuped_sd_factor = np.sqrt(1 - rho**2)
    cv_cuped = cv * cuped_sd_factor

    analysis = TTestIndPower()
    print("=" * 60)
    print("POWER — GMV per stream")
    print("=" * 60)
    print(f"mean GMV/stream: {mean:,.0f}   CV: {cv:.2f}   corr w/ pre: {rho:.2f}\n")

    print("Streams per arm needed to detect a given lift (80% power):")
    for lift in (0.02, 0.03, 0.05):
        d_raw = lift / cv
        d_cuped = lift / cv_cuped
        n_raw = analysis.solve_power(effect_size=d_raw, alpha=alpha, power=power)
        n_cuped = analysis.solve_power(effect_size=d_cuped, alpha=alpha, power=power)
        print(f"  {lift:>4.0%} lift:  no CUPED {n_raw:>8,.0f}   with CUPED {n_cuped:>8,.0f}")

    # MDE for the sample we actually have
    n_per_arm = len(df) / 2
    d_detectable = analysis.solve_power(nobs1=n_per_arm, alpha=alpha, power=power)
    print(f"\nWith {len(df):,} streams ({n_per_arm:,.0f}/arm):")
    print(f"  MDE no CUPED:   {d_detectable * cv:.2%} lift")
    print(f"  MDE with CUPED: {d_detectable * cv_cuped:.2%} lift")
    print("=" * 60)

    return {"cv": cv, "rho": rho, "mde_cuped_pct": d_detectable * cv_cuped * 100}


if __name__ == "__main__":
    run()