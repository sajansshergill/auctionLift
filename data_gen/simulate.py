"""
Simulate live-commerce auction data with a KNOWN, planted treatment effect.

The whole project rests on this: because we generate the data from a ground
truth we control, the analysis has something concrete to recover. If the
pipeline can't recover the effect we planted here, the pipeline is wrong.

Design of the world
-------------------
- Sellers run streams. Randomization is at the STREAM level (a stream is
  either all soft-close or all hard-close). This mirrors reality: you can't
  flip the timer rule mid-stream per bid.
- Each stream has a fixed time budget. Auctions consume duration, so a
  stream can only fit so many. This is what creates the throughput tradeoff.
- Some auctions get "sniped" (a bid in the final seconds).

Planted effect of SOFT CLOSE (treatment) on a sniped auction:
- price:    winning price is lifted by PRICE_UPLIFT   (bidding war)
- duration: the auction runs longer by DURATION_EXTRA (timer extends)

The longer duration means fewer auctions fit in the stream's time budget ->
a throughput penalty. Net GMV-per-stream effect is the price lift minus the
throughput loss, and it is deliberately positive-but-modest so the
experiment has to actually work to detect it.

Outputs raw tables into a DuckDB file for dbt to build on:
  raw_streams, raw_auctions, raw_bids, raw_purchases, raw_experiment_assignment
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import duckdb


# ---- ground-truth parameters (the effect we are planting) -------------------
PRICE_UPLIFT = 0.06        # +6% winning price on a sniped auction under soft close
DURATION_EXTRA = 18.0      # +18s duration on a sniped auction under soft close
P_SNIPE = 0.35             # share of auctions that get sniped
TREATMENT_SHARE = 0.5      # fraction of streams assigned to soft close


def simulate(n_streams: int = 2000, seed: int = 42) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    # --- streams -------------------------------------------------------------
    stream_ids = np.arange(1, n_streams + 1)
    treatment = (rng.random(n_streams) < TREATMENT_SHARE).astype(int)

    # seller "size" drives volume; power-law-ish (few big, long tail of small)
    size = rng.lognormal(mean=0.0, sigma=0.6, size=n_streams)
    time_budget = 900 + 2200 * size            # seconds of live selling time
    planned_auctions = np.clip((size * 40).astype(int), 8, 200)

    # pre-period GMV per stream: correlated with size, used later for CUPED.
    # Independent of treatment (measured before the experiment started).
    pre_gmv = 1200 * size + rng.normal(0, 300, n_streams)
    pre_gmv = np.clip(pre_gmv, 50, None)

    streams = pd.DataFrame(
        {"stream_id": stream_ids, "seller_size": size,
         "time_budget_s": time_budget, "planned_auctions": planned_auctions,
         "pre_period_gmv": pre_gmv}
    )
    assignment = pd.DataFrame({"stream_id": stream_ids, "variant": treatment})

    # --- auctions + bids -----------------------------------------------------
    auction_rows, bid_rows, purchase_rows = [], [], []
    auction_id = 0
    bid_id = 0

    for i in range(n_streams):
        sid = int(stream_ids[i])
        is_treat = int(treatment[i])
        budget = time_budget[i]
        spent = 0.0

        for _ in range(int(planned_auctions[i])):
            base_value = rng.lognormal(mean=3.0, sigma=0.5)      # ~$20 typical
            base_duration = rng.uniform(45, 120)                 # seconds
            sniped = int(rng.random() < P_SNIPE)

            # planted effect applies only to sniped auctions under treatment
            price = base_value * (1 + (PRICE_UPLIFT if (is_treat and sniped) else 0.0))
            price *= (1 + rng.normal(0, 0.05))                   # measurement noise
            price = max(price, 1.0)

            duration = base_duration + (DURATION_EXTRA if (is_treat and sniped) else 0.0)

            # throughput constraint: stop once the stream's time budget is used
            if spent + duration > budget:
                break
            spent += duration
            auction_id += 1

            n_bids = rng.integers(1, 12) + (2 if sniped else 0)
            auction_rows.append(
                {"auction_id": auction_id, "stream_id": sid,
                 "sale_price": round(price, 2), "duration_s": round(duration, 1),
                 "n_bids": int(n_bids), "was_sniped": sniped}
            )

            # a small number of bid records per auction (timestamps to the second)
            for b in range(int(n_bids)):
                bid_id += 1
                secs_before_end = 0.5 if (b == n_bids - 1 and sniped) else rng.uniform(1, duration)
                bid_rows.append(
                    {"bid_id": bid_id, "auction_id": auction_id, "stream_id": sid,
                     "amount": round(price * rng.uniform(0.6, 1.0), 2),
                     "seconds_before_end": round(secs_before_end, 1)}
                )

            # winning bid -> a purchase
            purchase_rows.append(
                {"auction_id": auction_id, "stream_id": sid, "amount": round(price, 2)}
            )

    return {
        "raw_streams": streams,
        "raw_experiment_assignment": assignment,
        "raw_auctions": pd.DataFrame(auction_rows),
        "raw_bids": pd.DataFrame(bid_rows),
        "raw_purchases": pd.DataFrame(purchase_rows),
    }


def true_effects(tables: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Compute the realized ground-truth effect from the generated data,
    so the analysis has a target to hit."""
    a = tables["raw_auctions"].merge(tables["raw_experiment_assignment"], on="stream_id")
    stream_gmv = (
        a.groupby(["stream_id", "variant"])["sale_price"].sum().reset_index(name="gmv")
    )
    t = stream_gmv.loc[stream_gmv.variant == 1, "gmv"].mean()
    c = stream_gmv.loc[stream_gmv.variant == 0, "gmv"].mean()

    items = a.groupby(["stream_id", "variant"]).size().reset_index(name="items")
    it = items.loc[items.variant == 1, "items"].mean()
    ic = items.loc[items.variant == 0, "items"].mean()

    return {
        "true_gmv_per_stream_lift_pct": 100 * (t - c) / c,
        "true_gmv_treatment": t,
        "true_gmv_control": c,
        "true_items_lift_pct": 100 * (it - ic) / ic,
        "planted_price_uplift_pct": 100 * PRICE_UPLIFT,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--streams", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--db", type=str, default="auctionlift.duckdb")
    args = p.parse_args()

    tables = simulate(args.streams, args.seed)
    con = duckdb.connect(args.db)
    for name, df in tables.items():
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM df")
    con.close()

    eff = true_effects(tables)
    n_auctions = len(tables["raw_auctions"])
    print(f"Wrote {len(tables['raw_streams'])} streams, {n_auctions} auctions to {args.db}")
    print("--- ground truth (what the analysis should recover) ---")
    for k, v in eff.items():
        print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    main()