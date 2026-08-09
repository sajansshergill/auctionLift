"""
Experiment-validity gate — run this BEFORE trusting any effect estimate.

This is the QA/SDET differentiator: most experiment analyses skip these
checks and quietly ship bad reads. Every check returns pass/fail with a
number, and the module exits non-zero if any hard check fails, so it can
sit in CI or as a pre-analysis gate.

Checks
------
1. Sample Ratio Mismatch (SRM): is the split actually 50/50? A skew means
   assignment or logging is broken and the whole experiment is suspect.
2. Pre-experiment balance: is the pre-period covariate balanced across arms?
   Imbalance means randomization didn't work.
3. Pipeline integrity: no orphan auctions/bids, no null prices, no negative
   amounts, every auction maps to an assigned stream.
"""
from __future__ import annotations

import sys
import duckdb
import numpy as np
import pandas as pd
from scipy import stats


def _load(db: str):
    con = duckdb.connect(db, read_only=True)
    assign = con.execute("SELECT * FROM raw_experiment_assignment").fetchdf()
    streams = con.execute("SELECT * FROM raw_streams").fetchdf()
    auctions = con.execute("SELECT * FROM raw_auctions").fetchdf()
    bids = con.execute("SELECT * FROM raw_bids").fetchdf()
    con.close()
    return assign, streams, auctions, bids


def check_srm(assign: pd.DataFrame, expected=0.5, alpha=0.001) -> tuple[bool, str]:
    counts = assign["variant"].value_counts().sort_index()
    n = counts.sum()
    chi2, p = stats.chisquare(
        [counts.get(0, 0), counts.get(1, 0)], [n * (1 - expected), n * expected]
    )
    ok = p > alpha
    return ok, f"SRM: split {counts.get(1,0)}/{counts.get(0,0)}, p={p:.3f} (fail if <{alpha})"


def check_balance(assign, streams, alpha=0.01) -> tuple[bool, str]:
    df = streams.merge(assign, on="stream_id")
    t = df.loc[df.variant == 1, "pre_period_gmv"]
    c = df.loc[df.variant == 0, "pre_period_gmv"]
    _, p = stats.ttest_ind(t, c, equal_var=False)
    ok = p > alpha
    return ok, f"Balance: pre-GMV means {t.mean():.0f} vs {c.mean():.0f}, p={p:.3f} (fail if <{alpha})"


def check_integrity(assign, streams, auctions, bids) -> list[tuple[bool, str]]:
    out = []
    assigned = set(assign["stream_id"])
    orphan_streams = set(auctions["stream_id"]) - assigned
    out.append((len(orphan_streams) == 0,
                f"Integrity: {len(orphan_streams)} auction streams with no assignment"))

    out.append((auctions["sale_price"].notna().all() and (auctions["sale_price"] > 0).all(),
                f"Integrity: {(auctions['sale_price'] <= 0).sum()} non-positive sale prices"))

    orphan_bids = set(bids["auction_id"]) - set(auctions["auction_id"])
    out.append((len(orphan_bids) == 0,
                f"Integrity: {len(orphan_bids)} bids referencing missing auctions"))

    out.append(((bids["amount"] >= 0).all(),
                f"Integrity: {(bids['amount'] < 0).sum()} negative bid amounts"))
    return out


def run(db: str = "auctionlift.duckdb") -> bool:
    assign, streams, auctions, bids = _load(db)
    results = [check_srm(assign), check_balance(assign, streams)]
    results += check_integrity(assign, streams, auctions, bids)

    print("=" * 60)
    print("EXPERIMENT VALIDITY GATE")
    print("=" * 60)
    all_ok = True
    for ok, msg in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
        all_ok = all_ok and ok
    print("=" * 60)
    print("RESULT:", "all checks passed — safe to analyze" if all_ok
          else "FAILURES present — do not trust the effect estimate")
    return all_ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)