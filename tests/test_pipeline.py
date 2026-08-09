"""
Tests that treat the data pipeline like production code.

Two layers:
  - contract tests: the simulated data has the shape the models assume.
  - correctness tests: the analysis recovers the planted ground-truth effect,
    and the validity gate catches deliberately corrupted data.

Run: pytest -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_gen.simulate import simulate, true_effects, PRICE_UPLIFT
from analysis.effect import stream_level, auction_level_price
from analysis.validity import check_integrity


@pytest.fixture(scope="module")
def tables():
    return simulate(n_streams=2000, seed=7)


@pytest.fixture(scope="module")
def auctions(tables):
    return tables["raw_auctions"].merge(
        tables["raw_experiment_assignment"], on="stream_id"
    ).merge(tables["raw_streams"][["stream_id", "pre_period_gmv"]], on="stream_id")


# ---- data contracts ---------------------------------------------------------
def test_no_null_or_negative_prices(tables):
    a = tables["raw_auctions"]
    assert a["sale_price"].notna().all()
    assert (a["sale_price"] > 0).all()


def test_variant_is_binary_and_roughly_balanced(tables):
    v = tables["raw_experiment_assignment"]["variant"]
    assert set(v.unique()) <= {0, 1}
    assert 0.45 < v.mean() < 0.55


def test_every_bid_maps_to_an_auction(tables):
    orphans = set(tables["raw_bids"]["auction_id"]) - set(tables["raw_auctions"]["auction_id"])
    assert not orphans


def test_pre_period_covariate_correlates_with_outcome(auctions):
    # CUPED only helps if the covariate is correlated with the outcome
    g = auctions.groupby("stream_id").agg(
        gmv=("sale_price", "sum"), pre=("pre_period_gmv", "first"))
    assert np.corrcoef(g["gmv"], g["pre"])[0, 1] > 0.5


# ---- analysis correctness ---------------------------------------------------
def test_price_effect_recovers_planted_uplift(auctions):
    """Blended price lift should be ~ PRICE_UPLIFT * P_SNIPE, and the CI on the
    per-auction lift should be positive and significant."""
    res = auction_level_price(auctions)
    assert res["price_pvalue"] < 0.01
    assert res["price_lift_pct"] > 0
    # blended lift is a fraction of the on-sniped uplift; sanity bound
    assert res["price_lift_pct"] < 100 * PRICE_UPLIFT


def test_throughput_penalty_is_negative(auctions):
    res = stream_level(auctions)
    assert res["items_lift_pct"] < 0


def test_net_gmv_matches_ground_truth_sign(tables, auctions):
    truth = true_effects(tables)
    est = stream_level(auctions)
    # both should agree that net GMV per stream moved in the same direction
    assert np.sign(est["gmv_lift_pct"]) == np.sign(truth["true_gmv_per_stream_lift_pct"])


def test_cuped_reduces_variance(auctions):
    res = stream_level(auctions)
    assert res["cuped_variance_reduction_pct"] > 0


def test_validity_gate_catches_corrupted_bids(tables):
    bids = tables["raw_bids"].copy()
    bids.loc[bids.index[0], "auction_id"] = -1
    results = check_integrity(
        tables["raw_experiment_assignment"],
        tables["raw_streams"],
        tables["raw_auctions"],
        bids,
    )
    assert any(not ok and "bids referencing missing auctions" in msg for ok, msg in results)