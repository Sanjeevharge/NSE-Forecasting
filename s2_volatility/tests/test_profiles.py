import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from features import H  # noqa: E402
from profiles import (ANN, PROFILES, Profile, alert_flag, build_panel, evaluate,  # noqa: E402
                      position_multiplier, rebalance_dates)


def synthetic_preds(n_days=120, tickers=("A", "B", "C"), seed=0, oracle=False):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    rows = []
    for t in tickers:
        fwd = rng.uniform(0.008, 0.03, n_days)                      # daily vol, ~13%-48% annualised
        pred = fwd.copy() if oracle else fwd * rng.lognormal(0, 0.3, n_days)
        rows.append(pd.DataFrame({"fold": np.arange(n_days) // 30, "date": dates, "ticker": t, "model": "m",
                                  "regime": "mid", "pred": pred, "fwd_rv": fwd}))
    return pd.concat(rows, ignore_index=True)


def test_multiplier_respects_cap_and_is_decreasing_in_forecast():
    p = Profile("x", target_ann_vol=0.15, alert_ann_vol=0.3, cap=1.0)
    pred = np.linspace(0.001, 0.05, 200)
    m = position_multiplier(pred, p)
    assert (m <= p.cap + 1e-12).all() and (m > 0).all()
    assert (np.diff(m) <= 1e-12).all()


def test_sizing_and_alerts_ignore_realized_vol():
    preds = synthetic_preds()
    shuffled = preds.copy()
    shuffled["fwd_rv"] = np.random.default_rng(1).permutation(shuffled["fwd_rv"].values)
    a, b = build_panel(preds), build_panel(shuffled)
    assert np.allclose(a["multiplier"].values, b["multiplier"].values)
    assert (a["alert"].values == b["alert"].values).all()


def test_rebalance_dates_do_not_overlap():
    dates = pd.bdate_range("2020-01-01", periods=200)
    keep = rebalance_dates(dates, H)
    pos = np.searchsorted(dates.values, keep)
    assert (np.diff(pos) == H).all()      # forward windows t+1..t+H of consecutive rebalances are disjoint


def test_oracle_forecast_has_zero_tracking_error_and_no_breaches():
    panel = build_panel(synthetic_preds(oracle=True), profiles=(Profile("low", 0.05, 0.25),))
    res = evaluate(panel)
    r = res[res["regime"] == "all"].iloc[0]
    assert r["capped_share"] == 0.0
    assert r["track_err"] < 1e-9 and r["breach_rate"] == 0.0
    assert r["alert_precision"] == 1.0 and r["alert_recall"] == 1.0


def test_alert_flag_threshold_is_annualised():
    p = Profile("x", 0.1, alert_ann_vol=0.30)
    below, above = 0.29 / ANN, 0.31 / ANN
    assert alert_flag([below, above], p).tolist() == [False, True]


def test_default_profiles_are_ordered_by_risk_appetite():
    targets = [p.target_ann_vol for p in PROFILES]
    alerts = [p.alert_ann_vol for p in PROFILES]
    assert targets == sorted(targets) and alerts == sorted(alerts)
