"""Risk-profile layer: turns volatility forecasts into profile-specific position sizes and alerts.

IMPORTANT: the investor profiles below are SIMULATED. There are no real users in this project.
Prices, forecasts and realized volatility are real; only the profiles' risk budgets are assumed.

What it does, for each profile and each forecast model (naive, EWMA, HAR, LightGBM):
  1. Sizing  - volatility targeting: multiplier = min(cap, target_vol / forecast_vol), held for H days.
  2. Alerts  - flag a stock when forecast annualised vol exceeds the profile's alert threshold.

How it is scored (always against realized forward volatility, never against the forecast itself):
  - Tracking error : |realized vol of the scaled position - target| / target (uncapped positions only)
  - Breach rate    : share of periods where the scaled position's realized vol > BREACH_MULT x target
  - Alert quality  : precision / recall / F1 of "forecast > threshold" vs "realized > threshold"

No look-ahead: sizing and alerts read ONLY the `pred` column. `fwd_rv` is used afterwards, for scoring.
Rebalancing happens every H trading days, so 5-day forward windows never overlap.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from features import H

ANN = np.sqrt(252.0)          # daily vol -> annualised vol
BREACH_MULT = 1.5             # "breach" = realized position vol above 1.5x the profile's target
PREDICTIONS = Path(__file__).parent / "data" / "predictions.csv"
PROFILE_RESULTS = Path(__file__).parent / "data" / "profile_results.csv"


@dataclass(frozen=True)
class Profile:
    name: str
    target_ann_vol: float   # risk budget the position is scaled to
    alert_ann_vol: float    # alert when forecast annualised vol exceeds this
    cap: float = 1.0        # max position multiplier (1.0 = no leverage)


# Simulated risk budgets. Set them BEFORE looking at which model wins, and document any change in the README.
# Targets sit below the typical large-cap single-stock vol (~20% annualised, median over the test period) so the
# no-leverage cap rarely binds; a 30% target left 79-91% of positions capped, i.e. the profile did nothing.
PROFILES = (
    Profile("conservative", target_ann_vol=0.08, alert_ann_vol=0.20),
    Profile("balanced", target_ann_vol=0.12, alert_ann_vol=0.28),
    Profile("aggressive", target_ann_vol=0.18, alert_ann_vol=0.40),
)


# ----------------------------------------------------------------------------- core logic
def position_multiplier(pred_daily_vol, profile: Profile) -> np.ndarray:
    """Position multiplier from a DAILY vol forecast only (no access to realized values)."""
    pred = np.maximum(np.asarray(pred_daily_vol, dtype=float), 1e-8)
    return np.minimum(profile.cap, (profile.target_ann_vol / ANN) / pred)


def alert_flag(pred_daily_vol, profile: Profile) -> np.ndarray:
    return np.asarray(pred_daily_vol, dtype=float) * ANN > profile.alert_ann_vol


def rebalance_dates(dates, h: int = H, offset: int = 0) -> np.ndarray:
    """Every h-th trading date, so the forward windows t+1..t+h of consecutive dates do not overlap."""
    return np.sort(pd.unique(pd.to_datetime(pd.Series(dates))))[offset::h]


def build_panel(preds: pd.DataFrame, profiles=PROFILES, h: int = H, offset: int = 0) -> pd.DataFrame:
    """One row per (profile, model, ticker, rebalance date) with sizing, alert and scoring columns."""
    need = {"fold", "date", "ticker", "model", "regime", "pred", "fwd_rv"}
    missing = need - set(preds.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {sorted(missing)}")
    preds = preds.copy()
    preds["date"] = pd.to_datetime(preds["date"])
    keep = rebalance_dates(preds["date"], h, offset)
    sub = preds[preds["date"].isin(keep)]

    frames = []
    for p in profiles:
        d = sub.copy()
        d["profile"] = p.name
        m = position_multiplier(d["pred"].values, p)                 # uses forecast only
        d["multiplier"] = m
        d["capped"] = m >= p.cap - 1e-12
        d["pos_vol_ann"] = m * d["fwd_rv"].values * ANN              # scoring: realized vol of the scaled position
        d["ratio"] = d["pos_vol_ann"] / p.target_ann_vol
        d["rel_err"] = (d["ratio"] - 1.0).abs()
        d["breach"] = d["pos_vol_ann"] > BREACH_MULT * p.target_ann_vol
        d["alert"] = alert_flag(d["pred"].values, p)
        d["actual_high"] = d["fwd_rv"].values * ANN > p.alert_ann_vol
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------------- scoring
def _summarise(g: pd.DataFrame) -> pd.Series:
    unc = ~g["capped"]
    tp = int((g["alert"] & g["actual_high"]).sum())
    n_alert, n_actual = int(g["alert"].sum()), int(g["actual_high"].sum())
    prec = tp / n_alert if n_alert else np.nan
    rec = tp / n_actual if n_actual else np.nan
    f1 = 2 * prec * rec / (prec + rec) if (n_alert and n_actual and (prec + rec) > 0) else np.nan
    return pd.Series({
        "n": len(g),
        "track_err": g.loc[unc, "rel_err"].mean(),
        "median_ratio": g.loc[unc, "ratio"].median(),   # >1 means realized risk overshoots the target
        "breach_rate": g["breach"].mean(),
        "capped_share": g["capped"].mean(),
        "alert_rate": g["alert"].mean(),
        "alert_precision": prec,
        "alert_recall": rec,
        "alert_f1": f1,
    })


def evaluate(panel: pd.DataFrame) -> pd.DataFrame:
    """Metrics per (profile, model, regime), plus regime='all'."""
    both = pd.concat([panel, panel.assign(regime="all")], ignore_index=True)
    out = both.groupby(["profile", "model", "regime"], sort=False).apply(_summarise, include_groups=False)
    return out.reset_index()


def bootstrap_ci(panel: pd.DataFrame, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> pd.DataFrame:
    """Fold-level bootstrap CIs for tracking error and breach rate (regime='all').

    Resamples whole walk-forward folds with replacement, so serial and cross-sectional dependence
    inside a fold is preserved.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for (prof, model), g in panel.groupby(["profile", "model"], sort=False):
        unc = ~g["capped"]
        agg = g.assign(err_sum=g["rel_err"].where(unc, 0.0), unc=unc.astype(int), br=g["breach"].astype(int)) \
               .groupby("fold").agg(err_sum=("err_sum", "sum"), unc=("unc", "sum"), br=("br", "sum"), n=("br", "size"))
        es, un, br, n = (agg[c].values for c in ["err_sum", "unc", "br", "n"])
        idx = rng.integers(0, len(agg), size=(n_boot, len(agg)))
        te = es[idx].sum(1) / np.maximum(un[idx].sum(1), 1)
        bk = br[idx].sum(1) / n[idx].sum(1)
        lo, hi = alpha / 2 * 100, (1 - alpha / 2) * 100
        rows.append(dict(profile=prof, model=model,
                         track_err_lo=np.percentile(te, lo), track_err_hi=np.percentile(te, hi),
                         breach_lo=np.percentile(bk, lo), breach_hi=np.percentile(bk, hi)))
    return pd.DataFrame(rows)


def paired_breach_diff(panel: pd.DataFrame, ref: str = "EWMA", n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Paired fold-bootstrap of breach_rate(model) - breach_rate(ref). A CI that excludes 0 means the gap is not noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for prof, g in panel.groupby("profile", sort=False):
        tab = g.groupby(["fold", "model"]).agg(br=("breach", "sum"), n=("breach", "size")).unstack("model")
        k = len(tab)
        idx = rng.integers(0, k, size=(n_boot, k))
        rate = lambda m: tab[("br", m)].values[idx].sum(1) / tab[("n", m)].values[idx].sum(1)
        base = rate(ref)
        for m in tab["br"].columns:
            if m == ref:
                continue
            d = rate(m) - base
            rows.append(dict(profile=prof, model=m, vs=ref, diff=float(np.mean(d)),
                             lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5))))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- I/O + CLI
def load_predictions(refresh: bool = False) -> pd.DataFrame:
    """Out-of-sample forecasts from the walk-forward run (cached to data/predictions.csv)."""
    if PREDICTIONS.exists() and not refresh:
        return pd.read_csv(PREDICTIONS, parse_dates=["date"])
    import features
    import walkforward
    from data import PRICES
    prices = pd.read_csv(PRICES, parse_dates=["date"])
    _, preds = walkforward.run(features.build(prices), return_preds=True)
    PREDICTIONS.parent.mkdir(exist_ok=True, parents=True)
    preds.to_csv(PREDICTIONS, index=False)
    return preds


def main(refresh: bool = False):
    preds = load_predictions(refresh)
    panel = build_panel(preds)
    res = evaluate(panel)
    ci = bootstrap_ci(panel)
    PROFILE_RESULTS.parent.mkdir(exist_ok=True, parents=True)
    res.to_csv(PROFILE_RESULTS, index=False)

    allr = res[res["regime"] == "all"].merge(ci, on=["profile", "model"])
    print("\nSIMULATED investor profiles (risk budgets are assumed; prices and forecasts are real)")
    print(f"Rebalance every {H} trading days; breach = realized position vol > {BREACH_MULT}x target\n")
    for prof, g in allr.groupby("profile", sort=False):
        p = next(x for x in PROFILES if x.name == prof)
        print(f"--- {prof}: target vol {p.target_ann_vol:.0%}, alert > {p.alert_ann_vol:.0%}, cap {p.cap:.1f}x ---")
        t = g.set_index("model")[["n", "track_err", "track_err_lo", "track_err_hi", "median_ratio",
                                  "breach_rate", "breach_lo", "breach_hi", "capped_share",
                                  "alert_precision", "alert_recall", "alert_f1"]]
        print(t.round(3).to_string())
        print()
    pd_ = paired_breach_diff(panel)
    print("Breach-rate difference vs EWMA (paired fold bootstrap, 95% CI; CI excluding 0 = not noise):")
    print(pd_.round(3).to_string(index=False) + "\n")
    hi = res[(res["regime"] == "high")].pivot(index="model", columns="profile", values="breach_rate")
    print("Breach rate in HIGH-volatility regime:\n" + hi.round(3).to_string())
    print(f"\nSaved {PROFILE_RESULTS}")
    return res


if __name__ == "__main__":
    main()
