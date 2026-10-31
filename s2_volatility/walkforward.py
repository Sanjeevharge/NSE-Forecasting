"""Walk-forward evaluation with an embargo of H days between train and test."""
import sys
from pathlib import Path

# Ensure module path resolution works across working directories
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

from features import FEATURES, H, HAR, build
from data import PRICES, RESULTS

INIT_TRAIN_DAYS = 756   # ~3 years
TEST_DAYS = 63          # ~1 quarter per fold


def qlike(true_vol, pred_vol, eps=1e-8):
    tv, pv = np.maximum(true_vol, eps) ** 2, np.maximum(pred_vol, eps) ** 2
    return float(np.mean(tv / pv - np.log(tv / pv) - 1))


def smear(model, train, cols):
    """Duan smearing: corrects the downward bias from exponentiating a log-space forecast."""
    return float(np.mean(np.exp(train["y"].values - model.predict(train[cols]))))


def fit_predict(train, test):
    preds = {
        "naive (rv5)": test["rv5"].values,
        "EWMA": test["ewma"].values,
    }
    har = Ridge(alpha=1e-3).fit(train[HAR], train["y"])
    preds["HAR (ridge)"] = smear(har, train, HAR) * np.exp(har.predict(test[HAR]))
    gbm = LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15,
                        min_child_samples=100, subsample=0.8, subsample_freq=1,
                        colsample_bytree=0.8, random_state=42, verbose=-1)
    gbm.fit(train[FEATURES], train["y"])
    preds["LightGBM"] = smear(gbm, train, FEATURES) * np.exp(gbm.predict(test[FEATURES]))
    return preds


def run(df: pd.DataFrame):
    df = df.dropna(subset=["y"]).copy()
    dates = np.sort(df["date"].unique())
    rows = []
    start = INIT_TRAIN_DAYS
    fold = 0
    while start + TEST_DAYS <= len(dates):
        test_start, test_end = dates[start], dates[start + TEST_DAYS - 1]
        # embargo: train targets (t+1..t+H) must end before test_start
        train_end = dates[start - H - 1]
        train = df[df["date"] <= train_end]
        test = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
        # regime label from information known at test time (terciles of rv22 in train)
        q1, q2 = train["rv22"].quantile([1 / 3, 2 / 3])
        test["regime"] = np.where(test["rv22"] <= q1, "low", np.where(test["rv22"] <= q2, "mid", "high"))
        for name, p in fit_predict(train, test).items():
            for reg, idx in [("all", np.ones(len(test), bool))] + [(r, (test["regime"] == r).values) for r in ["low", "mid", "high"]]:
                if idx.sum() == 0:
                    continue
                tv = test["fwd_rv"].values[idx]
                rows.append(dict(fold=fold, model=name, regime=reg,
                                 mae=float(np.mean(np.abs(tv - p[idx]))), qlike=qlike(tv, p[idx]), n=int(idx.sum())))
        fold += 1
        start += TEST_DAYS
    return pd.DataFrame(rows)


def summarize(res: pd.DataFrame):
    allr = res[res["regime"] == "all"]
    tab = allr.groupby("model")[["mae", "qlike"]].mean()
    base = tab.loc["EWMA"]
    tab["mae_vs_EWMA_%"] = 100 * (tab["mae"] / base["mae"] - 1)
    tab["qlike_vs_EWMA_%"] = 100 * (tab["qlike"] / base["qlike"] - 1)
    print(f"\nFolds: {allr['fold'].nunique()}  (negative % = better than EWMA)\n")
    print(tab.round(4).to_string())
    pv = allr.pivot(index="fold", columns="model", values="qlike")
    print("\nFolds where LightGBM beats HAR on QLIKE:",
          int((pv["LightGBM"] < pv["HAR (ridge)"]).sum()), "of", len(pv))
    print("\nBy regime (mean QLIKE):")
    print(res[res["regime"] != "all"].groupby(["regime", "model"])["qlike"].mean().unstack().round(4).to_string())


if __name__ == "__main__":
    prices = pd.read_csv(PRICES, parse_dates=["date"])
    print(f"Building features from {len(prices):,} price records across {prices['ticker'].nunique()} tickers...")
    feature_df = build(prices)
    print(f"Running walk-forward evaluation across rolling quarterly folds (H={H} days embargo)...")
    res = run(feature_df)
    RESULTS.parent.mkdir(exist_ok=True, parents=True)
    res.to_csv(RESULTS, index=False)
    print(f"Saved evaluation metrics to {RESULTS}")
    summarize(res)
    
    # Auto-generate visual charts if generate_visuals exists
    try:
        from generate_visuals import (
            ASSETS_DIR,
            generate_fold_comparison_chart,
            generate_regime_chart,
            generate_relative_improvement_chart,
        )
        generate_regime_chart(res, ASSETS_DIR / "regime_performance.svg")
        generate_fold_comparison_chart(res, ASSETS_DIR / "walkforward_folds.svg")
        generate_relative_improvement_chart(res, ASSETS_DIR / "relative_improvement.svg")
        print(f"Updated visual charts in {ASSETS_DIR}")
    except Exception as e:
        print(f"Note: Could not auto-generate charts ({e}).")
