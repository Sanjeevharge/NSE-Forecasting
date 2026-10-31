"""Features use only information available at the close of day t."""
import numpy as np
import pandas as pd

H = 5          # forecast horizon (trading days)
EPS = 1e-6


def make_features(g: pd.DataFrame, h: int = H) -> pd.DataFrame:
    g = g.sort_values("date").reset_index(drop=True).copy()
    r = np.log(g["close"]).diff()
    r2 = r ** 2
    g["r"] = r
    g["rv1"] = r.abs()
    g["rv5"] = np.sqrt(r2.rolling(5).mean())
    g["rv22"] = np.sqrt(r2.rolling(22).mean())
    g["ewma"] = np.sqrt(r2.ewm(alpha=0.06, adjust=False).mean())     # RiskMetrics lambda=0.94
    g["neg_r"] = (r < 0).astype(float) * r.abs()
    g["vol_ratio"] = np.log((g["volume"] + 1) / (g["volume"].rolling(22).mean() + 1))
    for c in ["rv1", "rv5", "rv22"]:
        g[f"log_{c}"] = np.log(g[c] + EPS)
    # target: realized vol over days t+1..t+h. rolling(h) at t+h covers t+1..t+h.
    fwd_var = r2.rolling(h).mean().shift(-h)
    g["fwd_rv"] = np.sqrt(fwd_var)
    g["y"] = np.log(g["fwd_rv"] + EPS)
    return g


FEATURES = ["log_rv1", "log_rv5", "log_rv22", "r", "neg_r", "vol_ratio"]
HAR = ["log_rv1", "log_rv5", "log_rv22"]


def build(prices: pd.DataFrame) -> pd.DataFrame:
    out = [make_features(g) for _, g in prices.groupby("ticker")]
    df = pd.concat(out, ignore_index=True)
    return df.dropna(subset=FEATURES + ["ewma"]).reset_index(drop=True)
