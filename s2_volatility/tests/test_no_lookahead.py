import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from features import FEATURES, make_features  # noqa: E402


def synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=n),
                         "close": close, "volume": rng.integers(1e5, 1e6, n)})


def test_features_unchanged_by_future_data():
    full = make_features(synthetic())
    for t in [60, 150, 300]:
        trunc = make_features(synthetic().iloc[: t + 1])
        a = full.loc[t, FEATURES + ["ewma"]].astype(float).values
        b = trunc.loc[t, FEATURES + ["ewma"]].astype(float).values
        assert np.allclose(a, b, equal_nan=True), f"look-ahead at row {t}"


def test_target_uses_only_future_returns():
    g = make_features(synthetic())
    r = np.log(g["close"]).diff()
    t, h = 100, 5
    expected = np.sqrt((r.iloc[t + 1 : t + 1 + h] ** 2).mean())
    assert np.isclose(g.loc[t, "fwd_rv"], expected)
