"""Unit tests for financial loss functions and statistical corrections."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from walkforward import qlike, smear  # noqa: E402


def test_qlike_zero_for_perfect_prediction():
    """QLIKE should be zero when forecast matches true volatility."""
    true_vol = np.array([0.015, 0.02, 0.035])
    assert np.isclose(qlike(true_vol, true_vol), 0.0, atol=1e-7)


def test_qlike_strictly_positive_for_errors():
    """QLIKE must be strictly positive when forecast diverges from truth."""
    true_vol = np.array([0.02])
    pred_vol = np.array([0.025])
    assert qlike(true_vol, pred_vol) > 0.0


def test_qlike_asymmetric_penalty():
    """QLIKE penalizes underestimating volatility more than overestimating it.
    
    If true vol is 0.02:
    - Underpredicting by 0.01 (predicting 0.01) represents high unhedged tail risk.
    - Overpredicting by 0.01 (predicting 0.03) is cautious/conservative.
    QLIKE(true=0.02, pred=0.01) must be strictly greater than QLIKE(true=0.02, pred=0.03).
    """
    true_vol = np.array([0.02])
    under_pred = np.array([0.01])
    over_pred = np.array([0.03])
    
    loss_under = qlike(true_vol, under_pred)
    loss_over = qlike(true_vol, over_pred)
    
    assert loss_under > loss_over, f"Expected loss_under ({loss_under}) > loss_over ({loss_over})"


def test_duan_smearing_factor():
    """Duan smearing factor must be positive and close to 1 for zero-mean residuals."""
    class MockModel:
        def predict(self, X):
            return X["feat"].values

    import pandas as pd
    # When y = feat (perfect prediction), residuals are 0, exp(0) = 1.0
    df = pd.DataFrame({"feat": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    factor = smear(MockModel(), df, ["feat"])
    assert np.isclose(factor, 1.0)

