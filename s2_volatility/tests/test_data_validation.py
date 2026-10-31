"""Unit tests for data sanitization and validation checks."""
import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from data import validate  # noqa: E402


def test_validate_rejects_duplicates():
    df = pd.DataFrame({
        "ticker": ["RELIANCE.NS", "RELIANCE.NS"],
        "date": ["2023-01-01", "2023-01-01"],
        "close": [2500.0, 2510.0],
        "volume": [10000, 12000]
    })
    with pytest.raises(ValueError, match="duplicate"):
        validate(df)


def test_validate_rejects_non_positive_prices():
    df = pd.DataFrame({
        "ticker": ["TCS.NS", "TCS.NS"],
        "date": ["2023-01-01", "2023-01-02"],
        "close": [3200.0, -5.0],
        "volume": [10000, 12000]
    })
    with pytest.raises(ValueError, match="non-positive"):
        validate(df)


def test_validate_flags_anomalous_gaps():
    df = pd.DataFrame({
        "ticker": ["INFY.NS", "INFY.NS"],
        "date": ["2023-01-01", "2023-01-20"],  # 19 day gap
        "close": [1400.0, 1420.0],
        "volume": [50000, 60000]
    })
    issues = validate(df, max_gap_days=7)
    assert any("gaps > 7 days" in iss for iss in issues)

