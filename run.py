"""Root CLI entrypoint for NSE Volatility Forecasting Pipeline."""
import argparse
import sys
from pathlib import Path

# Add package directory to sys.path
BASE_DIR = Path(__file__).parent
PKG_DIR = BASE_DIR / "s2_volatility"
sys.path.insert(0, str(PKG_DIR))


def main():
    parser = argparse.ArgumentParser(
        description="NSE Volatility Forecasting: Econometrics vs. Machine Learning Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --evaluate       # Run walk-forward backtest & update visualizations
  python run.py --test           # Run unit test suite (look-ahead, metrics, validation)
  python run.py --visualize      # Re-generate SVG charts in assets/
  python run.py --download       # Fetch fresh NSE price data from Yahoo Finance
        """
    )
    parser.add_argument("--evaluate", action="store_true", help="Run walk-forward simulation across all folds")
    parser.add_argument("--test", action="store_true", help="Execute pytest verification suite")
    parser.add_argument("--visualize", action="store_true", help="Generate SVG charts from results.csv")
    parser.add_argument("--profiles", action="store_true", help="Run the risk-profile layer (sizing + vol alerts) on out-of-sample forecasts")
    parser.add_argument("--download", action="store_true", help="Download raw prices from Yahoo Finance")
    
    args = parser.parse_args()

    # Default action if no arguments provided
    if not any([args.evaluate, args.test, args.visualize, args.download, args.profiles]):
        parser.print_help()
        print("\n[INFO] Running complete walk-forward evaluation by default:\n")
        args.evaluate = True

    if args.test:
        import pytest
        print("\n>>> Running Test Suite...")
        ret = pytest.main(["-v", str(PKG_DIR / "tests")])
        if ret != 0:
            sys.exit(ret)

    if args.download:
        import data
        print("\n>>> Downloading fresh price data from Yahoo Finance...")
        df = data.download()
        issues = data.validate(df)
        data.PRICES.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(data.PRICES, index=False)
        print(f"Saved {len(df):,} price records -> {data.PRICES}")
        if issues:
            print("Validation warnings:\n" + "\n".join(issues))
        else:
            print("Validation checks: PASSED")

    if args.evaluate:
        import pandas as pd
        from data import PRICES, RESULTS
        import features
        import walkforward

        print("\n>>> Loading Historical Price Data...")
        if not PRICES.exists():
            print(f"Error: {PRICES} not found. Run with --download first.")
            sys.exit(1)
        prices = pd.read_csv(PRICES, parse_dates=["date"])
        print(f"Loaded {len(prices):,} rows across {prices['ticker'].nunique()} tickers.")

        print(">>> Engineering Financial Features...")
        feature_df = features.build(prices)

        print(">>> Running Walk-Forward Cross-Validation (Embargo H=5)...")
        res = walkforward.run(feature_df)
        RESULTS.parent.mkdir(exist_ok=True, parents=True)
        res.to_csv(RESULTS, index=False)
        print(f"Metrics saved to {RESULTS}")
        
        walkforward.summarize(res)

        # Trigger visual generation
        try:
            import generate_visuals
            from generate_visuals import (
                ASSETS_DIR,
                generate_fold_comparison_chart,
                generate_regime_chart,
                generate_relative_improvement_chart,
            )
            generate_regime_chart(res, ASSETS_DIR / "regime_performance.svg")
            generate_fold_comparison_chart(res, ASSETS_DIR / "walkforward_folds.svg")
            generate_relative_improvement_chart(res, ASSETS_DIR / "relative_improvement.svg")
            print(f">>> Charts updated in {ASSETS_DIR}")
        except Exception as e:
            print(f"Note: Visuals update skipped: {e}")

    if args.profiles:
        import profiles
        profiles.main()

    if args.visualize and not args.evaluate:
        import pandas as pd
        from data import RESULTS
        import generate_visuals
        from generate_visuals import (
            ASSETS_DIR,
            generate_fold_comparison_chart,
            generate_regime_chart,
            generate_relative_improvement_chart,
        )
        if not RESULTS.exists():
            print(f"Error: {RESULTS} not found. Run --evaluate first.")
            sys.exit(1)
        res = pd.read_csv(RESULTS)
        generate_regime_chart(res, ASSETS_DIR / "regime_performance.svg")
        generate_fold_comparison_chart(res, ASSETS_DIR / "walkforward_folds.svg")
        generate_relative_improvement_chart(res, ASSETS_DIR / "relative_improvement.svg")
        print(f">>> Charts updated in {ASSETS_DIR}")


if __name__ == "__main__":
    main()

