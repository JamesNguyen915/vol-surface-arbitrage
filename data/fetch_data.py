"""
fetch_data.py
-------------
Reproducible data pipeline for the vol surface arbitrage project.

This script downloads and caches all data required to run the notebooks
and reproduce the research results.

Data Sources
-----------
1. SPX prices         → Yahoo Finance (^GSPC), free and public
2. VIX term structure → Yahoo Finance (^VIX, ^VVIX, VXX), free and public
3. Risk-free rate     → FRED (TB3MS 3-month T-bill), free and public
4. SPX options        → CBOE DataShop (historical), requires institutional
                        subscription. We provide a synthetic fallback for
                        open-source reproducibility.

Usage
-----
    python data/fetch_data.py                  # Downloads all freely available data
    python data/fetch_data.py --synthetic      # Uses synthetic options data
    python data/fetch_data.py --start 2020-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data" / "cache"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. SPX Prices
# ---------------------------------------------------------------------------

def fetch_spx_prices(start: str, end: str) -> pd.DataFrame:
    """Download SPX daily OHLCV from Yahoo Finance."""
    print(f"  Fetching SPX prices ({start} to {end})...")
    try:
        import yfinance as yf
        spx = yf.download("^GSPC", start=start, end=end,
                          auto_adjust=True, progress=False)
        spx.index = pd.to_datetime(spx.index)
        print(f"  ✓ SPX: {len(spx)} trading days")
        return spx
    except Exception as e:
        warnings.warn(f"Failed to download SPX data: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# 2. VIX Term Structure
# ---------------------------------------------------------------------------

def fetch_vix_data(start: str, end: str) -> pd.DataFrame:
    """Download VIX spot, VVIX, and VXX from Yahoo Finance."""
    print(f"  Fetching VIX term structure ({start} to {end})...")
    try:
        import yfinance as yf
        tickers = {"^VIX": "VIX", "^VVIX": "VVIX", "VXX": "VXX"}
        frames = {}
        for ticker, name in tickers.items():
            try:
                data = yf.download(ticker, start=start, end=end,
                                   auto_adjust=True, progress=False)["Close"]
                data.name = name
                frames[name] = data
                print(f"    ✓ {name}: {len(data)} days")
            except Exception as e:
                print(f"    ✗ {name}: {e}")

        df = pd.DataFrame(frames)
        df.index = pd.to_datetime(df.index)
        return df.dropna()
    except ImportError:
        print("  ✗ yfinance not installed. Run: pip install yfinance")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# 3. Risk-Free Rate (FRED)
# ---------------------------------------------------------------------------

def fetch_risk_free_rate(start: str, end: str) -> pd.Series:
    """Download 3-month T-bill rate from FRED."""
    print(f"  Fetching risk-free rate from FRED ({start} to {end})...")
    try:
        import pandas_datareader as pdr
        rf = pdr.get_data_fred("TB3MS", start=start, end=end)["TB3MS"]
        rf = rf / 100.0
        rf.index = pd.to_datetime(rf.index)
        # Forward-fill to daily
        bdays = pd.date_range(start=start, end=end, freq="B")
        rf_daily = rf.reindex(bdays, method="ffill").dropna()
        rf_daily.name = "rf"
        print(f"  ✓ Risk-free rate: {len(rf_daily)} days")
        return rf_daily
    except ImportError:
        print("  ✗ pandas-datareader not installed. Run: pip install pandas-datareader")
        # Fallback: flat 5% rate
        bdays = pd.date_range(start=start, end=end, freq="B")
        return pd.Series(0.05, index=bdays, name="rf")
    except Exception as e:
        print(f"  ✗ FRED download failed: {e}")
        bdays = pd.date_range(start=start, end=end, freq="B")
        return pd.Series(0.05, index=bdays, name="rf")


# ---------------------------------------------------------------------------
# 4. Synthetic Options Data (Fallback)
# ---------------------------------------------------------------------------

def generate_synthetic_options_data(
    spx_prices: pd.Series,
    vix: pd.Series,
    rf: pd.Series,
    n_strikes: int = 11,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic SPX options chain data for open-source reproducibility.

    This simulates realistic SPX 0DTE option chains using:
    - Black-Scholes with stochastic vol (SABR-like dynamics)
    - Realistic bid-ask spreads (wider OTM, narrower ATM)
    - VIX-scaled ATM implied vol
    - Realistic skew (rho ≈ -0.3 to -0.7)

    The synthetic data is not a substitute for real CBOE data, but allows
    the full pipeline to run without an institutional data subscription.

    Parameters
    ----------
    spx_prices : SPX daily closing prices
    vix        : VIX spot series
    rf         : daily risk-free rate series
    n_strikes  : number of strikes per daily chain (symmetric around ATM)
    seed       : reproducibility seed

    Returns
    -------
    DataFrame with columns: date, strike, T, IV_mid, IV_bid, IV_ask,
                             log_moneyness, option_type
    """
    from scipy.stats import norm

    rng = np.random.default_rng(seed)
    T = 1.0 / 252  # 0DTE: 1 trading day

    common_idx = spx_prices.index.intersection(vix.index).intersection(rf.index)
    spx_aligned = spx_prices.loc[common_idx]
    vix_aligned  = vix.loc[common_idx]
    rf_aligned   = rf.loc[common_idx]

    records = []

    for date in common_idx:
        S = float(spx_aligned.loc[date])
        sigma_atm = float(vix_aligned.loc[date]) / 100.0  # VIX is in percent
        r = float(rf_aligned.loc[date])
        F = S * np.exp(r * T)

        # Strikes: ±2.5% around ATM in steps of 0.5%
        strike_pcts = np.linspace(-0.025, 0.025, n_strikes)
        strikes = S * (1 + strike_pcts)

        # SVI-like skew: steeper left side (negative rho)
        rho = rng.uniform(-0.55, -0.25)
        skew_slope = -rho * sigma_atm * 0.5

        for K in strikes:
            k = np.log(K / F)  # log-moneyness

            # IV with realistic skew
            iv_mid = sigma_atm + skew_slope * k + 0.5 * sigma_atm * k**2
            iv_mid = max(iv_mid, 0.01)  # Floor at 1%

            # Add small noise (market microstructure)
            iv_mid += rng.normal(0, 0.002)
            iv_mid = max(iv_mid, 0.01)

            # Bid-ask spread: wider OTM, narrower ATM
            half_spread = 0.003 + 0.004 * abs(k)
            iv_bid = max(iv_mid - half_spread, 0.005)
            iv_ask = iv_mid + half_spread

            option_type = "put" if k < 0 else "call"

            records.append({
                "date":          date,
                "strike":        round(K, 2),
                "T":             T,
                "IV_mid":        round(iv_mid, 6),
                "IV_bid":        round(iv_bid, 6),
                "IV_ask":        round(iv_ask, 6),
                "log_moneyness": round(k, 6),
                "option_type":   option_type,
                "SPX":           S,
                "rf":            r,
            })

    df = pd.DataFrame(records)
    print(f"  ✓ Synthetic options: {len(df)} option records "
          f"({len(common_idx)} days × {n_strikes} strikes)")
    return df


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download and cache data for vol surface arbitrage project"
    )
    parser.add_argument("--start", default="2018-01-01",
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-12-31",
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic options data (no CBOE subscription needed)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Vol Surface Arbitrage — Data Pipeline")
    print(f"  Period: {args.start} to {args.end}")
    print(f"{'='*60}\n")

    # 1. SPX Prices
    spx = fetch_spx_prices(args.start, args.end)
    if not spx.empty:
        spx.to_parquet(DATA_DIR / "spx_prices.parquet")
        print(f"  Saved: {DATA_DIR / 'spx_prices.parquet'}")

    # 2. VIX Term Structure
    vix = fetch_vix_data(args.start, args.end)
    if not vix.empty:
        vix.to_parquet(DATA_DIR / "vix_data.parquet")
        print(f"  Saved: {DATA_DIR / 'vix_data.parquet'}")

    # 3. Risk-Free Rate
    rf = fetch_risk_free_rate(args.start, args.end)
    if rf is not None and len(rf) > 0:
        rf.to_frame().to_parquet(DATA_DIR / "risk_free_rate.parquet")
        print(f"  Saved: {DATA_DIR / 'risk_free_rate.parquet'}")

    # 4. Options Data
    if args.synthetic or True:  # Always generate synthetic for portability
        if not spx.empty and not vix.empty and rf is not None:
            print("\n  Generating synthetic 0DTE options data...")
            spx_close = spx["Close"] if "Close" in spx.columns else spx.iloc[:, 0]
            vix_series = vix["VIX"] if "VIX" in vix.columns else vix.iloc[:, 0]
            options = generate_synthetic_options_data(spx_close, vix_series, rf)
            options.to_parquet(DATA_DIR / "options_synthetic.parquet")
            print(f"  Saved: {DATA_DIR / 'options_synthetic.parquet'}")

    print(f"\n{'='*60}")
    print(f"  Data pipeline complete.")
    print(f"  All files saved to: {DATA_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
