"""
utils.py
--------
Core utility functions: Black-Scholes Greeks, data loading,
and performance metric calculations.

All functions are pure (no side effects) and fully type-annotated
so the test suite can verify them in isolation.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Black-Scholes Greeks
# ---------------------------------------------------------------------------

def bs_d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Compute d1 from the Black-Scholes formula.

    Parameters
    ----------
    S     : spot price
    K     : strike price
    T     : time to expiry in years (must be > 0)
    r     : continuously compounded risk-free rate
    sigma : implied volatility (annualised)

    Returns
    -------
    float : d1 value
    """
    if T <= 0 or sigma <= 0:
        raise ValueError(f"T and sigma must be positive. Got T={T}, sigma={sigma}")
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def bs_d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute d2 = d1 - sigma*sqrt(T)."""
    return bs_d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call price."""
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes put price via put-call parity."""
    call = bs_call_price(S, K, T, r, sigma)
    return call - S + K * np.exp(-r * T)


def bs_delta(S: float, K: float, T: float, r: float, sigma: float,
             option_type: str = "call") -> float:
    """
    Black-Scholes delta.

    Parameters
    ----------
    option_type : "call" or "put"
    """
    d1 = bs_d1(S, K, T, r, sigma)
    if option_type.lower() == "call":
        return norm.cdf(d1)
    elif option_type.lower() == "put":
        return norm.cdf(d1) - 1.0
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")


def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes gamma (same for calls and puts)."""
    d1 = bs_d1(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes vega.

    Returns change in option price per 1-point (100%) move in IV.
    Divide by 100 to get price change per 1% IV move.
    """
    d1 = bs_d1(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)


def bs_theta(S: float, K: float, T: float, r: float, sigma: float,
             option_type: str = "call") -> float:
    """
    Black-Scholes theta (per calendar day).

    The standard formula gives theta per year; we divide by 365.
    """
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if option_type.lower() == "call":
        theta_annual = term1 - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        theta_annual = term1 + r * K * np.exp(-r * T) * norm.cdf(-d2)
    return theta_annual / 365.0


def implied_vol_bisection(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    tol: float = 1e-6,
    max_iter: int = 200,
) -> float:
    """
    Compute implied volatility via bisection search.

    Uses a bounded [1e-4, 10.0] search range (1bp to 1000% vol).
    Returns NaN if the market price is outside the no-arbitrage bounds.

    Parameters
    ----------
    market_price : observed mid-price of the option
    tol          : convergence tolerance on the IV value
    max_iter     : maximum bisection iterations

    Returns
    -------
    float : implied volatility (annualised), or NaN if not solvable
    """
    if T <= 0:
        return np.nan

    price_fn = bs_call_price if option_type.lower() == "call" else bs_put_price

    # Intrinsic value bounds check
    intrinsic = max(0.0, S - K * np.exp(-r * T)) if option_type == "call" \
        else max(0.0, K * np.exp(-r * T) - S)
    if market_price < intrinsic - tol:
        return np.nan

    lo, hi = 1e-4, 10.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if price_fn(S, K, T, r, mid) < market_price:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Log-moneyness helper
# ---------------------------------------------------------------------------

def log_moneyness(S: float, K: float, T: float, r: float) -> float:
    """
    Compute log-moneyness k = log(K / F) where F = S * exp(r * T)
    is the forward price.

    This is the standard x-axis for SVI parameterization.
    """
    F = S * np.exp(r * T)
    return np.log(K / F)


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Annualised Sharpe ratio (assuming risk-free rate = 0 for excess returns).

    Parameters
    ----------
    returns          : daily P&L returns (not percent)
    periods_per_year : trading days per year (252 for equities/options)
    """
    if returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(periods_per_year)


def max_drawdown(returns: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown of a return series.

    Returns a negative number (e.g. -0.142 for a 14.2% drawdown).
    """
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calmar ratio = Annualised Return / |Max Drawdown|.
    Returns NaN if max_drawdown is 0.
    """
    ann_return = (1 + returns.mean()) ** periods_per_year - 1
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return np.nan
    return ann_return / mdd


def annualised_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Geometric annualised return."""
    return (1 + returns.mean()) ** periods_per_year - 1


def annualised_vol(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualised volatility of returns."""
    return returns.std() * np.sqrt(periods_per_year)


def performance_summary(returns: pd.Series,
                        periods_per_year: int = 252) -> dict:
    """
    Compute a full set of performance metrics in one call.

    Returns
    -------
    dict with keys: sharpe, annualised_return, annualised_vol,
                    max_drawdown, calmar, win_rate, num_days
    """
    return {
        "sharpe":            round(sharpe_ratio(returns, periods_per_year), 4),
        "annualised_return": round(annualised_return(returns, periods_per_year), 4),
        "annualised_vol":    round(annualised_vol(returns, periods_per_year), 4),
        "max_drawdown":      round(max_drawdown(returns), 4),
        "calmar":            round(calmar_ratio(returns, periods_per_year), 4),
        "win_rate":          round((returns > 0).mean(), 4),
        "num_days":          len(returns),
    }


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_spx_prices(start: str = "2018-01-01",
                    end: str = "2024-12-31") -> pd.Series:
    """
    Load SPX daily closing prices from Yahoo Finance.

    Returns a pd.Series indexed by date.
    Requires: pip install yfinance
    """
    try:
        import yfinance as yf
        spx = yf.download("^GSPC", start=start, end=end,
                          auto_adjust=True, progress=False)["Close"]
        spx.index = pd.to_datetime(spx.index)
        spx.name = "SPX"
        return spx.dropna()
    except ImportError:
        raise ImportError("yfinance is required. Run: pip install yfinance")


def load_vix_data(start: str = "2018-01-01",
                  end: str = "2024-12-31") -> pd.DataFrame:
    """
    Load VIX spot and VIX futures term structure proxies.

    VIX  → ^VIX  (CBOE Volatility Index, 30-day)
    VVIX → ^VVIX (Volatility of VIX, measures vol-of-vol)
    VX1  → approximated via VXX (short-term VIX futures ETF)

    Returns a DataFrame with columns: VIX, VVIX, VXX
    """
    try:
        import yfinance as yf
        tickers = {"^VIX": "VIX", "^VVIX": "VVIX", "VXX": "VXX"}
        frames = {}
        for ticker, name in tickers.items():
            try:
                s = yf.download(ticker, start=start, end=end,
                                auto_adjust=True, progress=False)["Close"]
                s.name = name
                frames[name] = s
            except Exception:
                warnings.warn(f"Could not load {ticker}")
        df = pd.DataFrame(frames).dropna()
        df.index = pd.to_datetime(df.index)
        return df
    except ImportError:
        raise ImportError("yfinance is required. Run: pip install yfinance")


def load_risk_free_rate(start: str = "2018-01-01",
                        end: str = "2024-12-31") -> pd.Series:
    """
    Load 3-month US T-bill rate from FRED as a proxy for the risk-free rate.

    Returns annualised rate as a decimal (e.g. 0.053 for 5.3%).
    Requires: pip install pandas-datareader
    """
    try:
        import pandas_datareader as pdr
        rf = pdr.get_data_fred("TB3MS", start=start, end=end)["TB3MS"]
        rf = rf / 100.0  # convert percent to decimal
        rf.index = pd.to_datetime(rf.index)
        rf.name = "rf"
        # Forward-fill to daily frequency
        idx = pd.date_range(start=start, end=end, freq="B")
        rf = rf.reindex(idx, method="ffill")
        return rf.dropna()
    except ImportError:
        raise ImportError(
            "pandas-datareader is required. Run: pip install pandas-datareader"
        )
