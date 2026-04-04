"""
market_impact.py
----------------
Almgren-Chriss (2001) optimal execution and market impact model.

This module answers the critical question every prop desk asks before
allocating to a new strategy: "What is the capacity?"

The Almgren-Chriss Model
------------------------
We model execution of N shares (or option contracts) over T periods.
At each period, we trade x_t contracts. The execution price deviates
from the mid-price by:

    Temporary impact:   eta * (x_t / V_t)^0.6 * sigma
    Permanent impact:   gamma * (x_t / V_t) * sigma

where V_t is average daily volume and sigma is daily volatility.

The total expected shortfall (cost of execution) is:

    E[cost] = 0.5 * gamma * X^2 * sigma^2 / V
              + eta * sigma * sqrt(X/V) * f(T)

where X is total order size and f(T) is a function of the trading horizon.

We use this to:
    1. Compute all-in transaction costs at James's current strategy size
    2. Find the notional at which execution costs exceed the strategy's edge
    3. Plot the efficient frontier: execution speed vs. total impact cost

Reference:
    Almgren, R. & Chriss, N. (2001). Optimal execution of portfolio transactions.
    Journal of Risk, 3(2), 5–39.

    Kyle, A.S. (1985). Continuous auctions and insider trading.
    Econometrica, 53(6), 1315–1335.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Market Parameters Container
# ---------------------------------------------------------------------------

@dataclass
class MarketParams:
    """
    Market microstructure parameters for SPX options execution.

    Attributes
    ----------
    daily_volume       : average daily notional volume in dollars
                         (SPX options: ~$50B/day for liquid strikes)
    daily_vol          : underlying daily volatility (e.g. 0.012 for 1.2%/day)
    bid_ask_spread_pct : bid-ask spread as percentage of mid-price
                         (SPX ATM options: ~0.10% to 0.25%)
    eta                : temporary impact coefficient (calibrated to SPX)
    gamma              : permanent impact coefficient (calibrated to SPX)
    """
    daily_volume: float = 50_000_000_000.0    # $50B SPX options daily volume
    daily_vol: float = 0.012                   # ~19% annualised vol / sqrt(252)
    bid_ask_spread_pct: float = 0.0015         # 15 bps bid-ask
    eta: float = 0.142                         # Almgren-Chriss temp impact
    gamma: float = 0.314                       # Almgren-Chriss perm impact


# ---------------------------------------------------------------------------
# Core Market Impact Calculations
# ---------------------------------------------------------------------------

def temporary_impact(
    order_size: float,
    daily_volume: float,
    daily_vol: float,
    eta: float = 0.142,
) -> float:
    """
    Compute temporary (transient) market impact cost per dollar traded.

    Temporary impact is the additional cost of trading quickly versus slowly,
    and recovers after the trade is complete.

    Impact_temp = eta * (order_size / daily_volume)^0.6 * daily_vol

    Parameters
    ----------
    order_size   : size of order in dollars
    daily_volume : average daily volume in dollars
    daily_vol    : daily price volatility (decimal)
    eta          : temporary impact coefficient

    Returns
    -------
    float : fractional cost (e.g. 0.002 = 20bps of order value)
    """
    participation_rate = order_size / daily_volume
    return eta * (participation_rate ** 0.6) * daily_vol


def permanent_impact(
    order_size: float,
    daily_volume: float,
    daily_vol: float,
    gamma: float = 0.314,
) -> float:
    """
    Compute permanent market impact cost per dollar traded.

    Permanent impact is the lasting price change caused by the trade
    (information content of order flow).

    Impact_perm = 0.5 * gamma * (order_size / daily_volume) * daily_vol

    The 0.5 factor accounts for the fact that impact accumulates linearly.
    """
    participation_rate = order_size / daily_volume
    return 0.5 * gamma * participation_rate * daily_vol


def total_execution_cost(
    order_size: float,
    params: MarketParams,
    n_periods: int = 5,
) -> Dict[str, float]:
    """
    Compute total all-in execution cost for a given order size.

    Splits the order evenly over n_periods (days) and sums:
        1. Bid-ask spread cost (paid once, regardless of periods)
        2. Temporary impact (scales with order size per period)
        3. Permanent impact (paid on entire size, regardless of n_periods)

    Parameters
    ----------
    order_size : total order notional in dollars
    params     : MarketParams object
    n_periods  : number of trading periods to spread execution over

    Returns
    -------
    dict with keys:
        bid_ask_cost_bps  : bid-ask cost in basis points
        temp_impact_bps   : temporary impact cost in basis points
        perm_impact_bps   : permanent impact cost in basis points
        total_cost_bps    : total all-in cost in basis points
        total_cost_dollars: total dollar cost
    """
    period_size = order_size / n_periods

    bid_ask_bps = params.bid_ask_spread_pct * 10_000

    # Each period we trade period_size; total temp impact sums over n_periods.
    # Because temp_impact has a 0.6 power law:
    #   total = n_periods * eta * (period_size/V)^0.6 * sigma
    #         = n_periods * eta * (X/(n*V))^0.6 * sigma
    #         = n^{1-0.6} * eta * (X/V)^0.6 * sigma
    # So total temp impact decreases as n_periods increases (n^0.4 factor).
    temp_impact_per_period = temporary_impact(
        period_size, params.daily_volume, params.daily_vol, params.eta
    )
    temp_bps = temp_impact_per_period * n_periods * 10_000

    perm_bps = (
        permanent_impact(order_size, params.daily_volume, params.daily_vol, params.gamma)
        * 10_000
    )

    total_bps = bid_ask_bps + temp_bps + perm_bps

    return {
        "bid_ask_cost_bps":   round(bid_ask_bps, 2),
        "temp_impact_bps":    round(temp_bps, 4),
        "perm_impact_bps":    round(perm_bps, 4),
        "total_cost_bps":     round(total_bps, 2),
        "total_cost_dollars": round(order_size * total_bps / 10_000, 2),
    }


# ---------------------------------------------------------------------------
# Strategy Capacity Analysis
# ---------------------------------------------------------------------------

def capacity_curve(
    params: MarketParams,
    strategy_edge_bps: float,
    order_sizes: Optional[np.ndarray] = None,
    n_periods: int = 5,
) -> pd.DataFrame:
    """
    Compute strategy capacity as a function of order size.

    The strategy's edge (net alpha) is consumed by execution costs as
    order size grows. The breakeven point — where total_cost_bps =
    strategy_edge_bps — is the practical capacity ceiling.

    Parameters
    ----------
    params            : MarketParams
    strategy_edge_bps : net alpha in basis points per trade
                        (from backtesting, before execution costs)
    order_sizes       : array of order sizes to evaluate (dollars)
                        default: $10K to $10B log-spaced
    n_periods         : execution periods

    Returns
    -------
    DataFrame with columns:
        order_size_usd, total_cost_bps, net_edge_bps, profitable (bool)
    """
    if order_sizes is None:
        order_sizes = np.logspace(4, 10, 100)  # $10K to $10B

    rows = []
    for size in order_sizes:
        costs = total_execution_cost(size, params, n_periods)
        net_edge = strategy_edge_bps - costs["total_cost_bps"]
        rows.append({
            "order_size_usd":  size,
            "order_size_label": f"${size/1e6:.1f}M" if size >= 1e6 else f"${size/1e3:.0f}K",
            "total_cost_bps":  costs["total_cost_bps"],
            "net_edge_bps":    net_edge,
            "profitable":      net_edge > 0,
        })

    return pd.DataFrame(rows)


def find_capacity_limit(
    params: MarketParams,
    strategy_edge_bps: float,
    n_periods: int = 5,
) -> float:
    """
    Find the notional order size at which strategy edge is fully consumed
    by execution costs (the capacity ceiling).

    Uses binary search between $1K and $100B.

    Returns
    -------
    float : capacity limit in dollars (order size where net_edge = 0)
    """
    lo, hi = 1_000.0, 100_000_000_000.0
    tol = 1_000.0  # $1K precision

    for _ in range(100):
        mid = 0.5 * (lo + hi)
        costs = total_execution_cost(mid, params, n_periods)
        net_edge = strategy_edge_bps - costs["total_cost_bps"]
        if net_edge > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break

    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Efficient Frontier: Execution Speed vs. Impact Cost
# ---------------------------------------------------------------------------

def execution_frontier(
    order_size: float,
    params: MarketParams,
    n_periods_range: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Compute the efficient frontier between execution speed and impact cost.

    Trading faster (fewer periods) increases temporary impact but reduces
    market risk (the underlying can move adversely while executing).
    Trading slower (more periods) reduces temporary impact but increases
    market risk exposure.

    Parameters
    ----------
    order_size      : total order notional in dollars
    params          : MarketParams
    n_periods_range : array of execution horizons (number of days)

    Returns
    -------
    DataFrame with columns:
        n_periods, total_cost_bps, market_risk_bps, total_risk_adjusted_cost
    """
    if n_periods_range is None:
        n_periods_range = np.arange(1, 21)

    rows = []
    for n in n_periods_range:
        costs = total_execution_cost(order_size, params, int(n))

        # Market risk: sigma * sqrt(n_periods) * 10000 (bps of underlying move)
        # This is the P&L variance from the residual position during execution
        market_risk_bps = params.daily_vol * np.sqrt(n) * 10_000

        rows.append({
            "n_periods":        int(n),
            "execution_days":   int(n),
            "impact_cost_bps":  costs["total_cost_bps"],
            "market_risk_bps":  round(market_risk_bps, 2),
            # Variance-adjusted total: sqrt(impact^2 + market_risk^2)
            "total_risk_bps":   round(
                np.sqrt(costs["total_cost_bps"]**2 + market_risk_bps**2), 2
            ),
        })

    return pd.DataFrame(rows)
