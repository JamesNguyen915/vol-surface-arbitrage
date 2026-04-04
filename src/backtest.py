"""
backtest.py
-----------
Walk-forward backtesting engine with regime-aware out-of-sample splits.

Design Principles
-----------------
1. No lookahead bias: all signals are computed using only data available
   at time t-1. The signal on day t uses parameters estimated through day t-1.

2. Realistic transaction costs: each trade pays bid-ask spread + estimated
   market impact proportional to order size.

3. Regime-aware out-of-sample: we test on regimes NOT seen during training,
   not just on dates after the training period. This is a stricter and more
   honest test of generalisability.

4. Vectorised where possible: the engine uses pandas/numpy operations
   to be both fast and auditable.

Strategy Logic
--------------
Given:
    signal_t   : z-scored IV surface residual at time t
    regime_t   : HMM regime label at time t
    beta_t     : Kalman filter hedge ratio at time t

Entry:
    If signal_t > entry_z(regime_t) → sell the vol residual (short vega)
    If signal_t < -entry_z(regime_t) → buy the vol residual (long vega)
    Size the position by: max_position_scale(regime_t) * base_notional

Exit:
    Close position when |signal_t| < exit_z(regime_t)
    Force-close at end of each trading day (0DTE: no overnight risk)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .hmm_regime import get_regime_entry_thresholds
from .market_impact import MarketParams, total_execution_cost
from .utils import performance_summary


# ---------------------------------------------------------------------------
# Trade Record
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """Record of a single completed trade."""
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    direction: int          # +1 = long vol, -1 = short vol
    entry_signal: float
    exit_signal: float
    entry_regime: int
    exit_regime: int
    gross_pnl: float
    transaction_cost: float
    net_pnl: float
    holding_days: int


# ---------------------------------------------------------------------------
# Backtest Configuration
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """
    Configuration for the walk-forward backtest.

    Parameters
    ----------
    base_notional      : base position size in dollars per trade
    transaction_cost_bps: flat transaction cost assumption (bps)
                          used when market_params is None
    market_params      : optional MarketParams for Almgren-Chriss costs
    force_daily_close  : if True, close all positions at end of day
                         (appropriate for 0DTE options: they expire worthless)
    max_holding_days   : force-close after this many days (risk limit)
    """
    base_notional: float = 50_000.0
    transaction_cost_bps: float = 20.0
    market_params: Optional[MarketParams] = None
    force_daily_close: bool = True
    max_holding_days: int = 1


# ---------------------------------------------------------------------------
# Walk-Forward Backtest Engine
# ---------------------------------------------------------------------------

class WalkForwardBacktest:
    """
    Walk-forward backtesting engine with regime-aware out-of-sample splits.

    Usage
    -----
    bt = WalkForwardBacktest(config=config)
    results = bt.run(signals, regimes, returns_proxy)
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self._trades: List[Trade] = []
        self._daily_pnl: Optional[pd.Series] = None

    def _compute_transaction_cost(self, notional: float) -> float:
        """Compute dollar transaction cost for a given notional."""
        if self.config.market_params is not None:
            costs = total_execution_cost(notional, self.config.market_params)
            return notional * costs["total_cost_bps"] / 10_000
        else:
            return notional * self.config.transaction_cost_bps / 10_000

    def run(
        self,
        signals: pd.Series,
        regimes: pd.Series,
        returns_proxy: pd.Series,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Execute the walk-forward backtest.

        Parameters
        ----------
        signals       : z-scored IV surface residuals, indexed by date
        regimes       : HMM regime labels (0/1/2), indexed by date
        returns_proxy : daily return of the vol signal (estimated P&L per
                        unit of notional when signal mean-reverts by 1 z-score)
                        indexed by date
        verbose       : if True, print trade-by-trade details

        Returns
        -------
        results_df : DataFrame indexed by date with columns:
                        position, gross_pnl, transaction_cost, net_pnl,
                        cumulative_pnl, regime, signal
        """
        # Align all series to common index
        idx = signals.index.intersection(regimes.index).intersection(
            returns_proxy.index
        )
        signals = signals.reindex(idx)
        regimes = regimes.reindex(idx)
        returns_proxy = returns_proxy.reindex(idx)

        n = len(idx)
        position = np.zeros(n)
        gross_pnl = np.zeros(n)
        net_pnl = np.zeros(n)
        tcost = np.zeros(n)

        current_position = 0
        entry_idx: Optional[int] = None
        entry_signal: Optional[float] = None
        entry_regime: Optional[int] = None

        for t in range(n):
            sig = signals.iloc[t]
            reg = int(regimes.iloc[t])
            ret = returns_proxy.iloc[t]

            if np.isnan(sig) or np.isnan(reg) or np.isnan(ret):
                position[t] = current_position
                continue

            thresholds = get_regime_entry_thresholds(reg)
            entry_z = thresholds["entry_z"]
            exit_z = thresholds["exit_z"]
            size_scale = thresholds["max_position_scale"]
            notional = self.config.base_notional * size_scale

            # P&L from existing position (before close/open decisions)
            # returns_proxy represents the daily P&L per unit notional
            # for the direction we are holding
            if current_position != 0:
                daily_gross = current_position * ret * notional
                gross_pnl[t] = daily_gross

            # --- Exit logic ---
            should_exit = False

            if current_position != 0:
                # Exit on mean-reversion
                if abs(sig) < exit_z:
                    should_exit = True
                # Force-close 0DTE options at end of day
                if self.config.force_daily_close:
                    should_exit = True
                # Max holding period exceeded
                if entry_idx is not None and (t - entry_idx) >= self.config.max_holding_days:
                    should_exit = True

            if should_exit and current_position != 0:
                # Record trade
                cost = self._compute_transaction_cost(notional)
                tcost[t] += cost
                net_pnl[t] = gross_pnl[t] - cost

                if entry_idx is not None:
                    self._trades.append(Trade(
                        entry_date=idx[entry_idx],
                        exit_date=idx[t],
                        direction=current_position,
                        entry_signal=entry_signal,
                        exit_signal=sig,
                        entry_regime=entry_regime,
                        exit_regime=reg,
                        gross_pnl=gross_pnl[t],
                        transaction_cost=cost,
                        net_pnl=gross_pnl[t] - cost,
                        holding_days=t - entry_idx,
                    ))

                current_position = 0
                entry_idx = None

            # --- Entry logic ---
            if current_position == 0:
                new_position = 0
                if sig > entry_z:
                    new_position = -1  # Short vol: sell the overpriced residual
                elif sig < -entry_z:
                    new_position = 1   # Long vol: buy the underpriced residual

                if new_position != 0:
                    cost = self._compute_transaction_cost(notional)
                    tcost[t] += cost
                    net_pnl[t] -= cost

                    current_position = new_position
                    entry_idx = t
                    entry_signal = sig
                    entry_regime = reg

            position[t] = current_position

        # Net P&L column (entry cost already subtracted above)
        net_pnl_series = pd.Series(net_pnl, index=idx)
        # Add gross for days we had a position but no trade (no cost)
        for t in range(n):
            if position[t] != 0 and tcost[t] == 0:
                net_pnl[t] = gross_pnl[t]

        results_df = pd.DataFrame({
            "position":       position,
            "gross_pnl":      gross_pnl,
            "transaction_cost": tcost,
            "net_pnl":        net_pnl,
            "cumulative_pnl": np.cumsum(net_pnl),
            "regime":         regimes.values,
            "signal":         signals.values,
        }, index=idx)

        self._daily_pnl = pd.Series(net_pnl, index=idx, name="net_pnl")
        return results_df

    @property
    def trades(self) -> List[Trade]:
        return self._trades

    def get_performance_metrics(self) -> dict:
        """Compute full performance summary on net P&L."""
        if self._daily_pnl is None:
            raise RuntimeError("Run backtest first.")
        pnl_returns = self._daily_pnl / self.config.base_notional
        return performance_summary(pnl_returns)

    def get_regime_breakdown(self) -> pd.DataFrame:
        """
        Break down performance by regime label.

        Returns a DataFrame showing Sharpe, return, and number of trades
        for each regime.
        """
        if self._daily_pnl is None:
            raise RuntimeError("Run backtest first.")

        rows = []
        for regime_id in [0, 1, 2]:
            regime_names = {0: "Low-vol", 1: "Transition", 2: "Stress"}
            regime_pnl = self._daily_pnl[
                pd.Series(
                    [t.entry_regime == regime_id for t in self._trades],
                    index=range(len(self._trades))
                ).reindex(range(len(self._trades)), fill_value=False).values
            ] if self._trades else pd.Series(dtype=float)

            trade_pnl = [t.net_pnl for t in self._trades if t.entry_regime == regime_id]
            n_trades = len(trade_pnl)
            if n_trades == 0:
                continue

            win_rate = sum(1 for p in trade_pnl if p > 0) / n_trades
            avg_pnl = np.mean(trade_pnl)
            rows.append({
                "regime":    regime_names.get(regime_id, str(regime_id)),
                "n_trades":  n_trades,
                "win_rate":  round(win_rate, 3),
                "avg_pnl":   round(avg_pnl, 2),
                "total_pnl": round(sum(trade_pnl), 2),
            })

        return pd.DataFrame(rows).set_index("regime") if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Regime-Aware Train/Test Split
# ---------------------------------------------------------------------------

def regime_aware_split(
    data: pd.DataFrame,
    regimes: pd.Series,
    test_regime_ids: List[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/test sets based on regime labels.

    Test set: observations where the regime is in test_regime_ids
              AND was not observed during training.

    This is a strictly harder test than time-based splitting because:
    - The model is never shown test-regime data during training
    - The test set contains regimes that require genuine generalisation

    Parameters
    ----------
    data             : full dataset (signals, returns, etc.)
    regimes          : regime label series aligned to data.index
    test_regime_ids  : list of regime IDs to hold out for testing

    Returns
    -------
    (train_data, test_data) : DataFrames for training and testing
    """
    test_mask = regimes.isin(test_regime_ids)
    train_mask = ~test_mask

    return data[train_mask], data[test_mask]


# ---------------------------------------------------------------------------
# Synthetic Data Generator (for testing without live data)
# ---------------------------------------------------------------------------

def generate_synthetic_backtest_data(
    n_days: int = 1000,
    signal_mean_reversion: float = 0.85,
    signal_vol: float = 0.8,
    pnl_noise: float = 0.005,
    seed: int = 42,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Generate synthetic signals, regimes, and returns for unit testing.

    The signal is an AR(1) process (mean-reverting), which is the
    theoretical ideal for this type of strategy.

    Parameters
    ----------
    n_days                : number of trading days
    signal_mean_reversion : AR(1) coefficient (0.85 → strong mean-reversion)
    signal_vol            : signal volatility (controls z-score magnitudes)
    pnl_noise             : noise in the P&L relative to signal

    Returns
    -------
    (signals, regimes, returns_proxy)
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-02", periods=n_days, freq="B")

    # AR(1) signal: z_t = rho * z_{t-1} + epsilon_t
    signals = np.zeros(n_days)
    for t in range(1, n_days):
        signals[t] = signal_mean_reversion * signals[t - 1] + rng.normal(0, signal_vol)
    signal_series = pd.Series(signals, index=dates, name="signal")

    # Synthetic regimes: 3-state Markov chain
    regime_transitions = np.array([
        [0.97, 0.02, 0.01],   # Low-vol → ...
        [0.10, 0.85, 0.05],   # Transition → ...
        [0.05, 0.15, 0.80],   # Stress → ...
    ])
    regime_states = np.zeros(n_days, dtype=int)
    for t in range(1, n_days):
        probs = regime_transitions[regime_states[t - 1]]
        regime_states[t] = rng.choice(3, p=probs)
    regime_series = pd.Series(regime_states, index=dates, name="regime")

    # Returns proxy: higher when signal magnitude is high (mean-reversion)
    # Short vol pays off when the z-score contracts back to zero
    # Returns = -delta(signal) * scale + noise
    signal_changes = np.diff(signals, prepend=signals[0])
    returns = -0.001 * signal_changes + rng.normal(0, pnl_noise, n_days)
    # Attenuate returns in stress regime
    stress_mask = regime_states == 2
    returns[stress_mask] *= 0.4
    returns_series = pd.Series(returns, index=dates, name="returns")

    return signal_series, regime_series, returns_series
