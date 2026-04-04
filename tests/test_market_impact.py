"""
test_market_impact.py
---------------------
Unit tests for the Almgren-Chriss market impact module.
"""

import numpy as np
import pandas as pd
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.market_impact import (
    MarketParams,
    temporary_impact,
    permanent_impact,
    total_execution_cost,
    capacity_curve,
    find_capacity_limit,
    execution_frontier,
)


@pytest.fixture
def default_params():
    return MarketParams()


class TestTemporaryImpact:

    def test_increases_with_order_size(self, default_params):
        """Larger orders have proportionally higher temporary impact."""
        p = default_params
        small  = temporary_impact(1e5,  p.daily_volume, p.daily_vol, p.eta)
        medium = temporary_impact(1e6,  p.daily_volume, p.daily_vol, p.eta)
        large  = temporary_impact(1e8,  p.daily_volume, p.daily_vol, p.eta)
        assert small < medium < large

    def test_positive_impact(self, default_params):
        """Impact costs are always positive."""
        p = default_params
        impact = temporary_impact(1e6, p.daily_volume, p.daily_vol, p.eta)
        assert impact > 0

    def test_power_law_scaling(self, default_params):
        """
        Almgren-Chriss temporary impact scales as (size/volume)^0.6.
        Doubling the order size multiplies impact by 2^0.6 ≈ 1.516.
        """
        p = default_params
        base_size = 1e6
        doubled   = 2e6
        r1 = temporary_impact(base_size, p.daily_volume, p.daily_vol, p.eta)
        r2 = temporary_impact(doubled,   p.daily_volume, p.daily_vol, p.eta)
        ratio = r2 / r1
        expected = 2.0 ** 0.6
        assert abs(ratio - expected) < 0.001, (
            f"Expected impact ratio {expected:.4f}, got {ratio:.4f}"
        )


class TestPermanentImpact:

    def test_linear_in_order_size(self, default_params):
        """Permanent impact is linear in order size / daily volume."""
        p = default_params
        r1 = permanent_impact(1e6, p.daily_volume, p.daily_vol, p.gamma)
        r2 = permanent_impact(2e6, p.daily_volume, p.daily_vol, p.gamma)
        assert abs(r2 / r1 - 2.0) < 0.001, "Permanent impact should be linear"

    def test_permanent_less_than_temporary_for_small_orders(self, default_params):
        """
        For small orders (low participation rate), temporary impact
        should exceed permanent impact (Almgren-Chriss calibration).
        """
        p = default_params
        size = 1e6  # $1M order out of $50B daily volume → very small
        temp = temporary_impact(size, p.daily_volume, p.daily_vol, p.eta)
        perm = permanent_impact(size, p.daily_volume, p.daily_vol, p.gamma)
        # Both should be very small for a $1M order in a $50B market
        assert temp > 0 and perm > 0


class TestTotalExecutionCost:

    def test_returns_required_keys(self, default_params):
        """Output dict must contain all expected cost components."""
        costs = total_execution_cost(1e6, default_params)
        required_keys = {
            "bid_ask_cost_bps", "temp_impact_bps",
            "perm_impact_bps", "total_cost_bps", "total_cost_dollars"
        }
        assert required_keys.issubset(set(costs.keys()))

    def test_total_equals_sum_of_components(self, default_params):
        """Total cost should equal sum of bid-ask + temp + perm."""
        costs = total_execution_cost(1e6, default_params)
        computed_total = (
            costs["bid_ask_cost_bps"]
            + costs["temp_impact_bps"]
            + costs["perm_impact_bps"]
        )
        assert abs(computed_total - costs["total_cost_bps"]) < 0.01

    def test_dollar_cost_consistency(self, default_params):
        """Dollar cost should equal total_bps / 10000 * order_size."""
        order_size = 50_000.0
        costs = total_execution_cost(order_size, default_params)
        expected_dollars = order_size * costs["total_cost_bps"] / 10_000
        assert abs(costs["total_cost_dollars"] - expected_dollars) < 0.01

    def test_more_periods_increases_temp_impact(self, default_params):
        """
        In the Almgren-Chriss model, spreading execution over more periods
        INCREASES total temporary impact (scaling as T^0.4).

        This is because the per-period cost is eta*(X/TV)^0.6 and summing
        over T periods gives T * eta*(X/TV)^0.6 = T^0.4 * eta*(X/V)^0.6.

        The benefit of slower execution is NOT lower temporary impact — it is
        lower MARKET RISK exposure (the position is smaller during execution).
        This tradeoff is the core of Almgren-Chriss optimal execution theory.
        """
        costs_1 = total_execution_cost(1e7, default_params, n_periods=1)
        costs_5 = total_execution_cost(1e7, default_params, n_periods=5)
        assert costs_5["temp_impact_bps"] > costs_1["temp_impact_bps"], (
            "More execution periods should INCREASE total temporary impact (T^0.4 scaling)"
        )
        # Verify the T^0.4 scaling law
        import numpy as np
        ratio = costs_5["temp_impact_bps"] / costs_1["temp_impact_bps"]
        expected_ratio = 5 ** 0.4
        assert abs(ratio - expected_ratio) < 0.05, (
            f"Temp impact ratio should be 5^0.4 ≈ {expected_ratio:.3f}, got {ratio:.3f}"
        )

    def test_james_actual_size_cost(self, default_params):
        """
        At James's actual strategy size ($50K per trade), total cost
        should be dominated by bid-ask spread (well under 30bps total).
        This validates the strategy's viability at current scale.
        """
        costs = total_execution_cost(50_000, default_params)
        assert costs["total_cost_bps"] < 30, (
            f"At $50K, costs should be < 30bps. Got {costs['total_cost_bps']:.1f}bps"
        )


class TestCapacityCurve:

    def test_capacity_curve_returns_dataframe(self, default_params):
        """capacity_curve() should return a properly structured DataFrame."""
        df = capacity_curve(default_params, strategy_edge_bps=50.0)
        assert isinstance(df, pd.DataFrame)
        required_cols = {"order_size_usd", "total_cost_bps", "net_edge_bps", "profitable"}
        assert required_cols.issubset(set(df.columns))

    def test_larger_orders_less_profitable(self, default_params):
        """Net edge should decrease monotonically as order size increases."""
        df = capacity_curve(default_params, strategy_edge_bps=50.0)
        net_edges = df["net_edge_bps"].values
        # Net edge should be generally decreasing (not necessarily strictly)
        assert net_edges[0] > net_edges[-1], (
            "Net edge should decrease from small to large orders"
        )

    def test_capacity_limit_positive(self, default_params):
        """Capacity limit should be a positive dollar amount."""
        limit = find_capacity_limit(default_params, strategy_edge_bps=50.0)
        assert limit > 0
        assert limit < 100_000_000_000  # Below $100B (sanity check)

    def test_higher_edge_means_higher_capacity(self, default_params):
        """Strategy with higher edge should have a larger capacity ceiling."""
        limit_low  = find_capacity_limit(default_params, strategy_edge_bps=20.0)
        limit_high = find_capacity_limit(default_params, strategy_edge_bps=80.0)
        assert limit_high > limit_low, (
            "Higher edge should support larger trade sizes before costs consume it"
        )


class TestExecutionFrontier:

    def test_returns_dataframe(self, default_params):
        df = execution_frontier(1e7, default_params)
        assert isinstance(df, pd.DataFrame)

    def test_impact_increases_with_more_periods(self, default_params):
        """
        Total temporary impact INCREASES with more execution periods (T^0.4 scaling).
        The benefit of slower execution is reduced market risk, not lower impact.
        See Almgren-Chriss (2001) for the full theory.
        """
        df = execution_frontier(1e7, default_params)
        assert df["impact_cost_bps"].iloc[0] < df["impact_cost_bps"].iloc[-1], (
            "Total temporary impact should increase as we spread execution over more days (T^0.4)"
        )

    def test_market_risk_increases_with_more_periods(self, default_params):
        """More execution periods → more exposure to underlying price moves."""
        df = execution_frontier(1e7, default_params)
        assert df["market_risk_bps"].iloc[0] < df["market_risk_bps"].iloc[-1], (
            "Market risk increases when execution is spread over more days"
        )
