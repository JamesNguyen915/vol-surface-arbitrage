"""
test_kalman.py
--------------
Unit tests for the Kalman filter hedge ratio estimation module.
"""

import numpy as np
import pandas as pd
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.kalman import (
    KalmanHedgeFilter,
    KalmanState,
    kalman_batch,
    rolling_ols_beta,
    estimate_noise_params,
)


# ---------------------------------------------------------------------------
# Test 1: Basic filter mechanics
# ---------------------------------------------------------------------------

class TestKalmanFilterBasics:

    def test_initial_state(self):
        """Filter should initialise to specified beta_0 and P_0."""
        kf = KalmanHedgeFilter(beta_0=-1.0, P_0=1.0)
        assert abs(kf.beta - (-1.0)) < 1e-12
        assert abs(kf.uncertainty - 1.0) < 1e-12

    def test_single_update_returns_state(self):
        """A single update should return a KalmanState object."""
        kf = KalmanHedgeFilter()
        state = kf.update(x_t=0.5, y_t=0.01)
        assert isinstance(state, KalmanState)
        assert isinstance(state.beta_hat, float)
        assert isinstance(state.P, float)

    def test_uncertainty_decreases_with_observations(self):
        """
        The posterior uncertainty P should decrease as we observe more data,
        provided the observations carry information (x_t != 0).
        This is the fundamental property of the Kalman filter.
        """
        kf = KalmanHedgeFilter(Q=1e-5, R=1e-2, beta_0=0.0, P_0=5.0)
        rng = np.random.default_rng(42)
        initial_P = kf.uncertainty

        for _ in range(50):
            x = rng.normal(1.0, 0.1)  # non-zero signal
            y = -1.0 * x + rng.normal(0, 0.01)  # y ≈ -x
            kf.update(x_t=x, y_t=y)

        # After 50 observations, uncertainty should be much lower
        assert kf.uncertainty < initial_P * 0.5, (
            f"Uncertainty should decrease. Initial: {initial_P:.4f}, "
            f"Final: {kf.uncertainty:.4f}"
        )

    def test_variance_stays_positive(self):
        """
        Posterior variance must always remain positive (numerical stability).
        Even with adversarial inputs, P should never go negative.
        """
        kf = KalmanHedgeFilter(Q=1e-6, R=1e-6, P_0=0.001)
        rng = np.random.default_rng(0)
        for _ in range(200):
            x = rng.normal(0, 10)  # Large signals
            y = rng.normal(0, 10)
            kf.update(x_t=x, y_t=y)
            assert kf.uncertainty > 0, "Posterior variance must stay positive"

    def test_reset_restores_initial_state(self):
        """reset() should restore the filter to its initial state."""
        kf = KalmanHedgeFilter(beta_0=-0.5, P_0=2.0)
        for _ in range(20):
            kf.update(x_t=1.0, y_t=-0.5)

        kf.reset()
        assert abs(kf.beta - (-0.5)) < 1e-12
        assert abs(kf.uncertainty - 2.0) < 1e-12
        assert len(kf._history) == 0


# ---------------------------------------------------------------------------
# Test 2: Convergence to true value
# ---------------------------------------------------------------------------

class TestKalmanConvergence:

    def test_converges_to_true_beta(self):
        """
        Given data y_t = beta_true * x_t + noise, the Kalman filter
        should converge to beta_true.
        """
        beta_true = -0.85
        rng = np.random.default_rng(123)
        n = 500

        kf = KalmanHedgeFilter(Q=1e-5, R=0.01, beta_0=0.0, P_0=10.0)
        for _ in range(n):
            x = rng.normal(1.0, 0.5)
            y = beta_true * x + rng.normal(0, 0.1)
            kf.update(x_t=x, y_t=y)

        # Filter should converge to within 5% of the true value
        assert abs(kf.beta - beta_true) < abs(beta_true) * 0.05, (
            f"Expected beta ≈ {beta_true:.3f}, got {kf.beta:.3f}"
        )

    def test_tracks_slowly_varying_beta(self):
        """
        When the true beta drifts slowly, the filter should track it.
        This is the key advantage over static OLS in non-stationary markets.
        """
        rng = np.random.default_rng(42)
        n = 300
        # True beta starts at -1 and drifts to -1.5
        true_betas = np.linspace(-1.0, -1.5, n)

        kf = KalmanHedgeFilter(Q=1e-3, R=0.02, beta_0=-1.0, P_0=1.0)
        final_estimates = []

        for t in range(n):
            x = rng.normal(1.0, 0.3)
            y = true_betas[t] * x + rng.normal(0, 0.05)
            state = kf.update(x_t=x, y_t=y)
            if t >= n - 50:
                final_estimates.append(state.beta_hat)

        avg_final = np.mean(final_estimates)
        # Final 50 estimates should be within 15% of true final value
        assert abs(avg_final - (-1.5)) < 0.25, (
            f"Expected tracked beta ≈ -1.5, got {avg_final:.3f}"
        )


# ---------------------------------------------------------------------------
# Test 3: Batch Kalman filter
# ---------------------------------------------------------------------------

class TestKalmanBatch:

    def test_batch_matches_sequential(self):
        """
        kalman_batch() should produce identical results to running
        KalmanHedgeFilter.update() sequentially.
        """
        rng = np.random.default_rng(7)
        n = 100
        signals = rng.normal(1.0, 0.5, n)
        pnl = -0.9 * signals + rng.normal(0, 0.05, n)
        Q, R, b0, P0 = 1e-4, 1e-2, 0.0, 1.0

        # Sequential
        kf = KalmanHedgeFilter(Q=Q, R=R, beta_0=b0, P_0=P0)
        seq_betas = []
        for t in range(n):
            state = kf.update(signals[t], pnl[t])
            seq_betas.append(state.beta_hat)

        # Batch
        batch_betas, _ = kalman_batch(signals, pnl, Q=Q, R=R, beta_0=b0, P_0=P0)

        np.testing.assert_allclose(seq_betas, batch_betas, rtol=1e-8,
            err_msg="Sequential and batch Kalman results must be identical")

    def test_batch_output_shape(self):
        """Batch output arrays should have same length as input."""
        n = 200
        signals = np.random.randn(n)
        pnl = np.random.randn(n)
        betas, uncertainties = kalman_batch(signals, pnl)
        assert len(betas) == n
        assert len(uncertainties) == n

    def test_batch_uncertainties_positive(self):
        """All posterior variances from batch filter must be positive."""
        signals = np.random.randn(100)
        pnl = np.random.randn(100)
        _, uncertainties = kalman_batch(signals, pnl, Q=1e-4, R=1e-2)
        assert np.all(uncertainties > 0)


# ---------------------------------------------------------------------------
# Test 4: Kalman vs. Rolling OLS comparison
# ---------------------------------------------------------------------------

class TestKalmanVsOLS:

    def test_kalman_outperforms_ols_in_regime_transition(self):
        """
        Kalman filter should track regime transitions faster than rolling OLS
        when the true beta shifts abruptly at the midpoint.

        This demonstrates the core value of the Kalman approach for our strategy.
        """
        rng = np.random.default_rng(99)
        n = 400
        signals = rng.normal(1.0, 0.3, n)

        # True beta: -1.0 for first half, -1.8 for second half (regime shift)
        true_beta = np.where(np.arange(n) < n // 2, -1.0, -1.8)
        pnl = true_beta * signals + rng.normal(0, 0.05, n)

        # Kalman filter
        batch_betas, _ = kalman_batch(signals, pnl, Q=5e-4, R=0.02)

        # Rolling OLS (60-day window)
        signal_s = pd.Series(signals)
        pnl_s = pd.Series(pnl)
        ols_betas = rolling_ols_beta(signal_s, pnl_s, window=60).values

        # After the regime shift, Kalman should be closer to true value
        # than OLS (which is slow to forget the old regime)
        post_shift = slice(n // 2 + 30, n // 2 + 90)  # 30-90 days after shift
        kalman_error = np.nanmean(np.abs(batch_betas[post_shift] - (-1.8)))
        ols_error = np.nanmean(np.abs(ols_betas[post_shift] - (-1.8)))

        assert kalman_error < ols_error, (
            f"Kalman error {kalman_error:.4f} should be less than "
            f"OLS error {ols_error:.4f} after regime shift"
        )


# ---------------------------------------------------------------------------
# Test 5: History logging
# ---------------------------------------------------------------------------

class TestKalmanHistory:

    def test_history_length(self):
        """History should record one entry per update call."""
        kf = KalmanHedgeFilter()
        for _ in range(30):
            kf.update(x_t=1.0, y_t=-0.9)
        history = kf.get_history()
        assert len(history) == 30

    def test_history_dataframe_columns(self):
        """History DataFrame should have expected columns."""
        kf = KalmanHedgeFilter()
        for _ in range(5):
            kf.update(x_t=0.5, y_t=0.1)
        history = kf.get_history()
        expected_cols = {"beta_hat", "uncertainty", "innovation", "kalman_gain"}
        assert expected_cols.issubset(set(history.columns))

    def test_empty_history_before_updates(self):
        """History should be empty before any update calls."""
        kf = KalmanHedgeFilter()
        history = kf.get_history()
        assert len(history) == 0
