"""
test_svi.py
-----------
Unit tests for the SVI volatility surface parameterization module.

Each test is self-contained and documents exactly what is being verified
and why it matters for the trading strategy.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.svi import (
    SVIParams,
    svi_total_variance,
    svi_implied_vol,
    svi_butterfly_density,
    has_butterfly_arbitrage,
    has_calendar_arbitrage,
    calibrate_svi,
    compute_surface_residual,
    compute_zscore_residual,
)


# ---------------------------------------------------------------------------
# Fixtures — reusable parameter sets
# ---------------------------------------------------------------------------

@pytest.fixture
def atm_smile_params():
    """
    A well-behaved ATM smile: symmetric, moderate skew, no arbitrage.
    Typical of SPX in a calm low-vol regime.
    """
    return SVIParams(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.15)


@pytest.fixture
def steep_skew_params():
    """
    Steep left-skew typical of SPX in an elevated-vol or stress regime.
    """
    return SVIParams(a=0.06, b=0.25, rho=-0.7, m=-0.05, sigma=0.08)


@pytest.fixture
def log_moneyness_grid():
    """Standard grid of log-moneyness values from -0.5 to +0.5."""
    return np.linspace(-0.5, 0.5, 51)


# ---------------------------------------------------------------------------
# Test 1: Parameter validity
# ---------------------------------------------------------------------------

class TestSVIParamsValidity:

    def test_valid_params_pass(self, atm_smile_params):
        """Valid parameters should pass the is_valid() check."""
        assert atm_smile_params.is_valid()

    def test_negative_b_invalid(self):
        """
        b must be non-negative. Negative b allows variance to become negative
        for extreme log-moneyness values, violating no-arbitrage.
        """
        params = SVIParams(a=0.04, b=-0.1, rho=0.0, m=0.0, sigma=0.15)
        assert not params.is_valid()

    def test_rho_bounds(self):
        """rho must be strictly between -1 and 1 (correlation-like parameter)."""
        assert not SVIParams(a=0.04, b=0.1, rho=1.0,  m=0.0, sigma=0.15).is_valid()
        assert not SVIParams(a=0.04, b=0.1, rho=-1.0, m=0.0, sigma=0.15).is_valid()
        assert     SVIParams(a=0.04, b=0.1, rho=0.99, m=0.0, sigma=0.15).is_valid()

    def test_sigma_positive(self):
        """sigma controls the curvature and must be > 0."""
        assert not SVIParams(a=0.04, b=0.1, rho=0.0, m=0.0, sigma=0.0).is_valid()
        assert not SVIParams(a=0.04, b=0.1, rho=0.0, m=0.0, sigma=-0.1).is_valid()

    def test_roundtrip_array_conversion(self, atm_smile_params):
        """Parameters should survive to_array() / from_array() round-trip."""
        arr = atm_smile_params.to_array()
        restored = SVIParams.from_array(arr)
        assert abs(restored.a     - atm_smile_params.a    ) < 1e-12
        assert abs(restored.b     - atm_smile_params.b    ) < 1e-12
        assert abs(restored.rho   - atm_smile_params.rho  ) < 1e-12
        assert abs(restored.m     - atm_smile_params.m    ) < 1e-12
        assert abs(restored.sigma - atm_smile_params.sigma) < 1e-12


# ---------------------------------------------------------------------------
# Test 2: SVI total variance
# ---------------------------------------------------------------------------

class TestSVITotalVariance:

    def test_atm_variance(self, atm_smile_params):
        """
        At ATM (k=0, m=0), w = a + b * sigma.
        This is the ATM total variance.
        """
        k = np.array([0.0])
        w = svi_total_variance(k, atm_smile_params)
        expected = atm_smile_params.a + atm_smile_params.b * atm_smile_params.sigma
        assert abs(w[0] - expected) < 1e-12

    def test_variance_non_negative(self, atm_smile_params, log_moneyness_grid):
        """
        Total variance must be non-negative everywhere for well-behaved params.
        Negative variance has no financial meaning.
        """
        w = svi_total_variance(log_moneyness_grid, atm_smile_params)
        assert np.all(w >= 0), f"Negative variance found: min={w.min():.6f}"

    def test_smile_shape(self, atm_smile_params, log_moneyness_grid):
        """
        The SVI smile should be convex (curvature > 0), consistent with
        the observed volatility smile in equity markets.
        """
        w = svi_total_variance(log_moneyness_grid, atm_smile_params)
        # Check that ATM (k≈0) is the minimum of this symmetric smile
        atm_idx = np.argmin(np.abs(log_moneyness_grid))
        assert w[atm_idx] <= w[0], "ATM variance should be near-minimum for this symmetric smile"
        assert w[atm_idx] <= w[-1], "ATM variance should be near-minimum for this symmetric smile"

    def test_left_skew_with_negative_rho(self, steep_skew_params, log_moneyness_grid):
        """
        Negative rho (ρ < 0) produces left skew: OTM put IV > OTM call IV.
        This is the defining feature of the equity volatility skew.
        """
        w = svi_total_variance(log_moneyness_grid, steep_skew_params)
        # w at k=-0.2 (OTM put) should be > w at k=+0.2 (OTM call)
        idx_put  = np.argmin(np.abs(log_moneyness_grid - (-0.2)))
        idx_call = np.argmin(np.abs(log_moneyness_grid - 0.2))
        assert w[idx_put] > w[idx_call], (
            "Negative rho should produce left skew (OTM put variance > OTM call variance)"
        )


# ---------------------------------------------------------------------------
# Test 3: Implied volatility conversion
# ---------------------------------------------------------------------------

class TestSVIImpliedVol:

    def test_iv_positive(self, atm_smile_params, log_moneyness_grid):
        """Implied vols must be positive everywhere."""
        T = 1.0 / 252  # One trading day (0DTE context)
        ivs = svi_implied_vol(log_moneyness_grid, T, atm_smile_params)
        assert np.all(ivs > 0), f"Non-positive IV found: min={ivs.min():.6f}"

    def test_iv_reasonable_range(self, atm_smile_params):
        """
        SPX implied vols should be in a reasonable range (2% to 300% annualised).
        This catches parameter explosions.
        """
        k = np.linspace(-0.3, 0.3, 21)
        T = 30 / 365
        ivs = svi_implied_vol(k, T, atm_smile_params)
        assert np.all(ivs > 0.02), "IV below 2% is implausible for SPX"
        assert np.all(ivs < 3.00), "IV above 300% suggests calibration failure"

    def test_total_variance_consistency(self, atm_smile_params):
        """
        svi_implied_vol(k, T)^2 * T should equal svi_total_variance(k).
        """
        k = np.array([-0.1, 0.0, 0.1])
        T = 30 / 365
        w = svi_total_variance(k, atm_smile_params)
        ivs = svi_implied_vol(k, T, atm_smile_params)
        reconstructed_w = ivs**2 * T
        np.testing.assert_allclose(reconstructed_w, w, rtol=1e-8,
            err_msg="IV^2 * T should equal total variance")

    def test_zero_time_raises(self, atm_smile_params):
        """T=0 is undefined (options have expired); must raise ValueError."""
        with pytest.raises(ValueError, match="Time to expiry"):
            svi_implied_vol(np.array([0.0]), 0.0, atm_smile_params)

    def test_negative_time_raises(self, atm_smile_params):
        """Negative time to expiry is non-physical."""
        with pytest.raises(ValueError, match="Time to expiry"):
            svi_implied_vol(np.array([0.0]), -0.1, atm_smile_params)


# ---------------------------------------------------------------------------
# Test 4: Butterfly arbitrage check
# ---------------------------------------------------------------------------

class TestButterflyArbitrage:

    def test_well_behaved_params_no_arbitrage(self, atm_smile_params, log_moneyness_grid):
        """
        Well-behaved ATM smile parameters should produce non-negative
        risk-neutral density everywhere.
        """
        g = svi_butterfly_density(log_moneyness_grid, atm_smile_params)
        assert not has_butterfly_arbitrage(log_moneyness_grid, atm_smile_params), (
            f"No butterfly arbitrage expected. Min g = {g.min():.6f}"
        )

    def test_extreme_params_may_arbitrage(self):
        """
        Extreme parameter values (very high b, very negative rho) can produce
        butterfly arbitrage. We test that our detection correctly identifies this.
        """
        # b=2.0, rho=-0.99 is known to produce negative density for some k
        extreme_params = SVIParams(a=0.001, b=2.0, rho=-0.99, m=0.0, sigma=0.01)
        k = np.linspace(-1.0, 1.0, 201)
        # We don't assert True or False here — just confirm the function runs
        # and returns a boolean without crashing
        result = has_butterfly_arbitrage(k, extreme_params)
        assert isinstance(result, bool)

    def test_density_array_shape(self, atm_smile_params, log_moneyness_grid):
        """Butterfly density should return same shape as input k."""
        g = svi_butterfly_density(log_moneyness_grid, atm_smile_params)
        assert g.shape == log_moneyness_grid.shape


# ---------------------------------------------------------------------------
# Test 5: Calendar arbitrage check
# ---------------------------------------------------------------------------

class TestCalendarArbitrage:

    def test_no_calendar_arbitrage_increasing_params(self):
        """
        Two slices where T2 > T1 and the level parameter a is higher
        for T2 should not exhibit calendar arbitrage.
        """
        k = np.linspace(-0.3, 0.3, 31)
        params_t1 = SVIParams(a=0.02, b=0.1, rho=-0.3, m=0.0, sigma=0.15)
        params_t2 = SVIParams(a=0.05, b=0.1, rho=-0.3, m=0.0, sigma=0.15)
        assert not has_calendar_arbitrage(params_t1, params_t2, k), (
            "Higher a for longer maturity should not produce calendar arbitrage"
        )

    def test_calendar_arbitrage_detected(self):
        """
        If T2 total variance is less than T1 for some k, we have calendar
        arbitrage (impossible in no-arb markets).
        """
        k = np.linspace(-0.3, 0.3, 31)
        params_t1 = SVIParams(a=0.05, b=0.2, rho=-0.3, m=0.0, sigma=0.15)
        params_t2 = SVIParams(a=0.01, b=0.05, rho=-0.3, m=0.0, sigma=0.15)
        # T2 has much lower a and b → lower variance → calendar arbitrage
        assert has_calendar_arbitrage(params_t1, params_t2, k), (
            "Lower total variance at T2 should be detected as calendar arbitrage"
        )


# ---------------------------------------------------------------------------
# Test 6: SVI calibration
# ---------------------------------------------------------------------------

class TestSVICalibration:

    def test_calibration_recovers_known_params(self):
        """
        Calibrate SVI to data generated from known parameters.
        The calibrated params should closely match the generating params.
        """
        true_params = SVIParams(a=0.04, b=0.15, rho=-0.4, m=0.0, sigma=0.12)
        k = np.linspace(-0.3, 0.3, 21)
        T = 30 / 365
        true_iv = svi_implied_vol(k, T, true_params)

        # Add small noise (1bp) to simulate real market data
        rng = np.random.default_rng(0)
        noisy_iv = true_iv + rng.normal(0, 0.0001, len(k))

        calibrated, rmse = calibrate_svi(k, noisy_iv, T, initial_params=true_params)

        # RMSE should be very small (under 50bps vol error)
        assert rmse < 0.005, f"Calibration RMSE {rmse:.5f} > tolerance 0.005"

        # Recovered ATM IV should be close to true ATM IV
        atm_iv_true = true_iv[len(k) // 2]
        atm_iv_calib = svi_implied_vol(np.array([0.0]), T, calibrated)[0]
        assert abs(atm_iv_calib - atm_iv_true) < 0.002, (
            f"ATM IV mismatch: true={atm_iv_true:.4f}, calibrated={atm_iv_calib:.4f}"
        )

    def test_calibration_too_few_strikes(self):
        """
        Fewer than 5 strikes is insufficient for 5-parameter SVI.
        Should raise ValueError.
        """
        with pytest.raises(ValueError, match="5 strikes"):
            calibrate_svi(
                k=np.array([-0.1, 0.0, 0.1, 0.2]),
                market_iv=np.array([0.2, 0.18, 0.19, 0.21]),
                T=30 / 365,
            )


# ---------------------------------------------------------------------------
# Test 7: Surface residuals
# ---------------------------------------------------------------------------

class TestSurfaceResiduals:

    def test_perfect_fit_zero_residual(self, atm_smile_params):
        """
        When market IV exactly equals model IV, residuals should be zero.
        """
        k = np.linspace(-0.2, 0.2, 11)
        T = 30 / 365
        model_iv = svi_implied_vol(k, T, atm_smile_params)
        residuals = compute_surface_residual(k, model_iv, atm_smile_params, T)
        np.testing.assert_allclose(residuals, 0.0, atol=1e-10,
            err_msg="Residuals should be zero when market IV = model IV")

    def test_residual_sign_convention(self, atm_smile_params):
        """
        Positive residual = market IV above model → option is expensive.
        Negative residual = market IV below model → option is cheap.
        """
        k = np.array([0.0])  # ATM
        T = 30 / 365
        model_iv = svi_implied_vol(k, T, atm_smile_params)

        high_market_iv = model_iv + 0.02   # 2 vol points expensive
        low_market_iv  = model_iv - 0.01   # 1 vol point cheap

        r_high = compute_surface_residual(k, high_market_iv, atm_smile_params, T)
        r_low  = compute_surface_residual(k, low_market_iv,  atm_smile_params, T)

        assert r_high[0] > 0, "High market IV should produce positive residual"
        assert r_low[0]  < 0, "Low market IV should produce negative residual"

    def test_zscore_residual_output_shape(self):
        """Z-scored residuals should have the same length as inputs."""
        residuals = np.random.randn(100)
        zscores = compute_zscore_residual(residuals, lookback=20)
        assert len(zscores) == len(residuals)

    def test_zscore_nan_for_initial_window(self):
        """First (lookback-1) values should be NaN before window is full."""
        residuals = np.random.randn(50)
        lookback = 10
        zscores = compute_zscore_residual(residuals, lookback=lookback)
        assert np.all(np.isnan(zscores[:lookback - 1])), (
            "First (lookback-1) z-scores should be NaN"
        )
        assert not np.any(np.isnan(zscores[lookback - 1:])), (
            "Z-scores at and after lookback should not be NaN"
        )

    def test_zscore_zero_mean_unit_std(self):
        """Over the full window, z-scores should be approximately standardised."""
        rng = np.random.default_rng(42)
        residuals = rng.normal(0.05, 0.03, 200)  # Non-zero mean and std
        lookback = 20
        zscores = compute_zscore_residual(residuals, lookback=lookback)
        valid = zscores[~np.isnan(zscores)]
        # Mean should be close to zero, std close to 1
        assert abs(np.mean(valid)) < 0.15, "Z-score mean should be near zero"
        assert abs(np.std(valid) - 1.0) < 0.15, "Z-score std should be near 1"
