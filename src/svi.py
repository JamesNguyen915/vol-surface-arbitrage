"""
svi.py
------
Stochastic Volatility Inspired (SVI) model for implied volatility surface
parameterization.

Reference:
    Gatheral, J. (2004). A parsimonious arbitrage-free implied volatility
    parameterization with application to the valuation of volatility
    derivatives. Presentation at Global Derivatives & Risk Management,
    Madrid. Available at: https://mfe.baruch.cuny.edu/wp-content/uploads/2019/12/Madrid2004.pdf

    Gatheral, J. & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces.
    Quantitative Finance, 14(1), 59–71.

The SVI parameterization expresses total implied variance w(k) as a function
of log-moneyness k = log(K/F):

    w(k; a, b, rho, m, sigma) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))

where:
    a     : vertical shift (overall variance level)
    b     : angle between the left and right asymptotes (wings)
    rho   : rotation (skew)
    m     : horizontal translation (ATM offset)
    sigma : curvature (smile width)

Arbitrage-free conditions (Gatheral & Jacquier 2014):
    1. Calendar spread: w(k; T1) <= w(k; T2) for all k when T1 < T2
    2. Butterfly: g(k) >= 0 for all k, where g(k) is a function of w and its
       derivatives that guarantees non-negative probability density.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize, Bounds


# ---------------------------------------------------------------------------
# SVI Parameter Container
# ---------------------------------------------------------------------------

@dataclass
class SVIParams:
    """
    Container for a single-slice SVI parameter set.

    Attributes
    ----------
    a     : level parameter (can be negative)
    b     : angle parameter (b >= 0)
    rho   : correlation/skew parameter (-1 < rho < 1)
    m     : ATM offset
    sigma : curvature parameter (sigma > 0)
    """
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def to_array(self) -> np.ndarray:
        return np.array([self.a, self.b, self.rho, self.m, self.sigma])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "SVIParams":
        return cls(a=arr[0], b=arr[1], rho=arr[2], m=arr[3], sigma=arr[4])

    def is_valid(self) -> bool:
        """Check parameter bounds."""
        return (
            self.b >= 0
            and -1.0 < self.rho < 1.0
            and self.sigma > 0
        )


# ---------------------------------------------------------------------------
# Core SVI Formula
# ---------------------------------------------------------------------------

def svi_total_variance(k: np.ndarray, params: SVIParams) -> np.ndarray:
    """
    Compute SVI total variance w(k) = sigma_BS^2 * T for a vector of
    log-moneyness values k = log(K/F).

    Parameters
    ----------
    k      : array of log-moneyness values
    params : SVIParams object

    Returns
    -------
    w : array of total implied variances (non-negative by construction
        when parameters satisfy validity conditions)
    """
    a, b, rho, m, sigma = params.a, params.b, params.rho, params.m, params.sigma
    xi = k - m
    w = a + b * (rho * xi + np.sqrt(xi**2 + sigma**2))
    return w


def svi_implied_vol(k: np.ndarray, T: float, params: SVIParams) -> np.ndarray:
    """
    Convert SVI total variance to implied volatility.

    Parameters
    ----------
    k      : log-moneyness array
    T      : time to expiry in years
    params : SVIParams

    Returns
    -------
    sigma_iv : implied volatility array (annualised)
    """
    if T <= 0:
        raise ValueError(f"Time to expiry T must be positive. Got T={T}")
    w = svi_total_variance(k, params)
    # Clip to avoid negative variance from numerical noise
    w = np.maximum(w, 1e-10)
    return np.sqrt(w / T)


# ---------------------------------------------------------------------------
# Butterfly Arbitrage Check (Gatheral & Jacquier 2014)
# ---------------------------------------------------------------------------

def svi_butterfly_density(k: np.ndarray, params: SVIParams) -> np.ndarray:
    """
    Compute the risk-neutral density proxy g(k) that must be >= 0 for
    the absence of butterfly arbitrage.

    From Gatheral & Jacquier (2014) Eq. 2.1:

        g(k) = (1 - k*w'/(2w))^2 - (w'/2)^2 * (1/w + 1/4) + w''/2

    where w' and w'' are first and second derivatives of w w.r.t. k.

    Returns
    -------
    g : array — should be >= 0 everywhere for no butterfly arbitrage
    """
    a, b, rho, m, sigma = params.a, params.b, params.rho, params.m, params.sigma
    xi = k - m
    sqrt_term = np.sqrt(xi**2 + sigma**2)

    w = a + b * (rho * xi + sqrt_term)
    w_prime = b * (rho + xi / sqrt_term)
    w_double_prime = b * sigma**2 / (sqrt_term**3)

    # Guard against zero variance
    w = np.maximum(w, 1e-10)

    g = (
        (1.0 - 0.5 * k * w_prime / w) ** 2
        - 0.25 * w_prime**2 * (1.0 / w + 0.25)
        + 0.5 * w_double_prime
    )
    return g


def has_butterfly_arbitrage(k: np.ndarray, params: SVIParams,
                             tol: float = -1e-4) -> bool:
    """
    Return True if the SVI slice admits butterfly arbitrage.

    We use a small negative tolerance (tol = -1e-4) to allow for
    floating-point noise near zero.
    """
    g = svi_butterfly_density(k, params)
    return bool(np.any(g < tol))


def has_calendar_arbitrage(params_t1: SVIParams, params_t2: SVIParams,
                            k: np.ndarray) -> bool:
    """
    Return True if there is calendar spread arbitrage between two slices.

    Calendar arbitrage condition: w(k; T1) > w(k; T2) for any k
    (total variance must be non-decreasing in T).
    """
    w1 = svi_total_variance(k, params_t1)
    w2 = svi_total_variance(k, params_t2)
    return bool(np.any(w1 > w2 + 1e-6))


# ---------------------------------------------------------------------------
# SVI Calibration (Fitting to Market Data)
# ---------------------------------------------------------------------------

def _svi_objective(params_arr: np.ndarray,
                   k: np.ndarray,
                   market_iv: np.ndarray,
                   T: float,
                   butterfly_penalty: float = 1e4) -> float:
    """
    Objective function for SVI calibration.

    Minimises weighted sum of:
        1. Squared IV fitting error (RMSE in IV space)
        2. Butterfly arbitrage penalty (soft constraint)

    Parameters
    ----------
    params_arr        : 5-element array [a, b, rho, m, sigma]
    k                 : log-moneyness array
    market_iv         : observed implied volatility array
    T                 : time to expiry
    butterfly_penalty : penalty weight for arbitrage violations
    """
    params = SVIParams.from_array(params_arr)
    try:
        model_iv = svi_implied_vol(k, T, params)
    except Exception:
        return 1e10

    # IV fitting error
    fit_error = np.mean((model_iv - market_iv) ** 2)

    # Butterfly penalty — penalise negative density
    g = svi_butterfly_density(k, params)
    penalty = butterfly_penalty * np.sum(np.maximum(-g, 0.0) ** 2)

    return fit_error + penalty


def calibrate_svi(
    k: np.ndarray,
    market_iv: np.ndarray,
    T: float,
    initial_params: Optional[SVIParams] = None,
    butterfly_penalty: float = 1e4,
    n_restarts: int = 5,
) -> Tuple[SVIParams, float]:
    """
    Calibrate SVI parameters to a single option chain (one expiry slice).

    Uses L-BFGS-B with parameter bounds and multiple random restarts
    to avoid local minima.

    Parameters
    ----------
    k              : log-moneyness array, shape (N,)
    market_iv      : observed implied volatility, shape (N,)
    T              : time to expiry in years
    initial_params : starting parameters (optional; random restarts used if None)
    butterfly_penalty : soft penalty weight for butterfly arbitrage violations
    n_restarts     : number of random restarts for global search

    Returns
    -------
    best_params : SVIParams with best-fit parameters
    rmse        : root mean squared IV error at best fit (in vol points)
    """
    if len(k) < 5:
        raise ValueError(
            f"Need at least 5 strikes for SVI calibration. Got {len(k)}."
        )

    # Parameter bounds: a ∈ ℝ, b ≥ 0, -1 < rho < 1, m ∈ ℝ, sigma > 0
    bounds = Bounds(
        lb=[-np.inf, 0.0, -0.999, -np.inf, 1e-4],
        ub=[ np.inf, np.inf,  0.999,  np.inf, np.inf],
    )

    best_result = None
    best_cost = np.inf

    # Generate initial guesses
    candidates = []
    if initial_params is not None:
        candidates.append(initial_params.to_array())

    # Add data-driven initial guess
    atm_iv = float(np.interp(0.0, k, market_iv))
    atm_var = atm_iv**2 * T
    candidates.append(np.array([atm_var * 0.8, 0.1, -0.3, 0.0, 0.2]))

    # Random restarts
    rng = np.random.default_rng(seed=42)
    for _ in range(n_restarts):
        a0 = rng.uniform(0.001, 0.5)
        b0 = rng.uniform(0.01, 1.0)
        rho0 = rng.uniform(-0.9, 0.9)
        m0 = rng.uniform(-0.3, 0.3)
        sigma0 = rng.uniform(0.05, 0.5)
        candidates.append(np.array([a0, b0, rho0, m0, sigma0]))

    for x0 in candidates:
        try:
            result = minimize(
                _svi_objective,
                x0,
                args=(k, market_iv, T, butterfly_penalty),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
            )
            if result.fun < best_cost:
                best_cost = result.fun
                best_result = result
        except Exception:
            continue

    if best_result is None:
        raise RuntimeError("SVI calibration failed for all starting points.")

    best_params = SVIParams.from_array(best_result.x)
    model_iv = svi_implied_vol(k, T, best_params)
    rmse = float(np.sqrt(np.mean((model_iv - market_iv) ** 2)))

    return best_params, rmse


# ---------------------------------------------------------------------------
# IV Surface Residual (the Trading Signal)
# ---------------------------------------------------------------------------

def compute_surface_residual(
    k: np.ndarray,
    observed_iv: np.ndarray,
    params: SVIParams,
    T: float,
) -> np.ndarray:
    """
    Compute the residual between observed market IV and SVI model IV.

    residual[i] = observed_iv[i] - svi_implied_vol(k[i], T, params)

    A positive residual means the market IV is *above* the model-fair value
    (options are expensive relative to the surface). A negative residual
    means they are cheap.

    This residual is the primary trading signal: we sell overpriced options
    (positive residual) and buy underpriced ones (negative residual),
    delta-hedged to isolate the volatility mispricing.

    Parameters
    ----------
    k            : log-moneyness of the options
    observed_iv  : market-observed implied volatilities
    params       : SVI parameters calibrated to this slice
    T            : time to expiry

    Returns
    -------
    residual : array of IV residuals (observed minus model)
    """
    model_iv = svi_implied_vol(k, T, params)
    return observed_iv - model_iv


def compute_zscore_residual(
    residuals: np.ndarray,
    lookback: int = 20,
) -> np.ndarray:
    """
    Z-score the surface residuals over a rolling window.

    This standardises the signal so that entry thresholds are
    dimensionless and comparable across different volatility regimes.

    Parameters
    ----------
    residuals : 1D array of IV residuals (one per day)
    lookback  : rolling window length in days

    Returns
    -------
    zscores : array of z-scored residuals, same length as input
              (NaN for the first lookback-1 observations)
    """
    residuals = np.asarray(residuals, dtype=float)
    n = len(residuals)
    zscores = np.full(n, np.nan)

    for i in range(lookback - 1, n):
        window = residuals[max(0, i - lookback + 1): i + 1]
        mu = np.mean(window)
        sigma = np.std(window, ddof=1)
        if sigma > 1e-10:
            zscores[i] = (residuals[i] - mu) / sigma
        else:
            zscores[i] = 0.0

    return zscores
