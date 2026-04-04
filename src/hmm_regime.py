"""
hmm_regime.py
-------------
Hidden Markov Model (HMM) for volatility regime detection.

We model three latent volatility regimes:
    State 0 — Low-vol / Risk-On  : VIX < ~15, flat term structure
    State 1 — Transition         : VIX 15–25, moderately inverted term structure
    State 2 — Stress / Risk-Off  : VIX > 25, strongly inverted term structure

The feature set is designed to capture the VIX term structure shape:
    1. VIX level (log-transformed for stationarity)
    2. VIX/VVIX ratio (vol-of-vol relative to spot vol)
    3. VXX/VIX ratio (front-month futures premium/discount to spot)
    4. 10-day realised vol of VIX changes (regime transition speed)

Reference:
    Hamilton, J.D. (1989). A new approach to the economic analysis of
    nonstationary time series and the business cycle. Econometrica, 57(2), 357–384.

    Ang, A. & Bekaert, G. (2002). Regime switches in interest rates.
    Journal of Business & Economic Statistics, 20(2), 163–182.

Implementation:
    Uses hmmlearn.GaussianHMM with full covariance matrices.
    Requires: pip install hmmlearn
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import warnings


# ---------------------------------------------------------------------------
# Regime Labels
# ---------------------------------------------------------------------------

REGIME_NAMES: Dict[int, str] = {
    0: "low_vol",
    1: "transition",
    2: "stress",
}

# Intuitive colour mapping for plotting
REGIME_COLORS: Dict[int, str] = {
    0: "#2ecc71",   # green — calm
    1: "#f39c12",   # amber — caution
    2: "#e74c3c",   # red   — stress
}


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def build_hmm_features(vix_data: pd.DataFrame,
                        realised_vol_window: int = 10) -> pd.DataFrame:
    """
    Build the 4-feature matrix used to train and predict the HMM.

    Parameters
    ----------
    vix_data            : DataFrame with columns VIX, VVIX, VXX
                          (as returned by utils.load_vix_data)
    realised_vol_window : window for computing realised VIX vol

    Returns
    -------
    features : DataFrame with columns:
                log_vix          — log(VIX) for stationarity
                vix_vvix_ratio   — VIX / VVIX (normalised vol-of-vol)
                vxx_vix_ratio    — VXX / VIX (term structure shape)
                realised_vix_vol — rolling std of VIX daily changes
    """
    df = vix_data.copy().dropna()

    features = pd.DataFrame(index=df.index)

    # Feature 1: log-VIX level
    features["log_vix"] = np.log(df["VIX"])

    # Feature 2: VIX-to-VVIX ratio
    # High ratio → vol elevated relative to vol-of-vol → potential mean-reversion
    features["vix_vvix_ratio"] = df["VIX"] / df["VVIX"]

    # Feature 3: VXX/VIX ratio (term structure)
    # VXX tracks short-dated VIX futures. Ratio > 1 → contango (calm)
    #                                      Ratio < 1 → backwardation (stress)
    features["vxx_vix_ratio"] = df["VXX"] / df["VIX"]

    # Feature 4: Realised volatility of VIX changes
    vix_changes = df["VIX"].diff().dropna()
    realised_vix_vol = vix_changes.rolling(realised_vol_window).std()
    features["realised_vix_vol"] = realised_vix_vol

    features = features.dropna()
    return features


def standardise_features(features: pd.DataFrame,
                          mean: Optional[pd.Series] = None,
                          std: Optional[pd.Series] = None
                          ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Z-score standardise features.

    If mean and std are provided (from training), use them (for out-of-sample
    application). Otherwise compute from the data.

    Returns
    -------
    (standardised_features, mean, std)
    """
    if mean is None:
        mean = features.mean()
    if std is None:
        std = features.std()

    std_clipped = std.copy()
    std_clipped[std_clipped < 1e-10] = 1.0  # avoid division by zero

    return (features - mean) / std_clipped, mean, std


# ---------------------------------------------------------------------------
# HMM Model Wrapper
# ---------------------------------------------------------------------------

@dataclass
class RegimeModel:
    """
    Trained HMM regime model with fitted parameters and training statistics.

    Attributes
    ----------
    n_states        : number of hidden states
    feature_mean    : mean used for feature standardisation
    feature_std     : std used for feature standardisation
    model           : fitted hmmlearn.GaussianHMM object
    training_regimes: regime labels on the training set
    transition_matrix: (n_states × n_states) state transition probabilities
    regime_stats    : dict of per-regime feature statistics
    """
    n_states: int
    feature_mean: pd.Series
    feature_std: pd.Series
    model: object  # hmmlearn.GaussianHMM
    training_regimes: pd.Series
    transition_matrix: np.ndarray
    regime_stats: Dict[int, Dict]


def train_regime_model(
    vix_data: pd.DataFrame,
    n_states: int = 3,
    n_iter: int = 200,
    n_init: int = 10,
    random_state: int = 42,
) -> RegimeModel:
    """
    Train a Gaussian HMM for volatility regime detection.

    Parameters
    ----------
    vix_data     : DataFrame with columns VIX, VVIX, VXX
    n_states     : number of hidden states (default 3)
    n_iter       : EM algorithm maximum iterations
    n_init       : number of random initialisations (best log-likelihood kept)
    random_state : seed for reproducibility

    Returns
    -------
    RegimeModel containing the trained model and metadata
    """
    try:
        from hmmlearn import hmm as hmmlearn_hmm
    except ImportError:
        raise ImportError("hmmlearn is required. Run: pip install hmmlearn")

    # Build and standardise features
    features = build_hmm_features(vix_data)
    features_std, feat_mean, feat_std = standardise_features(features)
    X = features_std.values

    # Train multiple initialisations, keep best by log-likelihood
    best_model = None
    best_score = -np.inf
    rng = np.random.default_rng(random_state)

    for i in range(n_init):
        seed = int(rng.integers(0, 10000))
        model = hmmlearn_hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=n_iter,
            tol=1e-6,
            random_state=seed,
            init_params="stmc",
            params="stmc",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model.fit(X)
                score = model.score(X)
                if score > best_score:
                    best_score = score
                    best_model = model
            except Exception:
                continue

    if best_model is None:
        raise RuntimeError("HMM training failed for all initialisations.")

    # Predict regime labels on training data
    raw_labels = best_model.predict(X)

    # Sort states by mean VIX level (State 0 = lowest VIX, State 2 = highest)
    log_vix_col = features_std.columns.get_loc("log_vix")
    state_means = [
        best_model.means_[s, log_vix_col] for s in range(n_states)
    ]
    sorted_states = np.argsort(state_means)
    remap = {old: new for new, old in enumerate(sorted_states)}
    labels = np.array([remap[l] for l in raw_labels])

    regime_series = pd.Series(labels, index=features.index, name="regime")

    # Compute per-regime statistics for interpretability
    regime_stats = {}
    for s in range(n_states):
        mask = labels == s
        regime_stats[s] = {
            "name":        REGIME_NAMES[s],
            "n_days":      int(mask.sum()),
            "pct_days":    round(mask.mean() * 100, 1),
            "mean_vix":    round(float(vix_data.loc[features.index, "VIX"][mask].mean()), 1),
            "mean_vvix":   round(float(vix_data.loc[features.index, "VVIX"][mask].mean()), 1),
        }

    # Remap transition matrix to match sorted labels
    T_raw = best_model.transmat_
    T_sorted = np.zeros_like(T_raw)
    for i in range(n_states):
        for j in range(n_states):
            T_sorted[remap[i], remap[j]] = T_raw[i, j]

    return RegimeModel(
        n_states=n_states,
        feature_mean=feat_mean,
        feature_std=feat_std,
        model=best_model,
        training_regimes=regime_series,
        transition_matrix=T_sorted,
        regime_stats=regime_stats,
    )


def predict_regimes(
    regime_model: RegimeModel,
    vix_data: pd.DataFrame,
) -> pd.Series:
    """
    Predict regime labels for new (out-of-sample) VIX data.

    Uses the feature standardisation parameters from training to avoid
    data leakage.

    Parameters
    ----------
    regime_model : trained RegimeModel
    vix_data     : new VIX data (same column structure as training data)

    Returns
    -------
    pd.Series of regime labels (0, 1, or 2), indexed by date
    """
    features = build_hmm_features(vix_data)
    features_std, _, _ = standardise_features(
        features,
        mean=regime_model.feature_mean,
        std=regime_model.feature_std,
    )
    X = features_std.values

    raw_labels = regime_model.model.predict(X)

    # Apply the same remap as during training (low VIX = State 0)
    log_vix_col = features.columns.get_loc("log_vix")
    state_means = [
        regime_model.model.means_[s, log_vix_col]
        for s in range(regime_model.n_states)
    ]
    sorted_states = np.argsort(state_means)
    remap = {old: new for new, old in enumerate(sorted_states)}
    labels = np.array([remap.get(l, l) for l in raw_labels])

    return pd.Series(labels, index=features.index, name="regime")


def get_regime_entry_thresholds(regime: int) -> Dict[str, float]:
    """
    Return regime-conditional strategy entry and exit thresholds.

    In low-vol regimes, the IV surface is stable and mean-reversion
    is fast — tighter thresholds are appropriate.
    In stress regimes, the surface can gap — wider thresholds required
    to avoid being whipsawed by non-stationarity.

    Parameters
    ----------
    regime : 0 (low-vol), 1 (transition), 2 (stress)

    Returns
    -------
    dict with keys: entry_z, exit_z, max_position_scale
    """
    thresholds = {
        0: {"entry_z": 1.5, "exit_z": 0.3, "max_position_scale": 1.0},
        1: {"entry_z": 2.0, "exit_z": 0.5, "max_position_scale": 0.6},
        2: {"entry_z": 2.5, "exit_z": 0.8, "max_position_scale": 0.3},
    }
    return thresholds.get(regime, thresholds[1])
