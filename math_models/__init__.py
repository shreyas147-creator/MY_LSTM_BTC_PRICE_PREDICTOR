"""
math_models/__init__.py
Clean public API for all mathematical model modules.
"""

from .sde import simulate_gbm, simulate_jump_diffusion, simulate_heston, euler_maruyama, path_statistics
from .fokker_planck import FokkerPlanckSolver, evolve_price_pdf, pdf_confidence_bands
from .garch import GARCHModel, fit_best_garch, realised_volatility, sabr_vol_surface
from .bayesian import (
    bayesian_gbm_params,
    variational_gbm_params,
    BayesianOnlineEstimator,
)
from .cointegration import (
    engle_granger_test,
    johansen_test,
    VECModel,
    cointegration_pipeline,
    spread_zscore,
    half_life,
)
from .kalman import (
    KalmanFilter,
    local_level_model,
    local_linear_trend,
    AdaptiveKalmanRegression,
    extract_trend,
)
from .spectral import (
    fft_power_spectrum,
    dominant_cycles,
    bandpass_filter,
    morlet_cwt,
    hilbert_analysis,
    spectral_features,
)
from .hmm import (
    HMMRegimeDetector,
    regime_statistics,
    select_hmm_states,
    build_hmm_features,
)
from .copula import (
    GaussianCopula,
    ClaytonCopula,
    GumbelCopula,
    FrankCopula,
    empirical_tail_dependence,
    select_copula,
    to_uniform,
)
from .evt import (
    GPDTailModel,
    GEVModel,
    hill_estimator,
    mean_excess_function,
    evt_pipeline,
)
from .tda import (
    sliding_window_embedding,
    compute_persistence,
    persistence_features,
    rolling_tda_features,
    tda_change_detection,
)
from .information import (
    shannon_entropy,
    empirical_entropy,
    kl_divergence,
    js_divergence,
    mutual_information_discrete,
    normalised_mutual_information,
    transfer_entropy,
    transfer_entropy_matrix,
    information_feature_ranking,
    rolling_entropy_features,
    kl_returns_vs_gaussian,
)

__all__ = [
    # SDE
    "simulate_gbm", "simulate_jump_diffusion", "simulate_heston",
    "euler_maruyama", "path_statistics",
    # Fokker-Planck
    "FokkerPlanckSolver", "evolve_price_pdf", "pdf_confidence_bands",
    # GARCH
    "GARCHModel", "fit_best_garch", "realised_volatility", "sabr_vol_surface",
    # Bayesian
    "bayesian_gbm_params", "variational_gbm_params", "BayesianOnlineEstimator",
    # Cointegration
    "engle_granger_test", "johansen_test", "VECModel",
    "cointegration_pipeline", "spread_zscore", "half_life",
    # Kalman
    "KalmanFilter", "local_level_model", "local_linear_trend",
    "AdaptiveKalmanRegression", "extract_trend",
    # Spectral
    "fft_power_spectrum", "dominant_cycles", "bandpass_filter",
    "morlet_cwt", "hilbert_analysis", "spectral_features",
    # HMM
    "HMMRegimeDetector", "regime_statistics", "select_hmm_states", "build_hmm_features",
    # Copula
    "GaussianCopula", "ClaytonCopula", "GumbelCopula", "FrankCopula",
    "empirical_tail_dependence", "select_copula", "to_uniform",
    # EVT
    "GPDTailModel", "GEVModel", "hill_estimator", "mean_excess_function", "evt_pipeline",
    # TDA
    "sliding_window_embedding", "compute_persistence", "persistence_features",
    "rolling_tda_features", "tda_change_detection",
    # Information
    "shannon_entropy", "empirical_entropy", "kl_divergence", "js_divergence",
    "mutual_information_discrete", "normalised_mutual_information",
    "transfer_entropy", "transfer_entropy_matrix",
    "information_feature_ranking", "rolling_entropy_features", "kl_returns_vs_gaussian",
]