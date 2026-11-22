"""
House Price Prediction - Residual Analysis Module
---------------------------------------------------
Computes residual statistics and diagnostics.
"""

import numpy as np
from scipy import stats


def compute_residual_stats(y_true, y_pred):
    """Compute residual statistics for diagnostics."""
    residuals = np.array(y_true) - np.array(y_pred)
    return {
        "residuals": residuals,
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals)),
        "median": float(np.median(residuals)),
        "min": float(np.min(residuals)),
        "max": float(np.max(residuals)),
        "skewness": float(stats.skew(residuals)),
        "kurtosis": float(stats.kurtosis(residuals)),
        "shapiro_stat": float(stats.shapiro(residuals[:min(5000, len(residuals))])[0]),
        "shapiro_p": float(stats.shapiro(residuals[:min(5000, len(residuals))])[1]),
    }
