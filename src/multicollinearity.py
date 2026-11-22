"""
House Price Prediction - Multicollinearity Module
---------------------------------------------------
Computes VIF and high correlations for feature analysis.
"""

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.data_loader import FEATURE_COLS


def compute_vif(df, features=None):
    """Compute Variance Inflation Factor for each feature."""
    if features is None:
        features = FEATURE_COLS
    X = df[features].dropna()
    vif_data = []
    for i, feat in enumerate(features):
        try:
            vif_val = variance_inflation_factor(X.values, i)
        except Exception:
            vif_val = float("nan")
        vif_data.append({"Feature": feat, "VIF": round(vif_val, 4)})
    return pd.DataFrame(vif_data)


def interpret_vif(vif_val):
    """Interpret a VIF value."""
    if vif_val < 1:
        return "Not correlated"
    elif vif_val < 5:
        return "Low multicollinearity"
    elif vif_val < 10:
        return "Moderate multicollinearity"
    else:
        return "High multicollinearity ⚠️"


def get_high_correlations(df, features=None, threshold=0.7):
    """Find feature pairs with absolute correlation above threshold."""
    if features is None:
        features = FEATURE_COLS
    corr = df[features].corr()
    pairs = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            val = corr.iloc[i, j]
            if abs(val) >= threshold:
                pairs.append({
                    "Feature 1": features[i],
                    "Feature 2": features[j],
                    "Correlation": round(val, 4),
                })
    return pd.DataFrame(pairs) if pairs else pd.DataFrame(columns=["Feature 1", "Feature 2", "Correlation"])
