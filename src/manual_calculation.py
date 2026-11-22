"""
House Price Prediction - Manual Calculation Module
---------------------------------------------------
Implements the Normal Equation step-by-step for educational purposes.
Uses a small sample (5-8 observations) with 2 features.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.data_loader import load_dataset, TARGET_COL, FEATURE_COLS


def manual_regression(df=None, feature1=None, feature2=None, n_samples=6, random_state=42):
    """
    Perform manual linear regression using the Normal Equation on a small sample.

    Steps:
    1. Select a small sample of observations.
    2. Build the design matrix X (with ones column).
    3. Compute X^T X, X^T y.
    4. Solve (X^T X) b = X^T y for coefficients.
    5. Compare with sklearn.
    """
    if df is None:
        df = load_dataset()

    if feature1 is None:
        feature1 = FEATURE_COLS[0]  # Square_Footage
    if feature2 is None:
        feature2 = FEATURE_COLS[1]  # Num_Bedrooms

    # Select a small sample
    sample = df.sample(n=n_samples, random_state=random_state).reset_index(drop=True)

    X_raw = sample[[feature1, feature2]].values
    y = sample[TARGET_COL].values

    # Design matrix with intercept column
    ones = np.ones((len(X_raw), 1))
    X_design = np.hstack([ones, X_raw])

    # Step-by-step Normal Equation
    Xt = X_design.T                          # X transpose
    XtX = Xt @ X_design                      # X^T X
    Xty = Xt @ y                             # X^T y

    # Solve using numpy (handles near-singular)
    try:
        cond = np.linalg.cond(XtX)
        if cond > 1e12:
            # Use least-squares for ill-conditioned
            coeffs, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
            solve_method = "numpy.linalg.lstsq (ill-conditioned matrix)"
        else:
            coeffs = np.linalg.solve(XtX, Xty)
            solve_method = "numpy.linalg.solve (Normal Equation)"
    except np.linalg.LinAlgError:
        coeffs, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
        solve_method = "numpy.linalg.lstsq (singular matrix fallback)"

    b0, b1, b2 = coeffs[0], coeffs[1], coeffs[2]

    # Prediction for first observation
    pred_first = b0 + b1 * X_raw[0, 0] + b2 * X_raw[0, 1]
    residual_first = y[0] - pred_first

    # Sklearn comparison
    sk_model = LinearRegression()
    sk_model.fit(X_raw, y)
    sk_b0 = sk_model.intercept_
    sk_b1 = sk_model.coef_[0]
    sk_b2 = sk_model.coef_[1]

    return {
        "feature1": feature1,
        "feature2": feature2,
        "n_samples": n_samples,
        "sample_df": sample,
        "X_raw": X_raw,
        "y": y,
        "X_design": X_design,
        "Xt": Xt,
        "XtX": XtX,
        "Xty": Xty,
        "coefficients": coeffs,
        "b0": float(b0),
        "b1": float(b1),
        "b2": float(b2),
        "solve_method": solve_method,
        "condition_number": float(np.linalg.cond(XtX)),
        "prediction_obs1": float(pred_first),
        "actual_obs1": float(y[0]),
        "residual_obs1": float(residual_first),
        "sklearn_b0": float(sk_b0),
        "sklearn_b1": float(sk_b1),
        "sklearn_b2": float(sk_b2),
        "diff_b0": float(abs(b0 - sk_b0)),
        "diff_b1": float(abs(b1 - sk_b1)),
        "diff_b2": float(abs(b2 - sk_b2)),
    }
