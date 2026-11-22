"""
House Price Prediction - Model Training Module
-----------------------------------------------
Trains baseline, single-feature, and multiple linear regression models.
Computes metrics including Adjusted R-squared.
Saves model and metadata with Joblib.
"""

import os
import json
import datetime
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from src.data_loader import load_dataset, TARGET_COL, FEATURE_COLS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "house_price_model.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")

RANDOM_STATE = 42
TEST_SIZE = 0.2


def split_data(df, target_col=TARGET_COL, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Split data into train/test sets. Returns X_train, X_test, y_train, y_test."""
    X = df[FEATURE_COLS]
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def compute_metrics(y_true, y_pred, n_features=None):
    """Compute MAE, MSE, RMSE, R2, and Adjusted R2."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    if n_features is not None and n > n_features + 1:
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    else:
        adj_r2 = None
    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2),
        "Adjusted_R2": float(adj_r2) if adj_r2 is not None else None,
    }


def train_baseline(y_train, y_test):
    """Mean-based baseline: predict training mean for every test sample."""
    mean_price = float(y_train.mean())
    y_pred = np.full(len(y_test), mean_price)
    metrics = compute_metrics(y_test, y_pred, n_features=0)
    return {
        "method": "Mean Baseline",
        "mean_price": mean_price,
        "predictions": y_pred,
        "metrics": metrics,
    }


def train_single_feature(X_train, X_test, y_train, y_test):
    """Train single-feature LR using the feature most correlated with target."""
    correlations = X_train.corrwith(y_train).abs()
    best_feature = correlations.idxmax()
    model = LinearRegression()
    model.fit(X_train[[best_feature]], y_train)
    y_pred = model.predict(X_test[[best_feature]])
    metrics = compute_metrics(y_test, y_pred, n_features=1)
    return {
        "method": "Single-Feature Linear Regression",
        "feature": best_feature,
        "intercept": float(model.intercept_),
        "coefficient": float(model.coef_[0]),
        "predictions": y_pred,
        "metrics": metrics,
        "model": model,
    }


def train_multiple_regression(X_train, X_test, y_train, y_test):
    """Train Multiple Linear Regression using all features."""
    preprocessor = ColumnTransformer(
        transformers=[("num", SimpleImputer(strategy="median"), FEATURE_COLS)],
        remainder="drop",
    )
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, n_features=len(FEATURE_COLS))

    regressor = pipeline.named_steps["regressor"]

    # Standardized coefficients
    X_train_processed = pipeline.named_steps["preprocessor"].transform(X_train)
    std_X = np.std(X_train_processed, axis=0)
    std_y = np.std(y_train)
    std_coefs = []
    for i, c in enumerate(regressor.coef_):
        if std_y != 0 and std_X[i] != 0:
            std_coefs.append(float(c * std_X[i] / std_y))
        else:
            std_coefs.append(0.0)

    coefficients = {feat: float(c) for feat, c in zip(FEATURE_COLS, regressor.coef_)}

    return {
        "method": "Multiple Linear Regression",
        "intercept": float(regressor.intercept_),
        "coefficients": coefficients,
        "standardized_coefficients": {feat: sc for feat, sc in zip(FEATURE_COLS, std_coefs)},
        "predictions": y_pred,
        "metrics": metrics,
        "pipeline": pipeline,
        "model": regressor,
    }


def save_model(pipeline, metadata, model_path=MODEL_PATH, metadata_path=METADATA_PATH):
    """Save model pipeline and metadata."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4, default=str)


def load_model(model_path=MODEL_PATH, metadata_path=METADATA_PATH):
    """Load saved model and metadata."""
    if not os.path.exists(model_path):
        return None, None
    pipeline = joblib.load(model_path)
    metadata = None
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    return pipeline, metadata


def train_all_models(df=None):
    """Train all three models and save the multiple regression pipeline."""
    if df is None:
        df = load_dataset()
    X_train, X_test, y_train, y_test = split_data(df)

    baseline = train_baseline(y_train, y_test)
    single = train_single_feature(X_train, X_test, y_train, y_test)
    multiple = train_multiple_regression(X_train, X_test, y_train, y_test)

    # Compute training ranges for extrapolation warnings
    train_ranges = {}
    for col in FEATURE_COLS:
        train_ranges[col] = {
            "min": float(X_train[col].min()),
            "max": float(X_train[col].max()),
        }

    import sklearn
    metadata = {
        "model_name": "Multiple Linear Regression",
        "dataset_name": "house_price_regression_dataset.csv",
        "training_date": datetime.datetime.now().isoformat(),
        "input_features": FEATURE_COLS,
        "target_variable": TARGET_COL,
        "train_test_split": f"{int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)}",
        "random_state": RANDOM_STATE,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "evaluation_metrics": multiple["metrics"],
        "train_ranges": train_ranges,
        "library_versions": {
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
    }
    save_model(multiple["pipeline"], metadata)

    return {
        "baseline": baseline,
        "single": single,
        "multiple": multiple,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "metadata": metadata,
    }
