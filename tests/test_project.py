"""
House Price Prediction - Automated Tests
-----------------------------------------
Pytest test suite covering all major project modules.
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_dataset, get_dataset_info, TARGET_COL, FEATURE_COLS
from src.model_training import (
    split_data, train_baseline, train_single_feature,
    train_multiple_regression, compute_metrics, train_all_models,
    save_model, load_model,
)
from src.manual_calculation import manual_regression
from src.multicollinearity import compute_vif, interpret_vif, get_high_correlations
from src.residual_analysis import compute_residual_stats


class TestDataLoading:
    """Tests for dataset loading module."""

    def test_load_dataset_returns_dataframe(self):
        df = load_dataset()
        assert isinstance(df, pd.DataFrame)

    def test_dataset_not_empty(self):
        df = load_dataset()
        assert len(df) > 0

    def test_dataset_has_target_column(self):
        df = load_dataset()
        assert TARGET_COL in df.columns

    def test_dataset_has_all_feature_columns(self):
        df = load_dataset()
        for col in FEATURE_COLS:
            assert col in df.columns, f"Missing feature column: {col}"

    def test_dataset_shape(self):
        df = load_dataset()
        assert df.shape[0] == 1000
        assert df.shape[1] == 8

    def test_get_dataset_info(self):
        df = load_dataset()
        info = get_dataset_info(df)
        assert info["rows"] == 1000
        assert info["cols"] == 8
        assert info["total_missing"] == 0
        assert info["duplicates"] == 0

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_dataset("nonexistent_file.csv")


class TestPreprocessing:
    """Tests for data splitting and preprocessing."""

    def test_split_data_sizes(self):
        df = load_dataset()
        X_train, X_test, y_train, y_test = split_data(df)
        total = len(X_train) + len(X_test)
        assert total == len(df)
        assert len(X_train) == 800
        assert len(X_test) == 200

    def test_split_data_no_leakage(self):
        df = load_dataset()
        X_train, X_test, y_train, y_test = split_data(df)
        # No overlap in indices
        train_idx = set(X_train.index.tolist())
        test_idx = set(X_test.index.tolist())
        assert len(train_idx & test_idx) == 0

    def test_features_are_numerical(self):
        df = load_dataset()
        for col in FEATURE_COLS:
            assert pd.api.types.is_numeric_dtype(df[col])


class TestManualCalculation:
    """Tests for manual Normal Equation calculation."""

    def test_manual_regression_runs(self):
        result = manual_regression()
        assert "b0" in result
        assert "b1" in result
        assert "b2" in result

    def test_coefficients_match_sklearn(self):
        result = manual_regression()
        assert result["diff_b0"] < 1e-4, "b0 differs too much from sklearn"
        assert result["diff_b1"] < 1e-4, "b1 differs too much from sklearn"
        assert result["diff_b2"] < 1e-4, "b2 differs too much from sklearn"

    def test_design_matrix_shape(self):
        result = manual_regression(n_samples=6)
        assert result["X_design"].shape == (6, 3)

    def test_prediction_exists(self):
        result = manual_regression()
        assert result["prediction_obs1"] is not None
        assert isinstance(result["residual_obs1"], float)


class TestModelTraining:
    """Tests for model training functions."""

    def test_baseline_r2_near_zero(self):
        df = load_dataset()
        X_train, X_test, y_train, y_test = split_data(df)
        baseline = train_baseline(y_train, y_test)
        # Baseline R² should be very low (possibly negative)
        assert baseline["metrics"]["R2"] < 0.1

    def test_single_feature_better_than_baseline(self):
        df = load_dataset()
        X_train, X_test, y_train, y_test = split_data(df)
        baseline = train_baseline(y_train, y_test)
        single = train_single_feature(X_train, X_test, y_train, y_test)
        assert single["metrics"]["R2"] > baseline["metrics"]["R2"]

    def test_multiple_better_than_single(self):
        df = load_dataset()
        X_train, X_test, y_train, y_test = split_data(df)
        single = train_single_feature(X_train, X_test, y_train, y_test)
        multiple = train_multiple_regression(X_train, X_test, y_train, y_test)
        assert multiple["metrics"]["R2"] >= single["metrics"]["R2"]

    def test_multiple_has_all_coefficients(self):
        df = load_dataset()
        X_train, X_test, y_train, y_test = split_data(df)
        multiple = train_multiple_regression(X_train, X_test, y_train, y_test)
        for feat in FEATURE_COLS:
            assert feat in multiple["coefficients"]

    def test_compute_metrics(self):
        y_true = np.array([100, 200, 300, 400])
        y_pred = np.array([110, 190, 310, 390])
        metrics = compute_metrics(y_true, y_pred, n_features=2)
        assert "MAE" in metrics
        assert "MSE" in metrics
        assert "RMSE" in metrics
        assert "R2" in metrics
        assert "Adjusted_R2" in metrics
        assert metrics["MAE"] == 10.0
        assert metrics["R2"] > 0.9


class TestModelPrediction:
    """Tests for prediction and model persistence."""

    def test_save_and_load_model(self, tmp_path):
        df = load_dataset()
        X_train, X_test, y_train, y_test = split_data(df)
        result = train_multiple_regression(X_train, X_test, y_train, y_test)
        model_path = str(tmp_path / "test_model.pkl")
        meta_path = str(tmp_path / "test_metadata.json")
        save_model(result["pipeline"], {"test": True}, model_path, meta_path)
        loaded_pipeline, loaded_meta = load_model(model_path, meta_path)
        assert loaded_pipeline is not None
        assert loaded_meta["test"] is True

    def test_prediction_returns_float(self):
        df = load_dataset()
        X_train, X_test, y_train, y_test = split_data(df)
        result = train_multiple_regression(X_train, X_test, y_train, y_test)
        pred = result["pipeline"].predict(X_test.head(1))
        assert isinstance(pred[0], (float, np.floating))

    def test_prediction_positive(self):
        df = load_dataset()
        X_train, X_test, y_train, y_test = split_data(df)
        result = train_multiple_regression(X_train, X_test, y_train, y_test)
        preds = result["pipeline"].predict(X_test)
        # Most predictions should be positive for house prices
        assert np.mean(preds > 0) > 0.95


class TestMulticollinearity:
    """Tests for VIF and correlation analysis."""

    def test_compute_vif_returns_dataframe(self):
        df = load_dataset()
        vif_df = compute_vif(df)
        assert isinstance(vif_df, pd.DataFrame)
        assert len(vif_df) == len(FEATURE_COLS)

    def test_vif_values_positive(self):
        df = load_dataset()
        vif_df = compute_vif(df)
        assert all(vif_df["VIF"] > 0)

    def test_interpret_vif(self):
        assert "Low" in interpret_vif(2.0)
        assert "High" in interpret_vif(15.0)

    def test_high_correlations(self):
        df = load_dataset()
        result = get_high_correlations(df)
        assert isinstance(result, pd.DataFrame)


class TestEvaluationMetrics:
    """Tests for evaluation metric calculations."""

    def test_perfect_prediction(self):
        y = np.array([100, 200, 300])
        metrics = compute_metrics(y, y, n_features=1)
        assert metrics["MAE"] == 0.0
        assert metrics["R2"] == 1.0

    def test_adjusted_r2_less_than_r2(self):
        np.random.seed(42)
        y_true = np.random.randn(100) * 100 + 500
        y_pred = y_true + np.random.randn(100) * 10
        metrics = compute_metrics(y_true, y_pred, n_features=5)
        assert metrics["Adjusted_R2"] <= metrics["R2"]


class TestInvalidInputs:
    """Tests for error handling with invalid inputs."""

    def test_empty_dataframe(self):
        with pytest.raises(Exception):
            split_data(pd.DataFrame())

    def test_missing_target_column(self):
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        with pytest.raises(Exception):
            split_data(df)

    def test_residual_stats_with_arrays(self):
        result = compute_residual_stats(
            np.array([100, 200, 300]),
            np.array([110, 190, 310])
        )
        assert "mean" in result
        assert "skewness" in result

