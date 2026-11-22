"""
House Price Prediction - Model Training & Evaluation Module
-----------------------------------------------------------
Uses Scikit-Learn Linear Regression inside an end-to-end Pipeline
with ColumnTransformer preprocessing, generates EDA visualizations,
evaluates metrics (MAE, MSE, RMSE, R²), and persists the model.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for generating figures
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import (
    load_dataset,
    inspect_dataset,
    build_preprocessor,
    prepare_train_test_data,
    identify_target_column,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "model", "house_price_model.pkl")
DEFAULT_METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")
DEFAULT_VIS_DIR = os.path.join(BASE_DIR, "visualizations")


def create_visualizations(df, y_test, y_pred, output_dir=None):
    """
    Generates and saves the required EDA and evaluation graphs:
    1. Correlation Heatmap
    2. House Price Distribution
    3. Feature vs House Price Plots
    4. Actual vs Predicted House Prices
    5. Residual Plot
    """
    if output_dir is None:
        output_dir = DEFAULT_VIS_DIR

    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    target_col = identify_target_column(df)

    print(f"\nGenerating and saving visualizations to '{output_dir}' directory...")

    # 1. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap: Features vs House Price", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    corr_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(corr_path, dpi=300)
    plt.close()
    print(f"  [✓] Saved: {corr_path}")

    # 2. House Price Distribution
    plt.figure(figsize=(9, 6))
    sns.histplot(df[target_col], kde=True, color="#2b5c8f", bins=30)
    mean_price = df[target_col].mean()
    median_price = df[target_col].median()
    plt.axvline(mean_price, color="red", linestyle="--", linewidth=1.5, label=f"Mean: ${mean_price:,.2f}")
    plt.axvline(median_price, color="green", linestyle="-.", linewidth=1.5, label=f"Median: ${median_price:,.2f}")
    plt.title(f"Distribution of {target_col}", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel(f"{target_col} ($)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    plt.tight_layout()
    dist_path = os.path.join(output_dir, "house_price_distribution.png")
    plt.savefig(dist_path, dpi=300)
    plt.close()
    print(f"  [✓] Saved: {dist_path}")

    # 3. Feature vs House Price Plots
    feature_cols = [c for c in numeric_df.columns if c != target_col]
    n_features = len(feature_cols)
    cols_per_row = 3
    n_rows = int(np.ceil(n_features / cols_per_row))

    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]

    for idx, feature in enumerate(feature_cols):
        sns.regplot(
            data=df,
            x=feature,
            y=target_col,
            ax=axes[idx],
            scatter_kws={"alpha": 0.4, "color": "#1f77b4"},
            line_kws={"color": "red", "linewidth": 1.5}
        )
        axes[idx].set_title(f"{feature} vs {target_col}", fontsize=11, fontweight="bold")
        axes[idx].set_xlabel(feature, fontsize=10)
        axes[idx].set_ylabel(target_col, fontsize=10)

    # Hide unused subplots if any
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Feature vs House Price Relationships (with Linear Fit)", fontsize=16, fontweight="bold", y=1.00)
    plt.tight_layout()
    feat_path = os.path.join(output_dir, "feature_vs_house_price.png")
    plt.savefig(feat_path, dpi=300)
    plt.close()
    print(f"  [✓] Saved: {feat_path}")

    # 4. Actual vs Predicted House Prices
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.6, color="#2b5c8f", edgecolors="w", s=50, label="Predictions")
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="crimson", linestyle="--", linewidth=2, label="Ideal Fit (y = x)")
    plt.title("Actual vs Predicted House Prices", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Actual House Price ($)", fontsize=12)
    plt.ylabel("Predicted House Price ($)", fontsize=12)
    plt.legend()
    plt.tight_layout()
    act_pred_path = os.path.join(output_dir, "actual_vs_predicted.png")
    plt.savefig(act_pred_path, dpi=300)
    plt.close()
    print(f"  [✓] Saved: {act_pred_path}")

    # 5. Residual Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(9, 6))
    plt.scatter(y_pred, residuals, alpha=0.6, color="#ff7f0e", edgecolors="w", s=50)
    plt.axhline(0, color="black", linestyle="--", linewidth=1.5)
    plt.title("Residual Plot (Errors vs Predicted Values)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Predicted House Price ($)", fontsize=12)
    plt.ylabel("Residuals (Actual - Predicted) ($)", fontsize=12)
    plt.tight_layout()
    res_path = os.path.join(output_dir, "residual_plot.png")
    plt.savefig(res_path, dpi=300)
    plt.close()
    print(f"  [✓] Saved: {res_path}")


def display_model_parameters(pipeline, numerical_cols, categorical_cols):
    """
    Displays the trained Linear Regression coefficients and intercept,
    mapping them cleanly to their respective feature names.
    """
    regressor = pipeline.named_steps["regressor"]
    preprocessor = pipeline.named_steps["preprocessor"]

    feature_names = []
    if numerical_cols:
        feature_names.extend(numerical_cols)
    if categorical_cols:
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_features = cat_encoder.get_feature_names_out(categorical_cols).tolist()
        feature_names.extend(cat_features)

    intercept = regressor.intercept_
    coefficients = regressor.coef_

    print("\n" + "=" * 60)
    print("             LINEAR REGRESSION MODEL PARAMETERS")
    print("=" * 60)
    print(f"\nModel Intercept (beta_0): ${intercept:,.4f}")
    print("\nModel Coefficients (Weights):")
    print(f"{'Feature Name':<30} | {'Coefficient Value':<18} | {'Impact on Price ($)'}")
    print("-" * 65)

    coef_dict = {}
    for feat, coef in zip(feature_names, coefficients):
        coef_dict[feat] = float(coef)
        sign = "+" if coef >= 0 else "-"
        print(f"{feat:<30} | {coef:>18.4f} | {sign}${abs(coef):,.2f} per unit")
    print("=" * 60 + "\n")

    return intercept, coef_dict


def evaluate_model(y_test, y_pred, display=True):
    """
    Calculates and displays standard regression metrics:
    - Mean Absolute Error (MAE)
    - Mean Squared Error (MSE)
    - Root Mean Squared Error (RMSE)
    - R² Score (Coefficient of Determination)
    Provides beginner-friendly academic explanations for each.
    """
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2_Score": float(r2),
    }

    if display:
        print("\n" + "=" * 65)
        print("                   MODEL EVALUATION METRICS")
        print("=" * 65)

        print(f"\n1. Mean Absolute Error (MAE): ${mae:,.2f}")
        print("   Explanation: Average magnitude of the absolute errors.")
        print("   On average, the model's price prediction is off by this amount.")

        print(f"\n2. Mean Squared Error (MSE): {mse:,.2f}")
        print("   Explanation: Average of the squared prediction errors.")
        print("   Penalizes larger outliers more heavily due to quadratic squaring.")

        print(f"\n3. Root Mean Squared Error (RMSE): ${rmse:,.2f}")
        print("   Explanation: Square root of MSE, bringing error back to price units ($).")
        print("   Indicates typical deviation between actual and predicted prices.")

        print(f"\n4. R² Score (Coefficient of Determination): {r2:.4f} ({r2 * 100:.2f}%)")
        print("   Explanation: Proportion of house price variance explained by the features.")
        print(f"   The model explains {r2 * 100:.2f}% of the variation in house prices.")
        print("=" * 65 + "\n")

    return metrics


def train_model(dataset_path=None, model_save_path=None, metrics_save_path=None, output_vis_dir=None):
    """
    End-to-end model training routine:
    1. Loads and inspects dataset.
    2. Prepares train/test split (80/20, random_state=42).
    3. Builds Pipeline with ColumnTransformer + LinearRegression.
    4. Fits pipeline strictly on training data (no data leakage).
    5. Evaluates model on test data.
    6. Generates and saves all visualizations.
    7. Displays coefficients and intercept.
    8. Persists complete pipeline using joblib.
    """
    if model_save_path is None:
        model_save_path = DEFAULT_MODEL_PATH
    if metrics_save_path is None:
        metrics_save_path = DEFAULT_METRICS_PATH
    if output_vis_dir is None:
        output_vis_dir = DEFAULT_VIS_DIR

    df = load_dataset(dataset_path)
    inspect_dataset(df)

    # 80/20 train/test split with random_state=42
    X_train, X_test, y_train, y_test, num_cols, cat_cols, target_col = prepare_train_test_data(
        df, test_size=0.2, random_state=42
    )

    print(f"Dataset split completed:")
    print(f"  - Training samples: {len(X_train)} (80%)")
    print(f"  - Testing samples:  {len(X_test)} (20%)")

    # Build Preprocessor & Full Pipeline
    preprocessor = build_preprocessor(num_cols, cat_cols)
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    print("\nTraining Linear Regression Pipeline (fit strictly on training data)...")
    pipeline.fit(X_train, y_train)
    print("Training successfully completed.")

    # Generate test predictions
    y_pred = pipeline.predict(X_test)

    # Model evaluation
    metrics = evaluate_model(y_test, y_pred, display=True)

    # Display coefficients and intercept
    intercept, coef_dict = display_model_parameters(pipeline, num_cols, cat_cols)

    # Visualizations
    create_visualizations(df, y_test, y_pred, output_dir=output_vis_dir)

    # Persist the full pipeline
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(pipeline, model_save_path)
    print(f"\n[✓] Saved complete ML pipeline to: {os.path.abspath(model_save_path)}")

    # Also persist metrics and feature metadata for evaluation display
    metadata = {
        "metrics": metrics,
        "intercept": float(intercept),
        "coefficients": coef_dict,
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "target_col": target_col,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }
    with open(metrics_save_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"[✓] Saved model evaluation metadata to: {os.path.abspath(metrics_save_path)}")

    return pipeline, metrics


if __name__ == "__main__":
    train_model()
