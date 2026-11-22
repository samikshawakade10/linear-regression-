"""
House Price Prediction - Data Preprocessing Module
--------------------------------------------------
Handles dataset discovery, loading, inspection, data splitting,
and Scikit-Learn ColumnTransformer pipeline generation.
"""

import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


def locate_dataset(search_dir=None):
    """
    Locates the existing house price CSV dataset automatically.
    Prioritizes the dataset directory, then the project root directory.
    """
    if search_dir is None:
        search_dir = os.path.dirname(os.path.abspath(__file__))

    # Preferred locations
    candidate_paths = [
        os.path.join(search_dir, "dataset", "house_price_regression_dataset.csv"),
        os.path.join(search_dir, "dataset", "existing_dataset.csv"),
        os.path.join(search_dir, "house_price_regression_dataset.csv"),
    ]

    for path in candidate_paths:
        if os.path.isfile(path):
            return os.path.abspath(path)

    # Search for any CSV file inside dataset/
    dataset_dir_csvs = glob.glob(os.path.join(search_dir, "dataset", "*.csv"))
    if dataset_dir_csvs:
        return os.path.abspath(dataset_dir_csvs[0])

    # Search for any CSV file in project root
    root_csvs = glob.glob(os.path.join(search_dir, "*.csv"))
    if root_csvs:
        return os.path.abspath(root_csvs[0])

    raise FileNotFoundError(
        "Could not locate house price dataset CSV file in project or dataset folder."
    )


def load_dataset(filepath=None):
    """
    Loads dataset from CSV file into a pandas DataFrame.
    Does not modify or overwrite the original dataset file.
    """
    if filepath is None:
        filepath = locate_dataset()

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    print(f"Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    return df


def inspect_dataset(df):
    """
    Inspects the dataset and prints summary details:
    - Shape and column names
    - First 5 records
    - Data types
    - Statistical summary
    - Missing values and duplicates
    - Numerical and categorical column identification
    - Automatic target column detection
    """
    print("\n" + "=" * 60)
    print("                DATASET INSPECTION REPORT")
    print("=" * 60)

    # Shape
    rows, cols = df.shape
    print(f"\n1. DATASET DIMENSIONS:")
    print(f"   - Total Records (Rows): {rows}")
    print(f"   - Total Features (Columns): {cols}")

    # Column names
    print(f"\n2. COLUMN NAMES:")
    for idx, col in enumerate(df.columns, 1):
        print(f"   {idx}. {col} ({df[col].dtype})")

    # First 5 records
    print(f"\n3. FIRST 5 RECORDS:")
    print(df.head().to_string())

    # Data types and missing values
    print(f"\n4. DATA TYPES & MISSING VALUES:")
    missing = df.isnull().sum()
    dtype_info = pd.DataFrame({
        "Data Type": df.dtypes,
        "Missing Values": missing,
        "Missing (%)": (missing / rows * 100).round(2)
    })
    print(dtype_info.to_string())

    # Duplicate records
    duplicate_count = df.duplicated().sum()
    print(f"\n5. DUPLICATE RECORDS:")
    print(f"   - Number of duplicate rows: {duplicate_count}")

    # Statistical summary
    print(f"\n6. STATISTICAL SUMMARY (NUMERICAL FEATURES):")
    print(df.describe().to_string())

    # Identify numerical and categorical columns
    target_col = identify_target_column(df)
    features_df = df.drop(columns=[target_col]) if target_col in df.columns else df

    numerical_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = features_df.select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"\n7. FEATURE CLASSIFICATION:")
    print(f"   - Target Column: '{target_col}'")
    print(f"   - Numerical Features ({len(numerical_cols)}): {numerical_cols}")
    print(f"   - Categorical Features ({len(categorical_cols)}): {categorical_cols}")
    print("=" * 60 + "\n")

    return {
        "shape": (rows, cols),
        "target_col": target_col,
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
        "missing_count": int(missing.sum()),
        "duplicate_count": int(duplicate_count),
    }


def identify_target_column(df):
    """
    Automatically identifies the target column related to house price.
    """
    candidates = ["house_price", "price", "saleprice", "sale_price", "target"]
    for col in df.columns:
        clean_name = col.strip().lower()
        if clean_name in candidates or any(cand in clean_name for cand in ["price"]):
            return col

    # Fallback to the last column if no matching name found
    return df.columns[-1]


def build_preprocessor(numerical_cols, categorical_cols):
    """
    Constructs a Scikit-Learn ColumnTransformer.
    - Numerical: SimpleImputer (median)
    - Categorical: SimpleImputer (most_frequent) + OneHotEncoder (handle_unknown='ignore')
    """
    transformers = []

    if numerical_cols:
        num_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ])
        transformers.append(("num", num_pipeline, numerical_cols))

    if categorical_cols:
        cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", cat_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )
    return preprocessor


def prepare_train_test_data(df, target_col=None, test_size=0.2, random_state=42):
    """
    Splits the dataset into 80% training and 20% testing sets.
    Prevents data leakage by strictly separating features (X) and target (y).
    """
    if target_col is None:
        target_col = identify_target_column(df)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, numerical_cols, categorical_cols, target_col


if __name__ == "__main__":
    # Test script standalone
    df_sample = load_dataset()
    inspect_dataset(df_sample)
