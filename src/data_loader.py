"""
House Price Prediction - Data Loader Module
--------------------------------------------
Centralised dataset loading, feature info, and constants.
"""

import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "house_price_regression_dataset.csv")

TARGET_COL = "House_Price"
FEATURE_COLS = [
    "Square_Footage", "Num_Bedrooms", "Num_Bathrooms",
    "Year_Built", "Lot_Size", "Garage_Size", "Neighborhood_Quality"
]

FEATURE_INFO = {
    "Square_Footage": {
        "description": "Total living area of the house",
        "unit": "sq ft",
        "why": "Larger houses generally command higher prices due to more living space.",
    },
    "Num_Bedrooms": {
        "description": "Number of bedrooms",
        "unit": "count",
        "why": "More bedrooms increase utility for families, raising demand and price.",
    },
    "Num_Bathrooms": {
        "description": "Number of bathrooms",
        "unit": "count",
        "why": "Additional bathrooms add convenience and are a sought-after feature.",
    },
    "Year_Built": {
        "description": "Year the house was constructed",
        "unit": "year",
        "why": "Newer houses often have modern amenities and require less maintenance.",
    },
    "Lot_Size": {
        "description": "Total lot area",
        "unit": "acres",
        "why": "Larger lots provide more outdoor space and potential for expansion.",
    },
    "Garage_Size": {
        "description": "Garage capacity",
        "unit": "cars",
        "why": "Garage space adds storage and vehicle accommodation value.",
    },
    "Neighborhood_Quality": {
        "description": "Subjective neighborhood quality rating",
        "unit": "1-10 scale",
        "why": "Better neighborhoods have better schools, safety, and amenities.",
    },
}


def load_dataset(filepath=None):
    """Load the house price dataset. Never modifies the original file."""
    if filepath is None:
        filepath = DATASET_PATH
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    df = pd.read_csv(filepath)
    return df


def get_dataset_info(df):
    """Return a dict of dataset inspection info."""
    missing = df.isnull().sum()
    duplicates = df.duplicated().sum()
    numerical = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return {
        "name": "house_price_regression_dataset.csv",
        "source": "Synthetic dataset for academic ML demonstration",
        "rows": len(df),
        "cols": len(df.columns),
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": missing.to_dict(),
        "total_missing": int(missing.sum()),
        "duplicates": int(duplicates),
        "numerical": numerical,
        "categorical": categorical,
    }
