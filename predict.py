"""
House Price Prediction - Prediction Module
------------------------------------------
Loads the persisted trained Pipeline (preprocessing + Linear Regression)
and offers interactive, validated terminal-based house price estimation.
"""

import os
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "model", "house_price_model.pkl")


def load_trained_model(model_path=None):
    """
    Loads the trained Scikit-Learn pipeline from disk.
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            "Please train the model first by selecting Option 1 in the main menu."
        )

    pipeline = joblib.load(model_path)
    return pipeline


def prompt_float(prompt_text, min_val=None, max_val=None):
    """
    Prompts the user for a floating point number with boundary validation.
    """
    while True:
        try:
            val_str = input(prompt_text).strip()
            val = float(val_str)
            if min_val is not None and val < min_val:
                print(f"  [!] Value must be at least {min_val}. Please try again.")
                continue
            if max_val is not None and val > max_val:
                print(f"  [!] Value must be at most {max_val}. Please try again.")
                continue
            return val
        except ValueError:
            print("  [!] Invalid input. Please enter a valid numerical value.")


def prompt_int(prompt_text, min_val=None, max_val=None):
    """
    Prompts the user for an integer number with boundary validation.
    """
    while True:
        try:
            val_str = input(prompt_text).strip()
            val = int(val_str)
            if min_val is not None and val < min_val:
                print(f"  [!] Value must be at least {min_val}. Please try again.")
                continue
            if max_val is not None and val > max_val:
                print(f"  [!] Value must be at most {max_val}. Please try again.")
                continue
            return val
        except ValueError:
            print("  [!] Invalid input. Please enter a valid whole integer number.")


def collect_house_features():
    """
    Interactively collects house features from user input with clear prompts and validation.
    Does NOT ask for the target House_Price.
    """
    print("\nPlease enter the details of the house for price estimation:\n")

    # 1. Square Footage
    sq_ft = prompt_float(
        "1. Square Footage (e.g., 500 to 10000 sq ft): ",
        min_val=100.0,
        max_val=50000.0
    )

    # 2. Number of Bedrooms
    bedrooms = prompt_int(
        "2. Number of Bedrooms (e.g., 1 to 10): ",
        min_val=1,
        max_val=20
    )

    # 3. Number of Bathrooms
    bathrooms = prompt_int(
        "3. Number of Bathrooms (e.g., 1 to 10): ",
        min_val=1,
        max_val=20
    )

    # 4. Year Built
    year_built = prompt_int(
        "4. Year Built (e.g., 1850 to 2026): ",
        min_val=1800,
        max_val=2030
    )

    # 5. Lot Size in acres
    lot_size = prompt_float(
        "5. Lot Size in Acres (e.g., 0.1 to 10.0): ",
        min_val=0.01,
        max_val=100.0
    )

    # 6. Garage Size
    garage_size = prompt_int(
        "6. Garage Size (Number of cars, e.g., 0 to 5): ",
        min_val=0,
        max_val=10
    )

    # 7. Neighborhood Quality rating
    neighborhood_quality = prompt_int(
        "7. Neighborhood Quality (Rating scale 1 to 10): ",
        min_val=1,
        max_val=10
    )

    input_data = {
        "Square_Footage": [sq_ft],
        "Num_Bedrooms": [bedrooms],
        "Num_Bathrooms": [bathrooms],
        "Year_Built": [year_built],
        "Lot_Size": [lot_size],
        "Garage_Size": [garage_size],
        "Neighborhood_Quality": [neighborhood_quality],
    }

    return pd.DataFrame(input_data)


def predict_price(pipeline, input_df):
    """
    Feeds input features through the saved Pipeline and returns the estimated price.
    """
    predicted_val = pipeline.predict(input_df)[0]
    return float(predicted_val)


def interactive_prediction(model_path=None):
    """
    Terminal prediction loop allowing repeated predictions without restarting.
    """
    try:
        pipeline = load_trained_model(model_path)
    except FileNotFoundError as err:
        print(f"\n[ERROR] {err}")
        return

    print("\n" + "=" * 50)
    print("        HOUSE PRICE PREDICTION TERMINAL")
    print("=" * 50)

    while True:
        input_df = collect_house_features()

        print("\nEstimating price using trained Linear Regression pipeline...")
        predicted_price = predict_price(pipeline, input_df)

        print("\n" + "-" * 50)
        print("                 PREDICTION RESULT")
        print("-" * 50)
        print(f"  Estimated House Price: ${predicted_price:,.2f}")
        print("-" * 50)

        # Quick breakdown of input features
        print("\nSubmitted House Specifications:")
        for col in input_df.columns:
            val = input_df[col].iloc[0]
            if isinstance(val, float):
                print(f"  • {col:<22}: {val:,.2f}")
            else:
                print(f"  • {col:<22}: {val}")

        print("\n" + "=" * 50)
        again = input("\nWould you like to predict another house price? (y/n): ").strip().lower()
        if again not in ("y", "yes"):
            print("Exiting prediction interface. Returning to main menu.\n")
            break


if __name__ == "__main__":
    interactive_prediction()
