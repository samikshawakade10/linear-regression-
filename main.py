"""
========================================
       HOUSE PRICE PREDICTION
       Linear Regression Model
========================================
Main Entry Point & Multi-Interface Hub
Supports:
- Terminal Menu
- Native Desktop GUI (Tkinter)
- Web Browser Dashboard
"""

import os
import sys
import json
import argparse

from preprocessing import load_dataset, inspect_dataset, locate_dataset
from train import train_model, evaluate_model, display_model_parameters
from predict import interactive_prediction, load_trained_model, predict_price

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "house_price_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")
VIS_DIR = os.path.join(BASE_DIR, "visualizations")


def display_menu():
    """
    Displays the interactive main menu.
    """
    print("\n" + "=" * 42)
    print("        HOUSE PRICE PREDICTION")
    print("        Linear Regression Model")
    print("=" * 42)
    print()
    print("  1. Train Model")
    print("  2. Evaluate Model")
    print("  3. Predict House Price (Terminal)")
    print("  4. Launch Desktop GUI (Tkinter Window)")
    print("  5. Launch Web Dashboard (Browser)")
    print("  6. Exit")
    print()


def handle_train():
    """
    Executes model training, visualization generation, and pipeline saving.
    """
    print("\n>>> Selected: 1. Train Model")
    train_model(
        dataset_path=None,
        model_save_path=MODEL_PATH,
        metrics_save_path=METRICS_PATH,
        output_vis_dir=VIS_DIR
    )
    print("\n[SUCCESS] Model training and evaluation completed successfully!")


def handle_evaluate():
    """
    Loads saved model evaluation metrics, coefficients, and visualization paths.
    """
    print("\n>>> Selected: 2. Evaluate Model")

    if not os.path.isfile(MODEL_PATH) or not os.path.isfile(METRICS_PATH):
        print(f"\n[!] Model or metrics not found at '{MODEL_PATH}'.")
        print("    Training model first to generate evaluation metrics...")
        handle_train()
        return

    try:
        with open(METRICS_PATH, "r") as f:
            metadata = json.load(f)

        metrics = metadata.get("metrics", {})
        intercept = metadata.get("intercept", 0.0)
        coefficients = metadata.get("coefficients", {})
        train_samples = metadata.get("train_samples", 800)
        test_samples = metadata.get("test_samples", 200)

        print("\n" + "=" * 65)
        print("               SAVED MODEL EVALUATION SUMMARY")
        print("=" * 65)
        print(f"Algorithm: Linear Regression (Ordinary Least Squares)")
        print(f"Data Split: 80% Train ({train_samples} samples) / 20% Test ({test_samples} samples)")
        print(f"Random State: 42 (Reproducible)")

        print("\n" + "-" * 65)
        print("1. Mean Absolute Error (MAE):")
        print(f"   Value: ${metrics.get('MAE', 0.0):,.2f}")
        print("   Meaning: The average magnitude of error in dollar units ($).")

        print("\n2. Mean Squared Error (MSE):")
        print(f"   Value: {metrics.get('MSE', 0.0):,.2f}")
        print("   Meaning: Squared average error, heavily penalizing large deviations.")

        print("\n3. Root Mean Squared Error (RMSE):")
        print(f"   Value: ${metrics.get('RMSE', 0.0):,.2f}")
        print("   Meaning: Standard deviation of residuals; error measured in dollars ($).")

        print("\n4. R² Score (Coefficient of Determination):")
        r2 = metrics.get('R2_Score', 0.0)
        print(f"   Value: {r2:.4f} ({r2 * 100:.2f}%)")
        print(f"   Meaning: {r2 * 100:.2f}% of price variability is explained by the features.")
        print("-" * 65)

        print(f"\nModel Intercept (beta_0): ${intercept:,.4f}")
        print("\nFeature Coefficients (Weights):")
        print(f"{'Feature Name':<30} | {'Coefficient':<15} | Impact")
        print("-" * 65)
        for feat, coef in coefficients.items():
            sign = "+" if coef >= 0 else "-"
            print(f"{feat:<30} | {coef:>15.4f} | {sign}${abs(coef):,.2f} per unit")

        print("\n" + "=" * 65)
        print("Generated Visualization Plots:")
        print(f"Location: {VIS_DIR}")
        for idx, fname in enumerate(os.listdir(VIS_DIR), 1):
            if fname.endswith(".png"):
                full_p = os.path.join(VIS_DIR, fname)
                print(f"  {idx}. {fname} ({os.path.getsize(full_p):,} bytes)")
        print("=" * 65 + "\n")

    except Exception as err:
        print(f"[ERROR] Failed to load model evaluation: {err}")


def handle_predict():
    """
    Launches interactive terminal prediction interface.
    """
    print("\n>>> Selected: 3. Predict House Price (Terminal)")
    if not os.path.isfile(MODEL_PATH):
        print("\n[!] Trained model not found. Training model first...")
        handle_train()
    interactive_prediction(model_path=MODEL_PATH)


def handle_gui():
    """
    Launches native Tkinter Desktop GUI.
    """
    print("\n>>> Selected: 4. Launch Desktop GUI...")
    from gui import launch_gui
    launch_gui()


def handle_web():
    """
    Launches local Web Dashboard.
    """
    print("\n>>> Selected: 5. Launch Web Dashboard...")
    from web_app import launch_web_server
    launch_web_server(port=5000, open_browser=True)


def run_cli_demo():
    """
    Automated demo prediction without waiting for user input.
    """
    import pandas as pd
    pipeline = load_trained_model(MODEL_PATH)
    sample_house = pd.DataFrame({
        "Square_Footage": [2800.0],
        "Num_Bedrooms": [3],
        "Num_Bathrooms": [2],
        "Year_Built": [2000],
        "Lot_Size": [2.5],
        "Garage_Size": [2],
        "Neighborhood_Quality": [7],
    })
    pred = predict_price(pipeline, sample_house)
    print("\n[DEMO PREDICTION VERIFICATION]")
    print("Input Features:", sample_house.to_dict(orient="records")[0])
    print(f"Predicted Price: ${pred:,.2f}")
    return pred


def main():
    """
    Main loop parsing commands or displaying the interactive menu.
    """
    parser = argparse.ArgumentParser(description="House Price Prediction - Linear Regression")
    parser.add_argument("--train", action="store_true", help="Train model and save pipeline")
    parser.add_argument("--evaluate", action="store_true", help="Display model evaluation metrics")
    parser.add_argument("--predict", action="store_true", help="Launch interactive terminal prediction")
    parser.add_argument("--gui", action="store_true", help="Launch native Desktop GUI")
    parser.add_argument("--web", action="store_true", help="Launch Web Browser Dashboard")
    parser.add_argument("--demo", action="store_true", help="Run automated test prediction")
    parser.add_argument("--inspect", action="store_true", help="Inspect dataset")
    args = parser.parse_args()

    if args.train:
        handle_train()
        return
    if args.evaluate:
        handle_evaluate()
        return
    if args.predict:
        handle_predict()
        return
    if args.gui:
        handle_gui()
        return
    if args.web:
        handle_web()
        return
    if args.demo:
        run_cli_demo()
        return
    if args.inspect:
        df = load_dataset()
        inspect_dataset(df)
        return

    # Interactive Loop
    while True:
        display_menu()
        choice = input("Enter your choice (1-6): ").strip()

        if choice == "1":
            handle_train()
        elif choice == "2":
            handle_evaluate()
        elif choice == "3":
            handle_predict()
        elif choice == "4":
            handle_gui()
        elif choice == "5":
            handle_web()
        elif choice == "6":
            print("\nThank you for using House Price Prediction System. Goodbye!\n")
            sys.exit(0)
        else:
            print("\n[!] Invalid selection. Please enter a number between 1 and 6.")


if __name__ == "__main__":
    main()
