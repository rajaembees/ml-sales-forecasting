import os
import sys
import pandas as pd

# Make src/ importable without a package __init__.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_generator import generate_sales_data
from eda import run_eda
from models import build_models
from evaluation import evaluate_models
from reporter import generate_reports


def main():
    data_path = os.path.join("data", "Sales.csv")

    if not os.path.exists(data_path):
        print("Sales.csv not found - generating synthetic data...")
        generate_sales_data(data_path)
    else:
        print(f"Loading existing data from {data_path}")

    sales = pd.read_csv(data_path)
    data = sales.copy()

    print(f"\nDataset: {data.shape[0]} rows x {data.shape[1]} columns")
    print("\nFirst 5 rows:")
    print(data.head().to_string())

    print("\nStatistical Summary:")
    print(data.describe().to_string())

    print("\nMissing values:", data.isnull().sum().sum())
    print("Duplicates    :", data.duplicated().sum())

    print("\n--- EDA ---")
    run_eda(data, output_dir=os.path.join("outputs", "plots"))

    print("\n--- Model Building ---")
    models, X_train, X_test, y_train, y_test = build_models(
        data, output_dir=os.path.join("outputs", "plots")
    )

    print("\n--- Model Evaluation ---")
    train_df, test_df = evaluate_models(models, X_train, X_test, y_train, y_test, output_dir="outputs")

    print("\n--- Generating Reports ---")
    generate_reports(
        data, models, X_train, train_df, test_df,
        plot_dir=os.path.join("outputs", "plots"),
        output_dir="outputs",
    )

    print("\nAll done! Check the outputs/ folder for plots, CSVs, and reports.")


if __name__ == "__main__":
    main()
