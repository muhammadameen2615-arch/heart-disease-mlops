import pandas as pd
from pathlib import Path


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset path
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "heart.csv"


# Dataset column names
columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target"
]


def inspect_data():

    # Check dataset existence
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    # Load dataset
    df = pd.read_csv(
        DATA_PATH,
        header=None,
        names=columns,
        na_values=["?"]
    )

    # Dataset shape
    print("\n========== DATASET SHAPE ==========")
    print(df.shape)

    # First five records
    print("\n========== FIRST 5 ROWS ==========")
    print(df.head())

    # Last five records
    print("\n========== LAST 5 ROWS ==========")
    print(df.tail())

    # Dataset information
    print("\n========== DATA INFORMATION ==========")
    df.info()

    # Statistical summary
    print("\n========== STATISTICAL SUMMARY ==========")
    print(df.describe())

    # Data types
    print("\n========== DATA TYPES ==========")
    print(df.dtypes)

    # Missing values
    print("\n========== MISSING VALUES ==========")
    print(df.isnull().sum())

    print("\nData inspection completed successfully!")


if __name__ == "__main__":
    inspect_data()