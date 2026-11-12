import pandas as pd
from pathlib import Path


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Input dataset
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "heart.csv"


# Output dataset
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "heart_cleaned.csv"


# Dataset column names
COLUMNS = [
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


def clean_data():

    print("\n========== DATA CLEANING STARTED ==========")

    # Check raw dataset
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_DATA_PATH}"
        )

    # Load raw dataset
    df = pd.read_csv(
        RAW_DATA_PATH,
        header=None,
        names=COLUMNS,
        na_values=["?"]
    )

    print("\nOriginal Dataset Shape:", df.shape)

    # Convert all columns to numeric
    for column in COLUMNS:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Check missing values
    print("\nMissing Values Before Cleaning:")

    print(df.isnull().sum())

    # Remove duplicate records if present
    duplicate_count = df.duplicated().sum()

    print("\nDuplicate Records Found:", duplicate_count)

    if duplicate_count > 0:

        df = df.drop_duplicates()

        print("Duplicate Records Removed:", duplicate_count)

    else:

        print("No duplicate records found.")

    # Validate dataset
    if df.empty:
        raise ValueError("Dataset is empty after cleaning.")

    # Check target values
    valid_targets = [0, 1, 2, 3, 4]

    invalid_targets = df[
        ~df["target"].isin(valid_targets)
    ]

    if not invalid_targets.empty:

        raise ValueError(
            "Invalid target values found."
        )

    # Create processed directory
    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save cleaned dataset
    df.to_csv(
        PROCESSED_DATA_PATH,
        index=False
    )

    print("\n========== CLEANING SUMMARY ==========")

    print("Final Dataset Shape:", df.shape)

    print(
        "Total Missing Values:",
        df.isnull().sum().sum()
    )

    print(
        "Total Duplicate Records:",
        df.duplicated().sum()
    )

    print("\nCleaned Dataset Saved Successfully!")

    print("Saved Location:", PROCESSED_DATA_PATH)

    print("\nDATA CLEANING COMPLETED!")


if __name__ == "__main__":

    clean_data()