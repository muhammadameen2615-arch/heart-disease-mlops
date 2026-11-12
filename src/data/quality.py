import pandas as pd
from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset path
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "heart.csv"


# Column names
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


def analyze_data_quality():

    # Load dataset
    df = pd.read_csv(
        DATA_PATH,
        header=None,
        names=columns,
        na_values=["?"]
    )

    # 1. Missing Values
    print("\n========== MISSING VALUES ==========")

    print(df.isnull().sum())

    print(
        "\nTotal Missing Values:",
        df.isnull().sum().sum()
    )

    # 2. Duplicate Records
    print("\n========== DUPLICATE RECORDS ==========")

    duplicates = df.duplicated().sum()

    print("Total Duplicates:", duplicates)

    # 3. Target Distribution
    print("\n========== TARGET DISTRIBUTION ==========")

    print(df["target"].value_counts().sort_index())

    # 4. Outlier Detection using IQR
    print("\n========== OUTLIER ANALYSIS ==========")

    numerical_columns = [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak"
    ]

    for col in numerical_columns:

        Q1 = df[col].quantile(0.25)

        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR

        upper_bound = Q3 + 1.5 * IQR

        outliers = df[
            (df[col] < lower_bound) |
            (df[col] > upper_bound)
        ]

        print(f"\nColumn: {col}")

        print("Lower Bound:", lower_bound)

        print("Upper Bound:", upper_bound)

        print("Outliers:", len(outliers))

    # 5. Invalid Value Checks
    print("\n========== INVALID VALUES ==========")

    valid_values = {
        "sex": [0, 1],
        "cp": [1, 2, 3, 4],
        "fbs": [0, 1],
        "restecg": [0, 1, 2],
        "exang": [0, 1],
        "slope": [1, 2, 3],
        "ca": [0, 1, 2, 3],
        "thal": [3, 6, 7],
        "target": [0, 1, 2, 3, 4]
    }

    for col, valid in valid_values.items():

        invalid = df[
            df[col].notna() &
            ~df[col].isin(valid)
        ]

        print(
            f"{col}: {len(invalid)} invalid values"
        )

    print("\nData Quality Analysis Completed!")


if __name__ == "__main__":

    analyze_data_quality()