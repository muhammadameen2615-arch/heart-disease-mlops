import pandas as pd
from pathlib import Path


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Input dataset
INPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "heart_cleaned.csv"
)


# Output dataset
OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "heart_features.csv"
)


def create_features():

    print("\n========== FEATURE ENGINEERING STARTED ==========")

    # Load cleaned dataset
    df = pd.read_csv(INPUT_PATH)

    print("\nOriginal Shape:", df.shape)

    # Validate original target values
    valid_targets = [0, 1, 2, 3, 4]

    if not df["target"].isin(valid_targets).all():
        raise ValueError("Invalid target values found!")

    # Convert multiclass target into binary target
    df["target"] = (df["target"] > 0).astype(int)

    # Display new target distribution
    print("\nNew Target Distribution:")

    print(df["target"].value_counts())

    # Check missing values
    print("\nMissing Values:")

    print(df.isnull().sum())

    # Save feature-engineered dataset
    df.to_csv(OUTPUT_PATH, index=False)

    print("\nFinal Dataset Shape:", df.shape)

    print("\nFeature Engineered Dataset Saved!")

    print("Location:", OUTPUT_PATH)

    print("\nFEATURE ENGINEERING COMPLETED!")


if __name__ == "__main__":

    create_features()