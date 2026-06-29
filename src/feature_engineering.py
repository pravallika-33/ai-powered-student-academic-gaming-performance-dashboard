"""Feature engineering for the Gaming Academic Performance project."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "Gaming_Academic_Performance_cleaned.csv"
TARGET_DATA_PATH = PROJECT_ROOT / "data" / "Gaming_Academic_Performance_with_target.csv"


TARGET_COLUMN = "academic_status"
TARGET_RULES = {
    "Risk": "grades < 60",
    "Medium": "60 <= grades < 85",
    "Safe": "grades >= 85",
}


def add_academic_status(df: pd.DataFrame) -> pd.DataFrame:
    """Create academic status target classes from cleaned grades."""
    df = df.copy()

    conditions = [
        df["grades"] < 60,
        (df["grades"] >= 60) & (df["grades"] < 85),
        df["grades"] >= 85,
    ]
    choices = ["Risk", "Medium", "Safe"]

    df[TARGET_COLUMN] = np.select(conditions, choices, default="Unknown")
    return df


def create_target_dataset(
    clean_path: Path = CLEAN_DATA_PATH,
    target_path: Path = TARGET_DATA_PATH,
) -> pd.DataFrame:
    """Load the cleaned dataset, add target classes, and save the result."""
    df = pd.read_csv(clean_path)
    df = add_academic_status(df)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_path, index=False)
    return df


if __name__ == "__main__":
    dataset = create_target_dataset()
    print(f"Saved target dataset with shape {dataset.shape}.")
    print(dataset[TARGET_COLUMN].value_counts().to_string())
