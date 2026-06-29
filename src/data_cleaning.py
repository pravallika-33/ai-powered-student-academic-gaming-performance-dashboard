"""Data cleaning workflow for the Gaming Academic Performance dataset.

This module reads the raw CSV, applies transparent cleaning steps, writes a
cleaned dataset, and creates a Markdown report that explains every action.
"""

from pathlib import Path

import numpy as np
import pandas as pd


RAW_DATA_PATH = Path(
    "/Users/mohanapravallikapakala/Downloads/Data Sets/Gaming_Academic_Performance.csv"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "Gaming_Academic_Performance_cleaned.csv"
REPORT_PATH = PROJECT_ROOT / "data" / "cleaning_report.md"


NUMERIC_COLUMNS = [
    "student_id",
    "age",
    "gaming_hours",
    "study_hours",
    "sleep_hours",
    "attendance",
    "social_activity",
    "device_usage",
    "reaction_time_ms",
    "addiction_score",
    "grades",
]

CATEGORICAL_COLUMNS = ["gender", "gaming_genre", "stress_level"]


def _iqr_outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return an IQR-based outlier summary for numeric columns."""
    rows = []

    for column in NUMERIC_COLUMNS:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = int(((df[column] < lower_bound) | (df[column] > upper_bound)).sum())

        rows.append(
            {
                "column": column,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_count": outlier_count,
            }
        )

    return pd.DataFrame(rows)


def clean_dataset(
    raw_path: Path = RAW_DATA_PATH,
    clean_path: Path = CLEAN_DATA_PATH,
    report_path: Path = REPORT_PATH,
) -> tuple[pd.DataFrame, list[str]]:
    """Clean the dataset and save both the cleaned CSV and cleaning report."""
    notes = []

    df = pd.read_csv(raw_path)
    original_shape = df.shape
    notes.append(
        f"Loaded raw dataset with {original_shape[0]:,} rows and {original_shape[1]:,} columns."
    )

    df.columns = df.columns.str.strip().str.lower()
    notes.append(
        "Standardized column names by trimming spaces and converting names to lowercase."
    )

    before_duplicate_rows = len(df)
    df = df.drop_duplicates()
    duplicate_rows_removed = before_duplicate_rows - len(df)
    notes.append(
        f"Checked full-row duplicates and removed {duplicate_rows_removed:,} duplicate rows."
    )

    before_duplicate_ids = len(df)
    df = df.drop_duplicates(subset="student_id", keep="first")
    duplicate_ids_removed = before_duplicate_ids - len(df)
    notes.append(
        f"Checked duplicate student IDs and removed {duplicate_ids_removed:,} duplicate ID records."
    )

    numeric_conversion_issues = {}
    for column in NUMERIC_COLUMNS:
        invalid_before = pd.to_numeric(df[column], errors="coerce").isna().sum() - df[column].isna().sum()
        df[column] = pd.to_numeric(df[column], errors="coerce")
        numeric_conversion_issues[column] = int(invalid_before)
    notes.append(
        "Validated numeric columns by converting them to numeric data types; "
        f"conversion issues found: {sum(numeric_conversion_issues.values()):,}."
    )

    for column in CATEGORICAL_COLUMNS:
        df[column] = df[column].astype(str).str.strip()
    notes.append("Trimmed extra spaces from categorical columns.")

    missing_before = int(df.isna().sum().sum())
    for column in NUMERIC_COLUMNS:
        if df[column].isna().any():
            df[column] = df[column].fillna(df[column].median())

    for column in CATEGORICAL_COLUMNS:
        if df[column].isna().any():
            df[column] = df[column].fillna(df[column].mode(dropna=True)[0])
    missing_after = int(df.isna().sum().sum())
    notes.append(
        f"Checked missing values. Missing values before imputation: {missing_before:,}; "
        f"missing values after imputation: {missing_after:,}."
    )

    negative_addiction_count = int((df["addiction_score"] < 0).sum())
    df["addiction_score"] = df["addiction_score"].clip(lower=0)
    notes.append(
        f"Corrected addiction score validity by setting {negative_addiction_count:,} "
        "negative addiction_score values to 0."
    )

    grade_above_100_count = int((df["grades"] > 100).sum())
    grade_below_0_count = int((df["grades"] < 0).sum())
    df["grades"] = df["grades"].clip(lower=0, upper=100)
    notes.append(
        f"Corrected grade validity by clipping {grade_below_0_count:,} grades below 0 "
        f"and {grade_above_100_count:,} grades above 100 to the valid 0-100 range."
    )

    range_rules = {
        "age": (0, 120),
        "gaming_hours": (0, 24),
        "study_hours": (0, 24),
        "sleep_hours": (0, 24),
        "attendance": (0, 100),
        "social_activity": (0, 24),
        "device_usage": (0, 24),
        "reaction_time_ms": (0, np.inf),
    }
    range_issues = {
        column: int(((df[column] < lower) | (df[column] > upper)).sum())
        for column, (lower, upper) in range_rules.items()
    }
    notes.append(
        "Checked business ranges for age, time-use fields, attendance, and reaction time; "
        f"remaining invalid range values found: {sum(range_issues.values()):,}."
    )

    outlier_summary = _iqr_outlier_summary(df)
    total_outliers = int(outlier_summary["outlier_count"].sum())
    notes.append(
        "Checked statistical outliers using the IQR method; "
        f"{total_outliers:,} outlier values were detected, so no IQR treatment was applied."
    )

    clean_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(clean_path, index=False)
    notes.append(f"Saved cleaned dataset to {clean_path}.")

    report = _build_cleaning_report(df, notes, outlier_summary)
    report_path.write_text(report, encoding="utf-8")

    return df, notes


def _build_cleaning_report(
    df: pd.DataFrame, notes: list[str], outlier_summary: pd.DataFrame
) -> str:
    """Build a Markdown cleaning report."""
    missing_summary = df.isna().sum().to_frame("missing_values")
    dtype_summary = df.dtypes.astype(str).to_frame("data_type")

    return "\n".join(
        [
            "# Data Cleaning Report",
            "",
            "## Cleaning Notes",
            "",
            *[f"- {note}" for note in notes],
            "",
            "## Final Dataset Shape",
            "",
            f"- Rows: {len(df):,}",
            f"- Columns: {len(df.columns):,}",
            "",
            "## Final Data Types",
            "",
            _dataframe_to_markdown(dtype_summary.reset_index(names="column")),
            "",
            "## Final Missing Values",
            "",
            _dataframe_to_markdown(missing_summary.reset_index(names="column")),
            "",
            "## IQR Outlier Summary",
            "",
            _dataframe_to_markdown(outlier_summary),
            "",
        ]
    )


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Convert a small DataFrame to a Markdown table without optional packages."""
    headers = [str(column) for column in df.columns]
    rows = df.astype(str).values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


if __name__ == "__main__":
    _, cleaning_notes = clean_dataset()
    for note in cleaning_notes:
        print(note)
