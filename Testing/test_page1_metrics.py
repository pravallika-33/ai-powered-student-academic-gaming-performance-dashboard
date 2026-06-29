"""Manual validation checks for Page 1 Analytics Dashboard metrics.

Run this script from the project root to compare direct Pandas calculations
against the Streamlit dashboard values.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "Gaming_Academic_Performance_with_target.csv"


def print_section(title: str) -> None:
    """Print a readable section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    """Load the target dataset and print validation metrics."""
    df = pd.read_csv(DATA_PATH)

    print_section("Dataset Overview")
    print(f"Dataset path: {DATA_PATH}")
    print(f"Dataset shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")

    print_section("Column Names")
    for column in df.columns:
        print(f"- {column}")

    print_section("KPI Validation Using Direct Pandas Formulas")
    total_students = len(df)
    print(f"Total students: {total_students:,}")

    if "grades" in df.columns:
        average_grade = df["grades"].mean()
        print(f"Average academic score / grade: {average_grade:.2f}")
    else:
        print("Average academic score / grade: grades column not found")

    if "gaming_hours" in df.columns:
        average_gaming_hours = df["gaming_hours"].mean()
        print(f"Average gaming hours: {average_gaming_hours:.2f}")
    else:
        print("Average gaming hours: gaming_hours column not found")

    if "academic_status" in df.columns:
        risk_count = (df["academic_status"] == "Risk").sum()
        risk_percentage = risk_count / total_students * 100
        print(f"Risk count: {risk_count:,}")
        print(f"Risk percentage: {risk_percentage:.2f}%")
    else:
        print("Risk count: academic_status column not found")
        print("Risk percentage: academic_status column not found")

    print_section("Risk Percentage by Gaming Hours Bins")
    if {"gaming_hours", "academic_status"}.issubset(df.columns):
        gaming_bins = [0, 2, 4, 6, 8]
        gaming_labels = ["0-2 hrs", "2-4 hrs", "4-6 hrs", "6-8 hrs"]
        df["gaming_hours_bin"] = pd.cut(
            df["gaming_hours"],
            bins=gaming_bins,
            labels=gaming_labels,
            include_lowest=True,
            right=True,
        )

        risk_by_bin = (
            df.groupby("gaming_hours_bin", observed=False)
            .agg(
                total_students=("student_id", "count"),
                risk_students=("academic_status", lambda status: (status == "Risk").sum()),
            )
            .reset_index()
        )
        risk_by_bin["risk_percentage"] = (
            risk_by_bin["risk_students"] / risk_by_bin["total_students"] * 100
        ).round(2)

        print(risk_by_bin.to_string(index=False))
    else:
        print("Cannot calculate risk percentage by gaming hours bins.")
        print("Required columns: gaming_hours and academic_status")

    print_section("Gender-wise Academic Status Counts")
    if {"gender", "academic_status"}.issubset(df.columns):
        gender_status_counts = pd.crosstab(df["gender"], df["academic_status"])
        print(gender_status_counts.to_string())
    else:
        print("Cannot print gender-wise academic status counts.")
        print("Required columns: gender and academic_status")

    print_section("Missing Values Per Column")
    print(df.isna().sum().to_string())


if __name__ == "__main__":
    main()
