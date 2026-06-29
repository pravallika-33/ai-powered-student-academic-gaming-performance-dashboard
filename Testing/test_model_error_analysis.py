"""Full-dataset error analysis for the saved academic status model.

Grades are printed only for validation/debugging. They are not used as model
input features.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "Gaming_Academic_Performance_with_target.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_academic_status_model.joblib"
MISMATCH_OUTPUT_PATH = PROJECT_ROOT / "model_mismatch_analysis.csv"

TARGET_COLUMN = "academic_status"
ML_FEATURES = [
    "age",
    "gender",
    "gaming_hours",
    "study_hours",
    "sleep_hours",
    "attendance",
    "gaming_genre",
    "social_activity",
    "device_usage",
    "reaction_time_ms",
    "addiction_score",
    "stress_level",
]

MISMATCH_COLUMNS = [
    "student_id",
    "grades",
    "actual_academic_status",
    "predicted_academic_status",
    "gaming_hours",
    "study_hours",
    "sleep_hours",
    "attendance",
    "addiction_score",
    "stress_level",
]


def print_section(title: str) -> None:
    """Print a readable section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    """Run full-dataset model error analysis."""
    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    X = df[ML_FEATURES]
    y_actual = df[TARGET_COLUMN]
    y_predicted = model.predict(X)

    analysis_df = df.copy()
    analysis_df["actual_academic_status"] = y_actual
    analysis_df["predicted_academic_status"] = y_predicted
    analysis_df["prediction_matches"] = (
        analysis_df["actual_academic_status"]
        == analysis_df["predicted_academic_status"]
    )

    total_records = len(analysis_df)
    correct_predictions = int(analysis_df["prediction_matches"].sum())
    incorrect_predictions = total_records - correct_predictions
    overall_accuracy = correct_predictions / total_records

    print_section("Model Error Analysis Summary")
    print(f"Dataset path: {DATA_PATH}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Total records: {total_records:,}")
    print(f"Correct predictions: {correct_predictions:,}")
    print(f"Incorrect predictions: {incorrect_predictions:,}")
    print(f"Overall accuracy: {overall_accuracy:.4f}")
    print("\nInput features used by the model:")
    for feature in ML_FEATURES:
        print(f"- {feature}")
    print("\nExcluded from model input: student_id, grades, academic_status")

    labels = sorted(y_actual.unique())

    print_section("Confusion Matrix")
    matrix = confusion_matrix(y_actual, y_predicted, labels=labels)
    confusion_df = pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in labels],
        columns=[f"predicted_{label}" for label in labels],
    )
    print(confusion_df.to_string())

    print_section("Classification Report")
    print(classification_report(y_actual, y_predicted, labels=labels))

    mismatch_df = analysis_df[~analysis_df["prediction_matches"]][MISMATCH_COLUMNS]
    mismatch_df.to_csv(MISMATCH_OUTPUT_PATH, index=False)

    print_section("First 10 Mismatched Predictions")
    if mismatch_df.empty:
        print("No mismatched predictions found.")
    else:
        print(mismatch_df.head(10).to_string(index=False))

    print_section("Mismatch Export")
    print(f"Saved mismatch results to: {MISMATCH_OUTPUT_PATH}")
    print("Reminder: grades were used only for validation/debugging, not model input.")


if __name__ == "__main__":
    main()
