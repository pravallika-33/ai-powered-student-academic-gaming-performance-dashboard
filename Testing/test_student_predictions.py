"""Validate saved model predictions for selected student IDs.

This script is for manual debugging only. It prints grades for validation, but
grades are never used as an ML input feature.
"""

from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "Gaming_Academic_Performance_with_target.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_academic_status_model.joblib"

TARGET_COLUMN = "academic_status"
ML_FEATURES = [
    "age",
    "gaming_hours",
    "study_hours",
    "sleep_hours",
    "attendance",
    "social_activity",
    "device_usage",
    "reaction_time_ms",
    "addiction_score",
    "gender",
    "gaming_genre",
    "stress_level",
]

SAMPLE_STUDENT_IDS = [1, 2, 3, 4, 5]


def print_section(title: str) -> None:
    """Print a readable section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def parse_student_ids(raw_input: str) -> list[int]:
    """Parse comma-separated student IDs, or return sample IDs when blank."""
    if not raw_input.strip():
        return SAMPLE_STUDENT_IDS

    return [int(value.strip()) for value in raw_input.split(",") if value.strip()]


def print_prediction_probabilities(model, prediction_input: pd.DataFrame) -> None:
    """Print class probabilities when the saved model supports predict_proba."""
    if not hasattr(model, "predict_proba"):
        print("Prediction probabilities: not available for this model")
        return

    probabilities = model.predict_proba(prediction_input)[0]
    print("Prediction probabilities:")
    for class_name, probability in zip(model.classes_, probabilities):
        print(f"  {class_name}: {probability * 100:.2f}%")


def validate_student_prediction(df: pd.DataFrame, model, student_id: int) -> None:
    """Print actual vs predicted academic status for one student ID."""
    print_section(f"Student ID: {student_id}")

    matching_rows = df[df["student_id"] == student_id]
    if matching_rows.empty:
        print("Student ID not found. Please check the ID and try again.")
        return

    student = matching_rows.iloc[0]
    prediction_input = student[ML_FEATURES].to_frame().T
    predicted_status = model.predict(prediction_input)[0]
    actual_status = student[TARGET_COLUMN]

    print("Behavior features used for prediction:")
    for feature in ML_FEATURES:
        print(f"  {feature}: {student[feature]}")

    print("\nValidation fields:")
    print(f"  Actual grades: {student['grades']:.2f}")
    print(f"  Actual academic_status: {actual_status}")
    print(f"  Predicted academic_status: {predicted_status}")

    print_prediction_probabilities(model, prediction_input)

    if predicted_status == actual_status:
        print("\nMatch result: MATCH")
    else:
        print("\nMatch result: NOT A MATCH")


def main() -> None:
    """Run manual prediction validation for selected student IDs."""
    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    print_section("Prediction Validation")
    print(f"Dataset path: {DATA_PATH}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Dataset shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")
    print("Grades are printed only for validation/debugging.")
    print("Grades are not used as ML input features.")

    raw_student_ids = input(
        f"\nEnter student_id values separated by commas, or press Enter to use "
        f"sample IDs {SAMPLE_STUDENT_IDS}: "
    )
    student_ids = parse_student_ids(raw_student_ids)

    for student_id in student_ids:
        validate_student_prediction(df, model, student_id)


if __name__ == "__main__":
    main()
