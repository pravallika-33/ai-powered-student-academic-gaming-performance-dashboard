"""Student-only page for academic status prediction by student ID."""

from pathlib import Path
import sys

import joblib
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from auth import render_sidebar_navigation, require_roles  # noqa: E402
from model_training import (  # noqa: E402
    ML_FEATURES,
    MODEL_PATH,
    get_recommendations,
    load_ml_dataset,
)


st.set_page_config(page_title="Student Prediction", layout="wide")

require_roles(["Student"])
render_sidebar_navigation()


@st.cache_data
def get_student_dataset():
    """Load the dataset used for student ID lookup."""
    return load_ml_dataset()


@st.cache_resource
def load_saved_model():
    """Load the saved best model pipeline if it exists."""
    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


def find_student_record(student_df, student_id):
    """Return the matching student row for the submitted student ID."""
    match = student_df[student_df["student_id"] == student_id]

    if match.empty:
        return None

    return match.iloc[0]


def render_prediction(prediction):
    """Display the predicted academic status with status-specific styling."""
    if prediction == "Risk":
        st.error(f"Predicted Academic Status: {prediction}")
    elif prediction == "Medium":
        st.warning(f"Predicted Academic Status: {prediction}")
    else:
        st.success(f"Predicted Academic Status: {prediction}")


def render_student_record(student_record):
    """Display the matched student record in grouped dashboard sections."""
    st.subheader("Student Profile")
    profile_cols = st.columns(3)
    profile_cols[0].metric("Student ID", int(student_record["student_id"]))
    profile_cols[1].metric("Age", int(student_record["age"]))
    profile_cols[2].metric("Gender", student_record["gender"])

    st.subheader("Behavior Summary")
    behavior_cols = st.columns(4)
    behavior_cols[0].metric("Gaming Hours", f"{student_record['gaming_hours']:.2f}")
    behavior_cols[1].metric("Study Hours", f"{student_record['study_hours']:.2f}")
    behavior_cols[2].metric("Sleep Hours", f"{student_record['sleep_hours']:.2f}")
    behavior_cols[3].metric("Attendance", f"{student_record['attendance']:.2f}%")

    st.subheader("Gaming and Lifestyle")
    lifestyle_row_1 = st.columns(3)
    lifestyle_row_1[0].metric("Gaming Genre", student_record["gaming_genre"])
    lifestyle_row_1[1].metric("Social Activity", f"{student_record['social_activity']:.2f}")
    lifestyle_row_1[2].metric("Device Usage", f"{student_record['device_usage']:.2f}")

    lifestyle_row_2 = st.columns(3)
    lifestyle_row_2[0].metric(
        "Reaction Time MS", f"{student_record['reaction_time_ms']:.2f}"
    )
    lifestyle_row_2[1].metric(
        "Addiction Score", f"{student_record['addiction_score']:.2f}"
    )
    lifestyle_row_2[2].metric("Stress Level", student_record["stress_level"])


st.title("Student Prediction")
st.write(
    "Enter your student ID to view your record, predicted academic status, "
    "and personalized recommendations."
)

data = get_student_dataset()

if st.button("Show Student ID Format Hint"):
    st.write(
        f"Student IDs are stored as whole numbers from "
        f"`{int(data['student_id'].min())}` to `{int(data['student_id'].max())}`."
    )
    st.write("Example student IDs:", ", ".join(map(str, data["student_id"].head(5))))

with st.form("student_prediction_form"):
    student_id = st.number_input(
        "Student ID",
        min_value=int(data["student_id"].min()),
        max_value=int(data["student_id"].max()),
        step=1,
        value=int(data["student_id"].min()),
    )
    submitted = st.form_submit_button("Find My Prediction")

if not submitted:
    st.stop()

student_record = find_student_record(data, int(student_id))

if student_record is None:
    st.error("Student ID not found. Please check the ID and try again.")
    st.stop()

render_student_record(student_record)

model = load_saved_model()

if model is None:
    st.error("Model file not found. Please ask Admin to train the model first.")
    st.stop()

# Use only approved ML input features. Exclude student_id, grades, and academic_status.
prediction_input = student_record[ML_FEATURES].to_frame().T
prediction = model.predict(prediction_input)[0]

st.subheader("Prediction Result")
render_prediction(prediction)

st.write(
    "This prediction is based on behavior and lifestyle features. "
    "Grades are not used by the model."
)

st.subheader("Recommendation Output")
for recommendation in get_recommendations(prediction):
    st.write(f"- {recommendation}")
