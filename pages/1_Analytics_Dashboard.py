"""Analytics Dashboard page for gaming behavior and academic performance."""

from pathlib import Path
import sys

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from eda import (  # noqa: E402
    TARGET_COLUMN,
    create_dynamic_performance_bar_chart,
    create_dynamic_risk_percentage_bar_chart,
    create_gaming_hours_vs_grades_scatter_plot,
    create_risk_progression_line_chart,
    create_risk_student_summary_table,
    get_average_addiction_score,
    get_average_attendance,
    get_average_gaming_hours,
    get_average_grade,
    get_average_sleep_hours,
    get_average_study_hours,
    get_risk_students_count,
    get_risk_students_percentage,
    get_total_students,
    load_target_dataset,
)
from auth import render_sidebar_navigation, require_roles  # noqa: E402


CHART_FEATURES = [
    "Gaming Hours",
    "Study Hours",
    "Sleep Hours",
    "Attendance",
    "Addiction Score",
    "Device Usage",
    "Social Activity",
]

PERFORMANCE_METRICS = ["Average Grade", "Student Count"]


st.set_page_config(
    page_title="Analytics Dashboard",
    layout="wide",
)

require_roles(["Admin", "Analyst"])
render_sidebar_navigation()


@st.cache_data
def get_dashboard_data():
    """Load the cleaned target dataset once for dashboard use."""
    return load_target_dataset()


def filter_dashboard_data(df):
    """Apply sidebar filters for demographic, gaming, stress, and academic segments."""
    st.sidebar.header("Filters")

    selected_gender = st.sidebar.multiselect(
        "Gender",
        options=sorted(df["gender"].unique()),
        default=sorted(df["gender"].unique()),
    )

    gaming_min = float(df["gaming_hours"].min())
    gaming_max = float(df["gaming_hours"].max())
    
    selected_gaming_range = st.sidebar.slider(
        "Gaming Hours",
        min_value=gaming_min,
        max_value=gaming_max,
        value=(gaming_min, gaming_max),
        step=0.25,
    )

    selected_stress_level = st.sidebar.multiselect(
        "Stress Level",
        options=sorted(df["stress_level"].unique()),
        default=sorted(df["stress_level"].unique()),
    )

    selected_academic_status = st.sidebar.multiselect(
        "Academic Status",
        options=["Risk", "Medium", "Safe"],
        default=["Risk", "Medium", "Safe"],
    )

    return df[
        df["gender"].isin(selected_gender)
        & df["stress_level"].isin(selected_stress_level)
        & df[TARGET_COLUMN].isin(selected_academic_status)
        & df["gaming_hours"].between(
            selected_gaming_range[0],
            selected_gaming_range[1],
            inclusive="both",
        )
    ].copy()


def render_kpi_cards(df):
    """Render KPI cards using existing KPI calculation functions."""
    kpis = [
        get_total_students(df),
        get_average_grade(df),
        get_risk_students_count(df),
        get_risk_students_percentage(df),
        get_average_gaming_hours(df),
        get_average_study_hours(df),
        get_average_sleep_hours(df),
        get_average_attendance(df),
        get_average_addiction_score(df),
    ]

    for row_start in range(0, len(kpis), 3):
        columns = st.columns(3)
        for column, kpi in zip(columns, kpis[row_start : row_start + 3]):
            column.metric(kpi["name"], kpi["display_value"])


def render_insights(df):
    """Display concise business insights based on the currently filtered data."""
    summary = create_risk_student_summary_table(df)

    if not {"Risk", "Safe"}.issubset(set(summary[TARGET_COLUMN])):
        st.subheader("Insights")
        st.info(
            "Select both Risk and Safe academic status filters to compare behavior gaps."
        )
        return

    risk_row = summary[summary[TARGET_COLUMN] == "Risk"].iloc[0]
    safe_row = summary[summary[TARGET_COLUMN] == "Safe"].iloc[0]
    gaming_gap = risk_row["avg_gaming_hours"] - safe_row["avg_gaming_hours"]
    study_gap = safe_row["avg_study_hours"] - risk_row["avg_study_hours"]
    addiction_gap = risk_row["avg_addiction_score"] - safe_row["avg_addiction_score"]

    st.subheader("Insights")
    st.markdown(
        f"""
        - Risk students represent **{risk_row['percentage']:.2f}%** of the selected student group.
        - Risk students game about **{gaming_gap:.2f} more hours** than Safe students on average.
        - Safe students study about **{study_gap:.2f} more hours** than Risk students on average.
        - Risk students have an average addiction score **{addiction_gap:.2f} points higher** than Safe students.
        """
    )


st.title("Analytics Dashboard")
st.caption("Gaming behavior, lifestyle habits, academic performance, and risk segmentation")

data = get_dashboard_data()
filtered_data = filter_dashboard_data(data)

if filtered_data.empty:
    st.warning("No records match the selected filters. Adjust the sidebar filters to continue.")
    st.stop()

render_kpi_cards(filtered_data)

st.divider()

left_controls, right_controls = st.columns(2)
with left_controls:
    performance_feature = st.selectbox(
        "Performance Feature",
        options=CHART_FEATURES,
        index=0,
    )
with right_controls:
    performance_metric = st.selectbox(
        "Performance Metric",
        options=PERFORMANCE_METRICS,
        index=0,
    )

st.plotly_chart(
    create_dynamic_performance_bar_chart(
        filtered_data,
        performance_feature,
        performance_metric,
    ),
    width="stretch",
)

left_chart, right_chart = st.columns(2)
with left_chart:
    risk_feature = st.selectbox(
        "Risk Feature",
        options=CHART_FEATURES,
        index=0,
    )
    st.plotly_chart(
        create_dynamic_risk_percentage_bar_chart(filtered_data, risk_feature),
        width="stretch",
    )

with right_chart:
    progression_feature = st.selectbox(
        "Risk Progression Feature",
        options=CHART_FEATURES,
        index=0,
    )
    st.plotly_chart(
        create_risk_progression_line_chart(filtered_data, progression_feature),
        width="stretch",
    )

st.subheader("Gaming Hours vs Grades")
st.plotly_chart(
    create_gaming_hours_vs_grades_scatter_plot(filtered_data),
    width="stretch",
)

st.subheader("Risk Student Summary")
st.dataframe(
    create_risk_student_summary_table(filtered_data),
    width="stretch",
    hide_index=True,
)

render_insights(filtered_data)
