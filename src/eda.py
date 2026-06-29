"""Exploratory data analysis utilities for the Streamlit dashboard project."""

from pathlib import Path

import pandas as pd
import plotly.express as px

from feature_engineering import TARGET_COLUMN, TARGET_RULES, create_target_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_DATA_PATH = PROJECT_ROOT / "data" / "Gaming_Academic_Performance_with_target.csv"
EDA_REPORT_PATH = PROJECT_ROOT / "data" / "eda_report.md"


DIMENSION_COLUMNS = [
    "student_id",
    "gender",
    "gaming_genre",
    "stress_level",
    TARGET_COLUMN,
]

MEASURE_COLUMNS = [
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

NUMERICAL_INPUT_FEATURES = [
    "age",
    "gaming_hours",
    "study_hours",
    "sleep_hours",
    "attendance",
    "social_activity",
    "device_usage",
    "reaction_time_ms",
    "addiction_score",
]

SUPPORTED_CHART_FEATURES = {
    "gaming hours": "gaming_hours",
    "gaming_hours": "gaming_hours",
    "study hours": "study_hours",
    "study_hours": "study_hours",
    "sleep hours": "sleep_hours",
    "sleep_hours": "sleep_hours",
    "attendance": "attendance",
    "addiction score": "addiction_score",
    "addiction_score": "addiction_score",
    "device usage": "device_usage",
    "device_usage": "device_usage",
    "social activity": "social_activity",
    "social_activity": "social_activity",
}

FEATURE_BIN_CONFIG = {
    "gaming_hours": {
        "labels": ["0-2 hrs", "2-4 hrs", "4-6 hrs", "6-8 hrs"],
        "bins": [0, 2, 4, 6, 8],
        "title": "Gaming Hours",
    },
    "study_hours": {
        "labels": ["1-3 hrs", "3-5 hrs", "5-7 hrs", "7-10 hrs"],
        "bins": [1, 3, 5, 7, 10],
        "title": "Study Hours",
    },
    "sleep_hours": {
        "labels": ["<5 hrs", "5-6 hrs", "6-7 hrs", "7-8 hrs", "8+ hrs"],
        "bins": [0, 5, 6, 7, 8, 24],
        "title": "Sleep Hours",
    },
    "attendance": {
        "labels": ["60-70%", "70-80%", "80-90%", "90-100%"],
        "bins": [60, 70, 80, 90, 100],
        "title": "Attendance",
    },
    "addiction_score": {
        "labels": ["0-5", "5-10", "10-15", "15-20", "20+"],
        "bins": [0, 5, 10, 15, 20, 100],
        "title": "Addiction Score",
    },
    "device_usage": {
        "labels": ["0-4 hrs", "4-7 hrs", "7-10 hrs", "10+ hrs"],
        "bins": [0, 4, 7, 10, 24],
        "title": "Device Usage",
    },
    "social_activity": {
        "labels": ["0-1 hrs", "1-2 hrs", "2-3 hrs", "3-4 hrs", "4-5 hrs"],
        "bins": [0, 1, 2, 3, 4, 5],
        "title": "Social Activity",
    },
}

PERFORMANCE_METRICS = {
    "average grade": "Average Grade",
    "student count": "Student Count",
}

STATUS_COLOR_MAP = {
    "Risk": "#d62728",
    "Medium": "#ffbf00",
    "Safe": "#2ca02c",
}

ACADEMIC_STATUS_ORDER = ["Risk", "Medium", "Safe"]


def load_target_dataset(data_path: Path = TARGET_DATA_PATH) -> pd.DataFrame:
    """Load the cleaned target dataset, creating it first if needed."""
    if data_path.exists():
        return pd.read_csv(data_path)

    return create_target_dataset()


def _kpi_card(name: str, raw_value: float | int, display_value: str) -> dict[str, float | int | str]:
    """Return a consistent dashboard-ready KPI card payload."""
    return {
        "name": name,
        "raw_value": raw_value,
        "display_value": display_value,
    }


def get_total_students(df: pd.DataFrame) -> dict[str, float | int | str]:
    """Formula: count total student records in the dataset."""
    value = int(len(df))
    return _kpi_card("Total Students", value, f"{value:,}")


def get_average_grade(df: pd.DataFrame) -> dict[str, float | int | str]:
    """Formula: average grade = sum(grades) / total students."""
    value = round(float(df["grades"].mean()), 2)
    return _kpi_card("Average Grade", value, f"{value:.2f}")


def get_average_gaming_hours(df: pd.DataFrame) -> dict[str, float | int | str]:
    """Formula: average gaming hours = sum(gaming_hours) / total students."""
    value = round(float(df["gaming_hours"].mean()), 2)
    return _kpi_card("Average Gaming Hours", value, f"{value:.2f} hrs")


def get_average_study_hours(df: pd.DataFrame) -> dict[str, float | int | str]:
    """Formula: average study hours = sum(study_hours) / total students."""
    value = round(float(df["study_hours"].mean()), 2)
    return _kpi_card("Average Study Hours", value, f"{value:.2f} hrs")


def get_average_sleep_hours(df: pd.DataFrame) -> dict[str, float | int | str]:
    """Formula: average sleep hours = sum(sleep_hours) / total students."""
    value = round(float(df["sleep_hours"].mean()), 2)
    return _kpi_card("Average Sleep Hours", value, f"{value:.2f} hrs")


def get_average_attendance(df: pd.DataFrame) -> dict[str, float | int | str]:
    """Formula: average attendance = sum(attendance) / total students."""
    value = round(float(df["attendance"].mean()), 2)
    return _kpi_card("Average Attendance", value, f"{value:.2f}%")


def get_risk_students_count(df: pd.DataFrame) -> dict[str, float | int | str]:
    """Formula: count students where academic_status equals Risk."""
    value = int((df[TARGET_COLUMN] == "Risk").sum())
    return _kpi_card("Risk Students Count", value, f"{value:,}")


def get_risk_students_percentage(df: pd.DataFrame) -> dict[str, float | int | str]:
    """Formula: risk percentage = risk student count / total students * 100."""
    value = round(float((df[TARGET_COLUMN] == "Risk").mean() * 100), 2)
    return _kpi_card("Risk Students Percentage", value, f"{value:.2f}%")


def get_average_addiction_score(df: pd.DataFrame) -> dict[str, float | int | str]:
    """Formula: average addiction score = sum(addiction_score) / total students."""
    value = round(float(df["addiction_score"].mean()), 2)
    return _kpi_card("Average Addiction Score", value, f"{value:.2f}")


def calculate_all_kpis(data_path: Path = TARGET_DATA_PATH) -> list[dict[str, float | int | str]]:
    """Return all Page 1 KPI cards in dashboard display order."""
    df = load_target_dataset(data_path)

    return [
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


def create_risk_student_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize student count, percentage, and average behavior metrics by academic status."""
    total_students = len(df)

    # Group by academic status to compare Risk, Medium, and Safe students side by side.
    summary = (
        df.groupby(TARGET_COLUMN)
        .agg(
            student_count=("student_id", "count"),
            avg_grade=("grades", "mean"),
            avg_gaming_hours=("gaming_hours", "mean"),
            avg_study_hours=("study_hours", "mean"),
            avg_sleep_hours=("sleep_hours", "mean"),
            avg_attendance=("attendance", "mean"),
            avg_addiction_score=("addiction_score", "mean"),
        )
        .reset_index()
    )

    # Percentage = students in academic status / total students * 100.
    summary["percentage"] = (summary["student_count"] / total_students * 100).round(2)

    average_columns = [
        "avg_grade",
        "avg_gaming_hours",
        "avg_study_hours",
        "avg_sleep_hours",
        "avg_attendance",
        "avg_addiction_score",
    ]
    summary[average_columns] = summary[average_columns].round(2)

    # Keep the table in intervention-first order for dashboard readability.
    summary[TARGET_COLUMN] = pd.Categorical(
        summary[TARGET_COLUMN],
        categories=ACADEMIC_STATUS_ORDER,
        ordered=True,
    )
    summary = summary.sort_values(TARGET_COLUMN)
    summary[TARGET_COLUMN] = summary[TARGET_COLUMN].astype(str)

    return summary[
        [
            TARGET_COLUMN,
            "student_count",
            "percentage",
            "avg_grade",
            "avg_gaming_hours",
            "avg_study_hours",
            "avg_sleep_hours",
            "avg_attendance",
            "avg_addiction_score",
        ]
    ].reset_index(drop=True)


def normalize_chart_feature(selected_feature: str) -> str:
    """Return the dataset column name for a supported dashboard feature."""
    feature_key = selected_feature.strip().lower()

    if feature_key not in SUPPORTED_CHART_FEATURES:
        supported = ", ".join(sorted(SUPPORTED_CHART_FEATURES))
        raise ValueError(f"Unsupported feature '{selected_feature}'. Use one of: {supported}.")

    return SUPPORTED_CHART_FEATURES[feature_key]


def add_feature_bins(df: pd.DataFrame, selected_feature: str) -> tuple[pd.DataFrame, str, list[str]]:
    """Create ordered feature bins for the selected numeric behavior/lifestyle field."""
    feature_column = normalize_chart_feature(selected_feature)
    config = FEATURE_BIN_CONFIG[feature_column]
    binned_column = f"{feature_column}_bin"
    binned_df = df.copy()

    # pd.cut converts continuous values into readable dashboard groups.
    binned_df[binned_column] = pd.cut(
        binned_df[feature_column],
        bins=config["bins"],
        labels=config["labels"],
        include_lowest=True,
        right=True,
    )

    return binned_df, binned_column, config["labels"]


def create_dynamic_performance_bar_chart(
    df: pd.DataFrame,
    selected_feature: str,
    selected_metric: str,
):
    """Create a bar chart for Average Grade or Student Count by selected feature bin."""
    metric_key = selected_metric.strip().lower()

    if metric_key not in PERFORMANCE_METRICS:
        raise ValueError("selected_metric must be 'Average Grade' or 'Student Count'.")

    binned_df, binned_column, category_order = add_feature_bins(df, selected_feature)
    feature_column = normalize_chart_feature(selected_feature)
    feature_title = FEATURE_BIN_CONFIG[feature_column]["title"]

    if metric_key == "average grade":
        # Average Grade = average of grades within each selected feature bin.
        chart_data = (
            binned_df.groupby(binned_column, observed=False)["grades"]
            .mean()
            .round(2)
            .reset_index(name="value")
        )
        y_axis_title = "Average Grade"
        chart_title = f"Average Grade by {feature_title}"
    else:
        # Student Count = number of students within each selected feature bin.
        chart_data = (
            binned_df.groupby(binned_column, observed=False)["student_id"]
            .count()
            .reset_index(name="value")
        )
        y_axis_title = "Student Count"
        chart_title = f"Student Count by {feature_title}"

    fig = px.bar(
        chart_data,
        x=binned_column,
        y="value",
        text="value",
        category_orders={binned_column: category_order},
        labels={binned_column: feature_title, "value": y_axis_title},
        title=chart_title,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_title=feature_title, yaxis_title=y_axis_title)

    return fig


def create_dynamic_risk_percentage_bar_chart(
    df: pd.DataFrame,
    selected_feature: str,
):
    """Create a bar chart showing Risk students as a percentage of each feature bin."""
    chart_data, binned_column, category_order, feature_title = _risk_percentage_by_bin(
        df, selected_feature
    )

    # Risk Percentage = Risk students in bin / total students in bin * 100.
    fig = px.bar(
        chart_data,
        x=binned_column,
        y="risk_percentage",
        text="risk_percentage",
        category_orders={binned_column: category_order},
        labels={binned_column: feature_title, "risk_percentage": "Risk Percentage (%)"},
        title=f"Risk Percentage by {feature_title}",
    )
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(xaxis_title=feature_title, yaxis_title="Risk Percentage (%)")

    return fig


def create_risk_progression_line_chart(
    df: pd.DataFrame,
    selected_feature: str,
):
    """Create a line chart showing how Risk Percentage changes across feature bins."""
    chart_data, binned_column, category_order, feature_title = _risk_percentage_by_bin(
        df, selected_feature
    )

    # The line connects ordered bins to show risk progression as the feature increases.
    fig = px.line(
        chart_data,
        x=binned_column,
        y="risk_percentage",
        markers=True,
        category_orders={binned_column: category_order},
        labels={binned_column: feature_title, "risk_percentage": "Risk Percentage (%)"},
        title=f"Risk Progression by {feature_title}",
    )
    fig.update_traces(text=chart_data["risk_percentage"], texttemplate="%{text:.2f}%")
    fig.update_layout(xaxis_title=feature_title, yaxis_title="Risk Percentage (%)")

    return fig


def create_gaming_hours_vs_grades_scatter_plot(df: pd.DataFrame):
    """Create a scatter plot of gaming hours against grades colored by academic status."""
    # Each point represents one student, making individual risk patterns visible.
    fig = px.scatter(
        df,
        x="gaming_hours",
        y="grades",
        color=TARGET_COLUMN,
        color_discrete_map=STATUS_COLOR_MAP,
        hover_data=[
            "student_id",
            "study_hours",
            "sleep_hours",
            "attendance",
            "addiction_score",
        ],
        labels={
            "gaming_hours": "Gaming Hours",
            "grades": "Grades",
            TARGET_COLUMN: "Academic Status",
            "student_id": "Student ID",
            "study_hours": "Study Hours",
            "sleep_hours": "Sleep Hours",
            "attendance": "Attendance",
            "addiction_score": "Addiction Score",
        },
        title="Gaming Hours vs Grades",
    )
    fig.update_layout(xaxis_title="Gaming Hours", yaxis_title="Grades")

    return fig


def _risk_percentage_by_bin(
    df: pd.DataFrame,
    selected_feature: str,
) -> tuple[pd.DataFrame, str, list[str], str]:
    """Aggregate Risk Percentage by selected feature bin for risk charts."""
    binned_df, binned_column, category_order = add_feature_bins(df, selected_feature)
    feature_column = normalize_chart_feature(selected_feature)
    feature_title = FEATURE_BIN_CONFIG[feature_column]["title"]

    chart_data = (
        binned_df.groupby(binned_column, observed=False)
        .agg(
            total_students=("student_id", "count"),
            risk_students=(TARGET_COLUMN, lambda status: (status == "Risk").sum()),
        )
        .reset_index()
    )
    chart_data["risk_percentage"] = (
        chart_data["risk_students"] / chart_data["total_students"] * 100
    ).round(2)
    chart_data["risk_percentage"] = chart_data["risk_percentage"].fillna(0)

    return chart_data, binned_column, category_order, feature_title


def generate_eda_report(
    data_path: Path = TARGET_DATA_PATH,
    report_path: Path = EDA_REPORT_PATH,
) -> dict[str, pd.DataFrame]:
    """Create an EDA summary report for dimensions, measures, and target classes."""
    df = load_target_dataset(data_path)

    dataset_shape = pd.DataFrame(
        [{"rows": len(df), "columns": len(df.columns)}]
    )
    dimensions = pd.DataFrame(
        {
            "dimension": DIMENSION_COLUMNS,
            "unique_values": [df[column].nunique() for column in DIMENSION_COLUMNS],
        }
    )
    measures = pd.DataFrame(
        {
            "measure": MEASURE_COLUMNS,
            "data_type": [str(df[column].dtype) for column in MEASURE_COLUMNS],
        }
    )
    averages = (
        df[NUMERICAL_INPUT_FEATURES]
        .mean()
        .round(2)
        .reset_index()
        .rename(columns={"index": "numerical_input_feature", 0: "average"})
    )
    summary_statistics = df[NUMERICAL_INPUT_FEATURES + ["grades"]].describe().round(2)
    target_distribution = (
        df[TARGET_COLUMN]
        .value_counts()
        .rename_axis(TARGET_COLUMN)
        .reset_index(name="student_count")
    )
    target_distribution["percentage"] = (
        target_distribution["student_count"] / len(df) * 100
    ).round(2)

    report = _build_report(
        dataset_shape,
        dimensions,
        measures,
        averages,
        summary_statistics,
        target_distribution,
    )
    report_path.write_text(report, encoding="utf-8")

    return {
        "dataset_shape": dataset_shape,
        "dimensions": dimensions,
        "measures": measures,
        "averages": averages,
        "summary_statistics": summary_statistics,
        "target_distribution": target_distribution,
    }


def _build_report(
    dataset_shape: pd.DataFrame,
    dimensions: pd.DataFrame,
    measures: pd.DataFrame,
    averages: pd.DataFrame,
    summary_statistics: pd.DataFrame,
    target_distribution: pd.DataFrame,
) -> str:
    """Build a Markdown EDA report."""
    target_rules = "\n".join(
        f"- {status}: `{rule}`" for status, rule in TARGET_RULES.items()
    )

    return "\n".join(
        [
            "# Exploratory Data Analysis Report",
            "",
            "## Dataset Size",
            "",
            _dataframe_to_markdown(dataset_shape),
            "",
            "## Dimensions",
            "",
            "Dimensions are categorical or identifier fields used for filtering, grouping, and segmentation.",
            "",
            _dataframe_to_markdown(dimensions),
            "",
            "## Measures",
            "",
            "Measures are numerical fields used for aggregation, comparison, and trend analysis.",
            "",
            _dataframe_to_markdown(measures),
            "",
            "## Numerical Input Feature Averages",
            "",
            _dataframe_to_markdown(averages),
            "",
            "## Summary Statistics",
            "",
            _dataframe_to_markdown(summary_statistics.reset_index(names="statistic")),
            "",
            "## Target Column Decision",
            "",
            f"Target column: `{TARGET_COLUMN}`",
            "",
            "Target classes:",
            "",
            target_rules,
            "",
            "## Target Distribution",
            "",
            _dataframe_to_markdown(target_distribution),
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
    summaries = generate_eda_report()
    print("Generated EDA report.")
    print(summaries["dataset_shape"].to_string(index=False))
    print(summaries["target_distribution"].to_string(index=False))
