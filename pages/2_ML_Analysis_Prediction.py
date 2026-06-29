"""Admin and Analyst ML analysis and prediction page."""

from pathlib import Path
import sys

import plotly.express as px
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from auth import render_sidebar_navigation, require_roles  # noqa: E402
from ai_agent import (  # noqa: E402
    AIInsightsAPIError,
    AIInsightsResponseError,
    AIInsightsTimeoutError,
    build_ai_insights_context,
    build_fallback_summary,
    generate_ai_insights,
)
from model_training import (  # noqa: E402
    CATEGORICAL_FEATURES,
    EXCLUDED_COLUMNS,
    ML_FEATURES,
    MODEL_PATH,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    get_random_forest_feature_importance,
    get_recommendations,
    load_ml_dataset,
    predict_academic_status,
    train_and_evaluate_models,
)


st.set_page_config(page_title="ML Analysis & Prediction", layout="wide")

require_roles(["Admin", "Analyst"])
render_sidebar_navigation()


@st.cache_data
def get_ml_dataset():
    """Load the ML dataset once for this Streamlit page."""
    return load_ml_dataset()


@st.cache_resource
def get_trained_model_results():
    """Train required models once and cache the fitted pipelines."""
    dataset = load_ml_dataset()
    return train_and_evaluate_models(dataset)


def render_dataset_overview(df):
    """Show high-level dataset and ML feature information."""
    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", f"{df.shape[1]:,}")
    col3.metric("ML Input Features", len(ML_FEATURES))

    st.write("Target column:", f"`{TARGET_COLUMN}`")
    st.write("Excluded from training:", ", ".join(f"`{col}`" for col in EXCLUDED_COLUMNS))
    st.write("Numeric features:", ", ".join(f"`{col}`" for col in NUMERIC_FEATURES))
    st.write("Categorical features:", ", ".join(f"`{col}`" for col in CATEGORICAL_FEATURES))


def render_target_distribution(df):
    """Display target class counts and percentages."""
    st.subheader("Existing Academic Status Distribution in Dataset")
    st.write(
        "This table shows how many students in the dataset are labeled as Risk, "
        "Medium, or Safe. These labels are used as the target classes for model training."
    )

    distribution = (
        df[TARGET_COLUMN]
        .value_counts()
        .rename_axis(TARGET_COLUMN)
        .reset_index(name="student_count")
    )
    distribution["percentage"] = (
        distribution["student_count"] / len(df) * 100
    ).round(2)

    left, right = st.columns([1, 2])
    with left:
        st.dataframe(distribution, hide_index=True, width="stretch")
    with right:
        fig = px.bar(
            distribution,
            x=TARGET_COLUMN,
            y="student_count",
            text="student_count",
            title="Academic Status Distribution",
            labels={TARGET_COLUMN: "Academic Status", "student_count": "Student Count"},
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, width="stretch")


def render_model_results(comparison_df, best_model_name, best_scores):
    """Display model comparison and best model details."""
    st.subheader("Model Training and Evaluation")
    st.write(
        "Models are trained using behavior and lifestyle features only. "
        "`student_id`, `grades`, and `academic_status` are excluded from input features."
    )

    st.subheader("Model Comparison Table")
    st.dataframe(comparison_df, hide_index=True, width="stretch")
    st.info(
        f"{best_model_name} performed best based on weighted F1-score. "
        "`grades` was excluded to avoid data leakage because it was used to create "
        "`academic_status`. The model predicts `academic_status` using behavior and "
        "lifestyle features."
    )

    st.subheader("Best Model Summary")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Best Model", best_model_name)
    col2.metric("Accuracy", f"{best_scores['accuracy']:.4f}")
    col3.metric("Precision", f"{best_scores['precision_weighted']:.4f}")
    col4.metric("Recall", f"{best_scores['recall_weighted']:.4f}")
    col5.metric("F1-score", f"{best_scores['f1_weighted']:.4f}")
    st.success(f"Best model saved to `{MODEL_PATH}`")


def render_random_forest_importance(trained_models):
    """Display Random Forest feature importance table and chart."""
    st.subheader("Random Forest Feature Importance")

    importance_df = get_random_forest_feature_importance(trained_models)
    if importance_df.empty:
        st.info("Random Forest feature importance is unavailable.")
        return

    top_importance = importance_df.head(15)
    left, right = st.columns([1, 2])
    with left:
        st.dataframe(top_importance, hide_index=True, width="stretch")
    with right:
        fig = px.bar(
            top_importance.sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            title="Top Random Forest Feature Importances",
            labels={"importance": "Importance", "feature": "Feature"},
        )
        st.plotly_chart(fig, width="stretch")


def get_explanation_flags(input_data):
    """Create human-readable flags for explanation only, not model prediction."""
    flags = []

    if input_data["study_hours"] < 3:
        flags.append("Low study hours: study_hours is below 3.")
    if input_data["sleep_hours"] < 6:
        flags.append("Low sleep: sleep_hours is below 6.")
    if input_data["attendance"] < 75:
        flags.append("Low attendance: attendance is below 75%.")
    if input_data["gaming_hours"] > 5:
        flags.append("High gaming: gaming_hours is above 5.")
    if input_data["addiction_score"] > 10:
        flags.append("High addiction: addiction_score is above 10.")
    if input_data["stress_level"] in ["Medium", "High"]:
        flags.append("Medium/High stress: stress_level is Medium or High.")

    return flags


def render_prediction_probabilities(best_model, input_data):
    """Display class probabilities when the trained model supports predict_proba."""
    if not hasattr(best_model, "predict_proba"):
        st.info("Prediction probabilities are not available for this model.")
        return

    input_df = pd.DataFrame([input_data], columns=ML_FEATURES)
    probabilities = best_model.predict_proba(input_df)[0]
    probability_df = pd.DataFrame(
        {
            "academic_status": best_model.classes_,
            "probability": probabilities,
        }
    )
    probability_df["probability_percentage"] = (
        probability_df["probability"] * 100
    ).round(2)

    st.subheader("Prediction Class Probabilities")
    st.dataframe(
        probability_df[["academic_status", "probability_percentage"]],
        hide_index=True,
        width="stretch",
    )

    fig = px.bar(
        probability_df,
        x="academic_status",
        y="probability_percentage",
        text="probability_percentage",
        title="Predicted Class Probability",
        labels={
            "academic_status": "Academic Status",
            "probability_percentage": "Probability (%)",
        },
    )
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    st.plotly_chart(fig, width="stretch")


def render_prediction_explanation(input_data):
    """Display entered feature summary and simple explanation flags."""
    st.subheader("Prediction Explanation")
    st.write(
        "The model uses all behavior and lifestyle features together. "
        "It does not decide only from gaming_hours or study_hours."
    )

    summary = pd.DataFrame(
        [
            {"feature": "gaming_hours", "value": input_data["gaming_hours"]},
            {"feature": "study_hours", "value": input_data["study_hours"]},
            {"feature": "sleep_hours", "value": input_data["sleep_hours"]},
            {"feature": "attendance", "value": input_data["attendance"]},
            {"feature": "addiction_score", "value": input_data["addiction_score"]},
            {"feature": "stress_level", "value": input_data["stress_level"]},
        ]
    )
    st.dataframe(summary, hide_index=True, width="stretch")

    flags = get_explanation_flags(input_data)
    st.write(
        "These flags are only for human explanation. The ML model prediction still "
        "comes from the trained model."
    )

    if flags:
        for flag in flags:
            st.warning(flag)
    else:
        st.success("No simple explanation flags were triggered for this input.")


def render_manual_prediction_form(df, best_model):
    """Render form inputs and predict academic status using the best model."""
    st.subheader("Manual Prediction Form")

    with st.form("manual_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input(
                "Age",
                min_value=int(df["age"].min()),
                max_value=int(df["age"].max()),
                value=int(df["age"].median()),
            )
            gender = st.selectbox("Gender", sorted(df["gender"].unique()))
            gaming_hours = st.slider(
                "Gaming Hours",
                min_value=float(df["gaming_hours"].min()),
                max_value=float(df["gaming_hours"].max()),
                value=float(df["gaming_hours"].median()),
                step=0.25,
            )
            study_hours = st.slider(
                "Study Hours",
                min_value=float(df["study_hours"].min()),
                max_value=float(df["study_hours"].max()),
                value=float(df["study_hours"].median()),
                step=0.25,
            )

        with col2:
            sleep_hours = st.slider(
                "Sleep Hours",
                min_value=float(df["sleep_hours"].min()),
                max_value=float(df["sleep_hours"].max()),
                value=float(df["sleep_hours"].median()),
                step=0.25,
            )
            attendance = st.slider("Attendance", min_value=0.0, max_value=100.0, value=80.0)
            gaming_genre = st.selectbox("Gaming Genre", sorted(df["gaming_genre"].unique()))
            social_activity = st.slider(
                "Social Activity",
                min_value=float(df["social_activity"].min()),
                max_value=float(df["social_activity"].max()),
                value=float(df["social_activity"].median()),
                step=0.25,
            )

        with col3:
            device_usage = st.slider(
                "Device Usage",
                min_value=float(df["device_usage"].min()),
                max_value=float(df["device_usage"].max()),
                value=float(df["device_usage"].median()),
                step=0.25,
            )
            reaction_time_ms = st.number_input(
                "Reaction Time MS",
                min_value=float(df["reaction_time_ms"].min()),
                max_value=float(df["reaction_time_ms"].max()),
                value=float(df["reaction_time_ms"].median()),
                step=1.0,
            )
            addiction_score = st.slider(
                "Addiction Score",
                min_value=float(df["addiction_score"].min()),
                max_value=float(df["addiction_score"].max()),
                value=float(df["addiction_score"].median()),
                step=0.25,
            )
            stress_level = st.selectbox("Stress Level", sorted(df["stress_level"].unique()))

        submitted = st.form_submit_button("Predict Academic Status")

    if not submitted:
        return

    input_data = {
        "age": age,
        "gender": gender,
        "gaming_hours": gaming_hours,
        "study_hours": study_hours,
        "sleep_hours": sleep_hours,
        "attendance": attendance,
        "gaming_genre": gaming_genre,
        "social_activity": social_activity,
        "device_usage": device_usage,
        "reaction_time_ms": reaction_time_ms,
        "addiction_score": addiction_score,
        "stress_level": stress_level,
    }

    prediction = predict_academic_status(best_model, input_data)

    st.subheader("Prediction Result")
    st.metric("Predicted Academic Status", prediction)

    render_prediction_probabilities(best_model, input_data)
    render_prediction_explanation(input_data)

    st.subheader("Recommendation Output")
    for recommendation in get_recommendations(prediction):
        st.write(f"- {recommendation}")


def render_ai_insights_agent(
    df,
    comparison_df,
    best_model_name,
    best_scores,
    trained_models,
):
    """Render grounded AI commentary for the calculated ML page results."""
    st.subheader("AI Insights Agent")
    st.write(
        "Generate a plain-language interpretation of the model results using "
        "the calculated values already shown on this page."
    )

    distribution = (
        df[TARGET_COLUMN]
        .value_counts()
        .rename_axis(TARGET_COLUMN)
        .reset_index(name="student_count")
    )
    distribution["percentage"] = (
        distribution["student_count"] / len(df) * 100
    ).round(2)

    feature_importance = get_random_forest_feature_importance(trained_models).head(10)
    context = build_ai_insights_context(
        total_rows=len(df),
        total_columns=len(df.columns),
        ml_feature_count=len(ML_FEATURES),
        target_column=TARGET_COLUMN,
        excluded_columns=EXCLUDED_COLUMNS,
        target_distribution=distribution.to_dict(orient="records"),
        model_comparison=comparison_df.to_dict(orient="records"),
        best_model_name=best_model_name,
        best_scores=best_scores,
        top_feature_importances=feature_importance.to_dict(orient="records"),
    )

    with st.expander("View structured context sent to the AI model"):
        st.json(context)

    if not st.button("Generate AI Insights Summary"):
        return

    fallback_summary = build_fallback_summary(context)

    try:
        api_token = st.secrets["HUGGINGFACE_API_TOKEN"]
    except (KeyError, FileNotFoundError):
        st.warning(
            "Hugging Face token is missing. Showing the local fallback summary."
        )
        st.markdown(fallback_summary)
        st.caption("Source: Local fallback summary because API was unavailable")
        return

    if not isinstance(api_token, str) or not api_token.strip():
        st.warning(
            "Hugging Face token is empty. Showing the local fallback summary."
        )
        st.markdown(fallback_summary)
        st.caption("Source: Local fallback summary because API was unavailable")
        return

    try:
        with st.spinner("Generating AI insights..."):
            summary = generate_ai_insights(api_token, context)
    except AIInsightsTimeoutError:
        st.warning(
            "The Hugging Face request timed out. Showing the local fallback summary."
        )
        st.markdown(fallback_summary)
        st.caption("Source: Local fallback summary because API was unavailable")
        return
    except AIInsightsResponseError:
        st.warning(
            "The AI model returned an unexpected or unavailable response. "
            "Showing the local fallback summary."
        )
        st.markdown(fallback_summary)
        st.caption("Source: Local fallback summary because API was unavailable")
        return
    except AIInsightsAPIError as exc:
        st.warning(f"{exc} Showing the local fallback summary.")
        st.markdown(fallback_summary)
        st.caption("Source: Local fallback summary because API was unavailable")
        return
    except Exception:
        st.warning(
            "An unexpected AI service error occurred. "
            "Showing the local fallback summary."
        )
        st.markdown(fallback_summary)
        st.caption("Source: Local fallback summary because API was unavailable")
        return

    st.subheader("AI-Generated Summary")
    st.markdown(summary)
    st.caption("Source: Hugging Face open-source model API")


st.title("ML Analysis & Prediction")
st.caption("Academic status classification using behavior and lifestyle features")

data = get_ml_dataset()
comparison, models, best_name, scores = get_trained_model_results()
best_model = models[best_name]

render_dataset_overview(data)
st.divider()

render_target_distribution(data)
st.divider()

render_model_results(comparison, best_name, scores)
st.divider()

render_random_forest_importance(models)
st.divider()

render_manual_prediction_form(data, best_model)
st.divider()

render_ai_insights_agent(data, comparison, best_name, scores, models)
