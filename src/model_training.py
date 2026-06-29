"""Model training utilities for academic status classification."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "Gaming_Academic_Performance_with_target.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_academic_status_model.joblib"

TARGET_COLUMN = "academic_status"
EXCLUDED_COLUMNS = ["student_id", "grades", TARGET_COLUMN]

NUMERIC_FEATURES = [
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

CATEGORICAL_FEATURES = ["gender", "gaming_genre", "stress_level"]

ML_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_ml_dataset(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the target dataset used for Admin/Analyst ML analysis."""
    return pd.read_csv(data_path)


def prepare_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create X and y using only approved non-leakage ML columns."""
    X = df[ML_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Scale numeric features and one-hot encode categorical features."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_model_pipeline(model) -> Pipeline:
    """Attach preprocessing and a classifier in one reusable sklearn pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )


def get_candidate_models() -> dict[str, object]:
    """Return the three classifiers required for comparison."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }


def train_and_evaluate_models(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Pipeline], str, dict[str, float]]:
    """Train models, compare weighted metrics, save the best model, and return results."""
    X, y = prepare_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    trained_models = {}
    results = []

    for model_name, model in get_candidate_models().items():
        pipeline = build_model_pipeline(model)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        trained_models[model_name] = pipeline
        results.append(
            {
                "model_name": model_name,
                "accuracy": round(accuracy_score(y_test, predictions), 4),
                "precision_weighted": round(
                    precision_score(y_test, predictions, average="weighted"), 4
                ),
                "recall_weighted": round(
                    recall_score(y_test, predictions, average="weighted"), 4
                ),
                "f1_weighted": round(
                    f1_score(y_test, predictions, average="weighted"), 4
                ),
            }
        )

    comparison_df = pd.DataFrame(results).sort_values(
        by="f1_weighted", ascending=False
    )
    best_model_name = comparison_df.iloc[0]["model_name"]
    best_scores = comparison_df.iloc[0].drop(labels=["model_name"]).to_dict()

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(trained_models[best_model_name], MODEL_PATH)

    return comparison_df, trained_models, best_model_name, best_scores


def get_random_forest_feature_importance(
    trained_models: dict[str, Pipeline],
) -> pd.DataFrame:
    """Return Random Forest feature importances with encoded feature names."""
    random_forest_pipeline = trained_models.get("Random Forest")

    if random_forest_pipeline is None:
        return pd.DataFrame(columns=["feature", "importance"])

    preprocessor = random_forest_pipeline.named_steps["preprocessor"]
    model = random_forest_pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    )
    importance_df["importance"] = importance_df["importance"].round(4)

    return importance_df.sort_values(by="importance", ascending=False).reset_index(
        drop=True
    )


def predict_academic_status(model_pipeline: Pipeline, input_data: dict) -> str:
    """Predict academic_status from a single-row input dictionary."""
    input_df = pd.DataFrame([input_data], columns=ML_FEATURES)
    return model_pipeline.predict(input_df)[0]


def get_recommendations(prediction: str) -> list[str]:
    """Return recommendation text based on the predicted academic status."""
    if prediction == "Risk":
        return [
            "Reduce gaming hours and set clear daily gaming limits.",
            "Increase study hours with a consistent study schedule.",
            "Improve sleep routine and aim for more regular rest.",
            "Monitor attendance closely and avoid unnecessary absences.",
            "Seek academic support if performance concerns continue.",
        ]

    if prediction == "Medium":
        return [
            "Maintain a balanced gaming schedule.",
            "Improve study consistency across the week.",
            "Monitor stress and sleep quality.",
            "Avoid increasing gaming hours beyond the current level.",
        ]

    return [
        "Continue current habits.",
        "Maintain balance between gaming and academics.",
        "Keep consistent study and sleep routines.",
    ]
