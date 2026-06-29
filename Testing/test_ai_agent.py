"""Manual tests for AI Insights prompt building and fallback behavior.

This script does not read secrets and does not call the Hugging Face API.
Run it with:

    python test_ai_agent.py
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from ai_agent import (  # noqa: E402
    build_ai_insights_context,
    build_ai_insights_prompt,
    build_fallback_summary,
)


def report_test(test_name: str, passed: bool, details: str = "") -> bool:
    """Print a clear PASS or FAIL result."""
    status = "PASS" if passed else "FAIL"
    message = f"[{status}] {test_name}"
    if details:
        message += f" - {details}"
    print(message)
    return passed


def build_sample_context() -> dict:
    """Build sample ML metrics without reading any secret or calling an API."""
    return build_ai_insights_context(
        total_rows=8000,
        total_columns=15,
        ml_feature_count=12,
        target_column="academic_status",
        excluded_columns=["student_id", "grades", "academic_status"],
        target_distribution=[
            {
                "academic_status": "Risk",
                "student_count": 3131,
                "percentage": 39.14,
            },
            {
                "academic_status": "Medium",
                "student_count": 2986,
                "percentage": 37.33,
            },
            {
                "academic_status": "Safe",
                "student_count": 1883,
                "percentage": 23.54,
            },
        ],
        model_comparison=[
            {
                "model_name": "Logistic Regression",
                "accuracy": 0.86,
                "precision_weighted": 0.8609,
                "recall_weighted": 0.86,
                "f1_weighted": 0.8603,
            }
        ],
        best_model_name="Logistic Regression",
        best_scores={
            "accuracy": 0.86,
            "precision_weighted": 0.8609,
            "recall_weighted": 0.86,
            "f1_weighted": 0.8603,
        },
        top_feature_importances=[
            {"feature": "study_hours", "importance": 0.3583},
            {"feature": "gaming_hours", "importance": 0.1158},
            {"feature": "sleep_hours", "importance": 0.0946},
        ],
    )


def main() -> None:
    """Run prompt and fallback tests without network access."""
    print("=" * 72)
    print("AI Insights Agent Manual Test")
    print("No secrets are read and no external API is called.")
    print("=" * 72)

    results = []
    context = build_sample_context()
    prompt = build_ai_insights_prompt(context)

    results.append(
        report_test(
            "Prompt created successfully",
            isinstance(prompt, str) and bool(prompt.strip()),
        )
    )

    required_text = [
        "Logistic Regression",
        "grades",
        "academic_status",
        "data leakage",
        "feature importance",
    ]
    for text in required_text:
        results.append(
            report_test(
                f"Prompt contains '{text}'",
                text.lower() in prompt.lower(),
            )
        )

    forbidden_text = ["hf_", "HUGGINGFACE_API_TOKEN"]
    for text in forbidden_text:
        results.append(
            report_test(
                f"Prompt does not contain '{text}'",
                text not in prompt,
            )
        )

    fallback_summary = build_fallback_summary(context)
    results.append(
        report_test(
            "Fallback summary returns non-empty text",
            isinstance(fallback_summary, str) and bool(fallback_summary.strip()),
        )
    )

    print("-" * 72)
    passed_count = sum(results)
    total_count = len(results)
    print(f"Result: {passed_count}/{total_count} tests passed")

    if passed_count != total_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
