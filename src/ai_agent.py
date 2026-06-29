"""Hugging Face powered AI insights for ML dashboard results."""

import json
from typing import Any

import requests


HUGGINGFACE_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b:fastest"
DEFAULT_TIMEOUT_SECONDS = 45


class AIInsightsError(Exception):
    """Base error for AI insights generation."""


class AIInsightsTimeoutError(AIInsightsError):
    """Raised when the Hugging Face request times out."""


class AIInsightsAPIError(AIInsightsError):
    """Raised when the Hugging Face API returns an error."""


class AIInsightsResponseError(AIInsightsError):
    """Raised when the API response does not contain usable generated text."""


def build_ai_insights_context(
    total_rows: int,
    total_columns: int,
    ml_feature_count: int,
    target_column: str,
    excluded_columns: list[str],
    target_distribution: list[dict[str, Any]],
    model_comparison: list[dict[str, Any]],
    best_model_name: str,
    best_scores: dict[str, float],
    top_feature_importances: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a JSON-serializable context using calculated dashboard values only."""
    return {
        "dataset": {
            "total_rows": total_rows,
            "total_columns": total_columns,
            "ml_input_feature_count": ml_feature_count,
            "target_column": target_column,
            "excluded_columns": excluded_columns,
        },
        "academic_status_distribution": target_distribution,
        "model_comparison": model_comparison,
        "best_model": {
            "name": best_model_name,
            "accuracy": best_scores["accuracy"],
            "precision_weighted": best_scores["precision_weighted"],
            "recall_weighted": best_scores["recall_weighted"],
            "f1_weighted": best_scores["f1_weighted"],
        },
        "top_feature_importances": top_feature_importances,
    }


def build_ai_insights_prompt(context: dict[str, Any]) -> str:
    """Create a grounded prompt that asks the LLM to explain real ML results."""
    context_json = json.dumps(context, indent=2)

    return f"""
You are a senior data analyst explaining a multi-class academic risk model.
Use only the structured context below. Do not invent metrics, features, causes,
or conclusions that are not supported by the context.

STRUCTURED CONTEXT:
{context_json}

Write a short, presentation-friendly summary with exactly these headings:
1. Key Finding
2. Best Model
3. Important Features
4. Data Leakage Prevention
5. Final Recommendation

Requirements:
- Keep the complete response under 250 words.
- Use simple language suitable for non-technical users.
- Use only values provided in the structured context.
- Do not invent or estimate numbers.
- Identify the best-performing model from the supplied values.
- Briefly explain what the supplied performance metrics mean.
- State that feature importance shows learned patterns, not guaranteed causation.
- State that grades and academic_status were excluded to avoid data leakage.
- Explain that grades created academic_status and academic_status is the target.
- Mention that student_id is an identifier, not a predictive behavior feature.
- End with a practical recommendation for using the model responsibly.
""".strip()


def generate_ai_insights(
    api_token: str,
    context: dict[str, Any],
    model: str = DEFAULT_MODEL,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Call Hugging Face Inference Providers and return generated summary text."""
    prompt = build_ai_insights_prompt(context)

    try:
        response = requests.post(
            HUGGINGFACE_CHAT_URL,
            headers={
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Answer only from supplied ML metrics and context. "
                            "Do not expose secrets or fabricate results."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "temperature": 0.2,
                "max_tokens": 700,
            },
            timeout=timeout_seconds,
        )
    except requests.Timeout as exc:
        raise AIInsightsTimeoutError(
            "The Hugging Face request timed out."
        ) from exc
    except requests.RequestException as exc:
        raise AIInsightsAPIError(
            "The Hugging Face API request failed."
        ) from exc

    if response.status_code != 200:
        detail = _extract_api_error(response)
        raise AIInsightsAPIError(
            f"Hugging Face returned status {response.status_code}: {detail}"
        )

    try:
        payload = response.json()
        summary = payload["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise AIInsightsResponseError(
            "Hugging Face returned an unexpected response format."
        ) from exc

    if not isinstance(summary, str) or not summary.strip():
        raise AIInsightsResponseError(
            "Hugging Face returned an empty or unavailable model response."
        )

    return summary.strip()


def build_fallback_summary(context: dict[str, Any]) -> str:
    """Create a deterministic local summary when the AI API is unavailable."""
    best = context["best_model"]
    distribution = context["academic_status_distribution"]
    importances = context["top_feature_importances"]

    distribution_text = ", ".join(
        f"{row['academic_status']}: {row['student_count']} "
        f"({row['percentage']:.2f}%)"
        for row in distribution
    )
    feature_text = (
        ", ".join(row["feature"] for row in importances[:5])
        if importances
        else "No feature importance values were available"
    )

    return f"""
### Overall ML Result
The model comparison selected **{best['name']}** using the highest weighted
F1-score. Its accuracy is **{best['accuracy']:.4f}** and its weighted F1-score
is **{best['f1_weighted']:.4f}**. The dataset contains
**{context['dataset']['total_rows']:,}** records. The target distribution is:
{distribution_text}.

### Metric Meaning
Accuracy is the share of all predictions that were correct. Weighted precision
describes how often class predictions were correct, weighted by class size.
Weighted recall describes how many actual class records were found. Weighted
F1-score balances precision and recall while accounting for class sizes.

### Important Features
The leading Random Forest importance signals are: **{feature_text}**. These are
model associations and should not be interpreted as proof that a feature causes
academic risk.

### Data Leakage Prevention
`grades` was excluded because `academic_status` was created from grades.
`academic_status` is the target output, and `student_id` is only an identifier.
Using any of them as model inputs would produce misleading evaluation results.

### Academic Risk Meaning
The model estimates Risk, Medium, or Safe status from behavior and lifestyle
patterns. It can support early review, but it should complement rather than
replace academic or student-support judgment.
""".strip()


def _extract_api_error(response: requests.Response) -> str:
    """Extract a short API error without exposing request credentials."""
    try:
        payload = response.json()
    except ValueError:
        return "The provider returned a non-JSON error response."

    if isinstance(payload, dict):
        for key in ("error", "message", "detail"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:300]

    return "The selected model may be loading, unavailable, or inaccessible."
