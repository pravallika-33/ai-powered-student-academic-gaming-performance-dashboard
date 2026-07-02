# Gaming Academic Performance Dashboard

## Project Overview

This project is a role-based Streamlit application for analyzing how gaming
behavior and lifestyle factors relate to student academic performance.

It combines:

- Interactive analytics and KPI reporting
- Multi-class machine learning
- Student-level prediction using a saved model
- Role-based access control
- AI-generated explanations of model results
- Manual validation and model error-analysis scripts

The application predicts whether a student is in `Risk`, `Medium`, or `Safe`
academic status using behavior and lifestyle features.

## Problem Statement

Gaming time, study habits, sleep, attendance, device usage, addiction score,
stress, and other lifestyle factors may be associated with academic outcomes.

The goals of this project are to:

1. Explore gaming and lifestyle patterns through an interactive dashboard.
2. Identify academic-risk patterns across student groups.
3. Train and compare classification models that predict `academic_status`.
4. Provide Admin and Analyst users with model evaluation and manual prediction.
5. Allow Student users to retrieve their record and receive a prediction using
   `student_id`.
6. Convert technical ML results into clear language using an AI Insights Agent.

## Features

### RBAC Login

The application uses `st.session_state` for simple role-based access control.

| Role | Access |
|---|---|
| Admin | Analytics Dashboard and ML Analysis & Prediction |
| Analyst | Analytics Dashboard and ML Analysis & Prediction |
| Student | Student Prediction |

The sidebar displays navigation links based on the logged-in role.

### Analytics Dashboard

The Analytics Dashboard provides:

- Sidebar filters for gender, gaming hours, stress level, and academic status
- KPI cards
- Dynamic performance bar charts
- Dynamic risk-percentage charts
- Risk progression line charts
- Gaming-hours versus grades scatter plot
- Risk student summary table
- Filter-aware business insights

### ML Analysis & Prediction

Admin and Analyst users can:

- Review the dataset and target-class distribution
- Compare Logistic Regression, Decision Tree, and Random Forest
- Review accuracy, weighted precision, weighted recall, and weighted F1-score
- View the best model summary
- Inspect Random Forest feature importance
- Enter values through a manual prediction form
- View predicted class probabilities
- Review human-readable explanation flags
- Receive recommendation output

### Student Prediction

Student users can:

- Enter a valid `student_id`
- View grouped profile, behavior, and lifestyle information
- Receive a prediction from the saved best model
- View Risk, Medium, or Safe status
- Receive personalized recommendations

Grades are not displayed or used as prediction input on this page.

### AI Insights Agent

The AI Insights Agent is included at the bottom of the ML Analysis & Prediction
page. It:

- Collects calculated dataset and ML evaluation metrics
- Builds structured context
- Shows the context in an expandable section
- Calls an open-source model through Hugging Face Inference Providers
- Produces a short, presentation-friendly summary
- Clearly labels whether the result came from Hugging Face or the local fallback
- Uses a deterministic local summary when the external API is unavailable

## Dataset

Main application dataset:

```text
data/Gaming_Academic_Performance_with_target.csv
```

Cleaned dataset:

```text
data/Gaming_Academic_Performance_cleaned.csv
```

The target classes are created from `grades`:

| Grade Range | Academic Status |
|---|---|
| Below 60 | Risk |
| 60 to below 85 | Medium |
| 85 and above | Safe |

The final dataset contains 8,000 student records.

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Matplotlib
- Seaborn
- scikit-learn
- joblib
- Requests
- Hugging Face Inference Providers

## Project Structure

```text
Dashboard 2/
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml                  # Local only; do not commit
├── app.py
├── pages/
│   ├── 1_Analytics_Dashboard.py
│   ├── 2_ML_Analysis_Prediction.py
│   └── 3_Student_Prediction.py
├── src/
│   ├── auth.py
│   ├── ai_agent.py
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── utils.py
├── data/
│   ├── Gaming_Academic_Performance_cleaned.csv
│   └── Gaming_Academic_Performance_with_target.csv
├── models/
│   └── best_academic_status_model.joblib
├── test_page1_metrics.py
├── test_student_predictions.py
├── test_model_error_analysis.py
├── test_ai_agent.py
├── requirements.txt
└── README.md
```

## Machine Learning Methodology

### Target

```text
academic_status
```

### Leakage Prevention

`academic_status` was created directly from `grades`. Therefore, the following
columns are excluded from model input:

```text
student_id
grades
academic_status
```

- `grades` would reveal the target and cause data leakage.
- `academic_status` is the target being predicted.
- `student_id` is an identifier, not a behavior or lifestyle feature.

### Input Features

Numeric features:

```text
age, gaming_hours, study_hours, sleep_hours, attendance,
social_activity, device_usage, reaction_time_ms, addiction_score
```

Categorical features:

```text
gender, gaming_genre, stress_level
```

### Preprocessing

- Numeric features are standardized with `StandardScaler`.
- Categorical features are encoded with
  `OneHotEncoder(handle_unknown="ignore")`.
- Preprocessing and classification are combined in a scikit-learn `Pipeline`.

### Models Compared

- Logistic Regression
- Decision Tree
- Random Forest

The models are evaluated using:

- Accuracy
- Weighted precision
- Weighted recall
- Weighted F1-score

The best model is selected using the highest weighted F1-score and saved as:

```text
models/best_academic_status_model.joblib
```

## AI Insights Agent

The AI Insights Agent uses only calculated ML context, including:

- Dataset dimensions
- Number of ML input features
- Target and excluded columns
- Academic-status distribution
- Model comparison metrics
- Best model and scores
- Top feature importances

The structured context is converted into a constrained prompt. The prompt asks
the model to explain:

- The key ML finding
- The best model
- Important features
- Data leakage prevention
- A practical recommendation

The application calls a Hugging Face open-source model API. It does not send the
secret token inside the prompt.

If the API is unavailable, times out, returns an invalid response, or the token
is missing, the application displays a local rule-based summary so the demo
continues to work.

Feature importance represents patterns learned by the model. It does not prove
that a feature causes academic risk.

## Setup Instructions


### 1. Clone the repository

```bash

git clone <your-repository-url>

cd <your-project-folder>
```

### 2. Create A Virtual Environment

```bash
python3 -m venv venv
```

### 3. Activate The Environment

macOS/Linux:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

## How To Run Locally

Using the project virtual environment directly is recommended:

```bash
venv/bin/python -m streamlit run app.py
```

Or activate the environment first:

```bash
source venv/bin/activate
python -m streamlit run app.py
```

Then open the local URL displayed by Streamlit.

## Secrets Setup

Create this local file:

```text
.streamlit/secrets.toml
```

Store the Hugging Face token using this variable name:

```toml
HUGGINGFACE_API_TOKEN = "your_token_here"
```

The application reads it only with:

```python
st.secrets["HUGGINGFACE_API_TOKEN"]
```

Do not commit `.streamlit/secrets.toml` or place a real token in source code,
documentation, screenshots, or logs.

## Testing Scripts

### Validate Analytics KPIs

```bash
python test_page1_metrics.py
```

### Validate Selected Student Predictions

```bash
python test_student_predictions.py
```

### Analyze Model Errors

```bash
python test_model_error_analysis.py
```

This creates:

```text
model_mismatch_analysis.csv
```

### Test The AI Agent Offline

```bash
python test_ai_agent.py
```

This validates prompt construction and fallback behavior without reading secrets
or calling the real API.

## Limitations

- The authentication system uses hardcoded demo users and is not suitable for
  production.
- The dataset is static and does not contain a real time dimension.
- `academic_status` is derived from grades rather than collected independently.
- Model predictions represent learned associations, not guaranteed causation.
- Full-dataset evaluation may be optimistic compared with evaluation on new,
  unseen institutional data.
- The AI summary depends on an external service when the Hugging Face path is
  used.
- Recommendations are general guidance and do not replace academic counseling.

## Future Improvements

- Replace hardcoded users with a secure database-backed authentication system.
- Store hashed passwords and implement stronger RBAC permissions.
- Add cross-validation and hyperparameter tuning.
- Add calibration and class-specific model diagnostics.
- Monitor model drift using new student records.
- Add explainability methods such as SHAP.
- Connect to a database instead of static CSV files.
- Add semester or date data for real trend analysis.
- Add student consent, privacy controls, and audit logging.
- Deploy securely with managed secrets and production monitoring.


