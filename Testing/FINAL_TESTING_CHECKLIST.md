# Final Testing Checklist

Use this checklist to validate the Streamlit RBAC + Analytics + ML Prediction application.

Blank fields are still pending manual testing. Completed script-based checks are
marked below.

| Test Case ID | Test Scenario | User Role | Steps | Expected Result | Actual Result | Status | Notes |
|---|---|---|---|---|---|---|---|
| TC-001 | Login with valid Admin credentials | Admin | Open app, enter `admin` / `admin123`, click Login | Admin logs in successfully and sees Admin navigation |  |  |  |
| TC-002 | Login with valid Analyst credentials | Analyst | Open app, enter `analyst` / `analyst123`, click Login | Analyst logs in successfully and sees Analyst navigation |  |  |  |
| TC-003 | Login with valid Student credentials | Student | Open app, enter `student` / `student123`, click Login | Student logs in successfully and sees Student Prediction navigation only |  |  |  |
| TC-004 | Invalid login | Any | Enter invalid username/password and click Login | App shows invalid username or password message |  |  |  |
| TC-005 | Logout | Admin/Analyst/Student | Log in, click Logout in sidebar | Session clears and login screen is shown |  |  |  |
| TC-006 | Admin role access | Admin | Log in as Admin and check available pages | Admin can access Analytics Dashboard and ML Analysis & Prediction |  |  |  |
| TC-007 | Analyst role access | Analyst | Log in as Analyst and check available pages | Analyst can access Analytics Dashboard and ML Analysis & Prediction |  |  |  |
| TC-008 | Student role access | Student | Log in as Student and check available pages | Student can access Student Prediction only |  |  |  |
| TC-009 | Analytics Dashboard page loading | Admin/Analyst | Log in, open Analytics Dashboard | Analytics Dashboard loads without errors |  |  |  |
| TC-010 | KPI cards display | Admin/Analyst | Open Analytics Dashboard and view top section | KPI cards display total students, grade, risk, behavior, and attendance metrics |  |  |  |
| TC-011 | KPI validation using `test_page1_metrics.py` | Admin/Analyst | Run `python3 test_page1_metrics.py` and compare values with dashboard | Script values match dashboard KPIs before filters | 8,000 students; average grade 66.08; 3,131 Risk students; Risk percentage 39.14%. Risk increases from 10.08% at 0-2 gaming hours to 67.99% at 6-8 hours. | Passed | Direct Pandas calculations match the dashboard values. |
| TC-012 | Chart rendering on Analytics Dashboard | Admin/Analyst | Open Analytics Dashboard and scroll through charts | Dynamic performance chart, risk chart, line chart, and scatter plot render |  |  |  |
| TC-013 | Sidebar filters on Analytics Dashboard | Admin/Analyst | Change Gender, Gaming Hours, Stress Level, and Academic Status filters | KPIs, charts, table, and insights update based on filters |  |  |  |
| TC-014 | ML Analysis & Prediction page loading | Admin/Analyst | Open ML Analysis & Prediction page | Page loads without errors |  |  |  |
| TC-015 | Dataset overview on ML page | Admin/Analyst | View Dataset Overview section | Rows, columns, ML feature count, target, excluded columns, numeric features, and categorical features display |  |  |  |
| TC-016 | Existing academic status distribution | Admin/Analyst | View Existing Academic Status Distribution section | Distribution table and chart show Risk, Medium, and Safe classes |  |  |  |
| TC-017 | Model comparison table | Admin/Analyst | View Model Comparison Table | Logistic Regression, Decision Tree, and Random Forest metrics display |  |  |  |
| TC-018 | Best model summary | Admin/Analyst | View Best Model Summary | Best model name and accuracy, precision, recall, F1-score display |  |  |  |
| TC-019 | Saved model file check | Admin/Analyst | Check `models/best_academic_status_model.joblib` exists | Saved model file is present |  |  |  |
| TC-020 | Random Forest feature importance | Admin/Analyst | View Random Forest Feature Importance section | Feature importance table and bar chart display |  |  |  |
| TC-021 | Manual prediction form | Admin/Analyst | Fill all form inputs and click Predict Academic Status | Form accepts inputs and returns predicted class |  |  |  |
| TC-022 | Prediction class probabilities | Admin/Analyst | Submit manual prediction form | Probability table/chart displays if model supports `predict_proba` |  |  |  |
| TC-023 | Prediction explanation flags | Admin/Analyst | Submit form with low study, low sleep, low attendance, high gaming, high addiction, or Medium/High stress | Explanation flags display and clarify they are only for human explanation |  |  |  |
| TC-024 | Recommendation output for Risk | Admin/Analyst | Submit input likely to predict Risk | Risk prediction displays with Risk recommendations |  |  |  |
| TC-025 | Recommendation output for Medium | Admin/Analyst | Submit input likely to predict Medium | Medium prediction displays with Medium recommendations |  |  |  |
| TC-026 | Recommendation output for Safe | Admin/Analyst | Submit input likely to predict Safe | Safe prediction displays with Safe recommendations |  |  |  |
| TC-027 | Student Prediction page with valid `student_id` | Student | Log in as Student, enter valid ID such as `1`, click Find My Prediction | Matching student record, prediction, and recommendations display | Validation script completed: ID 1 actual Safe, predicted Medium; ID 3300 actual Risk, predicted Risk. | Passed | ID 1 is a valid expected mismatch because grades are excluded from model input. ID 3300 is a match. |
| TC-028 | Student Prediction page with invalid `student_id` | Student | Enter an ID not found in the dataset if allowed by UI or test with a dataset gap | App shows `Student ID not found. Please check the ID and try again.` |  |  |  |
| TC-029 | Student record display | Student | Enter a valid student ID | Student Profile, Behavior Summary, and Gaming and Lifestyle sections display; grades do not display |  |  |  |
| TC-030 | Student prediction output | Student | Enter a valid student ID | Prediction displays clearly using Risk error, Medium warning, or Safe success styling |  |  |  |
| TC-031 | Student recommendation output | Student | Enter a valid student ID | Recommendations display below prediction |  |  |  |
| TC-032 | Data leakage check | Admin/Analyst | Review ML feature list and scripts | `grades` is not used for ML input, `academic_status` is only target, `student_id` is only lookup/identifier |  |  |  |
| TC-033 | Model validation using `test_model_error_analysis.py` | Admin/Analyst | Run `python3 test_model_error_analysis.py` | Script prints total records, correct/incorrect predictions, accuracy, confusion matrix, classification report, and mismatches | 8,000 records; 6,874 correct; 1,126 incorrect; accuracy 0.8592. Risk precision and recall are approximately 0.91. | Passed | Results confirm strong Risk-class performance while preserving realistic model errors. |
| TC-034 | Model mismatch analysis file check | Admin/Analyst | Run model error analysis script and check project folder | `model_mismatch_analysis.csv` is created | `model_mismatch_analysis.csv` created successfully. | Passed | Contains records where predicted and actual status differ. |
| TC-035 | AI agent prompt and fallback validation | Admin/Analyst | Run `python3 test_ai_agent.py` | Prompt contains required ML context, exposes no token, and fallback summary is generated | 9/9 checks passed. Prompt contains key ML context, no token exposure was detected, and fallback summary returned non-empty text. | Passed | Test runs offline and does not call the Hugging Face API. |

## Latest Test Results

### Analytics KPI Validation

`test_page1_metrics.py` passed:

- Total students: `8,000`
- Average grade: `66.08`
- Risk count: `3,131`
- Risk percentage: `39.14%`
- Risk rises from `10.08%` in the `0-2` gaming-hours group to `67.99%`
  in the `6-8` gaming-hours group.

### Student Prediction Validation

`test_student_predictions.py` passed:

- Student ID `1`: actual `Safe`, predicted `Medium` - not a match.
- Student ID `3300`: actual `Risk`, predicted `Risk` - match.
- Some mismatches are expected because the model intentionally excludes
  `grades`. It predicts from behavior and lifestyle features instead of using
  the value that created the target.

### Model Error Analysis

`test_model_error_analysis.py` passed:

- Total records: `8,000`
- Correct predictions: `6,874`
- Incorrect predictions: `1,126`
- Overall accuracy: `0.8592`
- Risk precision and recall: approximately `0.91`

### AI Agent Validation

`test_ai_agent.py` passed all `9/9` checks:

- Prompt includes the required ML context.
- No Hugging Face token is exposed.
- Local fallback summary works without calling the real API.

## Suggested Test Commands

Start the app:

```bash
cd "/Users/mohanapravallikapakala/Documents/Dashboard 2"
streamlit run app.py
```

Run KPI validation:

```bash
python3 test_page1_metrics.py
```

Run student prediction validation:

```bash
python3 test_student_predictions.py
```

Run model error analysis:

```bash
python3 test_model_error_analysis.py
```

Run the offline AI agent validation:

```bash
python3 test_ai_agent.py
```
