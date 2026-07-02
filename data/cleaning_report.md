# Data Cleaning Report

## Cleaning Notes

- Loaded raw dataset with 8,000 rows and 14 columns.
- Standardized column names by trimming spaces and converting names to lowercase.
- Checked full-row duplicates and removed 0 duplicate rows.
- Checked duplicate student IDs and removed 0 duplicate ID records.
- Validated numeric columns by converting them to numeric data types; conversion issues found: 0.
- Trimmed extra spaces from categorical columns.
- Checked missing values. Missing values before imputation: 0; missing values after imputation: 0.
- Corrected addiction score validity by setting 107 negative addiction_score values to 0.
- Corrected grade validity by clipping 0 grades below 0 and 134 grades above 100 to the valid 0-100 range.
- Checked business ranges for age, time-use fields, attendance, and reaction time; remaining invalid range values found: 0.
- Checked statistical outliers using the IQR method; 0 outlier values were detected, so no IQR treatment was applied.
- Saved cleaned dataset as Gaming_Academic_Performance_cleaned.csv.

## Final Dataset Shape

- Rows: 8,000
- Columns: 14

## Final Data Types

| column | data_type |
| --- | --- |
| student_id | int64 |
| age | int64 |
| gender | object |
| gaming_hours | float64 |
| study_hours | float64 |
| sleep_hours | float64 |
| attendance | float64 |
| gaming_genre | object |
| social_activity | float64 |
| device_usage | float64 |
| reaction_time_ms | float64 |
| addiction_score | float64 |
| stress_level | object |
| grades | float64 |

## Final Missing Values

| column | missing_values |
| --- | --- |
| student_id | 0 |
| age | 0 |
| gender | 0 |
| gaming_hours | 0 |
| study_hours | 0 |
| sleep_hours | 0 |
| attendance | 0 |
| gaming_genre | 0 |
| social_activity | 0 |
| device_usage | 0 |
| reaction_time_ms | 0 |
| addiction_score | 0 |
| stress_level | 0 |
| grades | 0 |

## IQR Outlier Summary

| column | lower_bound | upper_bound | outlier_count |
| --- | --- | --- | --- |
| student_id | -3998.5 | 11999.5 | 0 |
| age | 12.0 | 28.0 | 0 |
| gaming_hours | -3.7649999999999997 | 11.954999999999998 | 0 |
| study_hours | -3.3899999999999997 | 14.29 | 0 |
| sleep_hours | 1.505 | 11.465 | 0 |
| attendance | 39.30000000000001 | 120.57999999999998 | 0 |
| social_activity | -2.4212499999999992 | 7.468749999999999 | 0 |
| device_usage | -0.5000000000000009 | 15.66 | 0 |
| reaction_time_ms | 175.865 | 365.985 | 0 |
| addiction_score | -5.99 | 25.77 | 0 |
| grades | -1.2887285694924486 | 135.1607945990998 | 0 |
