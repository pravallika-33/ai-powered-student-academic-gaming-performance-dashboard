# Exploratory Data Analysis Report

## Dataset Size

| rows | columns |
| --- | --- |
| 8000 | 15 |

## Dimensions

Dimensions are categorical or identifier fields used for filtering, grouping, and segmentation.

| dimension | unique_values |
| --- | --- |
| student_id | 8000 |
| gender | 3 |
| gaming_genre | 3 |
| stress_level | 3 |
| academic_status | 3 |

## Measures

Measures are numerical fields used for aggregation, comparison, and trend analysis.

| measure | data_type |
| --- | --- |
| age | int64 |
| gaming_hours | float64 |
| study_hours | float64 |
| sleep_hours | float64 |
| attendance | float64 |
| social_activity | float64 |
| device_usage | float64 |
| reaction_time_ms | float64 |
| addiction_score | float64 |
| grades | float64 |

## Numerical Input Feature Averages

| numerical_input_feature | average |
| --- | --- |
| age | 19.98 |
| gaming_hours | 4.09 |
| study_hours | 5.46 |
| sleep_hours | 6.49 |
| attendance | 79.89 |
| social_activity | 2.51 |
| device_usage | 7.59 |
| reaction_time_ms | 271.11 |
| addiction_score | 9.92 |

## Summary Statistics

| statistic | age | gaming_hours | study_hours | sleep_hours | attendance | social_activity | device_usage | reaction_time_ms | addiction_score | grades |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| count | 8000.0 | 8000.0 | 8000.0 | 8000.0 | 8000.0 | 8000.0 | 8000.0 | 8000.0 | 8000.0 | 8000.0 |
| mean | 19.98 | 4.09 | 5.46 | 6.49 | 79.89 | 2.51 | 7.59 | 271.11 | 9.92 | 66.08 |
| std | 2.59 | 2.31 | 2.58 | 1.44 | 11.58 | 1.44 | 2.71 | 29.44 | 5.01 | 22.25 |
| min | 16.0 | 0.0 | 1.0 | 4.0 | 60.0 | 0.0 | 1.1 | 183.26 | 0.0 | 0.0 |
| 25% | 18.0 | 2.13 | 3.24 | 5.24 | 69.78 | 1.29 | 5.56 | 247.16 | 5.92 | 49.88 |
| 50% | 20.0 | 4.13 | 5.46 | 6.5 | 79.69 | 2.5 | 7.61 | 270.48 | 10.0 | 67.07 |
| 75% | 22.0 | 6.06 | 7.66 | 7.73 | 90.1 | 3.76 | 9.6 | 294.69 | 13.86 | 83.99 |
| max | 24.0 | 8.0 | 10.0 | 9.0 | 100.0 | 5.0 | 13.95 | 347.87 | 23.16 | 100.0 |

## Target Column Decision

Target column: `academic_status`

Target classes:

- Risk: `grades < 60`
- Medium: `60 <= grades < 85`
- Safe: `grades >= 85`

## Target Distribution

| academic_status | student_count | percentage |
| --- | --- | --- |
| Risk | 3131 | 39.14 |
| Medium | 2986 | 37.33 |
| Safe | 1883 | 23.54 |
