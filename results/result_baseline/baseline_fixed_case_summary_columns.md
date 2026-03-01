# Baseline Fixed Case Summary: Column Guide

This file explains the columns in `baseline_fixed_case_summary.csv` and how to interpret them for the baseline scenario.

## Column Dictionary

| Column | Meaning | How to interpret |
|---|---|---|
| `sample_id` | GA sample identifier used for the selected baseline row. | Use it to trace the exact schedule and triage records that produced this baseline summary. |
| `Doctor` | Number of doctors in the selected baseline staffing tuple. | Higher values usually increase service capacity at doctor-required tasks. |
| `Nurse` | Number of nurses in the selected baseline staffing tuple. | Higher values usually reduce nurse-related queues and waiting. |
| `Assistant` | Number of assistants in the selected baseline staffing tuple. | Higher values usually increase throughput in assistant-required steps. |
| `Specialist` | Number of specialists in the selected baseline staffing tuple. | Higher values usually reduce specialist bottlenecks. |
| `target_patients_24h` | Required patients to be completed within 24 hours (daily target). | This is the baseline demand target to compare against served counts. |
| `generated_patients_ga` | Number of unique patients generated/observed in the GA baseline sample. | If this is below target, demand generation itself may limit achievable completions. |
| `served_24h_ga` | Number of patients completed within the 24h horizon (`last_end <= horizon_min`). | Primary achieved throughput under baseline conditions. |
| `served_24h_rate` | Hourly completion rate in 24h horizon. Formula: `served_24h_ga / 24`. | Compare across scenarios as “patients completed per hour”. |
| `served_share_generated_pct` | Percent of generated patients completed within 24h. Formula: `100 * served_24h_ga / generated_patients_ga`. | Shows service efficiency relative to generated cohort. Higher is better. |
| `served_share_target_pct` | Percent of daily target completed within 24h. Formula: `100 * served_24h_ga / target_patients_24h`. | Shows target attainment. `100%` means full target met. |
| `backlog_vs_generated` | Generated patients not completed within 24h. Formula: `generated_patients_ga - served_24h_ga`. | Immediate carry-over workload from generated cohort. Lower is better. |
| `backlog_vs_target` | Shortfall to daily target within 24h. Formula: `max(0, target_patients_24h - served_24h_ga)`. | Operational deficit vs planned daily capacity. `0` means target met. |
| `completion_time_max_min` | Latest patient completion time (minutes) in this baseline sample. | If this exceeds `horizon_min`, some patients complete after 24h. |
| `horizon_min` | Time horizon in minutes used to define 24h service window (normally `1440`). | Baseline 24h cutoff for served/unserved classification. |
| `schedule_feasible_24h_target` | Baseline feasibility flag for daily target. `True` if target served within horizon. | `True` means no target shortfall and no beyond-horizon completion constraint for feasibility rule. |
| `missing_first_wait_rows` | Count of patients with missing `first_wait` in triage data, filled as `0` in computation. | Data quality indicator. Non-zero can affect waiting-time precision. |
| `ga_wait_all_finished` | Total GA waiting metric for all finished patients in selected baseline row. | Use as total burden indicator; compare only with same KPI definition. |
| `ga_lambda_fixed_row` | Fuzzy objective score (`lambda`) for selected baseline row. | Higher generally indicates better balance against fuzzy goals. |
| `ga_feasible_fixed_row` | GA feasibility flag from baseline row in `results.csv`. | Feasibility under GA model constraints, separate from 24h target feasibility flag. |
| `baseline_source_subdir` | Run subfolder used to extract baseline data (for example `fixed`). | Provenance field for reproducibility and audit of baseline source. |

## Quick Interpretation Workflow

1. Check `schedule_feasible_24h_target`.
2. If `False`, inspect `served_share_target_pct`, `backlog_vs_target`, and `completion_time_max_min` vs `horizon_min`.
3. Use `served_24h_rate` to compare operational pace with other baselines.
4. Use `ga_lambda_fixed_row` and `ga_feasible_fixed_row` for optimization-quality context.
5. Check `missing_first_wait_rows` before drawing strong conclusions from waiting metrics.

## Practical Reading Rules

- `served_share_target_pct < 100` means daily target was not fully achieved in 24h.
- `completion_time_max_min > horizon_min` means at least one patient finished after the 24h window.
- `backlog_vs_generated > 0` means carry-over exists even for generated demand.
- `backlog_vs_target > 0` means target capacity gap remains at day-end.
