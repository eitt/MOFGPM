# Baseline Fixed Case Summary: Column Guide

This file explains `baseline_fixed_case_summary.csv`.
By default it contains `20` baseline runs:
- `10` runs for `Assistant=1`
- `10` runs for `Assistant=2`

## Structure

Each row is one baseline run (`baseline_run_id`) for a given staffing tuple and replicate index.
GA baseline diagnostics are merged into each row by staffing tuple.

## Column Dictionary

| Column | Meaning | How to interpret |
|---|---|---|
| `Scenario` | FlexSim scenario label associated with the baseline run. | Traceability label for the source scenario row. |
| `Doctor` | Number of doctors in the baseline tuple. | Common baseline setting unless overridden. |
| `Nurse` | Number of nurses in the baseline tuple. | Common baseline setting unless overridden. |
| `Assistant` | Number of assistants in the baseline tuple (`1` or `2` by default). | Baseline variant selector. |
| `Specialist` | Number of specialists in the baseline tuple. | Common baseline setting unless overridden. |
| `Patient` | FlexSim served-patient value in source row. | Throughput context for the baseline run. |
| `baseline_run_id` | Baseline run index (`1..10` by default per assistant variant). | Run-level identifier within each baseline variant. |
| `replicate_column` | Original replication column name used from FlexSim workbook. | Helps map the run back to source replication column. |
| `FS_wait_total` | Baseline run total wait value from source replication. | Run-level wait KPI value. |
| `baseline_variant` | Baseline label (`assistant_1`, `assistant_2`, or `assistant_other`). | Grouping label for comparisons. |
| `ga_replications_n` | GA replication count used for baseline runs. | `0` by design for GA baseline in this table. |
| `ga_ci95` | GA confidence-interval width for baseline runs. | `0.0` by design when no GA replications are used. |
| `sample_id` | GA sample identifier for the matched staffing tuple. | Trace to GA schedules and triage rows. |
| `target_patients_24h` | Target patients required within 24h. | Daily target benchmark. |
| `generated_patients_ga` | GA generated patients in baseline sample. | Demand generation context. |
| `served_24h_ga` | GA patients served within 24h horizon. | Main GA baseline throughput KPI. |
| `served_24h_rate` | Hourly GA served rate (`served_24h_ga / 24`). | Compare pace across variants. |
| `served_share_generated_pct` | Percent of generated GA patients served in 24h. | Service efficiency indicator. |
| `served_share_target_pct` | Percent of target served in 24h. | Target attainment indicator. |
| `backlog_vs_generated` | GA backlog against generated patients. | Carry-over load from generated cohort. |
| `backlog_vs_target` | GA backlog against target. | Day-end target gap indicator. |
| `completion_time_max_min` | Latest GA patient completion time (minutes). | > `horizon_min` means spill beyond 24h. |
| `horizon_min` | Horizon used for 24h classification. | Usually `1440`. |
| `schedule_feasible_24h_target` | GA feasibility flag for 24h target. | `True` means baseline meets target in horizon. |
| `missing_first_wait_rows` | Count of GA rows with missing first wait filled as `0`. | Data-quality signal. |
| `ga_wait_all_finished` | GA wait KPI for all finished patients. | Total burden indicator. |
| `ga_lambda_fixed_row` | GA fuzzy objective score (`lambda`). | Higher generally indicates better goal balance. |
| `ga_feasible_fixed_row` | GA feasibility flag from `results.csv`. | Model-feasibility context. |
| `baseline_source_subdir` | Source GA subfolder used for baseline extraction. | Provenance field for reproducibility. |

## Quick Interpretation Workflow

1. Filter rows by `Assistant` and compare run-level spread for `1` vs `2`.
2. Use `FS_wait_total` for run-level baseline dispersion.
3. Use GA columns (`served_24h_ga`, `schedule_feasible_24h_target`, `ga_lambda_fixed_row`) for baseline feasibility context.
4. Use `ga_replications_n=0` and `ga_ci95=0.0` to indicate no GA replication CI in this baseline table.
