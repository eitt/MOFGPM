# MOFGPM

Multi Objective Fuzzy Goal Programming (MOFGPM) workflow for ED resource planning, plus external validation against FlexSim.

## Requirements

### Python

- Python 3.10+ (recommended: 3.11 or 3.12)
- Install packages:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### System tools

- `pdflatex` on PATH (MiKTeX or TeX Live) for PDF generation from LaTeX.
- Microsoft Word is not required for `.docx` creation, but open Word files can block overwrite.

### Quick validation

```powershell
python -c "import numpy, pandas, matplotlib, scipy, docx; print('python deps ok')"
pdflatex --version
```

## Main Scripts

- `main.py`: MOFGPM + GA optimization run.
- `run_all_pipeline.py`: end-to-end runner for fixed, sensitivity, combinatorics, methodology, and merged history.
- `pipeline_flexsim_validation.py`: GA vs FlexSim 24h served-only validation pipeline.
- `results_report.py`: Results Word draft generator.
- `methdology.py`: Methodology document generator.
- `plotting_utils.py`: shared plotting utilities.

## Folder Layout

```text
outputs/
  run_<timestamp>/
    fixed/
    sensitivity/
    combinatorics/
    methodology/
  analysis/
    sensitivity_history/

results/
  run_<timestamp>/
    combinatorics/
      results.csv
      schedule_by_sample.csv
      triage_waits_by_sample.csv
      settings.json
  Flexsim/
    10_replicas.xlsx
  result_baseline/
    baseline_fixed_case_summary.csv
    figB1_baseline_completion_profile.png
  validation_flexsim_24h/
    table1_cohort_capacity_reconciliation.csv
    table2_kpi_comparison_served_only.csv
    table3_statistical_check.csv
    validation_matched_dataset_24h.csv
    ga_patient_level_waits_24h.csv
    baseline_fixed_case_summary.csv
    sensitivity_top_staffing_by_served24h.csv
    figB1_baseline_completion_profile.png
    figS1_ga_served24h_by_staff_grid.png
    figS2_flexsim_patient_by_staff_grid.png
    figS3_gap_flexsim_minus_ga_by_staff_grid.png
    fig1_aggregated_ratio_view.png
    fig2_aggregated_kpi_summary.png
    fig3_per_served_patient_kpi_bar.png
    fig4_served_rate_per_hour_bar.png
    appendix_figA1_scenario_ci_vs_ga_scaled.png
    alt_text_figures.txt
    Results_Validation_FlexSim_24h.txt
    Results_Validation_FlexSim_24h.docx
    Unified_Results_Section_MOFGPM_FlexSim.txt
    Unified_Results_Section_MOFGPM_FlexSim.docx
    validation_summary.json
```

Notes:

- `run_all_pipeline.py` writes under `outputs/` by default.
- `pipeline_flexsim_validation.py` reads GA exports from `results/run_*`, writes full validation outputs to `results/validation_flexsim_24h`, and writes baseline-only artifacts to `results/result_baseline` by default.
- Baseline defaults in `pipeline_flexsim_validation.py` now target `Doctor=3`, `Nurse=3`, `Specialist=1` with assistant levels `1` and `2`, and export `10` FlexSim replications per baseline.

## Run Commands

### Optimization run

```powershell
python main.py
```

### End-to-end pipeline

```powershell
python run_all_pipeline.py --install-deps --modes both
```

Combinatorics-only:

```powershell
python run_all_pipeline.py --modes combinatorics
```

### GA vs FlexSim validation (24h served-only)

```powershell
python pipeline_flexsim_validation.py
```

Typical explicit command:

```powershell
python pipeline_flexsim_validation.py --results-dir results --ga-subdir combinatorics --flexsim-file results/Flexsim/10_replicas.xlsx --outdir results/validation_flexsim_24h --baseline-outdir results/result_baseline --baseline-ga-subdir combinatorics --baseline-doctor 3 --baseline-nurse 3 --baseline-assistants 1 2 --baseline-specialist 1 --baseline-replicates 10 --mismatch-threshold-pct 20
```

## Validation Function Map (`pipeline_flexsim_validation.py`)

- `_load_ga_exports`: loads GA `results.csv`, `schedule_by_sample.csv`, `triage_waits_by_sample.csv`, `settings.json`.
- `_load_flexsim_exports`: loads FlexSim sheet and replication columns, computes `FS_mean`, `FS_std`, `FS_ci95`.
- `_select_ga_rows_by_staff`: keeps one GA row per staffing combination (feasible first, then highest lambda, then lowest wait).
- `_compute_ga_patient_waits`: reconstructs patient-level wait and served-within-24h flag from schedule + triage waits.
- `_aggregate_ga_sample_metrics`: computes GA totals/averages for all finished and served-within-24h cohorts.
- `_extract_baseline_metrics`: extracts baseline feasibility metrics for one staffing tuple.
- `_build_baseline_flexsim_replications`: exports baseline FlexSim replications (default request: assistant levels `1` and `2`, `10` replications each).
- `_build_comparison_dataset`: merges GA and FlexSim and computes comparison metrics and mismatch flags.
- `_build_table1`, `_build_table2`, `_build_table3`: exports reconciliation, KPI, and statistical-check tables.
- `_plot_*`: generates validation figures and appendix figures.
- `_write_results_text`, `_write_word_report`, `_write_unified_results_*`: creates narrative outputs.
- `run_pipeline`: orchestrates complete validation workflow.

## GA vs FlexSim Comparison Logic

Validation uses a comparable denominator: patients served within 24h.

Key formulas:

```text
FS_mean = mean(FlexSim replications)
FS_ci95 = 1.96 * FS_std / sqrt(n_reps)

GA_scaled_total_24h = GA_avg_wait_per_served * Patient

diff_per_patient = GA_avg_wait_per_served - FS_avg_wait_per_served
diff_total_24h   = GA_scaled_total_24h - FS_mean

served_gap_patients = N_served_24h_ga - Patient
served_gap_pct      = 100 * served_gap_patients / Patient

capacity_horizon_mismatch = abs(served_gap_pct) > mismatch_threshold_pct

z_score = (GA_scaled_total_24h - FS_mean) / FS_std
z_interpretation:
  Consistent if |z| < 2
  Material gap if |z| >= 2
```

Sign conventions:

- In `table1`, `served_gap_patients > 0` means GA served more than FlexSim.
- In Figure S3, value plotted is `Patient - N_served_24h_ga` (FlexSim minus GA), so positive means FlexSim served more.

## How To Interpret Validation Figures

### Main comparison figures

- `fig1_aggregated_ratio_view.png`
  - Left panel: scenario-level ratio `GA_scaled_total_24h / FS_mean`.
  - Dashed line at 1.0 is parity.
  - Red points are capacity/horizon mismatches; blue points are non-mismatch.
  - Right panel: quantile-bin mean ratio with 95% CI plus mismatch rate bars.

- `fig2_aggregated_kpi_summary.png`
  - Left panel: overall mean level comparison (`FS_mean` vs `GA_scaled_total_24h`) with CI.
  - Right panel: ratio distribution split by mismatch and non-mismatch groups.

- `fig3_per_served_patient_kpi_bar.png`
  - Mean per-served-patient KPI: GA vs FlexSim.
  - Error bars show cross-scenario variation.

- `fig4_served_rate_per_hour_bar.png`
  - Mean served patients per hour: GA vs FlexSim.
  - Error bars show cross-scenario variation.

- `appendix_figA1_scenario_ci_vs_ga_scaled.png`
  - Scenario-by-scenario detail.
  - FlexSim mean with CI (horizontal bars) vs GA scaled point (x marker).

### Baseline and staffing heatmaps

- `figB1_baseline_completion_profile.png`
  - Patient completion times in a one-column, two-row layout:
    - Row 1: `Assistant=1`
    - Row 2: `Assistant=2`
  - Checks if completions remain within 24h horizon.

- `figS1_ga_served24h_by_staff_grid.png`
  - Heatmap of GA `N_served_24h_ga` by staffing mix.
  - Panels are Assistant x Specialist combinations; within each panel, x=Doctor and y=Nurse.

- `figS2_flexsim_patient_by_staff_grid.png`
  - Heatmap of FlexSim `Patient` count with the same panel and axis structure.

- `figS3_gap_flexsim_minus_ga_by_staff_grid.png`
  - Heatmap of `Patient - N_served_24h_ga`.
  - Near zero means strong agreement in served volume.
  - Large positive means FlexSim served more; large negative means GA served more.

All validation figures are now generated without figure titles to simplify manuscript insertion.

## Validation Tables and Artifacts

- `table1_cohort_capacity_reconciliation.csv`: served-volume reconciliation and mismatch flag.
- `table2_kpi_comparison_served_only.csv`: KPI comparison in served-only cohort.
- `table3_statistical_check.csv`: z-score consistency check.
- `validation_matched_dataset_24h.csv`: joined row-level dataset used for all comparisons.
- `ga_patient_level_waits_24h.csv`: patient-level GA waits used to build cohort aggregates.
- `baseline_fixed_case_summary.csv`: baseline run-level table with `20` rows by default (`10` runs for `Assistant=1` and `10` runs for `Assistant=2`), plus GA baseline diagnostics merged by staffing tuple.
- `sensitivity_top_staffing_by_served24h.csv`: top staffing combinations by GA served volume.
- `validation_summary.json`: concise machine-readable summary of run-level outcomes.

## Existing Per-Run and History Interpretation

### Per-run figures (`fixed/`, `sensitivity/`, `combinatorics/`)

- `plot_cost_wait_scatter_lambda.png`: feasible high-lambda candidate screening.
- `plot_membership_decomposition.png`: shows binding membership in `lambda = min(mu_cost, mu_wait)`.
- `plot_feasibility_heatmap_doctor_nurse.png`: feasibility boundary over Doctor x Nurse.
- `plot_triage_first_wait_boxplot.png` and `plot_triage_first_wait_ecdf.png`: triage service quality/equity checks.
- `plot_baseline_schedule_tmax1440.png`: one-day baseline schedule timeline.

### Merged-history figures (`outputs/analysis/sensitivity_history/`)

- `plot_history_best_lambda_by_scenario_key.png`: best lambda by operational scenario.
- `plot_history_lambda_boxplot_by_scenario_key.png`: lambda dispersion/robustness by scenario.
- `plot_history_lambda_density.png`: feasible vs infeasible lambda distribution.
- `plot_history_lambda_vs_resource_levels.png`: resource-level effects on lambda.
- `plot_history_operational_pareto_overview.png`: integrated operational and Pareto view.
- `plot_history_cost_wait_scatter_all_runs.png`: global cost-wait cloud across runs.
- `plot_history_mean_lambda_heatmap_doctor_nurse.png`: mean lambda over Doctor x Nurse.
- `plot_history_triage_first_wait_ecdf.png`: merged triage wait distributions.

## Recommended Reading Order

1. Start with `validation_summary.json` and `table1_cohort_capacity_reconciliation.csv`.
2. Read `fig1_aggregated_ratio_view.png` and `fig2_aggregated_kpi_summary.png` for top-level agreement.
3. Use `figS1`, `figS2`, `figS3` to localize agreement/disagreement by staffing combination.
4. Use `appendix_figA1_scenario_ci_vs_ga_scaled.png` and `table3_statistical_check.csv` for scenario-level statistical checks.
5. Use baseline outputs (`results/result_baseline/baseline_fixed_case_summary.csv`, `results/result_baseline/figB1_baseline_completion_profile.png`) for 24h feasibility and baseline run-level context.
