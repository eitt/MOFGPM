# MOFGPM

Multi Objective Fuzzy Goal Programming (MOFGPM) workflow for ED resource planning.

## Requirements

### Python

- Python 3.10+ (recommended: 3.11 or 3.12)
- Install Python packages:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### System tools (for all outputs)

- `pdflatex` on PATH (MiKTeX or TeX Live) to generate PDF from LaTeX.
- Word is not required to generate `.docx`, but if the `.docx` file is open in Word, overwrite can fail due to file lock.

### Quick validation

```powershell
python -c "import numpy, pandas, matplotlib, scipy, docx; print('python deps ok')"
pdflatex --version
```

## Files

- `main.py`: optimization and sensitivity workflow (GA + MOFGPM).
- `methdology.py`: generates Word/LaTeX methodology outputs.
- `run_all_pipeline.py`: one-command runner for fixed + sensitivity + methodology + merged analysis outputs.
- `results_report.py`: Word draft generator for Results section from produced outputs.
- `requirements.txt`: Python dependency list.

## Methodology Document Generation

Run:

```powershell
python methdology.py
```

Outputs:

- `Methodology_MOFGPM_Pipeline.docx` in project root.
- `Methodology_MOFGPM_Pipeline.tex` in project root.
- `Methodology_MOFGPM_Pipeline.pdf` if `pdflatex` is installed and available on PATH.

## Optimization Run

Run:

```powershell
python main.py
```

This runs the MOFGPM + GA workflow (baseline or random sensitivity mode based on environment settings) and produces summary tables/plots.

## One-Command Full Pipeline

Run everything (fixed + sensitivity + methodology):

```powershell
python run_all_pipeline.py --install-deps --modes both
```

### Output folder organization

```text
outputs/
  run_<timestamp>/
    fixed/
    sensitivity/
    methodology/
  analysis/
    sensitivity_history/
```

- `outputs/run_<timestamp>/fixed/`
  - Fixed staffing (baseline) run outputs.
- `outputs/run_<timestamp>/sensitivity/`
  - Random staffing samples for sensitivity analysis.
- `outputs/run_<timestamp>/methodology/`
  - Methodology files and results draft report.
- `outputs/analysis/sensitivity_history/`
  - Merged cross-run sensitivity tables and plots.

### Typical files in `fixed/` and `sensitivity/`

- `settings.json`
- `results.csv`
- `triage_waits_by_sample.csv`
- `schedule_by_sample.csv`
- `plot_cost_vs_wait.png`
- `plot_lambda_vs_wait.png`
- `plot_lambda_vs_cost.png`
- `plot_cost_wait_scatter_lambda.png`
- `plot_membership_decomposition.png`
- `plot_feasibility_heatmap_doctor_nurse.png`
- `plot_triage_first_wait_boxplot.png`
- `plot_triage_first_wait_ecdf.png`

Baseline-only helper (when `T_MAX_DAY <= 1440`):

- `plot_baseline_schedule_tmax1440.png`

### Typical files in `analysis/sensitivity_history/`

- `sensitivity_results_all_runs.csv`
- `triage_waits_all_runs.csv` (if available)
- `plot_history_best_lambda_by_scenario_key.png`
- `plot_history_lambda_boxplot_by_scenario_key.png`
- `plot_history_lambda_density.png`
- `plot_history_lambda_vs_resource_levels.png`
- `plot_history_operational_pareto_overview.png`
- `plot_history_cost_wait_scatter_all_runs.png`
- `plot_history_mean_lambda_heatmap_doctor_nurse.png`
- `plot_history_triage_first_wait_ecdf.png`

### Typical files in `methodology/`

- `Methodology_MOFGPM_Pipeline.docx`
- `Methodology_MOFGPM_Pipeline.tex`
- `Methodology_MOFGPM_Pipeline.pdf` (if `pdflatex` is available)
- `Results_Draft_MOFGPM.docx`

## CSV Dictionary (Column Meaning)

### `results.csv` (sample-level summary)

- `scenario`: fuzzy scenario solved (`optimistic`, `expected`, `pessimistic`).
- `sample_id`: staffing sample index in that run.
- `Doctor`, `Nurse`, `Assistant`, `Specialist`: staffing levels for the sample.
- `wmax`: wait tolerance used in memberships.
- `cost`: staffing cost objective.
- `wait`: total waiting-time objective.
- `mu_cost`, `mu_wait`: membership values for cost and wait.
- `lambda`: MOFGPM value (`min(mu_cost, mu_wait)`).
- `feasible`: feasibility flag.
- `triage_violation`: triage-related violation amount.
- `ga_time_s`: solve time for that sample.

### `triage_waits_by_sample.csv` (patient-level waits)

- `scenario`, `sample_id`, `patient`, `triage`
- `first_wait`: arrival to first provider wait.
- `early_wait_j_le_4`: cumulative early-task waiting (`j <= 4` tasks).
- `schedule_complete`: whether required early tasks were scheduled.
- `feasible`: parent sample feasibility.

### `schedule_by_sample.csv` (task-level schedule)

- `scenario`, `sample_id`, `patient`, `triage`, `activity`, `task_j`
- `start`, `end`, `duration`
- `required_resources`: comma-separated required resources.
- `req_Doctor`, `req_Nurse`, `req_Assistant`, `req_Specialist`: binary resource requirements.
- `Doctor`, `Nurse`, `Assistant`, `Specialist`: staffing context.

### `sensitivity_results_all_runs.csv` (merged history)

Includes `results.csv` columns plus run metadata, such as:

- `run_tag`, `run_dir`
- `configured_num_patients`, `t_max_day`, `n_samples_per_level`
- `fuzzy_scenario`, `goal_wait`, `max_wait`, `goal_cost`, `max_cost`
- `scenario_key`: operational signature used for merged grouping

## Figure Interpretation Guide

### Per-run figures (`fixed/` and `sensitivity/`)

- `plot_cost_wait_scatter_lambda.png`
  - Use first to screen candidates.
  - Prefer feasible points with high `lambda`, then compare cost/wait.
- `plot_membership_decomposition.png`
  - Shows `mu_cost`, `mu_wait`, `lambda` behavior.
  - The lower membership is binding (`lambda = min(mu_cost, mu_wait)`).
- `plot_feasibility_heatmap_doctor_nurse.png`
  - Doctor-Nurse staffing feasibility boundary.
  - Empty/NA cells imply no feasible solution in sampled data.
- `plot_triage_first_wait_boxplot.png` and `plot_triage_first_wait_ecdf.png`
  - Check triage-level service quality and equity.
- `plot_baseline_schedule_tmax1440.png`
  - Easy-to-read baseline schedule timeline for a one-day horizon.

### Merged-history figures (`analysis/sensitivity_history/`)

- `plot_history_best_lambda_by_scenario_key.png`
  - Best achieved lambda by operational setting.
- `plot_history_lambda_boxplot_by_scenario_key.png`
  - Robustness/dispersion of lambda by operational setting.
- `plot_history_lambda_density.png`
  - Feasible vs infeasible lambda distribution.
- `plot_history_lambda_vs_resource_levels.png`
  - Resource-level effects on lambda (subplots by resource).
- `plot_history_operational_pareto_overview.png`
  - Integrated operational-performance view and Pareto trend.
- `plot_history_cost_wait_scatter_all_runs.png`
  - Global cost-wait cloud across all runs.
- `plot_history_mean_lambda_heatmap_doctor_nurse.png`
  - Mean lambda landscape over Doctor x Nurse levels.
- `plot_history_triage_first_wait_ecdf.png`
  - Merged triage service distributions.

## Recommended Reading Order

1. Start with merged-history figures in `analysis/sensitivity_history/` to identify robust regions.
2. Use run-level `results.csv` and `plot_cost_wait_scatter_lambda.png` for candidate shortlist.
3. Verify triage-level quality with `triage_waits_by_sample.csv` and triage plots.
4. Use `schedule_by_sample.csv` and baseline schedule plot for operational explainability.
5. Use `methodology/Results_Draft_MOFGPM.docx` for manuscript/report-ready narrative.

## Example Full Command

```powershell
python run_all_pipeline.py --modes both --fuzzy-scenario expected --num-patients 36 --force-num-patients --data-seed 1951 --t-max-day 1440 --n-samples-per-level 25 --pop-size 200 --generations 100 --mut-rate 0.2 --ga-seed-base 202600 --goal-cost 650 --max-cost 22750 --goal-wait 50 --max-wait 200000 --triage-limit-1 0 --triage-limit-2 30 --triage-limit-3 240 --triage-limit-4 720 --triage-limit-5 inf --triage-penalty 1000 --infeas-base-penalty 1000000 --exp-center median --fixed-doctor 3 --fixed-nurse 3 --fixed-assistant 6 --fixed-specialist 1 --n-staff-samples 25 --bound-doctor-lo 1 --bound-doctor-hi 6 --bound-nurse-lo 1 --bound-nurse-hi 8 --bound-assistant-lo 1 --bound-assistant-hi 10 --bound-specialist-lo 0 --bound-specialist-hi 2
```
