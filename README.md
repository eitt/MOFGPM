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
- `methdology.py`: generates a Word methodology document (`.docx`) with formatted equations.
- `run_all_pipeline.py`: one-command runner for fixed + sensitivity + methodology outputs.
- `results_report.py`: Word draft generator for Results section from pipeline outputs.
- `requirements.txt`: Python dependency list.

## Methodology Document Generation

Run:

```powershell
python methdology.py
```

Output:

- `Methodology_MOFGPM_Pipeline.docx` in the project root.
- `Methodology_MOFGPM_Pipeline.tex` in the project root.
- `Methodology_MOFGPM_Pipeline.pdf` if `pdflatex` is installed and available on PATH.

## Optimization Run

Run:

```powershell
python main.py
```

This runs the MOFGPM + GA workflow (baseline or random sensitivity mode based on `main.py` settings) and shows summary tables/plots.

## One-Command Full Pipeline

Run everything (fixed + sensitivity + methodology):

```powershell
python run_all_pipeline.py --install-deps --modes both
```

Default output structure:

- `outputs/run_<timestamp>/fixed/`
- `outputs/run_<timestamp>/sensitivity/`
- `outputs/run_<timestamp>/methodology/`

The `fixed` and `sensitivity` folders include:

- `results.csv`
- `settings.json`
- `plot_cost_vs_wait.png`
- `plot_lambda_vs_wait.png`
- `plot_lambda_vs_cost.png`
- `plot_cost_wait_scatter_lambda.png` (feasible/infeasible markers, λ visual encoding)
- `plot_membership_decomposition.png` (\(\mu_{Cost}\), \(\mu_{Wait}\), and \(\lambda\) curves)
- `plot_feasibility_heatmap_doctor_nurse.png` (best feasible λ over Doctor × Nurse grid)
- `plot_triage_first_wait_boxplot.png` (distribution by triage)
- `plot_triage_first_wait_ecdf.png` (ECDF by triage)
- `triage_waits_by_sample.csv`

The `methodology` folder includes (when available):

- `Methodology_MOFGPM_Pipeline.docx`
- `Methodology_MOFGPM_Pipeline.tex`
- `Methodology_MOFGPM_Pipeline.pdf` (requires `pdflatex`)
- `Results_Draft_MOFGPM.docx` (auto-generated results section draft with tables and figures)

### Example with your settings

```powershell
python run_all_pipeline.py --modes both --fuzzy-scenario expected --num-patients 3 --data-seed 42 --t-max-day 1440 --n-samples-per-level 25 --pop-size 200 --generations 120 --mut-rate 0.2 --ga-seed-base 202600 --goal-cost 650 --max-cost 22750 --goal-wait 50 --max-wait 200000 --triage-limit-1 0 --triage-limit-2 30 --triage-limit-3 240 --triage-limit-4 720 --triage-limit-5 inf --triage-penalty 1000 --infeas-base-penalty 1000000 --exp-center median --fixed-doctor 3 --fixed-nurse 3 --fixed-assistant 6 --fixed-specialist 1 --n-staff-samples 25 --bound-doctor-lo 1 --bound-doctor-hi 6 --bound-nurse-lo 1 --bound-nurse-hi 8 --bound-assistant-lo 1 --bound-assistant-hi 10 --bound-specialist-lo 0 --bound-specialist-hi 2
```

## Equation Rendering Guarantee

`methdology.py` now uses a fail-safe equation pipeline:

- Accepts LaTeX-like input lines in code.
- Converts them to Word-safe math text (Greek symbols, inequalities, sums, fractions).
- Inserts them as Word equation objects (OMML), preventing raw LaTeX tokens like `\frac` or `\sigma` from appearing in the final document.

## Validation Scope

The methodology includes:

- Baseline case analysis (fixed staffing).
- Sensitivity analysis (sampled staffing).
- External validation with FlexSim simulation to compare GA+MOFGPM outputs (waiting time, service attainment, utilization).

## How To Interpret Outputs (Project Aim Alignment)

Project aim: support baseline-anchored ED staffing decisions under uncertainty by identifying:

- whether current staffing is feasible,
- where feasibility boundaries are,
- and what cost is required for better service performance.

### Core tables

- `results.csv`
  - One row per tested staffing plan.
  - Key fields:
    - `feasible`: whether triage constraints and MOFGPM conditions are satisfied.
    - `cost`, `wait`: raw tradeoff metrics.
    - `mu_cost`, `mu_wait`, `lambda`: fuzzy-goal satisfaction metrics.
  - Interpretation:
    - Start from feasible rows.
    - Rank by highest `lambda`, then lower `cost` / `wait`.
    - The smaller of `mu_cost` and `mu_wait` is the binding objective.

- `triage_waits_by_sample.csv`
  - Patient-level waiting metrics by triage and staffing sample.
  - Interpretation:
    - Use to verify service quality distribution (not only aggregate wait).
    - Compare triage classes for equity and SLA consistency.

### Figures and their decision meaning

- `plot_cost_vs_wait.png`
  - Basic cost vs waiting tradeoff cloud.
  - Interpretation: visual first-pass of efficiency vs service delay.

- `plot_lambda_vs_wait.png` and `plot_lambda_vs_cost.png`
  - Satisfaction index `lambda` against each objective.
  - Interpretation: identify regions where balanced performance is strong (`lambda` near 1).

- `plot_cost_wait_scatter_lambda.png`
  - Manager-ready tradeoff map.
  - Feasible/infeasible plans use different markers; `lambda` is encoded visually.
  - Interpretation: best actionable candidates are feasible points with high `lambda` and acceptable cost.

- `plot_membership_decomposition.png`
  - `mu_cost`, `mu_wait`, and `lambda` curves vs sample (or `W_max` if swept).
  - Interpretation: reveals which objective binds (`lambda = min(mu_cost, mu_wait)`), guiding whether to relax budget or service tolerance.

- `plot_feasibility_heatmap_doctor_nurse.png`
  - Staffing feasibility map over Doctors × Nurses.
  - Color shows best feasible `lambda`; missing cells indicate no feasible solution.
  - Interpretation: operational feasibility boundary for staffing policy.

- `plot_triage_first_wait_boxplot.png` and `plot_triage_first_wait_ecdf.png`
  - Distribution view of time-to-first-provider by triage class.
  - Interpretation: clinical relevance check; ensures aggregate improvements do not hide poor performance in critical triage groups.

### Results report draft

- `methodology/Results_Draft_MOFGPM.docx`
  - Narrative-ready results section for manuscript/reporting.
  - Interpretation flow:
    - baseline status,
    - scenario feasibility rates,
    - top feasible tradeoff solutions,
    - feasibility boundary diagnostics,
    - figure-supported managerial conclusions.
