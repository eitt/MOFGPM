import argparse
import importlib
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from results_report import generate_results_report_word


REQUIRED_MODULES = ["numpy", "pandas", "matplotlib", "scipy", "docx"]


def check_python_requirements() -> list[str]:
    missing = []
    for mod in REQUIRED_MODULES:
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(mod)
    return missing


def maybe_install_requirements(repo_dir: Path) -> None:
    req = repo_dir / "requirements.txt"
    if not req.exists():
        print("requirements.txt not found. Skipping install.")
        return
    print("Installing Python dependencies from requirements.txt ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req)], check=True)


def run_main(repo_dir: Path, env: dict, outdir: Path) -> None:
    env = dict(env)
    env["OUTPUT_DIR"] = str(outdir.resolve())
    env["SAVE_PLOTS"] = "1"
    env["SHOW_PLOTS"] = "0"
    env["MPLBACKEND"] = "Agg"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Running main.py -> {outdir}")
    subprocess.run([sys.executable, "main.py"], cwd=repo_dir, env=env, check=True)


def run_methodology(repo_dir: Path, method_dir: Path) -> None:
    method_dir.mkdir(parents=True, exist_ok=True)
    print("Running methdology.py ...")
    subprocess.run([sys.executable, "methdology.py"], cwd=repo_dir, check=True)

    # Collect generated methodology files.
    for name in [
        "Methodology_MOFGPM_Pipeline.docx",
        "Methodology_MOFGPM_Pipeline.tex",
        "Methodology_MOFGPM_Pipeline.pdf",
    ]:
        src = repo_dir / name
        if src.exists():
            shutil.copy2(src, method_dir / name)


def _collect_results_frames(fixed_dir: Path, random_dir: Path) -> pd.DataFrame:
    frames = []
    fixed_csv = fixed_dir / "results.csv"
    random_csv = random_dir / "results.csv"
    if fixed_csv.exists():
        dff = pd.read_csv(fixed_csv)
        dff["run_folder"] = "fixed"
        frames.append(dff)
    if random_csv.exists():
        dfr = pd.read_csv(random_csv)
        dfr["run_folder"] = "sensitivity"
        frames.append(dfr)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _build_figure_map(fixed_dir: Path, random_dir: Path) -> dict:
    mapping = {}
    candidates = [
        ("Figure 1. Cost-Wait scatter (lambda encoding) - Fixed", fixed_dir / "plot_cost_wait_scatter_lambda.png"),
        ("Figure 2. Membership decomposition - Fixed", fixed_dir / "plot_membership_decomposition.png"),
        ("Figure 3. Feasibility heatmap (Doctor x Nurse) - Fixed", fixed_dir / "plot_feasibility_heatmap_doctor_nurse.png"),
        ("Figure 4. Triage first-wait boxplot - Fixed", fixed_dir / "plot_triage_first_wait_boxplot.png"),
        ("Figure 5. Cost-Wait scatter (lambda encoding) - Sensitivity", random_dir / "plot_cost_wait_scatter_lambda.png"),
        ("Figure 6. Membership decomposition - Sensitivity", random_dir / "plot_membership_decomposition.png"),
        ("Figure 7. Feasibility heatmap (Doctor x Nurse) - Sensitivity", random_dir / "plot_feasibility_heatmap_doctor_nurse.png"),
        ("Figure 8. Triage first-wait ECDF - Sensitivity", random_dir / "plot_triage_first_wait_ecdf.png"),
    ]
    for caption, p in candidates:
        if p.exists():
            mapping[caption] = str(p)
    return mapping


def main():
    parser = argparse.ArgumentParser(
        description="Run MOFGPM pipeline end-to-end with fixed/sensitivity folders and methodology outputs."
    )
    parser.add_argument("--install-deps", action="store_true", help="Install Python deps from requirements.txt first.")
    parser.add_argument(
        "--modes",
        choices=["both", "fixed", "random"],
        default="both",
        help="Which pipeline modes to run.",
    )
    parser.add_argument("--outdir", default="outputs", help="Base output directory.")
    parser.add_argument("--tag", default=datetime.now().strftime("%Y%m%d_%H%M%S"), help="Run tag for output folder.")

    # Settings exposed from USER SETTINGS.
    parser.add_argument("--fuzzy-scenario", default="expected", choices=["optimistic", "expected", "pessimistic"])
    parser.add_argument("--fixed-doctor", type=int, default=3)
    parser.add_argument("--fixed-nurse", type=int, default=3)
    parser.add_argument("--fixed-assistant", type=int, default=6)
    parser.add_argument("--fixed-specialist", type=int, default=1)
    parser.add_argument("--n-staff-samples", type=int, default=5) # scenarios
    parser.add_argument("--bound-doctor-lo", type=int, default=1)
    parser.add_argument("--bound-doctor-hi", type=int, default=6)
    parser.add_argument("--bound-nurse-lo", type=int, default=1)
    parser.add_argument("--bound-nurse-hi", type=int, default=8)
    parser.add_argument("--bound-assistant-lo", type=int, default=1)
    parser.add_argument("--bound-assistant-hi", type=int, default=10)
    parser.add_argument("--bound-specialist-lo", type=int, default=0)
    parser.add_argument("--bound-specialist-hi", type=int, default=2)
    parser.add_argument("--random-staff-seed", type=int, default=202600)
    parser.add_argument("--num-patients", type=int, default=36)   #Number of patients
    parser.add_argument("--data-seed", type=int, default=1951)
    parser.add_argument("--t-max-day", type=float, default=1440.0)
    parser.add_argument("--n-samples-per-level", type=int, default=25)
    parser.add_argument("--pop-size", type=int, default=200)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--mut-rate", type=float, default=0.2)
    parser.add_argument("--ga-seed-base", type=int, default=202600)
    parser.add_argument("--goal-cost", type=float, default=650.0)
    parser.add_argument("--max-cost", type=float, default=22750.0)
    parser.add_argument("--goal-wait", type=float, default=50.0)
    parser.add_argument("--max-wait", type=float, default=200000.0)
    parser.add_argument("--triage-limit-1", type=float, default=0.0)
    parser.add_argument("--triage-limit-2", type=float, default=30.0)
    parser.add_argument("--triage-limit-3", type=float, default=240.0)
    parser.add_argument("--triage-limit-4", type=float, default=720.0)
    parser.add_argument("--triage-limit-5", type=float, default=float("inf"))
    parser.add_argument("--triage-penalty", type=float, default=1000.0)
    parser.add_argument("--infeas-base-penalty", type=float, default=1e6)
    parser.add_argument("--exp-center", default="median", choices=["median", "mean"])

    args = parser.parse_args()
    repo_dir = Path(__file__).resolve().parent

    if args.install_deps:
        maybe_install_requirements(repo_dir)

    missing = check_python_requirements()
    if missing:
        print(f"Missing Python modules: {missing}")
        print("Install with: python -m pip install -r requirements.txt")
        raise SystemExit(1)

    base_out = (repo_dir / args.outdir / f"run_{args.tag}").resolve()
    fixed_dir = base_out / "fixed"
    random_dir = base_out / "sensitivity"
    methodology_dir = base_out / "methodology"
    base_out.mkdir(parents=True, exist_ok=True)

    common_env = os.environ.copy()
    common_env.update(
        {
            "FUZZY_SCENARIO": args.fuzzy_scenario,
            "FIXED_DOCTOR": str(args.fixed_doctor),
            "FIXED_NURSE": str(args.fixed_nurse),
            "FIXED_ASSISTANT": str(args.fixed_assistant),
            "FIXED_SPECIALIST": str(args.fixed_specialist),
            "N_STAFF_SAMPLES": str(args.n_staff_samples),
            "BOUND_DOCTOR_LO": str(args.bound_doctor_lo),
            "BOUND_DOCTOR_HI": str(args.bound_doctor_hi),
            "BOUND_NURSE_LO": str(args.bound_nurse_lo),
            "BOUND_NURSE_HI": str(args.bound_nurse_hi),
            "BOUND_ASSISTANT_LO": str(args.bound_assistant_lo),
            "BOUND_ASSISTANT_HI": str(args.bound_assistant_hi),
            "BOUND_SPECIALIST_LO": str(args.bound_specialist_lo),
            "BOUND_SPECIALIST_HI": str(args.bound_specialist_hi),
            "RANDOM_STAFF_SEED": str(args.random_staff_seed),
            "NUM_PATIENTS": str(args.num_patients),
            "DATA_SEED": str(args.data_seed),
            "T_MAX_DAY": str(args.t_max_day),
            "N_SAMPLES_PER_LEVEL": str(args.n_samples_per_level),
            "POP_SIZE": str(args.pop_size),
            "GENERATIONS": str(args.generations),
            "MUT_RATE": str(args.mut_rate),
            "GA_SEED_BASE": str(args.ga_seed_base),
            "GOAL_COST": str(args.goal_cost),
            "MAX_COST": str(args.max_cost),
            "GOAL_WAIT": str(args.goal_wait),
            "MAX_WAIT": str(args.max_wait),
            "TRIAGE_LIMIT_1": str(args.triage_limit_1),
            "TRIAGE_LIMIT_2": str(args.triage_limit_2),
            "TRIAGE_LIMIT_3": str(args.triage_limit_3),
            "TRIAGE_LIMIT_4": str(args.triage_limit_4),
            "TRIAGE_LIMIT_5": str(args.triage_limit_5),
            "TRIAGE_PENALTY": str(args.triage_penalty),
            "INFEAS_BASE_PENALTY": str(args.infeas_base_penalty),
            "EXP_CENTER": str(args.exp_center),
        }
    )

    if args.modes in {"both", "fixed"}:
        fixed_env = dict(common_env)
        fixed_env["RUN_MODE"] = "fixed"
        run_main(repo_dir, fixed_env, fixed_dir)

    if args.modes in {"both", "random"}:
        random_env = dict(common_env)
        random_env["RUN_MODE"] = "random"
        run_main(repo_dir, random_env, random_dir)

    run_methodology(repo_dir, methodology_dir)

    # Generate Results draft report in Word from produced CSVs and figures.
    results_df = _collect_results_frames(fixed_dir, random_dir)
    if not results_df.empty:
        results_report_path = methodology_dir / "Results_Draft_MOFGPM.docx"
        wmax_col = "wmax" if "wmax" in results_df.columns else "mw"
        figures = _build_figure_map(fixed_dir, random_dir)
        baseline_staff = {
            "Doctor": args.fixed_doctor,
            "Nurse": args.fixed_nurse,
            "Assistant": args.fixed_assistant,
            "Specialist": args.fixed_specialist,
        }
        goals = {
            "G_cost": args.goal_cost,
            "C_max": args.max_cost,
            "G_wait": args.goal_wait,
            "W_max": args.max_wait,
        }
        try:
            generate_results_report_word(
                results_df,
                str(results_report_path),
                baseline_staff=baseline_staff,
                goals=goals,
                figures=figures,
                scenario_col="scenario",
                wmax_col=wmax_col,
            )
            print(f"Saved: {results_report_path}")
        except Exception as exc:
            print(f"[Warning] Could not generate Word results draft: {exc}")
    else:
        print("[Warning] No results.csv found; skipping Results draft report generation.")

    print(f"Done. Outputs in: {base_out}")


if __name__ == "__main__":
    main()
