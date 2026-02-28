import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
        ("Figure 9. Baseline schedule overview (t<=1440)", fixed_dir / "plot_baseline_schedule_tmax1440.png"),
    ]
    for caption, p in candidates:
        if p.exists():
            mapping[caption] = str(p)
    return mapping


def _build_history_figure_map(history_dir: Path) -> dict:
    mapping = {}
    candidates = [
        ("Figure 10. Best lambda by operational scenario key", history_dir / "plot_history_best_lambda_by_scenario_key.png"),
        ("Figure 11. Lambda distribution by operational scenario key", history_dir / "plot_history_lambda_boxplot_by_scenario_key.png"),
        ("Figure 12. Lambda density (feasible vs infeasible)", history_dir / "plot_history_lambda_density.png"),
        ("Figure 13. Lambda vs resource levels", history_dir / "plot_history_lambda_vs_resource_levels.png"),
        ("Figure 14. Operational Pareto overview", history_dir / "plot_history_operational_pareto_overview.png"),
        ("Figure 15. Cost-Wait scatter (all merged runs)", history_dir / "plot_history_cost_wait_scatter_all_runs.png"),
        ("Figure 16. Mean lambda heatmap (Doctor x Nurse)", history_dir / "plot_history_mean_lambda_heatmap_doctor_nurse.png"),
        ("Figure 17. Triage first-wait ECDF (merged)", history_dir / "plot_history_triage_first_wait_ecdf.png"),
    ]
    for caption, p in candidates:
        if p.exists():
            mapping[caption] = str(p)
    return mapping


def _extract_run_tag(run_dir: Path) -> str:
    name = run_dir.name
    return name[4:] if name.startswith("run_") else name


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:
        return {}


def _collect_sensitivity_history(base_outputs_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    triage_frames = []

    run_dirs = sorted([p for p in base_outputs_dir.glob("run_*") if p.is_dir()])
    for run_dir in run_dirs:
        sens_dir = run_dir / "sensitivity"
        results_csv = sens_dir / "results.csv"
        if not results_csv.exists():
            continue

        try:
            df = pd.read_csv(results_csv)
        except Exception:
            continue

        settings = _load_json(sens_dir / "settings.json")
        df["run_dir"] = str(run_dir.resolve())
        df["run_tag"] = _extract_run_tag(run_dir)
        df["fuzzy_scenario"] = settings.get("FUZZY_SCENARIO")
        df["configured_num_patients"] = settings.get("NUM_PATIENTS")
        df["t_max_day"] = settings.get("T_MAX_DAY")
        df["n_samples_per_level"] = settings.get("N_SAMPLES_PER_LEVEL")
        df["random_staff_seed"] = settings.get("RANDOM_STAFF_SEED")
        df["goal_wait"] = settings.get("GOAL_WAIT")
        df["max_wait"] = settings.get("MAX_WAIT")
        df["goal_cost"] = settings.get("GOAL_COST")
        df["max_cost"] = settings.get("MAX_COST")
        df["exp_center"] = settings.get("EXP_CENTER")
        df["pop_size"] = settings.get("POP_SIZE")
        df["generations"] = settings.get("GENERATIONS")
        df["mut_rate"] = settings.get("MUT_RATE")
        df["scenario_key"] = (
            "FZ="
            + df["fuzzy_scenario"].astype(str)
            + "|P="
            + df["configured_num_patients"].astype(str)
            + "|T="
            + df["t_max_day"].astype(str)
            + "|NS="
            + df["n_samples_per_level"].astype(str)
            + "|GW="
            + df["goal_wait"].astype(str)
            + "|MW="
            + df["max_wait"].astype(str)
        )
        frames.append(df)

        triage_csv = sens_dir / "triage_waits_by_sample.csv"
        if triage_csv.exists():
            try:
                tdf = pd.read_csv(triage_csv)
            except Exception:
                tdf = pd.DataFrame()
            if not tdf.empty:
                tdf["run_dir"] = str(run_dir.resolve())
                tdf["run_tag"] = _extract_run_tag(run_dir)
                triage_frames.append(tdf)

    history = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    triage_history = pd.concat(triage_frames, ignore_index=True) if triage_frames else pd.DataFrame()
    return history, triage_history


def _save_fig(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _pareto_mask_min_cost_wait(df: pd.DataFrame) -> np.ndarray:
    if df.empty:
        return np.array([], dtype=bool)
    vals = df[["cost", "wait"]].to_numpy(dtype=float)
    keep = np.ones(len(vals), dtype=bool)
    for i in range(len(vals)):
        if not keep[i]:
            continue
        dominated = (vals[:, 0] <= vals[i, 0]) & (vals[:, 1] <= vals[i, 1]) & (
            (vals[:, 0] < vals[i, 0]) | (vals[:, 1] < vals[i, 1])
        )
        if np.any(dominated):
            keep[i] = False
    return keep


def _plot_sensitivity_history(history: pd.DataFrame, triage_history: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_paths = []
    if history.empty:
        return out_paths

    history = history.copy()
    history["feasible"] = history["feasible"].astype(bool)
    history["total_staff"] = (
        history.get("Doctor", 0).fillna(0)
        + history.get("Nurse", 0).fillna(0)
        + history.get("Assistant", 0).fillna(0)
        + history.get("Specialist", 0).fillna(0)
    )

    # 1) Best lambda grouped by scenario characteristics.
    best = (
        history[history["feasible"]]
        .groupby("scenario_key", as_index=False)["lambda"]
        .max()
        .rename(columns={"lambda": "best_lambda"})
    )
    if not best.empty:
        best = best.sort_values("best_lambda", ascending=False)
        fig, ax = plt.subplots(figsize=(9.0, 4.8))
        ax.bar(best["scenario_key"], best["best_lambda"], color="#4C78A8")
        ax.set_xlabel("Operational scenario key")
        ax.set_ylabel("Best feasible lambda")
        ax.set_ylim(-0.02, 1.02)
        ax.tick_params(axis="x", rotation=45)
        out = out_dir / "plot_history_best_lambda_by_scenario_key.png"
        _save_fig(fig, out)
        out_paths.append(out)

    # 2) Lambda distribution by scenario characteristics.
    keys = sorted(history["scenario_key"].dropna().astype(str).unique().tolist())
    values = [history.loc[history["scenario_key"] == k, "lambda"].dropna().to_numpy() for k in keys]
    if keys and any(v.size > 0 for v in values):
        fig, ax = plt.subplots(figsize=(10.5, 5.6))
        ax.boxplot(values, tick_labels=keys, showfliers=False)
        ax.set_xlabel("Operational scenario key")
        ax.set_ylabel("Lambda distribution")
        ax.set_ylim(-0.02, 1.02)
        ax.tick_params(axis="x", rotation=45)
        out = out_dir / "plot_history_lambda_boxplot_by_scenario_key.png"
        _save_fig(fig, out)
        out_paths.append(out)

    # 3) Dense lambda distribution.
    lam_feas = history.loc[history["feasible"], "lambda"].dropna().to_numpy(dtype=float)
    lam_infeas = history.loc[~history["feasible"], "lambda"].dropna().to_numpy(dtype=float)
    if lam_feas.size > 0 or lam_infeas.size > 0:
        fig, ax = plt.subplots(figsize=(9.0, 5.2))
        bins = np.linspace(0.0, 1.0, 21)
        if lam_feas.size > 0:
            ax.hist(lam_feas, bins=bins, alpha=0.6, density=True, label="Feasible", color="tab:green")
        if lam_infeas.size > 0:
            ax.hist(lam_infeas, bins=bins, alpha=0.6, density=True, label="Infeasible", color="tab:red")
        ax.set_xlabel("Lambda")
        ax.set_ylabel("Density")
        ax.legend(loc="best")
        out = out_dir / "plot_history_lambda_density.png"
        _save_fig(fig, out)
        out_paths.append(out)

    # 4) Cost-wait scatter for all runs (lambda encoded).
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    sc = ax.scatter(
        history["cost"],
        history["wait"],
        c=history["lambda"].clip(lower=0.0, upper=1.0),
        cmap="viridis",
        alpha=0.8,
        edgecolors="#202020",
        linewidths=0.3,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Lambda")
    ax.set_xlabel("Cost")
    ax.set_ylabel("Total waiting time")
    out = out_dir / "plot_history_cost_wait_scatter_all_runs.png"
    _save_fig(fig, out)
    out_paths.append(out)

    # 5) Lambda vs resource levels as subplots.
    resources = ["Doctor", "Nurse", "Assistant", "Specialist"]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.4))
    axes = axes.ravel()
    plotted = False
    for i, r in enumerate(resources):
        ax = axes[i]
        if r not in history.columns:
            ax.axis("off")
            continue
        levels = sorted(history[r].dropna().astype(int).unique().tolist())
        if not levels:
            ax.axis("off")
            continue
        vals = [history.loc[history[r].astype(int) == lvl, "lambda"].dropna().to_numpy() for lvl in levels]
        if not any(v.size > 0 for v in vals):
            ax.axis("off")
            continue
        ax.boxplot(vals, tick_labels=levels, showfliers=False)
        ax.set_title(r)
        ax.set_xlabel("Staff level")
        ax.set_ylabel("Lambda")
        ax.set_ylim(-0.02, 1.02)
        plotted = True
    if plotted:
        out = out_dir / "plot_history_lambda_vs_resource_levels.png"
        _save_fig(fig, out)
        out_paths.append(out)
    else:
        plt.close(fig)

    # 6) Average lambda by staffing level (Doctor/Nurse heatmap).
    if {"Doctor", "Nurse"}.issubset(history.columns):
        grp = (
            history.groupby(["Nurse", "Doctor"], as_index=False)["lambda"]
            .mean()
            .rename(columns={"lambda": "mean_lambda"})
        )
        pivot = grp.pivot(index="Nurse", columns="Doctor", values="mean_lambda")
        if not pivot.empty:
            y_vals = sorted(pivot.index.tolist())
            x_vals = sorted(pivot.columns.tolist())
            arr = pivot.reindex(index=y_vals, columns=x_vals).to_numpy(dtype=float)
            masked = np.ma.masked_invalid(arr)

            cmap = plt.colormaps["viridis"].copy()
            cmap.set_bad("#d9d9d9")

            fig, ax = plt.subplots(figsize=(8.4, 6.1))
            im = ax.imshow(masked, origin="lower", cmap=cmap, aspect="auto", vmin=0.0, vmax=1.0)
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Mean lambda")
            ax.set_xlabel("Doctors")
            ax.set_ylabel("Nurses")
            ax.set_xticks(range(len(x_vals)))
            ax.set_xticklabels(x_vals)
            ax.set_yticks(range(len(y_vals)))
            ax.set_yticklabels(y_vals)
            out = out_dir / "plot_history_mean_lambda_heatmap_doctor_nurse.png"
            _save_fig(fig, out)
            out_paths.append(out)

    # 7) Operational-performance integrated Pareto overview.
    feas = history[history["feasible"]].copy()
    if not feas.empty:
        pareto = feas[_pareto_mask_min_cost_wait(feas)].copy()
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
        ax1, ax2 = axes

        ax1.scatter(
            feas["cost"], feas["wait"], c="#c7c7c7", alpha=0.6, s=30, edgecolors="none", label="Feasible samples"
        )
        if not pareto.empty:
            pareto = pareto.sort_values("cost")
            ax1.plot(pareto["cost"], pareto["wait"], color="tab:red", marker="o", linewidth=1.8, label="Pareto front")
        ax1.set_xlabel("Cost")
        ax1.set_ylabel("Total waiting time")
        ax1.legend(loc="best")

        sc2 = ax2.scatter(
            feas["wait"],
            feas["lambda"],
            c=feas["cost"],
            s=20 + 12 * feas["total_staff"].clip(lower=0),
            cmap="plasma",
            alpha=0.8,
            edgecolors="#202020",
            linewidths=0.25,
        )
        cbar2 = fig.colorbar(sc2, ax=ax2)
        cbar2.set_label("Cost")
        ax2.set_xlabel("Total waiting time")
        ax2.set_ylabel("Lambda")
        ax2.set_ylim(-0.02, 1.02)
        out = out_dir / "plot_history_operational_pareto_overview.png"
        _save_fig(fig, out)
        out_paths.append(out)

    # 8) Cross-run triage first-wait ECDF.
    if not triage_history.empty and "first_wait" in triage_history.columns:
        fig, ax = plt.subplots(figsize=(9.0, 5.4))
        tri = triage_history.copy()
        tri = tri[tri["first_wait"].notna()]
        tri = tri[tri["first_wait"] >= 0]
        if "feasible" in tri.columns and tri["feasible"].notna().any():
            tri = tri[tri["feasible"].astype(bool)]
        for t in sorted(tri["triage"].dropna().astype(int).unique().tolist()):
            x = np.sort(tri.loc[tri["triage"].astype(int) == t, "first_wait"].to_numpy(dtype=float))
            if x.size == 0:
                continue
            y = np.arange(1, x.size + 1) / x.size
            ax.step(x, y, where="post", label=f"T{t}")
        if ax.has_data():
            ax.set_xlabel("Time to first provider (minutes)")
            ax.set_ylabel("ECDF")
            ax.set_ylim(0.0, 1.0)
            ax.legend(loc="lower right")
            out = out_dir / "plot_history_triage_first_wait_ecdf.png"
            _save_fig(fig, out)
            out_paths.append(out)
        else:
            plt.close(fig)

    return out_paths


def _plot_baseline_schedule(fixed_dir: Path) -> Path | None:
    schedule_csv = fixed_dir / "schedule_by_sample.csv"
    triage_csv = fixed_dir / "triage_waits_by_sample.csv"
    settings = _load_json(fixed_dir / "settings.json")
    if not schedule_csv.exists():
        return None
    if float(settings.get("T_MAX_DAY", 0.0)) > 1440.0:
        return None

    try:
        sched = pd.read_csv(schedule_csv)
    except Exception:
        return None
    if sched.empty:
        return None

    sample_id = int(sched["sample_id"].min())
    sched = sched[sched["sample_id"] == sample_id].copy()
    if sched.empty:
        return None

    # Use one row per patient from first start to last completion.
    gp = (
        sched.groupby("patient", as_index=False)
        .agg(first_start=("start", "min"), finish=("end", "max"), triage=("triage", "first"))
        .sort_values(["triage", "first_start", "patient"])
        .reset_index(drop=True)
    )
    if gp.empty:
        return None

    if triage_csv.exists():
        try:
            tri = pd.read_csv(triage_csv)
        except Exception:
            tri = pd.DataFrame()
        if not tri.empty:
            tri = tri[tri["sample_id"] == sample_id].copy()
            tri = tri[["patient", "first_wait"]].drop_duplicates()
            gp = gp.merge(tri, on="patient", how="left")
            gp["arrival"] = gp["first_start"] - gp["first_wait"]
        else:
            gp["arrival"] = np.nan
    else:
        gp["arrival"] = np.nan

    fig, ax = plt.subplots(figsize=(11.5, 8.0))
    triage_palette = {
        1: "#1f77b4",
        2: "#ff7f0e",
        3: "#2ca02c",
        4: "#d62728",
        5: "#9467bd",
    }
    y = np.arange(len(gp))
    for i, row in gp.iterrows():
        c = triage_palette.get(int(row["triage"]), "#7f7f7f")
        ax.hlines(i, row["first_start"], row["finish"], color=c, linewidth=2.0)
        ax.scatter(row["first_start"], i, color=c, s=18, marker="o")
        ax.scatter(row["finish"], i, color=c, s=16, marker="s")
        if np.isfinite(row["arrival"]):
            ax.scatter(row["arrival"], i, color="#111111", s=18, marker="|")

    ax.set_yticks(y)
    ax.set_yticklabels(gp["patient"].tolist())
    ax.invert_yaxis()
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Patient")
    ax.set_title("Baseline schedule overview (t<=1440)")
    out = fixed_dir / "plot_baseline_schedule_tmax1440.png"
    _save_fig(fig, out)
    return out


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
    parser.add_argument(
        "--history-analysis-dir",
        default="analysis/sensitivity_history",
        help="Folder (relative to --outdir) for cross-run sensitivity CSV/plots.",
    )

    # Settings exposed from USER SETTINGS.
    parser.add_argument("--fuzzy-scenario", default="expected", choices=["optimistic", "expected", "pessimistic"])
    parser.add_argument("--fixed-doctor", type=int, default=3)
    parser.add_argument("--fixed-nurse", type=int, default=3)
    parser.add_argument("--fixed-assistant", type=int, default=6)
    parser.add_argument("--fixed-specialist", type=int, default=1)
    parser.add_argument("--n-staff-samples", type=int, default=1) # scenarios
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
    parser.add_argument(
        "--force-num-patients",
        action="store_true",
        help="Require exactly --num-patients within t<=t-max-day; fail if not possible in max tries.",
    )
    parser.add_argument(
        "--max-patient-gen-tries",
        type=int,
        default=200,
        help="Maximum retries when searching for enough generated arrivals in a day.",
    )
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
            "FORCE_NUM_PATIENTS": "1" if args.force_num_patients else "0",
            "MAX_PATIENT_GEN_TRIES": str(args.max_patient_gen_tries),
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
        baseline_plot = _plot_baseline_schedule(fixed_dir)
        if baseline_plot is not None:
            print(f"Saved: {baseline_plot}")

    if args.modes in {"both", "random"}:
        random_env = dict(common_env)
        random_env["RUN_MODE"] = "random"
        run_main(repo_dir, random_env, random_dir)

    run_methodology(repo_dir, methodology_dir)

    # Cross-batch sensitivity aggregation (keeps each batch untouched).
    base_outputs_dir = (repo_dir / args.outdir).resolve()
    history_dir = (base_outputs_dir / args.history_analysis_dir).resolve()
    history_df, triage_history_df = _collect_sensitivity_history(base_outputs_dir)
    if history_df.empty:
        print("[Warning] No sensitivity results found across run_* folders; skipping history analysis.")
    else:
        history_dir.mkdir(parents=True, exist_ok=True)
        hist_csv = history_dir / "sensitivity_results_all_runs.csv"
        triage_csv = history_dir / "triage_waits_all_runs.csv"
        history_df.to_csv(hist_csv, index=False)
        if not triage_history_df.empty:
            triage_history_df.to_csv(triage_csv, index=False)
        plot_paths = _plot_sensitivity_history(history_df, triage_history_df, history_dir)
        print(f"Saved: {hist_csv}")
        if not triage_history_df.empty:
            print(f"Saved: {triage_csv}")
        for p in plot_paths:
            print(f"Saved: {p}")

    # Generate Results draft report in Word using merged scenarios when available.
    results_df = _collect_results_frames(fixed_dir, random_dir)
    report_df = history_df if not history_df.empty else results_df
    if report_df.empty:
        print("[Warning] No results.csv found; skipping Results draft report generation.")
    else:
        results_report_path = methodology_dir / "Results_Draft_MOFGPM.docx"
        wmax_col = "wmax" if "wmax" in report_df.columns else ("mw" if "mw" in report_df.columns else "wmax")
        figures = _build_figure_map(fixed_dir, random_dir)
        if history_dir.exists():
            figures.update(_build_history_figure_map(history_dir))

        scenario_col = "scenario_key" if "scenario_key" in report_df.columns else "scenario"
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
                report_df,
                str(results_report_path),
                baseline_staff=baseline_staff,
                goals=goals,
                figures=figures,
                scenario_col=scenario_col,
                wmax_col=wmax_col,
            )
            print(f"Saved: {results_report_path}")
        except Exception as exc:
            print(f"[Warning] Could not generate Word results draft: {exc}")

    print(f"Done. Outputs in: {base_out}")


if __name__ == "__main__":
    main()
