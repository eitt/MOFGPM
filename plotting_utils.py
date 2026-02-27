import numpy as np
import matplotlib.pyplot as plt


def plot_lambda_vs_wmax_letter(results_by_fuzzy, out_png="lambda_vs_Wmax_letter.png"):
    scenarios = ["optimistic", "expected", "pessimistic"]
    labels = {"optimistic": "Optimistic", "expected": "Expected value", "pessimistic": "Pessimistic"}
    colors = {"optimistic": "tab:blue", "expected": "tab:purple", "pessimistic": "tab:red"}

    def get_feasible(fs):
        mw = np.asarray(results_by_fuzzy[fs]["mw"], dtype=float)
        lam = np.asarray(results_by_fuzzy[fs]["lam"], dtype=float)
        wait = np.asarray(results_by_fuzzy[fs]["wait"], dtype=float)
        feas = np.asarray(results_by_fuzzy[fs]["feas"], dtype=bool)
        mask = feas & np.isfinite(mw) & np.isfinite(lam) & np.isfinite(wait)
        return mw[mask], lam[mask], wait[mask]

    all_mw = []
    for fs in scenarios:
        all_mw.extend(results_by_fuzzy[fs]["mw"])
    all_mw = np.asarray(all_mw, dtype=float)
    all_mw = all_mw[np.isfinite(all_mw)]
    x_min = 0.0
    x_max = float(np.max(all_mw)) if all_mw.size else 1.0

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "figure.dpi": 100,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)

    ax_all = axes[0, 0]
    ax_opt = axes[0, 1]
    ax_exp = axes[1, 0]
    ax_pes = axes[1, 1]

    for fs in scenarios:
        mw_f, lam_f, _ = get_feasible(fs)
        if mw_f.size == 0:
            continue
        idx = np.argsort(mw_f)
        ax_all.plot(mw_f[idx], lam_f[idx], marker="o", markersize=5, color=colors[fs], label=labels[fs])

    ax_all.set_title("λ vs $W_{max}$ (Feasible only) - All scenarios")
    ax_all.set_xlabel("$W_{max}$ (wait tolerance / goal upper bound)")
    ax_all.set_ylabel("λ (min satisfaction)")
    ax_all.set_xlim(x_min, x_max)
    ax_all.set_ylim(-0.02, 1.02)
    ax_all.grid(True, linestyle="--", alpha=0.35)
    ax_all.legend(loc="best")
    ax_all.axhline(0, linestyle="--", linewidth=1)

    def single_panel(ax, fs):
        mw_f, lam_f, wait_f = get_feasible(fs)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.axhline(0, linestyle="--", linewidth=1)

        if mw_f.size == 0:
            ax.set_title(f"λ vs $W_{{max}}$ ({labels[fs]}) - No feasible points")
            ax.set_xlabel("$W_{max}$ (wait tolerance)")
            ax.set_ylabel("λ (min satisfaction)")
            return

        idx = np.argsort(mw_f)
        mw_s, lam_s = mw_f[idx], lam_f[idx]
        min_feasible_wait = float(np.min(wait_f))

        ax.plot(mw_s, lam_s, marker="o", markersize=5, color=colors[fs], label=labels[fs])
        ax.set_title(f"λ vs $W_{{max}}$ ({labels[fs]})")
        ax.set_xlabel("$W_{max}$ (wait tolerance)")
        ax.set_ylabel("λ (min satisfaction)")
        ax.legend([f"Minimum feasible Wait metric: {min_feasible_wait:.2f}"], loc="best")

    single_panel(ax_opt, "optimistic")
    single_panel(ax_exp, "expected")
    single_panel(ax_pes, "pessimistic")

    fig.suptitle("MOFGPM sensitivity: feasibility boundary and satisfaction under uncertainty", y=1.01, fontsize=12)

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_png
