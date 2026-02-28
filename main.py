# ============================================================
# FULL, SINGLE-SNIPPET SCRIPT
# MOFGPM (Max–Min Fuzzy Goal Programming) + Jiménez linearization
# Run ONE fuzzy scenario with:
#   (A) fixed staff (fixed cost)  OR
#   (B) random staff samples to study sensitivity of cost/wait/lambda
#
# Requirements: numpy, pandas, matplotlib, scipy
# ============================================================

import json
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats


def _env(name, default, cast):
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return cast(raw)


def _env_bool(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}

# -------------------------
# USER SETTINGS (EDIT HERE)
# -------------------------

# Choose ONE fuzzy scenario to solve:
FUZZY_SCENARIO = _env("FUZZY_SCENARIO", "expected", str)  # "optimistic" | "expected" | "pessimistic"

# Choose run mode:
RUN_MODE = _env("RUN_MODE", "fixed", str)  # "fixed" | "random"

# If RUN_MODE == "fixed":
FIXED_STAFF = {
    "Doctor": _env("FIXED_DOCTOR", 3, int),
    "Nurse": _env("FIXED_NURSE", 3, int),
    "Assistant": _env("FIXED_ASSISTANT", 6, int),
    "Specialist": _env("FIXED_SPECIALIST", 1, int),
}

# If RUN_MODE == "random":
N_STAFF_SAMPLES = _env("N_STAFF_SAMPLES", 25, int)
STAFF_BOUNDS = {
    "Doctor": (_env("BOUND_DOCTOR_LO", 1, int), _env("BOUND_DOCTOR_HI", 6, int)),
    "Nurse": (_env("BOUND_NURSE_LO", 1, int), _env("BOUND_NURSE_HI", 8, int)),
    "Assistant": (_env("BOUND_ASSISTANT_LO", 1, int), _env("BOUND_ASSISTANT_HI", 10, int)),
    "Specialist": (_env("BOUND_SPECIALIST_LO", 0, int), _env("BOUND_SPECIALIST_HI", 2, int)),
}
RANDOM_STAFF_SEED = _env("RANDOM_STAFF_SEED", 202600, int)

# Day / patient generation:
NUM_PATIENTS = _env("NUM_PATIENTS", 36, int)
DATA_SEED = _env("DATA_SEED", 42, int)
T_MAX_DAY = _env("T_MAX_DAY", 1440.0, float)
N_SAMPLES_PER_LEVEL = _env("N_SAMPLES_PER_LEVEL", 25, int)
FORCE_NUM_PATIENTS = _env_bool("FORCE_NUM_PATIENTS", False)
MAX_PATIENT_GEN_TRIES = _env("MAX_PATIENT_GEN_TRIES", 200, int)

# GA params:
POP_SIZE = _env("POP_SIZE", 200, int)
GENERATIONS = _env("GENERATIONS", 120, int)
MUT_RATE = _env("MUT_RATE", 0.2, float)
GA_SEED_BASE = _env("GA_SEED_BASE", 202600, int)

# MOFGPM goals (piecewise linear membership, minimization):
GOAL_COST = _env("GOAL_COST", 650.0, float)
MAX_COST = _env("MAX_COST", 22750.0, float)
GOAL_WAIT = _env("GOAL_WAIT", 50.0, float)
MAX_WAIT = _env("MAX_WAIT", 200000.0, float)  # Can sweep this if you want

# Triage waiting-time hard limits (minutes) for early activities (j<=4)
TRIAGE_WAIT_LIMIT = {
    1: _env("TRIAGE_LIMIT_1", 0.0, float),
    2: _env("TRIAGE_LIMIT_2", 30.0, float),
    3: _env("TRIAGE_LIMIT_3", 240.0, float),
    4: _env("TRIAGE_LIMIT_4", 720.0, float),
    5: _env("TRIAGE_LIMIT_5", float("inf"), float),
}

# Penalties
TRIAGE_PENALTY = _env("TRIAGE_PENALTY", 1000.0, float)
INFEAS_BASE_PENALTY = _env("INFEAS_BASE_PENALTY", 1e6, float)

# Jimenez center for exponential linearization (use "median" to match your current approach)
EXP_CENTER = _env("EXP_CENTER", "median", str)  # "median" | "mean"

OUTPUT_DIR = Path(_env("OUTPUT_DIR", ".", str)).resolve()
SAVE_PLOTS = _env_bool("SAVE_PLOTS", True)
SHOW_PLOTS = _env_bool("SHOW_PLOTS", False)
PLOT_STYLE = _env("PLOT_STYLE", "ggplot", str)
PLOT_DPI = _env("PLOT_DPI", 300, int)

try:
    plt.style.use(PLOT_STYLE)
except OSError:
    plt.style.use("ggplot")

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "#f7f7f7",
        "axes.edgecolor": "#333333",
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.35,
        "savefig.facecolor": "white",
    }
)


# ============================================================
# 1) Distributions for activities
# ============================================================

dists = [
    {"name": "R&T",  "type": "uniform",     "a": 3.0,  "b": 5.0},

    {"name": "CPR",  "type": "exponential", "scale": 30.0},
    {"name": "SAUV", "type": "exponential", "scale": 30.0},
    {"name": "GP1",  "type": "exponential", "scale": 9.0},
    {"name": "T1",   "type": "exponential", "scale": 30.0},
    {"name": "BE",   "type": "exponential", "scale": 10.0},
    {"name": "T2",   "type": "exponential", "scale": 30.0},

    {"name": "R",    "type": "triangular",  "L": 15.0, "M": 30.0, "U": 45.0},

    {"name": "SA",   "type": "exponential", "scale": 5.0},

    {"name": "MS",   "type": "triangular",  "L": 10.0, "M": 15.0, "U": 30.0},

    {"name": "GP2",  "type": "exponential", "scale": 9.0},
    {"name": "T3",   "type": "exponential", "scale": 15.0},
]

activity_labels = [d["name"] for d in dists]

# ============================================================
# 2) Helpers: median/mean + Jiménez fuzzy + EI defuzzification
# ============================================================

def tri_median(L, M, U):
    if M >= (L + U) / 2.0:
        return L + np.sqrt((U - L) * (M - L) / 2.0)
    else:
        return U - np.sqrt((U - L) * (U - M) / 2.0)

def summarize_dist(d):
    typ = d["type"]
    if typ == "uniform":
        mean = (d["a"] + d["b"]) / 2.0
        median = mean
        return mean, median
    if typ == "exponential":
        s = d["scale"]
        mean = s
        median = s * np.log(2.0)
        return mean, median
    if typ == "triangular":
        L, M, U = d["L"], d["M"], d["U"]
        mean = (L + M + U) / 3.0
        median = tri_median(L, M, U)
        return mean, median
    raise ValueError("Unknown distribution type")

def jimenez_exponential_fuzzy(a, sigma):
    """
    Jiménez-style linearization (your approach):
      EI interval: [a, a + sigma]
      Center value (EV): a + sigma/2
    We interpret a fuzzy triple (p,m,o) = (a, a+sigma/2, a+sigma).
    """
    p = float(a)
    o = float(a + sigma)
    m = float(a + sigma / 2.0)
    return p, m, o

def EI_defuzz(p, m, o):
    """
    Paper-consistent defuzzification:
      EI(AT~) = (p + 2m + o)/4
    """
    return (float(p) + 2.0 * float(m) + float(o)) / 4.0

def build_fuzzy_triples_and_scenarios(dists, exp_center="median"):
    """
    Returns:
      durations_by_scenario["optimistic"/"expected"/"pessimistic"][activity] -> duration
    Construction:
      - Uniform: (p,m,o)=(a,(a+b)/2,b)
      - Triangular: (p,m,o)=(L,M,U)  (keep TRUE mode M)
      - Exponential: Jiménez (p,m,o)=(a, a+sigma/2, a+sigma), with a=median or mean
      - "expected" scenario uses EI_defuzz(p,m,o)
    """
    dur = {"optimistic": {}, "expected": {}, "pessimistic": {}}

    for d in dists:
        name, typ = d["name"], d["type"]
        mean, median = summarize_dist(d)
        a_center = median if exp_center == "median" else mean

        if typ == "uniform":
            p = float(d["a"])
            o = float(d["b"])
            m = float((d["a"] + d["b"]) / 2.0)
        elif typ == "triangular":
            p = float(d["L"])
            m = float(d["M"])
            o = float(d["U"])
        elif typ == "exponential":
            sigma = float(d["scale"])
            p, m, o = jimenez_exponential_fuzzy(a_center, sigma)
        else:
            raise ValueError(f"Unknown type: {typ}")

        dur["optimistic"][name]  = p
        dur["expected"][name]    = EI_defuzz(p, m, o)
        dur["pessimistic"][name] = o

    return dur

durations_by_scenario = build_fuzzy_triples_and_scenarios(dists, exp_center=EXP_CENTER)

# ============================================================
# 3) Model inputs TH_hj and R_cj 
# ============================================================

TH_hj = np.array([
    [0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1],
], dtype=float)

R_cj = np.array([
    [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
    [1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1],
    [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
], dtype=float)

# ============================================================
# 4) Arrival generator 
# ============================================================

def _rvs_expon(loc, scale, size, rng):
    return stats.expon.rvs(loc=loc, scale=scale, size=size, random_state=rng)

def _rvs_gamma(shape, loc, scale, size, rng):
    return stats.gamma.rvs(a=shape, loc=loc, scale=scale, size=size, random_state=rng)

def _rvs_lognorm(s, loc, scale, size, rng):
    return stats.lognorm.rvs(s=s, loc=loc, scale=scale, size=size, random_state=rng)

def generate_triage_simulation(n_samples=25, seed=None):
    rng = np.random.default_rng(seed)

    out = {}
    out["Triage I"]   = _rvs_expon(loc=159.66667, scale=21094.04259, size=n_samples, rng=rng)
    out["Triage II"]  = _rvs_gamma(shape=0.54286, loc=2.43333, scale=2543.79781, size=n_samples, rng=rng)
    out["Triage III"] = _rvs_lognorm(s=1.33412, loc=1.96094, scale=14.31896, size=n_samples, rng=rng)

    # Triage IV mixture lognormal without loc
    weights = np.array([0.45546588, 0.54453412], dtype=float)
    means   = np.array([7.460967112176844, 4.2544537009688], dtype=float)
    sigmas  = np.array([0.6885149071957068, 1.4155979087095545], dtype=float)

    comp = rng.choice([0, 1], size=n_samples, p=weights)
    samples_iv = np.empty(n_samples, dtype=float)
    for k in range(n_samples):
        c = comp[k]
        samples_iv[k] = rng.lognormal(mean=means[c], sigma=sigmas[c])
    out["Triage IV"] = samples_iv

    out["Triage V"] = _rvs_expon(loc=8.08333, scale=17909.49815, size=n_samples, rng=rng)

    return pd.DataFrame(out)

def per_level_cumsum_union_sort(df_sim: pd.DataFrame) -> pd.DataFrame:
    df_abs = df_sim.cumsum(axis=0)
    df_long = df_abs.melt(var_name="Triage Level", value_name="Arrival Time")
    df_long = df_long.sort_values("Arrival Time").reset_index(drop=True)
    return df_long

def assign_arrivals_and_triage_single_day(
    num_patients,
    seed=1952,
    t_max=1440.0,
    n_samples_per_level=36,
    enforce_exact=False,
    max_tries=200,
):
    rng = np.random.default_rng(seed)
    map_level = {"Triage I": 1, "Triage II": 2, "Triage III": 3, "Triage IV": 4, "Triage V": 5}
    best_df_day = None
    best_seed_day = None
    chosen_df_day = None
    chosen_seed_day = None

    for _ in range(max(1, int(max_tries))):
        seed_day = int(rng.integers(0, 2**31 - 1))
        df_sim = generate_triage_simulation(n_samples=n_samples_per_level, seed=seed_day)
        df_ord = per_level_cumsum_union_sort(df_sim)
        df_day = df_ord[df_ord["Arrival Time"] <= t_max].copy().reset_index(drop=True)

        if best_df_day is None or len(df_day) > len(best_df_day):
            best_df_day = df_day
            best_seed_day = seed_day

        if len(df_day) >= num_patients:
            chosen_df_day = df_day
            chosen_seed_day = seed_day
            break

    if chosen_df_day is None:
        if best_df_day is None or len(best_df_day) == 0:
            raise ValueError(f"No patients generated with t<= {t_max} after {max_tries} tries.")
        if enforce_exact:
            raise ValueError(
                f"Could not generate requested NUM_PATIENTS={num_patients} with t<= {t_max} "
                f"after {max_tries} tries (best={len(best_df_day)}, seed_day={best_seed_day}). "
                "Increase T_MAX_DAY and/or N_SAMPLES_PER_LEVEL."
            )
        chosen_df_day = best_df_day
        chosen_seed_day = best_seed_day

    take_n = min(num_patients, len(chosen_df_day))
    if take_n < num_patients:
        print(
            f"[Warning] Requested {num_patients} patients but only {take_n} were available "
            f"within t<= {t_max} (seed_day={chosen_seed_day})."
        )

    df_take = chosen_df_day.iloc[:take_n].copy().reset_index(drop=True)
    df_take["Patient"] = [f"P{i+1}" for i in range(take_n)]

    arrival_times, triage_level, patient_route_idx = {}, {}, {}
    for i in range(take_n):
        p = df_take.loc[i, "Patient"]
        lvl_str = str(df_take.loc[i, "Triage Level"])
        lvl = int(map_level[lvl_str])
        arr = float(df_take.loc[i, "Arrival Time"])
        arrival_times[p] = arr
        triage_level[p] = lvl
        patient_route_idx[p] = lvl - 1

    return arrival_times, triage_level, patient_route_idx, df_take

# ============================================================
# 5) ED data generator
# ============================================================

class ED_Data_Generator:
    def __init__(
        self,
        TH_hj,
        R_cj,
        activity_labels,
        duration_by_activity,
        arrival_times,
        triage_level,
        patient_route_idx,
        resource_names=None,
        sc_cost=None,
    ):
        self.TH_hj = np.array(TH_hj, dtype=float)
        self.R_cj = np.array(R_cj, dtype=float)

        self.H = self.TH_hj.shape[0]
        self.J = self.TH_hj.shape[1]
        if self.R_cj.shape[1] != self.J:
            raise ValueError("R_cj must have the same number of columns as TH_hj.")

        if resource_names is None:
            resource_names = ["Doctor", "Nurse", "Assistant", "Specialist"]
        if len(resource_names) != self.H:
            raise ValueError("resource_names must have length H.")
        self.resources = list(resource_names)

        if sc_cost is None:
            sc_cost = {"Doctor": 200, "Nurse": 100, "Assistant": 50, "Specialist": 300}
        for r in self.resources:
            if r not in sc_cost:
                raise ValueError(f"Missing cost for resource '{r}'.")
        self.sc_cost = dict(sc_cost)

        self.activity_labels = list(activity_labels)
        self.duration_by_activity = dict(duration_by_activity)

        self.arrival_times = dict(arrival_times)
        self.triage_level = dict(triage_level)
        self.patient_route_idx = dict(patient_route_idx)

        self.patients = sorted(list(self.arrival_times.keys()), key=lambda x: int(x[1:]))
        self.num_patients = len(self.patients)

        self.activities = {}
        self.task_j = {}
        self.reqs = {}
        self.eff_duration = {}
        self.all_tasks = []

        self._generate_clinical_pathways()

    def _route_activities_from_level(self, c):
        js = [j for j in range(self.J) if self.R_cj[c, j] > 0.5]
        js.sort()
        return js

    def _generate_clinical_pathways(self):
        C = self.R_cj.shape[0]
        for p in self.patients:
            c = int(self.patient_route_idx[p])
            if not (0 <= c < C):
                raise ValueError(f"Invalid route index for {p}: c={c}")

            act_js = self._route_activities_from_level(c)
            if len(act_js) == 0:
                raise ValueError(f"Route c={c} has no activities.")

            p_acts = [self.activity_labels[j] for j in act_js]
            self.activities[p] = p_acts

            for j_idx, act_label in zip(act_js, p_acts):
                task_id = (p, act_label)
                self.all_tasks.append(task_id)
                self.task_j[task_id] = j_idx + 1
                self.eff_duration[task_id] = float(self.duration_by_activity[act_label])

                for h, rname in enumerate(self.resources):
                    self.reqs[(p, act_label, rname)] = 1 if self.TH_hj[h, j_idx] > 0.5 else 0

# ============================================================
# 6) MOFGPM membership (piecewise linear) + GA scheduler
# ============================================================

def mu_minimize(x, goal, xmax):
    x = float(x); goal = float(goal); xmax = float(xmax)
    if xmax <= goal:
        return 1.0 if x <= goal else 0.0
    if x <= goal:
        return 1.0
    if x >= xmax:
        return 0.0
    return (xmax - x) / (xmax - goal)

class RobustGAScheduler:
    def __init__(self, data, staff_limits, pop_size=50, generations=50, mutation_rate=0.2,
                 triage_wait_limit=None):
        self.data = data
        self.staff_limits = dict(staff_limits)
        self.pop_size = int(pop_size)
        self.generations = int(generations)
        self.mutation_rate = float(mutation_rate)
        self.num_genes = len(data.all_tasks)

        for r in self.data.resources:
            if r not in self.staff_limits:
                raise ValueError(f"Missing staff_limits for '{r}'")

        if triage_wait_limit is None:
            triage_wait_limit = TRIAGE_WAIT_LIMIT
        self.triage_wait_limit = dict(triage_wait_limit)

    def _required_resources(self, p, act):
        return [r for r in self.data.resources if self.data.reqs.get((p, act, r), 0) == 1]

    def _wait_sum_j_leq(self, schedule, p, jmax=4):
        acts_in_route = self.data.activities[p]
        early_acts = [act for act in acts_in_route if self.data.task_j[(p, act)] <= jmax]
        if not early_acts:
            return 0.0

        total = 0.0
        t0 = (p, early_acts[0])
        total += schedule[t0] - self.data.arrival_times[p]

        for k in range(1, len(early_acts)):
            curr = (p, early_acts[k])
            prev = (p, early_acts[k - 1])
            prev_end = schedule[prev] + self.data.eff_duration[prev]
            total += schedule[curr] - prev_end

        return float(total)

    def _decode_schedule(self, task_permutation):
        priority_map = {task: i for i, task in enumerate(task_permutation)}

        unscheduled = set(self.data.all_tasks)
        schedule = {}

        res_free_times = {r: [0.0] * int(self.staff_limits[r]) for r in self.data.resources}
        completed = set()

        while unscheduled:
            ready = []
            for t in unscheduled:
                p, act = t
                idx = self.data.activities[p].index(act)

                if idx == 0:
                    ready.append((t, self.data.arrival_times[p]))
                else:
                    prev = (p, self.data.activities[p][idx - 1])
                    if prev in completed:
                        ready.append((t, schedule[prev] + self.data.eff_duration[prev]))

            if not ready:
                break

            ready.sort(key=lambda x: priority_map[x[0]])

            progressed = False
            for task, min_start in ready:
                p, act = task
                dur = self.data.eff_duration[task]
                req = self._required_resources(p, act)

                if not req:
                    schedule[task] = min_start
                    unscheduled.remove(task)
                    completed.add(task)
                    progressed = True
                    break

                chosen = {}
                candidate_starts = [min_start]
                feasible = True

                for r in req:
                    if len(res_free_times[r]) == 0:
                        feasible = False
                        break
                    best_st = None
                    best_u = None
                    for u, free_t in enumerate(res_free_times[r]):
                        st = max(float(free_t), float(min_start))
                        if best_st is None or st < best_st:
                            best_st, best_u = st, u
                    if best_st is None:
                        feasible = False
                        break
                    chosen[r] = best_u
                    candidate_starts.append(best_st)

                if not feasible:
                    continue

                st = max(candidate_starts)
                schedule[task] = st
                for r in req:
                    u = chosen[r]
                    res_free_times[r][u] = st + dur

                unscheduled.remove(task)
                completed.add(task)
                progressed = True
                break

            if not progressed:
                break

        return schedule

    def evaluate_mofgpm(self, schedule, goal_cost, max_cost, goal_wait, max_wait,
                       triage_penalty=TRIAGE_PENALTY, infeas_base_penalty=INFEAS_BASE_PENALTY):
        # cost from staff plan
        cost = sum(self.data.sc_cost[r] * self.staff_limits[r] for r in self.data.resources)

        # total waiting time (your current metric)
        total_wait = 0.0
        for p in self.data.patients:
            acts = self.data.activities[p]
            t0 = (p, acts[0])
            total_wait += schedule[t0] - self.data.arrival_times[p]
            for i in range(1, len(acts)):
                curr = (p, acts[i])
                prev = (p, acts[i - 1])
                prev_end = schedule[prev] + self.data.eff_duration[prev]
                total_wait += (schedule[curr] - prev_end)

        # triage hard constraint on early activities
        triage_violation = 0.0
        for p in self.data.patients:
            tri = self.data.triage_level[p]
            limit = self.triage_wait_limit.get(tri, float("inf"))
            w_early = self._wait_sum_j_leq(schedule, p, jmax=4)
            if w_early > limit:
                triage_violation += (w_early - limit)

        # memberships (piecewise linear)
        mu_c = mu_minimize(cost, goal_cost, max_cost)
        mu_w = mu_minimize(total_wait, goal_wait, max_wait)
        lam = min(mu_c, mu_w)

        feas = (mu_c > 0.0) and (mu_w > 0.0) and (triage_violation <= 0.0)

        if triage_violation > 0.0:
            fitness = -infeas_base_penalty - triage_penalty * triage_violation
        else:
            fitness = lam

        return fitness, lam, cost, total_wait, mu_c, mu_w, feas, triage_violation

    def solve(self, goal_cost, max_cost, goal_wait, max_wait, seed=0):
        random.seed(seed)
        np.random.seed(seed)

        base = list(self.data.all_tasks)

        population = []
        for _ in range(self.pop_size):
            p = base[:]
            random.shuffle(p)
            population.append(p)

        best = None
        best_fit = -float("inf")

        for _gen in range(self.generations):
            scores = []
            eval_cache = []

            for chrom in population:
                sched = self._decode_schedule(chrom)

                if len(sched) != len(self.data.all_tasks):
                    # not fully scheduled -> heavy penalty
                    fit = -1e9
                    lam = -1.0
                    cost = sum(self.data.sc_cost[r] * self.staff_limits[r] for r in self.data.resources)
                    tw = float("inf")
                    mu_c = 0.0
                    mu_w = 0.0
                    feas = False
                    vtri = float("inf")
                else:
                    fit, lam, cost, tw, mu_c, mu_w, feas, vtri = self.evaluate_mofgpm(
                        sched, goal_cost, max_cost, goal_wait, max_wait
                    )

                scores.append(fit)
                eval_cache.append((sched, fit, lam, cost, tw, mu_c, mu_w, feas, vtri))

                if fit > best_fit:
                    best_fit = fit
                    best = {
                        "sched": sched,
                        "fitness": float(fit),
                        "lambda": float(lam),
                        "cost": float(cost),
                        "wait": float(tw),
                        "mu_cost": float(mu_c),
                        "mu_wait": float(mu_w),
                        "feas": bool(feas),
                        "triage_violation": float(vtri),
                    }

            # selection (sort desc)
            idx = np.argsort(scores)[::-1]
            population = [population[i] for i in idx]

            elite = max(1, int(self.pop_size * 0.1))
            new_pop = population[:elite]

            # crossover + mutation
            while len(new_pop) < self.pop_size:
                p1 = population[random.randrange(0, max(2, self.pop_size // 2))]
                p2 = population[random.randrange(0, max(2, self.pop_size // 2))]

                start, end = sorted(random.sample(range(self.num_genes), 2))
                child = [None] * self.num_genes
                child[start:end] = p1[start:end]

                seg = set(p1[start:end])
                ptr = 0
                for gene in p2:
                    if gene not in seg:
                        while child[ptr] is not None:
                            ptr += 1
                        child[ptr] = gene

                if random.random() < self.mutation_rate:
                    i1, i2 = random.sample(range(self.num_genes), 2)
                    child[i1], child[i2] = child[i2], child[i1]

                new_pop.append(child)

            population = new_pop

        return best

# ============================================================
# 7) Run ONE scenario with fixed staff OR random staff sensitivity
# ============================================================

# Generate the day once (same patients)
arrival_times_day, triage_level_day, route_idx_day, df_take = assign_arrivals_and_triage_single_day(
    num_patients=NUM_PATIENTS,
    seed=DATA_SEED,
    t_max=T_MAX_DAY,
    n_samples_per_level=N_SAMPLES_PER_LEVEL,
    enforce_exact=FORCE_NUM_PATIENTS,
    max_tries=MAX_PATIENT_GEN_TRIES,
)

# Build ED data for selected fuzzy scenario
data = ED_Data_Generator(
    TH_hj=TH_hj,
    R_cj=R_cj,
    activity_labels=activity_labels,
    duration_by_activity=durations_by_scenario[FUZZY_SCENARIO],
    arrival_times=arrival_times_day,
    triage_level=triage_level_day,
    patient_route_idx=route_idx_day,
)

def sample_staff(bounds, rng):
    staff = {}
    for r, (lo, hi) in bounds.items():
        staff[r] = int(rng.integers(lo, hi + 1))
    return staff


def _extract_wait_metrics(data_obj, schedule, sample_id, scenario, is_feasible):
    rows = []
    for p in data_obj.patients:
        acts = data_obj.activities[p]
        first_task = (p, acts[0])
        if first_task not in schedule:
            rows.append(
                {
                    "scenario": scenario,
                    "sample_id": int(sample_id),
                    "patient": p,
                    "triage": int(data_obj.triage_level[p]),
                    "first_wait": np.nan,
                    "early_wait_j_le_4": np.nan,
                    "schedule_complete": False,
                    "feasible": bool(is_feasible),
                }
            )
            continue

        first_wait = float(schedule[first_task] - data_obj.arrival_times[p])

        early_acts = [act for act in acts if data_obj.task_j[(p, act)] <= 4]
        early_wait = 0.0
        schedule_complete = True
        if early_acts:
            t0 = (p, early_acts[0])
            if t0 not in schedule:
                schedule_complete = False
                early_wait = np.nan
            else:
                early_wait += schedule[t0] - data_obj.arrival_times[p]
            for k in range(1, len(early_acts)):
                curr = (p, early_acts[k])
                prev = (p, early_acts[k - 1])
                if curr not in schedule or prev not in schedule:
                    schedule_complete = False
                    early_wait = np.nan
                    break
                prev_end = schedule[prev] + data_obj.eff_duration[prev]
                early_wait += schedule[curr] - prev_end

        rows.append(
            {
                "scenario": scenario,
                "sample_id": int(sample_id),
                "patient": p,
                "triage": int(data_obj.triage_level[p]),
                "first_wait": float(first_wait),
                "early_wait_j_le_4": float(early_wait) if np.isfinite(early_wait) else np.nan,
                "schedule_complete": bool(schedule_complete),
                "feasible": bool(is_feasible),
            }
        )
    return rows


def _extract_schedule_rows(data_obj, schedule, sample_id, scenario, is_feasible, staff):
    rows = []
    for (p, act), start in schedule.items():
        duration = float(data_obj.eff_duration[(p, act)])
        end = float(start + duration)
        req_flags = {r: int(data_obj.reqs[(p, act, r)]) for r in data_obj.resources}
        req_active = [r for r, v in req_flags.items() if v == 1]
        row = {
            "scenario": scenario,
            "sample_id": int(sample_id),
            "patient": p,
            "triage": int(data_obj.triage_level[p]),
            "activity": act,
            "task_j": int(data_obj.task_j[(p, act)]),
            "start": float(start),
            "end": end,
            "duration": duration,
            "feasible": bool(is_feasible),
            "Doctor": int(staff.get("Doctor", 0)),
            "Nurse": int(staff.get("Nurse", 0)),
            "Assistant": int(staff.get("Assistant", 0)),
            "Specialist": int(staff.get("Specialist", 0)),
            "required_resources": ",".join(req_active),
        }
        for r, v in req_flags.items():
            row[f"req_{r}"] = int(v)
        rows.append(row)
    rows.sort(key=lambda x: (x["sample_id"], x["start"], x["patient"], x["task_j"]))
    return rows


def _save_plot(fig, filename):
    out = OUTPUT_DIR / filename
    fig.tight_layout()
    if SAVE_PLOTS:
        fig.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_cost_wait_lambda_map(df):
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    feas_mask = df["feasible"].astype(bool)
    lam = df["lambda"].astype(float).clip(lower=0.0, upper=1.0)
    sizes = 45.0 + 220.0 * lam.fillna(0.0)

    sc = None
    if feas_mask.any():
        sc = ax.scatter(
            df.loc[feas_mask, "cost"],
            df.loc[feas_mask, "wait"],
            c=lam.loc[feas_mask],
            s=sizes.loc[feas_mask],
            cmap="viridis",
            marker="o",
            edgecolors="#202020",
            linewidths=0.4,
            alpha=0.9,
            label="Feasible",
        )

    if (~feas_mask).any():
        ax.scatter(
            df.loc[~feas_mask, "cost"],
            df.loc[~feas_mask, "wait"],
            c="#9e9e9e",
            s=70,
            marker="X",
            edgecolors="#202020",
            linewidths=0.4,
            alpha=0.9,
            label="Infeasible",
        )

    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("λ (min satisfaction)")

    ax.set_xlabel("Cost")
    ax.set_ylabel("Total waiting time")
    ax.legend(loc="best")
    _save_plot(fig, "plot_cost_wait_scatter_lambda.png")


def plot_membership_decomposition(df):
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    x_col = "wmax" if "wmax" in df.columns and df["wmax"].nunique(dropna=True) > 1 else "sample_id"
    d = df.sort_values(x_col).reset_index(drop=True)

    ax.plot(d[x_col], d["mu_cost"], marker="o", markersize=5, color="tab:blue", label=r"$\mu_{Cost}$")
    ax.plot(d[x_col], d["mu_wait"], marker="s", markersize=5, color="tab:orange", label=r"$\mu_{Wait}$")
    ax.plot(d[x_col], d["lambda"], marker="^", markersize=5, color="tab:green", linestyle="--", label=r"$\lambda$")

    if x_col == "wmax":
        ax.set_xlabel(r"$W_{max}$ (wait tolerance)")
    else:
        ax.set_xlabel("Staff sample index")
    ax.set_ylabel("Satisfaction / membership")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="best")
    _save_plot(fig, "plot_membership_decomposition.png")


def plot_feasibility_heatmap(df):
    if not {"Doctor", "Nurse"}.issubset(df.columns):
        return

    rows = []
    for (nurse, doctor), g in df.groupby(["Nurse", "Doctor"]):
        feasible_vals = g.loc[g["feasible"], "lambda"]
        best_lambda = float(feasible_vals.max()) if not feasible_vals.empty else np.nan
        rows.append(
            {
                "Nurse": int(nurse),
                "Doctor": int(doctor),
                "best_lambda_feasible": best_lambda,
                "feasibility_ratio": float(g["feasible"].mean()),
            }
        )
    grouped = pd.DataFrame(rows)

    pivot = grouped.pivot(index="Nurse", columns="Doctor", values="best_lambda_feasible")
    if pivot.empty:
        return

    y_vals = sorted(pivot.index.tolist())
    x_vals = sorted(pivot.columns.tolist())
    arr = pivot.reindex(index=y_vals, columns=x_vals).to_numpy(dtype=float)
    masked = np.ma.masked_invalid(arr)

    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad("#d9d9d9")

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    im = ax.imshow(masked, origin="lower", cmap=cmap, aspect="auto", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Best feasible λ")

    ax.set_xlabel("Doctors")
    ax.set_ylabel("Nurses")
    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels(x_vals)
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels(y_vals)

    for i in range(len(y_vals)):
        for j in range(len(x_vals)):
            val = arr[i, j]
            txt = "NA" if np.isnan(val) else f"{val:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="#111111")

    _save_plot(fig, "plot_feasibility_heatmap_doctor_nurse.png")


def plot_triage_wait_distribution(triage_df):
    if triage_df.empty:
        return

    use_df = triage_df[triage_df["feasible"]].copy()
    if use_df.empty:
        use_df = triage_df.copy()

    triages = sorted(use_df["triage"].dropna().astype(int).unique().tolist())
    if not triages:
        return

    labels = [f"T{t}" for t in triages]
    values = [
        use_df.loc[use_df["triage"] == t, "first_wait"].astype(float).dropna().to_numpy()
        for t in triages
    ]
    if not any(v.size > 0 for v in values):
        return

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    bp = ax.boxplot(values, tick_labels=labels, patch_artist=True, widths=0.6, showfliers=False)
    fill_colors = ["#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6"]
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(fill_colors[i % len(fill_colors)])
        box.set_alpha(0.8)
    for med in bp["medians"]:
        med.set_color("#202020")
        med.set_linewidth(1.6)

    ax.set_xlabel("Triage class")
    ax.set_ylabel("Time to first provider (minutes)")
    _save_plot(fig, "plot_triage_first_wait_boxplot.png")

    # ECDF companion plot
    fig2, ax2 = plt.subplots(figsize=(8.8, 5.6))
    palette = {
        1: "tab:blue",
        2: "tab:orange",
        3: "tab:green",
        4: "tab:red",
        5: "tab:purple",
    }
    for t in triages:
        x = np.sort(use_df.loc[use_df["triage"] == t, "first_wait"].astype(float).dropna().to_numpy())
        if x.size == 0:
            continue
        y = np.arange(1, x.size + 1) / x.size
        ax2.step(x, y, where="post", color=palette.get(t, "tab:gray"), label=f"T{t}")

    ax2.set_xlabel("Time to first provider (minutes)")
    ax2.set_ylabel("ECDF")
    ax2.set_ylim(0.0, 1.0)
    ax2.legend(loc="lower right")
    _save_plot(fig2, "plot_triage_first_wait_ecdf.png")

records = []
triage_wait_records = []
schedule_records = []
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

run_settings = {
    "FUZZY_SCENARIO": FUZZY_SCENARIO,
    "RUN_MODE": RUN_MODE,
    "FIXED_STAFF": FIXED_STAFF,
    "N_STAFF_SAMPLES": N_STAFF_SAMPLES,
    "STAFF_BOUNDS": STAFF_BOUNDS,
    "RANDOM_STAFF_SEED": RANDOM_STAFF_SEED,
    "NUM_PATIENTS": NUM_PATIENTS,
    "DATA_SEED": DATA_SEED,
    "T_MAX_DAY": T_MAX_DAY,
    "N_SAMPLES_PER_LEVEL": N_SAMPLES_PER_LEVEL,
    "FORCE_NUM_PATIENTS": FORCE_NUM_PATIENTS,
    "MAX_PATIENT_GEN_TRIES": MAX_PATIENT_GEN_TRIES,
    "POP_SIZE": POP_SIZE,
    "GENERATIONS": GENERATIONS,
    "MUT_RATE": MUT_RATE,
    "GA_SEED_BASE": GA_SEED_BASE,
    "GOAL_COST": GOAL_COST,
    "MAX_COST": MAX_COST,
    "GOAL_WAIT": GOAL_WAIT,
    "MAX_WAIT": MAX_WAIT,
    "TRIAGE_WAIT_LIMIT": TRIAGE_WAIT_LIMIT,
    "TRIAGE_PENALTY": TRIAGE_PENALTY,
    "INFEAS_BASE_PENALTY": INFEAS_BASE_PENALTY,
    "EXP_CENTER": EXP_CENTER,
}

with open(OUTPUT_DIR / "settings.json", "w", encoding="utf-8") as fp:
    json.dump(run_settings, fp, indent=2)

if RUN_MODE == "fixed":
    staff_list = [FIXED_STAFF]
elif RUN_MODE == "random":
    rng = np.random.default_rng(RANDOM_STAFF_SEED)
    staff_list = [sample_staff(STAFF_BOUNDS, rng) for _ in range(N_STAFF_SAMPLES)]
else:
    raise ValueError("RUN_MODE must be 'fixed' or 'random'.")

print("\n" + "=" * 80)
print(f"Scenario: {FUZZY_SCENARIO.upper()} | Mode: {RUN_MODE.upper()}")
print(f"Patients used: {data.num_patients} (t<= {T_MAX_DAY})")
print("=" * 80)

for idx, staff in enumerate(staff_list, 1):
    solver = RobustGAScheduler(
        data=data,
        staff_limits=staff,
        pop_size=POP_SIZE,
        generations=GENERATIONS,
        mutation_rate=MUT_RATE,
        triage_wait_limit=TRIAGE_WAIT_LIMIT
    )

    seed = GA_SEED_BASE + idx
    t0 = time.time()
    best = solver.solve(GOAL_COST, MAX_COST, GOAL_WAIT, MAX_WAIT, seed=seed)
    dt = time.time() - t0

    rec = {
        "scenario": FUZZY_SCENARIO,
        "sample_id": idx,
        "Doctor": staff["Doctor"],
        "Nurse": staff["Nurse"],
        "Assistant": staff["Assistant"],
        "Specialist": staff["Specialist"],
        "wmax": MAX_WAIT,
        "cost": best["cost"],
        "wait": best["wait"],
        "mu_cost": best["mu_cost"],
        "mu_wait": best["mu_wait"],
        "lambda": best["lambda"],
        "feasible": best["feas"],
        "triage_violation": best["triage_violation"],
        "ga_time_s": dt,
    }
    records.append(rec)
    triage_wait_records.extend(
        _extract_wait_metrics(data, best["sched"], idx, FUZZY_SCENARIO, rec["feasible"])
    )
    schedule_records.extend(
        _extract_schedule_rows(data, best["sched"], idx, FUZZY_SCENARIO, rec["feasible"], staff)
    )

    status = "FEASIBLE" if rec["feasible"] else "INFEASIBLE"
    print(f"[{idx:02d}/{len(staff_list)}] staff={staff} -> "
          f"cost={rec['cost']:.1f}, wait={rec['wait']:.1f}, lambda={rec['lambda']:.4f}, {status}, time={dt:.1f}s")

df = pd.DataFrame(records)
df = df.sort_values(["feasible", "lambda"], ascending=[False, False]).reset_index(drop=True)
df.to_csv(OUTPUT_DIR / "results.csv", index=False)

print("\nTop results (sorted by feasible then lambda):")
print(df.head(10).to_string(index=False))

# ============================================================
# 8) Reporting plots (no in-figure titles; Word cross-references can be used)
# ============================================================

triage_df = pd.DataFrame(triage_wait_records)
triage_df.to_csv(OUTPUT_DIR / "triage_waits_by_sample.csv", index=False)
schedule_df = pd.DataFrame(schedule_records)
schedule_df.to_csv(OUTPUT_DIR / "schedule_by_sample.csv", index=False)

# Backward-compatible core scatters (aesthetic update, no titles)
fig, ax = plt.subplots(figsize=(8.8, 5.6))
ax.scatter(df["cost"], df["wait"], marker="o", color="tab:blue", edgecolors="#202020", linewidths=0.3, alpha=0.85)
ax.set_xlabel("Cost")
ax.set_ylabel("Total waiting time")
_save_plot(fig, "plot_cost_vs_wait.png")

fig, ax = plt.subplots(figsize=(8.8, 5.6))
ax.scatter(df["wait"], df["lambda"], marker="o", color="tab:green", edgecolors="#202020", linewidths=0.3, alpha=0.85)
ax.set_xlabel("Total waiting time")
ax.set_ylabel("λ (min satisfaction)")
ax.set_ylim(-0.02, 1.02)
_save_plot(fig, "plot_lambda_vs_wait.png")

fig, ax = plt.subplots(figsize=(8.8, 5.6))
ax.scatter(df["cost"], df["lambda"], marker="o", color="tab:purple", edgecolors="#202020", linewidths=0.3, alpha=0.85)
ax.set_xlabel("Cost")
ax.set_ylabel("λ (min satisfaction)")
ax.set_ylim(-0.02, 1.02)
_save_plot(fig, "plot_lambda_vs_cost.png")

# New manager-ready plots requested
plot_cost_wait_lambda_map(df)
plot_membership_decomposition(df)
plot_feasibility_heatmap(df)
plot_triage_wait_distribution(triage_df)

# ============================================================
# DONE.
# Notes:
# - This implements MOFGPM via piecewise-linear memberships and lambda = min(mu_cost, mu_wait).
# - Jiménez linearization is used for exponential activities (p,m,o)=(a, a+σ/2, a+σ).
# - "expected" scenario uses EI(p,m,o)=(p+2m+o)/4 per your paper.
# ============================================================
