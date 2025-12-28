import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

from Create_Dag import CreateDAG
from Dag_Env import DagSchedulingEnv
from scheduling_utils import run_policy_episode


# --- 1) Initialization and Data Loading ---
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path(sys.argv[0]).resolve().parent

DATASET_PATH = SCRIPT_DIR / "dag_dataset_a7_a12.csv"

TASK_ID = 0
SEED = 42
rng = np.random.RandomState(SEED)

# NOTE:
# Env itself loads DAG from CSV. We create one CreateDAG here only if you want to sanity-check
# but we will schedule inside env.
# (Keeping it minimal and consistent with your previous logic.)

# --- 2) Execute Scheduling Scenarios through Gymnasium Env ---
results = {
    "Makespan": {},
    "Total_Tardiness": {},
    "Total_Energy": {},
}

scheduling_scenarios = {
    "Dual A7 (Homogeneous)": ([0, 0], "EDF"),
    "Dual A12 (Homogeneous)": ([1, 1], "EDF"),
    "Random A7 + A12 (Heterogeneous)": ([0, 1], "RANDOM"),
    "HEFT A7 + A12 (Heterogeneous)": ([0, 1], "HEFT"),
}

for name, (processor_map, policy_name) in scheduling_scenarios.items():
    print(f"Running (Gym) scheduler: {name} ...")

    env = DagSchedulingEnv(
        csv_path=str(DATASET_PATH),
        row_index=TASK_ID,
        processor_map=processor_map,
        # reward weights don't affect heuristic policies, but env is ready for RL:
        reward_weights=(1.0, 1.0, 0.0),
        invalid_action_penalty=1.0,
        seed=SEED,
    )

    if policy_name == "RANDOM":
        assigned, start, finish, metrics, total_reward = run_policy_episode(env, policy_name, rng=rng)
    else:
        assigned, start, finish, metrics, total_reward = run_policy_episode(env, policy_name, rng=None)

    makespan = metrics["makespan"]
    total_tardiness = metrics["total_tardiness"]
    total_energy = metrics["total_energy"]

    results["Makespan"][name] = makespan
    results["Total_Tardiness"][name] = total_tardiness
    results["Total_Energy"][name] = total_energy

    print(
        f"   Makespan: {makespan:.2f}, "
        f"Total Tardiness: {total_tardiness:.2f}, "
        f"Total Energy: {total_energy:.2f}"
    )

# --- 3) Plotting Results (Tardiness + Energy ONLY, in TWO separate figures) ---

names = list(results["Total_Tardiness"].keys())
tardinesses = [results["Total_Tardiness"][n] for n in names]
energies = [results["Total_Energy"][n] for n in names]

x = np.arange(len(names))


def autolabel(ax, rects, fmt="{:.2f}"):
    for r in rects:
        h = r.get_height()
        ax.annotate(
            fmt.format(h),
            xy=(r.get_x() + r.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


# -------- Figure 1: Total Tardiness --------
fig1, ax1 = plt.subplots(figsize=(14, 7))
rects_t = ax1.bar(x, tardinesses, width=0.6, label="Total Tardiness (Total Delay)")

ax1.set_ylabel("Total Tardiness")
ax1.set_title(f"Total Tardiness Comparison for DAG Task {TASK_ID} (Gym)")
ax1.set_xticks(x)
ax1.set_xticklabels(names, rotation=15, ha="right", fontsize=10)
ax1.grid(axis="y", linestyle="--", alpha=0.6)
ax1.legend(loc="upper right")

autolabel(ax1, rects_t)

fig1.tight_layout()

# -------- Figure 2: Total Energy --------
fig2, ax2 = plt.subplots(figsize=(14, 7))
rects_e = ax2.bar(x, energies, width=0.6, label="Total Energy (Power × Time)")

ax2.set_ylabel("Total Energy (Power × Time)")
ax2.set_title(f"Total Energy Comparison for DAG Task {TASK_ID} (Gym)")
ax2.set_xticks(x)
ax2.set_xticklabels(names, rotation=15, ha="right", fontsize=10)
ax2.grid(axis="y", linestyle="--", alpha=0.6)
ax2.legend(loc="upper right")

autolabel(ax2, rects_e)

fig2.tight_layout()

plt.show()
