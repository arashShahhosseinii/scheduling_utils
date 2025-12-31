import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

from Dag_Env import DagSchedulingEnv
from scheduling_utils import run_policy_episode


# --- 1) Initialization and Data Loading ---
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path(sys.argv[0]).resolve().parent

DATASET_PATH = SCRIPT_DIR / "dag_dataset(1)_a7_a12.csv"

TASK_ID = 0
SEED = 42
rng = np.random.RandomState(SEED)

# --- 2) Execute Scheduling Scenarios through Gymnasium Env ---
results = {
    "Makespan": {},
    "Total_Tardiness": {},
    "Total_Energy": {},
}

# ✅ Dualها فقط با HEFT
scheduling_scenarios = {
    "Dual A7 (Homogeneous, HEFT)": ([0, 0], "HEFT"),
    "Dual A12 (Homogeneous, HEFT)": ([1, 1], "HEFT"),
    "Random A7 + A12 (Heterogeneous)": ([0, 1], "RANDOM"),
    "HEFT A7 + A12 (Heterogeneous)": ([0, 1], "HEFT"),
}

for name, (processor_map, policy_name) in scheduling_scenarios.items():
    print(f"Running (Gym) scheduler: {name} ...")

    env = DagSchedulingEnv(
        csv_path=str(DATASET_PATH),
        row_index=TASK_ID,
        processor_map=processor_map,
        reward_weights=(1.0, 1.0, 0.0),
        invalid_action_penalty=1.0,
        seed=SEED,
    )

    # ✅ Random فقط یک بار اجرا می‌شود (بدون میانگین)
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

# --- 3) Plotting Results (Makespan + Energy ONLY, in TWO separate figures) ---
names = list(results["Makespan"].keys())
makespans = [results["Makespan"][n] for n in names]
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


# -------- Figure 1: Makespan --------
fig1, ax1 = plt.subplots(figsize=(14, 7))
rects_m = ax1.bar(x, makespans, width=0.6, label="Makespan (Total Completion Time)")

ax1.set_ylabel("Makespan")
ax1.set_title(f"Makespan Comparison for DAG Task {TASK_ID} (Gym)")
ax1.set_xticks(x)
ax1.set_xticklabels(names, rotation=15, ha="right", fontsize=10)
ax1.grid(axis="y", linestyle="--", alpha=0.6)
ax1.legend(loc="upper right")

autolabel(ax1, rects_m)
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
