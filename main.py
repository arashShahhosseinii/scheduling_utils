import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

from Create_Dag import CreateDAG
from scheduling_utils import (
    compute_schedule_metrics,
    schedule_random_A7_A12,
    schedule_dual_A7,
    schedule_dual_A12,
    schedule_deadline_based_HEFT,
)

# --- 1) Initialization and Data Loading ---
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path(sys.argv[0]).resolve().parent

DATASET_PATH = SCRIPT_DIR / "dag_dataset_a7_a12.csv"

TASK_ID = 0
SEED = 42
rng = np.random.RandomState(SEED)

dag_creator = CreateDAG(str(DATASET_PATH), row_index=TASK_ID)
G = dag_creator.graph

# exec_times convention: col0=A7, col1=A12
exec_times = np.stack(
    [
        np.array(dag_creator.a7_times, dtype=float),
        np.array(dag_creator.a12_times, dtype=float),
    ],
    axis=1,
)

deadlines = np.array(dag_creator.deadlines, dtype=float)

# energy matrix: col0=A7 energy, col1=A12 energy
energy_mat = np.stack(
    [
        np.array(dag_creator.a7_energy, dtype=float),
        np.array(dag_creator.a12_energy, dtype=float),
    ],
    axis=1,
)

num_tasks = G.number_of_nodes()


def finish_dict_to_array(finish_dict):
    return np.array([finish_dict[i] for i in sorted(finish_dict.keys())], dtype=float)


def compute_total_energy(assigned_core_index: dict, processor_map: list) -> float:
    """
    assigned_core_index: task -> core_index
    processor_map: core_index -> proc_type (0=A7,1=A12)
    energy_mat: [task, proc_type] -> energy
    """
    total = 0.0
    for t in range(num_tasks):
        core_idx = assigned_core_index[t]
        proc_type = processor_map[core_idx]
        total += float(energy_mat[t, proc_type])
    return total


# --- 2) Execute Scheduling Algorithms ---
results = {
    "Makespan": {},
    "Total_Tardiness": {},
    "Total_Energy": {},
}

scheduling_scenarios = {
    "Dual A7 (Homogeneous)": (schedule_dual_A7, [0, 0]),
    "Dual A12 (Homogeneous)": (schedule_dual_A12, [1, 1]),
    "Random A7 + A12 (Heterogeneous)": (schedule_random_A7_A12, [0, 1]),
    "HEFT A7 + A12 (Heterogeneous)": (schedule_deadline_based_HEFT, [0, 1]),
}

for name, (scheduler_func, processor_map) in scheduling_scenarios.items():
    print(f"Running scheduler: {name}...")

    if "Random" in name:
        assigned, start, finish = scheduler_func(G, exec_times, deadlines, rng)
    else:
        assigned, start, finish = scheduler_func(G, exec_times, deadlines)

    finish_array = finish_dict_to_array(finish)
    makespan, total_tardiness = compute_schedule_metrics(finish_array, deadlines)
    total_energy = compute_total_energy(assigned, processor_map)

    results["Makespan"][name] = makespan
    results["Total_Tardiness"][name] = total_tardiness
    results["Total_Energy"][name] = total_energy

    # Makespan is still computed & printed, but will NOT be plotted
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
ax1.set_title(f"Total Tardiness Comparison for DAG Task {TASK_ID}")
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
ax2.set_title(f"Total Energy Comparison for DAG Task {TASK_ID}")
ax2.set_xticks(x)
ax2.set_xticklabels(names, rotation=15, ha="right", fontsize=10)
ax2.grid(axis="y", linestyle="--", alpha=0.6)
ax2.legend(loc="upper right")

autolabel(ax2, rects_e)

fig2.tight_layout()

plt.show()
