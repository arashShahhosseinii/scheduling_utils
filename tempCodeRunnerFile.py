import numpy as np
import matplotlib.pyplot as plt

from Create_Dag import CreateDAG

from scheduling_utils import (
    schedule_random_A7_A12,
    schedule_dual_A7,
    schedule_dual_A12,
    schedule_deadline_based_HEFT,
    compute_schedule_metrics,
)

def main():
    csv_path = "dag_dataset_a7_a12.csv"
    TASK_ID = 0
    SEED = 42
    rng = np.random.RandomState(SEED)

    dag = CreateDAG(csv_path, row_index=TASK_ID)
    G = dag.graph

    # Build execution time matrix (CONSISTENT WITH main.py):
    #   column 0 -> A7 (slower core) gets the slower original times
    #   column 1 -> A12 (faster core) gets the faster original times
    exec_times = np.stack(
        [
            np.array(dag.a12_times, dtype=float),  # index 0
            np.array(dag.a7_times, dtype=float),   # index 1
        ],
        axis=1,
    )  # shape: (num_tasks, 2)

    deadlines = np.array(dag.deadlines, dtype=float)

    def finish_dict_to_array(finish_dict):
        # EXACTLY like main.py: align finish times with task indices
        return np.array([finish_dict[i] for i in sorted(finish_dict.keys())], dtype=float)

    # ---------- 1) Dual A7 (Homogeneous) ----------
    a7_assigned, a7_start, a7_finish = schedule_dual_A7(G, exec_times, deadlines)
    a7_finish_arr = finish_dict_to_array(a7_finish)
    a7_makespan, a7_tardiness = compute_schedule_metrics(a7_finish_arr, deadlines)

    # ---------- 2) Dual A12 (Homogeneous) ----------
    a12_assigned, a12_start, a12_finish = schedule_dual_A12(G, exec_times, deadlines)
    a12_finish_arr = finish_dict_to_array(a12_finish)
    a12_makespan, a12_tardiness = compute_schedule_metrics(a12_finish_arr, deadlines)

    # ---------- 3) Random A7 + A12 (Heterogeneous) ----------
    rand_assigned, rand_start, rand_finish = schedule_random_A7_A12(G, exec_times, deadlines, rng)
    rand_finish_arr = finish_dict_to_array(rand_finish)
    rand_makespan, rand_tardiness = compute_schedule_metrics(rand_finish_arr, deadlines)

    # ---------- 4) EDF/HEFT-like A7 + A12 (Heterogeneous) ----------
    heft_assigned, heft_start, heft_finish = schedule_deadline_based_HEFT(G, exec_times, deadlines)
    heft_finish_arr = finish_dict_to_array(heft_finish)
    heft_makespan, heft_tardiness = compute_schedule_metrics(heft_finish_arr, deadlines)

    # ---------- Print numeric results ----------
    print("\n=== Scheduling Results (CONSISTENT WITH main.py) ===")
    print("Exec_times mapping: col 0 -> A7 (slower), col 1 -> A12 (faster)\n")

    print(f"Dual A7 (Homogeneous)             | Makespan: {a7_makespan:.2f} | Total Tardiness: {a7_tardiness:.2f}")
    print(f"Dual A12 (Homogeneous)            | Makespan: {a12_makespan:.2f} | Total Tardiness: {a12_tardiness:.2f}")
    print(f"Random A7 + A12 (Heterogeneous)   | Makespan: {rand_makespan:.2f} | Total Tardiness: {rand_tardiness:.2f}")
    print(f"EDF/HEFT A7 + A12 (Heterogeneous) | Makespan: {heft_makespan:.2f} | Total Tardiness: {heft_tardiness:.2f}")

    # ---------- Plot 1: Makespan comparison ----------
    algo_names = [
        "Dual A7",
        "Dual A12",
        "Random A7+A12",
        "EDF/HEFT A7+A12",
    ]
    makespans = [a7_makespan, a12_makespan, rand_makespan, heft_makespan]

    plt.figure(figsize=(10, 5))
    plt.bar(algo_names, makespans)
    plt.ylabel("Makespan (Total Completion Time)")
    plt.title(f"Makespan Comparison (Task {TASK_ID})")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    # ---------- Plot 2: Total Tardiness comparison ----------
    tardiness_values = [a7_tardiness, a12_tardiness, rand_tardiness, heft_tardiness]

    plt.figure(figsize=(10, 5))
    plt.bar(algo_names, tardiness_values)
    plt.ylabel("Total Tardiness")
    plt.title(f"Total Tardiness Comparison (Task {TASK_ID})")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
