import numpy as np
import matplotlib.pyplot as plt

from Create_Dag import CreateDAG

from scheduling_utils import (
    schedule_random,
    schedule_deadline_based,
    schedule_deadline_only_A12,
    schedule_deadline_only_A7,
    compute_schedule_metrics,
)

def main():
    csv_path = "dag_dataset_a7_a12.csv"

    dag = CreateDAG(csv_path)
    G = dag.graph

    # Build execution time matrix:
    #   column 0 -> A7  (utilization_proc1_vf)
    #   column 1 -> A12 (utilization_proc2_vf)
    exec_times = np.stack(
        [
            np.array(dag.a7_times, dtype=float),   # A7  times
            np.array(dag.a12_times, dtype=float),  # A12 times
        ],
        axis=1,
    )  # shape: (num_tasks, 2)

    deadlines = np.array(dag.deadlines, dtype=float)

    rng = np.random.RandomState(42)

    # ---------- 1. Random Walk ----------
    rand_assigned, rand_start, rand_finish = schedule_random(G, exec_times, rng)
    rand_finish_arr = np.array(list(rand_finish.values()))
    rand_makespan, rand_tardiness = compute_schedule_metrics(rand_finish_arr, deadlines)

    # ---------- 2. Deadline-based (A7 + A12) ----------
    mix_assigned, mix_start, mix_finish = schedule_deadline_based(G, exec_times, deadlines)
    mix_finish_arr = np.array(list(mix_finish.values()))
    mix_makespan, mix_tardiness = compute_schedule_metrics(mix_finish_arr, deadlines)

    # ---------- 3. Deadline-based Only-A12 ----------
    a12_assigned, a12_start, a12_finish = schedule_deadline_only_A12(
        G, exec_times, deadlines, a12_index=1
    )
    a12_finish_arr = np.array(list(a12_finish.values()))
    a12_makespan, a12_tardiness = compute_schedule_metrics(a12_finish_arr, deadlines)

    # ---------- 4. Deadline-based Only-A7 ----------
    a7_assigned, a7_start, a7_finish = schedule_deadline_only_A7(
        G, exec_times, deadlines, a7_index=0
    )
    a7_finish_arr = np.array(list(a7_finish.values()))
    a7_makespan, a7_tardiness = compute_schedule_metrics(a7_finish_arr, deadlines)

    # ---------- Print numeric results ----------
    print("\n=== Scheduling Results (column 0 = A7, column 1 = A12) ===")
    
    # Random Walk
    print(f"Random Walk                 | Makespan: {rand_makespan:.2f} | Total Tardiness: {rand_tardiness:.2f}")
    
    # Deadline (A7 + A12)
    print(f"Deadline (A7 + A12)         | Makespan: {mix_makespan:.2f}  | Total Tardiness: {mix_tardiness:.2f}")
    
    # Deadline Only-A7
    print(f"Deadline Only-A7 (proc 0)   | Makespan: {a7_makespan:.2f} | Total Tardiness: {a7_tardiness:.2f}")
    
    # Deadline Only-A12
    print(f"Deadline Only-A12 (proc 1)  | Makespan: {a12_makespan:.2f}  | Total Tardiness: {a12_tardiness:.2f}")

    # ---------- Plot 1: Makespan comparison ----------
    algo_names = ["Random", "Deadl A7+A12", "Only A7", "Only A12"] 
    makespans = [rand_makespan, mix_makespan, a7_makespan, a12_makespan] 

    plt.figure(figsize=(9, 5))
    plt.bar(algo_names, makespans)
    plt.ylabel("Makespan (Total Completion Time)")
    plt.title("Makespan Comparison of 4 Scheduling Algorithms")
    plt.tight_layout()

    # ---------- Plot 2: Total Tardiness comparison ----------
    tardiness_values = [rand_tardiness, mix_tardiness, a7_tardiness, a12_tardiness] 

    plt.figure(figsize=(9, 5))
    plt.bar(algo_names, tardiness_values)
    plt.ylabel("Total Tardiness")
    plt.title("Total Tardiness Comparison of 4 Scheduling Algorithms")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()