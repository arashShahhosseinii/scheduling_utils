import networkx as nx
import numpy as np
from typing import Tuple, Dict, List, Optional


def compute_schedule_metrics(
    finish_times: np.ndarray, deadlines: np.ndarray
) -> Tuple[float, float]:
    if finish_times.size == 0:
        return 0.0, 0.0

    makespan = float(finish_times.max())
    tardiness = np.maximum(0.0, finish_times - deadlines)
    total_tardiness = float(tardiness.sum())
    return makespan, total_tardiness


def _validate_inputs(
    G: nx.DiGraph,
    exec_times: np.ndarray,
    deadlines: np.ndarray,
    processor_map: List[int],
) -> None:
    """
    Convention:
      processor types: 0=A7, 1=A12
      exec_times[:,0]=A7, exec_times[:,1]=A12
    """
    if not isinstance(G, nx.DiGraph):
        raise TypeError("G must be a networkx.DiGraph")

    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Input graph is not a DAG (contains at least one directed cycle).")

    num_tasks = G.number_of_nodes()
    nodes_sorted = sorted(G.nodes())
    if nodes_sorted != list(range(num_tasks)):
        raise ValueError(f"Graph nodes must be exactly 0..{num_tasks-1} (got {nodes_sorted[:10]}...).")

    if deadlines.shape[0] != num_tasks:
        raise ValueError(f"deadlines length ({deadlines.shape[0]}) != num_tasks ({num_tasks})")

    if exec_times.ndim != 2 or exec_times.shape[0] != num_tasks or exec_times.shape[1] < 2:
        raise ValueError("exec_times must be shape (num_tasks, >=2) with columns [A7, A12].")

    if not processor_map:
        raise ValueError("processor_map must contain at least one core")

    if any(t not in (0, 1) for t in processor_map):
        raise ValueError("processor_map types must be 0 (A7) or 1 (A12)")


def _find_earliest_insertion_start(
    intervals: List[Tuple[float, float, int]],
    ready_time: float,
    duration: float,
) -> float:
    if duration < 0:
        raise ValueError("duration must be non-negative")

    if not intervals:
        return ready_time

    intervals = sorted(intervals, key=lambda x: x[0])

    # before first interval
    s = ready_time
    if s + duration <= intervals[0][0]:
        return s

    # gaps between intervals
    for i in range(len(intervals) - 1):
        gap_start = max(ready_time, intervals[i][1])
        gap_end = intervals[i + 1][0]
        if gap_start + duration <= gap_end:
            return gap_start

    # after last interval
    return max(ready_time, intervals[-1][1])


def _compute_upward_ranks(
    G: nx.DiGraph,
    exec_times: np.ndarray,
    processor_map: List[int],
) -> np.ndarray:
    """
    Real HEFT upward rank (communication cost assumed 0):
      rank_u(n) = avg_w(n) + max_{succ} rank_u(succ)
    avg_w(n) is mean execution time over processor types used.
    """
    num_tasks = G.number_of_nodes()
    types_used = sorted(set(processor_map))

    avg_w = np.zeros(num_tasks, dtype=float)
    for t in range(num_tasks):
        avg_w[t] = float(np.mean(exec_times[t, types_used]))

    topo = list(nx.topological_sort(G))
    rank_u = np.zeros(num_tasks, dtype=float)

    for n in reversed(topo):
        succs = list(G.successors(n))
        rank_u[n] = avg_w[n] if not succs else (avg_w[n] + max(rank_u[s] for s in succs))

    return rank_u


def _schedule_parallel_dag(
    G: nx.DiGraph,
    exec_times: np.ndarray,
    deadlines: np.ndarray,
    processor_map: List[int],        # per-core type (0=A7,1=A12)
    policy: str,                     # 'EDF', 'RANDOM', 'HEFT'
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[Dict[int, int], Dict[int, float], Dict[int, float]]:
    """
    Returns:
      assigned: task -> core_index (0..num_cores-1)
      start: task -> start time
      finish: task -> finish time
    """
    _validate_inputs(G, exec_times, deadlines, processor_map)

    num_tasks = G.number_of_nodes()
    num_cores = len(processor_map)

    if policy == "RANDOM" and rng is None:
        raise ValueError("rng must be provided for RANDOM policy")

    rank_u = _compute_upward_ranks(G, exec_times, processor_map) if policy == "HEFT" else None

    scheduled = np.zeros(num_tasks, dtype=bool)
    core_intervals: List[List[Tuple[float, float, int]]] = [[] for _ in range(num_cores)]

    assigned: Dict[int, int] = {}
    start: Dict[int, float] = {}
    finish: Dict[int, float] = {}

    def ready_time_of(task: int) -> float:
        preds = list(G.predecessors(task))
        return 0.0 if not preds else max(finish[p] for p in preds)

    while not bool(scheduled.all()):
        ready_tasks: List[int] = []
        for t in range(num_tasks):
            if scheduled[t]:
                continue
            preds = list(G.predecessors(t))
            if all(scheduled[p] for p in preds):
                ready_tasks.append(t)

        if not ready_tasks:
            remaining = [i for i in range(num_tasks) if not scheduled[i]]
            raise RuntimeError(f"No ready tasks found but unscheduled remain (e.g. {remaining[:10]}...).")

        # Choose task
        if policy == "RANDOM":
            chosen_task = int(rng.choice(ready_tasks))
        elif policy == "EDF":
            chosen_task = min(ready_tasks, key=lambda t: (deadlines[t], t))
        elif policy == "HEFT":
            chosen_task = max(ready_tasks, key=lambda t: (rank_u[t], -deadlines[t], -t))
        else:
            raise ValueError(f"Invalid policy: {policy}")

        # Allocate by EFT
        rt = ready_time_of(chosen_task)
        best_core = -1
        best_start = 0.0
        best_finish = float("inf")

        for core_idx in range(num_cores):
            proc_type = processor_map[core_idx]  # 0 or 1
            dur = float(exec_times[chosen_task, proc_type])

            s = _find_earliest_insertion_start(core_intervals[core_idx], rt, dur)
            f = s + dur

            if f < best_finish:
                best_finish = f
                best_core = core_idx
                best_start = s

        assigned[chosen_task] = best_core
        start[chosen_task] = float(best_start)
        finish[chosen_task] = float(best_finish)

        core_intervals[best_core].append((float(best_start), float(best_finish), chosen_task))
        core_intervals[best_core].sort(key=lambda x: x[0])

        scheduled[chosen_task] = True

    return assigned, start, finish


def schedule_random_A7_A12(G: nx.DiGraph, exec_times: np.ndarray, deadlines: np.ndarray, rng: np.random.RandomState):
    processor_map = [0, 1]  # core0=A7, core1=A12
    return _schedule_parallel_dag(G, exec_times, deadlines, processor_map, policy="RANDOM", rng=rng)


def schedule_dual_A7(G: nx.DiGraph, exec_times: np.ndarray, deadlines: np.ndarray):
    processor_map = [0, 0]
    return _schedule_parallel_dag(G, exec_times, deadlines, processor_map, policy="EDF")


def schedule_dual_A12(G: nx.DiGraph, exec_times: np.ndarray, deadlines: np.ndarray):
    processor_map = [1, 1]
    return _schedule_parallel_dag(G, exec_times, deadlines, processor_map, policy="EDF")


def schedule_deadline_based_HEFT(G: nx.DiGraph, exec_times: np.ndarray, deadlines: np.ndarray):
    processor_map = [0, 1]
    return _schedule_parallel_dag(G, exec_times, deadlines, processor_map, policy="HEFT")
