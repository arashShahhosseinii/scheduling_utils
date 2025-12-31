import networkx as nx
import numpy as np
from typing import Tuple, List, Optional


# -----------------------------
# Generic metric helper (optional)
# -----------------------------
def compute_schedule_metrics(
    finish_times: np.ndarray, deadlines: np.ndarray
) -> Tuple[float, float]:
    """
    Returns:
      makespan = max(finish_times)
      total_tardiness = sum(max(0, finish - deadline))
    """
    if finish_times.size == 0:
        return 0.0, 0.0

    makespan = float(finish_times.max())
    tardiness = np.maximum(0.0, finish_times - deadlines)
    total_tardiness = float(tardiness.sum())
    return makespan, total_tardiness


# -----------------------------
# (Optional) kept to avoid breaking older code
# -----------------------------
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
    HEFT upward rank (comm cost assumed 0):
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


# ------------------------------------------------------------
# Policies that work on DagSchedulingEnv (Gymnasium env)
# ------------------------------------------------------------
def decode_action(action: int, num_cores: int) -> Tuple[int, int]:
    task = int(action // num_cores)
    core = int(action % num_cores)
    return task, core


def encode_action(task: int, core: int, num_cores: int) -> int:
    return int(task * num_cores + core)


def _ready_tasks_from_obs(obs: dict) -> List[int]:
    ready_mask = obs["ready_mask"].astype(bool)
    done_mask = obs["done_mask"].astype(bool)
    return [i for i in range(len(ready_mask)) if ready_mask[i] and (not done_mask[i])]


# ✅ RANDOM: Random task, but core chosen by EFT (best_core_for_task)
def choose_action_random(env, obs: dict, rng: np.random.RandomState) -> int:
    ready = _ready_tasks_from_obs(obs)
    if not ready:
        return 0

    chosen_task = int(rng.choice(ready))
    best_core = env.best_core_for_task(chosen_task)
    return encode_action(chosen_task, best_core, env.num_cores)


def choose_action_edf(env, obs: dict) -> int:
    ready = _ready_tasks_from_obs(obs)
    if not ready:
        return 0

    chosen_task = min(ready, key=lambda t: (float(env.deadlines[t]), int(t)))
    best_core = env.best_core_for_task(chosen_task)
    return encode_action(chosen_task, best_core, env.num_cores)


def choose_action_heft(env, obs: dict) -> int:
    ready = _ready_tasks_from_obs(obs)
    if not ready:
        return 0

    chosen_task = max(
        ready,
        key=lambda t: (float(env.rank_u_raw[t]), -float(env.deadlines[t]), -int(t)),
    )
    best_core = env.best_core_for_task(chosen_task)
    return encode_action(chosen_task, best_core, env.num_cores)


def run_policy_episode(env, policy_name: str, rng: Optional[np.random.RandomState] = None):
    obs, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0.0

    while not (terminated or truncated):
        if policy_name == "RANDOM":
            if rng is None:
                raise ValueError("rng must be provided for RANDOM policy")
            action = choose_action_random(env, obs, rng)

        elif policy_name == "EDF":
            action = choose_action_edf(env, obs)

        elif policy_name == "HEFT":
            action = choose_action_heft(env, obs)

        else:
            raise ValueError(f"Unknown policy_name: {policy_name}")

        obs, reward, terminated, truncated, step_info = env.step(action)
        total_reward += float(reward)

    assigned, start, finish = env.get_schedule()
    metrics = env.get_metrics()
    return assigned, start, finish, metrics, total_reward
