import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional

from Create_Dag import CreateDAG


class DagSchedulingEnv(gym.Env):
    """
    A real DAG scheduling environment.

    - Actions choose (task, core).
    - The env performs actual list scheduling with insertion.
    - Tracks assigned/start/finish, makespan, total tardiness, total energy.
    - Provides action_mask for valid (task, core) pairs.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        csv_path: str,
        row_index: int = 0,
        processor_map: Optional[List[int]] = None,  # per-core proc type: 0=A7, 1=A12
        reward_weights: Tuple[float, float, float] = (1.0, 1.0, 0.0),  # (wE, wT, wM)
        invalid_action_penalty: float = 1.0,
        seed: Optional[int] = None,
    ):
        super().__init__()

        if processor_map is None:
            processor_map = [0, 1]  # default: one A7 + one A12

        self.processor_map: List[int] = list(processor_map)
        self.num_cores: int = len(self.processor_map)

        self.wE, self.wT, self.wM = reward_weights
        self.invalid_action_penalty = float(invalid_action_penalty)

        # Load DAG from CSV row
        self.dag = CreateDAG(csv_path, row_index=row_index)
        self.G: nx.DiGraph = self.dag.graph

        if not nx.is_directed_acyclic_graph(self.G):
            raise ValueError("Input graph is not a DAG (contains a cycle).")

        self.num_tasks: int = int(self.G.number_of_nodes())
        nodes_sorted = sorted(self.G.nodes())
        if nodes_sorted != list(range(self.num_tasks)):
            raise ValueError(f"Graph nodes must be exactly 0..{self.num_tasks-1} (got {nodes_sorted[:10]}...).")

        # Convention everywhere:
        # exec_times[:,0] = A7, exec_times[:,1] = A12
        self.exec_times = np.stack(
            [
                np.array(self.dag.a7_times, dtype=np.float32),
                np.array(self.dag.a12_times, dtype=np.float32),
            ],
            axis=1,
        )  # shape: (N,2)

        self.deadlines = np.array(self.dag.deadlines, dtype=np.float32)  # shape: (N,)

        # energy_mat[:,0] = A7 energy, energy_mat[:,1] = A12 energy
        self.energy_mat = np.stack(
            [
                np.array(self.dag.a7_energy, dtype=np.float32),
                np.array(self.dag.a12_energy, dtype=np.float32),
            ],
            axis=1,
        )  # shape: (N,2)

        # Other optional node fields
        self.periods = np.array(self.dag.periods, dtype=np.float32)
        self.m_i = np.array(self.dag.m_i_list, dtype=np.float32)

        # Precompute HEFT rank_u for this processor_map (types used)
        self.rank_u_raw = self._compute_upward_ranks(self.G, self.exec_times, self.processor_map).astype(np.float32)

        # -------- Normalization scales for observation --------
        # Upper bound-ish scale for time: max(deadline, N*max_exec)
        max_dead = float(self.deadlines.max()) if self.deadlines.size else 1.0
        max_exec = float(self.exec_times.max()) if self.exec_times.size else 1.0
        self.time_scale = max(max_dead, self.num_tasks * max_exec, 1e-6)

        max_energy = float(self.energy_mat.max()) if self.energy_mat.size else 1.0
        self.energy_scale = max(max_energy, 1e-6)

        max_period = float(self.periods.max()) if self.periods.size else 1.0
        self.period_scale = max(max_period, 1e-6)

        max_mi = float(self.m_i.max()) if self.m_i.size else 1.0
        self.mi_scale = max(max_mi, 1e-6)

        max_rank = float(self.rank_u_raw.max()) if self.rank_u_raw.size else 1.0
        self.rank_scale = max(max_rank, 1e-6)

        # -------- Observation features (node-wise) --------
        # We'll expose (N,7):
        # [period_norm, deadline_norm, m_i_norm, tA7_norm, tA12_norm, eA7_norm, eA12_norm]
        self.node_features = np.stack(
            [
                (self.periods / self.period_scale),
                (self.deadlines / self.time_scale),
                (self.m_i / self.mi_scale),
                (self.exec_times[:, 0] / self.time_scale),  # A7 time
                (self.exec_times[:, 1] / self.time_scale),  # A12 time
                (self.energy_mat[:, 0] / self.energy_scale),  # A7 energy
                (self.energy_mat[:, 1] / self.energy_scale),  # A12 energy
            ],
            axis=1,
        ).astype(np.float32)

        self.rank_u = (self.rank_u_raw / self.rank_scale).astype(np.float32)

        # -------- Spaces --------
        # action = task * num_cores + core
        self.action_space = spaces.Discrete(self.num_tasks * self.num_cores)

        self.observation_space = spaces.Dict(
            {
                "node_features": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.num_tasks, 7),
                    dtype=np.float32,
                ),
                "rank_u": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.num_tasks,),
                    dtype=np.float32,
                ),
                "done_mask": spaces.MultiBinary(self.num_tasks),
                "ready_mask": spaces.MultiBinary(self.num_tasks),
                "core_available": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.num_cores,),
                    dtype=np.float32,
                ),
                "action_mask": spaces.MultiBinary(self.num_tasks * self.num_cores),
            }
        )

        # -------- Internal state (reset each episode) --------
        self.done_mask = np.zeros(self.num_tasks, dtype=bool)
        self.ready_mask = np.zeros(self.num_tasks, dtype=bool)

        self.assigned: Dict[int, int] = {}  # task -> core_idx
        self.start: Dict[int, float] = {}
        self.finish: Dict[int, float] = {}

        # For insertion scheduling per core: list of (start, finish, task)
        self.core_intervals: List[List[Tuple[float, float, int]]] = [[] for _ in range(self.num_cores)]
        self.core_available_time = np.zeros(self.num_cores, dtype=np.float32)

        self.total_energy = 0.0
        self.total_tardiness = 0.0
        self.makespan = 0.0

        # Seed
        self._initial_seed = seed
        self.reset(seed=seed)

    # ----------------- Core scheduling helpers -----------------

    @staticmethod
    def _compute_upward_ranks(G: nx.DiGraph, exec_times: np.ndarray, processor_map: List[int]) -> np.ndarray:
        """
        HEFT upward rank (comm cost = 0):
          rank_u(n) = avg_w(n) + max_{succ} rank_u(succ)
        avg_w(n) is mean execution time over processor types used in processor_map.
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

    @staticmethod
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

    def ready_time_of(self, task: int) -> float:
        preds = list(self.G.predecessors(task))
        if not preds:
            return 0.0
        return float(max(self.finish[p] for p in preds))

    def estimate_start_finish(self, task: int, core_idx: int) -> Tuple[float, float]:
        proc_type = self.processor_map[core_idx]  # 0=A7, 1=A12
        dur = float(self.exec_times[task, proc_type])

        rt = self.ready_time_of(task)
        s = self._find_earliest_insertion_start(self.core_intervals[core_idx], rt, dur)
        f = s + dur
        return float(s), float(f)

    def best_core_for_task(self, task: int) -> int:
        """
        EFT core selection (Earliest Finish Time).
        Tie-breaking matches old logic: first core with strictly smaller finish wins.
        """
        best_core = 0
        best_finish = float("inf")
        for c in range(self.num_cores):
            _, f = self.estimate_start_finish(task, c)
            if f < best_finish:
                best_finish = f
                best_core = c
        return best_core

    # ----------------- Masks / Observation -----------------

    def _update_ready_mask(self) -> None:
        for t in range(self.num_tasks):
            if self.done_mask[t]:
                self.ready_mask[t] = False
                continue
            preds = list(self.G.predecessors(t))
            self.ready_mask[t] = all(self.done_mask[p] for p in preds)

    def _compute_action_mask(self) -> np.ndarray:
        # valid if task is ready and not done; any core allowed
        mask = np.zeros(self.num_tasks * self.num_cores, dtype=np.int8)
        for task in range(self.num_tasks):
            if self.ready_mask[task] and (not self.done_mask[task]):
                base = task * self.num_cores
                mask[base : base + self.num_cores] = 1
        return mask

    def _get_obs(self) -> Dict[str, np.ndarray]:
        # core available normalized
        core_av_norm = np.clip(self.core_available_time / self.time_scale, 0.0, 1.0).astype(np.float32)
        action_mask = self._compute_action_mask()

        return {
            "node_features": self.node_features.copy(),
            "rank_u": self.rank_u.copy(),
            "done_mask": self.done_mask.astype(np.int8),
            "ready_mask": self.ready_mask.astype(np.int8),
            "core_available": core_av_norm,
            "action_mask": action_mask,
        }

    # ----------------- Gymnasium API -----------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed if seed is not None else self._initial_seed)

        self.done_mask[:] = False
        self.ready_mask[:] = False

        self.assigned = {}
        self.start = {}
        self.finish = {}

        self.core_intervals = [[] for _ in range(self.num_cores)]
        self.core_available_time[:] = 0.0

        self.total_energy = 0.0
        self.total_tardiness = 0.0
        self.makespan = 0.0

        self._update_ready_mask()

        obs = self._get_obs()
        info = {
            "num_tasks": self.num_tasks,
            "num_cores": self.num_cores,
            "processor_map": self.processor_map,
        }
        return obs, info

    def step(self, action: int):
        # Decode
        task = int(action // self.num_cores)
        core = int(action % self.num_cores)

        # Validate action
        invalid = (
            task < 0
            or task >= self.num_tasks
            or core < 0
            or core >= self.num_cores
            or self.done_mask[task]
            or (not self.ready_mask[task])
        )

        if invalid:
            reward = -float(self.invalid_action_penalty)
            terminated = False
            truncated = False
            info = {"invalid_action": True}
            return self._get_obs(), reward, terminated, truncated, info

        # Schedule on chosen core with insertion
        s, f = self.estimate_start_finish(task, core)

        self.assigned[task] = core
        self.start[task] = float(s)
        self.finish[task] = float(f)

        self.core_intervals[core].append((float(s), float(f), task))
        self.core_intervals[core].sort(key=lambda x: x[0])

        # update core available time (simple upper bound, not exact earliest gap)
        self.core_available_time[core] = float(max(self.core_available_time[core], f))

        # update masks
        self.done_mask[task] = True
        self._update_ready_mask()

        # Metrics increments
        proc_type = self.processor_map[core]
        energy_inc = float(self.energy_mat[task, proc_type])
        tard_inc = float(max(0.0, f - float(self.deadlines[task])))

        self.total_energy += energy_inc
        self.total_tardiness += tard_inc
        self.makespan = float(max(self.makespan, f))

        # Reward shaping
        reward = -(self.wE * energy_inc + self.wT * tard_inc)

        terminated = bool(self.done_mask.all())
        truncated = False

        info = {
            "scheduled_task": task,
            "scheduled_core": core,
            "start": float(s),
            "finish": float(f),
            "energy_inc": energy_inc,
            "tardiness_inc": tard_inc,
        }

        if terminated:
            # Add final makespan penalty if wM > 0
            reward -= float(self.wM * self.makespan)

            info.update(
                {
                    "makespan": float(self.makespan),
                    "total_tardiness": float(self.total_tardiness),
                    "total_energy": float(self.total_energy),
                    "assigned": self.assigned.copy(),
                    "start_times": self.start.copy(),
                    "finish_times": self.finish.copy(),
                }
            )

        return self._get_obs(), float(reward), terminated, truncated, info

    # ----------------- Convenience -----------------

    def get_schedule(self) -> Tuple[Dict[int, int], Dict[int, float], Dict[int, float]]:
        return self.assigned.copy(), self.start.copy(), self.finish.copy()

    def get_metrics(self) -> Dict[str, float]:
        return {
            "makespan": float(self.makespan),
            "total_tardiness": float(self.total_tardiness),
            "total_energy": float(self.total_energy),
        }

    def render(self):
        done_nodes = [i for i, d in enumerate(self.done_mask) if d]
        ready_nodes = [i for i, r in enumerate(self.ready_mask) if r]
        print(f"Done (scheduled) tasks: {done_nodes}")
        print(f"Ready tasks: {ready_nodes}")
        print(f"Makespan: {self.makespan:.3f} | Tardiness: {self.total_tardiness:.3f} | Energy: {self.total_energy:.3f}")
