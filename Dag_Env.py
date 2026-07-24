from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces

from Create_Dag import CreateDAG


class DagSchedulingEnv(gym.Env):
    """
    Gymnasium environment for heterogeneous DAG scheduling.

    Action:
      Discrete(max_tasks * num_cores), decoded as:
        action = task * num_cores + core

    Observation:
      Dict with padded graph tensors:
        node_features:    (max_tasks, 12)
        edge_index:       (2, max_edges)
        edge_mask:        (max_edges,)
        node_mask:        (max_tasks,)
        core_features:    (num_cores, 3)
        global_features:  (6,)
        action_mask:      (max_tasks * num_cores,)

    Reward (QoS‑based):
      For each finished task:
        QoS = 1                          if Fi <= Di
              (x*Di - Fi) / ((x-1)*Di)   if Di < Fi <= x*Di
              0                          if Fi > x*Di
      Then reward = QoS * (max_global_energy / actual_energy)   (Proposal A)
                or QoS * exp(-actual_energy / max_global_energy) (Proposal B)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        csv_path: Optional[Union[str, Path]] = None,
        row_index: int = 0,
        csv_paths: Optional[Sequence[Union[str, Path]]] = None,
        dag_records: Optional[Sequence[Tuple[Union[str, Path], int]]] = None,
        processor_map: Optional[List[int]] = None,
        reward_weights: Tuple[float, float] = (0.5, 0.5),   # kept for compatibility, not used
        invalid_action_penalty: float = 2.0,
        max_tasks: Optional[int] = None,
        max_edges: Optional[int] = None,
        sample_dags: bool = True,
        seed: Optional[int] = None,
        qos_factor: float = 2.0,                 # new: x in the QoS formula
        reward_proposal: str = "A",              # "A" or "B"
    ) -> None:
        super().__init__()

        if processor_map is None:
            processor_map = [0, 1]

        self.processor_map = list(processor_map)
        self.num_cores = len(self.processor_map)

        if self.num_cores <= 0:
            raise ValueError("processor_map must contain at least one core.")

        if any(p not in (0, 1) for p in self.processor_map):
            raise ValueError("processor_map values must be 0=A7 or 1=A12.")

        # reward_weights are no longer used, but kept for compatibility with old code
        self.wE, self.wM = map(float, reward_weights)   # not used in reward

        self.invalid_action_penalty = float(invalid_action_penalty)
        self.sample_dags = bool(sample_dags)
        self._initial_seed = seed
        self.np_random = np.random.default_rng(seed)

        # QoS parameters
        self.qos_factor = float(qos_factor)          # x in the formula
        self.reward_proposal = reward_proposal.upper()
        if self.reward_proposal not in ("A", "B"):
            raise ValueError("reward_proposal must be 'A' or 'B'")

        if dag_records is not None:
            self.dag_records = [(str(p), int(i)) for p, i in dag_records]
        elif csv_paths is not None:
            self.dag_records = CreateDAG.list_records(csv_paths)
        elif csv_path is not None:
            self.dag_records = [(str(csv_path), int(row_index))]
        else:
            raise ValueError("Provide csv_path, csv_paths, or dag_records.")

        sizes = []
        edge_sizes = []

        for path, idx in self.dag_records:
            rec = CreateDAG.read_record(path, idx)
            sizes.append(rec.num_subtasks)
            edge_sizes.append(len(rec.edges))

        self.max_tasks = int(max_tasks or max(sizes))
        self.max_edges = int(max_edges or max(max(edge_sizes), 1))

        if self.max_tasks < max(sizes):
            raise ValueError("max_tasks is smaller than at least one DAG.")

        if self.max_edges < max(edge_sizes):
            raise ValueError("max_edges is smaller than at least one DAG edge count.")

        self.node_feature_dim = 12
        self.core_feature_dim = 3
        self.global_feature_dim = 6

        self.action_space = spaces.Discrete(self.max_tasks * self.num_cores)

        self.observation_space = spaces.Dict(
            {
                "node_features": spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(self.max_tasks, self.node_feature_dim),
                    dtype=np.float32,
                ),
                "edge_index": spaces.Box(
                    0,
                    self.max_tasks - 1,
                    shape=(2, self.max_edges),
                    dtype=np.int64,
                ),
                "edge_mask": spaces.MultiBinary(self.max_edges),
                "node_mask": spaces.MultiBinary(self.max_tasks),
                "core_features": spaces.Box(
                    0.0,
                    np.inf,
                    shape=(self.num_cores, self.core_feature_dim),
                    dtype=np.float32,
                ),
                "global_features": spaces.Box(
                    0.0,
                    np.inf,
                    shape=(self.global_feature_dim,),
                    dtype=np.float32,
                ),
                "action_mask": spaces.MultiBinary(self.max_tasks * self.num_cores),
            }
        )

        self.dag: CreateDAG
        self.G: nx.DiGraph
        self.num_tasks = 0
        self.edges: List[Tuple[int, int]] = []

        self.exec_times = np.zeros((1, 2), dtype=np.float32)
        self.energy_mat = np.zeros((1, 2), dtype=np.float32)

        self.deadlines = np.ones(1, dtype=np.float32)
        self.periods = np.ones(1, dtype=np.float32)
        self.m_i = np.ones(1, dtype=np.float32)
        self.rank_u_raw = np.ones(1, dtype=np.float32)

        # scaling factors (still used for normalising observations, not reward)
        self.time_scale = 1.0
        self.energy_scale = 1.0
        self.period_scale = 1.0
        self.mi_scale = 1.0
        self.rank_scale = 1.0

        # global max per-task energy for reward normalisation
        self.max_energy_global = 1.0

        # Denominator for graph-level energy normalisation:
        # energy(DAG) / energy(max v-f)
        self.max_vf_dag_energy = 1.0

        self.done_mask: np.ndarray
        self.ready_mask: np.ndarray
        self.assigned: Dict[int, int]
        self.start: Dict[int, float]
        self.finish: Dict[int, float]

        self.core_intervals: List[List[Tuple[float, float, int]]]
        self.core_available_time: np.ndarray

        self.total_energy = 0.0
        self.total_tardiness = 0.0
        self.makespan = 0.0
        self.steps = 0

        self.current_record: Tuple[str, int] = self.dag_records[0]

        self.reset(seed=seed)

    def _load_dag(self, record: Tuple[str, int]) -> None:
        self.current_record = (str(record[0]), int(record[1]))

        self.dag = CreateDAG(
            self.current_record[0],
            self.current_record[1],
        )

        self.G = self.dag.graph
        self.edges = list(self.G.edges())
        self.num_tasks = int(self.G.number_of_nodes())

        self.exec_times = np.stack(
            [
                np.asarray(self.dag.a7_times, dtype=np.float32),
                np.asarray(self.dag.a12_times, dtype=np.float32),
            ],
            axis=1,
        )

        self.energy_mat = np.stack(
            [
                np.asarray(self.dag.a7_energy, dtype=np.float32),
                np.asarray(self.dag.a12_energy, dtype=np.float32),
            ],
            axis=1,
        )

        self.deadlines = np.asarray(self.dag.deadlines, dtype=np.float32)
        self.periods = np.asarray(self.dag.periods, dtype=np.float32)
        self.m_i = np.asarray(self.dag.m_i_list, dtype=np.float32)

        self.rank_u_raw = self._compute_upward_ranks(
            self.G,
            self.exec_times,
            self.processor_map,
        ).astype(np.float32)

        # Compute energy normalisation values.
        # self.energy_mat already stores the energy of each task at the selected
        # maximum v-f entry for A7 and A12, as extracted in Create_Dag.py.
        available_proc_types = sorted(set(self.processor_map))
        available_energy = self.energy_mat[:, available_proc_types]

        # Per-task maximum energy, used by the existing reward proposal A/B.
        self.max_energy_global = (
            float(np.max(available_energy))
            if available_energy.size
            else 1.0
        )

        # Graph-level maximum-v-f energy denominator:
        # energy(max v-f) = sum over all DAG tasks of the largest available
        # max-v-f energy value for that task.
        self.max_vf_dag_energy = max(
            float(np.sum(np.max(available_energy, axis=1)))
            if available_energy.size
            else 1.0,
            1e-6,
        )

        max_deadline = float(np.max(self.deadlines)) if self.deadlines.size else 1.0
        max_exec = float(np.max(self.exec_times)) if self.exec_times.size else 1.0

        self.time_scale = max(max_deadline, self.num_tasks * max_exec, 1e-6)
        self.energy_scale = max(float(np.max(self.energy_mat)), 1e-6)
        self.period_scale = max(float(np.max(self.periods)), 1e-6)
        self.mi_scale = max(float(np.max(self.m_i)), 1e-6)
        self.rank_scale = max(float(np.max(self.rank_u_raw)), 1e-6)

    @staticmethod
    def _compute_upward_ranks(
        G: nx.DiGraph,
        exec_times: np.ndarray,
        processor_map: List[int],
    ) -> np.ndarray:
        num_tasks = G.number_of_nodes()
        types_used = sorted(set(processor_map))

        avg_w = np.zeros(num_tasks, dtype=float)

        for task in range(num_tasks):
            avg_w[task] = float(np.mean(exec_times[task, types_used]))

        rank_u = np.zeros(num_tasks, dtype=float)

        for node in reversed(list(nx.topological_sort(G))):
            successors = list(G.successors(node))

            if not successors:
                rank_u[node] = avg_w[node]
            else:
                rank_u[node] = avg_w[node] + max(rank_u[s] for s in successors)

        return rank_u

    @staticmethod
    def _find_earliest_insertion_start(
        intervals: List[Tuple[float, float, int]],
        ready_time: float,
        duration: float,
    ) -> float:
        if not intervals:
            return float(ready_time)

        intervals = sorted(intervals, key=lambda x: x[0])
        candidate = float(ready_time)

        if candidate + duration <= intervals[0][0]:
            return candidate

        for i in range(len(intervals) - 1):
            candidate = max(float(ready_time), intervals[i][1])

            if candidate + duration <= intervals[i + 1][0]:
                return candidate

        return max(float(ready_time), intervals[-1][1])

    def ready_time_of(self, task: int) -> float:
        preds = list(self.G.predecessors(task))

        if not preds:
            return 0.0

        return float(max(self.finish[p] for p in preds))

    def estimate_start_finish(self, task: int, core_idx: int) -> Tuple[float, float]:
        proc_type = self.processor_map[core_idx]
        duration = float(self.exec_times[task, proc_type])
        ready_time = self.ready_time_of(task)

        start = self._find_earliest_insertion_start(
            self.core_intervals[core_idx],
            ready_time,
            duration,
        )

        return float(start), float(start + duration)

    def best_core_for_task(self, task: int) -> int:
        best_core = 0
        best_finish = float("inf")

        for core in range(self.num_cores):
            _, finish = self.estimate_start_finish(task, core)

            if finish < best_finish:
                best_core = core
                best_finish = finish

        return best_core

    def _update_ready_mask(self) -> None:
        self.ready_mask[:] = False

        for task in range(self.num_tasks):
            if self.done_mask[task]:
                continue

            self.ready_mask[task] = all(
                self.done_mask[p]
                for p in self.G.predecessors(task)
            )

    def _action_mask(self) -> np.ndarray:
        mask = np.zeros(self.max_tasks * self.num_cores, dtype=np.int8)

        for task in range(self.num_tasks):
            if self.ready_mask[task] and not self.done_mask[task]:
                for core in range(self.num_cores):
                    mask[task * self.num_cores + core] = 1

        return mask

    def _node_features(self) -> np.ndarray:
        x = np.zeros(
            (self.max_tasks, self.node_feature_dim),
            dtype=np.float32,
        )

        for task in range(self.num_tasks):
            finish_norm = float(self.finish.get(task, 0.0)) / self.time_scale
            start_norm = float(self.start.get(task, 0.0)) / self.time_scale

            x[task] = np.asarray(
                [
                    self.periods[task] / self.period_scale,
                    self.deadlines[task] / self.time_scale,
                    self.m_i[task] / self.mi_scale,
                    self.exec_times[task, 0] / self.time_scale,
                    self.exec_times[task, 1] / self.time_scale,
                    self.energy_mat[task, 0] / self.energy_scale,
                    self.energy_mat[task, 1] / self.energy_scale,
                    self.rank_u_raw[task] / self.rank_scale,
                    float(self.done_mask[task]),
                    float(self.ready_mask[task]),
                    start_norm,
                    finish_norm,
                ],
                dtype=np.float32,
            )

        return x

    def _normalized_dag_energy(self) -> float:
        """Return graph energy as energy(DAG) / energy(max v-f)."""
        return float(self.total_energy) / max(float(self.max_vf_dag_energy), 1e-6)

    def _edge_index_and_mask(self) -> Tuple[np.ndarray, np.ndarray]:
        edge_index = np.zeros((2, self.max_edges), dtype=np.int64)
        edge_mask = np.zeros(self.max_edges, dtype=np.int8)

        for k, (u, v) in enumerate(self.edges[: self.max_edges]):
            edge_index[0, k] = int(u)
            edge_index[1, k] = int(v)
            edge_mask[k] = 1

        return edge_index, edge_mask

    def _obs(self) -> Dict[str, np.ndarray]:
        self._update_ready_mask()

        edge_index, edge_mask = self._edge_index_and_mask()

        node_mask = np.zeros(self.max_tasks, dtype=np.int8)
        node_mask[: self.num_tasks] = 1

        core_features = np.zeros(
            (self.num_cores, self.core_feature_dim),
            dtype=np.float32,
        )

        for core, proc_type in enumerate(self.processor_map):
            core_features[core] = np.asarray(
                [
                    float(proc_type),
                    self.core_available_time[core] / self.time_scale,
                    len(self.core_intervals[core]) / max(self.num_tasks, 1),
                ],
                dtype=np.float32,
            )

        global_features = np.asarray(
            [
                self.steps / max(self.num_tasks, 1),
                float(np.sum(self.done_mask[: self.num_tasks])) / max(self.num_tasks, 1),
                self._normalized_dag_energy(),
                self.total_tardiness / self.time_scale,
                self.makespan / self.time_scale,
                float(np.sum(self.ready_mask[: self.num_tasks])) / max(self.num_tasks, 1),
            ],
            dtype=np.float32,
        )

        return {
            "node_features": self._node_features(),
            "edge_index": edge_index,
            "edge_mask": edge_mask,
            "node_mask": node_mask,
            "core_features": core_features,
            "global_features": global_features,
            "action_mask": self._action_mask(),
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        if self.sample_dags and len(self.dag_records) > 1:
            idx = int(self.np_random.integers(0, len(self.dag_records)))
            record = self.dag_records[idx]
        else:
            record = self.dag_records[0]

        self._load_dag(record)

        self.done_mask = np.zeros(self.max_tasks, dtype=bool)
        self.ready_mask = np.zeros(self.max_tasks, dtype=bool)

        self.assigned = {}
        self.start = {}
        self.finish = {}

        self.core_intervals = [[] for _ in range(self.num_cores)]
        self.core_available_time = np.zeros(self.num_cores, dtype=np.float32)

        self.total_energy = 0.0
        self.total_tardiness = 0.0
        self.makespan = 0.0
        self.steps = 0

        obs = self._obs()

        return obs, {
            "csv_path": self.current_record[0],
            "row_index": self.current_record[1],
        }

    def step(self, action: int):
        action = int(action)

        task = action // self.num_cores
        core = action % self.num_cores

        valid = bool(
            0 <= task < self.num_tasks
            and 0 <= core < self.num_cores
            and self.ready_mask[task]
            and not self.done_mask[task]
        )

        if not valid:
            obs = self._obs()
            return (
                obs,
                -self.invalid_action_penalty,
                False,
                False,
                {
                    "invalid_action": True,
                    "task": task,
                    "core": core,
                },
            )

        prev_energy = self.total_energy
        prev_makespan = self.makespan      # <-- added to compute delta

        start, finish = self.estimate_start_finish(task, core)

        proc_type = self.processor_map[core]
        energy = float(self.energy_mat[task, proc_type])

        # Tardiness is calculated only as a metric (not used in reward)
        tardiness = max(0.0, finish - float(self.deadlines[task]))

        self.assigned[task] = core
        self.start[task] = start
        self.finish[task] = finish

        self.core_intervals[core].append((start, finish, task))
        self.core_intervals[core].sort(key=lambda x: x[0])

        self.core_available_time[core] = max(
            self.core_available_time[core],
            finish,
        )

        self.done_mask[task] = True
        self.steps += 1

        self.total_energy += energy
        self.total_tardiness += tardiness
        self.makespan = max(self.makespan, finish)

        # ========== REWARD CALCULATION ==========
        # 1. Compute QoS for this task
        di = float(self.deadlines[task])
        fi = finish
        x = self.qos_factor

        if fi <= di:
            qos = 1.0
        elif fi <= x * di:
            qos = (x * di - fi) / ((x - 1) * di)
        else:
            qos = 0.0

        # 2. Energy term
        max_e = self.max_energy_global
        if max_e <= 0:
            max_e = 1e-6

        if self.reward_proposal == "A":
            # reward = QoS * (max_global_energy / actual_energy)
            energy_term = max_e / (energy + 1e-8)
        else:   # proposal B
            # reward = QoS * exp(-actual_energy / max_global_energy)
            energy_term = np.exp(-energy / max_e)

        reward = qos * energy_term

        # 3. Makespan penalty (new)
        delta_makespan = self.makespan - prev_makespan
        makespan_penalty = 0.001 * (delta_makespan / self.time_scale)
        reward = reward - makespan_penalty

        # Optional clipping to prevent extreme values
        reward = np.clip(reward, -10.0, 10.0)
        # ===========================================

        # For info logging, we still can store delta values if needed
        delta_energy = self.total_energy - prev_energy

        terminated = bool(np.all(self.done_mask[: self.num_tasks]))
        obs = self._obs()

        info = {
            "invalid_action": False,
            "task": task,
            "core": core,
            "start": start,
            "finish": finish,
            "energy": energy,
            "tardiness": tardiness,
            "delta_energy": delta_energy,
            "delta_makespan": delta_makespan,
            "makespan": self.makespan,
            "total_energy": self._normalized_dag_energy(),
            "total_energy_raw": self.total_energy,
            "max_vf_dag_energy": self.max_vf_dag_energy,
            "total_tardiness": self.total_tardiness,
            "qos": qos,
            "reward_proposal": self.reward_proposal,
        }

        return obs, float(reward), terminated, False, info

    def get_schedule(self):
        assigned = np.full(self.num_tasks, -1, dtype=int)
        start = np.zeros(self.num_tasks, dtype=np.float32)
        finish = np.zeros(self.num_tasks, dtype=np.float32)

        for task in range(self.num_tasks):
            assigned[task] = int(self.assigned.get(task, -1))
            start[task] = float(self.start.get(task, 0.0))
            finish[task] = float(self.finish.get(task, 0.0))

        return assigned, start, finish

    def get_metrics(self) -> Dict[str, float]:
        return {
            "makespan": float(self.makespan),
            "total_tardiness": float(self.total_tardiness),
            "total_energy": self._normalized_dag_energy(),
            "total_energy_raw": float(self.total_energy),
            "max_vf_dag_energy": float(self.max_vf_dag_energy),
        }

    def render(self):
        print(
            f"DAG={Path(self.current_record[0]).name}:{self.current_record[1]} "
            f"steps={self.steps}/{self.num_tasks} "
            f"makespan={self.makespan:.3f} "
            f"energy={self._normalized_dag_energy():.6f} "
            f"raw_energy={self.total_energy:.3f} "
            f"tardiness={self.total_tardiness:.3f}"
        )