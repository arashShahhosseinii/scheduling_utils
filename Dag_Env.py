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
        Discrete(max_tasks * num_cores)

        action = task * num_cores + core

    Observation:
        node_features:   (max_tasks, 12)
        edge_index:      (2, max_edges)
        edge_mask:       (max_edges,)
        node_mask:       (max_tasks,)
        core_features:   (num_cores, 3)
        global_features: (6,)
        action_mask:     (max_tasks * num_cores,)

    QoS:
        QoS = 1
            if Fi <= Di

        QoS = (x * Di - Fi) / ((x - 1) * Di)
            if Di < Fi <= x * Di

        QoS = 0
            if Fi > x * Di

    Reward:
        energy_reward = QoS * energy_term

        makespan_penalty =
            delta_makespan / rank_scale

        reward =
            wE * energy_reward
            - wM * makespan_penalty
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        csv_path: Optional[Union[str, Path]] = None,
        row_index: int = 0,
        csv_paths: Optional[Sequence[Union[str, Path]]] = None,
        dag_records: Optional[
            Sequence[Tuple[Union[str, Path], int]]
        ] = None,
        processor_map: Optional[List[int]] = None,
        reward_weights: Tuple[float, float] = (0.5, 0.5),
        invalid_action_penalty: float = 2.0,
        max_tasks: Optional[int] = None,
        max_edges: Optional[int] = None,
        sample_dags: bool = True,
        seed: Optional[int] = None,
        qos_factor: float = 1.25,
        reward_proposal: str = "A",
    ) -> None:
        super().__init__()

        if processor_map is None:
            processor_map = [0, 1]

        self.processor_map = list(processor_map)
        self.num_cores = len(self.processor_map)

        if self.num_cores <= 0:
            raise ValueError(
                "processor_map must contain at least one core."
            )

        if any(
            processor_type not in (0, 1)
            for processor_type in self.processor_map
        ):
            raise ValueError(
                "processor_map values must be 0=A7 or 1=A12."
            )

        self.wE, self.wM = map(float, reward_weights)

        if self.wE < 0.0 or self.wM < 0.0:
            raise ValueError(
                "Reward weights must be non-negative."
            )

        self.invalid_action_penalty = float(
            invalid_action_penalty
        )
        self.sample_dags = bool(sample_dags)
        self._initial_seed = seed
        self.np_random = np.random.default_rng(seed)

        self.qos_factor = float(qos_factor)

        if self.qos_factor <= 1.0:
            raise ValueError(
                "qos_factor must be greater than 1.0."
            )

        self.reward_proposal = reward_proposal.upper()

        if self.reward_proposal not in ("A", "B"):
            raise ValueError(
                "reward_proposal must be 'A' or 'B'."
            )

        if dag_records is not None:
            self.dag_records = [
                (str(path), int(index))
                for path, index in dag_records
            ]
        elif csv_paths is not None:
            self.dag_records = CreateDAG.list_records(
                csv_paths
            )
        elif csv_path is not None:
            self.dag_records = [
                (str(csv_path), int(row_index))
            ]
        else:
            raise ValueError(
                "Provide csv_path, csv_paths, or dag_records."
            )

        sizes: List[int] = []
        edge_sizes: List[int] = []

        for path, index in self.dag_records:
            record = CreateDAG.read_record(
                path,
                index,
            )
            sizes.append(record.num_subtasks)
            edge_sizes.append(len(record.edges))

        self.max_tasks = int(
            max_tasks or max(sizes)
        )
        self.max_edges = int(
            max_edges or max(max(edge_sizes), 1)
        )

        if self.max_tasks < max(sizes):
            raise ValueError(
                "max_tasks is smaller than at least one DAG."
            )

        if self.max_edges < max(edge_sizes):
            raise ValueError(
                "max_edges is smaller than at least one "
                "DAG edge count."
            )

        self.node_feature_dim = 12
        self.core_feature_dim = 3
        self.global_feature_dim = 6

        self.action_space = spaces.Discrete(
            self.max_tasks * self.num_cores
        )

        self.observation_space = spaces.Dict(
            {
                "node_features": spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(
                        self.max_tasks,
                        self.node_feature_dim,
                    ),
                    dtype=np.float32,
                ),
                "edge_index": spaces.Box(
                    0,
                    self.max_tasks - 1,
                    shape=(2, self.max_edges),
                    dtype=np.int64,
                ),
                "edge_mask": spaces.MultiBinary(
                    self.max_edges
                ),
                "node_mask": spaces.MultiBinary(
                    self.max_tasks
                ),
                "core_features": spaces.Box(
                    0.0,
                    np.inf,
                    shape=(
                        self.num_cores,
                        self.core_feature_dim,
                    ),
                    dtype=np.float32,
                ),
                "global_features": spaces.Box(
                    0.0,
                    np.inf,
                    shape=(self.global_feature_dim,),
                    dtype=np.float32,
                ),
                "action_mask": spaces.MultiBinary(
                    self.max_tasks * self.num_cores
                ),
            }
        )

        self.dag: CreateDAG
        self.G: nx.DiGraph
        self.num_tasks = 0
        self.edges: List[Tuple[int, int]] = []

        self.exec_times = np.zeros(
            (1, 2),
            dtype=np.float32,
        )
        self.energy_mat = np.zeros(
            (1, 2),
            dtype=np.float32,
        )

        self.deadlines = np.ones(
            1,
            dtype=np.float32,
        )
        self.periods = np.ones(
            1,
            dtype=np.float32,
        )
        self.m_i = np.ones(
            1,
            dtype=np.float32,
        )
        self.rank_u_raw = np.ones(
            1,
            dtype=np.float32,
        )

        self.time_scale = 1.0
        self.energy_scale = 1.0
        self.period_scale = 1.0
        self.mi_scale = 1.0
        self.rank_scale = 1.0

        self.max_energy_global = 1.0
        self.max_vf_dag_energy = 1.0

        self.done_mask: np.ndarray
        self.ready_mask: np.ndarray

        self.assigned: Dict[int, int]
        self.start: Dict[int, float]
        self.finish: Dict[int, float]

        self.core_intervals: List[
            List[Tuple[float, float, int]]
        ]
        self.core_available_time: np.ndarray

        self.qos_values: List[float] = []

        self.total_energy = 0.0
        self.total_tardiness = 0.0
        self.makespan = 0.0
        self.steps = 0

        self.current_record: Tuple[str, int] = (
            self.dag_records[0]
        )

        self.reset(seed=seed)

    def _load_dag(
        self,
        record: Tuple[str, int],
    ) -> None:
        self.current_record = (
            str(record[0]),
            int(record[1]),
        )

        self.dag = CreateDAG(
            self.current_record[0],
            self.current_record[1],
        )

        self.G = self.dag.graph
        self.edges = list(self.G.edges())
        self.num_tasks = int(
            self.G.number_of_nodes()
        )

        self.exec_times = np.stack(
            [
                np.asarray(
                    self.dag.a7_times,
                    dtype=np.float32,
                ),
                np.asarray(
                    self.dag.a12_times,
                    dtype=np.float32,
                ),
            ],
            axis=1,
        )

        self.energy_mat = np.stack(
            [
                np.asarray(
                    self.dag.a7_energy,
                    dtype=np.float32,
                ),
                np.asarray(
                    self.dag.a12_energy,
                    dtype=np.float32,
                ),
            ],
            axis=1,
        )

        self.deadlines = np.asarray(
            self.dag.deadlines,
            dtype=np.float32,
        )
        self.periods = np.asarray(
            self.dag.periods,
            dtype=np.float32,
        )
        self.m_i = np.asarray(
            self.dag.m_i_list,
            dtype=np.float32,
        )

        self.rank_u_raw = self._compute_upward_ranks(
            self.G,
            self.exec_times,
            self.processor_map,
        ).astype(np.float32)

        available_processor_types = sorted(
            set(self.processor_map)
        )

        available_energy = self.energy_mat[
            :,
            available_processor_types,
        ]

        self.max_energy_global = (
            float(np.max(available_energy))
            if available_energy.size
            else 1.0
        )

        self.max_vf_dag_energy = max(
            (
                float(
                    np.sum(
                        np.max(
                            available_energy,
                            axis=1,
                        )
                    )
                )
                if available_energy.size
                else 1.0
            ),
            1e-6,
        )

        max_deadline = (
            float(np.max(self.deadlines))
            if self.deadlines.size
            else 1.0
        )

        max_execution_time = (
            float(np.max(self.exec_times))
            if self.exec_times.size
            else 1.0
        )

        self.time_scale = max(
            max_deadline,
            self.num_tasks * max_execution_time,
            1e-6,
        )

        self.energy_scale = max(
            float(np.max(self.energy_mat)),
            1e-6,
        )

        self.period_scale = max(
            float(np.max(self.periods)),
            1e-6,
        )

        self.mi_scale = max(
            float(np.max(self.m_i)),
            1e-6,
        )

        self.rank_scale = max(
            float(np.max(self.rank_u_raw)),
            1e-6,
        )

    @staticmethod
    def _compute_upward_ranks(
        graph: nx.DiGraph,
        execution_times: np.ndarray,
        processor_map: List[int],
    ) -> np.ndarray:
        number_of_tasks = graph.number_of_nodes()
        processor_types = sorted(
            set(processor_map)
        )

        average_execution_time = np.zeros(
            number_of_tasks,
            dtype=float,
        )

        for task in range(number_of_tasks):
            average_execution_time[task] = float(
                np.mean(
                    execution_times[
                        task,
                        processor_types,
                    ]
                )
            )

        upward_rank = np.zeros(
            number_of_tasks,
            dtype=float,
        )

        for node in reversed(
            list(nx.topological_sort(graph))
        ):
            successors = list(
                graph.successors(node)
            )

            if not successors:
                upward_rank[node] = (
                    average_execution_time[node]
                )
            else:
                upward_rank[node] = (
                    average_execution_time[node]
                    + max(
                        upward_rank[successor]
                        for successor in successors
                    )
                )

        return upward_rank

    @staticmethod
    def _find_earliest_insertion_start(
        intervals: List[
            Tuple[float, float, int]
        ],
        ready_time: float,
        duration: float,
    ) -> float:
        if not intervals:
            return float(ready_time)

        intervals = sorted(
            intervals,
            key=lambda interval: interval[0],
        )

        candidate = float(ready_time)

        if (
            candidate + duration
            <= intervals[0][0]
        ):
            return candidate

        for index in range(
            len(intervals) - 1
        ):
            candidate = max(
                float(ready_time),
                intervals[index][1],
            )

            if (
                candidate + duration
                <= intervals[index + 1][0]
            ):
                return candidate

        return max(
            float(ready_time),
            intervals[-1][1],
        )

    def ready_time_of(
        self,
        task: int,
    ) -> float:
        predecessors = list(
            self.G.predecessors(task)
        )

        if not predecessors:
            return 0.0

        return float(
            max(
                self.finish[predecessor]
                for predecessor in predecessors
            )
        )

    def estimate_start_finish(
        self,
        task: int,
        core_index: int,
    ) -> Tuple[float, float]:
        processor_type = self.processor_map[
            core_index
        ]

        duration = float(
            self.exec_times[
                task,
                processor_type,
            ]
        )

        ready_time = self.ready_time_of(task)

        start = self._find_earliest_insertion_start(
            self.core_intervals[core_index],
            ready_time,
            duration,
        )

        return (
            float(start),
            float(start + duration),
        )

    def best_core_for_task(
        self,
        task: int,
    ) -> int:
        best_core = 0
        best_finish = float("inf")

        for core in range(self.num_cores):
            _, finish = self.estimate_start_finish(
                task,
                core,
            )

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
                self.done_mask[predecessor]
                for predecessor
                in self.G.predecessors(task)
            )

    def _action_mask(self) -> np.ndarray:
        mask = np.zeros(
            self.max_tasks * self.num_cores,
            dtype=np.int8,
        )

        for task in range(self.num_tasks):
            if (
                self.ready_mask[task]
                and not self.done_mask[task]
            ):
                for core in range(
                    self.num_cores
                ):
                    action = (
                        task * self.num_cores
                        + core
                    )
                    mask[action] = 1

        return mask

    def _node_features(self) -> np.ndarray:
        features = np.zeros(
            (
                self.max_tasks,
                self.node_feature_dim,
            ),
            dtype=np.float32,
        )

        for task in range(self.num_tasks):
            finish_normalized = (
                float(
                    self.finish.get(
                        task,
                        0.0,
                    )
                )
                / self.time_scale
            )

            start_normalized = (
                float(
                    self.start.get(
                        task,
                        0.0,
                    )
                )
                / self.time_scale
            )

            features[task] = np.asarray(
                [
                    (
                        self.periods[task]
                        / self.period_scale
                    ),
                    (
                        self.deadlines[task]
                        / self.time_scale
                    ),
                    (
                        self.m_i[task]
                        / self.mi_scale
                    ),
                    (
                        self.exec_times[task, 0]
                        / self.time_scale
                    ),
                    (
                        self.exec_times[task, 1]
                        / self.time_scale
                    ),
                    (
                        self.energy_mat[task, 0]
                        / self.energy_scale
                    ),
                    (
                        self.energy_mat[task, 1]
                        / self.energy_scale
                    ),
                    (
                        self.rank_u_raw[task]
                        / self.rank_scale
                    ),
                    float(self.done_mask[task]),
                    float(self.ready_mask[task]),
                    start_normalized,
                    finish_normalized,
                ],
                dtype=np.float32,
            )

        return features

    def _normalized_dag_energy(self) -> float:
        """
        Return graph energy as:

            energy(DAG) / energy(max v-f)
        """
        return (
            float(self.total_energy)
            / max(
                float(
                    self.max_vf_dag_energy
                ),
                1e-6,
            )
        )

    def _edge_index_and_mask(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        edge_index = np.zeros(
            (2, self.max_edges),
            dtype=np.int64,
        )

        edge_mask = np.zeros(
            self.max_edges,
            dtype=np.int8,
        )

        for index, (source, target) in enumerate(
            self.edges[: self.max_edges]
        ):
            edge_index[0, index] = int(source)
            edge_index[1, index] = int(target)
            edge_mask[index] = 1

        return edge_index, edge_mask

    def _calculate_qos(
        self,
        finish_time: float,
        deadline: float,
    ) -> float:
        if deadline <= 0.0:
            return 0.0

        if finish_time <= deadline:
            return 1.0

        upper_limit = (
            self.qos_factor * deadline
        )

        if finish_time <= upper_limit:
            qos = (
                upper_limit - finish_time
            ) / (
                (
                    self.qos_factor - 1.0
                )
                * deadline
            )

            return float(
                np.clip(
                    qos,
                    0.0,
                    1.0,
                )
            )

        return 0.0

    def _qos_metrics(self) -> Dict[str, float]:
        if not self.qos_values:
            return {
                "average_qos": 0.0,
                "minimum_qos": 0.0,
                "maximum_qos": 0.0,
                "qos_std": 0.0,
                "on_time_task_ratio": 0.0,
                "zero_qos_task_ratio": 0.0,
            }

        qos_array = np.asarray(
            self.qos_values,
            dtype=np.float64,
        )

        return {
            "average_qos": float(
                np.mean(qos_array)
            ),
            "minimum_qos": float(
                np.min(qos_array)
            ),
            "maximum_qos": float(
                np.max(qos_array)
            ),
            "qos_std": float(
                np.std(qos_array)
            ),
            "on_time_task_ratio": float(
                np.mean(
                    np.isclose(
                        qos_array,
                        1.0,
                        atol=1e-8,
                    )
                )
            ),
            "zero_qos_task_ratio": float(
                np.mean(
                    qos_array <= 1e-8
                )
            ),
        }

    def _obs(self) -> Dict[str, np.ndarray]:
        self._update_ready_mask()

        edge_index, edge_mask = (
            self._edge_index_and_mask()
        )

        node_mask = np.zeros(
            self.max_tasks,
            dtype=np.int8,
        )
        node_mask[: self.num_tasks] = 1

        core_features = np.zeros(
            (
                self.num_cores,
                self.core_feature_dim,
            ),
            dtype=np.float32,
        )

        for core, processor_type in enumerate(
            self.processor_map
        ):
            core_features[core] = np.asarray(
                [
                    float(processor_type),
                    (
                        self.core_available_time[core]
                        / self.time_scale
                    ),
                    (
                        len(
                            self.core_intervals[core]
                        )
                        / max(
                            self.num_tasks,
                            1,
                        )
                    ),
                ],
                dtype=np.float32,
            )

        global_features = np.asarray(
            [
                (
                    self.steps
                    / max(
                        self.num_tasks,
                        1,
                    )
                ),
                (
                    float(
                        np.sum(
                            self.done_mask[
                                : self.num_tasks
                            ]
                        )
                    )
                    / max(
                        self.num_tasks,
                        1,
                    )
                ),
                self._normalized_dag_energy(),
                (
                    self.total_tardiness
                    / self.time_scale
                ),
                (
                    self.makespan
                    / self.time_scale
                ),
                (
                    float(
                        np.sum(
                            self.ready_mask[
                                : self.num_tasks
                            ]
                        )
                    )
                    / max(
                        self.num_tasks,
                        1,
                    )
                ),
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
            self.np_random = (
                np.random.default_rng(seed)
            )

        if (
            self.sample_dags
            and len(self.dag_records) > 1
        ):
            record_index = int(
                self.np_random.integers(
                    0,
                    len(self.dag_records),
                )
            )
            record = self.dag_records[
                record_index
            ]
        else:
            record = self.dag_records[0]

        self._load_dag(record)

        self.done_mask = np.zeros(
            self.max_tasks,
            dtype=bool,
        )
        self.ready_mask = np.zeros(
            self.max_tasks,
            dtype=bool,
        )

        self.assigned = {}
        self.start = {}
        self.finish = {}

        self.core_intervals = [
            []
            for _ in range(self.num_cores)
        ]

        self.core_available_time = np.zeros(
            self.num_cores,
            dtype=np.float32,
        )

        self.qos_values = []

        self.total_energy = 0.0
        self.total_tardiness = 0.0
        self.makespan = 0.0
        self.steps = 0

        observation = self._obs()

        return observation, {
            "csv_path": self.current_record[0],
            "row_index": self.current_record[1],
            "qos_factor": self.qos_factor,
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
            observation = self._obs()

            return (
                observation,
                -self.invalid_action_penalty,
                False,
                False,
                {
                    "invalid_action": True,
                    "task": task,
                    "core": core,
                },
            )

        previous_energy = self.total_energy
        previous_makespan = self.makespan

        start, finish = (
            self.estimate_start_finish(
                task,
                core,
            )
        )

        processor_type = self.processor_map[
            core
        ]

        energy = float(
            self.energy_mat[
                task,
                processor_type,
            ]
        )

        deadline = float(
            self.deadlines[task]
        )

        tardiness = max(
            0.0,
            finish - deadline,
        )

        self.assigned[task] = core
        self.start[task] = start
        self.finish[task] = finish

        self.core_intervals[core].append(
            (
                start,
                finish,
                task,
            )
        )

        self.core_intervals[core].sort(
            key=lambda interval: interval[0]
        )

        self.core_available_time[core] = max(
            self.core_available_time[core],
            finish,
        )

        self.done_mask[task] = True
        self.steps += 1

        self.total_energy += energy
        self.total_tardiness += tardiness
        self.makespan = max(
            self.makespan,
            finish,
        )

        qos = self._calculate_qos(
            finish_time=finish,
            deadline=deadline,
        )

        self.qos_values.append(qos)

        maximum_energy = max(
            self.max_energy_global,
            1e-6,
        )

        if self.reward_proposal == "A":
            energy_term = (
                maximum_energy
                / max(
                    energy,
                    1e-8,
                )
            )
        else:
            energy_term = np.exp(
                -energy / maximum_energy
            )

        energy_reward = (
            qos * energy_term
        )

        delta_makespan = (
            self.makespan
            - previous_makespan
        )

        # Ablation experiment:
        # reward uses only QoS and energy.
        # Makespan is still reported as a metric,
        # but it is not part of the reward.
        makespan_penalty = 0.0

        reward = energy_reward

        reward = float(
            np.clip(
                reward,
                -10.0,
                10.0,
            )
        )

        delta_energy = (
            self.total_energy
            - previous_energy
        )

        terminated = bool(
            np.all(
                self.done_mask[
                    : self.num_tasks
                ]
            )
        )

        observation = self._obs()
        qos_metrics = self._qos_metrics()

        info = {
            "invalid_action": False,
            "task": task,
            "core": core,
            "start": start,
            "finish": finish,
            "deadline": deadline,
            "energy": energy,
            "tardiness": tardiness,
            "delta_energy": delta_energy,
            "delta_makespan": delta_makespan,
            "makespan": self.makespan,
            "total_energy": (
                self._normalized_dag_energy()
            ),
            "total_energy_raw": (
                self.total_energy
            ),
            "max_vf_dag_energy": (
                self.max_vf_dag_energy
            ),
            "total_tardiness": (
                self.total_tardiness
            ),
            "qos": qos,
            "average_qos": qos_metrics[
                "average_qos"
            ],
            "energy_term": float(
                energy_term
            ),
            "energy_reward": float(
                energy_reward
            ),
            "makespan_penalty": float(
                makespan_penalty
            ),
            "reward_proposal": (
                self.reward_proposal
            ),
            "qos_factor": self.qos_factor,
            "reward_weight_energy": self.wE,
            "reward_weight_makespan": self.wM,
        }

        return (
            observation,
            reward,
            terminated,
            False,
            info,
        )

    def get_schedule(
        self,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        assigned = np.full(
            self.num_tasks,
            -1,
            dtype=int,
        )

        start = np.zeros(
            self.num_tasks,
            dtype=np.float32,
        )

        finish = np.zeros(
            self.num_tasks,
            dtype=np.float32,
        )

        for task in range(self.num_tasks):
            assigned[task] = int(
                self.assigned.get(
                    task,
                    -1,
                )
            )

            start[task] = float(
                self.start.get(
                    task,
                    0.0,
                )
            )

            finish[task] = float(
                self.finish.get(
                    task,
                    0.0,
                )
            )

        return assigned, start, finish

    def get_metrics(
        self,
    ) -> Dict[str, float]:
        metrics = {
            "makespan": float(
                self.makespan
            ),
            "total_tardiness": float(
                self.total_tardiness
            ),
            "total_energy": (
                self._normalized_dag_energy()
            ),
            "total_energy_raw": float(
                self.total_energy
            ),
            "max_vf_dag_energy": float(
                self.max_vf_dag_energy
            ),
        }

        metrics.update(
            self._qos_metrics()
        )

        return metrics

    def render(self) -> None:
        qos_metrics = self._qos_metrics()

        print(
            f"DAG="
            f"{Path(self.current_record[0]).name}:"
            f"{self.current_record[1]} "
            f"steps={self.steps}/{self.num_tasks} "
            f"makespan={self.makespan:.3f} "
            f"energy="
            f"{self._normalized_dag_energy():.6f} "
            f"raw_energy={self.total_energy:.3f} "
            f"average_qos="
            f"{qos_metrics['average_qos']:.6f} "
            f"tardiness="
            f"{self.total_tardiness:.3f}"
        )