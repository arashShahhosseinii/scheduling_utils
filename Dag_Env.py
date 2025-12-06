import gymnasium as gym
from gymnasium import spaces
import numpy as np

from Create_Dag import CreateDAG


class DagSchedulingEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, csv_path: str):
        super().__init__()

        # 1) Build the DAG and extract node features using CreateDAG
        self.dag = CreateDAG(csv_path)
        self.num_subtasks = self.dag.num_subtasks

        # 2) Convert node-level features to numpy arrays
        self.periods = np.array(self.dag.periods, dtype=np.float32)
        self.deadlines = np.array(self.dag.deadlines, dtype=np.float32)
        self.m_i = np.array(self.dag.m_i_list, dtype=np.float32)
        
        # proc1 (A7)
        self.proc1 = np.array(self.dag.a7_times, dtype=np.float32)
        # proc2 (A12)
        self.proc2 = np.array(self.dag.a12_times, dtype=np.float32)
        # ---------------------------------------------

        # 3) Compute normalization factors (max over nodes)
        max_period = float(self.periods.max()) if self.periods.size > 0 else 1.0
        max_deadline = float(self.deadlines.max()) if self.deadlines.size > 0 else 1.0
        max_m_i = float(self.m_i.max()) if self.m_i.size > 0 else 1.0
        max_p1 = float(self.proc1.max()) if self.proc1.size > 0 else 1.0
        max_p2 = float(self.proc2.max()) if self.proc2.size > 0 else 1.0

        # 4) Build normalized feature matrix for each node:
        period_norm = self.periods / (max_period if max_period > 0 else 1.0)
        deadline_norm = self.deadlines / (max_deadline if max_deadline > 0 else 1.0)
        m_i_norm = self.m_i / (max_m_i if max_m_i > 0 else 1.0)
        proc1_norm = self.proc1 / (max_p1 if max_p1 > 0 else 1.0)
        proc2_norm = self.proc2 / (max_p2 if max_p2 > 0 else 1.0)

        # Stack into shape (num_subtasks, 5)
        self.features = np.stack(
            [period_norm, deadline_norm, m_i_norm, proc1_norm, proc2_norm],
            axis=1,
        )

        # 5) Define observation and action spaces
        self.observation_space = spaces.Dict(
            {
                "features": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.num_subtasks, 5),
                    dtype=np.float32,
                ),
                "done_mask": spaces.MultiBinary(self.num_subtasks),
                "ready_mask": spaces.MultiBinary(self.num_subtasks),
            }
        )

        self.action_space = spaces.Discrete(self.num_subtasks)

        # 6) Internal state variables (reset every episode)
        self.done_mask = np.zeros(self.num_subtasks, dtype=bool)
        self.ready_mask = np.zeros(self.num_subtasks, dtype=bool)

        self._update_ready_mask()

    # ---------- Helper methods ----------

    def _update_ready_mask(self) -> None:
        for node in range(self.num_subtasks):
            if self.done_mask[node]:
                self.ready_mask[node] = False
                continue

            preds = list(self.dag.graph.predecessors(node))

            if all(self.done_mask[p] for p in preds):
                self.ready_mask[node] = True
            else:
                self.ready_mask[node] = False

    def _get_obs(self):
        return {
            "features": self.features.copy(),
            "done_mask": self.done_mask.astype(np.int8),
            "ready_mask": self.ready_mask.astype(np.int8),
        }

    # ---------- Gymnasium core methods ----------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.done_mask = np.zeros(self.num_subtasks, dtype=bool)
        self.ready_mask = np.zeros(self.num_subtasks, dtype=bool)
        self._update_ready_mask()
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):

        # 1) Validate the action
        invalid = (
            action < 0
            or action >= self.num_subtasks
            or self.done_mask[action]
            or not self.ready_mask[action]
        )

        if invalid:
            reward = -1.0
            terminated = False
            truncated = False
            info = {"invalid_action": True}
            return self._get_obs(), reward, terminated, truncated, info

        # 2) Valid action: "execute" the chosen node
        duration = float(self.proc1[action])

        self.done_mask[action] = True

        self._update_ready_mask()

        reward = -duration

        # 3) Check if all nodes are done
        terminated = bool(self.done_mask.all())
        truncated = False

        info = {}
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        done_nodes = [i for i, d in enumerate(self.done_mask) if d]
        ready_nodes = [i for i, r in enumerate(self.ready_mask) if r]
        print(f"Done nodes: {done_nodes}")
        print(f"Ready nodes: {ready_nodes}")