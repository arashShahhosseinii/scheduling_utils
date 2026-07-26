from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch


def compute_schedule_metrics(
    finish_times: np.ndarray,
    deadlines: np.ndarray,
) -> Tuple[float, float]:
    if finish_times.size == 0:
        return 0.0, 0.0

    makespan = float(
        np.max(finish_times)
    )

    total_tardiness = float(
        np.maximum(
            0.0,
            finish_times - deadlines,
        ).sum()
    )

    return makespan, total_tardiness


def encode_action(
    task: int,
    core: int,
    num_cores: int,
) -> int:
    return int(
        task * num_cores + core
    )


def decode_action(
    action: int,
    num_cores: int,
) -> Tuple[int, int]:
    return (
        int(action // num_cores),
        int(action % num_cores),
    )


def _ready_tasks_from_obs(
    obs: Dict[str, np.ndarray],
) -> List[int]:
    node_mask = obs.get(
        "node_mask",
        np.ones_like(
            obs["action_mask"]
        ),
    ).astype(bool)

    action_mask = obs[
        "action_mask"
    ].astype(bool)

    num_cores = int(
        len(action_mask)
        // len(node_mask)
    )

    ready: List[int] = []

    for task in range(
        len(node_mask)
    ):
        task_actions = action_mask[
            task * num_cores:
            (task + 1) * num_cores
        ]

        if (
            node_mask[task]
            and task_actions.any()
        ):
            ready.append(task)

    return ready


def choose_action_random(
    env,
    obs: Dict[str, np.ndarray],
    rng: np.random.Generator,
) -> int:
    ready = _ready_tasks_from_obs(
        obs
    )

    if not ready:
        return 0

    task = int(
        rng.choice(ready)
    )

    core = int(
        rng.integers(
            0,
            env.num_cores,
        )
    )

    return encode_action(
        task,
        core,
        env.num_cores,
    )


def choose_action_edf(
    env,
    obs: Dict[str, np.ndarray],
) -> int:
    ready = _ready_tasks_from_obs(
        obs
    )

    if not ready:
        return 0

    task = min(
        ready,
        key=lambda task_index: (
            float(
                env.deadlines[
                    task_index
                ]
            ),
            int(task_index),
        ),
    )

    core = env.best_core_for_task(
        task
    )

    return encode_action(
        task,
        core,
        env.num_cores,
    )


def choose_action_heft(
    env,
    obs: Dict[str, np.ndarray],
) -> int:
    ready = _ready_tasks_from_obs(
        obs
    )

    if not ready:
        return 0

    task = max(
        ready,
        key=lambda task_index: (
            float(
                env.rank_u_raw[
                    task_index
                ]
            ),
            -float(
                env.deadlines[
                    task_index
                ]
            ),
            -int(task_index),
        ),
    )

    core = env.best_core_for_task(
        task
    )

    return encode_action(
        task,
        core,
        env.num_cores,
    )


def choose_action_from_sb3_model(
    model,
    obs: Dict[str, np.ndarray],
    deterministic: bool = True,
    exploration_alpha: Optional[
        float
    ] = None,
) -> int:
    """
    Predict an action using the custom PPO+GAT policy.

    When deterministic=False, the policy samples from its
    exploration-aware distribution.
    """

    if exploration_alpha is not None:
        model.policy.set_exploration_alpha(
            exploration_alpha
        )

    model.policy.set_training_mode(
        False
    )

    device = model.policy.device

    boolean_keys = {
        "edge_mask",
        "node_mask",
        "action_mask",
    }

    observation_tensors: Dict[
        str,
        torch.Tensor,
    ] = {}

    for key, value in obs.items():
        array = np.asarray(value)

        if key == "edge_index":
            tensor = torch.as_tensor(
                array,
                dtype=torch.long,
                device=device,
            )
        elif key in boolean_keys:
            tensor = torch.as_tensor(
                array,
                dtype=torch.bool,
                device=device,
            )
        else:
            tensor = torch.as_tensor(
                array,
                dtype=torch.float32,
                device=device,
            )

        observation_tensors[key] = (
            tensor.unsqueeze(0)
        )

    with torch.no_grad():
        actions, _, _ = (
            model.policy.forward(
                observation_tensors,
                deterministic=deterministic,
            )
        )

    return int(
        actions.reshape(-1)[0].item()
    )


def run_policy_episode(
    env,
    policy_name: str,
    rng: Optional[
        np.random.Generator
    ] = None,
    model=None,
    ppo_deterministic: bool = True,
    ppo_exploration_alpha: Optional[
        float
    ] = None,
):
    observation, _ = env.reset()

    terminated = False
    truncated = False
    total_reward = 0.0
    step_rows = []

    if (
        policy_name
        in {"PPO_GAT", "GAT_PPO"}
    ):
        if model is None:
            raise ValueError(
                "model is required for "
                "PPO_GAT evaluation."
            )

        if ppo_exploration_alpha is not None:
            model.policy.set_exploration_alpha(
                ppo_exploration_alpha
            )

        model.policy.set_training_mode(
            False
        )

    while not (
        terminated or truncated
    ):
        if policy_name == "RANDOM":
            if rng is None:
                rng = (
                    np.random.default_rng()
                )

            action = choose_action_random(
                env,
                observation,
                rng,
            )

        elif policy_name == "EDF":
            action = choose_action_edf(
                env,
                observation,
            )

        elif policy_name == "HEFT":
            action = choose_action_heft(
                env,
                observation,
            )

        elif policy_name in {
            "PPO_GAT",
            "GAT_PPO",
        }:
            action = (
                choose_action_from_sb3_model(
                    model,
                    observation,
                    deterministic=(
                        ppo_deterministic
                    ),
                    exploration_alpha=(
                        ppo_exploration_alpha
                    ),
                )
            )

        else:
            raise ValueError(
                f"Unknown policy_name: "
                f"{policy_name}"
            )

        (
            next_observation,
            reward,
            terminated,
            truncated,
            step_info,
        ) = env.step(action)

        total_reward += float(reward)

        if not step_info.get(
            "invalid_action",
            False,
        ):
            step_rows.append(
                {
                    "action": int(action),
                    "reward": float(reward),
                    **step_info,
                }
            )

        observation = next_observation

    assigned, start, finish = (
        env.get_schedule()
    )

    metrics = env.get_metrics()

    return (
        assigned,
        start,
        finish,
        metrics,
        total_reward,
        step_rows,
    )


def validate_heft_inputs(
    graph: nx.DiGraph,
    execution_times: np.ndarray,
    processor_map: List[int],
) -> None:
    if not nx.is_directed_acyclic_graph(
        graph
    ):
        raise ValueError(
            "Input graph must be a DAG."
        )

    if (
        execution_times.ndim != 2
        or execution_times.shape[1] < 2
    ):
        raise ValueError(
            "exec_times must have shape "
            "(num_tasks, >=2)."
        )

    if any(
        processor_type not in (0, 1)
        for processor_type in processor_map
    ):
        raise ValueError(
            "processor_map values must be "
            "0=A7 or 1=A12."
        )