from __future__ import annotations

from pathlib import Path
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.env_checker import (
    check_env,
)
from stable_baselines3.common.monitor import (
    Monitor,
)

from Dag_Env import DagSchedulingEnv
from gat_sb3_policy import (
    MaskedGATActorCriticPolicy,
)


try:
    SCRIPT_DIR = (
        Path(__file__).resolve().parent
    )
except NameError:
    SCRIPT_DIR = (
        Path(sys.argv[0]).resolve().parent
    )


SELECTED_DATASET_KEY = "dataset1"
SELECTED_ROW_INDEX = 0

DATASETS = {
    "dataset1": (
        SCRIPT_DIR
        / "dag_dataset1_a7_a12.csv"
    ),
    "dataset2": (
        SCRIPT_DIR
        / "dag_dataset2_a7_a12.csv"
    ),
    "dataset3": (
        SCRIPT_DIR
        / "dag_dataset3_a7_a12.csv"
    ),
}

if SELECTED_DATASET_KEY not in DATASETS:
    valid_keys = ", ".join(
        DATASETS.keys()
    )

    raise ValueError(
        f"Invalid SELECTED_DATASET_KEY: "
        f"{SELECTED_DATASET_KEY}. "
        f"Valid options are: {valid_keys}"
    )

SELECTED_DATASET_PATH = DATASETS[
    SELECTED_DATASET_KEY
]

if not SELECTED_DATASET_PATH.exists():
    raise FileNotFoundError(
        "Missing selected dataset file: "
        f"{SELECTED_DATASET_PATH}"
    )


ARTIFACT_DIR = (
    SCRIPT_DIR / "artifacts"
)

MODEL_DIR = (
    ARTIFACT_DIR / "models"
)

CHECKPOINT_DIR = (
    MODEL_DIR / "checkpoints"
)

LOG_DIR = (
    ARTIFACT_DIR / "logs"
)

TENSORBOARD_DIR = (
    LOG_DIR / "tensorboard"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TENSORBOARD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SEED = 42
PROCESSOR_MAP = [0, 1]

REWARD_WEIGHTS = (
    0.008,
    45.0,
)

TOTAL_TIMESTEPS = 200_000
CHECKPOINT_FREQUENCY = 10_000

QOS_FACTOR = 1.25
REWARD_PROPOSAL = "A"

EXPLORATION_ALPHA_START = 0.60
EXPLORATION_ALPHA_END = 0.01


class LinearExplorationAlphaCallback(
    BaseCallback
):
    """
    Linearly decrease the exploration coefficient.

    The coefficient is changed only at the beginning
    of a rollout. Therefore it remains constant during:

        1. rollout collection
        2. the PPO update for that rollout

    This keeps stored and recalculated log-probabilities
    consistent.
    """

    def __init__(
        self,
        alpha_start: float,
        alpha_end: float,
        total_timesteps: int,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)

        self.alpha_start = float(
            alpha_start
        )
        self.alpha_end = float(
            alpha_end
        )
        self.total_training_timesteps = max(
            int(total_timesteps),
            1,
        )
        self.current_alpha = (
            self.alpha_start
        )

        if not (
            0.0
            <= self.alpha_end
            <= self.alpha_start
            <= 1.0
        ):
            raise ValueError(
                "Expected: "
                "0 <= alpha_end <= "
                "alpha_start <= 1."
            )

    def _calculate_alpha(
        self,
    ) -> float:
        progress = min(
            max(
                self.num_timesteps
                / self.total_training_timesteps,
                0.0,
            ),
            1.0,
        )

        return (
            self.alpha_start
            + progress
            * (
                self.alpha_end
                - self.alpha_start
            )
        )

    def _apply_alpha(
        self,
    ) -> None:
        self.current_alpha = (
            self._calculate_alpha()
        )

        self.model.policy.set_exploration_alpha(
            self.current_alpha
        )

        self.logger.record(
            "exploration/alpha",
            self.current_alpha,
        )

        if self.verbose >= 2:
            print(
                "Exploration alpha = "
                f"{self.current_alpha:.6f}"
            )

    def _on_training_start(
        self,
    ) -> None:
        self._apply_alpha()

    def _on_rollout_start(
        self,
    ) -> None:
        self._apply_alpha()

    def _on_step(
        self,
    ) -> bool:
        return True


def progress_bar_dependencies_available(
) -> bool:
    try:
        import rich  # noqa: F401
        from tqdm.rich import (  # noqa: F401
            tqdm as _rich_tqdm,
        )
    except ImportError:
        return False

    return True


def make_env() -> DagSchedulingEnv:
    return DagSchedulingEnv(
        csv_path=SELECTED_DATASET_PATH,
        row_index=SELECTED_ROW_INDEX,
        processor_map=PROCESSOR_MAP,
        reward_weights=REWARD_WEIGHTS,
        invalid_action_penalty=2.0,
        sample_dags=False,
        seed=SEED,
        qos_factor=QOS_FACTOR,
        reward_proposal=REWARD_PROPOSAL,
    )


def main() -> None:
    env = make_env()

    print()
    print(
        "=== Single-Dataset Training Mode ==="
    )
    print(
        "=== Selected dataset key  = "
        f"{SELECTED_DATASET_KEY} ==="
    )
    print(
        "=== Selected dataset file = "
        f"{SELECTED_DATASET_PATH.name} ==="
    )
    print(
        "=== Selected row index    = "
        f"{SELECTED_ROW_INDEX} ==="
    )
    print(
        "=== Environment max_tasks = "
        f"{env.max_tasks} ==="
    )
    print(
        "=== Environment max_edges = "
        f"{env.max_edges} ==="
    )
    print(
        "=== Action space size     = "
        f"{env.action_space.n} ==="
    )
    print(
        "=== Total timesteps       = "
        f"{TOTAL_TIMESTEPS} ==="
    )
    print(
        "=== Reward weights        = "
        f"wE={REWARD_WEIGHTS[0]}, "
        f"wM={REWARD_WEIGHTS[1]} ==="
    )
    print(
        "=== QoS factor            = "
        f"{QOS_FACTOR} ==="
    )
    print(
        "=== Reward proposal       = "
        f"{REWARD_PROPOSAL} ==="
    )
    print(
        "=== Exploration alpha     = "
        f"{EXPLORATION_ALPHA_START} "
        f"-> {EXPLORATION_ALPHA_END} ==="
    )
    print(
        "=== Reward formula        = "
        "wE * [QoS * energy_term] "
        "- wM * "
        "[delta_makespan / rank_scale] ==="
    )
    print()

    check_env(
        env,
        warn=True,
        skip_render_check=True,
    )

    env = Monitor(
        env,
        filename=str(
            LOG_DIR / "monitor.csv"
        ),
    )

    checkpoint_callback = (
        CheckpointCallback(
            save_freq=(
                CHECKPOINT_FREQUENCY
            ),
            save_path=str(
                CHECKPOINT_DIR
            ),
            name_prefix=(
                "ppo_gat_scheduler"
            ),
            verbose=1,
        )
    )

    exploration_callback = (
        LinearExplorationAlphaCallback(
            alpha_start=(
                EXPLORATION_ALPHA_START
            ),
            alpha_end=(
                EXPLORATION_ALPHA_END
            ),
            total_timesteps=(
                TOTAL_TIMESTEPS
            ),
            verbose=0,
        )
    )

    callbacks = CallbackList(
        [
            exploration_callback,
            checkpoint_callback,
        ]
    )

    model = PPO(
        policy=(
            MaskedGATActorCriticPolicy
        ),
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=8,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=SEED,
        device="cpu",
        tensorboard_log=str(
            TENSORBOARD_DIR
        ),
        policy_kwargs={
            "gat_hidden_dim": 64,
            "gat_heads": 4,
            "gat_layers": 2,
            "core_hidden_dim": 32,
            "global_hidden_dim": 32,
            "context_hidden_dim": 64,
            "actor_hidden_dim": 128,
            "critic_hidden_dim": 128,
            "dropout": 0.10,
            "attention_dropout": 0.10,
            "exploration_alpha": (
                EXPLORATION_ALPHA_START
            ),
        },
    )

    use_progress_bar = (
        progress_bar_dependencies_available()
    )

    if use_progress_bar:
        print(
            "=== Training progress bar "
            "is enabled. ==="
        )
    else:
        print(
            "=== WARNING: tqdm/rich were "
            "not found. Training continues "
            "without the optional progress bar. ==="
        )

    try:
        model.learn(
            total_timesteps=(
                TOTAL_TIMESTEPS
            ),
            callback=callbacks,
            progress_bar=use_progress_bar,
        )

        model.policy.set_exploration_alpha(
            EXPLORATION_ALPHA_END
        )

        final_path = (
            MODEL_DIR
            / "ppo_gat_scheduler"
        )

        model.save(
            str(final_path)
        )

        print()
        print(
            "=== Training completed "
            "successfully. ==="
        )
        print(
            "=== Final exploration alpha = "
            f"{EXPLORATION_ALPHA_END} ==="
        )
        print(
            "=== Trained only on dataset: "
            f"{SELECTED_DATASET_PATH.name} ==="
        )
        print(
            "=== Saved model to: "
            f"{final_path}.zip ==="
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()