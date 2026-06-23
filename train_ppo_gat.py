from __future__ import annotations

from pathlib import Path
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from Dag_Env import DagSchedulingEnv
from gat_sb3_policy import MaskedGATActorCriticPolicy


try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path(sys.argv[0]).resolve().parent


DATASET_PATHS = [
    SCRIPT_DIR / "dag_dataset1_a7_a12.csv",
    SCRIPT_DIR / "dag_dataset2_a7_a12.csv",
    SCRIPT_DIR / "dag_dataset3_a7_a12.csv",
]

for path in DATASET_PATHS:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")


ARTIFACT_DIR = SCRIPT_DIR / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
LOG_DIR = ARTIFACT_DIR / "logs"
TENSORBOARD_DIR = LOG_DIR / "tensorboard"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)


SEED = 42
PROCESSOR_MAP = [0, 1]
REWARD_WEIGHTS = (0.5, 0.5)   # kept for compatibility, not used in new reward

TOTAL_TIMESTEPS = 300_000      # increased from 200_000
CHECKPOINT_FREQUENCY = 10_000

# QoS parameters (as per master's request)
QOS_FACTOR = 1.1                 # reduced from 2.0 for sharper QoS
REWARD_PROPOSAL = "A"            # "A" or "B" – choose one


def progress_bar_dependencies_available() -> bool:
    """
    Check whether Stable-Baselines3 progress-bar dependencies exist.
    """
    try:
        import rich  # noqa: F401
        from tqdm.rich import tqdm as _rich_tqdm  # noqa: F401
    except ImportError:
        return False
    return True


def make_env(sample_dags: bool = True) -> DagSchedulingEnv:
    """
    Create the DAG scheduling environment used for PPO training
    with the new QoS‑based reward.
    """
    return DagSchedulingEnv(
        csv_paths=DATASET_PATHS,
        processor_map=PROCESSOR_MAP,
        reward_weights=REWARD_WEIGHTS,
        invalid_action_penalty=2.0,
        sample_dags=sample_dags,
        seed=SEED,
        qos_factor=QOS_FACTOR,
        reward_proposal=REWARD_PROPOSAL,
    )


def main() -> None:
    env = make_env(sample_dags=True)

    print(
        f"\n=== Environment created with max_tasks = "
        f"{env.max_tasks} ==="
    )
    print(
        f"=== Action space size = "
        f"{env.action_space.n} ==="
    )
    print(
        f"=== QoS factor (x) = {QOS_FACTOR} ==="
    )
    print(
        f"=== Reward proposal = {REWARD_PROPOSAL} ==="
    )
    print(
        f"=== Reward formula: "
        f"{'QoS * (max_global_energy / actual_energy)' if REWARD_PROPOSAL == 'A' else 'QoS * exp(-actual_energy / max_global_energy)'} ==="
    )
    print()

    # Check environment (warnings about Dict observations are expected)
    check_env(env, warn=True, skip_render_check=True)

    env = Monitor(
        env,
        filename=str(LOG_DIR / "monitor.csv"),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQUENCY,
        save_path=str(CHECKPOINT_DIR),
        name_prefix="ppo_gat_scheduler",
        verbose=1,
    )

    model = PPO(
        policy=MaskedGATActorCriticPolicy,
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=8,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,            # increased from 0.01
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=SEED,
        device="cpu",
        tensorboard_log=str(TENSORBOARD_DIR),
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
        },
    )

    use_progress_bar = progress_bar_dependencies_available()

    if use_progress_bar:
        print("=== Training progress bar is enabled (tqdm + rich found). ===")
    else:
        print("=== WARNING: tqdm/rich were not found. Training will continue without the optional progress bar. ===")
        print("=== To enable it later, run: python -m pip install tqdm rich ===")

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
            progress_bar=use_progress_bar,
        )

        final_path = MODEL_DIR / "ppo_gat_scheduler"
        model.save(str(final_path))

        print()
        print("=== Training completed successfully. ===")
        print(f"=== Saved model to: {final_path}.zip ===")

    finally:
        env.close()


if __name__ == "__main__":
    main()