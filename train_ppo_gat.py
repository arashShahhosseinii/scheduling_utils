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


# ============================================================
# Manual single-dataset training selection
# ============================================================
# Change this value only if you want to train on another CSV.
# Available options:
#   "dataset1" -> dag_dataset1_a7_a12.csv
#   "dataset2" -> dag_dataset2_a7_a12.csv
#   "dataset3" -> dag_dataset3_a7_a12.csv
#
# Current requested setting: train only on dataset1.
# ============================================================

SELECTED_DATASET_KEY = "dataset1"
SELECTED_ROW_INDEX = 0

DATASETS = {
    "dataset1": SCRIPT_DIR / "dag_dataset1_a7_a12.csv",
    "dataset2": SCRIPT_DIR / "dag_dataset2_a7_a12.csv",
    "dataset3": SCRIPT_DIR / "dag_dataset3_a7_a12.csv",
}

if SELECTED_DATASET_KEY not in DATASETS:
    valid_keys = ", ".join(DATASETS.keys())
    raise ValueError(
        f"Invalid SELECTED_DATASET_KEY: {SELECTED_DATASET_KEY}. "
        f"Valid options are: {valid_keys}"
    )

SELECTED_DATASET_PATH = DATASETS[SELECTED_DATASET_KEY]

if not SELECTED_DATASET_PATH.exists():
    raise FileNotFoundError(f"Missing selected dataset file: {SELECTED_DATASET_PATH}")


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

# Reward weights used inside Dag_Env.py:
#   wE = weight of the QoS-energy reward component
#   wM = weight of the makespan penalty component
# Final reward:
#   reward = wE * energy_reward - wM * (delta_makespan / rank_scale)
REWARD_WEIGHTS = (0.008, 45.0)

# Requested limit: training timesteps must not be more than 200,000.
TOTAL_TIMESTEPS = 200_000
CHECKPOINT_FREQUENCY = 10_000

# QoS parameters
QOS_FACTOR = 1.20
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


def make_env() -> DagSchedulingEnv:
    """
    Create the DAG scheduling environment used for PPO training.

    This training run uses only the manually selected CSV file.
    It does not sample from all datasets.
    """
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
    print("=== Single-Dataset Training Mode ===")
    print(f"=== Selected dataset key  = {SELECTED_DATASET_KEY} ===")
    print(f"=== Selected dataset file = {SELECTED_DATASET_PATH.name} ===")
    print(f"=== Selected row index    = {SELECTED_ROW_INDEX} ===")
    print(f"=== Environment max_tasks = {env.max_tasks} ===")
    print(f"=== Action space size     = {env.action_space.n} ===")
    print(f"=== Total timesteps       = {TOTAL_TIMESTEPS} ===")
    print(f"=== Reward weights        = wE={REWARD_WEIGHTS[0]}, wM={REWARD_WEIGHTS[1]} ===")
    print(f"=== QoS factor (x)        = {QOS_FACTOR} ===")
    print(f"=== Reward proposal       = {REWARD_PROPOSAL} ===")
    print(
        "=== Reward formula        = "
        "wE * [QoS * energy_term] - wM * [delta_makespan / rank_scale] ==="
    )
    print()

    # Check environment compatibility with Gymnasium / Stable-Baselines3.
    # Warnings about Dict observations are expected.
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
        ent_coef=0.05,
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
        print(f"=== Trained only on dataset: {SELECTED_DATASET_PATH.name} ===")
        print(f"=== Saved model to: {final_path}.zip ===")

    finally:
        env.close()


if __name__ == "__main__":
    main()
