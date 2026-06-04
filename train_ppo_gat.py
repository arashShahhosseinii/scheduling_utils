from __future__ import annotations

from pathlib import Path
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from Dag_Env import DagSchedulingEnv
from gat_sb3_policy import MaskedMLPActorCriticPolicy


try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path(sys.argv[0]).resolve().parent


DATASET_PATHS = [
    SCRIPT_DIR / "dag_dataset1_a7_a12.csv",
    SCRIPT_DIR / "dag_dataset2_a7_a12.csv",
    SCRIPT_DIR / "dag_dataset3_a7_a12.csv",
]

# Verify all files exist
for p in DATASET_PATHS:
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")

ARTIFACT_DIR = SCRIPT_DIR / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"
LOG_DIR = ARTIFACT_DIR / "logs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
PROCESSOR_MAP = [0, 1]
REWARD_WEIGHTS = (0.5, 0.5)
TOTAL_TIMESTEPS = 200_000


def make_env(sample_dags: bool = True) -> DagSchedulingEnv:
    env = DagSchedulingEnv(
        csv_paths=DATASET_PATHS,
        processor_map=PROCESSOR_MAP,
        reward_weights=REWARD_WEIGHTS,
        invalid_action_penalty=2.0,
        sample_dags=sample_dags,
        seed=SEED,
    )
    return env


def main() -> None:
    env = make_env(sample_dags=True)
    print(f"\n=== Environment created with max_tasks = {env.max_tasks} ===")
    print(f"=== Action space size = {env.action_space.n} ===\n")
    check_env(env, warn=True, skip_render_check=True)

    env = Monitor(env, filename=str(LOG_DIR / "monitor.csv"))

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=str(MODEL_DIR / "checkpoints"),
        name_prefix="ppo_mlp_scheduler",
    )

    model = PPO(
        policy=MaskedMLPActorCriticPolicy,
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=8,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=SEED,
        device="cpu",
        tensorboard_log=str(LOG_DIR / "tensorboard"),
        policy_kwargs=dict(
            features_extractor_kwargs=dict(features_dim=256),
        ),
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback, progress_bar=True)

    final_path = MODEL_DIR / "ppo_mlp_scheduler"
    model.save(str(final_path))
    print(f"Saved model to {final_path}.zip")


if __name__ == "__main__":
    main()