from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from Dag_Env import DagSchedulingEnv
from gat_sb3_policy import MaskedGATActorCriticPolicy
from scheduling_utils import run_policy_episode


try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path(sys.argv[0]).resolve().parent


DATASET_PATHS = [
    SCRIPT_DIR / "dag_dataset(1)_a7_a12.csv",
    SCRIPT_DIR / "dag_dataset(2)_a7_a12.csv",
    SCRIPT_DIR / "dag_dataset(3)_a7_a12.csv",
]


ARTIFACT_DIR = SCRIPT_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "models" / "ppo_gat_scheduler.zip"
EVAL_DIR = ARTIFACT_DIR / "evaluation"

EVAL_DIR.mkdir(parents=True, exist_ok=True)


SEED = 42

PROCESSOR_MAP = [0, 1]

# New reward weights:
# reward = -(0.5 * delta_energy + 0.5 * delta_makespan)
REWARD_WEIGHTS = (0.5, 0.5)


def build_env(
    csv_path: Path,
    row_index: int = 0,
    sample_dags: bool = False,
) -> DagSchedulingEnv:
    env = DagSchedulingEnv(
        csv_path=csv_path,
        row_index=row_index,
        processor_map=PROCESSOR_MAP,
        reward_weights=REWARD_WEIGHTS,
        invalid_action_penalty=2.0,
        sample_dags=sample_dags,
        seed=SEED,
    )

    return env


def evaluate() -> pd.DataFrame:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "First train it with: python train_ppo_gat.py"
        )

    load_env = DagSchedulingEnv(
        csv_paths=DATASET_PATHS,
        processor_map=PROCESSOR_MAP,
        reward_weights=REWARD_WEIGHTS,
        sample_dags=False,
        seed=SEED,
    )

    model = PPO.load(
        str(MODEL_PATH),
        env=load_env,
        custom_objects={
            "policy_class": MaskedGATActorCriticPolicy,
        },
    )

    rows: List[Dict[str, float]] = []
    rng = np.random.default_rng(SEED)

    scenarios = [
        "HEFT",
        "PPO_GAT",
    ]

    for csv_path in DATASET_PATHS:
        df = pd.read_csv(csv_path)

        for row_index in range(len(df)):
            for scenario in scenarios:
                env = build_env(
                    csv_path,
                    row_index=row_index,
                    sample_dags=False,
                )

                assigned, start, finish, metrics, total_reward, step_rows = run_policy_episode(
                    env,
                    scenario,
                    rng=rng,
                    model=model if scenario == "PPO_GAT" else None,
                )

                rows.append(
                    {
                        "dataset": csv_path.name,
                        "row_index": row_index,
                        "method": scenario,
                        "makespan": metrics["makespan"],
                        "total_energy": metrics["total_energy"],
                        "total_tardiness": metrics["total_tardiness"],
                        "total_reward": total_reward,
                    }
                )

    return pd.DataFrame(rows)


def save_plots(summary: pd.DataFrame) -> None:
    methods = summary["method"].tolist()

    fig1, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(methods, summary["makespan_mean"].tolist())
    ax1.set_ylabel("Mean Makespan")
    ax1.set_title("GAT+PPO vs HEFT: QoS / Makespan")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    fig1.tight_layout()
    fig1.savefig(EVAL_DIR / "makespan_comparison.png", dpi=180)

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.bar(methods, summary["total_energy_mean"].tolist())
    ax2.set_ylabel("Mean Total Energy")
    ax2.set_title("GAT+PPO vs HEFT: Energy")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    fig2.tight_layout()
    fig2.savefig(EVAL_DIR / "energy_comparison.png", dpi=180)


def main() -> None:
    detail = evaluate()

    detail_path = EVAL_DIR / "comparison_detail.csv"
    detail.to_csv(
        detail_path,
        index=False,
    )

    summary = (
        detail.groupby("method", as_index=False)
        .agg(
            makespan_mean=("makespan", "mean"),
            makespan_std=("makespan", "std"),
            total_energy_mean=("total_energy", "mean"),
            total_energy_std=("total_energy", "std"),
            total_tardiness_mean=("total_tardiness", "mean"),
            total_reward_mean=("total_reward", "mean"),
        )
        .fillna(0.0)
    )

    summary_path = EVAL_DIR / "comparison_summary.csv"
    summary.to_csv(
        summary_path,
        index=False,
    )

    save_plots(summary)

    print("\nDetailed comparison:")
    print(detail.to_string(index=False))

    print("\nSummary:")
    print(summary.to_string(index=False))

    print(f"\nSaved: {detail_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {EVAL_DIR / 'makespan_comparison.png'}")
    print(f"Saved: {EVAL_DIR / 'energy_comparison.png'}")


if __name__ == "__main__":
    main()