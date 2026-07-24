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


# ============================================================
# Single-dataset evaluation mode
# ============================================================
# The last training run was done only on dataset1.
# Therefore evaluation must also use dataset1 only.
#
# Important:
# The trained model observation space has:
#   max_tasks = 108
#   max_edges = 1152
#
# So this file forces both values to exactly match the trained model.
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
MODEL_PATH = ARTIFACT_DIR / "models" / "ppo_gat_scheduler.zip"
EVAL_DIR = ARTIFACT_DIR / "evaluation"

EVAL_DIR.mkdir(parents=True, exist_ok=True)


SEED = 42
PROCESSOR_MAP = [0, 1]

# These values must match the latest training setup.
REWARD_WEIGHTS = (0.008, 45.0)
QOS_FACTOR = 1.20
REWARD_PROPOSAL = "A"

# These values must match the trained model observation/action spaces.
# dataset1 has:
#   108 tasks
#   1152 edges
FORCED_MAX_TASKS = 108
FORCED_MAX_EDGES = 1152

EPS = 1e-12


def build_env(
    csv_path: Path,
    row_index: int = 0,
) -> DagSchedulingEnv:
    """
    Build an evaluation environment with exactly the same observation space
    as the trained model.

    This is critical because Stable-Baselines3 refuses to load a model
    when observation spaces do not match exactly.
    """
    env = DagSchedulingEnv(
        csv_path=csv_path,
        row_index=row_index,
        processor_map=PROCESSOR_MAP,
        reward_weights=REWARD_WEIGHTS,
        invalid_action_penalty=2.0,
        max_tasks=FORCED_MAX_TASKS,
        max_edges=FORCED_MAX_EDGES,
        sample_dags=False,
        seed=SEED,
        qos_factor=QOS_FACTOR,
        reward_proposal=REWARD_PROPOSAL,
    )
    return env


def evaluate() -> pd.DataFrame:
    """
    Evaluate HEFT and PPO_GAT only on the selected dataset.

    This avoids the max_edges mismatch problem:
        trained model: edge_index shape = (2, 1152)
        dataset2 env: edge_index shape = (2, 1527)

    Since the model was trained only on dataset1, evaluation must also be
    done only on dataset1.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "First train the model with: python train_ppo_gat.py"
        )

    dummy_env = build_env(
        SELECTED_DATASET_PATH,
        row_index=SELECTED_ROW_INDEX,
    )

    print()
    print("Loading trained PPO+GAT model...")
    print(f"Model path              = {MODEL_PATH}")
    print(f"Evaluation dataset      = {SELECTED_DATASET_PATH.name}")
    print(f"Forced max_tasks        = {FORCED_MAX_TASKS}")
    print(f"Forced max_edges        = {FORCED_MAX_EDGES}")
    print(f"Dummy observation space = {dummy_env.observation_space}")
    print()

    model = PPO.load(
        str(MODEL_PATH),
        env=dummy_env,
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

    # Evaluate only the row or rows inside the selected CSV.
    # Your current dataset files usually have one row, but this loop is safe
    # if the CSV contains multiple DAG rows.
    df = pd.read_csv(SELECTED_DATASET_PATH)

    for row_index in range(len(df)):
        for scenario in scenarios:
            env = build_env(
                SELECTED_DATASET_PATH,
                row_index=row_index,
            )

            assigned, start, finish, metrics, total_reward, step_rows = run_policy_episode(
                env,
                scenario,
                rng=rng,
                model=model if scenario == "PPO_GAT" else None,
            )

            rows.append(
                {
                    "dataset": SELECTED_DATASET_PATH.name,
                    "row_index": row_index,
                    "method": scenario,
                    "makespan": float(metrics["makespan"]),
                    "total_energy": float(metrics["total_energy"]),
                    "total_tardiness": float(metrics["total_tardiness"]),
                    "total_reward": float(total_reward),
                }
            )

            env.close()

    dummy_env.close()

    return pd.DataFrame(rows)


def percent_improvement(reference_value: float, candidate_value: float) -> float:
    """
    Percentage improvement of candidate compared with reference.

    Positive value:
        candidate is better/lower than reference.

    Negative value:
        candidate is worse/higher than reference.

    Formula:
        ((reference_value - candidate_value) / reference_value) * 100
    """
    reference_value = float(reference_value)
    candidate_value = float(candidate_value)

    if abs(reference_value) < EPS:
        return float("nan")

    return ((reference_value - candidate_value) / reference_value) * 100.0


def tradeoff_ratio(
    time_improvement_percent: float,
    energy_improvement_percent: float,
) -> float:
    """
    Ratio between time improvement and energy improvement/loss.

    It shows how much time improvement we gained per 1 percent energy change.

    Formula:
        time_improvement_percent / abs(energy_improvement_percent)
    """
    time_improvement_percent = float(time_improvement_percent)
    energy_improvement_percent = float(energy_improvement_percent)

    if np.isnan(time_improvement_percent) or np.isnan(energy_improvement_percent):
        return float("nan")

    denominator = abs(energy_improvement_percent)

    if denominator < EPS:
        if abs(time_improvement_percent) < EPS:
            return 0.0
        return float("inf") if time_improvement_percent > 0 else float("-inf")

    return time_improvement_percent / denominator


def build_improvement_report(detail: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-experiment comparison between PPO_GAT and HEFT.

    Requested metrics:
        1. time_improvement_percent
        2. energy_improvement_percent
        3. time_energy_tradeoff_ratio

    Baseline:
        HEFT

    Candidate:
        PPO_GAT
    """
    rows: List[Dict[str, float]] = []

    grouped = detail.groupby(["dataset", "row_index"], sort=False)

    for (dataset, row_index), group in grouped:
        heft_rows = group[group["method"] == "HEFT"]
        ppo_rows = group[group["method"] == "PPO_GAT"]

        if heft_rows.empty or ppo_rows.empty:
            continue

        heft = heft_rows.iloc[0]
        ppo = ppo_rows.iloc[0]

        time_improvement = percent_improvement(
            reference_value=heft["makespan"],
            candidate_value=ppo["makespan"],
        )

        energy_improvement = percent_improvement(
            reference_value=heft["total_energy"],
            candidate_value=ppo["total_energy"],
        )

        ratio = tradeoff_ratio(
            time_improvement_percent=time_improvement,
            energy_improvement_percent=energy_improvement,
        )

        rows.append(
            {
                "dataset": dataset,
                "row_index": int(row_index),
                "heft_makespan": float(heft["makespan"]),
                "ppo_gat_makespan": float(ppo["makespan"]),
                "time_improvement_percent": float(time_improvement),
                "heft_total_energy": float(heft["total_energy"]),
                "ppo_gat_total_energy": float(ppo["total_energy"]),
                "energy_improvement_percent": float(energy_improvement),
                "time_energy_tradeoff_ratio": float(ratio),
            }
        )

    return pd.DataFrame(rows)


def build_improvement_summary(improvement_report: pd.DataFrame) -> pd.DataFrame:
    """
    Build an average summary of the supervisor-requested metrics.
    """
    columns = [
        "time_improvement_percent_mean",
        "energy_improvement_percent_mean",
        "time_energy_tradeoff_ratio_mean",
    ]

    if improvement_report.empty:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(
        [
            {
                "time_improvement_percent_mean": float(
                    improvement_report["time_improvement_percent"].mean()
                ),
                "energy_improvement_percent_mean": float(
                    improvement_report["energy_improvement_percent"].mean()
                ),
                "time_energy_tradeoff_ratio_mean": float(
                    improvement_report["time_energy_tradeoff_ratio"].mean()
                ),
            }
        ]
    )


def save_plots(summary: pd.DataFrame) -> None:
    """
    Save makespan and energy comparison plots.
    """
    methods = summary["method"].tolist()

    fig1, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(methods, summary["makespan_mean"].tolist())
    ax1.set_ylabel("Mean Makespan")
    ax1.set_title("PPO+GAT vs HEFT: Makespan")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    fig1.tight_layout()
    fig1.savefig(EVAL_DIR / "makespan_comparison.png", dpi=180)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.bar(methods, summary["total_energy_mean"].tolist())
    ax2.set_ylabel("Mean Total Energy")
    ax2.set_title("PPO+GAT vs HEFT: Energy")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    fig2.tight_layout()
    fig2.savefig(EVAL_DIR / "energy_comparison.png", dpi=180)
    plt.close(fig2)


def main() -> None:
    print()
    print("============================================================")
    print(" Single-dataset evaluation")
    print("============================================================")
    print(f"Selected dataset key  = {SELECTED_DATASET_KEY}")
    print(f"Selected dataset file = {SELECTED_DATASET_PATH.name}")
    print(f"Selected row index    = {SELECTED_ROW_INDEX}")
    print(f"Forced max_tasks      = {FORCED_MAX_TASKS}")
    print(f"Forced max_edges      = {FORCED_MAX_EDGES}")
    print("============================================================")
    print()

    detail = evaluate()

    detail_path = EVAL_DIR / "comparison_detail.csv"
    detail.to_csv(detail_path, index=False)

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
    summary.to_csv(summary_path, index=False)

    improvement_report = build_improvement_report(detail)
    improvement_report_path = EVAL_DIR / "comparison_improvements.csv"
    improvement_report.to_csv(improvement_report_path, index=False)

    improvement_summary = build_improvement_summary(improvement_report)
    improvement_summary_path = EVAL_DIR / "comparison_improvements_summary.csv"
    improvement_summary.to_csv(improvement_summary_path, index=False)

    save_plots(summary)

    print()
    print("Detailed comparison:")
    print(detail.to_string(index=False))

    print()
    print("Summary:")
    print(summary.to_string(index=False))

    print()
    print("Improvement report: PPO_GAT compared with HEFT")
    print(improvement_report.to_string(index=False))

    print()
    print("Improvement summary:")
    print(improvement_summary.to_string(index=False))

    print()
    print("Saved files:")
    print(f"- {detail_path}")
    print(f"- {summary_path}")
    print(f"- {improvement_report_path}")
    print(f"- {improvement_summary_path}")
    print(f"- {EVAL_DIR / 'makespan_comparison.png'}")
    print(f"- {EVAL_DIR / 'energy_comparison.png'}")


if __name__ == "__main__":
    main()