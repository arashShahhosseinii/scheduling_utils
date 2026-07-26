from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from Dag_Env import DagSchedulingEnv
from gat_sb3_policy import (
    MaskedGATActorCriticPolicy,
)
from scheduling_utils import (
    run_policy_episode,
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

MODEL_PATH = (
    ARTIFACT_DIR
    / "models"
    / "ppo_gat_scheduler.zip"
)

EVAL_DIR = (
    ARTIFACT_DIR / "evaluation"
)

EVAL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SEED = 42
PROCESSOR_MAP = [0, 1]

REWARD_WEIGHTS = (
    0.008,
    45.0,
)

QOS_FACTOR = 1.25
REWARD_PROPOSAL = "A"

FORCED_MAX_TASKS = 108
FORCED_MAX_EDGES = 1152

EVALUATION_ALPHA = 0.01
EVALUATION_RUNS = 20

EPS = 1e-12


def build_env(
    csv_path: Path,
    row_index: int = 0,
) -> DagSchedulingEnv:
    return DagSchedulingEnv(
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


def metrics_to_row(
    *,
    row_index: int,
    method: str,
    evaluation_run: int,
    metrics: Dict[str, float],
    total_reward: float,
) -> Dict[str, object]:
    return {
        "dataset": (
            SELECTED_DATASET_PATH.name
        ),
        "row_index": int(row_index),
        "evaluation_run": int(
            evaluation_run
        ),
        "method": method,
        "makespan": float(
            metrics["makespan"]
        ),
        "total_energy": float(
            metrics["total_energy"]
        ),
        "total_energy_raw": float(
            metrics["total_energy_raw"]
        ),
        "total_tardiness": float(
            metrics["total_tardiness"]
        ),
        "average_qos": float(
            metrics["average_qos"]
        ),
        "minimum_qos": float(
            metrics["minimum_qos"]
        ),
        "maximum_qos": float(
            metrics["maximum_qos"]
        ),
        "qos_std": float(
            metrics["qos_std"]
        ),
        "on_time_task_ratio": float(
            metrics["on_time_task_ratio"]
        ),
        "zero_qos_task_ratio": float(
            metrics["zero_qos_task_ratio"]
        ),
        "total_reward": float(
            total_reward
        ),
    }


def evaluate() -> pd.DataFrame:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "First train the model with: "
            "python train_ppo_gat.py"
        )

    dummy_env = build_env(
        SELECTED_DATASET_PATH,
        row_index=SELECTED_ROW_INDEX,
    )

    print()
    print(
        "Loading trained PPO+GAT model..."
    )
    print(
        f"Model path              = "
        f"{MODEL_PATH}"
    )
    print(
        f"Evaluation dataset      = "
        f"{SELECTED_DATASET_PATH.name}"
    )
    print(
        f"Forced max_tasks        = "
        f"{FORCED_MAX_TASKS}"
    )
    print(
        f"Forced max_edges        = "
        f"{FORCED_MAX_EDGES}"
    )
    print(
        f"QoS factor              = "
        f"{QOS_FACTOR}"
    )
    print(
        f"Evaluation alpha        = "
        f"{EVALUATION_ALPHA}"
    )
    print(
        f"PPO evaluation runs     = "
        f"{EVALUATION_RUNS}"
    )
    print()

    model = PPO.load(
        str(MODEL_PATH),
        env=dummy_env,
        custom_objects={
            "policy_class": (
                MaskedGATActorCriticPolicy
            ),
        },
    )

    model.policy.set_exploration_alpha(
        EVALUATION_ALPHA
    )

    model.policy.set_training_mode(
        False
    )

    rows: List[Dict[str, object]] = []

    rng = np.random.default_rng(
        SEED
    )

    dataset_frame = pd.read_csv(
        SELECTED_DATASET_PATH
    )

    try:
        for row_index in range(
            len(dataset_frame)
        ):
            heft_env = build_env(
                SELECTED_DATASET_PATH,
                row_index=row_index,
            )

            try:
                (
                    _,
                    _,
                    _,
                    heft_metrics,
                    heft_reward,
                    _,
                ) = run_policy_episode(
                    env=heft_env,
                    policy_name="HEFT",
                    rng=rng,
                )

                rows.append(
                    metrics_to_row(
                        row_index=row_index,
                        method="HEFT",
                        evaluation_run=0,
                        metrics=heft_metrics,
                        total_reward=(
                            heft_reward
                        ),
                    )
                )
            finally:
                heft_env.close()

            for evaluation_run in range(
                1,
                EVALUATION_RUNS + 1,
            ):
                ppo_env = build_env(
                    SELECTED_DATASET_PATH,
                    row_index=row_index,
                )

                try:
                    (
                        _,
                        _,
                        _,
                        ppo_metrics,
                        ppo_reward,
                        _,
                    ) = run_policy_episode(
                        env=ppo_env,
                        policy_name="PPO_GAT",
                        rng=rng,
                        model=model,
                        ppo_deterministic=False,
                        ppo_exploration_alpha=(
                            EVALUATION_ALPHA
                        ),
                    )

                    rows.append(
                        metrics_to_row(
                            row_index=(
                                row_index
                            ),
                            method="PPO_GAT",
                            evaluation_run=(
                                evaluation_run
                            ),
                            metrics=(
                                ppo_metrics
                            ),
                            total_reward=(
                                ppo_reward
                            ),
                        )
                    )
                finally:
                    ppo_env.close()

    finally:
        dummy_env.close()

    return pd.DataFrame(rows)


def percent_improvement(
    reference_value: float,
    candidate_value: float,
) -> float:
    reference_value = float(
        reference_value
    )

    candidate_value = float(
        candidate_value
    )

    if abs(reference_value) < EPS:
        return float("nan")

    return (
        (
            reference_value
            - candidate_value
        )
        / reference_value
        * 100.0
    )


def tradeoff_ratio(
    time_improvement_percent: float,
    energy_improvement_percent: float,
) -> float:
    time_improvement_percent = float(
        time_improvement_percent
    )

    energy_improvement_percent = float(
        energy_improvement_percent
    )

    if (
        np.isnan(
            time_improvement_percent
        )
        or np.isnan(
            energy_improvement_percent
        )
    ):
        return float("nan")

    denominator = abs(
        energy_improvement_percent
    )

    if denominator < EPS:
        if (
            abs(
                time_improvement_percent
            )
            < EPS
        ):
            return 0.0

        return (
            float("inf")
            if time_improvement_percent > 0
            else float("-inf")
        )

    return (
        time_improvement_percent
        / denominator
    )


def build_improvement_report(
    detail: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []

    grouped = detail.groupby(
        [
            "dataset",
            "row_index",
        ],
        sort=False,
    )

    for (
        dataset,
        row_index,
    ), group in grouped:
        heft_rows = group[
            group["method"] == "HEFT"
        ]

        ppo_rows = group[
            group["method"] == "PPO_GAT"
        ]

        if (
            heft_rows.empty
            or ppo_rows.empty
        ):
            continue

        heft = heft_rows.iloc[0]

        for _, ppo in ppo_rows.iterrows():
            time_improvement = (
                percent_improvement(
                    reference_value=(
                        heft["makespan"]
                    ),
                    candidate_value=(
                        ppo["makespan"]
                    ),
                )
            )

            energy_improvement = (
                percent_improvement(
                    reference_value=(
                        heft["total_energy"]
                    ),
                    candidate_value=(
                        ppo["total_energy"]
                    ),
                )
            )

            ratio = tradeoff_ratio(
                time_improvement_percent=(
                    time_improvement
                ),
                energy_improvement_percent=(
                    energy_improvement
                ),
            )

            qos_difference = (
                float(
                    ppo["average_qos"]
                )
                - float(
                    heft["average_qos"]
                )
            )

            rows.append(
                {
                    "dataset": dataset,
                    "row_index": int(
                        row_index
                    ),
                    "evaluation_run": int(
                        ppo[
                            "evaluation_run"
                        ]
                    ),
                    "heft_makespan": float(
                        heft["makespan"]
                    ),
                    "ppo_gat_makespan": float(
                        ppo["makespan"]
                    ),
                    "time_improvement_percent": float(
                        time_improvement
                    ),
                    "heft_total_energy": float(
                        heft["total_energy"]
                    ),
                    "ppo_gat_total_energy": float(
                        ppo["total_energy"]
                    ),
                    "energy_improvement_percent": float(
                        energy_improvement
                    ),
                    "time_energy_tradeoff_ratio": float(
                        ratio
                    ),
                    "heft_average_qos": float(
                        heft["average_qos"]
                    ),
                    "ppo_gat_average_qos": float(
                        ppo["average_qos"]
                    ),
                    "qos_difference": float(
                        qos_difference
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_improvement_summary(
    improvement_report: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "time_improvement_percent_mean",
        "time_improvement_percent_std",
        "energy_improvement_percent_mean",
        "energy_improvement_percent_std",
        "time_energy_tradeoff_ratio_mean",
        "qos_difference_mean",
    ]

    if improvement_report.empty:
        return pd.DataFrame(
            columns=columns
        )

    return pd.DataFrame(
        [
            {
                "time_improvement_percent_mean": float(
                    improvement_report[
                        "time_improvement_percent"
                    ].mean()
                ),
                "time_improvement_percent_std": float(
                    improvement_report[
                        "time_improvement_percent"
                    ].std(
                        ddof=0
                    )
                ),
                "energy_improvement_percent_mean": float(
                    improvement_report[
                        "energy_improvement_percent"
                    ].mean()
                ),
                "energy_improvement_percent_std": float(
                    improvement_report[
                        "energy_improvement_percent"
                    ].std(
                        ddof=0
                    )
                ),
                "time_energy_tradeoff_ratio_mean": float(
                    improvement_report[
                        "time_energy_tradeoff_ratio"
                    ].mean()
                ),
                "qos_difference_mean": float(
                    improvement_report[
                        "qos_difference"
                    ].mean()
                ),
            }
        ]
    )


def build_qos_report(
    detail: pd.DataFrame,
) -> pd.DataFrame:
    return detail[
        [
            "dataset",
            "row_index",
            "evaluation_run",
            "method",
            "average_qos",
            "minimum_qos",
            "maximum_qos",
            "qos_std",
            "on_time_task_ratio",
            "zero_qos_task_ratio",
        ]
    ].copy()


def build_summary(
    detail: pd.DataFrame,
) -> pd.DataFrame:
    return (
        detail.groupby(
            "method",
            as_index=False,
        )
        .agg(
            evaluation_count=(
                "makespan",
                "count",
            ),
            makespan_mean=(
                "makespan",
                "mean",
            ),
            makespan_std=(
                "makespan",
                "std",
            ),
            total_energy_mean=(
                "total_energy",
                "mean",
            ),
            total_energy_std=(
                "total_energy",
                "std",
            ),
            total_tardiness_mean=(
                "total_tardiness",
                "mean",
            ),
            total_reward_mean=(
                "total_reward",
                "mean",
            ),
            average_qos_mean=(
                "average_qos",
                "mean",
            ),
            average_qos_std=(
                "average_qos",
                "std",
            ),
            minimum_qos_mean=(
                "minimum_qos",
                "mean",
            ),
            maximum_qos_mean=(
                "maximum_qos",
                "mean",
            ),
            on_time_task_ratio_mean=(
                "on_time_task_ratio",
                "mean",
            ),
            zero_qos_task_ratio_mean=(
                "zero_qos_task_ratio",
                "mean",
            ),
        )
        .fillna(0.0)
    )


def save_plots(
    summary: pd.DataFrame,
) -> None:
    methods = summary[
        "method"
    ].tolist()

    makespan_figure, makespan_axis = (
        plt.subplots(
            figsize=(9, 5)
        )
    )

    makespan_axis.bar(
        methods,
        summary[
            "makespan_mean"
        ].tolist(),
    )

    makespan_axis.set_ylabel(
        "Mean Makespan"
    )

    makespan_axis.set_title(
        "PPO+GAT vs HEFT: Makespan"
    )

    makespan_axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.5,
    )

    makespan_figure.tight_layout()

    makespan_figure.savefig(
        EVAL_DIR
        / "makespan_comparison.png",
        dpi=180,
    )

    plt.close(
        makespan_figure
    )

    energy_figure, energy_axis = (
        plt.subplots(
            figsize=(9, 5)
        )
    )

    energy_axis.bar(
        methods,
        summary[
            "total_energy_mean"
        ].tolist(),
    )

    energy_axis.set_ylabel(
        "Mean Normalized Energy"
    )

    energy_axis.set_title(
        "PPO+GAT vs HEFT: Energy"
    )

    energy_axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.5,
    )

    energy_figure.tight_layout()

    energy_figure.savefig(
        EVAL_DIR
        / "energy_comparison.png",
        dpi=180,
    )

    plt.close(
        energy_figure
    )

    qos_figure, qos_axis = plt.subplots(
    figsize=(9, 5)
    )

    qos_values = summary[
        "average_qos_mean"
    ].tolist()

    bars = qos_axis.bar(
        methods,
        qos_values,
    )

    qos_axis.set_ylabel(
        "Mean QoS"
    )

    qos_axis.set_title(
        "PPO+GAT vs HEFT: Average QoS"
    )

    qos_min = float(min(qos_values))
    qos_max = float(max(qos_values))

    if abs(qos_max - qos_min) < 1e-6:
        lower = max(0.0, qos_min - 0.01)
        upper = min(1.0, qos_max + 0.01)
    else:
        padding = max(0.002, 0.15 * (qos_max - qos_min))
        lower = max(0.0, qos_min - padding)
        upper = min(1.0, qos_max + padding)

    qos_axis.set_ylim(lower, upper)

    for bar, value in zip(bars, qos_values):
        qos_axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    qos_axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.5,
    )

    qos_figure.tight_layout()

    qos_figure.savefig(
        EVAL_DIR / "qos_comparison.png",
        dpi=180,
    )

    plt.close(qos_figure)


def main() -> None:
    print()
    print(
        "================================================"
    )
    print(
        " Single-dataset stochastic evaluation"
    )
    print(
        "================================================"
    )
    print(
        f"Selected dataset file = "
        f"{SELECTED_DATASET_PATH.name}"
    )
    print(
        f"Forced max_tasks      = "
        f"{FORCED_MAX_TASKS}"
    )
    print(
        f"Forced max_edges      = "
        f"{FORCED_MAX_EDGES}"
    )
    print(
        f"QoS factor            = "
        f"{QOS_FACTOR}"
    )
    print(
        f"Evaluation alpha      = "
        f"{EVALUATION_ALPHA}"
    )
    print(
        f"PPO evaluation runs   = "
        f"{EVALUATION_RUNS}"
    )
    print(
        "================================================"
    )
    print()

    detail = evaluate()
    summary = build_summary(
        detail
    )

    improvement_report = (
        build_improvement_report(
            detail
        )
    )

    improvement_summary = (
        build_improvement_summary(
            improvement_report
        )
    )

    qos_report = build_qos_report(
        detail
    )

    detail_path = (
        EVAL_DIR
        / "comparison_detail.csv"
    )

    summary_path = (
        EVAL_DIR
        / "comparison_summary.csv"
    )

    improvement_report_path = (
        EVAL_DIR
        / "comparison_improvements.csv"
    )

    improvement_summary_path = (
        EVAL_DIR
        / "comparison_improvements_summary.csv"
    )

    qos_report_path = (
        EVAL_DIR
        / "qos_report.csv"
    )

    detail.to_csv(
        detail_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    improvement_report.to_csv(
        improvement_report_path,
        index=False,
    )

    improvement_summary.to_csv(
        improvement_summary_path,
        index=False,
    )

    qos_report.to_csv(
        qos_report_path,
        index=False,
    )

    save_plots(
        summary
    )

    print(
        "Detailed comparison:"
    )
    print(
        detail.to_string(
            index=False
        )
    )

    print()
    print(
        "Average summary:"
    )
    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "PPO_GAT improvement report:"
    )
    print(
        improvement_report.to_string(
            index=False
        )
    )

    print()
    print(
        "Improvement summary:"
    )
    print(
        improvement_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Saved files:"
    )
    print(
        f"- {detail_path}"
    )
    print(
        f"- {summary_path}"
    )
    print(
        f"- {improvement_report_path}"
    )
    print(
        f"- {improvement_summary_path}"
    )
    print(
        f"- {qos_report_path}"
    )
    print(
        f"- "
        f"{EVAL_DIR / 'makespan_comparison.png'}"
    )
    print(
        f"- "
        f"{EVAL_DIR / 'energy_comparison.png'}"
    )
    print(
        f"- "
        f"{EVAL_DIR / 'qos_comparison.png'}"
    )


if __name__ == "__main__":
    main()