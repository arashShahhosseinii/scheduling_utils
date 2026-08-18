from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import json
import random

import networkx as nx
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================
# Extract results____.rar next to this script first.
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = SCRIPT_DIR / "results____"
WORKLOAD_GROUP = "60"
def _find_processor_folder(results_root: Path, workload_group: str, processor: str) -> Path:
    group_root = results_root / workload_group
    if not group_root.exists():
        raise FileNotFoundError(f"Workload group not found: {group_root}")
    candidates = [
        path for path in group_root.rglob(processor)
        if path.is_dir() and any(path.glob("*.csv"))
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one {processor} benchmark folder under {group_root}, "
            f"found: {candidates}"
        )
    return candidates[0]


def resolve_processor_folders() -> Tuple[Path, Path]:
    return (
        _find_processor_folder(RESULTS_ROOT, WORKLOAD_GROUP, "A7"),
        _find_processor_folder(RESULTS_ROOT, WORKLOAD_GROUP, "A12"),
    )

OUTPUT_CSV = SCRIPT_DIR / "dag_dataset_gang_a7_a12.csv"

NUM_DAGS = 1
NUM_SUBTASKS = 60
EDGE_PROB_RANGES: Sequence[Tuple[float, float]] = (
    (0.10, 0.30),
    (0.40, 0.60),
    (0.70, 0.90),
)
DEADLINE_FACTOR = 1.15
M_I_VALUES = (1, 2, 3, 4, 5, 6)
SEED = 42


# ============================================================
# VF levels supplied by the professor
# proc1 = A12, proc2 = A7
# ============================================================
VF_LEVELS_A12 = [
    {"level": 1, "frequency": 0.8, "voltage": 0.85},
    {"level": 2, "frequency": 1.0, "voltage": 0.90},
    {"level": 3, "frequency": 1.2, "voltage": 0.95},
    {"level": 4, "frequency": 1.4, "voltage": 1.05},
    {"level": 5, "frequency": 1.6, "voltage": 1.10},
    {"level": 6, "frequency": 1.8, "voltage": 1.15},
]

VF_LEVELS_A7 = [
    {"level": 1, "frequency": 0.5, "voltage": 0.85},
    {"level": 2, "frequency": 0.7, "voltage": 0.90},
    {"level": 3, "frequency": 0.9, "voltage": 0.95},
    {"level": 4, "frequency": 1.1, "voltage": 1.05},
    {"level": 5, "frequency": 1.3, "voltage": 1.10},
    {"level": 6, "frequency": 1.4, "voltage": 1.15},
]


def convert_np(obj):
    """Recursively convert NumPy values to ordinary Python values."""
    if isinstance(obj, dict):
        return {key: convert_np(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_np(value) for value in obj]
    if isinstance(obj, tuple):
        return [convert_np(value) for value in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def _find_column(df: pd.DataFrame, expected: str) -> str | None:
    expected_lower = expected.strip().lower()
    for column in df.columns:
        if str(column).strip().lower() == expected_lower:
            return str(column)
    return None


def _extract_last_numeric(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        raise ValueError("Could not find a numeric execution-time value.")
    return float(numeric.iloc[-1])


def extract_processor_data(folder_path: Path) -> Dict[str, dict]:
    """
    Read benchmark CSVs and return a dictionary keyed by lower-case filename.

    Matching A7/A12 by filename is important. The professor's original code
    used independent os.listdir() orders and later paired by position, which
    can mismatch workloads across processor types.
    """
    if not folder_path.exists():
        raise FileNotFoundError(f"Benchmark folder not found: {folder_path}")

    task_data: Dict[str, dict] = {}

    for file_path in sorted(folder_path.glob("*.csv"), key=lambda p: p.name.lower()):
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError(f"Empty benchmark CSV: {file_path}")

        execution_time_max_vf = _extract_last_numeric(df.iloc[:, 0])

        dynamic_col = _find_column(df, "Dynamic")
        static_col = _find_column(df, "Static")
        total_col = _find_column(df, "Total")

        avg_dynamic = 0.0
        avg_static = 0.0

        if dynamic_col is not None:
            dynamic_values = pd.to_numeric(df[dynamic_col], errors="coerce").dropna()
            if not dynamic_values.empty:
                avg_dynamic = float(dynamic_values.mean())

        if static_col is not None:
            static_values = pd.to_numeric(df[static_col], errors="coerce").dropna()
            if not static_values.empty:
                avg_static = float(static_values.mean())

        if dynamic_col is not None and static_col is not None:
            avg_total_power_max_vf = avg_dynamic + avg_static
        elif total_col is not None:
            total_values = pd.to_numeric(df[total_col], errors="coerce").dropna()
            if total_values.empty:
                raise ValueError(f"No numeric Total power values in {file_path}")
            avg_total_power_max_vf = float(total_values.mean())
        else:
            raise ValueError(
                f"{file_path} must contain Dynamic+Static or Total power columns."
            )

        if execution_time_max_vf <= 0.0 or avg_total_power_max_vf <= 0.0:
            raise ValueError(f"Non-positive time/power in benchmark: {file_path}")

        task_data[file_path.name.lower()] = {
            "file_name": file_path.name,
            "execution_time_max_vf": execution_time_max_vf,
            "avg_total_power_max_vf": avg_total_power_max_vf,
            "avg_dynamic_power_max_vf": avg_dynamic,
            "avg_static_power_max_vf": avg_static,
        }

    if not task_data:
        raise ValueError(f"No CSV files found in {folder_path}")

    return task_data


def pair_processor_data(
    a7_data: Dict[str, dict],
    a12_data: Dict[str, dict],
) -> List[dict]:
    """Pair the exact same benchmark filename across A7 and A12."""
    common_names = sorted(set(a7_data).intersection(a12_data))
    if not common_names:
        raise ValueError("No matching A7/A12 benchmark filenames were found.")

    missing_on_a12 = sorted(set(a7_data) - set(a12_data))
    missing_on_a7 = sorted(set(a12_data) - set(a7_data))

    if missing_on_a12:
        print(f"WARNING: {len(missing_on_a12)} A7 files have no A12 match.")
    if missing_on_a7:
        print(f"WARNING: {len(missing_on_a7)} A12 files have no A7 match.")

    return [
        {
            "benchmark_name": name,
            "A7": a7_data[name],
            "A12": a12_data[name],
        }
        for name in common_names
    ]


def update_task_vf_attributes(
    vf_level: dict,
    max_voltage: float,
    task_data: dict,
) -> Tuple[float, float]:
    """
    Preserve the professor-supplied voltage-scaling assumption:

        rho = V / Vmax
        T(V) = T_max / rho
        P(V) = rho^3 * P_max
    """
    rho = float(vf_level["voltage"]) / float(max_voltage)
    updated_execution_time = (
        float(task_data["execution_time_max_vf"]) / max(rho, 1e-12)
    )
    updated_power = (rho ** 3) * float(task_data["avg_total_power_max_vf"])
    return float(updated_execution_time), float(updated_power)


def build_vf_levels(a7_task: dict, a12_task: dict) -> List[dict]:
    """Build a parser-compatible v_f_levels list and retain time + power."""
    levels: List[dict] = []
    max_voltage_a12 = max(float(level["voltage"]) for level in VF_LEVELS_A12)
    max_voltage_a7 = max(float(level["voltage"]) for level in VF_LEVELS_A7)

    for level_a12, level_a7 in zip(VF_LEVELS_A12, VF_LEVELS_A7):
        time_a12, power_a12 = update_task_vf_attributes(
            level_a12, max_voltage_a12, a12_task
        )
        time_a7, power_a7 = update_task_vf_attributes(
            level_a7, max_voltage_a7, a7_task
        )

        levels.append(
            {
                "level": int(level_a12["level"]),
                # proc1 = A12
                "voltage_proc1": float(level_a12["voltage"]),
                "frequency_proc1": float(level_a12["frequency"]),
                "utilization_proc1_vf": time_a12,
                "avg_total_power_proc1": power_a12,
                # proc2 = A7
                "voltage_proc2": float(level_a7["voltage"]),
                "frequency_proc2": float(level_a7["frequency"]),
                "utilization_proc2_vf": time_a7,
                "avg_total_power_proc2": power_a7,
            }
        )

    return levels


def generate_node_characteristics(
    num_subtasks: int,
    paired_data: Sequence[dict],
    rng: random.Random,
) -> List[dict]:
    """Generate task properties without unbounded re-sampling loops."""
    if num_subtasks > len(paired_data):
        raise ValueError(
            f"num_subtasks={num_subtasks} exceeds the {len(paired_data)} "
            "matched A7/A12 benchmarks in the selected workload group."
        )

    selected = rng.sample(list(paired_data), k=num_subtasks)
    characteristics: List[dict] = []

    for pair in selected:
        a7_task = pair["A7"]
        a12_task = pair["A12"]
        vf_levels = build_vf_levels(a7_task, a12_task)

        fastest_local_time = min(
            float(a12_task["execution_time_max_vf"]),
            float(a7_task["execution_time_max_vf"]),
        )

        characteristics.append(
            {
                "benchmark_name": pair["benchmark_name"],
                "period": 1,
                "implicit_deadline": fastest_local_time * DEADLINE_FACTOR,
                "m_i": int(rng.choice(M_I_VALUES)),
                "v_f_levels": vf_levels,
            }
        )

    return characteristics


def generate_dag(
    num_subtasks: int,
    edge_prob: float,
    np_rng: np.random.Generator,
) -> nx.DiGraph:
    """Generate an acyclic directed graph by keeping only u < v edges."""
    seed = int(np_rng.integers(0, 2**31 - 1))
    graph = nx.gnp_random_graph(
        num_subtasks,
        edge_prob,
        seed=seed,
        directed=True,
    )
    dag = nx.DiGraph()
    dag.add_nodes_from(range(num_subtasks))
    dag.add_edges_from((u, v) for u, v in graph.edges() if u < v)
    return dag


def generate_dataset(
    num_dags: int,
    num_subtasks: int,
    edge_prob_ranges: Sequence[Tuple[float, float]],
    paired_data: Sequence[dict],
    seed: int,
) -> List[dict]:
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    dataset: List[dict] = []

    for _ in range(num_dags):
        low, high = edge_prob_ranges[int(np_rng.integers(0, len(edge_prob_ranges)))]
        edge_prob = float(np_rng.uniform(low, high))
        dag = generate_dag(num_subtasks, edge_prob, np_rng)
        characteristics = generate_node_characteristics(
            num_subtasks=num_subtasks,
            paired_data=paired_data,
            rng=py_rng,
        )

        dataset.append(
            {
                "num_subtasks": int(num_subtasks),
                "edge_prob": edge_prob,
                "characteristics": characteristics,
                "edges": [(int(u), int(v)) for u, v in dag.edges()],
            }
        )

    return dataset


def main() -> List[dict]:
    a7_folder, a12_folder = resolve_processor_folders()
    print(f"A7 folder : {a7_folder}")
    print(f"A12 folder: {a12_folder}")

    a7_data = extract_processor_data(a7_folder)
    a12_data = extract_processor_data(a12_folder)
    paired_data = pair_processor_data(a7_data, a12_data)

    print(f"Matched benchmark pairs: {len(paired_data)}")
    print(f"Requested DAG tasks     : {NUM_SUBTASKS}")

    dataset = generate_dataset(
        num_dags=NUM_DAGS,
        num_subtasks=NUM_SUBTASKS,
        edge_prob_ranges=EDGE_PROB_RANGES,
        paired_data=paired_data,
        seed=SEED,
    )

    serializable_rows = []
    for data in dataset:
        serializable_rows.append(
            {
                "num_subtasks": data["num_subtasks"],
                "edge_prob": data["edge_prob"],
                "characteristics": json.dumps(convert_np(data["characteristics"])),
                "edges": json.dumps(convert_np(data["edges"])),
            }
        )

    dataset_df = pd.DataFrame(serializable_rows)
    dataset_df.to_csv(OUTPUT_CSV, index=False)

    for row_index, data in enumerate(dataset):
        m_i_values = [int(task["m_i"]) for task in data["characteristics"]]
        print(f"DAG {row_index} m_i values: {m_i_values}")
        print(
            f"DAG {row_index}: tasks={data['num_subtasks']}, "
            f"edges={len(data['edges'])}, max_m_i={max(m_i_values)}"
        )

    print(f"Dataset saved to: {OUTPUT_CSV}")
    return dataset


if __name__ == "__main__":
    main()
