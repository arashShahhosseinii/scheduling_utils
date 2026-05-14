import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import networkx as nx
import pandas as pd


def _clean_numpy_wrappers(value: str) -> str:
    """Remove strings such as np.float64(1.2) before ast.literal_eval."""
    value = re.sub(r"np\.float64\(([^)]+)\)", r"\1", value)
    value = re.sub(r"np\.float32\(([^)]+)\)", r"\1", value)
    value = re.sub(r"np\.int64\(([^)]+)\)", r"\1", value)
    value = re.sub(r"np\.int32\(([^)]+)\)", r"\1", value)
    return value


@dataclass(frozen=True)
class DAGRecord:
    csv_path: str
    row_index: int
    num_subtasks: int
    edges: List[Tuple[int, int]]
    characteristics: List[Dict[str, Any]]


class CreateDAG:
    """
    Loads one DAG instance from your CSV format and exposes the fields needed by
    the Gymnasium scheduler.

    Processor convention used everywhere in the project:
      0 = A7
      1 = A12

    For the current phase we use the max-frequency entry for each processor type.
    The full v_f_levels list is kept in self.v_f_levels for a later DVFS phase.
    """

    def __init__(self, csv_path: Union[str, Path], row_index: int = 0) -> None:
        record = self.read_record(csv_path, row_index)

        self.csv_path = record.csv_path
        self.row_index = record.row_index
        self.num_subtasks = record.num_subtasks
        self.edges = record.edges
        self.characteristics = record.characteristics

        graph = nx.DiGraph()
        graph.add_nodes_from(range(self.num_subtasks))
        graph.add_edges_from(self.edges)

        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError(
                f"The DAG in {csv_path}, row {row_index}, contains a cycle."
            )

        self.graph = graph

        self.periods: List[float] = []
        self.deadlines: List[float] = []
        self.m_i_list: List[float] = []
        self.v_f_levels: List[List[Dict[str, Any]]] = []

        self.a7_times: List[float] = []
        self.a12_times: List[float] = []

        self.a7_power: List[float] = []
        self.a12_power: List[float] = []

        self.a7_energy: List[float] = []
        self.a12_energy: List[float] = []

        self.a7_freq: List[float] = []
        self.a12_freq: List[float] = []

        self._parse_characteristics(record.characteristics)

    @staticmethod
    def read_record(csv_path: Union[str, Path], row_index: int = 0) -> DAGRecord:
        csv_path = str(csv_path)
        df = pd.read_csv(csv_path)

        if row_index < 0 or row_index >= len(df):
            raise IndexError(
                f"row_index={row_index} is outside CSV length {len(df)} for {csv_path}."
            )

        row = df.iloc[row_index]

        num_subtasks = int(row["num_subtasks"])

        edges = [
            (int(u), int(v))
            for u, v in ast.literal_eval(str(row["edges"]))
        ]

        chars_str = _clean_numpy_wrappers(str(row["characteristics"]))
        characteristics = ast.literal_eval(chars_str)

        if len(characteristics) != num_subtasks:
            raise ValueError(
                f"Expected {num_subtasks} characteristic entries, "
                f"got {len(characteristics)} in {csv_path}."
            )

        return DAGRecord(
            csv_path=csv_path,
            row_index=row_index,
            num_subtasks=num_subtasks,
            edges=edges,
            characteristics=characteristics,
        )

    @staticmethod
    def list_records(
        csv_paths: Union[str, Path, Sequence[Union[str, Path]]]
    ) -> List[Tuple[str, int]]:
        """Return all (csv_path, row_index) pairs. Useful for multi-DAG PPO training."""

        if isinstance(csv_paths, (str, Path)):
            csv_paths = [csv_paths]

        records: List[Tuple[str, int]] = []

        for path in csv_paths:
            path = str(path)
            df = pd.read_csv(path)

            for row_idx in range(len(df)):
                records.append((path, row_idx))

        if not records:
            raise ValueError("No DAG records found.")

        return records

    def _parse_characteristics(self, characteristics: List[Dict[str, Any]]) -> None:
        for node_info in characteristics:
            self.periods.append(float(node_info.get("period", 1.0)))
            self.deadlines.append(float(node_info["implicit_deadline"]))
            self.m_i_list.append(float(node_info.get("m_i", 1.0)))

            vf_levels = node_info.get("v_f_levels")

            if not vf_levels:
                raise KeyError("Missing non-empty 'v_f_levels' in characteristics.")

            self.v_f_levels.append(vf_levels)

            vf_a12 = max(
                vf_levels,
                key=lambda d: float(d["frequency_proc1"]),
            )

            vf_a7 = max(
                vf_levels,
                key=lambda d: float(d["frequency_proc2"]),
            )

            t_a12 = float(vf_a12["utilization_proc1_vf"])
            t_a7 = float(vf_a7["utilization_proc2_vf"])

            p_a12 = float(vf_a12["avg_total_power_proc1"])
            p_a7 = float(vf_a7["avg_total_power_proc2"])

            self.a12_times.append(t_a12)
            self.a7_times.append(t_a7)

            self.a12_power.append(p_a12)
            self.a7_power.append(p_a7)

            self.a12_energy.append(p_a12 * t_a12)
            self.a7_energy.append(p_a7 * t_a7)

            self.a12_freq.append(float(vf_a12["frequency_proc1"]))
            self.a7_freq.append(float(vf_a7["frequency_proc2"]))