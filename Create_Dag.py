import pandas as pd
import ast
import re
import networkx as nx
from typing import List, Tuple


class CreateDAG:
    def __init__(self, csv_path: str, row_index: int = 0) -> None:
        # 1) Read CSV and select one row (one DAG)
        data_frame = pd.read_csv(csv_path)
        row = data_frame.iloc[row_index]

        # 2) Number of subtasks (nodes)
        self.num_subtasks: int = int(row["num_subtasks"])

        # 3) Parse edges
        edges_str = str(row["edges"])
        edges_list = ast.literal_eval(edges_str)
        self.edges: List[Tuple[int, int]] = [(int(u), int(v)) for u, v in edges_list]

        # 4) Build directed graph
        G = nx.DiGraph()
        G.add_nodes_from(range(self.num_subtasks))
        G.add_edges_from(self.edges)
        self.graph: nx.DiGraph = G

        # 5) Parse node characteristics (clean np.float64(...) wrappers)
        chars_str = str(row["characteristics"])
        chars_str = re.sub(r"np\.float64\(([^)]+)\)", r"\1", chars_str)
        chars_str = re.sub(r"np\.int64\(([^)]+)\)", r"\1", chars_str)

        chars_list = ast.literal_eval(chars_str)  # list length = num_subtasks

        periods = [0.0] * self.num_subtasks
        deadlines = [0.0] * self.num_subtasks
        m_i_list = [0.0] * self.num_subtasks

        # Times (col0=A7, col1=A12 will be built in main)
        a7_times = [0.0] * self.num_subtasks
        a12_times = [0.0] * self.num_subtasks

        # Power
        a7_power = [0.0] * self.num_subtasks
        a12_power = [0.0] * self.num_subtasks

        # Energy = Power * Time
        a7_energy = [0.0] * self.num_subtasks
        a12_energy = [0.0] * self.num_subtasks

        # Optional: store selected frequencies for sanity-checking
        a7_freq = [0.0] * self.num_subtasks
        a12_freq = [0.0] * self.num_subtasks

        for i, node_info in enumerate(chars_list):
            periods[i] = float(node_info["period"])
            deadlines[i] = float(node_info["implicit_deadline"])
            m_i_list[i] = float(node_info["m_i"])

            vf_levels = node_info.get("v_f_levels", None)
            if vf_levels is None or len(vf_levels) == 0:
                raise KeyError("Missing 'v_f_levels' in characteristics.")

            # Choose the VF level with the highest frequency (in your dataset it's the last entry)
            # To be robust, we compute argmax over (freq1 + freq2)
            best_idx = max(
                range(len(vf_levels)),
                key=lambda k: float(vf_levels[k]["frequency_proc1"]) + float(vf_levels[k]["frequency_proc2"]),
            )
            vf = vf_levels[best_idx]

            # IMPORTANT (based on your CSV):
            # proc1 is the faster core (higher frequency) => A12
            # proc2 is the slower core => A7
            t_a12 = float(vf["utilization_proc1_vf"])
            t_a7 = float(vf["utilization_proc2_vf"])

            p_a12 = float(vf["avg_total_power_proc1"])
            p_a7 = float(vf["avg_total_power_proc2"])

            a12_times[i] = t_a12
            a7_times[i] = t_a7

            a12_power[i] = p_a12
            a7_power[i] = p_a7

            a12_energy[i] = p_a12 * t_a12
            a7_energy[i] = p_a7 * t_a7

            a12_freq[i] = float(vf["frequency_proc1"])
            a7_freq[i] = float(vf["frequency_proc2"])

        # 6) Save attributes
        self.periods = periods
        self.deadlines = deadlines
        self.m_i_list = m_i_list

        self.a7_times = a7_times
        self.a12_times = a12_times

        self.a7_power = a7_power
        self.a12_power = a12_power

        self.a7_energy = a7_energy
        self.a12_energy = a12_energy

        self.a7_freq = a7_freq
        self.a12_freq = a12_freq
