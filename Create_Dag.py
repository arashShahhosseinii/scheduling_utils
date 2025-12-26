import pandas as pd
import ast
import re
import networkx as nx
from typing import List, Tuple


class CreateDAG:
    def __init__(self, csv_path: str, row_index: int = 0) -> None:
        # 1) Read CSV and select one row
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

        # 5) Parse node characteristics (CSV contains np.float64(...) so we must clean it)
        chars_str = str(row["characteristics"])
        chars_str = re.sub(r"np\.float64\(([^)]+)\)", r"\1", chars_str)
        chars_str = re.sub(r"np\.int64\(([^)]+)\)", r"\1", chars_str)

        chars_list = ast.literal_eval(chars_str)  # list of length num_subtasks

        periods = [0.0] * self.num_subtasks
        deadlines = [0.0] * self.num_subtasks
        m_i_list = [0.0] * self.num_subtasks

        # We will store REAL meanings:
        # A7  = slower core  = proc2 in your dataset
        # A12 = faster core  = proc1 in your dataset
        a7_times = [0.0] * self.num_subtasks
        a12_times = [0.0] * self.num_subtasks

        for i, node_info in enumerate(chars_list):
            # Scalar attributes
            periods[i] = float(node_info["period"])
            deadlines[i] = float(node_info["implicit_deadline"])
            m_i_list[i] = float(node_info["m_i"])

            # Your CSV uses v_f_levels (not Execution_Time)
            # pick the first VF level
            vf = node_info.get("v_f_levels", None)
            if vf is None:
                raise KeyError("CSV characteristics missing 'v_f_levels' key. Check dataset format.")
            exec_info = vf[0]

            # IMPORTANT FIX (based on your CSV):
            # proc1 is faster (higher frequency_proc1), so proc1 corresponds to A12
            # proc2 corresponds to A7
            a12 = float(exec_info["utilization_proc1_vf"])  # A12 (faster)
            a7 = float(exec_info["utilization_proc2_vf"])   # A7  (slower)

            a7_times[i] = a7
            a12_times[i] = a12

        # 6) Save as object attributes
        self.periods = periods
        self.deadlines = deadlines
        self.m_i_list = m_i_list
        self.a7_times = a7_times
        self.a12_times = a12_times
