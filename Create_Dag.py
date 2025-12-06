import pandas as pd
import ast
import networkx as nx
from typing import List, Tuple


class CreateDAG:


    def __init__(self, csv_path: str, row_index: int = 0) -> None:
        """
        :param csv_path: Path to the CSV file.
        :param row_index: Index of the row to use if the CSV has multiple rows.
        """
        # 1) Read CSV and select one row
        df = pd.read_csv(csv_path)
        row = df.iloc[row_index]

        # 2) Number of subtasks (nodes)
        self.num_subtasks: int = int(row["num_subtasks"])

        # 3) Parse edges from string to list of (u, v) tuples
        edges_str = str(row["edges"])
        edges_list = ast.literal_eval(edges_str)  # e.g. [[0, 1], [0, 2], ...]
        self.edges: List[Tuple[int, int]] = [(int(u), int(v)) for u, v in edges_list]

        # 4) Build directed graph with networkx
        G = nx.DiGraph()
        G.add_nodes_from(range(self.num_subtasks))
        G.add_edges_from(self.edges)
        self.graph: nx.DiGraph = G

        # 5) Parse node characteristics
        chars_str = str(row["characteristics"])
        chars_list = ast.literal_eval(chars_str)  # list of length num_subtasks

        periods = [0.0] * self.num_subtasks
        deadlines = [0.0] * self.num_subtasks
        m_i_list = [0.0] * self.num_subtasks
        a7_times = [0.0] * self.num_subtasks    # A7  -> utilization_proc1_vf
        a12_times = [0.0] * self.num_subtasks   # A12 -> utilization_proc2_vf

        for i, node_info in enumerate(chars_list):
            # Scalar attributes
            period = float(node_info["period"])
            deadline = float(node_info["implicit_deadline"])
            m_i = float(node_info["m_i"])

            # Execution-time information (one dict in the list)
            exec_info = node_info["Execution_Time"][0]

            # IMPORTANT: proc1 = A7, proc2 = A12
            a7 = float(exec_info["utilization_proc1_vf"])   # A7
            a12 = float(exec_info["utilization_proc2_vf"])  # A12

            periods[i] = period
            deadlines[i] = deadline
            m_i_list[i] = m_i
            a7_times[i] = a7
            a12_times[i] = a12

        # 6) Save as object attributes
        self.periods = periods
        self.deadlines = deadlines
        self.m_i_list = m_i_list
        self.a7_times = a7_times      # length = num_subtasks
        self.a12_times = a12_times    # length = num_subtasks
