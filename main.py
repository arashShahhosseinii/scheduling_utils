import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path  
import sys

# Assuming the CreateDAG class is defined in Create_Dag.py
from Create_Dag import CreateDAG 
from scheduling_utils import (
    compute_schedule_metrics,
    schedule_random_A7_A12,
    schedule_dual_A7,
    schedule_dual_A12,
    schedule_deadline_based_HEFT
)

# --- 1. Initialization and Data Loading ---

# Construct the absolute path based on the script's location
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path(sys.argv[0]).resolve().parent

DATASET_PATH = SCRIPT_DIR / 'dag_dataset_a7_a12.csv'

TASK_ID = 0
SEED = 42
rng = np.random.RandomState(SEED)

try:
    # Create instance and load data
    dag_creator = CreateDAG(str(DATASET_PATH), row_index=TASK_ID) 
    
    # Get required DAG components
    G = dag_creator.graph 
    
    exec_times = np.stack(
        [
            np.array(dag_creator.a12_times, dtype=float),  # A7 (index 0, slower core) gets the slower original times
            np.array(dag_creator.a7_times, dtype=float),   # A12 (index 1, faster core) gets the faster original times
        ],
        axis=1,
    )
    deadlines = np.array(dag_creator.deadlines, dtype=float)

except FileNotFoundError:
    print(f"Error: Data file {DATASET_PATH} not found. Ensure it is located in the same directory as main.py.")
    exit()
except Exception as e:
    print(f"Error during DAG creation: {e}")
    exit()


# --- 2. Execute Scheduling Algorithms (Dual-Core Scenarios) ---

results = {
    'Makespan': {},
    'Total_Tardiness': {}
}

scheduling_scenarios = {
    "Dual A7 (Homogeneous)": schedule_dual_A7,
    "Dual A12 (Homogeneous)": schedule_dual_A12,
    "Random A7 + A12 (Heterogeneous)": schedule_random_A7_A12,
    "EDF/HEFT A7 + A12 (Heterogeneous)": schedule_deadline_based_HEFT
}

for name, scheduler_func in scheduling_scenarios.items():
    print(f"Running scheduler: {name}...")
    
    # Call the scheduling function
    if "Random" in name:
        assigned, start, finish = scheduler_func(G, exec_times, deadlines, rng)
    else:
        assigned, start, finish = scheduler_func(G, exec_times, deadlines)
        
    # Convert finish dictionary to array for metric calculation (based on task order)
    finish_array = np.array([finish[i] for i in sorted(finish.keys())])
    
    # Calculate metrics
    makespan, total_tardiness = compute_schedule_metrics(finish_array, deadlines)
    
    results['Makespan'][name] = makespan
    results['Total_Tardiness'][name] = total_tardiness
    
    print(f"   Makespan: {makespan:.2f}, Total Tardiness: {total_tardiness:.2f}")


# --- 3. Plotting Results ---

# Prepare data for plotting
names = list(results['Makespan'].keys())
makespans = [results['Makespan'][name] for name in names]
tardinesses = [results['Total_Tardiness'][name] for name in names]

x = np.arange(len(names))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 7))
rects1 = ax.bar(x - width/2, makespans, width, label='Makespan (Total Completion Time)', color='#2ca02c') 
rects2 = ax.bar(x + width/2, tardinesses, width, label='Total Tardiness (Total Delay)', color='#ff7f0e') 

# Set labels and titles
ax.set_ylabel('Metric Value')
ax.set_title(f'Comparison of Dual-Core Scheduling Policies for DAG Task {TASK_ID}')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=20, ha='right', fontsize=10)
ax.legend()
ax.grid(axis='y', linestyle='--')

# Function to display the value above each bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.show()