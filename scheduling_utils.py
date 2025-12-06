import networkx as nx
import numpy as np
from typing import Tuple, Dict, List, Any


def compute_schedule_metrics(
    finish_times: np.ndarray, deadlines: np.ndarray
) -> Tuple[float, float]:
    """
    Calculate Makespan and Total Tardiness.
    """
    makespan = float(finish_times.max()) if finish_times.size > 0 else 0.0
    tardiness = np.maximum(0.0, finish_times - deadlines)
    total_tardiness = float(tardiness.sum())
    
    return makespan, total_tardiness


def _schedule_parallel_dag(
    G: nx.DiGraph, 
    exec_times: np.ndarray, 
    deadlines: np.ndarray,
    processor_map: List[int], # [Proc1_Type, Proc2_Type, ...] (0=A7, 1=A12)
    policy: str, # 'EDF', 'RANDOM', 'HEFT'
    rng: np.random.RandomState = None,
) -> Tuple[Dict[int, int], Dict[int, float], Dict[int, float]]:
    """
    Core function: Schedules tasks in parallel considering DAG dependencies.
    This function utilizes the Earliest Finish Time (EFT) heuristic for allocation.
    """
    num_tasks = G.number_of_nodes()
    num_processors = len(processor_map)
    
    # State variables
    task_status = {i: 'UNSCHEDULED' for i in range(num_tasks)}
    proc_free_time = np.zeros(num_processors) # Finish time of the last task on each core
    
    assigned = {}
    start = {}
    finish = {}
    
    # Predecessor finish time (Ready time for the current task)
    predecessor_finish_time = np.zeros(num_tasks)
    
    # Main scheduling loop
    while any(status != 'SCHEDULED' for status in task_status.values()):
        
        # 1. Identify Ready Tasks
        ready_tasks = []
        for i in range(num_tasks):
            if task_status[i] == 'UNSCHEDULED':
                # A task is ready if all its predecessors are finished
                predecessors_scheduled = True
                max_predecessor_finish = 0.0
                
                for pred in G.predecessors(i):
                    if task_status[pred] != 'SCHEDULED':
                        predecessors_scheduled = False
                        break
                    # Ready time for start: finish time of the last predecessor
                    max_predecessor_finish = max(max_predecessor_finish, finish.get(pred, 0.0))
                
                if predecessors_scheduled:
                    ready_tasks.append(i)
                    predecessor_finish_time[i] = max_predecessor_finish
        
        if not ready_tasks:
            # If no ready task exists, scheduling is complete
            break
        
        # 2. Select a task based on policy (EDF or RANDOM)
        if policy == 'RANDOM':
            # In RANDOM policy, randomly select one task
            chosen_task = rng.choice(ready_tasks)
        elif policy in ['EDF', 'HEFT']:
            # In deadline-based/HEFT policies, select the task with the earliest deadline
            deadlines_ready = {t: deadlines[t] for t in ready_tasks}
            chosen_task = min(deadlines_ready, key=deadlines_ready.get)
        else:
            raise ValueError(f"Invalid scheduling policy: {policy}")
            
        # 3. Processor assignment for the selected task (based on Earliest Finish Time - EFT)
        
        EFT_min = float('inf') 
        best_processor_index = -1
        best_start_time = 0.0
        
        # Check every available processor
        for p_idx in range(num_processors):
            proc_type = processor_map[p_idx] # 0=A7, 1=A12
            
            # Execution time on this processor
            exec_time_on_p = exec_times[chosen_task, proc_type]
            
            # Possible start time: max(processor free time, task ready time)
            start_time = max(proc_free_time[p_idx], predecessor_finish_time[chosen_task])
            
            # Finish time
            finish_time = start_time + exec_time_on_p
            
            # Update best choice
            if finish_time < EFT_min:
                EFT_min = finish_time
                best_processor_index = p_idx
                best_start_time = start_time
        
        # 4. Apply Assignment
        p_idx = best_processor_index
        
        assigned[chosen_task] = processor_map[p_idx] # Assigned core type (0 or 1)
        start[chosen_task] = best_start_time
        finish[chosen_task] = EFT_min
        
        # Update core status
        task_status[chosen_task] = 'SCHEDULED'
        proc_free_time[p_idx] = EFT_min 
        
    return assigned, start, finish


# Rewritten Scheduling Algorithms (Four Dual-Core Scenarios)

def schedule_random_A7_A12(G: nx.DiGraph, exec_times: np.ndarray, deadlines: np.ndarray, rng: np.random.RandomState):
    """
    Scenario 3: Random A7 and A12 (Heterogeneous Dual-Core)
    """
    processor_map = [0, 1] # One A7 core and one A12 core
    return _schedule_parallel_dag(G, exec_times, deadlines, processor_map, policy='RANDOM', rng=rng)


def schedule_dual_A7(G: nx.DiGraph, exec_times: np.ndarray, deadlines: np.ndarray):
    """
    Scenario 1: Dual A7 (Homogeneous Dual-Core)
    """
    processor_map = [0, 0] # Two A7 cores
    return _schedule_parallel_dag(G, exec_times, deadlines, processor_map, policy='EDF')


def schedule_dual_A12(G: nx.DiGraph, exec_times: np.ndarray, deadlines: np.ndarray):
    """
    Scenario 2: Dual A12 (Homogeneous Dual-Core)
    """
    processor_map = [1, 1] # Two A12 cores
    return _schedule_parallel_dag(G, exec_times, deadlines, processor_map, policy='EDF')


def schedule_deadline_based_HEFT(G: nx.DiGraph, exec_times: np.ndarray, deadlines: np.ndarray):
    """
    Scenario 4: EDF/HEFT-like A7 + A12 (Heterogeneous Dual-Core)
    Uses the Earliest Deadline First (EDF) policy combined with EFT assignment.
    """
    processor_map = [0, 1] # One A7 core and one A12 core
    return _schedule_parallel_dag(G, exec_times, deadlines, processor_map, policy='EDF')