# Heterogeneous DAG Scheduler with PPO + MLP

Reinforcement learning (PPO) based scheduler for DAG tasks on a heterogeneous two‑core processor (A7 / A12). The environment uses a custom Gymnasium space with graph observations, and the policy uses a fast MLP feature extractor with action masking. Trained with Stable‑Baselines3.

## Features

- DAG scheduling on two heterogeneous cores (A7 = 0, A12 = 1)
- Custom Gymnasium environment with:
  - Node features (deadline, execution times, energy, rank, mask, etc.)
  - Edge index and mask for graph structure
  - Core features (processor type, available time, queue length)
  - Global features (progress, energy, makespan, ready count)
  - Action masking (only ready tasks on any core)
- PPO policy with MLP mean‑pool feature extractor (fast, no GAT loops)
- Reward based on delta energy and delta makespan (tardiness not penalized in reward)
- Evaluation against HEFT baseline
- Training and evaluation scripts with artifact saving