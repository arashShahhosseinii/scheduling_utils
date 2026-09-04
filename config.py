from pathlib import Path

# ============================================================
# Project paths
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_FILENAME = "dag_dataset_gang_a7_a12.csv"
DATASET_PATH = SCRIPT_DIR / DATASET_FILENAME

ARTIFACT_DIR = SCRIPT_DIR / "artifacts_gang"
MODEL_DIR = ARTIFACT_DIR / "models"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
LOG_DIR = ARTIFACT_DIR / "logs"
TENSORBOARD_DIR = LOG_DIR / "tensorboard"
EVAL_DIR = ARTIFACT_DIR / "evaluation"
MODEL_PATH = MODEL_DIR / "ppo_gat_gang.zip"

# ============================================================
# Reproducibility
# ============================================================
SEED = 42

# ============================================================
# Gang hardware model
# ============================================================
# m_i is confirmed to be the exact number of cores that task i
# needs simultaneously.
# Since m_i is generated in [1, 6], using 6 A7 + 6 A12 cores
# guarantees that every task can be scheduled as a homogeneous gang:
# either m_i A7 cores or m_i A12 cores. Mixed A7/A12 gangs are disabled.
NUM_A7_CORES = 6
NUM_A12_CORES = 6
MAX_GANG_SIZE = 6
ALLOW_MIXED_GANGS = False

# Runtime model:
#   "perfect_linear_speedup"
#       service_rate = nA7/T_A7 + nA12/T_A12
#       T_gang = 1 / service_rate
#
#   "dataset_time_is_gang_time"
#       no inferred width speedup; for mixed gangs the slower
#       participating processor type determines the common duration.
RUNTIME_MODEL = "perfect_linear_speedup"

# ============================================================
# Reward / QoS
# ============================================================
QOS_FACTOR = 1.20
REWARD_PROPOSAL = "A"

# Main advisor-requested experiment:
#   reward = QoS * energy_term
# and makespan is retained only as an evaluation metric.
REWARD_MODE = "qos_energy_only"

# Kept so the earlier reward ablation can be restored with:
# REWARD_MODE = "qos_energy_makespan"
REWARD_WEIGHTS = (0.008, 45.0)  # (wE, wM)

# ============================================================
# PPO training
# ============================================================
TOTAL_TIMESTEPS = 200_000
CHECKPOINT_FREQUENCY = 10_000
LEARNING_RATE = 3e-4
N_STEPS = 2048
BATCH_SIZE = 256
N_EPOCHS = 8
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.05
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

# Exploration schedule requested by the advisor.
EXPLORATION_ALPHA_START = 0.60
EXPLORATION_ALPHA_END = 0.01

# ============================================================
# Evaluation
# ============================================================
EVALUATION_ALPHA = 0.01
EVALUATION_RUNS = 20