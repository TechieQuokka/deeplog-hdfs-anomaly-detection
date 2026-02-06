"""
Configuration file for DeepLog HDFS anomaly detection reproduction.
All hyperparameters match the original paper specification.
"""

import os
import torch

# ============================================================================
# Data Paths
# ============================================================================

# Raw HDFS data location (from /mnt/e/Big Data/HDFS/HDFS_v1)
DATA_ROOT = "/mnt/e/Big Data/HDFS/HDFS_v1"
RAW_LOG = os.path.join(DATA_ROOT, "HDFS.log")
PREPROCESSED_DIR = os.path.join(DATA_ROOT, "preprocessed")
EVENT_TRACES = os.path.join(PREPROCESSED_DIR, "Event_traces.csv")
ANOMALY_LABEL = os.path.join(PREPROCESSED_DIR, "anomaly_label.csv")
LOG_TEMPLATES = os.path.join(PREPROCESSED_DIR, "HDFS.log_templates.csv")

# Processed data output paths
PROCESSED_DIR = "./data/processed"
TRAIN_DATA = os.path.join(PROCESSED_DIR, "train_sequences.pkl")
TEST_DATA = os.path.join(PROCESSED_DIR, "test_sequences.pkl")

# ============================================================================
# Model Hyperparameters (Paper Specification)
# ============================================================================

# Paper: "By default, we use the following parameter values for DeepLog:
#         g = 9, h = 10, L = 2, and α = 64"

NUM_CLASSES = 29        # n: Number of distinct log keys in HDFS
WINDOW_SIZE = 10        # h: History window size for sequence prediction
NUM_LAYERS = 2          # L: Number of LSTM layers
HIDDEN_SIZE = 64        # α: Number of memory units in one LSTM block
EMBEDDING_DIM = 64      # Embedding dimension (not specified in paper, using common practice)
TOP_G = 9               # g: Top-g prediction threshold for anomaly detection

# ============================================================================
# Training Configuration
# ============================================================================

# Batch size (not specified in paper, using standard value)
BATCH_SIZE = 128

# Evaluation batch size (larger for faster inference)
EVAL_BATCH_SIZE = 512  # 4x larger for evaluation

# Learning rate (not specified in paper, will need tuning)
LEARNING_RATE = 0.001

# Number of epochs (not specified in paper, will need tuning)
NUM_EPOCHS = 100

# Early stopping patience
EARLY_STOPPING_PATIENCE = 10

# Train/validation split ratio (normal training data only)
TRAIN_VAL_SPLIT = 0.9  # 90% train, 10% validation

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# Data Split Configuration (Paper Specification)
# ============================================================================

# Paper: "In the case of HDFS log, only less than 1% of normal sessions
#         (4,855 sessions parsed from the first 100,000 log entries compared
#         to a total of 11,197,954) are used for training."

# Training data: First 100,000 log entries
TRAIN_LOG_LIMIT = 100000

# Expected training sessions (from paper)
EXPECTED_TRAIN_SESSIONS = 4855

# Expected test sessions (from paper)
EXPECTED_TEST_NORMAL_SESSIONS = 553366
EXPECTED_TEST_ABNORMAL_SESSIONS = 16838

# ============================================================================
# Checkpoint and Output Paths
# ============================================================================

CHECKPOINT_DIR = "./checkpoints"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

RESULTS_DIR = "./results"
METRICS_FILE = os.path.join(RESULTS_DIR, "metrics.json")

# ============================================================================
# Logging Configuration
# ============================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ============================================================================
# Target Performance Metrics (Paper Results)
# ============================================================================

# Paper Table 4: Number of FPs and FNs on HDFS log
TARGET_FP = 833
TARGET_FN = 619

# Calculated from paper results
TARGET_PRECISION = 0.951  # Approximately 95.1%
TARGET_RECALL = 0.964     # Approximately 96.4%
TARGET_F_MEASURE = 0.96   # 96% (reported in paper)

# ============================================================================
# Helper Functions
# ============================================================================

def create_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

def print_config():
    """Print current configuration."""
    print("=" * 80)
    print("DeepLog HDFS Configuration")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Data Root: {DATA_ROOT}")
    print(f"Event Traces: {EVENT_TRACES}")
    print(f"Anomaly Labels: {ANOMALY_LABEL}")
    print("-" * 80)
    print("Model Hyperparameters (Paper Spec):")
    print(f"  - Number of log keys (n): {NUM_CLASSES}")
    print(f"  - Window size (h): {WINDOW_SIZE}")
    print(f"  - LSTM layers (L): {NUM_LAYERS}")
    print(f"  - Hidden size (α): {HIDDEN_SIZE}")
    print(f"  - Top-g threshold: {TOP_G}")
    print("-" * 80)
    print("Training Configuration:")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Epochs: {NUM_EPOCHS}")
    print(f"  - Random seed: {RANDOM_SEED}")
    print("-" * 80)
    print("Target Performance (Paper):")
    print(f"  - False Positives: {TARGET_FP}")
    print(f"  - False Negatives: {TARGET_FN}")
    print(f"  - Precision: {TARGET_PRECISION:.1%}")
    print(f"  - Recall: {TARGET_RECALL:.1%}")
    print(f"  - F-measure: {TARGET_F_MEASURE:.1%}")
    print("=" * 80)

if __name__ == "__main__":
    print_config()
