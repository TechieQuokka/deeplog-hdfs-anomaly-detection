"""
Utility functions for DeepLog HDFS reproduction.
"""

import torch
import numpy as np
import random
import json
import logging
from typing import Dict, Any
import os

import config

# Configure logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


def set_seed(seed: int = config.RANDOM_SEED):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Random seed set to {seed}")


def get_device() -> torch.device:
    """
    Get the device to use for computation.

    Returns:
        torch.device (cuda or cpu)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.info("Using CPU device")

    return device


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_json(data: Dict[str, Any], filepath: str):
    """
    Save dictionary to JSON file.

    Args:
        data: Dictionary to save
        filepath: Output file path
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved data to {filepath}")


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load dictionary from JSON file.

    Args:
        filepath: Input file path

    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    logger.info(f"Loaded data from {filepath}")
    return data


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def print_config_summary():
    """Print a summary of the current configuration."""
    print("\n" + "=" * 80)
    print("DeepLog HDFS Reproduction - Configuration Summary")
    print("=" * 80)
    print(f"Device: {config.DEVICE}")
    print(f"Random Seed: {config.RANDOM_SEED}")
    print("-" * 80)
    print("Data:")
    print(f"  Data Root: {config.DATA_ROOT}")
    print(f"  Window Size: {config.WINDOW_SIZE}")
    print(f"  Train Log Limit: {config.TRAIN_LOG_LIMIT:,}")
    print("-" * 80)
    print("Model Architecture:")
    print(f"  Number of Classes: {config.NUM_CLASSES}")
    print(f"  Embedding Dim: {config.EMBEDDING_DIM}")
    print(f"  Hidden Size: {config.HIDDEN_SIZE}")
    print(f"  Num Layers: {config.NUM_LAYERS}")
    print(f"  Top-g Threshold: {config.TOP_G}")
    print("-" * 80)
    print("Training:")
    print(f"  Batch Size: {config.BATCH_SIZE}")
    print(f"  Learning Rate: {config.LEARNING_RATE}")
    print(f"  Num Epochs: {config.NUM_EPOCHS}")
    print(f"  Early Stopping Patience: {config.EARLY_STOPPING_PATIENCE}")
    print("-" * 80)
    print("Target Performance (Paper):")
    print(f"  FP: {config.TARGET_FP}")
    print(f"  FN: {config.TARGET_FN}")
    print(f"  Precision: {config.TARGET_PRECISION:.1%}")
    print(f"  Recall: {config.TARGET_RECALL:.1%}")
    print(f"  F-measure: {config.TARGET_F_MEASURE:.1%}")
    print("=" * 80 + "\n")


def check_data_files_exist() -> bool:
    """
    Check if required data files exist.

    Returns:
        True if all files exist, False otherwise
    """
    required_files = [
        config.EVENT_TRACES,
        config.ANOMALY_LABEL
    ]

    all_exist = True
    for filepath in required_files:
        if not os.path.exists(filepath):
            logger.error(f"Required file not found: {filepath}")
            all_exist = False
        else:
            logger.info(f"Found required file: {filepath}")

    return all_exist


def check_processed_data_exist() -> bool:
    """
    Check if preprocessed data files exist.

    Returns:
        True if processed data exists, False otherwise
    """
    return (
        os.path.exists(config.TRAIN_DATA) and
        os.path.exists(config.TEST_DATA)
    )


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        Update statistics.

        Args:
            val: New value
            n: Number of samples
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_metrics_table(metrics: Dict[str, float]):
    """
    Print metrics in a formatted table.

    Args:
        metrics: Dictionary of metrics
    """
    print("\n" + "=" * 80)
    print("Evaluation Metrics")
    print("=" * 80)

    for key, value in metrics.items():
        if isinstance(value, float):
            if 0 <= value <= 1:
                # Assume it's a ratio/percentage
                print(f"{key:30s} | {value:8.4f} ({value*100:6.2f}%)")
            else:
                print(f"{key:30s} | {value:8.4f}")
        else:
            print(f"{key:30s} | {value}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Test utilities
    print_config_summary()

    print("\nChecking data files...")
    if check_data_files_exist():
        print("✓ All required data files found")
    else:
        print("✗ Some data files are missing")

    print("\nChecking processed data...")
    if check_processed_data_exist():
        print("✓ Processed data found")
    else:
        print("✗ Processed data not found (run preprocessing first)")

    print("\nDevice information:")
    device = get_device()

    print("\nSetting random seed...")
    set_seed(42)
    print("✓ Random seed set")

    print("\nTest completed!")
