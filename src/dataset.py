"""
PyTorch Dataset and DataLoader for DeepLog HDFS sequences.
"""

import pickle
import logging
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple, Dict, List

import config

# Configure logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class HDFSDataset(Dataset):
    """
    PyTorch Dataset for HDFS log sequences.

    Each sample consists of:
        - Input: sequence of log key indices [window_size]
        - Label: next log key index (single integer)
    """

    def __init__(self, data_path: str):
        """
        Initialize dataset from preprocessed pickle file.

        Args:
            data_path: Path to preprocessed sequences pickle file
        """
        logger.info(f"Loading dataset from: {data_path}")

        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        self.sequences = data['sequences']  # List of (input_window, label) tuples
        self.block_ids = data['block_ids']  # Block IDs for each sequence
        self.session_labels = data['session_labels']  # Session-level labels
        self.log_key_to_idx = data['log_key_to_idx']
        self.idx_to_log_key = data['idx_to_log_key']

        logger.info(f"Loaded {len(self.sequences)} sequences")
        logger.info(f"Number of unique log keys: {len(self.log_key_to_idx)}")

    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sequence sample.

        Args:
            idx: Index of the sample

        Returns:
            input_seq: Tensor of shape [window_size] containing log key indices
            label: Tensor of shape [] (scalar) containing next log key index
        """
        input_window, label = self.sequences[idx]

        # Convert to tensors
        input_seq = torch.tensor(input_window, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)

        return input_seq, label

    def get_block_id(self, idx: int) -> str:
        """Get the BlockId for a given sequence index."""
        return self.block_ids[idx]

    def get_session_label(self, idx: int) -> str:
        """Get the session-level label (Normal/Anomaly) for a given sequence."""
        return self.session_labels[idx]

    def get_statistics(self) -> Dict:
        """
        Get dataset statistics.

        Returns:
            Dictionary with dataset statistics
        """
        normal_count = sum(1 for label in self.session_labels if label == 'Normal')
        anomaly_count = sum(1 for label in self.session_labels if label == 'Anomaly')

        return {
            'total_sequences': len(self.sequences),
            'normal_sequences': normal_count,
            'anomaly_sequences': anomaly_count,
            'unique_log_keys': len(self.log_key_to_idx),
            'window_size': len(self.sequences[0][0]) if len(self.sequences) > 0 else 0
        }


def create_data_loaders(
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = 4,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training, validation, and testing.

    Training data is split into train/val using config.TRAIN_VAL_SPLIT.
    Validation uses only normal data (same distribution as training).

    Args:
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes for data loading
        shuffle_train: Whether to shuffle training data

    Returns:
        train_loader, val_loader, test_loader
    """
    logger.info("Creating DataLoaders...")

    # Load datasets
    full_train_dataset = HDFSDataset(config.TRAIN_DATA)
    test_dataset = HDFSDataset(config.TEST_DATA)

    # Split training data into train/val
    train_size = int(len(full_train_dataset) * config.TRAIN_VAL_SPLIT)
    val_size = len(full_train_dataset) - train_size
    generator = torch.Generator().manual_seed(config.RANDOM_SEED)
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size], generator=generator
    )

    # Print statistics
    logger.info("Full training dataset statistics:")
    train_stats = full_train_dataset.get_statistics()
    for key, value in train_stats.items():
        logger.info(f"  {key}: {value}")
    logger.info(f"  train split: {train_size}, val split: {val_size}")

    logger.info("Test dataset statistics:")
    test_stats = test_dataset.get_statistics()
    for key, value in test_stats.items():
        logger.info(f"  {key}: {value}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    logger.info(f"Created DataLoaders with batch_size={batch_size}")
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader


def main():
    """Test function for dataset loading."""
    logger.info("Testing dataset loading...")

    # Load train dataset
    train_dataset = HDFSDataset(config.TRAIN_DATA)
    test_dataset = HDFSDataset(config.TEST_DATA)

    # Print statistics
    print("\n" + "=" * 80)
    print("Training Dataset Statistics")
    print("=" * 80)
    stats = train_dataset.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\n" + "=" * 80)
    print("Test Dataset Statistics")
    print("=" * 80)
    stats = test_dataset.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")

    # Test data access
    print("\n" + "=" * 80)
    print("Sample Data")
    print("=" * 80)
    input_seq, label = train_dataset[0]
    print(f"Input sequence shape: {input_seq.shape}")
    print(f"Input sequence: {input_seq}")
    print(f"Label: {label}")
    print(f"Block ID: {train_dataset.get_block_id(0)}")
    print(f"Session label: {train_dataset.get_session_label(0)}")

    # Test DataLoader
    print("\n" + "=" * 80)
    print("Testing DataLoader")
    print("=" * 80)
    train_loader, val_loader, test_loader = create_data_loaders(batch_size=32)

    # Get one batch
    for batch_inputs, batch_labels in train_loader:
        print(f"Batch inputs shape: {batch_inputs.shape}")
        print(f"Batch labels shape: {batch_labels.shape}")
        break

    print("\nDataset test completed successfully!")


if __name__ == "__main__":
    main()
