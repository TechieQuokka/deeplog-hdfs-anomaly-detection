"""
Data preprocessing pipeline for DeepLog HDFS dataset.

This module handles:
1. Loading Event_traces.csv and anomaly_label.csv
2. Splitting data into train/test following paper specification
3. Generating sliding windows for sequence prediction
4. Saving processed sequences to pickle files
"""

import pandas as pd
import numpy as np
import pickle
import logging
from typing import Dict, List, Tuple
from collections import defaultdict

import config

# Configure logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class HDFSPreprocessor:
    """
    Preprocessor for HDFS log data following DeepLog paper specification.

    Paper specification:
    - Training: First 100,000 log entries → 4,855 normal sessions
    - Testing: Remaining logs → 553,366 normal + 16,838 abnormal sessions
    - Window size: h = 10
    """

    def __init__(self):
        self.window_size = config.WINDOW_SIZE
        self.train_log_limit = config.TRAIN_LOG_LIMIT

        # Mapping from log key string to integer index
        self.log_key_to_idx = {}
        self.idx_to_log_key = {}

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load Event_traces.csv and anomaly_label.csv.

        Returns:
            event_traces: DataFrame with columns [BlockId, EventSequence]
            anomaly_labels: DataFrame with columns [BlockId, Label]
        """
        logger.info(f"Loading event traces from: {config.EVENT_TRACES}")
        event_traces = pd.read_csv(config.EVENT_TRACES)

        logger.info(f"Loading anomaly labels from: {config.ANOMALY_LABEL}")
        anomaly_labels = pd.read_csv(config.ANOMALY_LABEL)

        logger.info(f"Event traces shape: {event_traces.shape}")
        logger.info(f"Anomaly labels shape: {anomaly_labels.shape}")

        return event_traces, anomaly_labels

    def build_log_key_mapping(self, event_traces: pd.DataFrame) -> None:
        """
        Build mapping from log key strings to integer indices.

        Args:
            event_traces: DataFrame containing Features column
        """
        logger.info("Building log key to index mapping...")

        unique_keys = set()
        for seq in event_traces['Features']:
            # Features is in format "[E5,E22,E5,...]"
            # Parse and extract event keys
            if pd.isna(seq) or seq == '':
                continue
            # Remove brackets and split by comma
            seq_str = seq.strip('[]')
            keys = [k.strip() for k in seq_str.split(',')]
            unique_keys.update(keys)

        # Sort for deterministic ordering
        unique_keys = sorted(unique_keys)

        self.log_key_to_idx = {key: idx for idx, key in enumerate(unique_keys)}
        self.idx_to_log_key = {idx: key for key, idx in self.log_key_to_idx.items()}

        logger.info(f"Found {len(unique_keys)} unique log keys")

        # Verify against paper specification
        if len(unique_keys) != config.NUM_CLASSES:
            logger.warning(
                f"Number of log keys ({len(unique_keys)}) differs from "
                f"paper specification ({config.NUM_CLASSES})"
            )

    def split_train_test(
        self,
        event_traces: pd.DataFrame,
        anomaly_labels: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/test following paper specification.

        Paper: "In the case of HDFS log, only less than 1% of normal sessions
                (4,855 sessions parsed from the first 100,000 log entries)"

        Args:
            event_traces: DataFrame with [BlockId, EventSequence]
            anomaly_labels: DataFrame with [BlockId, Label]

        Returns:
            train_traces, train_labels, test_traces, test_labels
        """
        logger.info("Splitting train/test data...")

        # Count total log entries per session
        def count_events(x):
            if pd.isna(x) or x == '':
                return 0
            # Remove brackets and split by comma
            return len(x.strip('[]').split(','))

        event_traces['LogCount'] = event_traces['Features'].apply(count_events)

        # Calculate cumulative log count
        event_traces['CumulativeLogCount'] = event_traces['LogCount'].cumsum()

        # Split: first TRAIN_LOG_LIMIT log entries for training
        train_mask = event_traces['CumulativeLogCount'] <= self.train_log_limit

        train_traces = event_traces[train_mask].copy()
        test_traces = event_traces[~train_mask].copy()

        # Get corresponding labels
        train_block_ids = set(train_traces['BlockId'])
        test_block_ids = set(test_traces['BlockId'])

        train_labels = anomaly_labels[anomaly_labels['BlockId'].isin(train_block_ids)]
        test_labels = anomaly_labels[anomaly_labels['BlockId'].isin(test_block_ids)]

        # Filter training data to normal sessions only (paper: "only normal sessions for training")
        normal_train_block_ids = set(
            train_labels[train_labels['Label'] == 'Normal']['BlockId']
        )
        train_traces = train_traces[train_traces['BlockId'].isin(normal_train_block_ids)]
        train_labels = train_labels[train_labels['Label'] == 'Normal']

        # Statistics
        train_normal = (train_labels['Label'] == 'Normal').sum()
        train_abnormal = (train_labels['Label'] == 'Anomaly').sum()
        test_normal = (test_labels['Label'] == 'Normal').sum()
        test_abnormal = (test_labels['Label'] == 'Anomaly').sum()

        logger.info(f"Train sessions: {len(train_traces)} (Normal: {train_normal}, Abnormal: {train_abnormal})")
        logger.info(f"Test sessions: {len(test_traces)} (Normal: {test_normal}, Abnormal: {test_abnormal})")

        # Verify against paper specification
        if abs(train_normal - config.EXPECTED_TRAIN_SESSIONS) > 100:
            logger.warning(
                f"Training sessions ({train_normal}) differ significantly from "
                f"paper specification ({config.EXPECTED_TRAIN_SESSIONS})"
            )

        # Drop temporary columns
        train_traces = train_traces.drop(['LogCount', 'CumulativeLogCount'], axis=1)
        test_traces = test_traces.drop(['LogCount', 'CumulativeLogCount'], axis=1)

        return train_traces, train_labels, test_traces, test_labels

    def generate_sequences(
        self,
        traces: pd.DataFrame,
        labels: pd.DataFrame
    ) -> Dict[str, List]:
        """
        Generate sliding window sequences for training or testing.

        Args:
            traces: DataFrame with [BlockId, EventSequence]
            labels: DataFrame with [BlockId, Label]

        Returns:
            Dictionary with:
                - 'sequences': List of (input_window, label) tuples
                - 'block_ids': List of BlockId for each sequence
                - 'session_labels': List of session-level labels (Normal/Anomaly)
        """
        logger.info(f"Generating sequences with window size {self.window_size}...")

        sequences = []
        block_ids = []
        session_labels = []

        # Create label lookup
        label_dict = dict(zip(labels['BlockId'], labels['Label']))

        for _, row in traces.iterrows():
            block_id = row['BlockId']
            # Features is in format "[E5,E22,E5,...]"
            if pd.isna(row['Features']) or row['Features'] == '':
                continue
            # Remove brackets and split by comma
            seq_str = row['Features'].strip('[]')
            event_seq = [k.strip() for k in seq_str.split(',')]
            session_label = label_dict.get(block_id, 'Normal')

            # Convert log keys to indices
            event_indices = [self.log_key_to_idx[key] for key in event_seq]

            # Generate sliding windows
            for i in range(len(event_indices) - self.window_size):
                # Input: window of size h
                input_window = event_indices[i:i + self.window_size]
                # Label: next log key
                label = event_indices[i + self.window_size]

                sequences.append((input_window, label))
                block_ids.append(block_id)
                session_labels.append(session_label)

        logger.info(f"Generated {len(sequences)} sequences")

        return {
            'sequences': sequences,
            'block_ids': block_ids,
            'session_labels': session_labels
        }

    def save_sequences(self, data: Dict, filepath: str) -> None:
        """Save processed sequences to pickle file."""
        logger.info(f"Saving sequences to: {filepath}")
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logger.info("Sequences saved successfully")

    def preprocess(self) -> None:
        """
        Main preprocessing pipeline.

        Steps:
        1. Load raw data
        2. Build log key mapping
        3. Split train/test
        4. Generate sequences
        5. Save to pickle files
        """
        logger.info("=" * 80)
        logger.info("Starting HDFS data preprocessing pipeline")
        logger.info("=" * 80)

        # Ensure output directory exists
        config.create_directories()

        # Step 1: Load data
        event_traces, anomaly_labels = self.load_data()

        # Step 2: Build log key mapping
        self.build_log_key_mapping(event_traces)

        # Step 3: Split train/test
        train_traces, train_labels, test_traces, test_labels = self.split_train_test(
            event_traces, anomaly_labels
        )

        # Step 4: Generate sequences
        logger.info("Processing training data...")
        train_data = self.generate_sequences(train_traces, train_labels)
        train_data['log_key_to_idx'] = self.log_key_to_idx
        train_data['idx_to_log_key'] = self.idx_to_log_key

        logger.info("Processing test data...")
        test_data = self.generate_sequences(test_traces, test_labels)
        test_data['log_key_to_idx'] = self.log_key_to_idx
        test_data['idx_to_log_key'] = self.idx_to_log_key

        # Step 5: Save sequences
        self.save_sequences(train_data, config.TRAIN_DATA)
        self.save_sequences(test_data, config.TEST_DATA)

        logger.info("=" * 80)
        logger.info("Preprocessing completed successfully!")
        logger.info("=" * 80)
        logger.info(f"Training sequences: {len(train_data['sequences'])}")
        logger.info(f"Test sequences: {len(test_data['sequences'])}")
        logger.info(f"Number of unique log keys: {len(self.log_key_to_idx)}")


def main():
    """Main function for standalone execution."""
    preprocessor = HDFSPreprocessor()
    preprocessor.preprocess()


if __name__ == "__main__":
    main()
