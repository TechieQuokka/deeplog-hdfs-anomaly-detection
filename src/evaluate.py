"""
Evaluation pipeline for DeepLog with top-g prediction strategy.

Paper evaluation methodology:
- Top-g prediction: predict top-g (g=9) most probable log keys
- Session-level detection: a session is abnormal if ANY log key is detected as anomaly
- Metrics: FP, FN, Precision, Recall, F-measure
"""

import torch
import logging
import json
from typing import Dict, List, Tuple
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

import config
from model import DeepLogLSTM, create_model
from dataset import HDFSDataset
from train import load_checkpoint

# Configure logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluator for DeepLog model using top-g prediction strategy.

    Paper methodology:
    1. For each log key sequence, predict top-g most probable next keys
    2. If actual key is NOT in top-g → mark as anomaly
    3. Aggregate to session level: session is abnormal if ANY key is anomaly
    4. Compare with ground truth labels
    """

    def __init__(
        self,
        model: DeepLogLSTM,
        test_dataset: HDFSDataset,
        device: torch.device = config.DEVICE,
        top_g: int = config.TOP_G
    ):
        """
        Initialize evaluator.

        Args:
            model: Trained DeepLog model
            test_dataset: Test dataset
            device: Device to run evaluation on
            top_g: Top-g threshold (default: 9)
        """
        self.model = model
        self.test_dataset = test_dataset
        self.device = device
        self.top_g = top_g

        self.model.eval()
        logger.info(f"Evaluator initialized with top-g={top_g}")

    def detect_anomalies_sequence_level(self) -> Dict[int, bool]:
        """
        Detect anomalies at sequence level using top-g prediction.

        Returns:
            Dictionary mapping sequence index to anomaly flag (True = anomaly)
        """
        logger.info("Detecting anomalies at sequence level...")

        anomalies = {}
        num_anomalies = 0

        with torch.no_grad():
            for idx in range(len(self.test_dataset)):
                # Get sequence and label
                input_seq, true_label = self.test_dataset[idx]

                # Add batch dimension and move to device
                input_seq = input_seq.unsqueeze(0).to(self.device)

                # Get top-g predictions
                top_g_preds = self.model.predict_top_k(input_seq, k=self.top_g)

                # Check if true label is in top-g predictions
                is_anomaly = true_label.item() not in top_g_preds[0].cpu().tolist()

                anomalies[idx] = is_anomaly

                if is_anomaly:
                    num_anomalies += 1

                # Log progress
                if (idx + 1) % 10000 == 0:
                    logger.info(f"  Processed {idx + 1}/{len(self.test_dataset)} sequences")

        logger.info(f"Sequence-level anomalies detected: {num_anomalies}/{len(self.test_dataset)}")

        return anomalies

    def aggregate_to_session_level(
        self,
        sequence_anomalies: Dict[int, bool]
    ) -> Dict[str, bool]:
        """
        Aggregate sequence-level anomalies to session level.

        A session (BlockId) is marked as abnormal if ANY of its sequences
        is detected as anomaly.

        Args:
            sequence_anomalies: Dictionary mapping sequence index to anomaly flag

        Returns:
            Dictionary mapping BlockId to session-level anomaly flag
        """
        logger.info("Aggregating to session level...")

        session_anomalies = defaultdict(bool)

        for seq_idx, is_anomaly in sequence_anomalies.items():
            block_id = self.test_dataset.get_block_id(seq_idx)

            # If ANY sequence in session is anomaly, mark session as anomaly
            if is_anomaly:
                session_anomalies[block_id] = True

        # Ensure all sessions are in the dictionary
        for seq_idx in range(len(self.test_dataset)):
            block_id = self.test_dataset.get_block_id(seq_idx)
            if block_id not in session_anomalies:
                session_anomalies[block_id] = False

        num_abnormal_sessions = sum(session_anomalies.values())
        logger.info(
            f"Session-level anomalies: {num_abnormal_sessions}/{len(session_anomalies)}"
        )

        return dict(session_anomalies)

    def compute_metrics(
        self,
        session_predictions: Dict[str, bool]
    ) -> Dict:
        """
        Compute evaluation metrics by comparing predictions with ground truth.

        Args:
            session_predictions: Dictionary mapping BlockId to predicted anomaly flag

        Returns:
            Dictionary containing all metrics
        """
        logger.info("Computing evaluation metrics...")

        # Get ground truth labels
        ground_truth = {}
        for seq_idx in range(len(self.test_dataset)):
            block_id = self.test_dataset.get_block_id(seq_idx)
            session_label = self.test_dataset.get_session_label(seq_idx)
            ground_truth[block_id] = (session_label == 'Anomaly')

        # Align predictions and ground truth
        block_ids = sorted(ground_truth.keys())
        y_true = [ground_truth[bid] for bid in block_ids]
        y_pred = [session_predictions[bid] for bid in block_ids]

        # Compute confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Compute metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        metrics = {
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'total_sessions': len(block_ids),
            'abnormal_sessions': sum(y_true),
            'predicted_abnormal': sum(y_pred)
        }

        return metrics

    def evaluate(self) -> Dict:
        """
        Run complete evaluation pipeline.

        Returns:
            Dictionary containing all evaluation results
        """
        logger.info("=" * 80)
        logger.info("Starting evaluation")
        logger.info("=" * 80)

        # Step 1: Detect anomalies at sequence level
        sequence_anomalies = self.detect_anomalies_sequence_level()

        # Step 2: Aggregate to session level
        session_predictions = self.aggregate_to_session_level(sequence_anomalies)

        # Step 3: Compute metrics
        metrics = self.compute_metrics(session_predictions)

        # Log results
        self.print_metrics(metrics)

        # Compare with paper targets
        self.compare_with_paper(metrics)

        return metrics

    def print_metrics(self, metrics: Dict):
        """Print evaluation metrics in a formatted way."""
        logger.info("\n" + "=" * 80)
        logger.info("Evaluation Results")
        logger.info("=" * 80)
        logger.info(f"Total sessions: {metrics['total_sessions']}")
        logger.info(f"Abnormal sessions (ground truth): {metrics['abnormal_sessions']}")
        logger.info(f"Predicted abnormal sessions: {metrics['predicted_abnormal']}")
        logger.info("-" * 80)
        logger.info(f"True Positives (TP):  {metrics['true_positives']}")
        logger.info(f"True Negatives (TN):  {metrics['true_negatives']}")
        logger.info(f"False Positives (FP): {metrics['false_positives']}")
        logger.info(f"False Negatives (FN): {metrics['false_negatives']}")
        logger.info("-" * 80)
        logger.info(f"Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        logger.info(f"Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        logger.info(f"F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
        logger.info("=" * 80)

    def compare_with_paper(self, metrics: Dict):
        """Compare results with paper targets."""
        logger.info("\n" + "=" * 80)
        logger.info("Comparison with Paper Results")
        logger.info("=" * 80)

        comparisons = [
            ("False Positives", metrics['false_positives'], config.TARGET_FP),
            ("False Negatives", metrics['false_negatives'], config.TARGET_FN),
            ("Precision", metrics['precision'], config.TARGET_PRECISION),
            ("Recall", metrics['recall'], config.TARGET_RECALL),
            ("F-measure", metrics['f1_score'], config.TARGET_F_MEASURE)
        ]

        for metric_name, achieved, target in comparisons:
            if metric_name in ["False Positives", "False Negatives"]:
                diff = achieved - target
                diff_pct = (diff / target * 100) if target > 0 else 0
                logger.info(
                    f"{metric_name:20s} | Achieved: {achieved:8d} | "
                    f"Target: {target:8d} | Diff: {diff:+6d} ({diff_pct:+.1f}%)"
                )
            else:
                diff = achieved - target
                diff_pct = (diff / target * 100) if target > 0 else 0
                logger.info(
                    f"{metric_name:20s} | Achieved: {achieved:8.4f} | "
                    f"Target: {target:8.4f} | Diff: {diff:+.4f} ({diff_pct:+.1f}%)"
                )

        logger.info("=" * 80)


def save_results(metrics: Dict, filepath: str = config.METRICS_FILE):
    """Save evaluation results to JSON file."""
    logger.info(f"Saving results to: {filepath}")

    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info("Results saved successfully")


def main():
    """Main evaluation function."""
    logger.info("Starting evaluation pipeline...")

    # Ensure directories exist
    config.create_directories()

    # Load test dataset
    logger.info(f"Loading test dataset from: {config.TEST_DATA}")
    test_dataset = HDFSDataset(config.TEST_DATA)

    # Create and load model
    logger.info("Creating model...")
    model = create_model(device=config.DEVICE)

    logger.info(f"Loading best model from: {config.BEST_MODEL_PATH}")
    model = load_checkpoint(model, config.BEST_MODEL_PATH)

    # Create evaluator
    evaluator = Evaluator(
        model=model,
        test_dataset=test_dataset,
        device=config.DEVICE,
        top_g=config.TOP_G
    )

    # Run evaluation
    metrics = evaluator.evaluate()

    # Save results
    save_results(metrics)

    logger.info("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
