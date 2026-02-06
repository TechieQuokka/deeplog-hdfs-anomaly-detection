#!/usr/bin/env python3
"""
DeepLog HDFS Anomaly Detection - Main Pipeline

This script orchestrates the complete pipeline:
1. Data preprocessing
2. Model training
3. Model evaluation
4. Results comparison with paper

Usage:
    python main.py [--preprocess] [--train] [--evaluate] [--all]
"""

import argparse
import sys
import os
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import config
from preprocessing import HDFSPreprocessor
from train import main as train_main
from evaluate_optimized import main as evaluate_main
from utils import (
    set_seed,
    print_config_summary,
    check_data_files_exist,
    check_processed_data_exist
)

# Configure logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format=config.LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('deeplog.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DeepLog HDFS Anomaly Detection Reproduction"
    )

    parser.add_argument(
        '--preprocess',
        action='store_true',
        help='Run data preprocessing'
    )

    parser.add_argument(
        '--train',
        action='store_true',
        help='Run model training'
    )

    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Run model evaluation'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Run complete pipeline (preprocess + train + evaluate)'
    )

    parser.add_argument(
        '--skip-preprocess',
        action='store_true',
        help='Skip preprocessing if data already exists'
    )

    return parser.parse_args()


def run_preprocessing():
    """Run data preprocessing pipeline."""
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 1: Data Preprocessing")
    logger.info("=" * 80)

    # Check if raw data files exist
    if not check_data_files_exist():
        logger.error("Required data files not found!")
        logger.error(f"Please ensure data is available at: {config.DATA_ROOT}")
        return False

    # Run preprocessing
    preprocessor = HDFSPreprocessor()
    preprocessor.preprocess()

    logger.info("✓ Preprocessing completed successfully\n")
    return True


def run_training():
    """Run model training pipeline."""
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 2: Model Training")
    logger.info("=" * 80)

    # Check if preprocessed data exists
    if not check_processed_data_exist():
        logger.error("Preprocessed data not found!")
        logger.error("Please run preprocessing first: python main.py --preprocess")
        return False

    # Run training
    train_main()

    logger.info("✓ Training completed successfully\n")
    return True


def run_evaluation():
    """Run model evaluation pipeline."""
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 3: Model Evaluation")
    logger.info("=" * 80)

    # Check if trained model exists
    if not os.path.exists(config.BEST_MODEL_PATH):
        logger.error("Trained model not found!")
        logger.error("Please run training first: python main.py --train")
        return False

    # Run evaluation
    evaluate_main()

    logger.info("✓ Evaluation completed successfully\n")
    return True


def main():
    """Main pipeline orchestration."""
    # Parse arguments
    args = parse_arguments()

    # Print header
    print("\n" + "=" * 80)
    print("DeepLog HDFS Anomaly Detection - Reproduction Study")
    print("Paper: Min Du et al., CCS 2017")
    print("=" * 80)

    # Set random seed for reproducibility
    set_seed(config.RANDOM_SEED)

    # Print configuration
    print_config_summary()

    # Create directories
    config.create_directories()

    # Determine what to run
    run_preprocess = args.preprocess or args.all
    run_train = args.train or args.all
    run_eval = args.evaluate or args.all

    # If nothing specified, show help
    if not (run_preprocess or run_train or run_eval):
        logger.info("No action specified. Use --help to see available options.")
        logger.info("\nQuick start:")
        logger.info("  python main.py --all              # Run complete pipeline")
        logger.info("  python main.py --preprocess       # Run preprocessing only")
        logger.info("  python main.py --train            # Run training only")
        logger.info("  python main.py --evaluate         # Run evaluation only")
        return

    # Track success
    success = True

    # Stage 1: Preprocessing
    if run_preprocess:
        # Skip if data exists and --skip-preprocess is set
        if args.skip_preprocess and check_processed_data_exist():
            logger.info("Skipping preprocessing (data already exists)")
        else:
            success = run_preprocessing()
            if not success:
                logger.error("Preprocessing failed!")
                return

    # Stage 2: Training
    if run_train and success:
        success = run_training()
        if not success:
            logger.error("Training failed!")
            return

    # Stage 3: Evaluation
    if run_eval and success:
        success = run_evaluation()
        if not success:
            logger.error("Evaluation failed!")
            return

    # Final summary
    if success:
        logger.info("\n" + "=" * 80)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 80)

        if run_eval:
            logger.info(f"\nResults saved to: {config.METRICS_FILE}")
            logger.info(f"Best model saved to: {config.BEST_MODEL_PATH}")

        logger.info("\nNext steps:")
        if not run_eval:
            logger.info("  - Run evaluation: python main.py --evaluate")
        logger.info("  - Check results in: ./results/metrics.json")
        logger.info("  - View logs in: ./deeplog.log")


if __name__ == "__main__":
    main()
