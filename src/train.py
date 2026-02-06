"""
Training pipeline for DeepLog LSTM model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import time
from typing import Tuple, Dict
import os

import config
from model import DeepLogLSTM, create_model
from dataset import create_data_loaders

# Configure logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer class for DeepLog model.

    Handles:
    - Training loop with validation
    - Checkpoint saving
    - Early stopping
    - Learning rate scheduling
    """

    def __init__(
        self,
        model: DeepLogLSTM,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device = config.DEVICE,
        learning_rate: float = config.LEARNING_RATE,
        num_epochs: int = config.NUM_EPOCHS,
        early_stopping_patience: int = config.EARLY_STOPPING_PATIENCE
    ):
        """
        Initialize trainer.

        Args:
            model: DeepLog LSTM model
            train_loader: Training data loader
            val_loader: Validation data loader (normal data only)
            device: Device to train on
            learning_rate: Learning rate for optimizer
            num_epochs: Maximum number of training epochs
            early_stopping_patience: Patience for early stopping
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience

        # Loss function: CrossEntropyLoss (as specified in paper)
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer: Adam (common choice, not specified in paper)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )

        # Learning rate scheduler (optional)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

        logger.info("Trainer initialized")
        logger.info(f"Optimizer: Adam with lr={learning_rate}")
        logger.info(f"Loss function: CrossEntropyLoss")
        logger.info(f"Device: {device}")

    def train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            # Move data to device
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)

            # Compute loss
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Update weights
            self.optimizer.step()

            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1

            # Log progress
            if (batch_idx + 1) % 100 == 0:
                logger.info(
                    f"  Batch [{batch_idx + 1}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self) -> float:
        """
        Validate on test/validation set.

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                # Move data to device
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)

                # Compute loss
                loss = self.criterion(outputs, labels)

                # Accumulate loss
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch number
            val_loss: Validation loss
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': {
                'num_classes': config.NUM_CLASSES,
                'embedding_dim': config.EMBEDDING_DIM,
                'hidden_size': config.HIDDEN_SIZE,
                'num_layers': config.NUM_LAYERS,
                'window_size': config.WINDOW_SIZE
            }
        }

        if is_best:
            checkpoint_path = config.BEST_MODEL_PATH
            logger.info(f"Saving best model to {checkpoint_path}")
            torch.save(checkpoint, checkpoint_path)
        else:
            checkpoint_path = os.path.join(
                config.CHECKPOINT_DIR,
                f"checkpoint_epoch_{epoch}.pth"
            )
            torch.save(checkpoint, checkpoint_path)

    def train(self) -> Dict:
        """
        Main training loop.

        Returns:
            Dictionary containing training history
        """
        logger.info("=" * 80)
        logger.info("Starting training")
        logger.info("=" * 80)

        start_time = time.time()

        for epoch in range(1, self.num_epochs + 1):
            epoch_start = time.time()

            logger.info(f"\nEpoch {epoch}/{self.num_epochs}")
            logger.info("-" * 80)

            # Train for one epoch
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)

            # Update learning rate scheduler
            self.scheduler.step(val_loss)

            epoch_time = time.time() - epoch_start

            logger.info(
                f"Epoch {epoch} completed in {epoch_time:.2f}s | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

            # Check if validation loss improved
            if val_loss < self.best_val_loss:
                logger.info(
                    f"Validation loss improved from {self.best_val_loss:.4f} "
                    f"to {val_loss:.4f}"
                )
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch, val_loss, is_best=True)
            else:
                self.epochs_without_improvement += 1
                logger.info(
                    f"No improvement for {self.epochs_without_improvement} epochs "
                    f"(best: {self.best_val_loss:.4f})"
                )

            # Early stopping check
            if self.epochs_without_improvement >= self.early_stopping_patience:
                logger.info(
                    f"\nEarly stopping triggered after {epoch} epochs "
                    f"(patience: {self.early_stopping_patience})"
                )
                break

        total_time = time.time() - start_time

        logger.info("=" * 80)
        logger.info("Training completed!")
        logger.info(f"Total training time: {total_time:.2f}s")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info("=" * 80)

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'total_time': total_time
        }


def load_checkpoint(model: DeepLogLSTM, checkpoint_path: str) -> DeepLogLSTM:
    """
    Load model from checkpoint.

    Args:
        model: Model instance to load weights into
        checkpoint_path: Path to checkpoint file

    Returns:
        Model with loaded weights
    """
    logger.info(f"Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    logger.info(f"Validation loss: {checkpoint['val_loss']:.4f}")

    return model


def main():
    """Main training function."""
    # Set random seeds for reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.RANDOM_SEED)

    logger.info("Initializing training pipeline...")

    # Ensure directories exist
    config.create_directories()

    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        batch_size=config.BATCH_SIZE,
        num_workers=4
    )

    # Create model
    logger.info("Creating model...")
    model = create_model(device=config.DEVICE)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config.DEVICE,
        learning_rate=config.LEARNING_RATE,
        num_epochs=config.NUM_EPOCHS,
        early_stopping_patience=config.EARLY_STOPPING_PATIENCE
    )

    # Train model
    history = trainer.train()

    logger.info("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
