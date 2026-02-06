"""
DeepLog LSTM Model for log anomaly detection.

Architecture (Paper Specification):
- Embedding layer: input_dim=29, output_dim=64
- 2-layer LSTM: hidden_size=64
- Fully connected output: 29 classes

Paper parameters: L=2 (layers), α=64 (hidden units), n=29 (log keys)
"""

import torch
import torch.nn as nn
import logging

import config

# Configure logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class DeepLogLSTM(nn.Module):
    """
    DeepLog LSTM model for log key sequence prediction.

    Paper architecture:
    - Input: sequence of h log keys (window_size=10)
    - Embedding: map each log key to 64-dimensional vector
    - LSTM: 2 layers with 64 hidden units each
    - Output: probability distribution over n=29 log keys
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        embedding_dim: int = config.EMBEDDING_DIM,
        hidden_size: int = config.HIDDEN_SIZE,
        num_layers: int = config.NUM_LAYERS,
        dropout: float = 0.0
    ):
        """
        Initialize DeepLog LSTM model.

        Args:
            num_classes: Number of distinct log keys (n=29 for HDFS)
            embedding_dim: Dimension of embedding vectors (64)
            hidden_size: Number of hidden units in LSTM (α=64)
            num_layers: Number of LSTM layers (L=2)
            dropout: Dropout rate (0.0 for paper reproduction)
        """
        super(DeepLogLSTM, self).__init__()

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding layer: map log key index to dense vector
        self.embedding = nn.Embedding(
            num_embeddings=num_classes,
            embedding_dim=embedding_dim
        )

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, num_classes)

        logger.info(f"Initialized DeepLog model with:")
        logger.info(f"  - Num classes: {num_classes}")
        logger.info(f"  - Embedding dim: {embedding_dim}")
        logger.info(f"  - Hidden size: {hidden_size}")
        logger.info(f"  - Num layers: {num_layers}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, window_size)
               containing log key indices

        Returns:
            Output tensor of shape (batch_size, num_classes)
            representing probability distribution over next log key
        """
        # x shape: (batch_size, window_size)

        # Embedding layer
        # Output shape: (batch_size, window_size, embedding_dim)
        embedded = self.embedding(x)

        # LSTM layer
        # Output shape: (batch_size, window_size, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(embedded)

        # Take the output from the last time step
        # Shape: (batch_size, hidden_size)
        last_hidden = lstm_out[:, -1, :]

        # Fully connected layer
        # Output shape: (batch_size, num_classes)
        output = self.fc(last_hidden)

        return output

    def predict_top_k(self, x: torch.Tensor, k: int = config.TOP_G) -> torch.Tensor:
        """
        Predict top-k most probable log keys.

        Args:
            x: Input tensor of shape (batch_size, window_size)
            k: Number of top predictions to return (default: g=9)

        Returns:
            Tensor of shape (batch_size, k) containing indices of top-k predictions
        """
        # Get logits
        logits = self.forward(x)

        # Get top-k predictions
        # Returns values and indices
        _, top_k_indices = torch.topk(logits, k, dim=1)

        return top_k_indices

    def get_num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(device: torch.device = config.DEVICE) -> DeepLogLSTM:
    """
    Create and initialize DeepLog model.

    Args:
        device: Device to place the model on (CPU or CUDA)

    Returns:
        Initialized DeepLog model
    """
    model = DeepLogLSTM(
        num_classes=config.NUM_CLASSES,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS
    )

    model = model.to(device)

    num_params = model.get_num_parameters()
    logger.info(f"Model created with {num_params:,} trainable parameters")
    logger.info(f"Model moved to device: {device}")

    return model


def test_model():
    """Test model forward pass."""
    logger.info("Testing model forward pass...")

    # Create model
    model = create_model()

    # Create dummy input: batch_size=4, window_size=10
    batch_size = 4
    window_size = config.WINDOW_SIZE
    dummy_input = torch.randint(0, config.NUM_CLASSES, (batch_size, window_size))

    logger.info(f"Input shape: {dummy_input.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    logger.info(f"Output shape: {output.shape}")
    logger.info(f"Expected shape: (batch_size={batch_size}, num_classes={config.NUM_CLASSES})")

    # Test top-k prediction
    top_k = model.predict_top_k(dummy_input, k=config.TOP_G)
    logger.info(f"Top-{config.TOP_G} predictions shape: {top_k.shape}")
    logger.info(f"Top-{config.TOP_G} predictions:\n{top_k}")

    # Apply softmax to get probabilities
    probs = torch.softmax(output, dim=1)
    logger.info(f"Probability distribution shape: {probs.shape}")
    logger.info(f"Sum of probabilities (should be ~1.0): {probs[0].sum().item():.4f}")

    logger.info("\nModel test completed successfully!")


def print_model_architecture():
    """Print detailed model architecture."""
    model = create_model()

    print("\n" + "=" * 80)
    print("DeepLog LSTM Model Architecture")
    print("=" * 80)
    print(model)
    print("\n" + "=" * 80)
    print("Model Parameters")
    print("=" * 80)

    for name, param in model.named_parameters():
        print(f"{name:30s} | Shape: {str(list(param.shape)):20s} | Params: {param.numel():,}")

    print("\n" + "=" * 80)
    print(f"Total trainable parameters: {model.get_num_parameters():,}")
    print("=" * 80)


if __name__ == "__main__":
    test_model()
    print_model_architecture()
