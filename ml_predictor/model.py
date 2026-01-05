"""
Neural network model definition for SCALE-Sim performance prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from ml_predictor.config import MODEL_CONFIG, OUTPUT_TARGETS


class ScaleSimPredictor(nn.Module):
    """
    Multi-layer perceptron for predicting SCALE-Sim simulation outputs.
    Uses multi-task learning to predict multiple metrics simultaneously.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 5,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.2,
    ):
        """
        Initialize the model.

        Args:
            input_dim: Number of input features.
            output_dim: Number of output targets (default: 5 for COMPUTE_REPORT metrics).
            hidden_dims: List of hidden layer dimensions.
            dropout_rate: Dropout rate for regularization.
        """
        super(ScaleSimPredictor, self).__init__()

        hidden_dims = hidden_dims or MODEL_CONFIG["hidden_dims"]

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        return self.network(x)


class ScaleSimPredictorWithUncertainty(nn.Module):
    """
    Extended model that also predicts uncertainty (useful for active learning).
    Uses separate heads for mean and variance predictions.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 5,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.2,
    ):
        """
        Initialize the model with uncertainty estimation.

        Args:
            input_dim: Number of input features.
            output_dim: Number of output targets.
            hidden_dims: List of hidden layer dimensions.
            dropout_rate: Dropout rate for regularization.
        """
        super(ScaleSimPredictorWithUncertainty, self).__init__()

        hidden_dims = hidden_dims or MODEL_CONFIG["hidden_dims"]

        # Shared backbone
        backbone_layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims[:-1]:  # All but last layer
            backbone_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*backbone_layers)

        # Mean prediction head
        self.mean_head = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], output_dim),
        )

        # Log-variance prediction head (for numerical stability)
        self.logvar_head = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Tuple of (mean, logvar) tensors, each of shape (batch_size, output_dim).
        """
        features = self.backbone(x)
        mean = self.mean_head(features)
        logvar = self.logvar_head(features)
        return mean, logvar


def gaussian_nll_loss(
    mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """
    Gaussian negative log-likelihood loss for uncertainty-aware training.

    Args:
        mean: Predicted mean.
        logvar: Predicted log-variance.
        target: Ground truth target.

    Returns:
        Loss value.
    """
    var = torch.exp(logvar)
    loss = 0.5 * (logvar + (target - mean) ** 2 / var)
    return loss.mean()


def create_model(
    input_dim: int,
    output_dim: int = 5,
    with_uncertainty: bool = False,
    device: str = "cpu",
) -> nn.Module:
    """
    Factory function to create a model.

    Args:
        input_dim: Number of input features.
        output_dim: Number of output targets.
        with_uncertainty: Whether to include uncertainty estimation.
        device: Device to place the model on.

    Returns:
        Model instance.
    """
    if with_uncertainty:
        model = ScaleSimPredictorWithUncertainty(input_dim, output_dim)
    else:
        model = ScaleSimPredictor(input_dim, output_dim)

    return model.to(device)
