"""
Training script for SCALE-Sim ML Predictor.
Handles data loading, model training, validation, and saving.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

from ml_predictor.config import MODEL_CONFIG, OUTPUT_TARGETS
from ml_predictor.data_preprocessing import DataPreprocessor, load_and_preprocess
from ml_predictor.model import ScaleSimPredictor, create_model


class Trainer:
    """
    Trainer class for SCALE-Sim prediction model.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        learning_rate: float = 0.001,
        batch_size: int = 64,
        epochs: int = 100,
        early_stopping_patience: int = 10,
    ):
        """
        Initialize the trainer.

        Args:
            model: PyTorch model to train.
            device: Device to train on ('cpu' or 'cuda').
            learning_rate: Learning rate for optimizer.
            batch_size: Training batch size.
            epochs: Maximum number of training epochs.
            early_stopping_patience: Number of epochs without improvement before stopping.
        """
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience

        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        # Training history
        self.history = {"train_loss": [], "val_loss": [], "learning_rate": []}

        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.epochs_without_improvement = 0

    def _create_dataloader(
        self, X: np.ndarray, y: np.ndarray, shuffle: bool = True
    ) -> DataLoader:
        """Create a PyTorch DataLoader from numpy arrays."""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch and return average loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate(self, val_loader: DataLoader) -> float:
        """Validate and return average loss."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        verbose: bool = True,
    ) -> Dict:
        """
        Train the model.

        Args:
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features.
            y_val: Validation targets.
            verbose: Whether to print progress.

        Returns:
            Training history dictionary.
        """
        train_loader = self._create_dataloader(X_train, y_train, shuffle=True)
        val_loader = self._create_dataloader(X_val, y_val, shuffle=False)

        for epoch in range(self.epochs):
            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_loss = self.validate(val_loader)

            # Update learning rate scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["learning_rate"].append(current_lr)

            if verbose:
                print(
                    f"Epoch {epoch+1}/{self.epochs} - "
                    f"Train Loss: {train_loss:.6f} - "
                    f"Val Loss: {val_loss:.6f} - "
                    f"LR: {current_lr:.6f}"
                )

            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                self.epochs_without_improvement = 0
                if verbose:
                    print(f"  -> New best model (val_loss: {val_loss:.6f})")
            else:
                self.epochs_without_improvement += 1

            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return self.history

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray, preprocessor: DataPreprocessor
    ) -> Dict:
        """
        Evaluate the model on test data.

        Args:
            X_test: Test features (scaled).
            y_test: Test targets (scaled).
            preprocessor: Preprocessor for inverse transformation.

        Returns:
            Dictionary of evaluation metrics.
        """
        self.model.eval()

        X_tensor = torch.FloatTensor(X_test).to(self.device)

        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()

        # Inverse transform to original scale
        y_pred_original = preprocessor.inverse_transform_targets(predictions)
        y_true_original = preprocessor.inverse_transform_targets(y_test)

        # Calculate metrics for each target
        metrics = {}
        for i, target_name in enumerate(OUTPUT_TARGETS):
            pred = y_pred_original[:, i]
            true = y_true_original[:, i]

            # Mean Absolute Error
            mae = np.mean(np.abs(pred - true))

            # Mean Absolute Percentage Error (avoid division by zero)
            mask = true != 0
            mape = np.mean(np.abs((pred[mask] - true[mask]) / true[mask])) * 100

            # Root Mean Squared Error
            rmse = np.sqrt(np.mean((pred - true) ** 2))

            # R-squared
            ss_res = np.sum((true - pred) ** 2)
            ss_tot = np.sum((true - np.mean(true)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            metrics[target_name] = {"MAE": mae, "MAPE": mape, "RMSE": rmse, "R2": r2}

        # Overall metrics
        mse = np.mean((y_pred_original - y_true_original) ** 2)
        metrics["overall_mse"] = mse

        return metrics

    def save_model(
        self,
        model_path: str,
        preprocessor_path: str,
        preprocessor: DataPreprocessor,
        additional_info: Optional[Dict] = None,
    ):
        """
        Save the trained model and preprocessor.

        Args:
            model_path: Path to save the model.
            preprocessor_path: Path to save the preprocessor.
            preprocessor: Fitted preprocessor.
            additional_info: Additional information to save.
        """
        # Create directories
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        Path(preprocessor_path).parent.mkdir(parents=True, exist_ok=True)

        # Save model
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "input_dim": next(self.model.parameters()).shape[1],
            "output_dim": OUTPUT_TARGETS,
            "history": self.history,
            "best_val_loss": self.best_val_loss,
            "timestamp": datetime.now().isoformat(),
        }
        if additional_info:
            save_dict.update(additional_info)

        torch.save(save_dict, model_path)
        print(f"Model saved to: {model_path}")

        # Save preprocessor
        preprocessor.save(preprocessor_path)
        print(f"Preprocessor saved to: {preprocessor_path}")


def train_model(
    data_path: str,
    model_save_path: Optional[str] = None,
    preprocessor_save_path: Optional[str] = None,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, DataPreprocessor, Dict]:
    """
    Main function to train a model from data.

    Args:
        data_path: Path to training data CSV.
        model_save_path: Path to save trained model.
        preprocessor_save_path: Path to save preprocessor.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Learning rate.
        device: Training device ('cpu' or 'cuda').
        verbose: Whether to print progress.

    Returns:
        Tuple of (trained_model, preprocessor, metrics).
    """
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load and preprocess data
    print(f"Loading data from: {data_path}")
    preprocessor, X, y = load_and_preprocess(data_path)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
    print(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Create model
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    model = create_model(input_dim, output_dim, device=device)
    print(f"Model created: input_dim={input_dim}, output_dim={output_dim}")

    # Create trainer and train
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        early_stopping_patience=MODEL_CONFIG["early_stopping_patience"],
    )

    print("Starting training...")
    history = trainer.train(X_train, y_train, X_val, y_val, verbose=verbose)

    # Evaluate
    print("\nEvaluating on test set...")
    metrics = trainer.evaluate(X_test, y_test, preprocessor)

    # Print metrics
    print("\n" + "=" * 60)
    print("Test Set Evaluation Results:")
    print("=" * 60)
    for target_name, target_metrics in metrics.items():
        if isinstance(target_metrics, dict):
            print(f"\n{target_name}:")
            for metric_name, value in target_metrics.items():
                print(f"  {metric_name}: {value:.4f}")
    print("=" * 60)

    # Save model and preprocessor
    model_save_path = model_save_path or MODEL_CONFIG["model_save_path"]
    preprocessor_save_path = preprocessor_save_path or MODEL_CONFIG["scaler_save_path"]

    trainer.save_model(
        model_path=model_save_path,
        preprocessor_path=preprocessor_save_path,
        preprocessor=preprocessor,
        additional_info={"test_metrics": metrics},
    )

    return model, preprocessor, metrics


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Train SCALE-Sim ML Predictor")
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to training data CSV"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/scalesim_predictor.pt",
        help="Path to save trained model",
    )
    parser.add_argument(
        "--preprocessor_path",
        type=str,
        default="./models/preprocessor.pkl",
        help="Path to save preprocessor",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Training batch size"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--device", type=str, default=None, help="Training device ('cpu' or 'cuda')"
    )

    args = parser.parse_args()

    train_model(
        data_path=args.data_path,
        model_save_path=args.model_path,
        preprocessor_save_path=args.preprocessor_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()
