"""
Data preprocessing module for SCALE-Sim ML Predictor.
Handles feature engineering, encoding, and normalization.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from ml_predictor.config import (
    DATA_GENERATION_CONFIG,
    MODEL_CONFIG,
    INPUT_FEATURES,
    OUTPUT_TARGETS,
)


class DataPreprocessor:
    """
    Preprocesses data for training the SCALE-Sim prediction model.
    Handles feature engineering, encoding, and normalization.
    """

    def __init__(self, scaler_type: str = "standard"):
        """
        Initialize the preprocessor.

        Args:
            scaler_type: Type of scaler to use ('standard' or 'minmax').
        """
        self.scaler_type = scaler_type
        self.feature_scaler = (
            StandardScaler() if scaler_type == "standard" else MinMaxScaler()
        )
        self.target_scaler = StandardScaler()
        self.is_fitted = False

        # Feature columns
        self.feature_columns = []
        self.target_columns = OUTPUT_TARGETS

    def _one_hot_encode_dataflow(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        One-hot encode the dataflow column.

        Args:
            df: Input DataFrame with 'dataflow' column.

        Returns:
            DataFrame with one-hot encoded dataflow columns.
        """
        df = df.copy()

        # Create one-hot columns
        df["dataflow_os"] = (df["dataflow"] == "os").astype(int)
        df["dataflow_ws"] = (df["dataflow"] == "ws").astype(int)
        df["dataflow_is"] = (df["dataflow"] == "is").astype(int)

        # Drop original column
        df = df.drop(columns=["dataflow"])

        return df

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features if not already present.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with derived features.
        """
        df = df.copy()

        # Calculate output feature map dimensions if not present
        if "ofmap_height" not in df.columns:
            df["ofmap_height"] = (
                (df["ifmap_height"] - df["filter_height"]) // df["strides"] + 1
            ).astype(int)
        if "ofmap_width" not in df.columns:
            df["ofmap_width"] = (
                (df["ifmap_width"] - df["filter_width"]) // df["strides"] + 1
            ).astype(int)

        # Calculate derived features if not present
        if "total_macs" not in df.columns:
            df["total_macs"] = (
                df["ofmap_height"]
                * df["ofmap_width"]
                * df["filter_height"]
                * df["filter_width"]
                * df["channels"]
                * df["num_filter"]
            )

        if "ifmap_size" not in df.columns:
            df["ifmap_size"] = df["ifmap_height"] * df["ifmap_width"] * df["channels"]

        if "filter_size" not in df.columns:
            df["filter_size"] = (
                df["filter_height"]
                * df["filter_width"]
                * df["channels"]
                * df["num_filter"]
            )

        if "ofmap_size" not in df.columns:
            df["ofmap_size"] = df["ofmap_height"] * df["ofmap_width"] * df["num_filter"]

        if "compute_intensity" not in df.columns:
            total_data = df["ifmap_size"] + df["filter_size"] + df["ofmap_size"]
            df["compute_intensity"] = df["total_macs"] / total_data.replace(0, 1)

        return df

    def _apply_log_transform(
        self, df: pd.DataFrame, columns: List[str]
    ) -> pd.DataFrame:
        """
        Apply log transform to specified columns for better scaling.

        Args:
            df: Input DataFrame.
            columns: Columns to transform.

        Returns:
            DataFrame with transformed columns.
        """
        df = df.copy()
        for col in columns:
            if col in df.columns:
                # Use log1p to handle zeros
                df[col] = np.log1p(df[col])
        return df

    def preprocess(
        self, df: pd.DataFrame, fit: bool = True, log_transform_targets: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the data for model training/inference.

        Args:
            df: Input DataFrame with raw data.
            fit: Whether to fit the scalers (True for training, False for inference).
            log_transform_targets: Whether to apply log transform to target columns.

        Returns:
            Tuple of (X, y) numpy arrays.
        """
        df = df.copy()

        # One-hot encode dataflow
        if "dataflow" in df.columns:
            df = self._one_hot_encode_dataflow(df)

        # Add derived features
        df = self._add_derived_features(df)

        # Define feature columns
        self.feature_columns = [
            "array_height",
            "array_width",
            "ifmap_sram_sz_kb",
            "filter_sram_sz_kb",
            "ofmap_sram_sz_kb",
            "dataflow_os",
            "dataflow_ws",
            "dataflow_is",
            "bandwidth",
            "ifmap_height",
            "ifmap_width",
            "filter_height",
            "filter_width",
            "channels",
            "num_filter",
            "strides",
            "total_macs",
            "ifmap_size",
            "filter_size",
            "ofmap_size",
            "compute_intensity",
        ]

        # Select only available columns
        available_feature_cols = [
            col for col in self.feature_columns if col in df.columns
        ]
        available_target_cols = [
            col for col in self.target_columns if col in df.columns
        ]

        # Extract features and targets
        X = df[available_feature_cols].values.astype(np.float32)
        y = (
            df[available_target_cols].values.astype(np.float32)
            if available_target_cols
            else None
        )

        # Apply log transform to large values (MACs, sizes, cycles)
        log_feature_indices = [
            available_feature_cols.index(col)
            for col in ["total_macs", "ifmap_size", "filter_size", "ofmap_size"]
            if col in available_feature_cols
        ]
        for idx in log_feature_indices:
            X[:, idx] = np.log1p(X[:, idx])

        if y is not None and log_transform_targets:
            # Log transform cycles (large values)
            cycle_targets = [
                "total_cycles_with_prefetch",
                "total_cycles",
                "stall_cycles",
            ]
            cycle_indices = [
                available_target_cols.index(col)
                for col in cycle_targets
                if col in available_target_cols
            ]
            for idx in cycle_indices:
                y[:, idx] = np.log1p(y[:, idx])

        # Scale features
        if fit:
            X = self.feature_scaler.fit_transform(X)
            if y is not None:
                y = self.target_scaler.fit_transform(y)
            self.is_fitted = True
            self.feature_columns = available_feature_cols
            # Update target columns to match what was actually fitted (if any were filtered out)
            if available_target_cols:
                self.target_columns = available_target_cols
        else:
            if not self.is_fitted:
                raise ValueError("Preprocessor must be fitted before transform")
            X = self.feature_scaler.transform(X)
            if y is not None:
                y = self.target_scaler.transform(y)

        return X, y

    def inverse_transform_targets(
        self, y: np.ndarray, log_transformed: bool = True
    ) -> np.ndarray:
        """
        Inverse transform the target values back to original scale.

        Args:
            y: Scaled target values.
            log_transformed: Whether log transform was applied.

        Returns:
            Original scale target values.
        """
        y_inv = self.target_scaler.inverse_transform(y)

        if log_transformed:
            # Inverse log transform for cycles
            cycle_targets = [
                "total_cycles_with_prefetch",
                "total_cycles",
                "stall_cycles",
            ]
            cycle_indices = [
                i for i, col in enumerate(self.target_columns) if col in cycle_targets
            ]

            for idx in cycle_indices:
                if idx < y_inv.shape[1]:
                    # Clamp values to prevent overflow in expm1
                    # logs of total cycles typically won't exceed 20 (e^20 ~= 4.8e8)
                    y_inv[:, idx] = np.clip(y_inv[:, idx], a_min=None, a_max=50)
                    y_inv[:, idx] = np.expm1(y_inv[:, idx])
                    # Ensure non-negative
                    y_inv[:, idx] = np.maximum(y_inv[:, idx], 0)

        return y_inv

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split_ratio: Optional[List[float]] = None,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, ...]:
        """
        Split data into train/validation/test sets.

        Args:
            X: Feature array.
            y: Target array.
            split_ratio: List of [train, val, test] ratios.
            random_state: Random seed.

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
        """
        split_ratio = split_ratio or MODEL_CONFIG["train_val_test_split"]

        # First split: train + val vs test
        test_size = split_ratio[2]
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Second split: train vs val
        val_size = split_ratio[1] / (split_ratio[0] + split_ratio[1])
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def save(self, path: str):
        """Save the fitted preprocessor to disk."""
        save_dict = {
            "feature_scaler": self.feature_scaler,
            "target_scaler": self.target_scaler,
            "feature_columns": self.feature_columns,
            "target_columns": self.target_columns,
            "scaler_type": self.scaler_type,
            "is_fitted": self.is_fitted,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(save_dict, f)

    @classmethod
    def load(cls, path: str) -> "DataPreprocessor":
        """Load a fitted preprocessor from disk."""
        with open(path, "rb") as f:
            save_dict = pickle.load(f)

        preprocessor = cls(scaler_type=save_dict["scaler_type"])
        preprocessor.feature_scaler = save_dict["feature_scaler"]
        preprocessor.target_scaler = save_dict["target_scaler"]
        preprocessor.feature_columns = save_dict["feature_columns"]
        preprocessor.target_columns = save_dict["target_columns"]
        preprocessor.is_fitted = save_dict["is_fitted"]

        return preprocessor


def load_and_preprocess(
    data_path: str, preprocessor: Optional[DataPreprocessor] = None, fit: bool = True
) -> Tuple[DataPreprocessor, np.ndarray, np.ndarray]:
    """
    Load data from CSV and preprocess it.

    Args:
        data_path: Path to the CSV file.
        preprocessor: Existing preprocessor to use. If None, creates a new one.
        fit: Whether to fit the preprocessor.

    Returns:
        Tuple of (preprocessor, X, y).
    """
    df = pd.read_csv(data_path)
    preprocessor = preprocessor or DataPreprocessor()
    X, y = preprocessor.preprocess(df, fit=fit)
    return preprocessor, X, y
