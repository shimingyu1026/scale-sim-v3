"""
Inference module for SCALE-Sim ML Predictor.
Provides functions to load trained models and make predictions.
"""

import os
import argparse
import configparser
import csv
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

from ml_predictor.config import MODEL_CONFIG, OUTPUT_TARGETS
from ml_predictor.data_preprocessing import DataPreprocessor
from ml_predictor.model import ScaleSimPredictor, create_model


class Predictor:
    """
    Predictor class for making predictions with trained SCALE-Sim model.
    """

    def __init__(
        self, model_path: str, preprocessor_path: str, device: Optional[str] = None
    ):
        """
        Initialize the predictor.

        Args:
            model_path: Path to trained model file.
            preprocessor_path: Path to fitted preprocessor file.
            device: Inference device ('cpu' or 'cuda').
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load preprocessor
        self.preprocessor = DataPreprocessor.load(preprocessor_path)

        # Load model
        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False
        )
        input_dim = len(self.preprocessor.feature_columns)
        output_dim = len(OUTPUT_TARGETS)

        self.model = create_model(input_dim, output_dim, device=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        print(f"Loaded model from: {model_path}")
        print(f"Loaded preprocessor from: {preprocessor_path}")
        print(f"Using device: {self.device}")

    def _parse_config_file(self, config_path: str) -> Dict:
        """
        Parse a SCALE-Sim config file.

        Args:
            config_path: Path to config file.

        Returns:
            Dictionary of configuration values.
        """
        parser = configparser.ConfigParser()
        parser.read(config_path)

        config = {
            "array_height": int(parser.get("architecture_presets", "ArrayHeight")),
            "array_width": int(parser.get("architecture_presets", "ArrayWidth")),
            "ifmap_sram_sz_kb": int(
                parser.get("architecture_presets", "IfmapSramSzkB")
            ),
            "filter_sram_sz_kb": int(
                parser.get("architecture_presets", "FilterSramSzkB")
            ),
            "ofmap_sram_sz_kb": int(
                parser.get("architecture_presets", "OfmapSramSzkB")
            ),
            "dataflow": parser.get("architecture_presets", "Dataflow").strip(),
            "bandwidth": int(parser.get("architecture_presets", "Bandwidth")),
        }

        return config

    def _parse_topology_file(self, topology_path: str) -> List[Dict]:
        """
        Parse a SCALE-Sim topology file.

        Args:
            topology_path: Path to topology file.

        Returns:
            List of layer dictionaries.
        """
        layers = []

        with open(topology_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header

            for row in reader:
                if len(row) < 8 or not row[0].strip():
                    continue

                layer = {
                    "layer_name": row[0].strip(),
                    "ifmap_height": int(row[1].strip()),
                    "ifmap_width": int(row[2].strip()),
                    "filter_height": int(row[3].strip()),
                    "filter_width": int(row[4].strip()),
                    "channels": int(row[5].strip()),
                    "num_filter": int(row[6].strip()),
                    "strides": int(row[7].strip().rstrip(",")),
                }
                layers.append(layer)

        return layers

    def _prepare_input(self, hw_config: Dict, layer: Dict) -> pd.DataFrame:
        """
        Prepare input DataFrame for prediction.

        Args:
            hw_config: Hardware configuration dictionary.
            layer: Layer configuration dictionary.

        Returns:
            DataFrame ready for preprocessing.
        """
        # Combine config and layer parameters
        input_data = {**hw_config, **layer}

        # Calculate derived features
        ofmap_height = (layer["ifmap_height"] - layer["filter_height"]) // layer[
            "strides"
        ] + 1
        ofmap_width = (layer["ifmap_width"] - layer["filter_width"]) // layer[
            "strides"
        ] + 1

        input_data["ofmap_height"] = ofmap_height
        input_data["ofmap_width"] = ofmap_width

        total_macs = (
            ofmap_height
            * ofmap_width
            * layer["filter_height"]
            * layer["filter_width"]
            * layer["channels"]
            * layer["num_filter"]
        )
        ifmap_size = layer["ifmap_height"] * layer["ifmap_width"] * layer["channels"]
        filter_size = (
            layer["filter_height"]
            * layer["filter_width"]
            * layer["channels"]
            * layer["num_filter"]
        )
        ofmap_size = ofmap_height * ofmap_width * layer["num_filter"]

        input_data["total_macs"] = total_macs
        input_data["ifmap_size"] = ifmap_size
        input_data["filter_size"] = filter_size
        input_data["ofmap_size"] = ofmap_size
        input_data["compute_intensity"] = (
            total_macs / (ifmap_size + filter_size + ofmap_size)
            if (ifmap_size + filter_size + ofmap_size) > 0
            else 0
        )

        return pd.DataFrame([input_data])

    def predict_layer(self, hw_config: Dict, layer: Dict) -> Dict[str, float]:
        """
        Predict performance metrics for a single layer.

        Args:
            hw_config: Hardware configuration dictionary.
            layer: Layer configuration dictionary.

        Returns:
            Dictionary of predicted metrics.
        """
        # Prepare input
        df = self._prepare_input(hw_config, layer)

        # Preprocess (without fitting)
        X, _ = self.preprocessor.preprocess(df, fit=False)

        # Make prediction
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            prediction = self.model(X_tensor).cpu().numpy()

        # Inverse transform
        prediction_original = self.preprocessor.inverse_transform_targets(prediction)

        # Create result dictionary
        result = {}
        for i, target_name in enumerate(OUTPUT_TARGETS):
            result[target_name] = float(prediction_original[0, i])

        return result

    def predict_from_files(
        self, config_path: str, topology_path: str, output_path: Optional[str] = None
    ) -> List[Dict]:
        """
        Predict performance metrics from config and topology files.

        Args:
            config_path: Path to config file.
            topology_path: Path to topology file.
            output_path: Optional path to save results CSV.

        Returns:
            List of prediction dictionaries (one per layer).
        """
        # Parse files
        hw_config = self._parse_config_file(config_path)
        layers = self._parse_topology_file(topology_path)

        print(f"Config: {hw_config}")
        print(f"Found {len(layers)} layers in topology")

        # Predict for each layer
        results = []
        for i, layer in enumerate(layers):
            prediction = self.predict_layer(hw_config, layer)
            prediction["layer_id"] = i
            prediction["layer_name"] = layer["layer_name"]
            results.append(prediction)

            print(f"\nLayer {i} ({layer['layer_name']}):")
            for key, value in prediction.items():
                if key not in ["layer_id", "layer_name"]:
                    print(f"  {key}: {value:.4f}")

        # Save to CSV if requested
        if output_path:
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)
            print(f"\nResults saved to: {output_path}")

        return results

    def predict_batch(self, hw_configs: List[Dict], layers: List[Dict]) -> np.ndarray:
        """
        Make batch predictions.

        Args:
            hw_configs: List of hardware configuration dictionaries.
            layers: List of layer configuration dictionaries.

        Returns:
            Numpy array of predictions with shape (n_samples, n_outputs).
        """
        # Prepare all inputs
        dfs = []
        for hw_config, layer in zip(hw_configs, layers):
            dfs.append(self._prepare_input(hw_config, layer))

        combined_df = pd.concat(dfs, ignore_index=True)

        # Preprocess
        X, _ = self.preprocessor.preprocess(combined_df, fit=False)

        # Predict
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()

        # Inverse transform
        predictions_original = self.preprocessor.inverse_transform_targets(predictions)

        return predictions_original


def predict(
    config_path: str,
    topology_path: str,
    model_path: Optional[str] = None,
    preprocessor_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> List[Dict]:
    """
    Main function to make predictions.

    Args:
        config_path: Path to config file.
        topology_path: Path to topology file.
        model_path: Path to trained model.
        preprocessor_path: Path to preprocessor.
        output_path: Optional path to save results.

    Returns:
        List of prediction dictionaries.
    """
    model_path = model_path or MODEL_CONFIG["model_save_path"]
    preprocessor_path = preprocessor_path or MODEL_CONFIG["scaler_save_path"]

    predictor = Predictor(model_path, preprocessor_path)
    return predictor.predict_from_files(config_path, topology_path, output_path)


def main():
    """Main entry point for inference."""
    parser = argparse.ArgumentParser(description="SCALE-Sim ML Predictor Inference")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to hardware config file"
    )
    parser.add_argument(
        "--topology", type=str, required=True, help="Path to topology file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./models/scalesim_predictor.pt",
        help="Path to trained model",
    )
    parser.add_argument(
        "--preprocessor",
        type=str,
        default="./models/preprocessor.pkl",
        help="Path to preprocessor",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Path to save results CSV"
    )

    args = parser.parse_args()

    predict(
        config_path=args.config,
        topology_path=args.topology,
        model_path=args.model,
        preprocessor_path=args.preprocessor,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
