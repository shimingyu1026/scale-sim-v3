"""
Evaluation module for SCALE-Sim ML Predictor.
Provides functions to evaluate model performance on test data.
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Optional
import json

from ml_predictor.config import MODEL_CONFIG, OUTPUT_TARGETS
from ml_predictor.data_preprocessing import DataPreprocessor, load_and_preprocess
from ml_predictor.model import create_model


def evaluate_model(
    data_path: str,
    model_path: Optional[str] = None,
    preprocessor_path: Optional[str] = None,
    output_path: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict:
    """
    Evaluate trained model on test data.

    Args:
        data_path: Path to test data CSV.
        model_path: Path to trained model.
        preprocessor_path: Path to preprocessor.
        output_path: Optional path to save results.
        device: Inference device.

    Returns:
        Dictionary of evaluation metrics.
    """
    model_path = model_path or MODEL_CONFIG["model_save_path"]
    preprocessor_path = preprocessor_path or MODEL_CONFIG["scaler_save_path"]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from: {model_path}")
    print(f"Loading preprocessor from: {preprocessor_path}")
    print(f"Loading test data from: {data_path}")

    # Load preprocessor
    preprocessor = DataPreprocessor.load(preprocessor_path)

    # Load and preprocess test data (without re-fitting)
    df = pd.read_csv(data_path)
    X, y = preprocessor.preprocess(df, fit=False)

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    input_dim = len(preprocessor.feature_columns)
    output_dim = len(OUTPUT_TARGETS)

    model = create_model(input_dim, output_dim, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Make predictions
    X_tensor = torch.FloatTensor(X).to(device)

    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()

    # Inverse transform to original scale
    y_pred_original = preprocessor.inverse_transform_targets(predictions)
    y_true_original = preprocessor.inverse_transform_targets(y)

    # Calculate metrics for each target
    metrics = {"num_samples": len(X), "targets": {}}

    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)

    for i, target_name in enumerate(OUTPUT_TARGETS):
        pred = y_pred_original[:, i]
        true = y_true_original[:, i]

        # Mean Absolute Error
        mae = np.mean(np.abs(pred - true))

        # Mean Absolute Percentage Error
        mask = true != 0
        mape = np.mean(np.abs((pred[mask] - true[mask]) / true[mask])) * 100

        # Root Mean Squared Error
        rmse = np.sqrt(np.mean((pred - true) ** 2))

        # R-squared
        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - np.mean(true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Max error
        max_error = np.max(np.abs(pred - true))

        metrics["targets"][target_name] = {
            "MAE": float(mae),
            "MAPE": float(mape),
            "RMSE": float(rmse),
            "R2": float(r2),
            "MaxError": float(max_error),
        }

        print(f"\n{target_name}:")
        print(f"  MAE:       {mae:.4f}")
        print(f"  MAPE:      {mape:.2f}%")
        print(f"  RMSE:      {rmse:.4f}")
        print(f"  R²:        {r2:.4f}")
        print(f"  Max Error: {max_error:.4f}")

    # Overall MSE
    overall_mse = np.mean((y_pred_original - y_true_original) ** 2)
    metrics["overall_mse"] = float(overall_mse)

    print("\n" + "-" * 70)
    print(f"Overall MSE: {overall_mse:.6f}")
    print("=" * 70)

    # Save results if requested
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if output_path.endswith(".json"):
            with open(output_path, "w") as f:
                json.dump(metrics, f, indent=2)
        else:
            # Save as CSV with predictions vs ground truth
            results_df = pd.DataFrame(
                {
                    f"{target}_pred": y_pred_original[:, i]
                    for i, target in enumerate(OUTPUT_TARGETS)
                }
            )
            for i, target in enumerate(OUTPUT_TARGETS):
                results_df[f"{target}_true"] = y_true_original[:, i]
            results_df.to_csv(output_path, index=False)

        print(f"\nResults saved to: {output_path}")

    return metrics


def compare_with_simulation(
    config_path: str,
    topology_path: str,
    model_path: Optional[str] = None,
    preprocessor_path: Optional[str] = None,
) -> Dict:
    """
    Compare model predictions with actual SCALE-Sim simulation.

    Args:
        config_path: Path to config file.
        topology_path: Path to topology file.
        model_path: Path to trained model.
        preprocessor_path: Path to preprocessor.

    Returns:
        Comparison results.
    """
    import sys
    import tempfile
    import shutil
    from pathlib import Path

    # Add parent to path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from scalesim.scale_sim import scalesim
    from ml_predictor.predict import Predictor

    model_path = model_path or MODEL_CONFIG["model_save_path"]
    preprocessor_path = preprocessor_path or MODEL_CONFIG["scaler_save_path"]

    # Get ML predictions
    predictor = Predictor(model_path, preprocessor_path)
    ml_results = predictor.predict_from_files(config_path, topology_path)

    # Run actual simulation
    temp_dir = Path(tempfile.mkdtemp())
    try:
        sim = scalesim(
            save_disk_space=True,
            verbose=False,
            config=config_path,
            topology=topology_path,
            input_type_gemm=False,
        )
        sim.run_scale(top_path=str(temp_dir))

        # Find and parse results
        compute_report = None
        for subdir in temp_dir.iterdir():
            if subdir.is_dir():
                report_path = subdir / "COMPUTE_REPORT.csv"
                if report_path.exists():
                    compute_report = pd.read_csv(report_path)
                    break

        if compute_report is None:
            print("Warning: Could not find simulation results")
            return {"ml_predictions": ml_results, "simulation_results": None}

        # Compare
        print("\n" + "=" * 80)
        print("Comparison: ML Predictions vs Simulation")
        print("=" * 80)

        comparison = []
        for i, ml_result in enumerate(ml_results):
            if i < len(compute_report):
                sim_row = compute_report.iloc[i]

                comp = {"layer": ml_result["layer_name"], "metrics": {}}

                metric_map = {
                    "total_cycles": "Total Cycles",
                    "stall_cycles": "Stall Cycles",
                    "overall_util_percent": "Overall Util %",
                    "mapping_efficiency_percent": "Mapping Efficiency %",
                    "compute_util_percent": "Compute Util %",
                }

                print(f"\nLayer {i} ({ml_result['layer_name']}):")
                print(
                    f"{'Metric':<30} {'ML Prediction':>15} {'Simulation':>15} {'Error %':>10}"
                )
                print("-" * 70)

                for ml_key, sim_key in metric_map.items():
                    ml_val = ml_result[ml_key]
                    sim_val = float(str(sim_row[sim_key]).strip().rstrip(","))

                    error_pct = (
                        abs(ml_val - sim_val) / sim_val * 100 if sim_val != 0 else 0
                    )

                    comp["metrics"][ml_key] = {
                        "ml": ml_val,
                        "sim": sim_val,
                        "error_pct": error_pct,
                    }

                    print(
                        f"{ml_key:<30} {ml_val:>15.2f} {sim_val:>15.2f} {error_pct:>9.2f}%"
                    )

                comparison.append(comp)

        return {
            "ml_predictions": ml_results,
            "simulation_results": compute_report.to_dict("records"),
            "comparison": comparison,
        }

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Main entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate SCALE-Sim ML Predictor")
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to test data CSV"
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
        "--output", type=str, default=None, help="Path to save evaluation results"
    )

    args = parser.parse_args()

    evaluate_model(
        data_path=args.data_path,
        model_path=args.model,
        preprocessor_path=args.preprocessor,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
