"""
Main entry point for SCALE-Sim ML Predictor.
Provides a unified CLI for data generation, training, and prediction.
"""

import argparse
import sys
from pathlib import Path


def cmd_generate(args):
    """Generate training data."""
    from ml_predictor.data_generation import DataGenerator

    generator = DataGenerator(seed=args.seed)
    try:
        generator.generate(
            num_samples=args.num_samples,
            output_file=args.output,
            show_progress=True,
            num_workers=args.workers,
        )
    finally:
        generator.cleanup()


def cmd_train(args):
    """Train the model."""
    from ml_predictor.train import train_model

    train_model(
        data_path=args.data_path,
        model_save_path=args.model_path,
        preprocessor_save_path=args.preprocessor_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        verbose=True,
    )


def cmd_predict(args):
    """Make predictions."""
    from ml_predictor.predict import predict

    predict(
        config_path=args.config,
        topology_path=args.topology,
        model_path=args.model,
        preprocessor_path=args.preprocessor,
        output_path=args.output,
    )


def cmd_evaluate(args):
    """Evaluate model on test data."""
    from ml_predictor.evaluate import evaluate_model

    evaluate_model(
        data_path=args.data_path,
        model_path=args.model,
        preprocessor_path=args.preprocessor,
        output_path=args.output,
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SCALE-Sim ML Predictor - Predict simulation outputs using ML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate training data (5000 samples)
  python -m ml_predictor.main generate --num_samples 5000 --output ./data/raw/training_data.csv

  # Train the model
  python -m ml_predictor.main train --data_path ./data/raw/training_data.csv

  # Make predictions
  python -m ml_predictor.main predict --config ./configs/google.cfg --topology ./topologies/ispass25_models/alexnet.csv

  # Evaluate model
  python -m ml_predictor.main evaluate --data_path ./data/raw/test_data.csv --model ./models/scalesim_predictor.pt
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ================== Generate command ==================
    gen_parser = subparsers.add_parser("generate", help="Generate training data")
    gen_parser.add_argument(
        "--num_samples",
        type=int,
        default=5000,
        help="Number of samples to generate (default: 5000)",
    )
    gen_parser.add_argument(
        "--output",
        type=str,
        default="./data/raw/training_data.csv",
        help="Output CSV file path",
    )
    gen_parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    gen_parser.add_argument(
        "--workers", type=int, default=1, help="Number of parallel workers"
    )
    gen_parser.set_defaults(func=cmd_generate)

    # ================== Train command ==================
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--data_path", type=str, required=True, help="Path to training data CSV"
    )
    train_parser.add_argument(
        "--model_path",
        type=str,
        default="./models/scalesim_predictor.pt",
        help="Path to save trained model",
    )
    train_parser.add_argument(
        "--preprocessor_path",
        type=str,
        default="./models/preprocessor.pkl",
        help="Path to save preprocessor",
    )
    train_parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    train_parser.add_argument(
        "--batch_size", type=int, default=64, help="Training batch size"
    )
    train_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    train_parser.add_argument(
        "--device", type=str, default=None, help="Training device (cpu/cuda)"
    )
    train_parser.set_defaults(func=cmd_train)

    # ================== Predict command ==================
    pred_parser = subparsers.add_parser("predict", help="Make predictions")
    pred_parser.add_argument(
        "--config", type=str, required=True, help="Path to hardware config file"
    )
    pred_parser.add_argument(
        "--topology", type=str, required=True, help="Path to topology file"
    )
    pred_parser.add_argument(
        "--model",
        type=str,
        default="./models/scalesim_predictor.pt",
        help="Path to trained model",
    )
    pred_parser.add_argument(
        "--preprocessor",
        type=str,
        default="./models/preprocessor.pkl",
        help="Path to preprocessor",
    )
    pred_parser.add_argument(
        "--output", type=str, default=None, help="Path to save results CSV"
    )
    pred_parser.set_defaults(func=cmd_predict)

    # ================== Evaluate command ==================
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model on test data")
    eval_parser.add_argument(
        "--data_path", type=str, required=True, help="Path to test data CSV"
    )
    eval_parser.add_argument(
        "--model",
        type=str,
        default="./models/scalesim_predictor.pt",
        help="Path to trained model",
    )
    eval_parser.add_argument(
        "--preprocessor",
        type=str,
        default="./models/preprocessor.pkl",
        help="Path to preprocessor",
    )
    eval_parser.add_argument(
        "--output", type=str, default=None, help="Path to save evaluation results"
    )
    eval_parser.set_defaults(func=cmd_evaluate)

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()
