"""
Training script for traffic object detection model.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import yaml
from ultralytics import YOLO

from src.utils.data_utils import prepare_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train traffic light detection model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/detection_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data", type=str, default="data/dataset.yaml", help="Path to dataset configuration"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    parser.add_argument("--weights", type=str, default="yolov8s.pt", help="Initial weights path")
    parser.add_argument("--output", type=str, default="models", help="Output directory")
    parser.add_argument(
        "--device", type=str, default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument("--workers", type=int, default=8, help="Number of worker threads")
    parser.add_argument("--project", type=str, default="traffic_detection", help="Project name")
    parser.add_argument("--name", type=str, default="exp", help="Experiment name")

    return parser.parse_args()


def load_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("train")

    # Parse arguments
    args = parse_args()

    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")

        # Override args with config values
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    else:
        logger.warning(f"Configuration file {args.config} not found, using default values")

    # Prepare dataset if needed
    if not os.path.exists(args.data):
        logger.info(f"Dataset configuration not found at {args.data}, preparing dataset")
        prepare_dataset(args.data)

    # Initialize model
    logger.info(f"Initializing model with weights from {args.weights}")
    model = YOLO(args.weights)

    # Set training parameters
    training_args = {
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch_size,
        "imgsz": args.img_size,
        "project": args.project,
        "name": args.name,
        "workers": args.workers,
    }

    # If device is specified, add it to training args
    if args.device:
        training_args["device"] = args.device

    # Train model
    logger.info(f"Starting training with parameters: {training_args}")
    results = model.train(**training_args)

    # Save trained model
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    model_path = output_dir / "traffic_detector.pt"
    model.export(format="onnx")  # Export to ONNX format

    # Save best model
    if hasattr(results, "best") and os.path.exists(results.best):
        logger.info(f"Saving best model to {model_path}")
        import shutil

        shutil.copy(results.best, model_path)
    else:
        logger.warning("Best model not found, saving last model")
        model.save(model_path)

    logger.info(f"Training complete, model saved to {model_path}")


if __name__ == "__main__":
    main()
