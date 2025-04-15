#!/usr/bin/env python
"""
Instrument Timbre Analysis and Conversion System - Training Script
"""

import os
import argparse
import logging
import torch
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.model import InstrumentTimbreModel
from utils.data import prepare_dataloader, prepare_chinese_instrument_dataloader

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Instrument Timbre Analysis and Conversion System - Training Script"
    )

    # Dataset parameters
    parser.add_argument(
        "--data-dir", "--dataset-path", default="../wav", help="Training data directory"
    )
    parser.add_argument(
        "--model-path", default="./saved_models/model.pt", help="Model save path"
    )
    parser.add_argument(
        "--use-wav-files",
        action="store_true",
        help="Use WAV files directly as training data",
    )
    parser.add_argument(
        "--augment", action="store_true", help="Apply data augmentation"
    )

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--lr", "--learning-rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="Early stopping patience"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of data loading threads"
    )

    # Model parameters
    parser.add_argument(
        "--chinese-instruments",
        action="store_true",
        help="Optimize for Chinese traditional instruments",
    )
    parser.add_argument(
        "--feature-type",
        choices=["mel", "constant-q", "multi"],
        default="multi",
        help="Feature type (only applicable to Chinese instrument mode)",
    )
    parser.add_argument(
        "--pretrained", action="store_true", help="Use pretrained audio model"
    )

    # Other parameters
    parser.add_argument(
        "--cache-features", action="store_true", help="Enable feature caching"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--export-onnx", action="store_true", help="Export trained model to ONNX format"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "mps", "auto"],
        default="auto",
        help="Computation device",
    )

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    # Ensure model save directory exists
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    # Set device
    if args.device == "auto":
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")

    # Prepare data loader
    if args.chinese_instruments:
        logger.info("Using specialized data loader for Chinese traditional instruments")
        dataloader = prepare_chinese_instrument_dataloader(
            args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_wav_files=args.use_wav_files,
            augment=args.augment,
            debug=args.debug,
            feature_type=args.feature_type,
        )
    else:
        logger.info("Using standard data loader")
        dataloader = prepare_dataloader(
            args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_wav_files=args.use_wav_files,
            augment=args.augment,
            debug=args.debug,
        )

    # Initialize model
    model = InstrumentTimbreModel(
        use_pretrained=args.pretrained,
        chinese_instruments=args.chinese_instruments,
        feature_caching=args.cache_features,
        device=device,
    )

    # Train model
    logger.info("Starting model training...")
    model.train(
        dataloader=dataloader,
        epochs=args.epochs,
        learning_rate=args.lr,
        max_batches=100 if args.debug else None,
    )

    # Save model
    logger.info(f"Saving model to {args.model_path}")
    model.save_model(args.model_path)

    # Export ONNX (if needed)
    if args.export_onnx:
        onnx_path = args.model_path.replace(".pt", ".onnx")
        logger.info(f"Exporting ONNX model to {onnx_path}")
        model.export_to_onnx(onnx_path)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
