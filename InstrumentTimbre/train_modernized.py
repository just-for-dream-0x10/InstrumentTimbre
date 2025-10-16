#!/usr/bin/env python3
"""
Modernized training script for InstrumentTimbre.

Migrated from legacy train.py to use new architecture with
proper error handling, logging, and configuration management.
"""

import argparse
import sys
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.core import TimbreTrainingError, get_config, get_logger, setup_logging
from modules.services import TimbreTrainingService


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create command line argument parser.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Train InstrumentTimbre model with modern architecture",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../wav",
        help="Directory containing training audio files",
    )

    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Fraction of data to use for validation (0.0-0.5)",
    )

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )

    parser.add_argument(
        "--batch-size", type=int, default=32, help="Training batch size"
    )

    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate for optimizer"
    )

    parser.add_argument(
        "--patience", type=int, default=15, help="Early stopping patience"
    )

    # Model arguments
    parser.add_argument(
        "--model-size",
        choices=["tiny", "base", "large"],
        default="base",
        help="Model size configuration",
    )

    parser.add_argument(
        "--chinese-instruments",
        action="store_true",
        help="Optimize for Chinese traditional instruments",
    )

    # Data augmentation
    parser.add_argument(
        "--augment", action="store_true", help="Enable data augmentation"
    )

    parser.add_argument(
        "--use-wav-files",
        action="store_true",
        help="Use WAV files directly from data directory",
    )

    # Training control
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        help="Path to checkpoint file to resume training",
    )

    parser.add_argument(
        "--cache-features",
        action="store_true",
        help="Enable feature caching for faster training",
    )

    # Device and performance
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to use for training",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of data loading workers",
    )

    # Output and logging
    parser.add_argument(
        "--model-path",
        type=str,
        default="saved_models/timbre_model.pt",
        help="Path to save trained model",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode with additional logging"
    )

    return parser


def update_config_from_args(args) -> None:
    """
    Update global configuration from command line arguments.

    Args:
        args: Parsed command line arguments
    """
    config = get_config()

    # Update training configuration
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    config.training.early_stopping_patience = args.patience
    config.training.validation_split = args.validation_split
    config.training.model_size = args.model_size
    config.training.chinese_instruments_only = args.chinese_instruments
    config.training.enable_augmentation = args.augment

    # Update system configuration
    config.system.device = args.device
    config.system.max_workers = args.max_workers
    config.system.training_data_dir = args.data_dir
    config.system.enable_feature_cache = args.cache_features

    # Set log level
    if args.debug:
        config.system.log_level = "DEBUG"
    else:
        config.system.log_level = args.log_level

    # Validate updated configuration
    config.validate()


def main():
    """Main training function."""
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    # Setup logging
    log_level = getattr(__import__("logging"), args.log_level)
    setup_logging(level=log_level)

    logger = get_logger()
    logger.info("🎵 Starting InstrumentTimbre training with modern architecture")
    logger.info("=" * 60)

    try:
        # Update configuration from arguments
        update_config_from_args(args)

        # Log configuration
        config = get_config()
        logger.info(f"Training configuration:")
        logger.info(f"  Data directory: {config.system.training_data_dir}")
        logger.info(f"  Epochs: {config.training.epochs}")
        logger.info(f"  Batch size: {config.training.batch_size}")
        logger.info(f"  Learning rate: {config.training.learning_rate}")
        logger.info(f"  Model size: {config.training.model_size}")
        logger.info(
            f"  Chinese instruments: {config.training.chinese_instruments_only}"
        )
        logger.info(f"  Data augmentation: {config.training.enable_augmentation}")
        logger.info(f"  Device: {config.system.device}")
        logger.info("")

        # Check data directory
        data_path = Path(config.system.training_data_dir)
        if not data_path.exists():
            raise TimbreTrainingError(f"Training data directory not found: {data_path}")

        # Count audio files
        audio_files = list(data_path.glob("*.wav"))
        if len(audio_files) == 0:
            raise TimbreTrainingError(f"No WAV files found in {data_path}")

        logger.info(f"Found {len(audio_files)} audio files for training")

        # Initialize training service
        training_service = TimbreTrainingService()

        # Execute training
        logger.info("🚀 Starting model training...")
        result = training_service.safe_process(
            training_data_dir=str(data_path),
            validation_split=config.training.validation_split,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )

        # Handle results
        if result.success:
            logger.info("🎉 Training completed successfully!")
            logger.info("=" * 60)

            # Log training summary
            summary = result.get_training_summary()
            logger.info("Training Summary:")
            logger.info(f"  Epochs completed: {summary['epochs_completed']}")
            logger.info(
                f"  Best validation accuracy: {summary['best_val_accuracy']:.4f}"
            )
            logger.info(f"  Final training loss: {summary['final_train_loss']:.4f}")
            logger.info(f"  Final validation loss: {summary['final_val_loss']:.4f}")
            logger.info(f"  Total training time: {summary['total_training_time']:.2f}s")
            logger.info(f"  Model size: {summary['model_size_mb']:.2f} MB")
            logger.info(f"  Training samples: {summary['train_samples']}")
            logger.info(f"  Validation samples: {summary['val_samples']}")

            if result.model_path:
                logger.info(f"  Model saved to: {result.model_path}")

            # Log statistics
            stats = training_service.get_timbre_stats()
            logger.info(f"\nService Statistics:")
            logger.info(
                f"  Chinese instruments processed: {stats['chinese_instruments_processed']}"
            )
            logger.info(
                f"  Western instruments processed: {stats['western_instruments_processed']}"
            )

        else:
            logger.error("❌ Training failed!")
            logger.error(f"Error: {result.error_message}")
            sys.exit(1)

    except TimbreTrainingError as e:
        logger.error(f"❌ Training error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
