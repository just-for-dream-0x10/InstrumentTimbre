"""
Timbre training service for model development.

Provides comprehensive training pipeline for Chinese instrument
timbre recognition models with data augmentation and validation.
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ..core import (
    InstrumentType,
    TimbreTrainingError,
    TrainingResult,
    get_logger,
    log_training_epoch,
)
from .base_timbre_service import BaseTimbreService


class TimbreTrainingService(BaseTimbreService):
    """
    Professional training service for timbre recognition models.

    Specializes in Chinese instrument data with advanced augmentation
    and validation strategies for robust model development.
    """

    def __init__(self):
        """Initialize timbre training service."""
        super().__init__("TimbreTraining")
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

    def process(
        self,
        training_data_dir: str,
        validation_split: Optional[float] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> TrainingResult:
        """
        Train timbre recognition model with comprehensive validation.

        Args:
            training_data_dir: Directory containing training audio files
            validation_split: Fraction for validation (uses config default if None)
            resume_from_checkpoint: Path to checkpoint for resuming training

        Returns:
            TrainingResult: Comprehensive training results and metrics

        Raises:
            TimbreTrainingError: If training process fails
        """
        try:
            # Setup training configuration
            config = self.config.training
            val_split = validation_split or config.validation_split

            # Prepare data loaders
            with self._performance_monitor("data_preparation"):
                train_loader, val_loader = self._prepare_data_loaders(
                    training_data_dir, val_split
                )

            # Initialize model and training components
            with self._performance_monitor("model_initialization"):
                self._initialize_training_components()

            # Resume from checkpoint if provided
            start_epoch = 0
            if resume_from_checkpoint:
                start_epoch = self._load_checkpoint(resume_from_checkpoint)

            # Execute training loop
            with self._performance_monitor("training_execution"):
                result = self._execute_training_loop(
                    train_loader, val_loader, start_epoch
                )

            # Save final model
            final_model_path = self._save_final_model()
            result.model_path = final_model_path

            self.logger.info(
                f"Training completed: {result.epochs_completed} epochs, "
                f"best accuracy: {result.best_val_accuracy:.4f}"
            )

            return result

        except Exception as e:
            raise TimbreTrainingError(f"Training failed: {e}")

    def _prepare_data_loaders(
        self, data_dir: str, val_split: float
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare training and validation data loaders.

        Args:
            data_dir: Directory containing training data
            val_split: Validation split fraction

        Returns:
            Tuple of (train_loader, val_loader)
        """
        from ..utils.data_loader import ChineseInstrumentDataset

        # Create dataset
        dataset = ChineseInstrumentDataset(
            data_dir=data_dir,
            config=self.config.analysis,
            enable_augmentation=self.config.training.enable_augmentation,
        )

        # Split dataset
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=min(4, self.config.system.max_workers),
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=min(4, self.config.system.max_workers),
        )

        self.logger.info(
            f"Dataset prepared: {train_size} training, {val_size} validation samples"
        )

        return train_loader, val_loader
