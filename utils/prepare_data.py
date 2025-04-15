#!/usr/bin/env python
"""
Data Preparation Tools - Check and prepare training data
"""

import os
import argparse
import shutil
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_data_directory(data_dir):
    """Check if the data directory is valid"""
    data_path = Path(data_dir)

    if not data_path.exists():
        logger.error(f"Data directory does not exist: {data_path.absolute()}")
        return False

    # Check if there are instrument subdirectories
    instrument_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    if not instrument_dirs:
        logger.warning(
            f"No instrument subdirectories in data directory: {data_path.absolute()}"
        )
        return False

    # Check audio files in each instrument directory
    total_audio_files = 0
    for inst_dir in instrument_dirs:
        audio_files = [
            f
            for f in inst_dir.iterdir()
            if f.suffix.lower() in (".wav", ".mp3", ".flac")
        ]
        logger.info(f"Instrument {inst_dir.name}: {len(audio_files)} audio files")
        total_audio_files += len(audio_files)

    logger.info(
        f"Total: {len(instrument_dirs)} instruments, {total_audio_files} audio files"
    )
    return total_audio_files > 0


def create_sample_data(output_dir):
    """Create sample data directory structure"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Create instrument directories
    instruments = ["erhu", "guzheng", "pipa", "dizi", "guqin", "yangqin"]
    for inst in instruments:
        inst_dir = output_path / inst
        inst_dir.mkdir(exist_ok=True)

        # Create an empty placeholder file
        placeholder = inst_dir / ".placeholder"
        placeholder.touch()

        logger.info(f"Created instrument directory: {inst_dir}")

    logger.info(f"Sample data directory structure created: {output_path.absolute()}")
    logger.info("Please add audio files to each instrument directory")


def organize_audio_files(input_dir, output_dir, default_instrument=None):
    """Organize audio files into instrument directories"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_path.absolute()}")
        return False

    # Ensure output directory exists
    output_path.mkdir(exist_ok=True, parents=True)

    # Get all audio files
    audio_files = []
    for root, _, files in os.walk(input_path):
        for file in files:
            if file.lower().endswith((".wav", ".mp3", ".flac")):
                audio_files.append(Path(root) / file)

    logger.info(f"Found {len(audio_files)} audio files in {input_path}")

    if not audio_files:
        logger.warning("No audio files found")
        return False

    # Define instrument keyword mapping
    instrument_keywords = {
        "erhu": ["erhu", "erhu"],
        "guzheng": ["guzheng", "guzheng", "zheng"],
        "pipa": ["pipa", "pipa"],
        "dizi": ["dizi", "dizi", "flute"],
        "guqin": ["guqin", "guqin"],
        "yangqin": ["yangqin", "yangqin"],
    }

    # Create instrument directories
    for instrument in instrument_keywords.keys():
        (output_path / instrument).mkdir(exist_ok=True)

    # If default instrument is provided, ensure its directory exists
    if default_instrument:
        (output_path / default_instrument).mkdir(exist_ok=True)

    # Organize files
    organized_count = 0
    for audio_file in audio_files:
        file_name = audio_file.name.lower()
        target_instrument = None

        # Determine instrument type based on filename
        for instrument, keywords in instrument_keywords.items():
            if any(keyword.lower() in file_name for keyword in keywords):
                target_instrument = instrument
                break

        # If instrument type cannot be determined, use default instrument
        if not target_instrument and default_instrument:
            target_instrument = default_instrument

        # If there is a target instrument, copy the file
        if target_instrument:
            target_dir = output_path / target_instrument
            target_file = target_dir / audio_file.name

            # Copy file
            shutil.copy2(audio_file, target_file)
            logger.info(f"Copied {audio_file.name} to {target_instrument} directory")
            organized_count += 1
        else:
            logger.warning(
                f"Cannot determine instrument type for {audio_file.name}, skipped"
            )

    logger.info(f"Organized {organized_count}/{len(audio_files)} audio files")
    return True


def main():
    parser = argparse.ArgumentParser(description="Data Preparation Tool")
    parser.add_argument("--data-dir", default="../wav", help="Data directory path")
    parser.add_argument("--check", action="store_true", help="Check data directory")
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create sample data directory structure",
    )
    parser.add_argument("--output-dir", default="../wav", help="Output directory path")
    parser.add_argument(
        "--organize",
        action="store_true",
        help="Organize audio files into instrument directories",
    )
    parser.add_argument(
        "--default-instrument",
        default=None,
        help="Default instrument when type cannot be recognized",
    )

    args = parser.parse_args()

    if args.check:
        if check_data_directory(args.data_dir):
            logger.info("Data directory check passed")
        else:
            logger.error("Data directory check failed")

    if args.create_sample:
        create_sample_data(args.output_dir)

    if args.organize:
        organize_audio_files(args.data_dir, args.output_dir, args.default_instrument)


if __name__ == "__main__":
    main()
