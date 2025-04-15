#!/usr/bin/env python
"""
Timbre Feature Extraction Script

This script extracts timbre features from audio files using the trained model
and saves them in formats that can be reused for other applications.
"""

import os
import sys
import argparse
import json
import pickle
import numpy as np
import torch
from pathlib import Path

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import model
from models.model import InstrumentTimbreModel


def extract_and_save_timbre(audio_file, model, output_dir, formats=None):
    """
    Extract timbre features from audio file and save in specified formats

    Args:
        audio_file: Path to audio file
        model: Trained model to use
        output_dir: Directory to save output files
        formats: List of output formats ('json', 'pickle', 'numpy')

    Returns:
        Dictionary of paths to saved files
    """
    if formats is None:
        formats = ["json", "pickle", "numpy"]

    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(audio_file))[0]

    print(f"Extracting timbre features from {audio_file}...")

    # Extract features using the model at different feature levels
    features = {}
    output_files = {}

    try:
        # Extract features at different levels of detail
        for level in [1, 2, 3]:
            feature_vector = model.extract_timbre(
                audio_file, feature_level=level, normalize=True
            )
            features[f"level_{level}"] = (
                feature_vector.tolist()
                if isinstance(feature_vector, np.ndarray)
                else feature_vector
            )

        # Add metadata
        features["metadata"] = {
            "source_file": os.path.basename(audio_file),
            "extraction_time": str(np.datetime64("now")),
            "model_config": model.get_config()
            if hasattr(model, "get_config")
            else {"type": "InstrumentTimbreModel"},
        }

        # Save in requested formats
        if "json" in formats:
            json_path = os.path.join(output_dir, f"{base_name}_timbre.json")
            with open(json_path, "w") as f:
                json.dump(features, f, indent=2)
            output_files["json"] = json_path
            print(f"Saved JSON: {json_path}")

        if "pickle" in formats:
            pickle_path = os.path.join(output_dir, f"{base_name}_timbre.pkl")
            with open(pickle_path, "wb") as f:
                pickle.dump(features, f)
            output_files["pickle"] = pickle_path
            print(f"Saved Pickle: {pickle_path}")

        if "numpy" in formats:
            # Save just the feature vectors as numpy arrays
            for level in [1, 2, 3]:
                feature_key = f"level_{level}"
                if feature_key in features:
                    feature_vector = features[feature_key]
                    if isinstance(feature_vector, list):
                        feature_vector = np.array(feature_vector)
                    numpy_path = os.path.join(
                        output_dir, f"{base_name}_timbre_level{level}.npy"
                    )
                    np.save(numpy_path, feature_vector)
                    output_files[f"numpy_level{level}"] = numpy_path
                    print(f"Saved NumPy Level {level}: {numpy_path}")

        return output_files

    except Exception as e:
        print(f"Error extracting timbre from {audio_file}: {e}")
        return None


def extract_advanced_features(audio_file, model, output_dir):
    """
    Extract more detailed timbre features using direct model components
    This provides access to internal representations that might be useful
    for specific applications

    Args:
        audio_file: Path to audio file
        model: Trained model to use
        output_dir: Directory to save outputs

    Returns:
        Path to saved feature file
    """
    try:
        # Use internal model methods for advanced extraction
        # Load audio file
        audio_data, sample_rate = model._load_audio_fallback(audio_file)
        audio_tensor = torch.tensor(audio_data).float()

        # Ensure correct dimensions
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        # Create feature dictionary
        features = {}

        # Extract mel spectrogram
        mel_spec = model.to_mel_spectrogram(audio_tensor, sample_rate)

        # Extract chroma features
        chroma = model.to_chroma(audio_tensor, sample_rate)

        # Extract MFCC features
        mfcc = model.to_mfcc(audio_tensor, sample_rate)

        # Convert to numpy for storage
        features["mel_spectrogram"] = mel_spec.cpu().numpy()
        features["chroma"] = chroma.cpu().numpy()
        features["mfcc"] = mfcc.cpu().numpy()

        # Add some derived features
        features["spectral_contrast"] = np.std(mel_spec.cpu().numpy(), axis=1)
        features["tonal_centroids"] = np.mean(chroma.cpu().numpy(), axis=1)

        # Save as pickle
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        out_path = os.path.join(output_dir, f"{base_name}_advanced_features.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(features, f)

        print(f"Saved advanced features: {out_path}")
        return out_path

    except Exception as e:
        print(f"Error extracting advanced features: {e}")
        # Fallback method if internal methods fail
        try:
            fallback_out_path = os.path.join(
                output_dir, f"{os.path.basename(audio_file)}_fallback.npy"
            )
            feature_vector = model.extract_timbre(audio_file)
            np.save(fallback_out_path, feature_vector)
            print(f"Saved fallback features: {fallback_out_path}")
            return fallback_out_path
        except:
            print("Fallback extraction also failed")
            return None


def _load_audio_fallback(audio_file):
    """
    Fallback method for audio loading if model method is not available
    """
    import librosa

    y, sr = librosa.load(audio_file, sr=None)
    return y, sr


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Extract timbre features from audio files"
    )
    parser.add_argument(
        "--input-dir", default=".", help="Directory containing audio files"
    )
    parser.add_argument(
        "--output-dir",
        default="./extracted_features",
        help="Directory to save extracted features",
    )
    parser.add_argument(
        "--model-path", default="../saved_models/model.pt", help="Path to trained model"
    )
    parser.add_argument(
        "--formats",
        default="json,pickle,numpy",
        help="Output formats (comma-separated)",
    )
    parser.add_argument(
        "--extensions",
        default="wav",
        help="Audio file extensions to process (comma-separated)",
    )
    parser.add_argument(
        "--advanced", action="store_true", help="Extract advanced features"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the model
    model = InstrumentTimbreModel(model_path=args.model_path)

    # Add fallback method if needed
    if not hasattr(model, "_load_audio"):
        model._load_audio_fallback = _load_audio_fallback

    # Parse formats and extensions
    formats = [f.strip() for f in args.formats.split(",")]
    extensions = [ext.strip() for ext in args.extensions.split(",")]

    # Find audio files
    audio_files = []
    for ext in extensions:
        audio_files.extend(list(Path(args.input_dir).glob(f"*.{ext}")))

    if not audio_files:
        print(f"No audio files found in {args.input_dir} with extensions {extensions}")
        return

    print(f"Found {len(audio_files)} audio files")

    # Process each file
    results = {}
    for audio_file in audio_files:
        audio_path = str(audio_file)
        # Extract standard features
        feature_files = extract_and_save_timbre(
            audio_path, model, args.output_dir, formats
        )
        results[os.path.basename(audio_path)] = {"standard": feature_files}

        # Extract advanced features if requested
        if args.advanced:
            advanced_file = extract_advanced_features(
                audio_path, model, args.output_dir
            )
            if advanced_file:
                results[os.path.basename(audio_path)]["advanced"] = advanced_file

    # Save a summary of all extractions
    summary_path = os.path.join(args.output_dir, "extraction_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nExtraction complete! Summary saved to: {summary_path}")
    print(f"All extracted features saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
