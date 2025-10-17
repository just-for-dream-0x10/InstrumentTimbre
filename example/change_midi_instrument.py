import mido
import argparse
import os
import json
import random
import numpy as np

# Define instrument mapping (MIDI Program Numbers are 0-based in mido)
INSTRUMENT_MAP = {
    "piano": {"name": "Acoustic Grand Piano", "midi_program": 0},
    "violin": {"name": "Violin", "midi_program": 40},
    "cello": {"name": "Cello", "midi_program": 42},
    "trumpet": {"name": "Trumpet", "midi_program": 56},
    "flute": {"name": "Flute", "midi_program": 73},
    "guitar": {"name": "Acoustic Guitar (steel)", "midi_program": 25},
    "saxophone": {"name": "Alto Sax", "midi_program": 65},
    "erhu": {
        "name": "Erhu (often mapped to Violin or Synth Voice)",
        "midi_program": 110,
    },
    "electronic keyboard": {"name": "Synth Voice", "midi_program": 80},
    # Add more instruments as needed
}

# --- Helper Functions ---

import pickle


def load_timbre_features_any(filepath):
    if filepath.endswith(".json"):
        return load_timbre_features(filepath)
    elif filepath.endswith(".pkl"):
        with open(filepath, "rb") as f:
            features = pickle.load(f)
        print(f"Successfully loaded timbre features from: {filepath}")
        return features
    else:
        raise ValueError("Unsupported feature file type")


def load_timbre_features(filepath):
    """Loads timbre features from a JSON file."""
    try:
        with open(filepath, "r") as f:
            features = json.load(f)
        print(f"Successfully loaded timbre features from: {filepath}")
        # Basic validation (can be expanded)
        if not isinstance(features, dict):
            print("Warning: Expected features to be a dictionary.")
            return None
        return features
    except FileNotFoundError:
        print(f"Error: Timbre feature file not found at {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        return None
    except Exception as e:
        print(f"An error occurred loading timbre features: {e}")
        return None


def calculate_feature_stats(features):
    """Calculate basic statistics from feature values (assuming numerical lists/arrays)."""
    stats = {}
    if not features or not isinstance(features, dict):
        return stats

    # Example: Calculate stats for 'mfcc_mean' if it exists and is numerical
    if "mfcc_mean" in features and isinstance(
        features["mfcc_mean"], (list, np.ndarray)
    ):
        mfcc_mean_numeric = [
            x for x in features["mfcc_mean"] if isinstance(x, (int, float))
        ]
        if mfcc_mean_numeric:
            stats["mfcc_avg"] = np.mean(mfcc_mean_numeric)
            stats["mfcc_std"] = np.std(mfcc_mean_numeric)

    # Example: Calculate stats for 'spectral_centroid_mean'
    if "spectral_centroid_mean" in features and isinstance(
        features["spectral_centroid_mean"], (int, float)
    ):
        # Normalize or scale spectral centroid for CC mapping (0-127)
        # This scaling factor is arbitrary and needs tuning
        stats["brightness_proxy"] = min(
            127, int(features["spectral_centroid_mean"] / 100)
        )

    # Add more feature calculations as needed
    return stats


def change_midi_instrument(
    input_midi_path, target_instrument_name, output_midi_path, timbre_features_path=None
):
    """
    Changes the instrument (MIDI program) for all tracks in a MIDI file,
    optionally applying timbre features and weakening original characteristics.

    Args:
        input_midi_path (str): Path to the input MIDI file.
        target_instrument_name (str): The name of the target instrument.
        output_midi_path (str): Path to save the modified MIDI file.
        timbre_features_path (str, optional): Path to the timbre features JSON file.
    """
    if target_instrument_name not in INSTRUMENT_MAP:
        print(
            f"Error: Instrument '{target_instrument_name}' not found in INSTRUMENT_MAP."
        )
        print(f"Available instruments: {', '.join(INSTRUMENT_MAP.keys())}")
        return False

    target_program = INSTRUMENT_MAP[target_instrument_name]["midi_program"]
    print(
        f"Target instrument: {target_instrument_name} (MIDI Program: {target_program})"
    )

    # Load timbre features if path provided
    timbre_features = None
    feature_stats = {}
    apply_erhu_features = False
    erhu_cc11_value = 64
    erhu_velocity_scale = 1.0
    if timbre_features_path and target_instrument_name == "erhu":
        timbre_features = load_timbre_features_any(timbre_features_path)
        if timbre_features:
            feature_stats = calculate_feature_stats(timbre_features)
            erhu_cc11_value = int(
                min(127, max(0, feature_stats.get("brightness_proxy", 64)))
            )
            erhu_velocity_scale = float(feature_stats.get("energy_mean", 1.0))
            apply_erhu_features = True
            print(f"Feature stats for Erhu: {feature_stats}")
            print(
                f"Erhu timbre mapped: CC11={erhu_cc11_value}, velocity_scale={erhu_velocity_scale}"
            )

    try:
        mid = mido.MidiFile(input_midi_path)
        print(f"Loaded MIDI file: {input_midi_path} ({len(mid.tracks)} tracks)")

        for i, track in enumerate(mid.tracks):
            print(f"Processing Track {i}: {track.name}")
            has_program_change = False
            first_non_meta_idx = None
            first_channel = 0
            for idx, msg in enumerate(track):
                if not msg.is_meta:
                    first_non_meta_idx = idx
                    if hasattr(msg, "channel"):
                        first_channel = msg.channel
                    break
            new_track = mido.MidiTrack()
            for idx, msg in enumerate(track):
                # Remove sustain pedal
                if msg.type == "control_change" and msg.control == 64:
                    print(f"  - Removing Sustain Pedal (CC 64) at time {msg.time}")
                    continue
                # Erhu branch: insert expression, modulation, pitch bend, and velocity enhancement
                if target_instrument_name == "erhu" and apply_erhu_features:
                    import math

                    if msg.type == "program_change":
                        msg = msg.copy(program=40)  # Violin
                    if msg.type == "note_on" and msg.velocity > 0:
                        # Expression (CC11)
                        new_track.append(
                            mido.Message(
                                "control_change",
                                control=11,
                                value=erhu_cc11_value,
                                time=0,
                                channel=msg.channel,
                            )
                        )
                        # Vibrato/tremolo parameters
                        vib_cycles = 3
                        vib_depth = int(10 + 10 * feature_stats.get("mfcc_avg", 1))
                        bend_depth = int(200 + 20 * feature_stats.get("mfcc_std", 1))
                        # Glide (slide) start
                        new_track.append(
                            mido.Message(
                                "pitchwheel",
                                pitch=-bend_depth,
                                time=0,
                                channel=msg.channel,
                            )
                        )
                        # Vibrato CC1 (modulation) periodic changes
                        for vib_step in range(vib_cycles):
                            vib_val = int(
                                64
                                + vib_depth
                                * math.sin(2 * math.pi * vib_step / vib_cycles)
                            )
                            new_track.append(
                                mido.Message(
                                    "control_change",
                                    control=1,
                                    value=vib_val,
                                    time=0,
                                    channel=msg.channel,
                                )
                            )
                        # Slide process and tremolo (pitch bend periodic changes)
                        for bend_step in range(vib_cycles):
                            bend_val = int(
                                bend_depth
                                * math.sin(2 * math.pi * bend_step / vib_cycles)
                            )
                            new_track.append(
                                mido.Message(
                                    "pitchwheel",
                                    pitch=bend_val,
                                    time=5,
                                    channel=msg.channel,
                                )
                            )
                        # Reset pitch bend after slide
                        new_track.append(
                            mido.Message(
                                "pitchwheel", pitch=0, time=0, channel=msg.channel
                            )
                        )
                        # Velocity scaled by energy
                        new_msg = msg.copy(
                            velocity=int(
                                min(
                                    127, max(1, int(msg.velocity * erhu_velocity_scale))
                                )
                            )
                        )
                        new_track.append(new_msg)
                        continue
                elif target_instrument_name == "piano":
                    if msg.type == "program_change":
                        print(
                            f"  - Removing Program Change at time {msg.time} for piano safety"
                        )
                        continue
                else:
                    # 其他乐器正常替换
                    if msg.type == "program_change":
                        has_program_change = True
                        if msg.program != target_program:
                            print(
                                f"  - Changing Program Change from {msg.program} to {target_program} at time {msg.time}"
                            )
                        msg = msg.copy(program=target_program)
                new_track.append(msg)
            # 钢琴分支：不插入program_change
            if (
                target_instrument_name != "piano"
                and not has_program_change
                and first_non_meta_idx is not None
            ):
                msg0 = new_track[first_non_meta_idx]
                pc_msg = mido.Message(
                    "program_change",
                    program=target_program,
                    channel=first_channel,
                    time=msg0.time,
                )
                new_track.insert(first_non_meta_idx, pc_msg)
                # Reset delta time of the original message
                new_track[first_non_meta_idx + 1] = msg0.copy(time=0)
            mid.tracks[i] = new_track

        # Save the modified MIDI file
        print(f"Saving modified MIDI file to: {output_midi_path}")
        mid.save(output_midi_path)
        print(f"Successfully saved modified MIDI.")
        return True

    except FileNotFoundError:
        print(f"Error: Input MIDI file not found at {input_midi_path}")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Change instrument and apply timbre features to a MIDI file."
    )
    parser.add_argument("--input", required=True, help="Input MIDI file path (.mid)")
    parser.add_argument(
        "--instrument",
        required=True,
        choices=INSTRUMENT_MAP.keys(),
        help="Target instrument name",
    )
    parser.add_argument(
        "--output",
        help="Output MIDI file path (default: input_filename-<instrument>.mid)",
    )
    parser.add_argument(
        "--timbre",
        help="Path to timbre feature JSON file (optional, for erhu enhancement)",
    )

    args = parser.parse_args()

    if not args.output:
        input_dir, input_filename = os.path.split(args.input)
        filename_base, _ = os.path.splitext(input_filename)
        args.output = os.path.join(input_dir, f"{filename_base}-{args.instrument}.mid")

    change_midi_instrument(args.input, args.instrument, args.output, args.timbre)
