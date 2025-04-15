"""
Audio Processing Module

Provides audio loading, saving, feature extraction, and effects chain utilities for timbre analysis and conversion tasks.
"""
import os
import numpy as np
import torch
import librosa
import soundfile as sf
from pedalboard import Pedalboard
from pedalboard.io import AudioFile
import warnings


def load_audio(file_path, sr=22050, mono=True, duration=None, offset=0.0):
    """
    Load audio file with consistent parameters

    Args:
        file_path: Path to audio file
        sr: Sample rate
        mono: Convert to mono
        duration: Maximum duration to load
        offset: Start time offset

    Returns:
        audio: Audio data
        sr: Sample rate
    """
    try:
        audio, sr = librosa.load(
            file_path, sr=sr, mono=mono, duration=duration, offset=offset
        )
        return audio, sr
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None, None


def save_audio(audio, file_path, sr=22050):
    """
    Save audio file using soundfile (replacing deprecated librosa.output.write_wav)

    Args:
        audio: Audio data
        file_path: Output file path
        sr: Sample rate
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

    try:
        sf.write(file_path, audio, sr)
        return True
    except Exception as e:
        print(f"Error saving audio file {file_path}: {e}")
        return False


def extract_features(
    audio, sr, feature_type="mel", n_fft=2048, hop_length=512, n_mels=128
):
    """
    Extract audio features with multiple options

    Args:
        audio: Audio data
        sr: Sample rate
        feature_type: Type of feature (mel, stft, cqt, chroma, mfcc, multi)
        n_fft: FFT window size
        hop_length: Hop length
        n_mels: Number of mel bands

    Returns:
        features: Extracted features
    """
    if feature_type == "stft":
        # STFT
        return np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))

    elif feature_type == "mel":
        # Mel spectrogram
        return librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )

    elif feature_type == "log_mel":
        # Log-mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        return librosa.power_to_db(mel_spec, ref=np.max)

    elif feature_type == "cqt":
        # Constant-Q transform
        return np.abs(librosa.cqt(audio, sr=sr, hop_length=hop_length))

    elif feature_type == "chroma":
        # Chromagram
        return librosa.feature.chroma_stft(
            y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length
        )

    elif feature_type == "mfcc":
        # MFCCs
        return librosa.feature.mfcc(
            y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, n_mfcc=20
        )

    elif feature_type == "multi":
        # Multiple features concatenated
        mel = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        chroma = librosa.feature.chroma_stft(
            y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length
        )
        mfcc = librosa.feature.mfcc(
            y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, n_mfcc=20
        )

        # Resize to match dimensions
        if chroma.shape[1] < mel.shape[1]:
            chroma = np.pad(chroma, ((0, 0), (0, mel.shape[1] - chroma.shape[1])))
        elif chroma.shape[1] > mel.shape[1]:
            chroma = chroma[:, : mel.shape[1]]

        if mfcc.shape[1] < mel.shape[1]:
            mfcc = np.pad(mfcc, ((0, 0), (0, mel.shape[1] - mfcc.shape[1])))
        elif mfcc.shape[1] > mel.shape[1]:
            mfcc = mfcc[:, : mel.shape[1]]

        return {"mel": mel, "chroma": chroma, "mfcc": mfcc}

    else:
        warnings.warn(f"Unknown feature type: {feature_type}, using mel spectrogram")
        return librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )


def extract_chinese_instrument_features(audio, sr, instrument_category=None):
    """
    Extract specialized features for Chinese traditional instruments

    Args:
        audio: Audio data
        sr: Sample rate
        instrument_category: Category of Chinese instrument (bowed string, plucked string, wind, percussion)

    Returns:
        features: Dictionary of extracted features
    """
    # Basic spectral features
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=2048, hop_length=512, n_mels=128
    )

    # Extract pitch contour (important for 弓弦类 and 吹管类)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
    )

    # Extract onset strengths (important for 弹拨类 and 打击类)
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)

    # Harmonic and percussive components (useful for all categories)
    harmonic, percussive = librosa.effects.hpss(audio)

    # Category-specific features
    if instrument_category == "弓弦类":  # Bowed strings
        # Enhanced pitch contour features
        pitch_contour = f0.copy()
        # Fill NaN values with previous values for a smoother contour
        pitch_contour[np.isnan(pitch_contour)] = 0

        # Calculate pitch variability features (important for sliding tones)
        pitch_delta = np.diff(pitch_contour, append=0)
        pitch_delta_mean = np.mean(np.abs(pitch_delta))
        pitch_delta_var = np.var(pitch_delta)

        # Spectral contrast for timbre characteristics
        contrast = librosa.feature.spectral_contrast(y=harmonic, sr=sr)

        return {
            "mel_spectrogram": mel_spec,
            "pitch_contour": pitch_contour,
            "pitch_delta": pitch_delta,
            "pitch_delta_stats": np.array([pitch_delta_mean, pitch_delta_var]),
            "spectral_contrast": contrast,
            "harmonic_component": librosa.feature.melspectrogram(y=harmonic, sr=sr),
        }

    elif instrument_category == "弹拨类":  # Plucked strings
        # Onset detection is critical
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        # Attack characteristics
        if len(onset_frames) > 0:
            onset_strengths = onset_env[onset_frames]
        else:
            onset_strengths = np.array([0])

        # Spectral flatness for string resonance
        flatness = librosa.feature.spectral_flatness(y=audio)

        return {
            "mel_spectrogram": mel_spec,
            "onset_strengths": onset_strengths,
            "onset_times": onset_times,
            "spectral_flatness": flatness,
            "percussive_component": librosa.feature.melspectrogram(y=percussive, sr=sr),
        }

    elif instrument_category == "吹管类":  # Wind instruments
        # Breath noise detection
        zero_crossings = librosa.feature.zero_crossing_rate(audio)

        # Formants and harmonics
        harmonics = librosa.effects.harmonic(audio)

        # Spectral bandwidth for breath control
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)

        return {
            "mel_spectrogram": mel_spec,
            "pitch_contour": f0,
            "voiced_flag": voiced_flag,
            "zero_crossings": zero_crossings,
            "spectral_bandwidth": bandwidth,
            "harmonic_component": librosa.feature.melspectrogram(y=harmonics, sr=sr),
        }

    elif instrument_category == "打击类":  # Percussion
        # Rhythmic features
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)

        # Spectral centroid for percussion character
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)

        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)

        return {
            "mel_spectrogram": mel_spec,
            "percussive_component": librosa.feature.melspectrogram(y=percussive, sr=sr),
            "tempogram": tempogram,
            "spectral_centroid": centroid,
            "spectral_rolloff": rolloff,
        }

    else:
        # General features for unknown category
        return {
            "mel_spectrogram": mel_spec,
            "pitch_contour": f0,
            "onset_strength": onset_env,
            "harmonic_component": librosa.feature.melspectrogram(y=harmonic, sr=sr),
            "percussive_component": librosa.feature.melspectrogram(y=percussive, sr=sr),
        }


def apply_audio_effects(audio, sr, effect_chain=None, params=None):
    """
    Apply audio effects using Pedalboard

    Args:
        audio: Audio data
        sr: Sample rate
        effect_chain: List of effects to apply or predefined chain type
        params: Parameters for effects

    Returns:
        processed_audio: Audio after applying effects
    """
    if params is None:
        params = {}

    # If effect_chain is a string, create a predefined chain
    if isinstance(effect_chain, str):
        board = create_effect_chain(effect_chain, params)
    elif isinstance(effect_chain, Pedalboard):
        board = effect_chain
    else:
        # Default empty chain
        board = Pedalboard()

    # Process audio with pedalboard
    processed_audio = board(audio, sr)

    return processed_audio


def create_effect_chain(chain_type, params=None):
    """
    Create a predefined effect chain

    Args:
        chain_type: Type of effect chain
        params: Parameters for effects

    Returns:
        board: Pedalboard with effects
    """
    from pedalboard import (
        Compressor,
        Gain,
        Reverb,
        Delay,
        Distortion,
        Chorus,
        Phaser,
        LadderFilter,
        PitchShift,
        Convolution,
    )

    if params is None:
        params = {}

    board = Pedalboard()

    if chain_type == "string_resonance":
        # Effect chain for string instrument resonance
        board.append(
            Compressor(
                threshold_db=params.get("threshold_db", -20),
                ratio=params.get("ratio", 4),
                attack_ms=params.get("attack_ms", 5),
                release_ms=params.get("release_ms", 100),
            )
        )
        board.append(Gain(gain_db=params.get("gain_db", 3)))
        board.append(
            Reverb(
                room_size=params.get("room_size", 0.4),
                damping=params.get("damping", 0.4),
                wet_level=params.get("wet_level", 0.3),
                dry_level=params.get("dry_level", 0.7),
            )
        )

    elif chain_type == "erhu_simulation":
        # Effect chain to simulate erhu-like qualities
        board.append(PitchShift(semitones=params.get("semitones", 0)))
        board.append(
            LadderFilter(
                mode=params.get("filter_mode", LadderFilter.Mode.HPF24),
                cutoff_hz=params.get("cutoff_hz", 300),
            )
        )
        board.append(
            Chorus(
                rate_hz=params.get("chorus_rate", 0.5),
                depth=params.get("chorus_depth", 0.2),
                mix=params.get("chorus_mix", 0.3),
            )
        )
        board.append(
            Reverb(
                room_size=params.get("room_size", 0.3),
                damping=params.get("damping", 0.5),
                wet_level=params.get("wet_level", 0.2),
                dry_level=params.get("dry_level", 0.8),
            )
        )

    elif chain_type == "plucked_string":
        # Effect chain for plucked string instruments
        board.append(
            Compressor(
                threshold_db=params.get("threshold_db", -25),
                ratio=params.get("ratio", 3),
                attack_ms=params.get("attack_ms", 1),
                release_ms=params.get("release_ms", 70),
            )
        )
        board.append(
            Delay(
                delay_seconds=params.get("delay_seconds", 0.05),
                feedback=params.get("feedback", 0.1),
                mix=params.get("delay_mix", 0.2),
            )
        )
        board.append(
            Reverb(
                room_size=params.get("room_size", 0.2),
                damping=params.get("damping", 0.7),
                wet_level=params.get("wet_level", 0.15),
                dry_level=params.get("dry_level", 0.85),
            )
        )

    elif chain_type == "wind_instrument":
        # Effect chain for wind instruments
        board.append(
            LadderFilter(
                mode=params.get("filter_mode", LadderFilter.Mode.LPF24),
                cutoff_hz=params.get("cutoff_hz", 2000),
            )
        )
        board.append(
            Chorus(
                rate_hz=params.get("chorus_rate", 0.3),
                depth=params.get("chorus_depth", 0.15),
                mix=params.get("chorus_mix", 0.25),
            )
        )
        board.append(
            Reverb(
                room_size=params.get("room_size", 0.4),
                damping=params.get("damping", 0.3),
                wet_level=params.get("wet_level", 0.25),
                dry_level=params.get("dry_level", 0.75),
            )
        )

    else:
        # Default chain with just compressor and reverb
        board.append(
            Compressor(
                threshold_db=params.get("threshold_db", -20),
                ratio=params.get("ratio", 2),
                attack_ms=params.get("attack_ms", 10),
                release_ms=params.get("release_ms", 100),
            )
        )
        board.append(
            Reverb(
                room_size=params.get("room_size", 0.3),
                damping=params.get("damping", 0.5),
                wet_level=params.get("wet_level", 0.2),
                dry_level=params.get("dry_level", 0.8),
            )
        )

    return board
