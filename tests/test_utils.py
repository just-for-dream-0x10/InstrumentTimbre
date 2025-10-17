#!/usr/bin/env python3
"""
Test suite for utility functions
工具函数测试套件
"""

import pytest
import numpy as np
import librosa
import os
import tempfile
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestAudioProcessing:
    """Test audio processing utilities"""
    
    def test_librosa_installation(self):
        """Test librosa installation and basic functions"""
        assert librosa.__version__ is not None
        print(f"Librosa version: {librosa.__version__}")
    
    def test_audio_loading(self):
        """Test audio loading with synthetic data"""
        # Create temporary audio file
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # A4 note
        
        # Save to temporary wav file
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, audio, sr)
            
            # Load with librosa
            loaded_audio, loaded_sr = librosa.load(tmp_file.name, sr=sr)
            
            assert loaded_sr == sr
            assert len(loaded_audio) > 0
            
            # Cleanup
            os.unlink(tmp_file.name)
    
    def test_stft_computation(self):
        """Test STFT computation"""
        # Generate test audio
        sr = 22050
        audio = np.random.randn(sr)  # 1 second of noise
        
        # Compute STFT
        stft = librosa.stft(audio)
        
        assert stft.dtype == np.complex64 or stft.dtype == np.complex128
        assert stft.shape[1] > 0  # Should have time frames
        assert stft.shape[0] > 0  # Should have frequency bins
    
    def test_f0_extraction(self):
        """Test F0 extraction with known frequency"""
        # Generate pure tone
        sr = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        frequency = 440.0  # A4
        audio = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        # Extract F0
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'), 
            sr=sr
        )
        
        # Check that we detected something close to the input frequency
        valid_f0 = f0[~np.isnan(f0)]
        if len(valid_f0) > 0:
            median_f0 = np.median(valid_f0)
            # Should be within 5% of target frequency
            assert abs(median_f0 - frequency) / frequency < 0.05
    
    def test_mfcc_extraction(self):
        """Test MFCC feature extraction"""
        # Generate test audio
        sr = 22050
        audio = np.random.randn(sr)
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        
        assert mfccs.shape[0] == 13  # 13 MFCC coefficients
        assert mfccs.shape[1] > 0   # Some time frames
    
    def test_spectral_features(self):
        """Test spectral feature extraction"""
        # Generate test audio
        sr = 22050
        audio = np.random.randn(sr)
        
        # Extract spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        
        assert len(spectral_centroids) > 0
        assert len(spectral_rolloff) > 0
        assert len(spectral_bandwidth) > 0
        assert len(zcr) > 0
        
        # Values should be reasonable
        assert np.all(spectral_centroids > 0)
        assert np.all(spectral_rolloff > 0)
        assert np.all(spectral_bandwidth >= 0)
        assert np.all((zcr >= 0) & (zcr <= 1))


class TestFileOperations:
    """Test file operations and data handling"""
    
    def test_audio_file_discovery(self):
        """Test audio file discovery"""
        # Test with example directory if it exists
        if os.path.exists("example"):
            audio_extensions = {'.wav', '.mp3', '.flac', '.m4a'}
            audio_files = []
            
            for file_path in Path("example").rglob('*'):
                if file_path.suffix.lower() in audio_extensions:
                    audio_files.append(str(file_path))
            
            print(f"Found {len(audio_files)} audio files in example/")
            
            for audio_file in audio_files[:3]:  # Check first 3
                assert os.path.exists(audio_file)
                print(f"  - {os.path.basename(audio_file)}")
    
    def test_directory_creation(self):
        """Test directory creation"""
        test_dir = "test_temp_dir"
        
        # Create directory
        os.makedirs(test_dir, exist_ok=True)
        assert os.path.exists(test_dir)
        
        # Cleanup
        os.rmdir(test_dir)
    
    def test_file_path_operations(self):
        """Test file path operations"""
        test_path = "some/path/to/audio_file.wav"
        
        # Test pathlib operations
        path_obj = Path(test_path)
        
        assert path_obj.name == "audio_file.wav"
        assert path_obj.stem == "audio_file"
        assert path_obj.suffix == ".wav"
        assert str(path_obj.parent) == "some/path/to"


class TestDataValidation:
    """Test data validation utilities"""
    
    def test_audio_data_validation(self):
        """Test audio data validation"""
        # Valid audio
        valid_audio = np.random.randn(22050)  # 1 second
        sr = 22050
        
        # Test validation logic
        assert len(valid_audio) > 0
        assert sr > 0
        assert len(valid_audio) >= sr / 10  # At least 0.1 seconds
        
        # Invalid audio (too short)
        invalid_audio = np.random.randn(100)  # Very short
        assert len(invalid_audio) < sr / 10
        
        # Empty audio
        empty_audio = np.array([])
        assert len(empty_audio) == 0
    
    def test_feature_vector_validation(self):
        """Test feature vector validation"""
        # Valid feature vector
        valid_features = np.random.randn(50)
        
        assert len(valid_features) == 50
        assert np.all(np.isfinite(valid_features))  # No NaN or inf
        
        # Invalid feature vector (contains NaN)
        invalid_features = valid_features.copy()
        invalid_features[10] = np.nan
        
        assert not np.all(np.isfinite(invalid_features))
        
        # Fix NaN values
        fixed_features = np.nan_to_num(invalid_features, nan=0.0)
        assert np.all(np.isfinite(fixed_features))
    
    def test_parameter_range_validation(self):
        """Test parameter range validation"""
        # Test sliding threshold
        sliding_threshold = 20  # cents per frame
        assert 0 < sliding_threshold < 100  # Reasonable range
        
        # Test vibrato rate range
        vibrato_rate = 5.0  # Hz
        assert 2.0 <= vibrato_rate <= 15.0  # Typical range
        
        # Test pentatonic adherence
        pentatonic_adherence = 0.75
        assert 0.0 <= pentatonic_adherence <= 1.0  # Must be probability


class TestNumericalOperations:
    """Test numerical operations and computations"""
    
    def test_frequency_conversions(self):
        """Test frequency conversion utilities"""
        # Test Hz to MIDI
        a4_hz = 440.0
        a4_midi = librosa.hz_to_midi(a4_hz)
        assert abs(a4_midi - 69.0) < 0.1  # A4 is MIDI note 69
        
        # Test MIDI to Hz
        midi_69 = 69.0
        hz_a4 = librosa.midi_to_hz(midi_69)
        assert abs(hz_a4 - 440.0) < 0.1
        
        # Test note name conversions
        c4_hz = librosa.note_to_hz('C4')
        assert abs(c4_hz - 261.63) < 1.0  # C4 ≈ 261.63 Hz
    
    def test_cents_calculations(self):
        """Test cents calculations"""
        # Convert frequency ratio to cents
        f1 = 440.0  # A4
        f2 = 466.16  # A#4 (one semitone up)
        
        cents = 1200 * np.log2(f2 / f1)
        assert abs(cents - 100.0) < 1.0  # One semitone = 100 cents
        
        # Test cents to frequency ratio
        cents_100 = 100.0
        ratio = 2 ** (cents_100 / 1200.0)
        assert abs(ratio - (f2/f1)) < 0.01
    
    def test_signal_processing_ops(self):
        """Test signal processing operations"""
        # Test gradient calculation
        signal = np.array([1, 2, 4, 7, 11, 16])  # Quadratic-like sequence
        gradient = np.gradient(signal)
        
        assert len(gradient) == len(signal)
        assert np.all(gradient > 0)  # Should be increasing
        
        # Test smoothing with simple moving average
        noisy_signal = np.random.randn(100) + np.sin(np.linspace(0, 4*np.pi, 100))
        window_size = 5
        
        # Simple moving average
        smoothed = np.convolve(noisy_signal, np.ones(window_size)/window_size, mode='same')
        
        assert len(smoothed) == len(noisy_signal)
        # Smoothed signal should have less variance
        assert np.var(smoothed) < np.var(noisy_signal)
    
    def test_statistical_operations(self):
        """Test statistical operations"""
        data = np.random.randn(1000)
        
        # Basic statistics
        mean = np.mean(data)
        std = np.std(data)
        var = np.var(data)
        
        assert abs(mean) < 0.1  # Should be close to 0 for large sample
        assert abs(std - 1.0) < 0.1  # Should be close to 1
        assert abs(var - std**2) < 1e-10  # Variance = std^2
        
        # Percentiles
        median = np.median(data)
        q25 = np.percentile(data, 25)
        q75 = np.percentile(data, 75)
        
        assert q25 < median < q75


if __name__ == "__main__":
    pytest.main([__file__, "-v"])