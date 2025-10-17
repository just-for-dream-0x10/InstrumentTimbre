"""
Audio visualization utilities for InstrumentTimbre
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import logging

class AudioVisualizer:
    """
    Visualizer for audio data and analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize audio visualizer
        
        Args:
            config: Visualization configuration
        """
        self.config = config or {}
        self.figure_size = self.config.get('figure_size', (12, 8))
        self.dpi = self.config.get('dpi', 100)
        self.save_format = self.config.get('save_format', 'png')
        self.logger = logging.getLogger(__name__)
        
        # Set matplotlib style
        plt.style.use(self.config.get('style', 'default'))
    
    def plot_waveform(self, audio_data: np.ndarray, sample_rate: int,
                     title: str = "Waveform", save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot audio waveform
        
        Args:
            audio_data: Audio signal
            sample_rate: Sample rate
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        time_axis = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
        ax.plot(time_axis, audio_data, linewidth=0.5)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format=self.save_format, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Waveform plot saved to: {save_path}")
        
        return fig
    
    def plot_spectrogram(self, audio_data: np.ndarray, sample_rate: int,
                        title: str = "Spectrogram", save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot spectrogram
        
        Args:
            audio_data: Audio signal
            sample_rate: Sample rate
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        # Compute spectrogram
        D = librosa.stft(audio_data)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        img = librosa.display.specshow(
            S_db, sr=sample_rate, x_axis='time', y_axis='hz', ax=ax
        )
        
        ax.set_title(title)
        plt.colorbar(img, ax=ax, format='%+2.0f dB')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format=self.save_format, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Spectrogram plot saved to: {save_path}")
        
        return fig
    
    def plot_mfcc(self, audio_data: np.ndarray, sample_rate: int,
                  n_mfcc: int = 13, title: str = "MFCC", 
                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot MFCC features
        
        Args:
            audio_data: Audio signal
            sample_rate: Sample rate
            n_mfcc: Number of MFCC coefficients
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        # Compute MFCC
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc)
        
        img = librosa.display.specshow(
            mfccs, sr=sample_rate, x_axis='time', ax=ax
        )
        
        ax.set_ylabel('MFCC Coefficients')
        ax.set_title(title)
        plt.colorbar(img, ax=ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format=self.save_format, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"MFCC plot saved to: {save_path}")
        
        return fig
    
    def plot_chroma(self, audio_data: np.ndarray, sample_rate: int,
                   title: str = "Chroma Features", save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot chroma features (important for Chinese music analysis)
        
        Args:
            audio_data: Audio signal
            sample_rate: Sample rate
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        # Compute chroma
        chroma = librosa.feature.chroma_cqt(y=audio_data, sr=sample_rate)
        
        img = librosa.display.specshow(
            chroma, sr=sample_rate, x_axis='time', y_axis='chroma', ax=ax
        )
        
        ax.set_title(title)
        plt.colorbar(img, ax=ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format=self.save_format, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Chroma plot saved to: {save_path}")
        
        return fig
    
    def plot_f0_analysis(self, audio_data: np.ndarray, sample_rate: int,
                        title: str = "F0 Analysis", save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot fundamental frequency analysis
        
        Args:
            audio_data: Audio signal
            sample_rate: Sample rate
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figure_size, dpi=self.dpi)
        
        # Extract F0
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_data, fmin=80, fmax=2000, sr=sample_rate
        )
        
        # Time axis
        times = librosa.frames_to_time(range(len(f0)), sr=sample_rate)
        
        # Plot F0
        ax1.plot(times, f0, 'o-', markersize=2, linewidth=1)
        ax1.set_ylabel('F0 (Hz)')
        ax1.set_title('Fundamental Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Plot voicing probability
        ax2.plot(times, voiced_probs, 'r-', linewidth=1)
        ax2.fill_between(times, voiced_probs, alpha=0.3)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Voicing Probability')
        ax2.set_title('Voicing Detection')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format=self.save_format, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"F0 analysis plot saved to: {save_path}")
        
        return fig
    
    def plot_spectral_features(self, audio_data: np.ndarray, sample_rate: int,
                              title: str = "Spectral Features Analysis", 
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comprehensive spectral features analysis
        
        Args:
            audio_data: Audio signal
            sample_rate: Sample rate
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=self.dpi)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. Waveform
        time_axis = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
        axes[0, 0].plot(time_axis, audio_data, linewidth=0.5, color='#1f77b4')
        axes[0, 0].set_title('Waveform', fontweight='bold')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Spectrogram
        D = librosa.stft(audio_data)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        img1 = librosa.display.specshow(S_db, sr=sample_rate, x_axis='time', y_axis='hz', ax=axes[0, 1])
        axes[0, 1].set_title('Spectrogram', fontweight='bold')
        plt.colorbar(img1, ax=axes[0, 1], format='%+2.0f dB')
        
        # 3. Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=128)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        img2 = librosa.display.specshow(mel_db, sr=sample_rate, x_axis='time', y_axis='mel', ax=axes[0, 2])
        axes[0, 2].set_title('Mel Spectrogram', fontweight='bold')
        plt.colorbar(img2, ax=axes[0, 2], format='%+2.0f dB')
        
        # 4. MFCC
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        img3 = librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time', ax=axes[1, 0])
        axes[1, 0].set_title('MFCC', fontweight='bold')
        axes[1, 0].set_ylabel('MFCC Coefficients')
        plt.colorbar(img3, ax=axes[1, 0])
        
        # 5. Chroma Features
        chroma = librosa.feature.chroma_cqt(y=audio_data, sr=sample_rate)
        img4 = librosa.display.specshow(chroma, sr=sample_rate, x_axis='time', y_axis='chroma', ax=axes[1, 1])
        axes[1, 1].set_title('Chroma Features', fontweight='bold')
        plt.colorbar(img4, ax=axes[1, 1])
        
        # 6. Spectral Centroid and Rolloff
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)[0]
        frames = range(len(spectral_centroids))
        t = librosa.frames_to_time(frames, sr=sample_rate)
        
        axes[1, 2].plot(t, spectral_centroids, label='Spectral Centroid', color='#ff7f0e', linewidth=2)
        axes[1, 2].plot(t, spectral_rolloff, label='Spectral Rolloff', color='#2ca02c', linewidth=2)
        axes[1, 2].set_title('Spectral Features', fontweight='bold')
        axes[1, 2].set_xlabel('Time (s)')
        axes[1, 2].set_ylabel('Hz')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format=self.save_format, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Spectral features plot saved to: {save_path}")
        
        return fig
    
    def plot_chinese_instrument_analysis(self, audio_data: np.ndarray, sample_rate: int,
                                       instrument_type: str = "Chinese Instrument",
                                       title: Optional[str] = None,
                                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot enhanced analysis specifically for Chinese instruments
        
        Args:
            audio_data: Audio signal
            sample_rate: Sample rate
            instrument_type: Type of Chinese instrument
            title: Custom title
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure
        """
        if title is None:
            title = f"{instrument_type} - Enhanced Audio Analysis"
            
        fig, axes = plt.subplots(3, 2, figsize=(16, 18), dpi=self.dpi)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. Waveform with envelope
        time_axis = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
        envelope = np.abs(librosa.util.frame(audio_data, frame_length=2048, hop_length=512))
        envelope = np.mean(envelope, axis=0)
        envelope_times = librosa.frames_to_time(range(len(envelope)), sr=sample_rate)
        
        axes[0, 0].plot(time_axis, audio_data, alpha=0.7, linewidth=0.5, color='#1f77b4', label='Waveform')
        axes[0, 0].plot(envelope_times, envelope, color='#ff7f0e', linewidth=2, label='Envelope')
        axes[0, 0].set_title('Waveform & Amplitude Envelope', fontweight='bold')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. F0 tracking and harmonics
        f0, voiced_flag, voiced_probs = librosa.pyin(audio_data, fmin=80, fmax=2000, sr=sample_rate)
        times = librosa.frames_to_time(range(len(f0)), sr=sample_rate)
        
        axes[0, 1].plot(times, f0, 'o-', markersize=3, linewidth=1, color='#d62728', label='F0')
        axes[0, 1].fill_between(times, 0, f0, where=(voiced_probs > 0.5), alpha=0.3, color='#d62728')
        axes[0, 1].set_title('Fundamental Frequency (F0) Tracking', fontweight='bold')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Frequency (Hz)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # 3. Harmonic-Percussive Separation
        harmonic, percussive = librosa.effects.hpss(audio_data)
        
        # Plot harmonic component spectrogram
        D_harmonic = librosa.stft(harmonic)
        S_harmonic_db = librosa.amplitude_to_db(np.abs(D_harmonic), ref=np.max)
        img1 = librosa.display.specshow(S_harmonic_db, sr=sample_rate, x_axis='time', y_axis='hz', ax=axes[1, 0])
        axes[1, 0].set_title('Harmonic Component', fontweight='bold')
        plt.colorbar(img1, ax=axes[1, 0], format='%+2.0f dB')
        
        # Plot percussive component spectrogram
        D_percussive = librosa.stft(percussive)
        S_percussive_db = librosa.amplitude_to_db(np.abs(D_percussive), ref=np.max)
        img2 = librosa.display.specshow(S_percussive_db, sr=sample_rate, x_axis='time', y_axis='hz', ax=axes[1, 1])
        axes[1, 1].set_title('Percussive Component', fontweight='bold')
        plt.colorbar(img2, ax=axes[1, 1], format='%+2.0f dB')
        
        # 4. Chinese pentatonic scale analysis
        chroma = librosa.feature.chroma_cqt(y=audio_data, sr=sample_rate)
        # Highlight pentatonic notes (C, D, E, G, A = 0, 2, 4, 7, 9)
        pentatonic_mask = np.zeros_like(chroma)
        pentatonic_notes = [0, 2, 4, 7, 9]  # C, D, E, G, A
        pentatonic_mask[pentatonic_notes, :] = chroma[pentatonic_notes, :]
        
        img3 = librosa.display.specshow(pentatonic_mask, sr=sample_rate, x_axis='time', y_axis='chroma', ax=axes[2, 0])
        axes[2, 0].set_title('Pentatonic Scale Emphasis', fontweight='bold')
        plt.colorbar(img3, ax=axes[2, 0])
        
        # 5. Spectral features over time
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)[0]
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        
        frames = range(len(spectral_centroids))
        t = librosa.frames_to_time(frames, sr=sample_rate)
        
        ax_twin = axes[2, 1].twinx()
        
        line1 = axes[2, 1].plot(t, spectral_centroids, label='Spectral Centroid', color='#ff7f0e', linewidth=2)
        line2 = axes[2, 1].plot(t, spectral_rolloff, label='Spectral Rolloff', color='#2ca02c', linewidth=2)
        line3 = ax_twin.plot(t, zcr, label='Zero Crossing Rate', color='#9467bd', linewidth=2)
        
        axes[2, 1].set_title('Spectral Features Evolution', fontweight='bold')
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].set_ylabel('Frequency (Hz)')
        ax_twin.set_ylabel('Zero Crossing Rate')
        
        # Combine legends
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        axes[2, 1].legend(lines, labels, loc='upper right')
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format=self.save_format, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Chinese instrument analysis plot saved to: {save_path}")
        
        return fig
    
    def create_comprehensive_analysis(self, audio_data: np.ndarray, sample_rate: int,
                                    filename: str = "audio", 
                                    save_dir: Optional[str] = None,
                                    instrument_type: Optional[str] = None) -> Dict[str, plt.Figure]:
        """
        Create comprehensive audio analysis with multiple plots
        
        Args:
            audio_data: Audio signal
            sample_rate: Sample rate
            filename: Base filename for plots
            save_dir: Directory to save plots
            instrument_type: Type of instrument for specialized analysis
            
        Returns:
            Dictionary of matplotlib Figures
        """
        figures = {}
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)
        
        # Basic spectral features analysis
        save_path = str(save_dir / f"{filename}_spectral_features.{self.save_format}") if save_dir else None
        figures['spectral_features'] = self.plot_spectral_features(
            audio_data, sample_rate, f"{filename} - Spectral Features Analysis", save_path
        )
        
        # Chinese instrument enhanced analysis
        if instrument_type or 'chinese' in filename.lower():
            inst_type = instrument_type or filename
            save_path = str(save_dir / f"{filename}_enhanced_analysis.{self.save_format}") if save_dir else None
            figures['enhanced_analysis'] = self.plot_chinese_instrument_analysis(
                audio_data, sample_rate, inst_type, save_path=save_path
            )
        
        # Individual component plots
        save_path = str(save_dir / f"{filename}_waveform.{self.save_format}") if save_dir else None
        figures['waveform'] = self.plot_waveform(
            audio_data, sample_rate, f"{filename} - Waveform", save_path
        )
        
        save_path = str(save_dir / f"{filename}_spectrogram.{self.save_format}") if save_dir else None
        figures['spectrogram'] = self.plot_spectrogram(
            audio_data, sample_rate, f"{filename} - Spectrogram", save_path
        )
        
        save_path = str(save_dir / f"{filename}_mfcc.{self.save_format}") if save_dir else None
        figures['mfcc'] = self.plot_mfcc(
            audio_data, sample_rate, title=f"{filename} - MFCC", save_path=save_path
        )
        
        save_path = str(save_dir / f"{filename}_chroma.{self.save_format}") if save_dir else None
        figures['chroma'] = self.plot_chroma(
            audio_data, sample_rate, f"{filename} - Chroma", save_path
        )
        
        save_path = str(save_dir / f"{filename}_f0.{self.save_format}") if save_dir else None
        figures['f0'] = self.plot_f0_analysis(
            audio_data, sample_rate, f"{filename} - F0 Analysis", save_path
        )
        
        self.logger.info(f"Created {len(figures)} analysis plots for {filename}")
        
        return figures