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
    
    def create_comprehensive_analysis(self, audio_data: np.ndarray, sample_rate: int,
                                    filename: str = "audio", 
                                    save_dir: Optional[str] = None) -> Dict[str, plt.Figure]:
        """
        Create comprehensive audio analysis with multiple plots
        
        Args:
            audio_data: Audio signal
            sample_rate: Sample rate
            filename: Base filename for plots
            save_dir: Directory to save plots
            
        Returns:
            Dictionary of matplotlib Figures
        """
        figures = {}
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)
        
        # Waveform
        save_path = str(save_dir / f"{filename}_waveform.{self.save_format}") if save_dir else None
        figures['waveform'] = self.plot_waveform(
            audio_data, sample_rate, f"{filename} - Waveform", save_path
        )
        
        # Spectrogram
        save_path = str(save_dir / f"{filename}_spectrogram.{self.save_format}") if save_dir else None
        figures['spectrogram'] = self.plot_spectrogram(
            audio_data, sample_rate, f"{filename} - Spectrogram", save_path
        )
        
        # MFCC
        save_path = str(save_dir / f"{filename}_mfcc.{self.save_format}") if save_dir else None
        figures['mfcc'] = self.plot_mfcc(
            audio_data, sample_rate, title=f"{filename} - MFCC", save_path=save_path
        )
        
        # Chroma
        save_path = str(save_dir / f"{filename}_chroma.{self.save_format}") if save_dir else None
        figures['chroma'] = self.plot_chroma(
            audio_data, sample_rate, f"{filename} - Chroma", save_path
        )
        
        # F0 Analysis
        save_path = str(save_dir / f"{filename}_f0.{self.save_format}") if save_dir else None
        figures['f0'] = self.plot_f0_analysis(
            audio_data, sample_rate, f"{filename} - F0 Analysis", save_path
        )
        
        self.logger.info(f"Created {len(figures)} analysis plots for {filename}")
        
        return figures