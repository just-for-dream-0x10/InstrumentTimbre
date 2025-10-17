#!/usr/bin/env python3
"""
Enhanced Chinese Instrument Visualization Module
Integrated into InstrumentTimbre project with improved font handling
"""

import os
import sys
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
from scipy import signal
from pathlib import Path

# Add project path
sys.path.append('..')
try:
    from InstrumentTimbre.modules.utils.chinese_instrument_features import ChineseInstrumentAnalyzer
    from InstrumentTimbre.modules.core.models import InstrumentType
except ImportError:
    print("Warning: Could not import enhanced Chinese features, using fallback")
    ChineseInstrumentAnalyzer = None
    InstrumentType = None

# Fix font issues for better cross-platform compatibility
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
# Avoid Chinese characters to prevent encoding issues
plt.rcParams['font.size'] = 10

class EnhancedChineseInstrumentVisualizer:
    """Enhanced Chinese Instrument Visualizer with improved font handling"""
    
    def __init__(self):
        self.wu_sheng_scale = {0, 2, 4, 7, 9}  # Pentatonic scale
        self.instrument_names = {
            'erhu': 'Erhu (Er Hu)',
            'pipa': 'Pipa (Pi Pa)', 
            'guzheng': 'Guzheng (Gu Zheng)',
            'dizi': 'Dizi (Di Zi)',
            'guqin': 'Guqin (Gu Qin)',
            'xiao': 'Xiao',
            'suona': 'Suona'
        }
        
    def extract_chinese_features(self, audio_data, sr):
        """Extract Chinese instrument features with fallback"""
        features = {}
        
        # Extract F0 using librosa.pyin
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio_data, 
                fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C7'), 
                sr=sr
            )
        except Exception as e:
            print(f"Warning: F0 extraction failed: {e}")
            f0 = np.full(len(audio_data) // 512, np.nan)
        
        valid_f0 = f0[~np.isnan(f0)]
        
        if len(valid_f0) > 0:
            # Basic F0 features
            features['f0_mean'] = np.mean(valid_f0)
            features['f0_std'] = np.std(valid_f0)
            features['f0_range'] = np.max(valid_f0) - np.min(valid_f0)
            
            # Pentatonic adherence (Wu Sheng scale)
            midi_notes = librosa.hz_to_midi(valid_f0)
            semitones = midi_notes % 12
            
            pentatonic_count = 0
            for note in semitones:
                if any(abs(note - pent) < 0.5 for pent in self.wu_sheng_scale):
                    pentatonic_count += 1
            
            features['pentatonic_adherence'] = pentatonic_count / len(semitones)
            
            # Sliding analysis (Hua Yin)
            if len(valid_f0) > 10:
                log_f0 = np.log2(valid_f0 + 1e-10) * 1200
                velocity = np.gradient(log_f0)
                
                if len(velocity) >= 5:
                    try:
                        velocity_smooth = signal.savgol_filter(velocity, 5, 2)
                    except:
                        velocity_smooth = velocity
                else:
                    velocity_smooth = velocity
                
                sliding_threshold = 20  # cents per frame
                sliding_mask = np.abs(velocity_smooth) > sliding_threshold
                
                features['sliding_presence'] = np.sum(sliding_mask) / len(sliding_mask)
                features['sliding_velocity_mean'] = np.mean(np.abs(velocity_smooth))
                features['sliding_velocity_max'] = np.max(np.abs(velocity_smooth))
            else:
                features['sliding_presence'] = 0.0
                features['sliding_velocity_mean'] = 0.0
                features['sliding_velocity_max'] = 0.0
            
            # Vibrato analysis (Chan Yin)
            if len(valid_f0) > 50:
                log_f0 = np.log2(valid_f0 + 1e-10) * 1200
                detrended = signal.detrend(log_f0)
                
                # FFT analysis for vibrato
                hop_length = 512
                time_per_frame = hop_length / sr
                
                fft = np.fft.fft(detrended)
                freqs = np.fft.fftfreq(len(detrended), d=time_per_frame)
                
                # Focus on vibrato frequency range (2-15 Hz)
                vibrato_mask = (freqs >= 2.0) & (freqs <= 15.0)
                
                if np.any(vibrato_mask):
                    vibrato_spectrum = np.abs(fft[vibrato_mask])
                    
                    if len(vibrato_spectrum) > 0:
                        peak_idx = np.argmax(vibrato_spectrum)
                        vibrato_freq = freqs[vibrato_mask][peak_idx]
                        
                        features['vibrato_rate'] = vibrato_freq if vibrato_freq > 0 else 0.0
                        features['vibrato_extent'] = np.std(detrended)
                        
                        # Regularity
                        mean_energy = np.mean(vibrato_spectrum)
                        peak_energy = np.max(vibrato_spectrum)
                        features['vibrato_regularity'] = peak_energy / (mean_energy + 1e-10)
                    else:
                        features['vibrato_rate'] = 0.0
                        features['vibrato_extent'] = 0.0
                        features['vibrato_regularity'] = 0.0
                else:
                    features['vibrato_rate'] = 0.0
                    features['vibrato_extent'] = 0.0
                    features['vibrato_regularity'] = 0.0
            else:
                features['vibrato_rate'] = 0.0
                features['vibrato_extent'] = 0.0
                features['vibrato_regularity'] = 0.0
            
            # Ornament density (Zhuang Shi Yin)
            if len(valid_f0) > 5:
                midi_notes = librosa.hz_to_midi(valid_f0)
                pitch_diff = np.abs(np.diff(midi_notes))
                
                ornament_threshold = 0.5  # semitones
                ornaments = np.sum(pitch_diff > ornament_threshold)
                features['ornament_density'] = ornaments / len(pitch_diff)
            else:
                features['ornament_density'] = 0.0
        else:
            # Default values when no valid F0
            features.update({
                'f0_mean': 0, 'f0_std': 0, 'f0_range': 0,
                'pentatonic_adherence': 0, 'sliding_presence': 0,
                'sliding_velocity_mean': 0, 'sliding_velocity_max': 0,
                'vibrato_rate': 0, 'vibrato_extent': 0, 'vibrato_regularity': 0,
                'ornament_density': 0
            })
        
        return features, f0
    
    def detect_instrument_type(self, filename):
        """Detect instrument type from filename"""
        filename = filename.lower()
        for key, name in self.instrument_names.items():
            if key in filename:
                return name
        return 'Chinese Traditional Instrument'
    
    def create_comprehensive_visualization(self, audio_file, output_dir="visualizations"):
        """Create comprehensive visualization with fixed encoding"""
        print(f"Creating visualization for: {os.path.basename(audio_file)}")
        
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load audio
        try:
            audio_data, sr = librosa.load(audio_file, sr=22050)
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return None
            
        duration = len(audio_data) / sr
        
        # Extract enhanced Chinese features
        chinese_features, f0 = self.extract_chinese_features(audio_data, sr)
        
        # Detect instrument type
        instrument_name = self.detect_instrument_type(os.path.basename(audio_file))
        
        # Create visualization
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        fig.suptitle(f'Enhanced Chinese Instrument Analysis - {instrument_name}\n{os.path.basename(audio_file)}', 
                    fontsize=16, fontweight='bold')
        
        # 1. Waveform
        time = np.linspace(0, duration, len(audio_data))
        axes[0,0].plot(time, audio_data, alpha=0.7, color='blue')
        axes[0,0].set_title('Audio Waveform')
        axes[0,0].set_xlabel('Time (s)')
        axes[0,0].set_ylabel('Amplitude')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Spectrogram
        try:
            D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
            librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr, ax=axes[0,1])
            axes[0,1].set_title('Spectrogram')
            axes[0,1].set_ylim(0, 4000)
        except Exception as e:
            axes[0,1].text(0.5, 0.5, f'Spectrogram error: {str(e)[:50]}', 
                          ha='center', va='center', transform=axes[0,1].transAxes)
            axes[0,1].set_title('Spectrogram (Error)')
        
        # 3. F0 contour
        hop_length = 512
        f0_times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
        axes[0,2].plot(f0_times, f0, 'g-', alpha=0.8, linewidth=1.5)
        axes[0,2].set_title(f'F0 Contour (Mean: {chinese_features["f0_mean"]:.1f} Hz)')
        axes[0,2].set_xlabel('Time (s)')
        axes[0,2].set_ylabel('Frequency (Hz)')
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Sliding analysis (Hua Yin)
        valid_f0 = f0[~np.isnan(f0)]
        if len(valid_f0) > 10:
            log_f0 = np.log2(valid_f0 + 1e-10) * 1200
            velocity = np.gradient(log_f0)
            if len(velocity) >= 5:
                try:
                    velocity_smooth = signal.savgol_filter(velocity, 5, 2)
                except:
                    velocity_smooth = velocity
            else:
                velocity_smooth = velocity
            
            valid_times = f0_times[~np.isnan(f0)]
            axes[1,0].plot(valid_times, velocity_smooth, 'purple', alpha=0.7, linewidth=1)
            axes[1,0].set_title(f'Sliding Velocity (Hua Yin)\nPresence: {chinese_features["sliding_presence"]:.1%}')
            axes[1,0].set_xlabel('Time (s)')
            axes[1,0].set_ylabel('Cents/frame')
            axes[1,0].grid(True, alpha=0.3)
            
            # Mark sliding regions
            sliding_threshold = 20
            sliding_mask = np.abs(velocity_smooth) > sliding_threshold
            if np.any(sliding_mask):
                axes[1,0].fill_between(valid_times, 0, velocity_smooth, 
                                     where=sliding_mask, alpha=0.3, color='red', label='Sliding')
                axes[1,0].legend()
        else:
            axes[1,0].text(0.5, 0.5, 'Insufficient F0 data for sliding analysis', 
                          ha='center', va='center', transform=axes[1,0].transAxes)
            axes[1,0].set_title('Sliding Analysis (Insufficient Data)')
        
        # 5. Vibrato analysis (Chan Yin)
        if len(valid_f0) > 50:
            log_f0 = np.log2(valid_f0 + 1e-10) * 1200
            detrended = signal.detrend(log_f0)
            valid_times = f0_times[~np.isnan(f0)]
            axes[1,1].plot(valid_times, detrended, 'orange', alpha=0.7, linewidth=1)
            axes[1,1].set_title(f'Vibrato Pattern (Chan Yin)\nRate: {chinese_features["vibrato_rate"]:.1f} Hz')
            axes[1,1].set_xlabel('Time (s)')
            axes[1,1].set_ylabel('Detrended Cents')
            axes[1,1].grid(True, alpha=0.3)
        else:
            axes[1,1].text(0.5, 0.5, 'Insufficient F0 data for vibrato analysis', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Vibrato Analysis (Insufficient Data)')
        
        # 6. Feature radar chart
        categories = ['Pentatonic\nAdherence', 'Sliding\nPresence', 'Vibrato\nRate', 
                     'Ornament\nDensity', 'F0\nStability', 'Spectral\nRichness']
        
        values = [
            chinese_features['pentatonic_adherence'],
            chinese_features['sliding_presence'],
            min(1.0, chinese_features['vibrato_rate'] / 10.0),  # Normalize to 0-1
            chinese_features['ornament_density'],
            min(1.0, 1.0 / (chinese_features['f0_std'] + 1e-10) / 100),  # F0 stability
            0.5  # Placeholder for spectral richness
        ]
        
        # Normalize values to 0-1 range
        values = [min(1.0, max(0.0, v)) for v in values]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # Close the polygon
        angles += angles[:1]
        
        ax_radar = plt.subplot(1, 3, 3, projection='polar')
        ax_radar.plot(angles, values, 'o-', linewidth=2, color='red', alpha=0.7)
        ax_radar.fill(angles, values, alpha=0.25, color='red')
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(categories, fontsize=8)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('Feature Radar Chart', pad=20)
        ax_radar.grid(True)
        
        # Position the radar chart
        pos = axes[1,2].get_position()
        ax_radar.set_position([pos.x0, pos.y0, pos.width, pos.height])
        axes[1,2].remove()
        
        # 7. MFCC heatmap
        try:
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
            sns.heatmap(mfccs, ax=axes[2,0], cmap='viridis', cbar_kws={'label': 'MFCC'})
            axes[2,0].set_title('MFCC Features')
            axes[2,0].set_xlabel('Time Frames')
            axes[2,0].set_ylabel('MFCC Coefficients')
        except Exception as e:
            axes[2,0].text(0.5, 0.5, f'MFCC error: {str(e)[:50]}', 
                          ha='center', va='center', transform=axes[2,0].transAxes)
            axes[2,0].set_title('MFCC Features (Error)')
        
        # 8. Spectral features
        try:
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0]
            
            frames = range(len(spectral_centroids))
            times_spectral = librosa.frames_to_time(frames, sr=sr)
            
            axes[2,1].plot(times_spectral, spectral_centroids, label='Spectral Centroid', alpha=0.7)
            axes[2,1].plot(times_spectral, spectral_rolloff, label='Spectral Rolloff', alpha=0.7)
            axes[2,1].set_title('Spectral Features')
            axes[2,1].set_xlabel('Time (s)')
            axes[2,1].set_ylabel('Frequency (Hz)')
            axes[2,1].legend()
            axes[2,1].grid(True, alpha=0.3)
        except Exception as e:
            axes[2,1].text(0.5, 0.5, f'Spectral error: {str(e)[:50]}', 
                          ha='center', va='center', transform=axes[2,1].transAxes)
            axes[2,1].set_title('Spectral Features (Error)')
        
        # 9. Feature summary
        axes[2,2].axis('off')
        
        summary_text = f"""Enhanced Chinese Instrument Features:

Pentatonic Adherence: {chinese_features['pentatonic_adherence']:.3f}
(Wu Sheng scale conformity)

Sliding Presence: {chinese_features['sliding_presence']:.3f}
• Avg Velocity: {chinese_features['sliding_velocity_mean']:.1f} cents/frame
• Max Velocity: {chinese_features['sliding_velocity_max']:.1f} cents/frame

Vibrato Features (Chan Yin):
• Rate: {chinese_features['vibrato_rate']:.1f} Hz
• Extent: {chinese_features['vibrato_extent']:.1f} cents
• Regularity: {chinese_features['vibrato_regularity']:.1f}

Ornament Density: {chinese_features['ornament_density']:.3f}
(Zhuang Shi Yin)

F0 Statistics:
• Mean: {chinese_features['f0_mean']:.1f} Hz
• Std Dev: {chinese_features['f0_std']:.1f} Hz
• Range: {chinese_features['f0_range']:.1f} Hz

Audio Info:
• Duration: {duration:.1f} seconds
• Sample Rate: {sr} Hz

Instrument: {instrument_name}
        """
        
        axes[2,2].text(0.05, 0.95, summary_text, transform=axes[2,2].transAxes, 
                      fontsize=9, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        plt.tight_layout()
        
        # Save visualization
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}_enhanced_analysis.png")
        
        try:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Enhanced visualization saved: {output_file}")
        except Exception as e:
            print(f"Error saving visualization: {e}")
            output_file = None
        
        plt.close()
        
        return chinese_features, output_file
    
    def process_directory(self, input_dir, output_dir="visualizations"):
        """Process all audio files in a directory"""
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a'}
        audio_files = []
        
        for file_path in Path(input_dir).rglob('*'):
            if file_path.suffix.lower() in audio_extensions:
                audio_files.append(str(file_path))
        
        if not audio_files:
            print(f"No audio files found in {input_dir}")
            return []
        
        print(f"Found {len(audio_files)} audio files")
        
        results = []
        for audio_file in audio_files:
            try:
                features, output_file = self.create_comprehensive_visualization(audio_file, output_dir)
                if output_file:
                    results.append({
                        'audio_file': audio_file,
                        'features': features,
                        'visualization': output_file
                    })
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
        
        return results

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Chinese Instrument Visualization')
    parser.add_argument('--input', '-i', required=True, help='Input audio file or directory')
    parser.add_argument('--output', '-o', default='visualizations', help='Output directory')
    parser.add_argument('--recursive', '-r', action='store_true', help='Process directory recursively')
    
    args = parser.parse_args()
    
    visualizer = EnhancedChineseInstrumentVisualizer()
    
    if os.path.isfile(args.input):
        # Process single file
        features, output_file = visualizer.create_comprehensive_visualization(args.input, args.output)
        if output_file:
            print(f"Visualization completed: {output_file}")
        else:
            print("Visualization failed")
    elif os.path.isdir(args.input):
        # Process directory
        results = visualizer.process_directory(args.input, args.output)
        print(f"Processed {len(results)} files successfully")
        
        # Print summary
        if results:
            print("\nFeature Summary:")
            for result in results:
                features = result['features']
                filename = os.path.basename(result['audio_file'])
                print(f"{filename}:")
                print(f"  Pentatonic: {features['pentatonic_adherence']:.3f}")
                print(f"  Sliding: {features['sliding_presence']:.3f}")
                print(f"  Vibrato: {features['vibrato_rate']:.1f} Hz")
    else:
        print(f"Input path not found: {args.input}")

if __name__ == "__main__":
    main()