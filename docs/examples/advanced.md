# Advanced Examples

## Custom Feature Development

### Adding New Traditional Techniques

```python
def detect_custom_technique(f0, sr, technique_params):
    """Add detection for new traditional technique"""
    
    # Example: Detect "Dian Yin" (dotted notes) technique
    valid_f0 = f0[~np.isnan(f0)]
    if len(valid_f0) < 20:
        return 0.0
    
    # Convert to MIDI for semitone analysis
    midi_notes = librosa.hz_to_midi(valid_f0)
    
    # Detect rapid note changes characteristic of Dian Yin
    note_changes = np.abs(np.diff(midi_notes))
    rapid_changes = note_changes > technique_params['threshold']
    
    # Count clusters of rapid changes
    clusters = find_technique_clusters(rapid_changes, technique_params['min_cluster_size'])
    
    return len(clusters) / len(valid_f0) * 1000  # Normalize

# Usage
dian_yin_params = {'threshold': 2.0, 'min_cluster_size': 3}
dian_yin_presence = detect_custom_technique(f0, sr, dian_yin_params)
```

### Custom Instrument Analysis

```python
class CustomInstrumentAnalyzer(EnhancedChineseInstrumentVisualizer):
    """Extended analyzer for additional Chinese instruments"""
    
    def __init__(self):
        super().__init__()
        self.custom_instruments = {
            'yangqin': 'Yangqin (Yang Qin)',
            'ruan': 'Ruan',
            'sanxian': 'Sanxian (San Xian)'
        }
    
    def analyze_yangqin_features(self, audio_data, sr):
        """Specialized analysis for Yangqin (hammered dulcimer)"""
        features = {}
        
        # Yangqin-specific: detect tremolo patterns
        onset_envelope = librosa.onset.onset_strength(y=audio_data, sr=sr)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_envelope, sr=sr)
        
        # Analyze attack characteristics
        stft = librosa.stft(audio_data)
        attack_times = self.detect_hammer_attacks(stft, sr)
        
        features['tremolo_rate'] = self.calculate_tremolo_rate(attack_times)
        features['attack_sharpness'] = self.measure_attack_sharpness(stft, attack_times)
        features['resonance_decay'] = self.analyze_string_resonance(audio_data, sr)
        
        return features
```

## Multi-Instrument Ensemble Analysis

### Ensemble Feature Extraction

```python
def analyze_ensemble_performance(audio_file, instrument_roles):
    """Analyze Chinese traditional ensemble performance"""
    
    # Load and separate sources (if multi-channel or using source separation)
    audio_data, sr = librosa.load(audio_file, sr=22050)
    
    # Source separation (conceptual - requires advanced ML models)
    separated_sources = separate_instruments(audio_data, len(instrument_roles))
    
    ensemble_features = {
        'instruments': {},
        'ensemble_metrics': {}
    }
    
    # Analyze each instrument separately
    for i, (source, instrument) in enumerate(zip(separated_sources, instrument_roles)):
        analyzer = ChineseInstrumentAnalyzer()
        features = analyzer.extract_chinese_features(source, sr, instrument)
        ensemble_features['instruments'][instrument.value] = features
    
    # Ensemble-level analysis
    ensemble_features['ensemble_metrics'] = {
        'synchronization': calculate_ensemble_sync(separated_sources, sr),
        'harmonic_interaction': analyze_harmonic_relationships(separated_sources, sr),
        'dynamic_balance': measure_dynamic_balance(separated_sources),
        'traditional_form_adherence': analyze_traditional_form(ensemble_features['instruments'])
    }
    
    return ensemble_features

def calculate_ensemble_sync(sources, sr):
    """Calculate synchronization between ensemble members"""
    onset_times = []
    
    for source in sources:
        onsets = librosa.onset.onset_detect(y=source, sr=sr, units='time')
        onset_times.append(onsets)
    
    # Calculate synchronization score
    sync_score = 0.0
    tolerance = 0.05  # 50ms tolerance
    
    for i in range(len(onset_times)):
        for j in range(i+1, len(onset_times)):
            sync_score += calculate_onset_alignment(onset_times[i], onset_times[j], tolerance)
    
    return sync_score / (len(sources) * (len(sources) - 1) / 2)
```

## Cultural Style Classification

### Regional Style Detection

```python
class ChineseRegionalStyleClassifier:
    """Classify Chinese traditional music by regional styles"""
    
    def __init__(self):
        self.style_features = {
            'jiangnan': {  # Jiangnan (South of Yangtze River) style
                'ornament_density_range': (0.3, 0.7),
                'sliding_preference': 'subtle',
                'tempo_range': (60, 120),
                'pentatonic_adherence': (0.8, 1.0)
            },
            'shandong': {  # Shandong style
                'ornament_density_range': (0.1, 0.4),
                'sliding_preference': 'bold',
                'tempo_range': (80, 160),
                'pentatonic_adherence': (0.6, 0.9)
            },
            'guangdong': {  # Guangdong style
                'ornament_density_range': (0.4, 0.8),
                'sliding_preference': 'frequent',
                'tempo_range': (70, 140),
                'pentatonic_adherence': (0.7, 0.95)
            }
        }
    
    def classify_style(self, features):
        """Classify regional style based on extracted features"""
        scores = {}
        
        for style_name, style_params in self.style_features.items():
            score = self.calculate_style_score(features, style_params)
            scores[style_name] = score
        
        # Return most likely style
        best_style = max(scores.items(), key=lambda x: x[1])
        return {
            'predicted_style': best_style[0],
            'confidence': best_style[1],
            'all_scores': scores
        }
    
    def calculate_style_score(self, features, style_params):
        """Calculate how well features match a regional style"""
        score = 0.0
        total_weight = 0.0
        
        # Check ornament density
        if 'ornament_density' in features:
            ornament_range = style_params['ornament_density_range']
            if ornament_range[0] <= features['ornament_density'] <= ornament_range[1]:
                score += 0.3
            total_weight += 0.3
        
        # Check pentatonic adherence
        if 'pentatonic_adherence' in features:
            penta_range = style_params['pentatonic_adherence']
            if penta_range[0] <= features['pentatonic_adherence'] <= penta_range[1]:
                score += 0.4
            total_weight += 0.4
        
        # Check sliding characteristics
        sliding_score = self.evaluate_sliding_style(
            features.get('sliding_presence', 0),
            style_params['sliding_preference']
        )
        score += sliding_score * 0.3
        total_weight += 0.3
        
        return score / total_weight if total_weight > 0 else 0.0
```

## Performance Analysis Over Time

### Temporal Evolution Tracking

```python
def analyze_performance_evolution(audio_file, segment_length=10.0):
    """Analyze how performance characteristics evolve over time"""
    
    audio_data, sr = librosa.load(audio_file, sr=22050)
    segment_samples = int(segment_length * sr)
    
    evolution_data = {
        'time_segments': [],
        'features_over_time': [],
        'technique_usage': [],
        'emotional_trajectory': []
    }
    
    analyzer = EnhancedChineseInstrumentVisualizer()
    
    # Analyze each segment
    for start in range(0, len(audio_data), segment_samples):
        end = min(start + segment_samples, len(audio_data))
        segment = audio_data[start:end]
        
        if len(segment) < sr:  # Skip segments shorter than 1 second
            continue
        
        # Extract features for this segment
        features, f0 = analyzer.extract_chinese_features(segment, sr)
        
        # Time stamp
        time_point = start / sr
        evolution_data['time_segments'].append(time_point)
        evolution_data['features_over_time'].append(features)
        
        # Analyze technique usage patterns
        technique_usage = analyze_technique_density(features)
        evolution_data['technique_usage'].append(technique_usage)
        
        # Estimate emotional content (simplified)
        emotional_state = estimate_emotional_state(features)
        evolution_data['emotional_trajectory'].append(emotional_state)
    
    # Generate evolution report
    return generate_evolution_report(evolution_data)

def analyze_technique_density(features):
    """Analyze density of traditional techniques in a segment"""
    return {
        'sliding_density': features.get('sliding_presence', 0),
        'vibrato_activity': 1.0 if features.get('vibrato_rate', 0) > 1.0 else 0.0,
        'ornament_activity': features.get('ornament_density', 0),
        'overall_technique_score': (
            features.get('sliding_presence', 0) +
            (1.0 if features.get('vibrato_rate', 0) > 1.0 else 0.0) +
            features.get('ornament_density', 0)
        ) / 3.0
    }
```

## Cross-Cultural Analysis

### East-West Musical Comparison

```python
def compare_eastern_western_features(chinese_audio, western_audio):
    """Compare Chinese and Western musical characteristics"""
    
    # Analyze both audio files
    chinese_features, _ = analyzer.extract_chinese_features(*librosa.load(chinese_audio))
    western_features = extract_western_features(western_audio)  # Hypothetical function
    
    comparison = {
        'scale_systems': {
            'chinese_pentatonic_adherence': chinese_features['pentatonic_adherence'],
            'western_diatonic_adherence': western_features['diatonic_adherence'],
            'modal_difference': abs(chinese_features['pentatonic_adherence'] - 
                                   western_features['diatonic_adherence'])
        },
        'ornamentation': {
            'chinese_ornament_density': chinese_features['ornament_density'],
            'western_ornament_density': western_features['ornament_density'],
            'ornamentation_style_difference': calculate_ornamentation_difference(
                chinese_features, western_features
            )
        },
        'expressive_techniques': {
            'chinese_sliding_usage': chinese_features['sliding_presence'],
            'western_vibrato_usage': western_features['vibrato_presence'],
            'technique_philosophy_difference': analyze_technique_philosophy(
                chinese_features, western_features
            )
        }
    }
    
    return comparison
```

## Machine Learning Integration

### Custom Model Training

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class ChineseInstrumentDataset(Dataset):
    """Dataset for training Chinese instrument classification models"""
    
    def __init__(self, audio_files, labels, feature_extractor):
        self.audio_files = audio_files
        self.labels = labels
        self.feature_extractor = feature_extractor
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        # Load audio and extract features
        audio_data, sr = librosa.load(self.audio_files[idx])
        features, _ = self.feature_extractor.extract_chinese_features(audio_data, sr)
        
        # Convert to tensor
        feature_vector = self.features_to_vector(features)
        
        return torch.FloatTensor(feature_vector), torch.LongTensor([self.labels[idx]])
    
    def features_to_vector(self, features):
        """Convert feature dict to fixed-size vector"""
        vector = [
            features.get('pentatonic_adherence', 0),
            features.get('sliding_presence', 0),
            features.get('vibrato_rate', 0) / 15.0,  # Normalize
            features.get('ornament_density', 0),
            features.get('f0_mean', 440) / 1000.0,   # Normalize
            features.get('f0_std', 0) / 100.0        # Normalize
        ]
        return vector

class InstrumentClassificationModel(nn.Module):
    """Neural network for Chinese instrument classification"""
    
    def __init__(self, input_size=6, num_instruments=5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_instruments)
        )
    
    def forward(self, x):
        return self.network(x)

# Training loop
def train_instrument_classifier(audio_files, labels):
    """Train custom instrument classification model"""
    
    # Create dataset
    feature_extractor = EnhancedChineseInstrumentVisualizer()
    dataset = ChineseInstrumentDataset(audio_files, labels, feature_extractor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    model = InstrumentClassificationModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(100):
        total_loss = 0
        for features, labels_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels_batch.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}')
    
    return model
```

These advanced examples demonstrate the extensibility and power of the InstrumentTimbre system for sophisticated Chinese traditional music analysis.