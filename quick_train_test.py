#!/usr/bin/env python3
"""
Quick Training Test Script - Validate optimized performance
"""

import sys
import time
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def test_data_loading_speed():
    """Test data loading speed"""
    from InstrumentTimbre.core.data.fast_dataset import FastAudioDataset
    from InstrumentTimbre.core.features.fast import FastFeatureExtractor
    
    print("🧪 Testing data loading speed...")
    
    # Initialize fast feature extractor
    extractor = FastFeatureExtractor()
    
    # Create dataset (will compute and cache features on first run)
    start_time = time.time()
    dataset = FastAudioDataset(
        data_dir='./data/clips',
        feature_extractor=extractor
    )
    setup_time = time.time() - start_time
    
    print(f"Dataset setup time: {setup_time:.2f}s")
    print(f"Total samples: {len(dataset)}")
    print(f"Number of classes: {len(dataset.class_names)}")
    print(f"Classes: {dataset.class_names}")
    print(f"Feature dimension: {dataset.get_feature_dim()}")
    
    # Test batch loading speed
    print("\nTesting batch loading speed...")
    from torch.utils.data import DataLoader
    
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    start_time = time.time()
    for i, (features, labels) in enumerate(loader):
        if i >= 10:  # Test only first 10 batches
            break
        if i == 0:
            print(f"Batch shape: {features.shape}, Labels shape: {labels.shape}")
    
    load_time = time.time() - start_time
    samples_processed = min(10 * 32, len(dataset))
    
    print(f"Processed {samples_processed} samples in: {load_time:.3f}s")
    print(f"Average speed: {samples_processed/load_time:.1f} samples/sec")
    
    return dataset

if __name__ == '__main__':
    print("🚀 InstrumentTimbre Optimized Performance Test")
    print("=" * 50)
    
    # Check if data clipping is complete
    clips_dir = Path('./data/clips')
    if not clips_dir.exists():
        print("❌ Data clips directory not found, please run clipping script first")
        sys.exit(1)
    
    # Count clips
    total_clips = 0
    for class_dir in clips_dir.iterdir():
        if class_dir.is_dir():
            count = len(list(class_dir.glob('*.wav')))
            total_clips += count
            print(f"📁 {class_dir.name}: {count} clips")
    
    print(f"📊 Total: {total_clips} training clips")
    
    if total_clips == 0:
        print("❌ No training clips found")
        sys.exit(1)
    
    # Test data loading
    dataset = test_data_loading_speed()
    
    print("\n✅ Optimization complete! Ready for fast training")
    print("\n🎯 Recommended training command:")
    print("python train.py --data ./data/clips --batch-size 32 --epochs 10")
    print("\nExpected training speed: seconds/epoch (instead of minutes)")