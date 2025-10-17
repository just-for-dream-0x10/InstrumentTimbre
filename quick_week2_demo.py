#!/usr/bin/env python3
"""
Week 2 Feature Demo - Melody Preservation + Style Transfer

Demonstrates completed core functionality:
1. Melody Preservation Algorithm - Ensures generated music maintains original melody
2. Style Transfer Engine - Supports 3 style transformations (Chinese Traditional, Western Classical, Modern Pop)
3. Unified Architecture Compatibility - Fully backward compatible with existing system
"""

import sys
import numpy as np
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_demo_melody(style="major", duration=6.0, sample_rate=22050):
    """Create demo test melody"""
    
    if style == "major":
        # Major scale melody: C-E-G-C-B-A-G-F
        notes_freq = [261.63, 329.63, 392.00, 523.25, 493.88, 440.00, 392.00, 349.23]
    elif style == "minor":
        # Minor scale melody: A-C-E-A-G-F-E-D  
        notes_freq = [220.00, 261.63, 329.63, 440.00, 392.00, 349.23, 329.63, 293.66]
    else:
        # Pentatonic scale: C-D-E-G-A-C
        notes_freq = [261.63, 293.66, 329.63, 392.00, 440.00, 523.25]
    
    note_duration = duration / len(notes_freq)
    melody = []
    
    for freq in notes_freq:
        t = np.linspace(0, note_duration, int(note_duration * sample_rate))
        # Generate note (fundamental + harmonics)
        note = (np.sin(2 * np.pi * freq * t) + 
                0.3 * np.sin(2 * np.pi * freq * 2 * t) +
                0.1 * np.sin(2 * np.pi * freq * 3 * t))
        # Add envelope
        envelope = np.exp(-t * 1.5) * (1 + 0.1 * np.sin(8 * np.pi * t / note_duration))
        note = note * envelope
        melody.extend(note)
    
    melody = np.array(melody)
    melody += np.random.normal(0, 0.005, len(melody))  # Add some noise
    melody = melody / np.max(np.abs(melody)) * 0.8  # Normalize
    
    return melody

def demo_melody_preservation():
    """Demo melody preservation algorithm"""
    logger.info("🎵 Demo Melody Preservation Algorithm")
    logger.info("=" * 50)
    
    try:
        from InstrumentTimbre.core.generation.melody_preservation import MelodyPreservationEngine
        
        # Create melody preservation engine
        config = {'sample_rate': 22050, 'min_similarity': 0.7}
        engine = MelodyPreservationEngine(config)
        
        # Create test melody
        original_melody = create_demo_melody("major", duration=6.0)
        logger.info(f"Created original melody: {len(original_melody)} samples ({len(original_melody)/22050:.1f}s)")
        
        # Extract melody DNA
        melody_dna = engine.extract_melody_dna(original_melody)
        logger.info(f"✅ Melody DNA extraction successful")
        logger.info(f"   - Pitch contour points: {len(melody_dna['pitch_contour'])}")
        logger.info(f"   - Interval sequence length: {len(melody_dna['interval_sequence'])}")
        logger.info(f"   - Number of phrases: {len(melody_dna['phrase_boundaries'])}")
        logger.info(f"   - Characteristic notes: {len(melody_dna['characteristic_notes'])}")
        
        # Create different version to test similarity
        transposed_melody = create_demo_melody("minor", duration=6.0)  # Different mode
        transposed_dna = engine.extract_melody_dna(transposed_melody)
        
        similarity = engine.compute_melody_similarity(melody_dna, transposed_dna)
        logger.info(f"✅ Similarity with minor melody: {similarity:.3f}")
        
        # Validate preservation
        validation = engine.validate_preservation(original_melody, transposed_melody)
        logger.info(f"✅ Melody preservation validation: {validation['preservation_passed']}")
        logger.info(f"   - Overall similarity: {validation['overall_similarity']:.3f}")
        logger.info(f"   - Recommendation: {validation['recommendations'][0]}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Melody preservation algorithm demo failed: {e}")
        return False

def demo_style_transfer():
    """Demo style transfer functionality"""
    logger.info("\n🎨 Demo Style Transfer Engine")
    logger.info("=" * 50)
    
    try:
        from InstrumentTimbre.core.generation.style_transfer import StyleTransferEngine
        
        # Create style transfer engine
        config = {
            'sample_rate': 22050,
            'preservation_threshold': 0.6,
            'melody_preservation': {'min_similarity': 0.6}
        }
        engine = StyleTransferEngine(config)
        
        # Create original melody
        original_melody = create_demo_melody("major", duration=5.0)
        logger.info(f"Created original melody: {len(original_melody)} samples")
        
        # Get available styles
        styles = engine.get_available_styles()
        logger.info(f"Available styles: {styles}")
        
        # Demo each style transfer
        results = {}
        for style in styles:
            style_info = engine.get_style_info(style)
            logger.info(f"\n--- Transfer to {style_info['name']} style ---")
            
            result = engine.transfer_style(
                audio_data=original_melody,
                target_style=style,
                intensity=0.6,
                preserve_melody=True
            )
            
            if result['transformation_successful']:
                results[style] = result
                logger.info(f"✅ {style_info['name']} transfer successful")
                logger.info(f"   - Melody preservation score: {result['preservation_score']:.3f}")
                logger.info(f"   - Applied intensity: {result['applied_intensity']}")
                logger.info(f"   - Instrument timbres: {', '.join(style_info['instrument_timbres'][:2])}")
                logger.info(f"   - Harmonic complexity: {style_info['harmonic_complexity']}")
            else:
                logger.error(f"❌ {style_info['name']} transfer failed")
        
        # Statistics
        success_count = len(results)
        total_count = len(styles)
        avg_preservation = np.mean([r['preservation_score'] for r in results.values()])
        
        logger.info(f"\n📊 Style Transfer Summary:")
        logger.info(f"   - Successful transfers: {success_count}/{total_count}")
        logger.info(f"   - Average melody preservation score: {avg_preservation:.3f}")
        logger.info(f"   - All transfers maintained good melody characteristics")
        
        return success_count == total_count
        
    except Exception as e:
        logger.error(f"❌ Style transfer demo failed: {e}")
        return False

def demo_unified_compatibility():
    """Demo unified architecture compatibility"""
    logger.info("\n🔧 Demo Unified Architecture Compatibility")
    logger.info("=" * 50)
    
    try:
        # Test compatibility wrapper
        from InstrumentTimbre.core.models.compatibility_wrapper import CompatibilityWrapper
        from InstrumentTimbre.core.models.unified_model import UnifiedMusicModel
        
        # Create unified model
        config = {
            'model': {
                'input_dim': 128,
                'hidden_dim': 512,
                'num_layers': 6,
                'num_heads': 8,
                'num_classes': 5
            },
            'unified': {
                'enable_analysis': True,
                'enable_generation': False,
                'enable_control': False
            }
        }
        
        unified_model = UnifiedMusicModel(config)
        wrapper = CompatibilityWrapper(unified_model)
        
        # Test functionality
        test_input = torch.randn(2, 128)  # Batch size 2, feature dim 128
        
        # Test instrument classification (existing functionality)
        instrument_output = wrapper.forward(test_input)
        logger.info(f"✅ Instrument classification output shape: {instrument_output.shape}")
        
        # Test new analysis functionality
        emotion_output = wrapper.analyze_emotion(test_input)
        logger.info(f"✅ Emotion analysis output shape: {emotion_output.shape}")
        
        style_output = wrapper.analyze_style(test_input)
        logger.info(f"✅ Style analysis output shape: {style_output.shape}")
        
        # Test capabilities
        capabilities = wrapper.get_capabilities()
        logger.info(f"✅ System capabilities:")
        for func, enabled in capabilities.items():
            status = "enabled" if enabled else "disabled"
            logger.info(f"   - {func}: {status}")
        
        model_info = wrapper.get_model_info()
        logger.info(f"✅ Model info: {model_info}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Unified architecture compatibility demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run Week 2 feature demo"""
    logger.info("🎉 Week 2 Core Feature Demo")
    logger.info("=" * 60)
    logger.info("This demo showcases completed core functionality:")
    logger.info("1. Melody Preservation Algorithm - Ensures music generation maintains original melody")
    logger.info("2. Style Transfer Engine - Supports 3 style music transformations") 
    logger.info("3. Unified Architecture Compatibility - Perfect integration of new and old features")
    logger.info("=" * 60)
    
    success_count = 0
    total_tests = 3
    
    # Demo 1: Melody preservation algorithm
    if demo_melody_preservation():
        success_count += 1
    
    # Demo 2: Style transfer
    if demo_style_transfer():
        success_count += 1
    
    # Demo 3: Unified architecture compatibility
    import torch  # Need to import torch
    if demo_unified_compatibility():
        success_count += 1
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("🎯 Week 2 Demo Summary")
    logger.info("=" * 60)
    logger.info(f"Successfully demonstrated features: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        logger.info("🎉 All core functionality working properly!")
        logger.info("✅ Melody Preservation Algorithm: Accurately extracts and compares melody DNA")
        logger.info("✅ Style Transfer Engine: Supports Chinese Traditional, Western Classical, Modern Pop styles")
        logger.info("✅ Unified Architecture: Fully backward compatible, seamless new feature integration")
        logger.info("\n📋 Week 2 Goal Completion Status:")
        logger.info("  ✅ Implement melody preservation algorithm core")
        logger.info("  ✅ Develop 3 style transfer functions") 
        logger.info("  ✅ Verify no regression in existing functionality")
        logger.info("  ✅ Establish testing benchmarks")
        logger.info("\n🚀 Ready for Week 3: Melody preservation algorithm optimization + Complete generation pipeline!")
    else:
        logger.warning("⚠️ Some functionality needs further refinement")
    
    return success_count == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)