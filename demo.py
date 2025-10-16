#!/usr/bin/env python3
"""
InstrumentTimbre Demo Script
Demonstrates enhanced Chinese instrument analysis capabilities
"""

import os
import sys
from pathlib import Path

def demo_enhanced_analysis():
    """Demo enhanced Chinese instrument analysis"""
    print("🎵 InstrumentTimbre Enhanced Chinese Instrument Analysis Demo")
    print("=" * 70)
    
    # Check if example files exist
    example_files = [
        "example/erhu1.wav",
        "example/erhu2.wav"
    ]
    
    available_files = [f for f in example_files if os.path.exists(f)]
    
    if not available_files:
        print("❌ No example audio files found!")
        print("Please ensure example/erhu1.wav and example/erhu2.wav exist.")
        return
    
    print(f"✅ Found {len(available_files)} example files")
    
    # Run enhanced visualization
    print("\n🎨 Running Enhanced Visualization...")
    os.chdir("example")
    
    for audio_file in available_files:
        filename = os.path.basename(audio_file)
        print(f"\n📊 Processing: {filename}")
        
        cmd = f"python enhanced_chinese_visualization.py --input {filename} --output ../demo_output"
        result = os.system(cmd)
        
        if result == 0:
            print(f"✅ Successfully processed {filename}")
        else:
            print(f"❌ Error processing {filename}")
    
    os.chdir("..")
    
    # Check results
    output_dir = "demo_output"
    if os.path.exists(output_dir):
        output_files = list(Path(output_dir).glob("*.png"))
        print(f"\n🎉 Demo completed! Generated {len(output_files)} visualization files:")
        for file in output_files:
            print(f"   📈 {file}")
        
        print(f"\n📂 Results saved in: {os.path.abspath(output_dir)}")
        print("\n🔍 Features analyzed:")
        print("   • Pentatonic Adherence (Wu Sheng scale conformity)")
        print("   • Sliding Analysis (Hua Yin technique)")
        print("   • Vibrato Analysis (Chan Yin patterns)")
        print("   • Ornament Density (Zhuang Shi Yin)")
        print("   • Comprehensive audio visualizations")
        
    else:
        print("❌ No output files generated")

def demo_basic_features():
    """Demo basic feature extraction"""
    print("\n🎼 Running Basic Feature Extraction...")
    
    os.chdir("example")
    
    # Run existing visualization
    result = os.system("python timbre_extraction_visualization.py")
    
    if result == 0:
        print("✅ Basic feature extraction completed")
        print("📊 Check example/visualizations/ for basic analysis results")
    else:
        print("❌ Basic feature extraction failed")
    
    os.chdir("..")

def main():
    """Main demo function"""
    
    print("🚀 Welcome to InstrumentTimbre Demo!")
    print("\nThis demo will showcase:")
    print("1. Enhanced Chinese instrument analysis")
    print("2. Traditional technique detection")
    print("3. Comprehensive visualizations")
    
    try:
        # Enhanced analysis demo
        demo_enhanced_analysis()
        
        # Basic features demo
        demo_basic_features()
        
        print("\n🎉 Demo completed successfully!")
        print("\n📖 Next steps:")
        print("1. Check the generated visualization files")
        print("2. Read the README.md for detailed usage instructions")
        print("3. Try processing your own audio files")
        print("4. Explore the enhanced Chinese instrument features")
        
    except KeyboardInterrupt:
        print("\n❌ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()