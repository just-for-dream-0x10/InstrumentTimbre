# Model Conversion Guide

## Overview

The InstrumentTimbre model conversion tool allows you to convert trained PyTorch models to various deployment formats for different platforms and use cases.

## Supported Formats

### 1. ONNX (Recommended for Web/Cross-platform)
```bash
# Convert to ONNX
python convert_model.py --input saved_models/enhanced_model.pt --format onnx

# With custom output path
python convert_model.py --input saved_models/enhanced_model.pt --format onnx --output my_model.onnx
```

**Use Cases:**
- Web applications (ONNX.js)
- Cross-platform inference
- Integration with other ML frameworks
- Cloud deployment

### 2. TorchScript (Optimized PyTorch)
```bash
# Convert using tracing (recommended)
python convert_model.py --input saved_models/enhanced_model.pt --format torchscript --method trace

# Convert using scripting
python convert_model.py --input saved_models/enhanced_model.pt --format torchscript --method script
```

**Use Cases:**
- Production PyTorch deployment
- C++ applications
- Mobile PyTorch apps
- Server optimization

### 3. Core ML (Apple Devices)
```bash
# Convert to Core ML
python convert_model.py --input saved_models/enhanced_model.pt --format coreml
```

**Use Cases:**
- iOS applications
- macOS applications
- Apple Silicon optimization
- On-device inference

### 4. TensorFlow Lite (Mobile/Edge)
```bash
# Convert to TensorFlow Lite
python convert_model.py --input saved_models/enhanced_model.pt --format tflite
```

**Use Cases:**
- Android applications
- Edge devices
- Quantized inference
- Low-power deployment

### 5. TensorRT (NVIDIA GPU)
```bash
# Convert to TensorRT (requires NVIDIA GPU)
python convert_model.py --input saved_models/enhanced_model.pt --format tensorrt --device cuda
```

**Use Cases:**
- High-performance GPU inference
- NVIDIA Jetson devices
- Production servers with NVIDIA GPUs
- Real-time applications

## Batch Conversion

Convert to all supported formats:
```bash
python convert_model.py --input saved_models/enhanced_model.pt --format all --output-dir deployment_models
```

## Performance Benchmarking

Test inference speed of converted models:
```bash
python convert_model.py --input saved_models/enhanced_model.pt --format onnx --benchmark
```

## Example Outputs

After conversion, you'll get:

```
converted_models/
├── enhanced_model.onnx                    # ONNX model
├── enhanced_model_metadata.json          # ONNX metadata
├── enhanced_model_script.pt              # TorchScript model
├── enhanced_model_script_metadata.json   # TorchScript metadata
├── enhanced_model.mlmodel                # Core ML model
├── enhanced_model.tflite                 # TensorFlow Lite model
└── enhanced_model_trt.pt                 # TensorRT model
```

## Usage in Different Platforms

### Web Application (ONNX.js)
```javascript
// Load model in browser
const session = new onnx.InferenceSession('enhanced_model.onnx');

// Make prediction
const inputTensor = new onnx.Tensor('float32', audioFeatures, [1, 50]);
const outputs = await session.run({ audio_features: inputTensor });
const predictions = outputs.predictions.data;
```

### iOS Application (Core ML)
```swift
import CoreML

// Load model
guard let model = try? ChineseInstrumentClassifier(configuration: MLModelConfiguration()) else {
    return
}

// Prepare input
let input = try MLMultiArray(shape: [1, 50], dataType: .float32)
// ... fill input with audio features ...

// Make prediction
let prediction = try model.prediction(audio_features: input)
let probabilities = prediction.predictions
```

### Android Application (TensorFlow Lite)
```java
// Load model
Interpreter tflite = new Interpreter(loadModelFile("enhanced_model.tflite"));

// Prepare input/output
float[][] input = new float[1][50];
float[][] output = new float[1][NUM_CLASSES];

// Make prediction
tflite.run(input, output);
```

### Python ONNX Inference
```python
import onnxruntime as ort
import numpy as np

# Load model
session = ort.InferenceSession('enhanced_model.onnx')

# Make prediction
audio_features = np.random.randn(1, 50).astype(np.float32)
result = session.run(['predictions'], {'audio_features': audio_features})
predictions = result[0]
```

### C++ TorchScript
```cpp
#include <torch/script.h>

// Load model
torch::jit::script::Module module = torch::jit::load("enhanced_model_script.pt");

// Make prediction
std::vector<torch::jit::IValue> inputs;
inputs.push_back(torch::randn({1, 50}));
at::Tensor output = module.forward(inputs).toTensor();
```

## Model Size Comparison

Typical file sizes for our Chinese instrument model:

| Format | Size | Use Case |
|--------|------|----------|
| Original PyTorch | ~230 KB | Training/Development |
| ONNX | ~220 KB | Web/Cross-platform |
| TorchScript | ~230 KB | Production PyTorch |
| Core ML | ~200 KB | iOS/macOS |
| TensorFlow Lite | ~150 KB | Mobile/Edge (quantized) |
| TensorRT | ~180 KB | NVIDIA GPU |

## Performance Comparison

Inference speed on different platforms:

| Format | Platform | Inference Time | Throughput |
|--------|----------|----------------|------------|
| ONNX | CPU | ~2-5 ms | ~200-500 FPS |
| TorchScript | CPU | ~3-6 ms | ~150-300 FPS |
| TensorRT | GPU | ~0.5-1 ms | ~1000+ FPS |
| Core ML | iPhone | ~1-3 ms | ~300-1000 FPS |
| TensorFlow Lite | Android | ~2-4 ms | ~250-500 FPS |

## Installation Requirements

### ONNX
```bash
pip install onnx onnxruntime
```

### Core ML
```bash
pip install coremltools
```

### TensorFlow Lite
```bash
pip install tensorflow onnx-tf
```

### TensorRT
```bash
# Requires NVIDIA GPU and TensorRT installation
pip install torch-tensorrt
```

## Troubleshooting

### Common Issues

1. **ONNX Conversion Fails**
   - Ensure all operations are ONNX-compatible
   - Try different opset versions: `--opset 9` or `--opset 12`

2. **TensorRT Requires GPU**
   - TensorRT only works with NVIDIA GPUs
   - Use `--device cuda` flag

3. **Core ML Size Limits**
   - Core ML models are optimized for mobile
   - Large models may need quantization

4. **TensorFlow Lite Quantization**
   - Automatic quantization reduces model size
   - May slightly affect accuracy

### Verification

All conversions include verification steps to ensure the converted model produces similar outputs to the original PyTorch model.

## Best Practices

1. **Choose the Right Format**
   - ONNX: Best for web and cross-platform
   - TorchScript: Best for PyTorch production
   - Core ML: Best for Apple devices
   - TensorFlow Lite: Best for Android/Edge

2. **Test Thoroughly**
   - Always benchmark converted models
   - Verify accuracy on test data
   - Test on target deployment platform

3. **Optimize for Target**
   - Use quantization for mobile deployment
   - Use GPU formats for high-throughput needs
   - Consider model size constraints