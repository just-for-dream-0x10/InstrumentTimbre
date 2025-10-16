#!/usr/bin/env python3
"""
Model Conversion Tool for InstrumentTimbre
InstrumentTimbre 模型转换工具

Supports conversion to multiple formats for deployment:
- ONNX (Cross-platform inference)
- TorchScript (PyTorch optimized)
- TensorRT (NVIDIA GPU optimized)
- Core ML (Apple devices)
- TensorFlow Lite (Mobile devices)
"""

import argparse
import torch
import numpy as np
import os
import logging
from pathlib import Path
import json
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelConverter:
    """Enhanced model converter for multiple deployment formats"""
    
    def __init__(self, input_model_path, device='cpu'):
        """Initialize converter"""
        self.input_model_path = input_model_path
        self.device = torch.device(device)
        self.model = None
        self.model_info = {}
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the PyTorch model"""
        logger.info(f"📥 Loading model from: {self.input_model_path}")
        
        if not os.path.exists(self.input_model_path):
            raise FileNotFoundError(f"Model file not found: {self.input_model_path}")
        
        try:
            checkpoint = torch.load(self.input_model_path, map_location=self.device, weights_only=False)
            
            # Extract model information
            self.model_info = {
                'feature_size': checkpoint.get('feature_size', 50),
                'num_classes': checkpoint.get('num_classes', 5),
                'class_names': checkpoint.get('class_names', []),
                'enhanced_features': checkpoint.get('enhanced_features', True),
                'training_args': checkpoint.get('training_args', {}),
                'best_accuracy': checkpoint.get('best_accuracy', 'Unknown')
            }
            
            logger.info(f"📊 Model Info:")
            logger.info(f"   Classes: {self.model_info['class_names']}")
            logger.info(f"   Feature size: {self.model_info['feature_size']}")
            logger.info(f"   Training accuracy: {self.model_info['best_accuracy']}")
            
            # Recreate model architecture
            self.model = self._create_model_architecture(
                self.model_info['feature_size'], 
                self.model_info['num_classes']
            )
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("✅ Model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def _create_model_architecture(self, input_size, num_classes):
        """Recreate the model architecture"""
        class EnhancedChineseClassifier(torch.nn.Module):
            def __init__(self, input_size, num_classes):
                super().__init__()
                self.network = torch.nn.Sequential(
                    torch.nn.Linear(input_size, 256),
                    torch.nn.BatchNorm1d(256),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.3),
                    
                    torch.nn.Linear(256, 128),
                    torch.nn.BatchNorm1d(128),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.3),
                    
                    torch.nn.Linear(128, 64),
                    torch.nn.BatchNorm1d(64),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.2),
                    
                    torch.nn.Linear(64, 32),
                    torch.nn.ReLU(),
                    
                    torch.nn.Linear(32, num_classes)
                )
            
            def forward(self, x):
                return self.network(x)
        
        return EnhancedChineseClassifier(input_size, num_classes)
    
    def convert_to_onnx(self, output_path, opset_version=11):
        """Convert model to ONNX format"""
        logger.info(f"🔄 Converting to ONNX format...")
        
        try:
            # Create dummy input
            dummy_input = torch.randn(1, self.model_info['feature_size']).to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Export to ONNX
            torch.onnx.export(
                self.model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['audio_features'],
                output_names=['predictions'],
                dynamic_axes={
                    'audio_features': {0: 'batch_size'},
                    'predictions': {0: 'batch_size'}
                },
                verbose=False
            )
            
            # Verify the ONNX model
            self._verify_onnx_model(output_path, dummy_input)
            
            # Save metadata
            self._save_onnx_metadata(output_path)
            
            logger.info(f"✅ ONNX model saved: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ ONNX conversion failed: {e}")
            return False
    
    def _verify_onnx_model(self, onnx_path, dummy_input):
        """Verify ONNX model works correctly"""
        try:
            import onnx
            import onnxruntime as ort
            
            # Load ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            # Test inference
            ort_session = ort.InferenceSession(onnx_path)
            
            # Compare outputs
            with torch.no_grad():
                pytorch_output = self.model(dummy_input)
                onnx_output = ort_session.run(None, {'audio_features': dummy_input.cpu().numpy()})
                
                # Check if outputs are close
                np.testing.assert_allclose(
                    pytorch_output.cpu().numpy(), 
                    onnx_output[0], 
                    rtol=1e-03, 
                    atol=1e-05
                )
            
            logger.info("✅ ONNX model verification passed")
            
        except ImportError:
            logger.warning("⚠️  ONNX/ONNXRuntime not installed, skipping verification")
        except Exception as e:
            logger.warning(f"⚠️  ONNX verification failed: {e}")
    
    def _save_onnx_metadata(self, onnx_path):
        """Save ONNX model metadata"""
        metadata = {
            'model_type': 'chinese_instrument_classifier',
            'input_shape': [1, self.model_info['feature_size']],
            'output_shape': [1, self.model_info['num_classes']],
            'class_names': self.model_info['class_names'],
            'feature_size': self.model_info['feature_size'],
            'enhanced_features': self.model_info['enhanced_features'],
            'conversion_date': datetime.now().isoformat(),
            'usage': {
                'python': 'import onnxruntime; session = onnxruntime.InferenceSession("model.onnx")',
                'javascript': 'const session = new onnx.InferenceSession("model.onnx");',
                'input_name': 'audio_features',
                'output_name': 'predictions'
            }
        }
        
        metadata_path = onnx_path.replace('.onnx', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"📄 ONNX metadata saved: {metadata_path}")
    
    def convert_to_torchscript(self, output_path, method='trace'):
        """Convert model to TorchScript"""
        logger.info(f"🔄 Converting to TorchScript ({method})...")
        
        try:
            self.model.eval()
            
            if method == 'trace':
                # Tracing method
                dummy_input = torch.randn(1, self.model_info['feature_size']).to(self.device)
                traced_model = torch.jit.trace(self.model, dummy_input)
                scripted_model = traced_model
                
            elif method == 'script':
                # Scripting method
                scripted_model = torch.jit.script(self.model)
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Save the model
            scripted_model.save(output_path)
            
            # Verify the model
            self._verify_torchscript_model(output_path)
            
            # Save metadata
            self._save_torchscript_metadata(output_path)
            
            logger.info(f"✅ TorchScript model saved: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ TorchScript conversion failed: {e}")
            return False
    
    def _verify_torchscript_model(self, script_path):
        """Verify TorchScript model"""
        try:
            # Load and test
            loaded_model = torch.jit.load(script_path, map_location=self.device)
            dummy_input = torch.randn(1, self.model_info['feature_size']).to(self.device)
            
            with torch.no_grad():
                original_output = self.model(dummy_input)
                script_output = loaded_model(dummy_input)
                
                # Check if outputs are close
                torch.testing.assert_close(original_output, script_output, rtol=1e-03, atol=1e-05)
            
            logger.info("✅ TorchScript model verification passed")
            
        except Exception as e:
            logger.warning(f"⚠️  TorchScript verification failed: {e}")
    
    def _save_torchscript_metadata(self, script_path):
        """Save TorchScript metadata"""
        metadata = {
            'model_type': 'chinese_instrument_classifier',
            'format': 'torchscript',
            'input_shape': [1, self.model_info['feature_size']],
            'output_shape': [1, self.model_info['num_classes']],
            'class_names': self.model_info['class_names'],
            'conversion_date': datetime.now().isoformat(),
            'usage': {
                'python': f'import torch; model = torch.jit.load("{Path(script_path).name}")',
                'cpp': 'torch::jit::script::Module module = torch::jit::load("model.pt");'
            }
        }
        
        metadata_path = script_path.replace('.pt', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"📄 TorchScript metadata saved: {metadata_path}")
    
    def convert_to_tensorrt(self, output_path):
        """Convert model to TensorRT (requires NVIDIA GPU)"""
        logger.info(f"🔄 Converting to TensorRT...")
        
        try:
            import tensorrt as trt
            import torch_tensorrt
            
            # Check if CUDA is available
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available, TensorRT requires NVIDIA GPU")
            
            # Move model to GPU
            self.model = self.model.cuda()
            
            # Create example input
            dummy_input = torch.randn(1, self.model_info['feature_size']).cuda()
            
            # Compile with TensorRT
            trt_model = torch_tensorrt.compile(
                self.model,
                inputs=[dummy_input],
                enabled_precisions=torch.half,  # Use FP16 for better performance
                workspace_size=1 << 20  # 1MB workspace
            )
            
            # Save the model
            torch.jit.save(trt_model, output_path)
            
            # Verify
            self._verify_tensorrt_model(output_path, dummy_input)
            
            logger.info(f"✅ TensorRT model saved: {output_path}")
            return True
            
        except ImportError:
            logger.error("❌ TensorRT not installed. Install with: pip install torch-tensorrt")
            return False
        except Exception as e:
            logger.error(f"❌ TensorRT conversion failed: {e}")
            return False
    
    def _verify_tensorrt_model(self, trt_path, dummy_input):
        """Verify TensorRT model"""
        try:
            trt_model = torch.jit.load(trt_path)
            
            with torch.no_grad():
                original_output = self.model(dummy_input)
                trt_output = trt_model(dummy_input)
                
                # TensorRT might have slightly different precision
                torch.testing.assert_close(original_output, trt_output, rtol=1e-02, atol=1e-03)
            
            logger.info("✅ TensorRT model verification passed")
            
        except Exception as e:
            logger.warning(f"⚠️  TensorRT verification failed: {e}")
    
    def convert_to_coreml(self, output_path):
        """Convert model to Core ML (Apple devices)"""
        logger.info(f"🔄 Converting to Core ML...")
        
        try:
            import coremltools as ct
            
            # Create example input
            dummy_input = torch.randn(1, self.model_info['feature_size'])
            
            # Trace the model
            traced_model = torch.jit.trace(self.model.cpu(), dummy_input)
            
            # Convert to Core ML
            coreml_model = ct.convert(
                traced_model,
                inputs=[ct.TensorType(shape=dummy_input.shape, name="audio_features")],
                outputs=[ct.TensorType(name="predictions")],
                minimum_deployment_target=ct.target.iOS14,
                compute_units=ct.ComputeUnit.ALL
            )
            
            # Add metadata
            coreml_model.short_description = "Chinese Traditional Instrument Classifier"
            coreml_model.author = "InstrumentTimbre"
            coreml_model.version = "1.0"
            coreml_model.license = "MIT"
            
            # Add input/output descriptions
            coreml_model.input_description["audio_features"] = "50-dimensional audio feature vector"
            coreml_model.output_description["predictions"] = f"Predictions for {len(self.model_info['class_names'])} instrument classes"
            
            # Save the model
            coreml_model.save(output_path)
            
            # Save metadata
            self._save_coreml_metadata(output_path)
            
            logger.info(f"✅ Core ML model saved: {output_path}")
            return True
            
        except ImportError:
            logger.error("❌ Core ML Tools not installed. Install with: pip install coremltools")
            return False
        except Exception as e:
            logger.error(f"❌ Core ML conversion failed: {e}")
            return False
    
    def _save_coreml_metadata(self, coreml_path):
        """Save Core ML metadata"""
        metadata = {
            'model_type': 'chinese_instrument_classifier',
            'format': 'coreml',
            'platform': 'iOS/macOS',
            'input_name': 'audio_features',
            'output_name': 'predictions',
            'class_names': self.model_info['class_names'],
            'conversion_date': datetime.now().isoformat(),
            'usage': {
                'swift': '''
let model = try ChineseInstrumentClassifier(configuration: MLModelConfiguration())
let input = ChineseInstrumentClassifierInput(audio_features: features)
let prediction = try model.prediction(from: input)
                '''.strip(),
                'objective_c': '''
NSError *error;
ChineseInstrumentClassifier *model = [[ChineseInstrumentClassifier alloc] initWithConfiguration:[MLModelConfiguration new] error:&error];
                '''.strip()
            }
        }
        
        metadata_path = coreml_path.replace('.mlmodel', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"📄 Core ML metadata saved: {metadata_path}")
    
    def convert_to_tflite(self, output_path):
        """Convert model to TensorFlow Lite"""
        logger.info(f"🔄 Converting to TensorFlow Lite...")
        
        try:
            import tensorflow as tf
            import onnx
            from onnx_tf.backend import prepare
            
            # First convert to ONNX, then to TensorFlow, then to TFLite
            temp_onnx = "temp_model.onnx"
            
            # Convert to ONNX first
            if not self.convert_to_onnx(temp_onnx):
                raise RuntimeError("Failed to convert to ONNX first")
            
            # Load ONNX model
            onnx_model = onnx.load(temp_onnx)
            
            # Convert ONNX to TensorFlow
            tf_rep = prepare(onnx_model)
            
            # Convert to TensorFlow Lite
            converter = tf.lite.TFLiteConverter.from_concrete_functions(tf_rep.signatures)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Enable quantization for smaller model size
            converter.representative_dataset = self._representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.float32
            converter.inference_output_type = tf.float32
            
            tflite_model = converter.convert()
            
            # Save the model
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            # Clean up temporary file
            if os.path.exists(temp_onnx):
                os.remove(temp_onnx)
            
            # Save metadata
            self._save_tflite_metadata(output_path)
            
            logger.info(f"✅ TensorFlow Lite model saved: {output_path}")
            return True
            
        except ImportError:
            logger.error("❌ TensorFlow/ONNX-TF not installed. Install with: pip install tensorflow onnx-tf")
            return False
        except Exception as e:
            logger.error(f"❌ TensorFlow Lite conversion failed: {e}")
            return False
    
    def _representative_dataset(self):
        """Generate representative dataset for TFLite quantization"""
        for _ in range(100):
            yield [np.random.randn(1, self.model_info['feature_size']).astype(np.float32)]
    
    def _save_tflite_metadata(self, tflite_path):
        """Save TensorFlow Lite metadata"""
        metadata = {
            'model_type': 'chinese_instrument_classifier',
            'format': 'tflite',
            'platform': 'Android/Mobile',
            'quantized': True,
            'class_names': self.model_info['class_names'],
            'conversion_date': datetime.now().isoformat(),
            'usage': {
                'android_java': '''
Interpreter tflite = new Interpreter(loadModelFile());
float[][] input = new float[1][50];
float[][] output = new float[1][5];
tflite.run(input, output);
                '''.strip(),
                'python': '''
import tensorflow as tf
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
                '''.strip()
            }
        }
        
        metadata_path = tflite_path.replace('.tflite', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"📄 TensorFlow Lite metadata saved: {metadata_path}")
    
    def get_model_size(self, file_path):
        """Get model file size"""
        if os.path.exists(file_path):
            size_bytes = os.path.getsize(file_path)
            size_mb = size_bytes / (1024 * 1024)
            return f"{size_mb:.2f} MB ({size_bytes:,} bytes)"
        return "File not found"
    
    def benchmark_model(self, model_path, format_type, num_iterations=100):
        """Benchmark model inference speed"""
        logger.info(f"⏱️  Benchmarking {format_type} model...")
        
        try:
            import time
            
            if format_type == 'onnx':
                import onnxruntime as ort
                session = ort.InferenceSession(model_path)
                input_name = session.get_inputs()[0].name
                
                dummy_input = np.random.randn(1, self.model_info['feature_size']).astype(np.float32)
                
                # Warmup
                for _ in range(10):
                    session.run(None, {input_name: dummy_input})
                
                # Benchmark
                start_time = time.time()
                for _ in range(num_iterations):
                    session.run(None, {input_name: dummy_input})
                end_time = time.time()
                
            elif format_type == 'torchscript':
                model = torch.jit.load(model_path, map_location=self.device)
                dummy_input = torch.randn(1, self.model_info['feature_size']).to(self.device)
                
                # Warmup
                with torch.no_grad():
                    for _ in range(10):
                        model(dummy_input)
                
                # Benchmark
                start_time = time.time()
                with torch.no_grad():
                    for _ in range(num_iterations):
                        model(dummy_input)
                end_time = time.time()
            
            else:
                logger.warning(f"Benchmarking not implemented for {format_type}")
                return None
            
            avg_time = (end_time - start_time) / num_iterations * 1000  # ms
            fps = 1000 / avg_time if avg_time > 0 else 0
            
            return {
                'avg_inference_time_ms': avg_time,
                'fps': fps,
                'total_time_s': end_time - start_time,
                'iterations': num_iterations
            }
            
        except Exception as e:
            logger.error(f"Benchmarking failed: {e}")
            return None


def main():
    """Main conversion function"""
    parser = argparse.ArgumentParser(description='InstrumentTimbre Model Conversion Tool')
    
    parser.add_argument('--input', required=True, help='Input PyTorch model (.pt file)')
    parser.add_argument('--output', help='Output file path (will auto-generate if not provided)')
    parser.add_argument('--format', required=True, 
                       choices=['onnx', 'torchscript', 'tensorrt', 'coreml', 'tflite', 'all'],
                       help='Output format')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='Device for conversion')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset version')
    parser.add_argument('--method', default='trace', choices=['trace', 'script'], 
                       help='TorchScript conversion method')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark converted model')
    parser.add_argument('--output-dir', default='converted_models', help='Output directory for conversions')
    
    args = parser.parse_args()
    
    print("🔄 InstrumentTimbre Model Conversion Tool")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize converter
    try:
        converter = ModelConverter(args.input, args.device)
    except Exception as e:
        logger.error(f"Failed to initialize converter: {e}")
        return
    
    # Get base name for output files
    base_name = Path(args.input).stem
    
    # Define conversion functions
    conversions = {
        'onnx': lambda: converter.convert_to_onnx(
            args.output or f"{args.output_dir}/{base_name}.onnx", 
            args.opset
        ),
        'torchscript': lambda: converter.convert_to_torchscript(
            args.output or f"{args.output_dir}/{base_name}_script.pt", 
            args.method
        ),
        'tensorrt': lambda: converter.convert_to_tensorrt(
            args.output or f"{args.output_dir}/{base_name}_trt.pt"
        ),
        'coreml': lambda: converter.convert_to_coreml(
            args.output or f"{args.output_dir}/{base_name}.mlmodel"
        ),
        'tflite': lambda: converter.convert_to_tflite(
            args.output or f"{args.output_dir}/{base_name}.tflite"
        )
    }
    
    # Perform conversions
    results = {}
    
    if args.format == 'all':
        # Convert to all formats
        logger.info("🔄 Converting to all supported formats...")
        
        for format_name, convert_func in conversions.items():
            logger.info(f"\n--- Converting to {format_name.upper()} ---")
            success = convert_func()
            results[format_name] = success
            
    else:
        # Convert to specific format
        if args.format in conversions:
            success = conversions[args.format]()
            results[args.format] = success
        else:
            logger.error(f"Unsupported format: {args.format}")
            return
    
    # Print summary
    print(f"\n📊 Conversion Summary:")
    print("-" * 30)
    
    for format_name, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"  {format_name.upper()}: {status}")
        
        if success:
            # Show file size
            extensions = {
                'onnx': '.onnx',
                'torchscript': '_script.pt',
                'tensorrt': '_trt.pt',
                'coreml': '.mlmodel',
                'tflite': '.tflite'
            }
            
            file_path = f"{args.output_dir}/{base_name}{extensions[format_name]}"
            if os.path.exists(file_path):
                size = converter.get_model_size(file_path)
                print(f"    Size: {size}")
                
                # Benchmark if requested
                if args.benchmark and format_name in ['onnx', 'torchscript']:
                    bench_result = converter.benchmark_model(file_path, format_name)
                    if bench_result:
                        print(f"    Inference: {bench_result['avg_inference_time_ms']:.2f} ms/sample")
                        print(f"    Throughput: {bench_result['fps']:.1f} FPS")
    
    print(f"\n📁 Output directory: {os.path.abspath(args.output_dir)}")
    print("🎉 Model conversion completed!")


if __name__ == "__main__":
    main()