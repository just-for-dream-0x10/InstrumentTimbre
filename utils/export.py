import os
import torch
import torch.nn as nn
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path
import torch.quantization
import copy

# For TensorRT export
try:
    import tensorrt as trt
    import onnx_tensorrt.backend as backend

    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

# For CoreML export
try:
    # Suppress CoreML compatibility warnings
    import os
    import sys
    import io

    # Temporarily redirect stderr to suppress warnings
    original_stderr = sys.stderr
    sys.stderr = io.StringIO()
    import coremltools as ct

    # Restore stderr
    sys.stderr = original_stderr
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False

# For pruning and optimization
try:
    import torch.nn.utils.prune as prune
    from torch.quantization import quantize_dynamic
except ImportError:
    pass


def export_to_onnx(
    model,
    save_path,
    input_shape=(1, 1, 128, 128),
    dynamic_axes=None,
    optimize=True,
    opset_version=12,
    verbose=False,
):
    """
    Export a PyTorch model to ONNX format for deployment with enhanced optimization options

    Args:
        model: PyTorch model to export
        save_path: Path to save the ONNX model
        input_shape: Shape of the input tensor
        dynamic_axes: Dictionary specifying dynamic axes
        optimize: Whether to optimize the ONNX model after export
        opset_version: ONNX opset version to use
        verbose: Whether to print detailed logs

    Returns:
        True if export was successful
    """
    # Ensure model is in evaluation mode
    if hasattr(model, "eval"):
        model.eval()

    # Create dummy input
    dummy_input = torch.randn(input_shape, requires_grad=True)

    # Default dynamic axes if not provided
    if dynamic_axes is None:
        dynamic_axes = {
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size"},
        }

    try:
        # Export the model
        torch.onnx.export(
            model,  # model being run
            dummy_input,  # model input (or a tuple for multiple inputs)
            save_path,  # where to save the model
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=opset_version,  # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=["input"],  # the model's input names
            output_names=["output"],  # the model's output names
            dynamic_axes=dynamic_axes,  # variable length axes
            verbose=verbose,  # detailed logs during export
        )

        # Perform ONNX model optimization
        if optimize:
            try:
                import onnx
                from onnxoptimizer import optimize

                # Load the model
                onnx_model = onnx.load(save_path)

                # Check the model
                onnx.checker.check_model(onnx_model)

                # Optimize the model
                optimization_passes = [
                    "eliminate_identity",
                    "eliminate_nop_dropout",
                    "fuse_bn_into_conv",
                    "fuse_add_bias_into_conv",
                ]
                onnx_model = optimize(onnx_model, optimization_passes)

                # Save the optimized model
                opt_path = save_path.replace(".onnx", "_optimized.onnx")
                onnx.save(onnx_model, opt_path)
                print(f"Optimized ONNX model saved to: {opt_path}")

                # Use the optimized model as the final model
                if os.path.exists(opt_path):
                    os.replace(opt_path, save_path)
                    print(f"Replaced original with optimized model")
            except ImportError:
                print("ONNX optimizer not available. Skipping optimization.")

        print(f"Model exported to ONNX format: {save_path}")
        return True
    except Exception as e:
        print(f"Error exporting model to ONNX: {e}")
        return False


def export_model_metadata(model, save_path, config=None):
    """
    Export model metadata for deployment

    Args:
        model: Model to export metadata for
        save_path: Path to save metadata
        config: Additional configuration information

    Returns:
        True if export was successful
    """
    # Extract model information
    metadata = {
        "model_type": model.__class__.__name__,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "architecture": str(model),
        "parameters": sum(p.numel() for p in model.parameters()),
    }

    # Add configuration if provided
    if config is not None:
        metadata["config"] = config

    try:
        with open(save_path, "w") as f:
            json.dump(metadata, f, indent=2)
        return True
    except Exception as e:
        print(f"Error exporting model metadata: {e}")
        return False


def export_ensemble_model(models, save_path, method="weighted_average", weights=None):
    """
    Export an ensemble of models

    Args:
        models: List of models to ensemble
        save_path: Path to save the ensemble model
        method: Ensemble method (weighted_average, voting)
        weights: Weights for weighted average method

    Returns:
        True if export was successful
    """
    if weights is None and method == "weighted_average":
        # Default to equal weights
        weights = [1.0 / len(models)] * len(models)

    # Create ensemble configuration
    ensemble_config = {
        "ensemble_method": method,
        "model_count": len(models),
        "weights": weights if method == "weighted_average" else None,
        "models": [],
    }

    try:
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        # Save each model
        for i, model in enumerate(models):
            # Model file path
            model_path = os.path.join(save_path, f"model_{i}.pt")

            # Save model
            torch.save(model.state_dict(), model_path)

            # Add to ensemble configuration
            ensemble_config["models"].append(
                {
                    "index": i,
                    "model_path": f"model_{i}.pt",
                    "architecture": model.__class__.__name__,
                }
            )

        # Save ensemble configuration
        config_path = os.path.join(save_path, "ensemble_config.json")
        with open(config_path, "w") as f:
            json.dump(ensemble_config, f, indent=2)

        print(f"Ensemble model exported to: {save_path}")
        return True
    except Exception as e:
        print(f"Error exporting ensemble model: {e}")
        return False


class ModelExporter:
    """Utility class for exporting models in various formats with enhanced optimization capabilities"""

    @staticmethod
    def to_onnx(
        model,
        save_path,
        input_shape=(1, 1, 128, 128),
        dynamic_axes=None,
        optimize=True,
        opset_version=12,
    ):
        """Export model to ONNX format with optimization"""
        return export_to_onnx(
            model, save_path, input_shape, dynamic_axes, optimize, opset_version
        )

    @staticmethod
    def to_torchscript(model, save_path, example_input=None, optimize=True):
        """
        Export model to TorchScript format with optimization

        Args:
            model: PyTorch model to export
            save_path: Path to save the TorchScript model
            example_input: Example input tensor for tracing
            optimize: Whether to optimize the model
        """
        if hasattr(model, "eval"):
            model.eval()

        if example_input is None:
            # Default input if none provided
            example_input = torch.randn(1, 1, 128, 128)

        try:
            # Use tracing method
            traced_model = torch.jit.trace(model, example_input)

            # Apply optimizations if requested
            if optimize:
                traced_model = torch.jit.optimize_for_inference(traced_model)

            traced_model.save(save_path)

            print(f"Model exported to TorchScript format: {save_path}")
            return True
        except Exception as e:
            print(f"Error exporting model to TorchScript: {e}")
            return False

    @staticmethod
    def to_quantized(
        model,
        save_path,
        calibration_data=None,
        quantization_dtype=torch.qint8,
        quantization_scheme="static",
        qconfig_name="fbgemm",
    ):
        """
        Export quantized model for edge devices with enhanced options

        Args:
            model: PyTorch model to quantize
            save_path: Path to save the quantized model
            calibration_data: Data for calibration in static quantization
            quantization_dtype: Data type for quantization (qint8 or quint8)
            quantization_scheme: 'static', 'dynamic', or 'qat' (quantization-aware training)
            qconfig_name: Quantization configuration name ('fbgemm' for x86 or 'qnnpack' for ARM)
        """
        if hasattr(model, "eval"):
            model.eval()

        try:
            # Make a copy of the model to avoid modifying the original
            model_copy = copy.deepcopy(model)

            if quantization_scheme == "dynamic":
                # Dynamic quantization - simpler but less accurate
                quantized_model = torch.quantization.quantize_dynamic(
                    model_copy,
                    {
                        torch.nn.Linear,
                        torch.nn.LSTM,
                        torch.nn.LSTMCell,
                        torch.nn.RNNCell,
                        torch.nn.GRUCell,
                    },
                    dtype=quantization_dtype,
                )
            else:  # static or qat
                # Quantization configuration
                qconfig = torch.quantization.get_default_qconfig(qconfig_name)
                torch.quantization.qconfig.default_qconfig = qconfig

                # Prepare for quantization
                model_prepared = torch.quantization.prepare(model_copy)

                # Calibrate with data if provided
                if calibration_data is not None:
                    with torch.no_grad():
                        for data in calibration_data:
                            model_prepared(data)

                # Convert to quantized model
                quantized_model = torch.quantization.convert(model_prepared)

            # Save quantized model
            torch.save(quantized_model.state_dict(), save_path)

            print(f"Model exported to quantized format: {save_path}")
            return True
        except Exception as e:
            print(f"Error exporting quantized model: {e}")
            return False

    @staticmethod
    def to_tensorrt(
        model,
        save_path,
        input_shape=(1, 1, 128, 128),
        onnx_path=None,
        precision="fp32",
        workspace_size=1 << 30,
    ):
        """
        Export model to TensorRT format for GPU acceleration

        Args:
            model: PyTorch model to export
            save_path: Path to save the TensorRT model
            input_shape: Shape of the input tensor
            onnx_path: Path to an existing ONNX model, or None to create one
            precision: Precision to use ('fp32', 'fp16', or 'int8')
            workspace_size: Maximum workspace size for TensorRT
        """
        if not TENSORRT_AVAILABLE:
            print("TensorRT is not available. Please install it first.")
            return False

        try:
            # First export to ONNX if not provided
            if onnx_path is None:
                onnx_path = save_path.replace(".trt", ".onnx")
                export_to_onnx(model, onnx_path, input_shape)

            # Create TensorRT engine
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            config = builder.create_builder_config()
            config.max_workspace_size = workspace_size

            # Set precision
            if precision == "fp16" and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            elif precision == "int8" and builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                # Would need to add an int8 calibrator here for INT8 precision

            parser = trt.OnnxParser(network, logger)
            with open(onnx_path, "rb") as model_file:
                parser.parse(model_file.read())

            engine = builder.build_engine(network, config)

            # Save engine
            with open(save_path, "wb") as f:
                f.write(engine.serialize())

            print(f"Model exported to TensorRT format: {save_path}")
            return True
        except Exception as e:
            print(f"Error exporting model to TensorRT: {e}")
            return False

    @staticmethod
    def to_coreml(model, save_path, input_shape=(1, 1, 128, 128), compute_units="ALL"):
        """
        Export model to CoreML format for Apple devices

        Args:
            model: PyTorch model to export
            save_path: Path to save the CoreML model
            input_shape: Shape of the input tensor
            compute_units: CoreML compute units ('ALL', 'CPU_ONLY', 'CPU_AND_GPU', 'CPU_AND_NE')
        """
        if not COREML_AVAILABLE:
            print("CoreML Tools is not available. Please install it first.")
            return False

        try:
            # Ensure model is in evaluation mode
            if hasattr(model, "eval"):
                model.eval()

            # Create dummy input
            dummy_input = torch.randn(input_shape)

            # Trace the model with torch.jit
            traced_model = torch.jit.trace(model, dummy_input)

            # Convert to CoreML
            input_name = "input"
            output_name = "output"

            coreml_model = ct.convert(
                traced_model,
                inputs=[ct.TensorType(name=input_name, shape=input_shape)],
                compute_units=getattr(ct.ComputeUnit, compute_units),
            )

            # Save the model
            coreml_model.save(save_path)

            print(f"Model exported to CoreML format: {save_path}")
            return True
        except Exception as e:
            print(f"Error exporting model to CoreML: {e}")
            return False
            print(f"Quantized model exported to: {save_path}")
            return True
        except Exception as e:
            print(f"Error exporting quantized model: {e}")
            return False
