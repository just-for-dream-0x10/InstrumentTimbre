"""

Intelligent Music Editing and Optimization AI System - Main Entry

Emotion-Driven AI Tool for Music Editing and Optimization.
"""

import numpy as np
import librosa
import argparse
import os
import sys
import time
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any

# add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.controller import MusicEditingController, MusicEditRequest
from src.core.emotion_engine import EmotionType
from src.core.track_operator import OperationType, TrackRole
from src.core.workflow_orchestrator import (
    WorkflowOrchestrator, WorkflowRequest, WorkflowType, WorkflowStatus
)
from InstrumentTimbre.core.workflow_integration import (
    WorkflowIntegrationEngine, WorkflowRequest as IntegratedWorkflowRequest,
    WorkflowType as IntegratedWorkflowType
)
from src.core.exception_handler import (
    ExceptionHandler, ErrorContext, exception_handler,
    SystemException, BusinessLogicException, ValidationException
)

@exception_handler(attempt_recovery=True, reraise=False)
def main():
    """Enhanced main function with comprehensive workflow integration"""
    parser = argparse.ArgumentParser(description='Intelligent Music Editing and Optimization AI System')
    parser.add_argument('--mode', choices=['demo', 'api', 'analyze', 'edit', 'workflow'], 
                       default='demo', help='Operation mode')
    parser.add_argument('--input', type=str, help='Input audio file path')
    parser.add_argument('--output', type=str, help='Output audio file path')
    parser.add_argument('--operation', choices=['add', 'replace', 'modify', 'delete', 'enhance'],
                       default='add', help='Operation type')
    parser.add_argument('--role', choices=['bass', 'melody', 'harmony', 'rhythm'],
                       default='bass', help='Target track role')
    parser.add_argument('--instrument', type=str, default='bass_guitar', help='Instrument type')
    parser.add_argument('--volume', type=float, default=0.6, help='Volume level')
    parser.add_argument('--workflow-type', choices=['music_analysis', 'track_generation', 'audio_enhancement'],
                       default='music_analysis', help='Workflow type for workflow mode')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('music_ai_system.log')
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Intelligent Music Editing and Optimization AI System")
    
    try:
        # Initialize global exception handler
        global_handler = ExceptionHandler()
        
        # Load configuration if provided
        config = {}
        if args.config and os.path.exists(args.config):
            import json
            with open(args.config, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {args.config}")
        
        # Execute based on mode
        if args.mode == 'demo':
            run_enhanced_demo(config, global_handler)
        elif args.mode == 'api':
            run_api_server(config, global_handler)
        elif args.mode == 'workflow':
            run_workflow_mode(args, config, global_handler)
        elif args.mode == 'analyze':
            if not args.input:
                raise ValidationException("Input file path is required for analyze mode")
            analyze_audio_file_enhanced(args.input, config, global_handler)
        elif args.mode == 'edit':
            if not args.input:
                raise ValidationException("Input file path is required for edit mode")
            edit_audio_file_enhanced(
                args.input, args.output, args.operation, args.role, 
                args.instrument, args.volume, config, global_handler
            )
        
        logger.info("System execution completed successfully")
        
    except Exception as e:
        logger.error(f"System execution failed: {e}")
        context = ErrorContext(
            module_name=__name__,
            function_name="main",
            timestamp=time.time()
        )
        error_result = global_handler.handle_exception(e, context)
        logger.error(f"Error details: {error_result['user_message']}")
        return 1
    
    return 0

@exception_handler(attempt_recovery=True)
def run_enhanced_demo(config: Dict[str, Any], handler: ExceptionHandler):
    """Enhanced demo with workflow orchestration"""
    
    logger = logging.getLogger(__name__)
    logger.info("Starting enhanced demo mode")
    
    print("🎵 Intelligent Music Editing and Optimization AI System - Enhanced Demo")
    print("=" * 70)
    
    try:
        # Run async demo
        asyncio.run(_run_enhanced_demo_async(config, handler))
        
    except Exception as e:
        context = ErrorContext(
            module_name=__name__,
            function_name="run_enhanced_demo",
            timestamp=time.time()
        )
        error_result = handler.handle_exception(e, context)
        logger.error(f"Enhanced demo failed: {error_result['user_message']}")
        print(f"\n❌ Demo failed: {error_result['user_message']}")

async def _run_enhanced_demo_async(config: Dict[str, Any], handler: ExceptionHandler):
    """Async enhanced demo implementation"""
    
    logger = logging.getLogger(__name__)
    
    # Use the enhanced workflow integration engine
    try:
        integration_engine = WorkflowIntegrationEngine(config)
        logger.info("Using enhanced workflow integration engine")
    except Exception as e:
        logger.warning(f"Failed to initialize integration engine: {e}, falling back to basic orchestrator")
        orchestrator = WorkflowOrchestrator(config)
    
    try:
        # Generate test audio
        print("\n1. Generating test audio...")
        audio_data, sr = _generate_test_audio()
        print(f"✅ Test audio generated (duration: {len(audio_data)/sr:.1f}s)")
        
        # Run music analysis workflow
        print("\n2. Running music analysis workflow...")
        
        analysis_request = WorkflowRequest(
            workflow_type=WorkflowType.MUSIC_ANALYSIS,
            input_data={
                "audio_data": audio_data,
                "sample_rate": sr
            },
            parameters={
                "analysis_depth": "comprehensive"
            }
        )
        
        analysis_result = await orchestrator.execute_workflow(analysis_request)
        
        if analysis_result.status == WorkflowStatus.COMPLETED:
            print("✅ Music analysis completed")
            analysis_data = analysis_result.result_data.get("results", {})
            
            if "emotion_analysis" in analysis_data:
                emotion_result = analysis_data["emotion_analysis"]
                print(f"   Emotion: {emotion_result.get('primary', 'unknown')}")
                print(f"   Intensity: {emotion_result.get('intensity', 0):.3f}")
            
            if "feature_extraction" in analysis_data:
                features = analysis_data["feature_extraction"]
                print(f"   Key Features: {list(features.keys())[:3]}")
        else:
            print(f"❌ Analysis failed: {analysis_result.error_info}")
        
        print("\n🎉 Enhanced demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Enhanced demo execution failed: {e}")
        raise
    
    finally:
        orchestrator.shutdown()

def _generate_test_audio():
    """Generate enhanced test audio for demo"""
    
    sr = 22050
    duration = 8
    t = np.linspace(0, duration, sr * duration)
    
    # Create chord progression
    audio = np.zeros_like(t)
    chord_duration = duration / 4
    
    # Define chord frequencies
    chords = [
        [261.63, 329.63, 392.00],  # C major
        [220.00, 261.63, 329.63],  # A minor
        [174.61, 220.00, 261.63],  # F major
        [196.00, 246.94, 293.66]   # G major
    ]
    
    for i, chord_freqs in enumerate(chords):
        start_time = i * chord_duration
        end_time = (i + 1) * chord_duration
        
        mask = (t >= start_time) & (t < end_time)
        
        # Create chord with harmonics
        chord_signal = np.zeros_like(t)
        for freq in chord_freqs:
            chord_signal += np.sin(2 * np.pi * freq * t) * 0.3
        
        audio[mask] = chord_signal[mask]
    
    return audio, sr

@exception_handler(attempt_recovery=True)
async def run_workflow_mode(args, config: Dict[str, Any], handler: ExceptionHandler):
    """Run system in workflow mode with comprehensive orchestration"""
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting workflow mode: {args.workflow_type}")
    
    # Initialize workflow orchestrator
    orchestrator = WorkflowOrchestrator(config)
    
    try:
        # Load input audio if provided
        audio_data = None
        sample_rate = 22050
        
        if args.input:
            if not os.path.exists(args.input):
                raise ValidationException(f"Input file not found: {args.input}")
            
            audio_data, sample_rate = librosa.load(args.input, sr=sample_rate)
            logger.info(f"Loaded audio file: {args.input} (duration: {len(audio_data)/sample_rate:.2f}s)")
        
        # Create workflow request
        workflow_type = WorkflowType(args.workflow_type.upper())
        
        request = WorkflowRequest(
            workflow_type=workflow_type,
            input_data={
                "audio_data": audio_data,
                "sample_rate": sample_rate,
                "file_path": args.input
            },
            parameters={
                "operation": args.operation,
                "target_role": args.role,
                "instrument": args.instrument,
                "volume": args.volume,
                "output_path": args.output
            }
        )
        
        # Execute workflow
        logger.info("Executing workflow...")
        result = await orchestrator.execute_workflow(request)
        
        # Process results
        if result.status == WorkflowStatus.COMPLETED:
            logger.info(f"Workflow completed successfully in {result.execution_time:.2f}s")
            
            # Display metrics
            print("\n" + "="*50)
            print("🎵 WORKFLOW EXECUTION SUMMARY")
            print("="*50)
            print(f"Workflow ID: {result.workflow_id}")
            print(f"Type: {result.workflow_type.value}")
            print(f"Status: {result.status.value}")
            print(f"Execution Time: {result.execution_time:.2f}s")
            
        else:
            logger.error(f"Workflow failed: {result.error_info}")
            print(f"\n❌ Workflow failed: {result.error_info.get('message', 'Unknown error')}")
        
    except Exception as e:
        context = ErrorContext(
            module_name=__name__,
            function_name="run_workflow_mode",
            operation_id=f"workflow_{args.workflow_type}",
            timestamp=time.time()
        )
        error_result = handler.handle_exception(e, context)
        logger.error(f"Workflow mode failed: {error_result['user_message']}")
        raise
    
    finally:
        orchestrator.shutdown()

@exception_handler(attempt_recovery=True)
def analyze_audio_file_enhanced(
    input_file: str, 
    config: Dict[str, Any], 
    handler: ExceptionHandler
):
    """Enhanced audio file analysis with workflow orchestration"""
    
    logger = logging.getLogger(__name__)
    logger.info(f"Analyzing audio file: {input_file}")
    
    print(f"🔍 Enhanced Audio Analysis: {input_file}")
    print("=" * 50)
    
    try:
        if not os.path.exists(input_file):
            raise ValidationException(f"Input file not found: {input_file}")
        
        # Run async analysis
        asyncio.run(_analyze_audio_async(input_file, config, handler))
        
    except Exception as e:
        context = ErrorContext(
            module_name=__name__,
            function_name="analyze_audio_file_enhanced",
            input_data_summary={"file_path": input_file},
            timestamp=time.time()
        )
        error_result = handler.handle_exception(e, context)
        logger.error(f"Audio analysis failed: {error_result['user_message']}")
        print(f"\n❌ Analysis failed: {error_result['user_message']}")

async def _analyze_audio_async(
    input_file: str, 
    config: Dict[str, Any], 
    handler: ExceptionHandler
):
    """Async audio analysis implementation"""
    
    orchestrator = WorkflowOrchestrator(config)
    
    try:
        # Load audio
        audio_data, sr = librosa.load(input_file, sr=22050)
        print(f"✅ Audio loaded successfully - Duration: {len(audio_data)/sr:.2f}s")
        
        # Create analysis request
        request = WorkflowRequest(
            workflow_type=WorkflowType.MUSIC_ANALYSIS,
            input_data={
                "audio_data": audio_data,
                "sample_rate": sr,
                "file_path": input_file
            },
            parameters={
                "analysis_depth": "comprehensive"
            }
        )
        
        # Execute analysis workflow
        result = await orchestrator.execute_workflow(request)
        
        if result.status == WorkflowStatus.COMPLETED:
            print("\n📊 Analysis Results:")
            print(f"⏱️ Analysis completed in {result.execution_time:.2f}s")
        else:
            print(f"\n❌ Analysis failed: {result.error_info}")
    
    finally:
        orchestrator.shutdown()

@exception_handler(attempt_recovery=True)
def edit_audio_file_enhanced(
    input_file: str,
    output_file: str,
    operation: str,
    role: str,
    instrument: str,
    volume: float,
    config: Dict[str, Any],
    handler: ExceptionHandler
):
    """Enhanced audio file editing with workflow orchestration"""
    
    logger = logging.getLogger(__name__)
    logger.info(f"Editing audio file: {input_file}")
    
    print(f"🎛️ Enhanced Audio Editing: {input_file}")
    print(f"Operation: {operation}, Role: {role}, Instrument: {instrument}")
    print("=" * 50)
    
    try:
        if not os.path.exists(input_file):
            raise ValidationException(f"Input file not found: {input_file}")
        
        # Run async editing
        asyncio.run(_edit_audio_async(
            input_file, output_file, operation, role, 
            instrument, volume, config, handler
        ))
        
    except Exception as e:
        context = ErrorContext(
            module_name=__name__,
            function_name="edit_audio_file_enhanced",
            input_data_summary={
                "file_path": input_file,
                "operation": operation,
                "role": role
            },
            timestamp=time.time()
        )
        error_result = handler.handle_exception(e, context)
        logger.error(f"Audio editing failed: {error_result['user_message']}")
        print(f"\n❌ Editing failed: {error_result['user_message']}")

async def _edit_audio_async(
    input_file: str,
    output_file: str,
    operation: str,
    role: str,
    instrument: str,
    volume: float,
    config: Dict[str, Any],
    handler: ExceptionHandler
):
    """Async audio editing implementation"""
    
    orchestrator = WorkflowOrchestrator(config)
    
    try:
        # Load audio
        audio_data, sr = librosa.load(input_file, sr=22050)
        print(f"✅ Audio loaded successfully")
        
        # Determine workflow type based on operation
        if operation in ['enhance']:
            workflow_type = WorkflowType.AUDIO_ENHANCEMENT
        else:
            workflow_type = WorkflowType.TRACK_GENERATION
        
        # Create editing request
        request = WorkflowRequest(
            workflow_type=workflow_type,
            input_data={
                "audio_data": audio_data,
                "sample_rate": sr,
                "file_path": input_file
            },
            parameters={
                "operation": operation,
                "target_role": role,
                "instrument": instrument,
                "volume": volume,
                "output_path": output_file
            }
        )
        
        # Execute editing workflow
        result = await orchestrator.execute_workflow(request)
        
        if result.status == WorkflowStatus.COMPLETED:
            print("✅ Editing completed successfully!")
            print(f"⏱️ Editing completed in {result.execution_time:.2f}s")
        else:
            print(f"❌ Editing failed: {result.error_info}")
    
    finally:
        orchestrator.shutdown()

@exception_handler(attempt_recovery=True)
def run_api_server(config: Dict[str, Any], handler: ExceptionHandler):
    """Enhanced API server with workflow integration"""
    
    logger = logging.getLogger(__name__)
    logger.info("Starting enhanced API server")
    
    print("🚀 Starting Enhanced API Server...")
    print("API Address: http://localhost:5000")
    print("Health Check: http://localhost:5000/api/health")
    
    try:
        from flask import Flask, request, jsonify
        
        app = Flask(__name__)
        orchestrator = WorkflowOrchestrator(config)
        
        @app.route('/api/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            metrics = orchestrator.get_system_metrics()
            return jsonify({
                "status": "healthy",
                "timestamp": time.time(),
                "metrics": metrics
            })
        
        # Start server
        app.run(debug=False, host='0.0.0.0', port=5000)
        
    except ImportError:
        print("❌ Unable to start API server, please ensure Flask is installed")
    except Exception as e:
        context = ErrorContext(
            module_name=__name__,
            function_name="run_api_server",
            timestamp=time.time()
        )
        error_result = handler.handle_exception(e, context)
        print(f"❌ API server startup failed: {error_result['user_message']}")

def run_demo():
    """run demo """
    print("🎵 Emotion-Driven AI Tool for Music Editing and Optimization - Demo Mode")
    print("=" * 50)

    # Create controller
    controller = MusicEditingController()

    # Generate test audio (simple chord progression)
    print("1. Generate test audio...")
    sr = 22050
    duration = 8

    # Create simple C major chord progression
    t = np.linspace(0, duration, sr * duration)

    # C - Am - F - G chord progression
    audio = np.zeros_like(t)
    chord_duration = duration / 4

    # C major chord (C-E-G)
    c_chord = (np.sin(2 * np.pi * 261.63 * t) + 
               np.sin(2 * np.pi * 329.63 * t) + 
               np.sin(2 * np.pi * 392.00 * t)) / 3

    # Add to first 2 seconds
    mask1 = (t >= 0) & (t < chord_duration)
    audio[mask1] = c_chord[mask1] * 0.3

    # Am chord (A-C-E)
    am_chord = (np.sin(2 * np.pi * 220.00 * t) +
                np.sin(2 * np.pi * 261.63 * t) + 
                np.sin(2 * np.pi * 329.63 * t)) / 3
    
    mask2 = (t >= chord_duration) & (t < 2 * chord_duration)
    audio[mask2] = am_chord[mask2] * 0.3

    # F major chord (F-A-C)
    f_chord = (np.sin(2 * np.pi * 174.61 * t) + 
               np.sin(2 * np.pi * 220.00 * t) + 
               np.sin(2 * np.pi * 261.63 * t)) / 3
    
    mask3 = (t >= 2 * chord_duration) & (t < 3 * chord_duration)
    audio[mask3] = f_chord[mask3] * 0.3

    # G major chord (G-B-D)
    g_chord = (np.sin(2 * np.pi * 196.00 * t) + 
               np.sin(2 * np.pi * 246.94 * t) + 
               np.sin(2 * np.pi * 293.66 * t)) / 3
    
    mask4 = (t >= 3 * chord_duration) & (t <= duration)
    audio[mask4] = g_chord[mask4] * 0.3

    print("✅ generate test audio (C-Am-F-G chord progression)")

    # 2. Analyze original audio
    print("\n2. Analyze original audio...")
    original_analysis = controller._analyze_audio(audio, sr)

    print(f"   Emotion: {original_analysis['emotion']['primary']}")
    print(f"   Intensity: {original_analysis['emotion']['intensity']:.3f}")
    print(f"   Confidence: {original_analysis['emotion']['confidence']:.3f}")
    print(f"   Key Signature: {original_analysis['structure']['key_signature']}")
    print(f"   Overall Form: {original_analysis['structure']['overall_form']}")

    # 3. Execute track addition operation
    print("\n3. Execute track addition operation (Add bass line)...")

    request = MusicEditRequest(
        audio_data=audio,
        sr=sr,
        operation_type="add",
        target_role="bass",
        parameters={
            'instrument': 'bass_guitar',
            'volume': 0.5,
            'style': 'walking_bass'
        },
        preserve_emotion=True,
        quality_threshold=0.7
    )
    
    response = controller.edit_music(request)
    
    print(f"   : {response.success}")
    if response.success:
        print(f"   : {response.quality_metrics}")
        print(f"   : {response.final_analysis.get('emotion', {}).get('confidence', 0):.3f}")
        print(f"   : {response.recommendations}")
        
        # save
        if response.result_audio is not None:
            output_file = "demo_output.wav"
            import soundfile as sf
            sf.write(output_file, response.result_audio, sr)
            print(f"   : {output_file}")
    else:
        print(f"   : {response.operation_log}")
    
    # 4. System status
    print("\n4. :")
    status = controller.get_system_status()
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    print("\n🎉 !")

def analyze_audio_file(input_file: str):
    """Analyze audio file"""
    print(f"🔍 : {input_file}")
    
    if not os.path.exists(input_file):
        print(f"❌ : {input_file}")
        return
    
    try:
        # load
        audio_data, sr = librosa.load(input_file, sr=22050)
        print(f"✅  - : {len(audio_data)/sr:.2f}")
        
        # create
        controller = MusicEditingController()
        analysis = controller._analyze_audio(audio_data, sr)
        
        # Analysis results
        print("\n📊 :")
        print(f":")
        print(f"  : {analysis['emotion']['primary']}")
        print(f"  : {analysis['emotion']['intensity']:.3f}")
        print(f"  : {analysis['emotion']['confidence']:.3f}")
        
        print(f"\n:")
        print(f"  : {analysis['structure']['key_signature']}")
        print(f"  : {analysis['structure']['tempo']:.1f} BPM")
        print(f"  : {analysis['structure']['overall_form']}")
        
        print(f"\n:")
        for feature, value in analysis['features'].items():
            print(f"  {feature}: {value:.4f}")
        
    except Exception as e:
        print(f"❌ : {str(e)}")

def edit_audio_file(input_file: str, output_file: str, operation: str, 
                   role: str, instrument: str, volume: float):
    """Edit audio file"""
    print(f"🎛️ : {input_file}")
    print(f": {operation}, : {role}, : {instrument}")
    
    if not os.path.exists(input_file):
        print(f"❌ : {input_file}")
        return
    
    try:
        # load
        audio_data, sr = librosa.load(input_file, sr=22050)
        print(f"✅ ")
        
        # create
        request = MusicEditRequest(
            audio_data=audio_data,
            sr=sr,
            operation_type=operation,
            target_role=role,
            parameters={
                'instrument': instrument,
                'volume': volume
            },
            preserve_emotion=True,
            quality_threshold=0.7
        )
        
        # 
        controller = MusicEditingController()
        response = controller.edit_music(request)
        
        if response.success:
            print("✅ !")
            
            # save
            if output_file and response.result_audio is not None:
                import soundfile as sf
                sf.write(output_file, response.result_audio, sr)
                print(f"📁 : {output_file}")
            
            # Quality metrics
            print(f"📊 : {response.quality_metrics}")
            print(f"💡 : {response.recommendations}")
            
        else:
            print(f"❌ : {response.operation_log}")
            
    except Exception as e:
        print(f"❌ : {str(e)}")

def run_api_server():
    """Run API server"""
    print("🚀 API...")
    print("API: http://localhost:5000")
    print(": http://localhost:5000/api/health")
    
    try:
        from src.api.music_api import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except ImportError:
        print("❌ API，Flask")
    except Exception as e:
        print(f"❌ API: {str(e)}")

if __name__ == "__main__":
    main()