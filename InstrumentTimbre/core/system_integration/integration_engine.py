"""
Integration Engine

Main system integration engine that provides unified access to all InstrumentTimbre
functionality with comprehensive error handling and monitoring.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum

from .module_coordinator import ModuleCoordinator, ProcessingPipeline
from .error_handler import SystemErrorHandler, ErrorContext, handle_errors
from .system_monitor import SystemMonitor
from .exception_types import *
try:
    from config import get_config, validate_config, UltimateConfig
    Config = UltimateConfig
except ImportError:
    # Fallback for testing environment
    def get_config():
        return {}
    def validate_config(config):
        return True
    Config = dict


class OperationType(Enum):
    """Types of operations the system can perform"""
    MUSIC_GENERATION = "music_generation"
    AUDIO_ENHANCEMENT = "audio_enhancement"
    TRACK_OPERATIONS = "track_operations"
    MUSIC_ANALYSIS = "music_analysis"
    STYLE_TRANSFER = "style_transfer"
    QUALITY_ASSURANCE = "quality_assurance"
    BATCH_PROCESSING = "batch_processing"


@dataclass
class OperationRequest:
    """Request for system operation"""
    operation_type: OperationType
    input_data: Dict[str, Any]
    parameters: Dict[str, Any]
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    priority: str = "normal"  # low, normal, high, critical
    timeout: Optional[float] = None


@dataclass
class OperationResult:
    """Result of system operation"""
    operation_id: str
    operation_type: OperationType
    success: bool
    result_data: Optional[Dict[str, Any]] = None
    error_info: Optional[Dict[str, Any]] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = None


class IntegrationEngine:
    """
    Main integration engine that provides unified access to all InstrumentTimbre
    functionality with comprehensive error handling, monitoring, and optimization.
    """
    
    def __init__(self, config: Optional[Config] = None, enable_monitoring: bool = True):
        """
        Initialize the integration engine
        
        Args:
            config: System configuration
            enable_monitoring: Whether to enable system monitoring
        """
        
        # Configuration
        self.config = config or get_config()
        if not validate_config(self.config):
            raise InvalidConfigError("Invalid system configuration")
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.module_coordinator = ModuleCoordinator(max_workers=self.config.num_workers)
        self.error_handler = SystemErrorHandler()
        
        # System monitoring
        self.monitor = None
        if enable_monitoring:
            self.monitor = SystemMonitor(monitoring_interval=5.0)
            self.monitor.start_monitoring()
            
            # Add alert callback for critical issues
            self.monitor.add_alert_callback(self._handle_system_alert)
        
        # Operation tracking
        self.operation_counter = 0
        self.active_operations: Dict[str, OperationRequest] = {}
        self.completed_operations: List[OperationResult] = []
        
        # Performance tracking
        self.performance_stats = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "average_processing_time": 0.0,
            "operations_by_type": {}
        }
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the system and verify all components are working"""
        
        self.logger.info("Initializing InstrumentTimbre integration engine...")
        
        try:
            # Check system resources
            if self.monitor:
                health = self.monitor.get_system_health()
                if health["score"] < 50:
                    self.logger.warning(f"System health is poor (score: {health['score']})")
                    for issue in health["issues"]:
                        self.logger.warning(f"Health issue: {issue}")
            
            # Initialize core modules
            self.logger.info("Initializing core modules...")
            core_modules = ["feature_extractor", "model_manager"]
            
            for module_name in core_modules:
                success = self.module_coordinator.initialize_module(module_name)
                if not success:
                    raise ModuleIntegrationError(f"Failed to initialize core module: {module_name}")
            
            # Perform basic health check
            health_results = self.module_coordinator.health_check()
            if not health_results["overall_healthy"]:
                self.logger.warning("Some modules are not healthy")
                for module, status in health_results["modules"].items():
                    if not status["healthy"]:
                        self.logger.warning(f"Module {module} health check failed")
            
            self.logger.info("Integration engine initialization completed successfully")
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            raise SystemResourceError(f"Failed to initialize system: {e}")
    
    def _handle_system_alert(self, alert_info: Dict[str, Any]):
        """Handle system alerts from monitor"""
        
        severity = alert_info.get("severity", "info")
        message = alert_info.get("message", "Unknown alert")
        
        if severity == "critical":
            self.logger.critical(f"CRITICAL SYSTEM ALERT: {message}")
            # Could implement automatic recovery actions here
        else:
            self.logger.warning(f"System alert ({severity}): {message}")
    
    @handle_errors(auto_recover=True, context_module="integration_engine")
    def execute_operation(
        self, 
        operation_request: OperationRequest,
        callback: Optional[Callable] = None
    ) -> OperationResult:
        """
        Execute a system operation with full error handling and monitoring
        
        Args:
            operation_request: The operation to execute
            callback: Optional callback for progress updates
            
        Returns:
            Operation result with success status and data
        """
        
        start_time = time.time()
        operation_id = f"op_{self.operation_counter:06d}_{int(time.time())}"
        self.operation_counter += 1
        
        self.logger.info(f"Executing operation {operation_id}: {operation_request.operation_type.value}")
        
        # Track active operation
        self.active_operations[operation_id] = operation_request
        
        try:
            # Validate operation request
            self._validate_operation_request(operation_request)
            
            # Check system resources
            if self.monitor:
                resource_check = self._check_operation_resources(operation_request)
                if not resource_check["can_proceed"]:
                    raise SystemResourceError(
                        f"Insufficient resources for operation: {resource_check['blockers']}"
                    )
                
                # Log warnings if any
                for warning in resource_check["warnings"]:
                    self.logger.warning(f"Resource warning: {warning}")
            
            # Execute operation based on type
            result_data = self._execute_operation_by_type(operation_request, callback)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create successful result
            result = OperationResult(
                operation_id=operation_id,
                operation_type=operation_request.operation_type,
                success=True,
                result_data=result_data,
                processing_time=processing_time,
                metadata={
                    "config_used": self.config.__dict__,
                    "modules_used": self._get_modules_used(operation_request.operation_type),
                    "system_health": self.monitor.get_system_health() if self.monitor else None
                }
            )
            
            # Update statistics
            self._update_performance_stats(operation_request.operation_type, True, processing_time)
            
            self.logger.info(f"Operation {operation_id} completed successfully in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Handle error through error handler
            error_context = ErrorContext(
                module_name="integration_engine",
                function_name="execute_operation",
                operation_id=operation_id,
                user_id=operation_request.user_id,
                session_id=operation_request.session_id,
                input_data_info={
                    "operation_type": operation_request.operation_type.value,
                    "input_keys": list(operation_request.input_data.keys()),
                    "parameter_count": len(operation_request.parameters)
                }
            )
            
            error_result = self.error_handler.handle_error(e, error_context, auto_recover=True)
            
            # Create error result
            result = OperationResult(
                operation_id=operation_id,
                operation_type=operation_request.operation_type,
                success=error_result["recovery_successful"],
                result_data=None,
                error_info=error_result,
                processing_time=processing_time
            )
            
            # Update statistics
            self._update_performance_stats(operation_request.operation_type, False, processing_time)
            
            if not error_result["recovery_successful"]:
                self.logger.error(f"Operation {operation_id} failed: {error_result['user_message']}")
            else:
                self.logger.info(f"Operation {operation_id} recovered from error")
            
            return result
            
        finally:
            # Remove from active operations
            self.active_operations.pop(operation_id, None)
            
            # Add to completed operations (keep last 1000)
            self.completed_operations.append(result)
            if len(self.completed_operations) > 1000:
                self.completed_operations = self.completed_operations[-1000:]
    
    def _validate_operation_request(self, request: OperationRequest):
        """Validate operation request"""
        
        if not request.input_data:
            raise InvalidParameterError("input_data", None)
        
        if not isinstance(request.parameters, dict):
            raise InvalidParameterError("parameters", request.parameters)
        
        # Type-specific validation
        if request.operation_type == OperationType.MUSIC_GENERATION:
            if "style" not in request.parameters:
                raise InvalidParameterError("style", None)
        
        elif request.operation_type == OperationType.AUDIO_ENHANCEMENT:
            if "audio_data" not in request.input_data:
                raise InvalidParameterError("audio_data", None)
    
    def _check_operation_resources(self, request: OperationRequest) -> Dict[str, Any]:
        """Check if system has sufficient resources for operation"""
        
        # Define resource requirements for different operations
        resource_requirements = {
            OperationType.MUSIC_GENERATION: {
                "memory_mb": 1024,
                "cpu_percent": 30,
                "gpu_memory_mb": 512 if self.config.use_gpu else 0
            },
            OperationType.AUDIO_ENHANCEMENT: {
                "memory_mb": 512,
                "cpu_percent": 20
            },
            OperationType.BATCH_PROCESSING: {
                "memory_mb": 2048,
                "cpu_percent": 50,
                "gpu_memory_mb": 1024 if self.config.use_gpu else 0
            }
        }
        
        requirements = resource_requirements.get(request.operation_type, {})
        
        return self.monitor.check_resource_requirements(
            request.operation_type.value,
            **requirements
        )
    
    def _execute_operation_by_type(
        self, 
        request: OperationRequest, 
        callback: Optional[Callable]
    ) -> Dict[str, Any]:
        """Execute operation based on its type"""
        
        operation_type = request.operation_type
        
        if operation_type == OperationType.MUSIC_GENERATION:
            return self._execute_music_generation(request, callback)
        
        elif operation_type == OperationType.AUDIO_ENHANCEMENT:
            return self._execute_audio_enhancement(request, callback)
        
        elif operation_type == OperationType.TRACK_OPERATIONS:
            return self._execute_track_operations(request, callback)
        
        elif operation_type == OperationType.MUSIC_ANALYSIS:
            return self._execute_music_analysis(request, callback)
        
        elif operation_type == OperationType.STYLE_TRANSFER:
            return self._execute_style_transfer(request, callback)
        
        elif operation_type == OperationType.QUALITY_ASSURANCE:
            return self._execute_quality_assurance(request, callback)
        
        elif operation_type == OperationType.BATCH_PROCESSING:
            return self._execute_batch_processing(request, callback)
        
        else:
            raise InvalidParameterError("operation_type", operation_type)
    
    def _execute_music_generation(self, request: OperationRequest, callback: Optional[Callable]) -> Dict[str, Any]:
        """Execute music generation operation"""
        
        pipeline_result = self.module_coordinator.execute_pipeline(
            "complete_music_processing",
            request.input_data,
            **request.parameters
        )
        
        return {
            "generated_audio": pipeline_result.get("music_generator"),
            "analysis": pipeline_result.get("music_analyzer"),
            "quality_score": pipeline_result.get("quality_assurance"),
            "pipeline_metadata": pipeline_result
        }
    
    def _execute_audio_enhancement(self, request: OperationRequest, callback: Optional[Callable]) -> Dict[str, Any]:
        """Execute audio enhancement operation"""
        
        pipeline_result = self.module_coordinator.execute_pipeline(
            "audio_enhancement",
            request.input_data,
            **request.parameters
        )
        
        return {
            "enhanced_audio": pipeline_result.get("audio_processor"),
            "quality_metrics": pipeline_result.get("quality_assurance"),
            "processing_metadata": pipeline_result
        }
    
    def _execute_track_operations(self, request: OperationRequest, callback: Optional[Callable]) -> Dict[str, Any]:
        """Execute track operations (repair, replace, generate)"""
        
        # Initialize track operations module
        self.module_coordinator.initialize_module("track_operations")
        
        # Execute operation
        track_ops_instance = self.module_coordinator.module_instances["track_operations"]
        result = track_ops_instance.process(request.input_data, **request.parameters)
        
        return {
            "operation_result": result,
            "tracks_processed": len(request.input_data.get("tracks", {})),
            "operations_applied": request.parameters.get("operations", [])
        }
    
    def _execute_music_analysis(self, request: OperationRequest, callback: Optional[Callable]) -> Dict[str, Any]:
        """Execute music analysis operation"""
        
        pipeline_result = self.module_coordinator.execute_pipeline(
            "music_analysis",
            request.input_data,
            **request.parameters
        )
        
        return {
            "analysis_result": pipeline_result.get("music_analyzer"),
            "features": pipeline_result.get("feature_extractor"),
            "analysis_metadata": pipeline_result
        }
    
    def _execute_style_transfer(self, request: OperationRequest, callback: Optional[Callable]) -> Dict[str, Any]:
        """Execute style transfer operation"""
        
        # Custom pipeline for style transfer
        style_modules = ["feature_extractor", "model_manager", "music_generator", "audio_processor"]
        
        results = {}
        for module_name in style_modules:
            self.module_coordinator.initialize_module(module_name)
            module_instance = self.module_coordinator.module_instances[module_name]
            result = module_instance.process(request.input_data, **request.parameters)
            results[module_name] = result
        
        return {
            "transferred_audio": results.get("audio_processor"),
            "style_analysis": results.get("music_generator"),
            "processing_chain": results
        }
    
    def _execute_quality_assurance(self, request: OperationRequest, callback: Optional[Callable]) -> Dict[str, Any]:
        """Execute quality assurance operation"""
        
        self.module_coordinator.initialize_module("quality_assurance")
        qa_instance = self.module_coordinator.module_instances["quality_assurance"]
        
        result = qa_instance.process(request.input_data, **request.parameters)
        
        return {
            "quality_report": result,
            "score": result.get("overall_score", 0),
            "recommendations": result.get("recommendations", [])
        }
    
    def _execute_batch_processing(self, request: OperationRequest, callback: Optional[Callable]) -> Dict[str, Any]:
        """Execute batch processing operation"""
        
        batch_items = request.input_data.get("batch_items", [])
        batch_results = []
        
        for i, item in enumerate(batch_items):
            try:
                # Create sub-operation request
                sub_request = OperationRequest(
                    operation_type=OperationType(request.parameters.get("batch_operation_type", "music_analysis")),
                    input_data=item,
                    parameters=request.parameters.get("item_parameters", {}),
                    user_id=request.user_id,
                    session_id=request.session_id
                )
                
                # Execute sub-operation
                sub_result = self._execute_operation_by_type(sub_request, callback)
                batch_results.append({
                    "index": i,
                    "success": True,
                    "result": sub_result
                })
                
                # Progress callback
                if callback:
                    callback({
                        "progress": (i + 1) / len(batch_items),
                        "current_item": i + 1,
                        "total_items": len(batch_items)
                    })
                    
            except Exception as e:
                batch_results.append({
                    "index": i,
                    "success": False,
                    "error": str(e)
                })
        
        successful_count = sum(1 for r in batch_results if r["success"])
        
        return {
            "batch_results": batch_results,
            "total_items": len(batch_items),
            "successful_items": successful_count,
            "failed_items": len(batch_items) - successful_count,
            "success_rate": successful_count / len(batch_items) if batch_items else 0
        }
    
    def _get_modules_used(self, operation_type: OperationType) -> List[str]:
        """Get list of modules used for an operation type"""
        
        module_mapping = {
            OperationType.MUSIC_GENERATION: ["feature_extractor", "model_manager", "music_generator", "audio_processor"],
            OperationType.AUDIO_ENHANCEMENT: ["audio_processor", "quality_assurance"],
            OperationType.MUSIC_ANALYSIS: ["feature_extractor", "music_analyzer"],
            OperationType.TRACK_OPERATIONS: ["track_operations"],
            OperationType.STYLE_TRANSFER: ["feature_extractor", "model_manager", "music_generator"],
            OperationType.QUALITY_ASSURANCE: ["quality_assurance"],
            OperationType.BATCH_PROCESSING: ["varies"]
        }
        
        return module_mapping.get(operation_type, [])
    
    def _update_performance_stats(self, operation_type: OperationType, success: bool, processing_time: float):
        """Update performance statistics"""
        
        self.performance_stats["total_operations"] += 1
        
        if success:
            self.performance_stats["successful_operations"] += 1
        else:
            self.performance_stats["failed_operations"] += 1
        
        # Update average processing time
        total_ops = self.performance_stats["total_operations"]
        current_avg = self.performance_stats["average_processing_time"]
        self.performance_stats["average_processing_time"] = (
            (current_avg * (total_ops - 1) + processing_time) / total_ops
        )
        
        # Update by-type statistics
        type_key = operation_type.value
        if type_key not in self.performance_stats["operations_by_type"]:
            self.performance_stats["operations_by_type"][type_key] = {
                "count": 0,
                "success_count": 0,
                "average_time": 0.0
            }
        
        type_stats = self.performance_stats["operations_by_type"][type_key]
        type_stats["count"] += 1
        
        if success:
            type_stats["success_count"] += 1
        
        # Update type-specific average time
        type_count = type_stats["count"]
        current_type_avg = type_stats["average_time"]
        type_stats["average_time"] = (
            (current_type_avg * (type_count - 1) + processing_time) / type_count
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        return {
            "system_health": self.monitor.get_system_health() if self.monitor else {"status": "monitoring_disabled"},
            "module_status": self.module_coordinator.get_module_status(),
            "active_operations": len(self.active_operations),
            "performance_stats": self.performance_stats,
            "error_stats": self.error_handler.get_error_statistics(),
            "config": {
                "quality": self.config.quality.value,
                "use_gpu": self.config.use_gpu,
                "sample_rate": self.config.sample_rate,
                "num_workers": self.config.num_workers
            }
        }
    
    def shutdown(self):
        """Shutdown the integration engine and cleanup resources"""
        
        self.logger.info("Shutting down integration engine...")
        
        # Stop monitoring
        if self.monitor:
            self.monitor.stop_monitoring()
        
        # Shutdown module coordinator
        self.module_coordinator.shutdown()
        
        self.logger.info("Integration engine shutdown complete")
    
    def __del__(self):
        """Cleanup when engine is destroyed"""
        try:
            self.shutdown()
        except:
            pass  # Ignore errors during cleanup