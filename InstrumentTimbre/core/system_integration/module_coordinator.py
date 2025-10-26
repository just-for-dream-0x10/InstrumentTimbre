"""
Module Coordinator

Manages coordination and communication between different InstrumentTimbre modules,
handles dependencies, and ensures proper execution order.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Set, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, Future
import threading

from .exception_types import (
    ModuleIntegrationError, ModuleNotAvailableError, ModuleVersionError,
    InstrumentTimbreError
)
from .error_handler import SystemErrorHandler, ErrorContext


class ModuleStatus(Enum):
    """Module status states"""
    NOT_INITIALIZED = "not_initialized"
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    DISABLED = "disabled"


class ModulePriority(Enum):
    """Module execution priority"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


@dataclass
class ModuleInfo:
    """Information about a registered module"""
    name: str
    module_type: str
    version: str
    status: ModuleStatus = ModuleStatus.NOT_INITIALIZED
    priority: ModulePriority = ModulePriority.NORMAL
    dependencies: List[str] = field(default_factory=list)
    initialization_function: Optional[Callable] = None
    health_check_function: Optional[Callable] = None
    cleanup_function: Optional[Callable] = None
    last_health_check: Optional[float] = None
    error_count: int = 0
    processing_count: int = 0
    last_error: Optional[str] = None


@dataclass
class ProcessingPipeline:
    """Defines a processing pipeline with multiple modules"""
    name: str
    modules: List[str]
    parallel_groups: List[List[str]] = field(default_factory=list)
    timeout: float = 300.0
    retry_on_failure: bool = True
    max_retries: int = 3


class ModuleCoordinator:
    """
    Coordinates execution of multiple modules ensuring proper dependency management,
    error handling, and optimal resource utilization.
    """
    
    def __init__(self, max_workers: int = 4):
        self.logger = logging.getLogger(__name__)
        self.error_handler = SystemErrorHandler()
        
        # Module registry
        self.modules: Dict[str, ModuleInfo] = {}
        self.module_instances: Dict[str, Any] = {}
        
        # Execution management
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.active_tasks: Dict[str, Future] = {}
        self.pipeline_registry: Dict[str, ProcessingPipeline] = {}
        
        # Synchronization
        self.coordination_lock = threading.RLock()
        
        # Statistics
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0
        }
        
        # Initialize core modules
        self._register_core_modules()
    
    def _register_core_modules(self):
        """Register core InstrumentTimbre modules"""
        
        core_modules = [
            # Feature extraction
            {
                "name": "feature_extractor",
                "module_type": "features",
                "version": "1.0.0",
                "priority": ModulePriority.CRITICAL,
                "dependencies": []
            },
            
            # Models
            {
                "name": "model_manager",
                "module_type": "models",
                "version": "1.0.0",
                "priority": ModulePriority.CRITICAL,
                "dependencies": ["feature_extractor"]
            },
            
            # Audio processing
            {
                "name": "audio_processor",
                "module_type": "professional_audio",
                "version": "1.0.0",
                "priority": ModulePriority.HIGH,
                "dependencies": []
            },
            
            # Generation
            {
                "name": "music_generator",
                "module_type": "generation",
                "version": "1.0.0",
                "priority": ModulePriority.HIGH,
                "dependencies": ["model_manager", "feature_extractor"]
            },
            
            # Operations
            {
                "name": "track_operations",
                "module_type": "operations",
                "version": "1.0.0",
                "priority": ModulePriority.NORMAL,
                "dependencies": ["music_generator", "audio_processor"]
            },
            
            # Quality assurance
            {
                "name": "quality_assurance",
                "module_type": "quality_assurance",
                "version": "1.0.0",
                "priority": ModulePriority.HIGH,
                "dependencies": ["audio_processor"]
            },
            
            # Analysis
            {
                "name": "music_analyzer",
                "module_type": "analysis",
                "version": "1.0.0",
                "priority": ModulePriority.NORMAL,
                "dependencies": ["feature_extractor"]
            }
        ]
        
        for module_def in core_modules:
            self.register_module(**module_def)
        
        # Define standard processing pipelines
        self._register_standard_pipelines()
    
    def _register_standard_pipelines(self):
        """Register standard processing pipelines"""
        
        # Complete music processing pipeline
        self.register_pipeline(ProcessingPipeline(
            name="complete_music_processing",
            modules=[
                "feature_extractor",
                "model_manager", 
                "music_analyzer",
                "music_generator",
                "audio_processor",
                "quality_assurance"
            ],
            parallel_groups=[
                ["music_analyzer"],  # Can run in parallel with generation
                ["audio_processor", "quality_assurance"]  # Can run in parallel
            ],
            timeout=600.0
        ))
        
        # Fast preview pipeline
        self.register_pipeline(ProcessingPipeline(
            name="fast_preview",
            modules=[
                "feature_extractor",
                "model_manager",
                "music_generator"
            ],
            timeout=60.0,
            max_retries=1
        ))
        
        # Audio enhancement only pipeline
        self.register_pipeline(ProcessingPipeline(
            name="audio_enhancement",
            modules=[
                "audio_processor",
                "quality_assurance"
            ],
            parallel_groups=[["audio_processor", "quality_assurance"]],
            timeout=120.0
        ))
        
        # Analysis only pipeline
        self.register_pipeline(ProcessingPipeline(
            name="music_analysis",
            modules=[
                "feature_extractor",
                "music_analyzer"
            ],
            timeout=30.0
        ))
    
    def register_module(
        self,
        name: str,
        module_type: str,
        version: str,
        priority: ModulePriority = ModulePriority.NORMAL,
        dependencies: Optional[List[str]] = None,
        initialization_function: Optional[Callable] = None,
        health_check_function: Optional[Callable] = None,
        cleanup_function: Optional[Callable] = None
    ):
        """Register a module for coordination"""
        
        with self.coordination_lock:
            if name in self.modules:
                self.logger.warning(f"Module {name} already registered, updating...")
            
            module_info = ModuleInfo(
                name=name,
                module_type=module_type,
                version=version,
                priority=priority,
                dependencies=dependencies or [],
                initialization_function=initialization_function,
                health_check_function=health_check_function,
                cleanup_function=cleanup_function
            )
            
            self.modules[name] = module_info
            self.logger.info(f"Registered module: {name} ({module_type} v{version})")
    
    def register_pipeline(self, pipeline: ProcessingPipeline):
        """Register a processing pipeline"""
        
        # Validate pipeline modules exist
        for module_name in pipeline.modules:
            if module_name not in self.modules:
                raise ModuleIntegrationError(
                    f"Pipeline {pipeline.name} references unknown module: {module_name}"
                )
        
        self.pipeline_registry[pipeline.name] = pipeline
        self.logger.info(f"Registered pipeline: {pipeline.name}")
    
    def initialize_module(self, module_name: str, **kwargs) -> bool:
        """Initialize a specific module"""
        
        with self.coordination_lock:
            if module_name not in self.modules:
                raise ModuleNotAvailableError(module_name)
            
            module_info = self.modules[module_name]
            
            if module_info.status == ModuleStatus.READY:
                self.logger.info(f"Module {module_name} already initialized")
                return True
            
            self.logger.info(f"Initializing module: {module_name}")
            module_info.status = ModuleStatus.INITIALIZING
            
            try:
                # Check dependencies first
                self._ensure_dependencies(module_name)
                
                # Initialize the module
                if module_info.initialization_function:
                    result = module_info.initialization_function(**kwargs)
                    if result is not None:
                        self.module_instances[module_name] = result
                else:
                    # Try to import and initialize the module dynamically
                    module_instance = self._dynamic_module_initialization(module_name, module_info)
                    if module_instance:
                        self.module_instances[module_name] = module_instance
                
                module_info.status = ModuleStatus.READY
                self.logger.info(f"Module {module_name} initialized successfully")
                return True
                
            except Exception as e:
                module_info.status = ModuleStatus.ERROR
                module_info.last_error = str(e)
                module_info.error_count += 1
                
                context = ErrorContext(
                    module_name=module_name,
                    function_name="initialize_module"
                )
                
                self.error_handler.handle_error(e, context)
                self.logger.error(f"Failed to initialize module {module_name}: {e}")
                return False
    
    def _ensure_dependencies(self, module_name: str):
        """Ensure all dependencies of a module are initialized"""
        
        module_info = self.modules[module_name]
        
        for dependency in module_info.dependencies:
            if dependency not in self.modules:
                raise ModuleNotAvailableError(dependency)
            
            dep_status = self.modules[dependency].status
            if dep_status != ModuleStatus.READY:
                self.logger.info(f"Initializing dependency {dependency} for {module_name}")
                if not self.initialize_module(dependency):
                    raise ModuleIntegrationError(
                        f"Failed to initialize dependency {dependency} for {module_name}"
                    )
    
    def _dynamic_module_initialization(self, module_name: str, module_info: ModuleInfo) -> Optional[Any]:
        """Dynamically initialize a module based on its type"""
        
        try:
            module_type = module_info.module_type
            
            if module_type == "features":
                from InstrumentTimbre.core.features import unified_features
                return unified_features.UnifiedFeatureExtractor()
            
            elif module_type == "models":
                from InstrumentTimbre.core.models import unified_model
                return unified_model.UnifiedModel()
            
            elif module_type == "professional_audio":
                from InstrumentTimbre.core.professional_audio import ProfessionalAudioEngine
                from config import get_config
                return ProfessionalAudioEngine(simple_config=get_config())
            
            elif module_type == "generation":
                from InstrumentTimbre.core.generation import music_generation_pipeline
                return music_generation_pipeline.MusicGenerationPipeline()
            
            elif module_type == "operations":
                from InstrumentTimbre.core.operations import operation_dispatcher
                return operation_dispatcher.OperationDispatcher()
            
            elif module_type == "quality_assurance":
                from InstrumentTimbre.core.quality_assurance import quality_assurance_engine
                return quality_assurance_engine.QualityAssuranceEngine()
            
            elif module_type == "analysis":
                from InstrumentTimbre.core.analysis import music_understanding_engine
                return music_understanding_engine.MusicUnderstandingEngine()
            
            else:
                self.logger.warning(f"Unknown module type: {module_type}")
                return None
                
        except ImportError as e:
            self.logger.error(f"Failed to import module {module_name}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to initialize module {module_name}: {e}")
            return None
    
    def execute_pipeline(
        self, 
        pipeline_name: str, 
        input_data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Execute a complete processing pipeline"""
        
        if pipeline_name not in self.pipeline_registry:
            raise ModuleIntegrationError(f"Unknown pipeline: {pipeline_name}")
        
        pipeline = self.pipeline_registry[pipeline_name]
        start_time = time.time()
        
        self.logger.info(f"Executing pipeline: {pipeline_name}")
        
        try:
            # Initialize all required modules
            for module_name in pipeline.modules:
                if not self.initialize_module(module_name):
                    raise ModuleIntegrationError(f"Failed to initialize module: {module_name}")
            
            # Execute pipeline
            result = self._execute_pipeline_modules(pipeline, input_data, **kwargs)
            
            # Update statistics
            execution_time = time.time() - start_time
            self._update_execution_stats(True, execution_time)
            
            self.logger.info(f"Pipeline {pipeline_name} completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_execution_stats(False, execution_time)
            
            context = ErrorContext(
                module_name="pipeline_coordinator",
                function_name="execute_pipeline",
                operation_id=pipeline_name
            )
            
            self.error_handler.handle_error(e, context)
            
            # Retry if configured
            if pipeline.retry_on_failure and pipeline.max_retries > 0:
                self.logger.info(f"Retrying pipeline {pipeline_name}")
                # Implement retry logic here
            
            raise
    
    def _execute_pipeline_modules(
        self, 
        pipeline: ProcessingPipeline, 
        input_data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Execute modules in the pipeline according to dependencies and parallel groups"""
        
        executed_modules = set()
        results = {"input": input_data}
        
        # Execute modules in dependency order
        execution_order = self._calculate_execution_order(pipeline.modules)
        
        for module_group in execution_order:
            if len(module_group) == 1:
                # Single module execution
                module_name = module_group[0]
                result = self._execute_single_module(module_name, results, **kwargs)
                results[module_name] = result
                executed_modules.add(module_name)
            else:
                # Parallel execution
                group_results = self._execute_parallel_modules(module_group, results, **kwargs)
                results.update(group_results)
                executed_modules.update(module_group)
        
        return results
    
    def _calculate_execution_order(self, modules: List[str]) -> List[List[str]]:
        """Calculate optimal execution order considering dependencies"""
        
        # Simplified topological sort - returns groups that can be executed in parallel
        remaining_modules = set(modules)
        execution_order = []
        
        while remaining_modules:
            # Find modules with no remaining dependencies
            ready_modules = []
            for module_name in remaining_modules:
                module_deps = set(self.modules[module_name].dependencies)
                if not module_deps.intersection(remaining_modules):
                    ready_modules.append(module_name)
            
            if not ready_modules:
                # Circular dependency or missing dependency
                raise ModuleIntegrationError(
                    f"Circular dependency detected in modules: {remaining_modules}"
                )
            
            execution_order.append(ready_modules)
            remaining_modules -= set(ready_modules)
        
        return execution_order
    
    def _execute_single_module(
        self, 
        module_name: str, 
        context_data: Dict[str, Any],
        **kwargs
    ) -> Any:
        """Execute a single module"""
        
        module_info = self.modules[module_name]
        module_instance = self.module_instances.get(module_name)
        
        if not module_instance:
            raise ModuleIntegrationError(f"Module {module_name} not initialized")
        
        self.logger.debug(f"Executing module: {module_name}")
        module_info.status = ModuleStatus.PROCESSING
        module_info.processing_count += 1
        
        try:
            # Call the module's main processing method
            if hasattr(module_instance, 'process'):
                result = module_instance.process(context_data, **kwargs)
            elif hasattr(module_instance, 'execute'):
                result = module_instance.execute(context_data, **kwargs)
            elif hasattr(module_instance, '__call__'):
                result = module_instance(context_data, **kwargs)
            else:
                raise ModuleIntegrationError(
                    f"Module {module_name} has no process/execute/__call__ method"
                )
            
            module_info.status = ModuleStatus.READY
            return result
            
        except Exception as e:
            module_info.status = ModuleStatus.ERROR
            module_info.last_error = str(e)
            module_info.error_count += 1
            raise
    
    def _execute_parallel_modules(
        self, 
        module_group: List[str], 
        context_data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Execute a group of modules in parallel"""
        
        self.logger.debug(f"Executing modules in parallel: {module_group}")
        
        # Submit all modules to thread pool
        futures = {}
        for module_name in module_group:
            future = self.executor.submit(
                self._execute_single_module, 
                module_name, 
                context_data, 
                **kwargs
            )
            futures[module_name] = future
        
        # Collect results
        results = {}
        for module_name, future in futures.items():
            try:
                results[module_name] = future.result()
            except Exception as e:
                self.logger.error(f"Parallel execution failed for {module_name}: {e}")
                # Continue with other modules, but mark this as failed
                results[module_name] = None
                raise
        
        return results
    
    def _update_execution_stats(self, success: bool, execution_time: float):
        """Update execution statistics"""
        
        self.execution_stats["total_executions"] += 1
        
        if success:
            self.execution_stats["successful_executions"] += 1
        else:
            self.execution_stats["failed_executions"] += 1
        
        # Update average execution time
        total = self.execution_stats["total_executions"]
        current_avg = self.execution_stats["average_execution_time"]
        self.execution_stats["average_execution_time"] = (
            (current_avg * (total - 1) + execution_time) / total
        )
    
    def health_check(self, module_name: Optional[str] = None) -> Dict[str, Any]:
        """Perform health check on modules"""
        
        if module_name:
            return self._health_check_single_module(module_name)
        else:
            return self._health_check_all_modules()
    
    def _health_check_single_module(self, module_name: str) -> Dict[str, Any]:
        """Health check for a single module"""
        
        if module_name not in self.modules:
            return {"status": "not_found", "healthy": False}
        
        module_info = self.modules[module_name]
        
        try:
            # Use custom health check if available
            if module_info.health_check_function:
                healthy = module_info.health_check_function()
            else:
                # Basic health check - module should be ready and have low error rate
                healthy = (
                    module_info.status == ModuleStatus.READY and
                    module_info.error_count < 10  # Arbitrary threshold
                )
            
            module_info.last_health_check = time.time()
            
            return {
                "status": module_info.status.value,
                "healthy": healthy,
                "error_count": module_info.error_count,
                "processing_count": module_info.processing_count,
                "last_error": module_info.last_error
            }
            
        except Exception as e:
            return {
                "status": "health_check_failed",
                "healthy": False,
                "error": str(e)
            }
    
    def _health_check_all_modules(self) -> Dict[str, Any]:
        """Health check for all modules"""
        
        results = {}
        overall_healthy = True
        
        for module_name in self.modules:
            module_health = self._health_check_single_module(module_name)
            results[module_name] = module_health
            
            if not module_health["healthy"]:
                overall_healthy = False
        
        return {
            "overall_healthy": overall_healthy,
            "modules": results,
            "execution_stats": self.execution_stats
        }
    
    def get_module_status(self) -> Dict[str, Any]:
        """Get status of all registered modules"""
        
        status = {}
        for name, module_info in self.modules.items():
            status[name] = {
                "status": module_info.status.value,
                "type": module_info.module_type,
                "version": module_info.version,
                "dependencies": module_info.dependencies,
                "error_count": module_info.error_count,
                "processing_count": module_info.processing_count
            }
        
        return status
    
    def shutdown(self):
        """Shutdown coordinator and cleanup resources"""
        
        self.logger.info("Shutting down module coordinator")
        
        # Cleanup modules
        for module_name, module_info in self.modules.items():
            try:
                if module_info.cleanup_function:
                    module_info.cleanup_function()
                
                module_info.status = ModuleStatus.DISABLED
                
            except Exception as e:
                self.logger.error(f"Error cleaning up module {module_name}: {e}")
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.logger.info("Module coordinator shutdown complete")