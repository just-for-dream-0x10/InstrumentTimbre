"""
End-to-End Workflow Integration Engine

Comprehensive workflow system that integrates all InstrumentTimbre modules
with robust exception handling and monitoring capabilities.
"""

import logging
import asyncio
import time
from typing import Dict, Any, List, Optional, Union, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import traceback
from concurrent.futures import ThreadPoolExecutor, Future
import json

from .system_integration.integration_engine import IntegrationEngine, OperationType, OperationRequest
from .system_integration.error_handler import SystemErrorHandler, ErrorContext, handle_errors
from .system_integration.exception_types import *
from .system_integration.module_coordinator import ModuleCoordinator, ProcessingPipeline
from .system_integration.system_monitor import SystemMonitor


class WorkflowType(Enum):
    """Comprehensive workflow types"""
    COMPLETE_MUSIC_ANALYSIS = "complete_music_analysis"
    INTELLIGENT_TRACK_GENERATION = "intelligent_track_generation" 
    PROFESSIONAL_AUDIO_ENHANCEMENT = "professional_audio_enhancement"
    STYLE_TRANSFER_PIPELINE = "style_transfer_pipeline"
    QUALITY_ASSURANCE_WORKFLOW = "quality_assurance_workflow"
    BATCH_PROCESSING_PIPELINE = "batch_processing_pipeline"
    REAL_TIME_PROCESSING = "real_time_processing"


class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RECOVERING = "recovering"


@dataclass
class WorkflowStep:
    """Individual workflow step definition with enhanced capabilities"""
    step_id: str
    module_name: str
    function_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    timeout: Optional[float] = None
    retry_count: int = 3
    critical: bool = True
    parallel_execution: bool = False
    quality_gate: Optional[Callable] = None
    fallback_function: Optional[str] = None
    
    
@dataclass
class WorkflowRequest:
    """Enhanced workflow execution request"""
    workflow_type: WorkflowType
    input_data: Dict[str, Any]
    parameters: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    priority: str = "normal"  # low, normal, high, critical
    timeout: Optional[float] = None
    callback: Optional[Callable] = None
    quality_requirements: Dict[str, float] = field(default_factory=dict)
    execution_context: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class WorkflowResult:
    """Comprehensive workflow execution result"""
    workflow_id: str
    workflow_type: WorkflowType
    status: WorkflowStatus
    result_data: Optional[Dict[str, Any]] = None
    error_info: Optional[Dict[str, Any]] = None
    step_results: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    recovery_actions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowIntegrationEngine:
    """
    Comprehensive end-to-end workflow integration engine that orchestrates
    all InstrumentTimbre functionality with robust error handling.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the workflow integration engine
        
        Args:
            config: System configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Core integration components
        self.integration_engine = IntegrationEngine(enable_monitoring=True)
        self.error_handler = SystemErrorHandler()
        self.module_coordinator = ModuleCoordinator(max_workers=8)
        
        # Workflow tracking and management
        self.workflow_counter = 0
        self.active_workflows: Dict[str, WorkflowRequest] = {}
        self.completed_workflows: List[WorkflowResult] = []
        self.workflow_history: List[Dict[str, Any]] = []
        
        # Performance and quality metrics
        self.metrics = {
            "total_workflows": 0,
            "successful_workflows": 0,
            "failed_workflows": 0,
            "recovered_workflows": 0,
            "average_execution_time": 0.0,
            "quality_scores": [],
            "workflows_by_type": {},
            "error_recovery_rate": 0.0
        }
        
        # Initialize predefined workflows
        self.workflow_definitions = self._initialize_comprehensive_workflows()
        
        # Setup quality gates and monitoring
        self._setup_quality_gates()
        self._setup_monitoring()
        
        self.logger.info("Workflow Integration Engine initialized successfully")
    
    def _initialize_comprehensive_workflows(self) -> Dict[WorkflowType, List[WorkflowStep]]:
        """Initialize comprehensive workflow definitions"""
        
        workflows = {}
        
        # Complete Music Analysis Workflow
        workflows[WorkflowType.COMPLETE_MUSIC_ANALYSIS] = [
            WorkflowStep(
                step_id="audio_validation",
                module_name="audio_processor", 
                function_name="validate_audio_input",
                critical=True,
                timeout=30.0
            ),
            WorkflowStep(
                step_id="feature_extraction",
                module_name="feature_extractor",
                function_name="extract_comprehensive_features", 
                dependencies=["audio_validation"],
                timeout=120.0
            ),
            WorkflowStep(
                step_id="emotion_analysis",
                module_name="emotion_engine",
                function_name="analyze_emotion",
                dependencies=["feature_extraction"],
                parallel_execution=True
            ),
            WorkflowStep(
                step_id="structure_analysis",
                module_name="music_analyzer", 
                function_name="analyze_structure",
                dependencies=["feature_extraction"],
                parallel_execution=True
            ),
            WorkflowStep(
                step_id="quality_assessment",
                module_name="quality_assessor",
                function_name="assess_audio_quality",
                dependencies=["emotion_analysis", "structure_analysis"],
                quality_gate=lambda result: result.get("quality_score", 0) > 0.6
            )
        ]
        
        # Intelligent Track Generation Workflow
        workflows[WorkflowType.INTELLIGENT_TRACK_GENERATION] = [
            WorkflowStep(
                step_id="input_analysis", 
                module_name="music_analyzer",
                function_name="analyze_for_generation",
                critical=True
            ),
            WorkflowStep(
                step_id="generation_planning",
                module_name="generation_planner",
                function_name="plan_track_generation",
                dependencies=["input_analysis"]
            ),
            WorkflowStep(
                step_id="track_generation",
                module_name="music_generator", 
                function_name="generate_track",
                dependencies=["generation_planning"],
                timeout=300.0,
                fallback_function="generate_simple_track"
            ),
            WorkflowStep(
                step_id="quality_validation",
                module_name="quality_validator",
                function_name="validate_generated_track",
                dependencies=["track_generation"],
                quality_gate=lambda result: result.get("harmonic_consistency", 0) > 0.7
            ),
            WorkflowStep(
                step_id="integration",
                module_name="track_integrator",
                function_name="integrate_with_original",
                dependencies=["quality_validation"]
            )
        ]
        
        # Professional Audio Enhancement Workflow  
        workflows[WorkflowType.PROFESSIONAL_AUDIO_ENHANCEMENT] = [
            WorkflowStep(
                step_id="audio_analysis",
                module_name="professional_audio_analyzer",
                function_name="analyze_for_enhancement", 
                critical=True
            ),
            WorkflowStep(
                step_id="dynamic_range_optimization",
                module_name="dynamic_range_optimizer",
                function_name="optimize_dynamic_range",
                dependencies=["audio_analysis"],
                parallel_execution=True
            ),
            WorkflowStep(
                step_id="eq_balancing",
                module_name="intelligent_eq_balancer", 
                function_name="balance_frequencies",
                dependencies=["audio_analysis"],
                parallel_execution=True
            ),
            WorkflowStep(
                step_id="spatial_enhancement",
                module_name="spatial_processor",
                function_name="enhance_spatial_positioning", 
                dependencies=["dynamic_range_optimization", "eq_balancing"]
            ),
            WorkflowStep(
                step_id="final_mastering",
                module_name="mastering_engine",
                function_name="apply_mastering",
                dependencies=["spatial_enhancement"],
                quality_gate=lambda result: result.get("mastering_quality", 0) > 0.8
            )
        ]
        
        return workflows
    
    def _setup_quality_gates(self):
        """Setup quality validation gates for workflow steps"""
        
        self.quality_validators = {
            "audio_quality": lambda data: data.get("snr_db", 0) > 20,
            "harmonic_consistency": lambda data: data.get("harmonic_score", 0) > 0.7,
            "emotion_preservation": lambda data: data.get("emotion_similarity", 0) > 0.8,
            "generation_quality": lambda data: data.get("generation_score", 0) > 0.6,
            "mastering_quality": lambda data: data.get("loudness_lufs", -50) > -23
        }
    
    def _setup_monitoring(self):
        """Setup comprehensive monitoring and alerting"""
        
        # Performance thresholds
        self.performance_thresholds = {
            "max_execution_time": 600.0,  # 10 minutes
            "min_success_rate": 0.85,
            "max_error_rate": 0.15,
            "min_quality_score": 0.7
        }
        
        # Setup alerts for critical issues
        if hasattr(self.integration_engine, 'monitor') and self.integration_engine.monitor:
            self.integration_engine.monitor.add_alert_callback(self._handle_system_alert)
    
    # @handle_errors - decorator temporarily disabled for compatibility
    async def execute_workflow(self, request: WorkflowRequest) -> WorkflowResult:
        """
        Execute a complete workflow with comprehensive error handling
        
        Args:
            request: Workflow execution request
            
        Returns:
            WorkflowResult: Complete execution results with metrics
        """
        
        workflow_id = f"workflow_{self.workflow_counter}"
        self.workflow_counter += 1
        
        start_time = time.time()
        
        # Initialize result object
        result = WorkflowResult(
            workflow_id=workflow_id,
            workflow_type=request.workflow_type,
            status=WorkflowStatus.INITIALIZING
        )
        
        try:
            self.logger.info(f"Starting workflow {workflow_id}: {request.workflow_type.value}")
            
            # Add to active workflows
            self.active_workflows[workflow_id] = request
            
            # Validate input and setup execution context
            await self._validate_workflow_input(request)
            await self._setup_execution_context(request)
            
            # Get workflow definition
            workflow_steps = self.workflow_definitions.get(request.workflow_type)
            if not workflow_steps:
                raise WorkflowConfigurationError(
                    f"No workflow definition found for {request.workflow_type.value}"
                )
            
            result.status = WorkflowStatus.RUNNING
            
            # Execute workflow steps
            step_results = await self._execute_workflow_steps(
                workflow_steps, request, workflow_id
            )
            
            result.step_results = step_results
            
            # Validate final results
            final_validation = await self._validate_workflow_results(step_results, request)
            
            if final_validation["valid"]:
                result.status = WorkflowStatus.COMPLETED
                result.result_data = final_validation["data"]
                result.quality_metrics = final_validation["quality_metrics"]
                
                self.metrics["successful_workflows"] += 1
                self.logger.info(f"Workflow {workflow_id} completed successfully")
            else:
                result.status = WorkflowStatus.FAILED
                result.error_info = final_validation["errors"]
                
                self.metrics["failed_workflows"] += 1
                self.logger.error(f"Workflow {workflow_id} failed validation")
                
        except Exception as e:
            result.status = WorkflowStatus.FAILED
            result.error_info = {
                "error_type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc()
            }
            
            # Attempt recovery
            recovery_result = await self._attempt_workflow_recovery(
                request, workflow_id, e
            )
            if recovery_result["recovered"]:
                result.status = WorkflowStatus.COMPLETED
                result.result_data = recovery_result["data"]
                result.recovery_actions = recovery_result["actions"]
                self.metrics["recovered_workflows"] += 1
            else:
                self.metrics["failed_workflows"] += 1
            
            self.logger.error(f"Workflow {workflow_id} failed: {e}")
        
        finally:
            # Cleanup and metrics
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            # Update metrics
            self.metrics["total_workflows"] += 1
            self._update_performance_metrics(execution_time, result.status)
            
            # Remove from active workflows
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
            
            # Add to history
            self.completed_workflows.append(result)
            self._add_to_workflow_history(workflow_id, request, result)
        
        return result
    
    async def _validate_workflow_input(self, request: WorkflowRequest):
        """Validate workflow input data and parameters"""
        
        if not request.input_data:
            raise ValidationException("Workflow input data is required")
        
        # Validate audio data if present
        if "audio_data" in request.input_data:
            audio_data = request.input_data["audio_data"]
            if audio_data is None or len(audio_data) == 0:
                raise AudioProcessingError("Invalid audio data provided")
        
        # Validate required parameters based on workflow type
        required_params = self._get_required_parameters(request.workflow_type)
        for param in required_params:
            if param not in request.parameters:
                raise ValidationException(f"Required parameter missing: {param}")
    
    def _get_required_parameters(self, workflow_type: WorkflowType) -> List[str]:
        """Get required parameters for each workflow type"""
        
        param_mapping = {
            WorkflowType.COMPLETE_MUSIC_ANALYSIS: ["analysis_depth"],
            WorkflowType.INTELLIGENT_TRACK_GENERATION: ["target_role", "instrument"],
            WorkflowType.PROFESSIONAL_AUDIO_ENHANCEMENT: ["enhancement_level"],
            WorkflowType.STYLE_TRANSFER_PIPELINE: ["target_style", "source_style"],
            WorkflowType.QUALITY_ASSURANCE_WORKFLOW: ["quality_threshold"],
            WorkflowType.BATCH_PROCESSING_PIPELINE: ["batch_size", "processing_mode"],
            WorkflowType.REAL_TIME_PROCESSING: ["buffer_size", "latency_target"]
        }
        
        return param_mapping.get(workflow_type, [])
    
    async def _setup_execution_context(self, request: WorkflowRequest):
        """Setup execution context for workflow"""
        
        context = {
            "start_time": time.time(),
            "user_id": request.user_id,
            "session_id": request.session_id,
            "priority": request.priority,
            "system_resources": self._get_system_resources(),
            "execution_environment": "production"
        }
        
        request.execution_context.update(context)
    
    def _get_system_resources(self) -> Dict[str, Any]:
        """Get current system resource status"""
        
        if hasattr(self.integration_engine, 'monitor') and self.integration_engine.monitor:
            return self.integration_engine.monitor.get_system_health()
        
        return {"cpu_percent": 0, "memory_percent": 0, "disk_usage": 0}
    
    async def _execute_workflow_steps(
        self, 
        steps: List[WorkflowStep], 
        request: WorkflowRequest,
        workflow_id: str
    ) -> Dict[str, Any]:
        """Execute workflow steps with dependency management and parallel execution"""
        
        step_results = {}
        completed_steps = set()
        failed_steps = set()
        
        # Create dependency graph
        dependency_graph = self._build_dependency_graph(steps)
        
        # Execute steps respecting dependencies
        while len(completed_steps) < len(steps):
            # Find ready steps (all dependencies completed)
            ready_steps = []
            for step in steps:
                if (step.step_id not in completed_steps and 
                    step.step_id not in failed_steps and
                    all(dep in completed_steps for dep in step.dependencies)):
                    ready_steps.append(step)
            
            if not ready_steps:
                # Check for circular dependencies or failures blocking progress
                remaining_steps = [s for s in steps if s.step_id not in completed_steps]
                if remaining_steps:
                    raise WorkflowExecutionError(
                        f"Workflow blocked: remaining steps {[s.step_id for s in remaining_steps]}"
                    )
                break
            
            # Group parallel steps
            parallel_groups = self._group_parallel_steps(ready_steps)
            
            # Execute each group
            for group in parallel_groups:
                if len(group) == 1:
                    # Single step execution
                    step = group[0]
                    try:
                        result = await self._execute_single_step(step, request, step_results)
                        step_results[step.step_id] = result
                        completed_steps.add(step.step_id)
                        
                        self.logger.info(f"Step {step.step_id} completed successfully")
                        
                    except Exception as e:
                        if step.critical:
                            # Critical step failure - attempt recovery
                            recovery_result = await self._attempt_step_recovery(
                                step, request, e, workflow_id
                            )
                            if recovery_result["recovered"]:
                                step_results[step.step_id] = recovery_result["result"]
                                completed_steps.add(step.step_id)
                            else:
                                failed_steps.add(step.step_id)
                                raise WorkflowStepException(
                                    step.step_id, f"Critical step failed: {e}", e
                                )
                        else:
                            # Non-critical step failure - log and continue
                            self.logger.warning(f"Non-critical step {step.step_id} failed: {e}")
                            step_results[step.step_id] = {"error": str(e), "status": "failed"}
                            failed_steps.add(step.step_id)
                else:
                    # Parallel execution
                    parallel_results = await self._execute_parallel_steps(
                        group, request, step_results, workflow_id
                    )
                    
                    for step_id, result in parallel_results.items():
                        if result.get("status") != "failed":
                            step_results[step_id] = result
                            completed_steps.add(step_id)
                        else:
                            failed_steps.add(step_id)
        
        return step_results
    
    def _build_dependency_graph(self, steps: List[WorkflowStep]) -> Dict[str, List[str]]:
        """Build dependency graph for workflow steps"""
        
        graph = {}
        for step in steps:
            graph[step.step_id] = step.dependencies.copy()
        
        return graph
    
    def _group_parallel_steps(self, steps: List[WorkflowStep]) -> List[List[WorkflowStep]]:
        """Group steps that can be executed in parallel"""
        
        parallel_steps = [step for step in steps if step.parallel_execution]
        sequential_steps = [step for step in steps if not step.parallel_execution]
        
        groups = []
        
        # Add parallel group if exists
        if parallel_steps:
            groups.append(parallel_steps)
        
        # Add sequential steps as individual groups
        for step in sequential_steps:
            groups.append([step])
        
        return groups
    
    async def _execute_single_step(
        self, 
        step: WorkflowStep, 
        request: WorkflowRequest,
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single workflow step"""
        
        self.logger.info(f"Executing step: {step.step_id}")
        
        try:
            # Prepare step parameters
            step_params = step.parameters.copy()
            step_params.update(request.parameters)
            
            # Add context from previous steps
            if step.dependencies:
                dependency_data = {}
                for dep_id in step.dependencies:
                    if dep_id in previous_results:
                        dependency_data[dep_id] = previous_results[dep_id]
                step_params["dependency_results"] = dependency_data
            
            # Create operation request
            operation_request = OperationRequest(
                operation_type=self._map_step_to_operation(step),
                input_data=request.input_data.copy(),
                parameters=step_params,
                user_id=request.user_id,
                session_id=request.session_id,
                timeout=step.timeout
            )
            
            # Execute through integration engine
            operation_result = await self.integration_engine.execute_operation(operation_request)
            
            if not operation_result.success:
                raise WorkflowStepException(
                    step.step_id, 
                    f"Step execution failed: {operation_result.error_info}"
                )
            
            result = operation_result.result_data or {}
            
            # Apply quality gate if defined
            if step.quality_gate and not step.quality_gate(result):
                raise QualityGateException(
                    f"Quality gate failed for step {step.step_id}"
                )
            
            result["execution_time"] = operation_result.processing_time
            result["status"] = "completed"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Step {step.step_id} execution failed: {e}")
            raise
    
    def _map_step_to_operation(self, step: WorkflowStep) -> OperationType:
        """Map workflow step to integration engine operation type"""
        
        mapping = {
            "audio_processor": OperationType.AUDIO_ENHANCEMENT,
            "feature_extractor": OperationType.MUSIC_ANALYSIS,
            "emotion_engine": OperationType.MUSIC_ANALYSIS,
            "music_analyzer": OperationType.MUSIC_ANALYSIS,
            "music_generator": OperationType.MUSIC_GENERATION,
            "quality_assessor": OperationType.QUALITY_ASSURANCE,
            "professional_audio_analyzer": OperationType.AUDIO_ENHANCEMENT,
            "dynamic_range_optimizer": OperationType.AUDIO_ENHANCEMENT,
            "intelligent_eq_balancer": OperationType.AUDIO_ENHANCEMENT,
            "spatial_processor": OperationType.AUDIO_ENHANCEMENT,
            "mastering_engine": OperationType.AUDIO_ENHANCEMENT
        }
        
        return mapping.get(step.module_name, OperationType.MUSIC_ANALYSIS)
    
    async def _execute_parallel_steps(
        self,
        steps: List[WorkflowStep],
        request: WorkflowRequest, 
        previous_results: Dict[str, Any],
        workflow_id: str
    ) -> Dict[str, Any]:
        """Execute multiple steps in parallel"""
        
        self.logger.info(f"Executing {len(steps)} steps in parallel")
        
        # Create tasks for parallel execution
        tasks = []
        for step in steps:
            task = asyncio.create_task(
                self._execute_single_step(step, request, previous_results)
            )
            tasks.append((step.step_id, task))
        
        # Wait for all tasks to complete
        results = {}
        for step_id, task in tasks:
            try:
                result = await task
                results[step_id] = result
            except Exception as e:
                self.logger.error(f"Parallel step {step_id} failed: {e}")
                results[step_id] = {"error": str(e), "status": "failed"}
        
        return results
    
    async def _validate_workflow_results(
        self, 
        step_results: Dict[str, Any], 
        request: WorkflowRequest
    ) -> Dict[str, Any]:
        """Validate final workflow results against quality requirements"""
        
        validation_result = {
            "valid": True,
            "data": {},
            "quality_metrics": {},
            "errors": []
        }
        
        try:
            # Aggregate results from all steps
            aggregated_data = {}
            quality_scores = []
            
            for step_id, result in step_results.items():
                if result.get("status") == "failed":
                    validation_result["errors"].append(f"Step {step_id} failed")
                    continue
                
                # Collect quality metrics
                if "quality_score" in result:
                    quality_scores.append(result["quality_score"])
                
                # Aggregate step data
                aggregated_data[step_id] = result
            
            # Calculate overall quality score
            if quality_scores:
                overall_quality = sum(quality_scores) / len(quality_scores)
                validation_result["quality_metrics"]["overall_quality"] = overall_quality
                
                # Check against quality requirements
                min_quality = request.quality_requirements.get("min_quality", 0.6)
                if overall_quality < min_quality:
                    validation_result["valid"] = False
                    validation_result["errors"].append(
                        f"Quality below threshold: {overall_quality:.3f} < {min_quality}"
                    )
            
            validation_result["data"] = aggregated_data
            
        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    async def _attempt_workflow_recovery(
        self, 
        request: WorkflowRequest, 
        workflow_id: str, 
        error: Exception
    ) -> Dict[str, Any]:
        """Attempt to recover from workflow failure"""
        
        recovery_result = {
            "recovered": False,
            "data": None,
            "actions": []
        }
        
        try:
            self.logger.info(f"Attempting recovery for workflow {workflow_id}")
            
            # Determine recovery strategy based on error type
            if isinstance(error, AudioProcessingError):
                # Try with degraded audio quality
                recovery_result = await self._recover_with_degraded_quality(request)
            elif isinstance(error, ModelError):
                # Try with fallback model
                recovery_result = await self._recover_with_fallback_model(request)
            elif isinstance(error, SystemResourceError):
                # Try with reduced resource requirements
                recovery_result = await self._recover_with_reduced_resources(request)
            else:
                # Generic recovery - try simplified workflow
                recovery_result = await self._recover_with_simplified_workflow(request)
            
            if recovery_result["recovered"]:
                self.logger.info(f"Recovery successful for workflow {workflow_id}")
            else:
                self.logger.warning(f"Recovery failed for workflow {workflow_id}")
                
        except Exception as recovery_error:
            self.logger.error(f"Recovery attempt failed: {recovery_error}")
            recovery_result["actions"].append(f"Recovery failed: {str(recovery_error)}")
        
        return recovery_result
    
    async def _recover_with_degraded_quality(self, request: WorkflowRequest) -> Dict[str, Any]:
        """Recover by reducing quality requirements"""
        
        # Create modified request with lower quality thresholds
        recovery_request = WorkflowRequest(
            workflow_type=request.workflow_type,
            input_data=request.input_data.copy(),
            parameters=request.parameters.copy(),
            quality_requirements={"min_quality": 0.4}  # Lower threshold
        )
        
        # Add degraded quality parameters
        recovery_request.parameters["quality_mode"] = "degraded"
        recovery_request.parameters["enable_fallbacks"] = True
        
        try:
            result = await self.execute_workflow(recovery_request)
            if result.status == WorkflowStatus.COMPLETED:
                return {
                    "recovered": True,
                    "data": result.result_data,
                    "actions": ["Applied degraded quality mode"]
                }
        except Exception:
            pass
        
        return {"recovered": False, "data": None, "actions": ["Degraded quality recovery failed"]}
    
    async def _attempt_step_recovery(
        self, 
        step: WorkflowStep, 
        request: WorkflowRequest, 
        error: Exception,
        workflow_id: str
    ) -> Dict[str, Any]:
        """Attempt to recover from individual step failure"""
        
        recovery_result = {"recovered": False, "result": None}
        
        # Try fallback function if available
        if step.fallback_function:
            try:
                self.logger.info(f"Trying fallback function for step {step.step_id}")
                
                # Create modified step with fallback function
                fallback_step = WorkflowStep(
                    step_id=f"{step.step_id}_fallback",
                    module_name=step.module_name,
                    function_name=step.fallback_function,
                    parameters=step.parameters.copy(),
                    timeout=step.timeout
                )
                
                result = await self._execute_single_step(fallback_step, request, {})
                recovery_result["recovered"] = True
                recovery_result["result"] = result
                
                self.logger.info(f"Fallback successful for step {step.step_id}")
                
            except Exception as fallback_error:
                self.logger.warning(f"Fallback failed for step {step.step_id}: {fallback_error}")
        
        # Try retry if not recovered and retries available
        if not recovery_result["recovered"] and step.retry_count > 0:
            for attempt in range(step.retry_count):
                try:
                    self.logger.info(f"Retry attempt {attempt + 1} for step {step.step_id}")
                    
                    # Add small delay between retries
                    await asyncio.sleep(1.0 * (attempt + 1))
                    
                    result = await self._execute_single_step(step, request, {})
                    recovery_result["recovered"] = True
                    recovery_result["result"] = result
                    
                    self.logger.info(f"Retry successful for step {step.step_id}")
                    break
                    
                except Exception as retry_error:
                    self.logger.warning(f"Retry {attempt + 1} failed for step {step.step_id}: {retry_error}")
        
        return recovery_result
    
    def _update_performance_metrics(self, execution_time: float, status: WorkflowStatus):
        """Update performance metrics"""
        
        # Update execution time average
        total_workflows = self.metrics["total_workflows"]
        if total_workflows > 0:
            current_avg = self.metrics["average_execution_time"]
            self.metrics["average_execution_time"] = (
                (current_avg * (total_workflows - 1) + execution_time) / total_workflows
            )
        else:
            self.metrics["average_execution_time"] = execution_time
        
        # Update success/failure rates
        if status == WorkflowStatus.COMPLETED:
            self.metrics["successful_workflows"] += 1
        
        # Calculate error recovery rate
        total_errors = self.metrics["failed_workflows"]
        recovered_errors = self.metrics["recovered_workflows"]
        
        if total_errors > 0:
            self.metrics["error_recovery_rate"] = recovered_errors / total_errors
    
    def _add_to_workflow_history(
        self, 
        workflow_id: str, 
        request: WorkflowRequest, 
        result: WorkflowResult
    ):
        """Add workflow execution to history"""
        
        history_entry = {
            "workflow_id": workflow_id,
            "workflow_type": request.workflow_type.value,
            "status": result.status.value,
            "execution_time": result.execution_time,
            "timestamp": time.time(),
            "user_id": request.user_id,
            "success": result.status == WorkflowStatus.COMPLETED,
            "quality_score": result.quality_metrics.get("overall_quality", 0.0)
        }
        
        self.workflow_history.append(history_entry)
        
        # Keep only last 1000 entries
        if len(self.workflow_history) > 1000:
            self.workflow_history = self.workflow_history[-1000:]
    
    def _handle_system_alert(self, alert: Dict[str, Any]):
        """Handle system alerts from monitoring"""
        
        self.logger.warning(f"System alert received: {alert}")
        
        # Take action based on alert type
        alert_type = alert.get("type", "unknown")
        
        if alert_type == "high_memory_usage":
            self._handle_memory_pressure()
        elif alert_type == "high_cpu_usage":
            self._handle_cpu_pressure()
        elif alert_type == "disk_space_low":
            self._handle_disk_pressure()
    
    def _handle_memory_pressure(self):
        """Handle high memory usage"""
        
        self.logger.warning("High memory usage detected - reducing concurrent workflows")
        
        # Reduce max workers temporarily
        if hasattr(self.module_coordinator, 'executor'):
            current_workers = self.module_coordinator.executor._max_workers
            if current_workers > 2:
                new_workers = max(2, current_workers // 2)
                self.logger.info(f"Reducing workers from {current_workers} to {new_workers}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        return {
            "active_workflows": len(self.active_workflows),
            "total_workflows_executed": self.metrics["total_workflows"],
            "success_rate": (
                self.metrics["successful_workflows"] / max(1, self.metrics["total_workflows"])
            ),
            "average_execution_time": self.metrics["average_execution_time"],
            "error_recovery_rate": self.metrics["error_recovery_rate"],
            "system_health": self._get_system_resources(),
            "workflow_types_supported": [wt.value for wt in WorkflowType],
            "last_workflow_history": self.workflow_history[-10:] if self.workflow_history else []
        }
    
    async def shutdown(self):
        """Gracefully shutdown the workflow integration engine"""
        
        self.logger.info("Shutting down Workflow Integration Engine...")
        
        # Cancel active workflows
        for workflow_id in list(self.active_workflows.keys()):
            self.logger.info(f"Cancelling active workflow: {workflow_id}")
        
        # Shutdown components
        if hasattr(self.integration_engine, 'shutdown'):
            await self.integration_engine.shutdown()
        
        if hasattr(self.module_coordinator, 'shutdown'):
            self.module_coordinator.shutdown()
        
        self.logger.info("Workflow Integration Engine shutdown complete")