"""
End-to-End Workflow Orchestrator

Comprehensive workflow integration system that coordinates all modules
with robust exception handling and monitoring capabilities.
"""

import logging
import asyncio
import time
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import traceback

from .controller import MusicEditingController
from .emotion_engine import EmotionAnalysisEngine
from .track_operator import TrackOperationEngine
from .music_analyzer import MusicAnalyzer


class WorkflowType(Enum):
    """Supported workflow types"""
    MUSIC_ANALYSIS = "music_analysis"
    TRACK_GENERATION = "track_generation"
    AUDIO_ENHANCEMENT = "audio_enhancement"
    STYLE_TRANSFER = "style_transfer"
    BATCH_PROCESSING = "batch_processing"
    QUALITY_ASSURANCE = "quality_assurance"


class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowStep:
    """Individual workflow step definition"""
    step_id: str
    module_name: str
    function_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    timeout: Optional[float] = None
    retry_count: int = 3
    critical: bool = True


@dataclass
class WorkflowRequest:
    """Workflow execution request"""
    workflow_type: WorkflowType
    input_data: Dict[str, Any]
    parameters: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    priority: str = "normal"
    callback: Optional[Callable] = None


@dataclass
class WorkflowResult:
    """Workflow execution result"""
    workflow_id: str
    workflow_type: WorkflowType
    status: WorkflowStatus
    result_data: Optional[Dict[str, Any]] = None
    error_info: Optional[Dict[str, Any]] = None
    step_results: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowException(Exception):
    """Base workflow exception"""
    pass


class WorkflowStepException(WorkflowException):
    """Step execution exception"""
    def __init__(self, step_id: str, message: str, original_error: Exception = None):
        super().__init__(message)
        self.step_id = step_id
        self.original_error = original_error


class WorkflowTimeoutException(WorkflowException):
    """Workflow timeout exception"""
    pass


class WorkflowOrchestrator:
    """
    End-to-end workflow orchestrator that integrates all system modules
    with comprehensive error handling and monitoring.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize workflow orchestrator
        
        Args:
            config: System configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.music_controller = MusicEditingController()
        self.emotion_engine = EmotionAnalysisEngine()
        self.track_operator = TrackOperationEngine()
        self.music_analyzer = MusicAnalyzer()
        
        # Workflow tracking
        self.workflow_counter = 0
        self.active_workflows: Dict[str, WorkflowRequest] = {}
        self.completed_workflows: List[WorkflowResult] = []
        
        # Performance metrics
        self.metrics = {
            "total_workflows": 0,
            "successful_workflows": 0,
            "failed_workflows": 0,
            "average_execution_time": 0.0,
            "workflows_by_type": {}
        }
        
        # Predefined workflows
        self.workflow_definitions = self._initialize_workflow_definitions()
        
        self.logger.info("Workflow orchestrator initialized successfully")
    
    def _initialize_workflow_definitions(self) -> Dict[WorkflowType, List[WorkflowStep]]:
        """Initialize predefined workflow definitions"""
        
        return {
            WorkflowType.MUSIC_ANALYSIS: [
                WorkflowStep(
                    step_id="audio_preprocessing",
                    module_name="music_analyzer",
                    function_name="preprocess_audio",
                    timeout=30.0
                ),
                WorkflowStep(
                    step_id="emotion_analysis",
                    module_name="emotion_engine",
                    function_name="analyze_emotion",
                    dependencies=["audio_preprocessing"],
                    timeout=15.0
                ),
                WorkflowStep(
                    step_id="feature_extraction",
                    module_name="music_analyzer",
                    function_name="extract_features",
                    dependencies=["audio_preprocessing"],
                    timeout=20.0
                ),
                WorkflowStep(
                    step_id="final_analysis",
                    module_name="music_analyzer",
                    function_name="comprehensive_analysis",
                    dependencies=["emotion_analysis", "feature_extraction"],
                    timeout=10.0
                )
            ],
            
            WorkflowType.TRACK_GENERATION: [
                WorkflowStep(
                    step_id="input_validation",
                    module_name="track_operator",
                    function_name="validate_input",
                    timeout=5.0
                ),
                WorkflowStep(
                    step_id="emotion_context",
                    module_name="emotion_engine",
                    function_name="get_emotion_context",
                    dependencies=["input_validation"],
                    timeout=10.0
                ),
                WorkflowStep(
                    step_id="track_generation",
                    module_name="track_operator",
                    function_name="generate_track",
                    dependencies=["emotion_context"],
                    timeout=60.0
                ),
                WorkflowStep(
                    step_id="quality_check",
                    module_name="music_controller",
                    function_name="validate_quality",
                    dependencies=["track_generation"],
                    timeout=15.0,
                    critical=False
                )
            ],
            
            WorkflowType.AUDIO_ENHANCEMENT: [
                WorkflowStep(
                    step_id="audio_analysis",
                    module_name="music_analyzer",
                    function_name="analyze_audio_quality",
                    timeout=20.0
                ),
                WorkflowStep(
                    step_id="enhancement_planning",
                    module_name="music_controller",
                    function_name="plan_enhancement",
                    dependencies=["audio_analysis"],
                    timeout=10.0
                ),
                WorkflowStep(
                    step_id="audio_processing",
                    module_name="track_operator",
                    function_name="enhance_audio",
                    dependencies=["enhancement_planning"],
                    timeout=90.0
                ),
                WorkflowStep(
                    step_id="quality_validation",
                    module_name="music_controller",
                    function_name="validate_enhancement",
                    dependencies=["audio_processing"],
                    timeout=15.0
                )
            ]
        }
    
    async def execute_workflow(self, request: WorkflowRequest) -> WorkflowResult:
        """
        Execute a complete workflow with error handling and monitoring
        
        Args:
            request: Workflow execution request
            
        Returns:
            Workflow execution result
        """
        start_time = time.time()
        workflow_id = f"wf_{self.workflow_counter:06d}_{int(time.time())}"
        self.workflow_counter += 1
        
        self.logger.info(
            f"Starting workflow {workflow_id}: {request.workflow_type.value}"
        )
        
        # Track active workflow
        self.active_workflows[workflow_id] = request
        
        try:
            # Validate request
            self._validate_workflow_request(request)
            
            # Get workflow definition
            workflow_steps = self.workflow_definitions.get(request.workflow_type)
            if not workflow_steps:
                raise WorkflowException(
                    f"No workflow definition found for {request.workflow_type.value}"
                )
            
            # Execute workflow steps
            step_results = await self._execute_workflow_steps(
                workflow_steps, request, workflow_id
            )
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Create success result
            result = WorkflowResult(
                workflow_id=workflow_id,
                workflow_type=request.workflow_type,
                status=WorkflowStatus.COMPLETED,
                result_data=self._aggregate_step_results(step_results),
                step_results=step_results,
                execution_time=execution_time,
                metadata={
                    "total_steps": len(workflow_steps),
                    "successful_steps": len([r for r in step_results.values() if r.get("success", False)]),
                    "config_used": self.config
                }
            )
            
            # Update metrics
            self._update_workflow_metrics(request.workflow_type, True, execution_time)
            
            self.logger.info(
                f"Workflow {workflow_id} completed successfully in {execution_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Handle workflow error
            error_info = self._handle_workflow_error(e, workflow_id, request)
            
            # Create error result
            result = WorkflowResult(
                workflow_id=workflow_id,
                workflow_type=request.workflow_type,
                status=WorkflowStatus.FAILED,
                error_info=error_info,
                execution_time=execution_time
            )
            
            # Update metrics
            self._update_workflow_metrics(request.workflow_type, False, execution_time)
            
            self.logger.error(
                f"Workflow {workflow_id} failed after {execution_time:.2f}s: {error_info['message']}"
            )
            
            return result
            
        finally:
            # Remove from active workflows
            self.active_workflows.pop(workflow_id, None)
            
            # Add to completed workflows (keep last 100)
            self.completed_workflows.append(result)
            if len(self.completed_workflows) > 100:
                self.completed_workflows = self.completed_workflows[-100:]
    
    async def _execute_workflow_steps(
        self, 
        steps: List[WorkflowStep], 
        request: WorkflowRequest, 
        workflow_id: str
    ) -> Dict[str, Any]:
        """Execute workflow steps with dependency management"""
        
        step_results = {}
        completed_steps = set()
        
        # Create dependency graph
        step_map = {step.step_id: step for step in steps}
        
        while len(completed_steps) < len(steps):
            # Find ready steps (dependencies satisfied)
            ready_steps = []
            for step in steps:
                if (step.step_id not in completed_steps and 
                    all(dep in completed_steps for dep in step.dependencies)):
                    ready_steps.append(step)
            
            if not ready_steps:
                raise WorkflowException("Circular dependency detected in workflow steps")
            
            # Execute ready steps concurrently
            tasks = []
            for step in ready_steps:
                task = self._execute_single_step(step, request, step_results, workflow_id)
                tasks.append((step.step_id, task))
            
            # Wait for step completion
            for step_id, task in tasks:
                try:
                    result = await task
                    step_results[step_id] = result
                    completed_steps.add(step_id)
                    
                    self.logger.debug(f"Step {step_id} completed successfully")
                    
                    # Progress callback
                    if request.callback:
                        await self._notify_progress(
                            request.callback, 
                            len(completed_steps), 
                            len(steps), 
                            step_id
                        )
                        
                except Exception as e:
                    step = step_map[step_id]
                    if step.critical:
                        raise WorkflowStepException(step_id, str(e), e)
                    else:
                        # Non-critical step failure
                        step_results[step_id] = {
                            "success": False,
                            "error": str(e),
                            "critical": False
                        }
                        completed_steps.add(step_id)
                        self.logger.warning(f"Non-critical step {step_id} failed: {e}")
        
        return step_results
    
    async def _execute_single_step(
        self, 
        step: WorkflowStep, 
        request: WorkflowRequest, 
        previous_results: Dict[str, Any],
        workflow_id: str
    ) -> Dict[str, Any]:
        """Execute a single workflow step with retry logic"""
        
        attempt = 0
        last_error = None
        
        while attempt < step.retry_count:
            try:
                self.logger.debug(f"Executing step {step.step_id} (attempt {attempt + 1})")
                
                # Get module instance
                module = self._get_module_instance(step.module_name)
                
                # Prepare step input data
                step_input = self._prepare_step_input(
                    step, request.input_data, request.parameters, previous_results
                )
                
                # Execute step with timeout
                if step.timeout:
                    result = await asyncio.wait_for(
                        self._call_module_function(module, step.function_name, step_input),
                        timeout=step.timeout
                    )
                else:
                    result = await self._call_module_function(
                        module, step.function_name, step_input
                    )
                
                return {
                    "success": True,
                    "result": result,
                    "attempt": attempt + 1,
                    "execution_time": time.time()
                }
                
            except asyncio.TimeoutError:
                last_error = WorkflowTimeoutException(
                    f"Step {step.step_id} timed out after {step.timeout}s"
                )
                self.logger.warning(f"Step {step.step_id} timed out on attempt {attempt + 1}")
                
            except Exception as e:
                last_error = e
                self.logger.warning(
                    f"Step {step.step_id} failed on attempt {attempt + 1}: {e}"
                )
            
            attempt += 1
            
            # Wait before retry (exponential backoff)
            if attempt < step.retry_count:
                wait_time = min(2 ** attempt, 10)  # Max 10 seconds
                await asyncio.sleep(wait_time)
        
        # All attempts failed
        raise WorkflowStepException(
            step.step_id, 
            f"Step failed after {step.retry_count} attempts",
            last_error
        )
    
    def _get_module_instance(self, module_name: str) -> Any:
        """Get module instance by name"""
        
        module_map = {
            "music_controller": self.music_controller,
            "emotion_engine": self.emotion_engine,
            "track_operator": self.track_operator,
            "music_analyzer": self.music_analyzer
        }
        
        module = module_map.get(module_name)
        if not module:
            raise WorkflowException(f"Unknown module: {module_name}")
        
        return module
    
    async def _call_module_function(
        self, 
        module: Any, 
        function_name: str, 
        input_data: Dict[str, Any]
    ) -> Any:
        """Call module function safely"""
        
        if not hasattr(module, function_name):
            raise WorkflowException(
                f"Module {module.__class__.__name__} does not have function {function_name}"
            )
        
        func = getattr(module, function_name)
        
        # Handle both sync and async functions
        if asyncio.iscoroutinefunction(func):
            return await func(**input_data)
        else:
            # Run sync function in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: func(**input_data))
    
    def _prepare_step_input(
        self, 
        step: WorkflowStep, 
        input_data: Dict[str, Any], 
        parameters: Dict[str, Any],
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare input data for step execution"""
        
        step_input = {
            **input_data,
            **parameters,
            **step.parameters
        }
        
        # Add results from dependency steps
        for dep_step_id in step.dependencies:
            if dep_step_id in previous_results:
                dep_result = previous_results[dep_step_id]
                if dep_result.get("success") and "result" in dep_result:
                    step_input[f"{dep_step_id}_result"] = dep_result["result"]
        
        return step_input
    
    def _validate_workflow_request(self, request: WorkflowRequest) -> None:
        """Validate workflow request parameters"""
        
        if not request.input_data:
            raise WorkflowException("input_data is required")
        
        if not isinstance(request.parameters, dict):
            raise WorkflowException("parameters must be a dictionary")
        
        # Type-specific validation
        if request.workflow_type == WorkflowType.MUSIC_ANALYSIS:
            if "audio_data" not in request.input_data:
                raise WorkflowException("audio_data is required for music analysis")
        
        elif request.workflow_type == WorkflowType.TRACK_GENERATION:
            if "target_role" not in request.parameters:
                raise WorkflowException("target_role is required for track generation")
    
    def _aggregate_step_results(self, step_results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results from all workflow steps"""
        
        aggregated = {
            "successful_steps": [],
            "failed_steps": [],
            "results": {}
        }
        
        for step_id, result in step_results.items():
            if result.get("success", False):
                aggregated["successful_steps"].append(step_id)
                if "result" in result:
                    aggregated["results"][step_id] = result["result"]
            else:
                aggregated["failed_steps"].append(step_id)
        
        return aggregated
    
    def _handle_workflow_error(
        self, 
        error: Exception, 
        workflow_id: str, 
        request: WorkflowRequest
    ) -> Dict[str, Any]:
        """Handle workflow errors with detailed information"""
        
        error_info = {
            "error_type": type(error).__name__,
            "message": str(error),
            "workflow_id": workflow_id,
            "workflow_type": request.workflow_type.value,
            "timestamp": time.time(),
            "traceback": traceback.format_exc()
        }
        
        # Add step-specific error info for WorkflowStepException
        if isinstance(error, WorkflowStepException):
            error_info.update({
                "failed_step": error.step_id,
                "original_error": str(error.original_error) if error.original_error else None
            })
        
        # Log error with context
        self.logger.error(
            f"Workflow {workflow_id} error: {error_info['message']}", 
            extra={"workflow_context": error_info}
        )
        
        return error_info
    
    async def _notify_progress(
        self, 
        callback: Callable, 
        completed_steps: int, 
        total_steps: int, 
        current_step: str
    ) -> None:
        """Notify progress callback"""
        
        try:
            progress_info = {
                "progress": completed_steps / total_steps,
                "completed_steps": completed_steps,
                "total_steps": total_steps,
                "current_step": current_step
            }
            
            if asyncio.iscoroutinefunction(callback):
                await callback(progress_info)
            else:
                callback(progress_info)
                
        except Exception as e:
            self.logger.warning(f"Progress callback failed: {e}")
    
    def _update_workflow_metrics(
        self, 
        workflow_type: WorkflowType, 
        success: bool, 
        execution_time: float
    ) -> None:
        """Update workflow performance metrics"""
        
        self.metrics["total_workflows"] += 1
        
        if success:
            self.metrics["successful_workflows"] += 1
        else:
            self.metrics["failed_workflows"] += 1
        
        # Update average execution time
        total = self.metrics["total_workflows"]
        current_avg = self.metrics["average_execution_time"]
        self.metrics["average_execution_time"] = (
            (current_avg * (total - 1) + execution_time) / total
        )
        
        # Update by-type metrics
        type_key = workflow_type.value
        if type_key not in self.metrics["workflows_by_type"]:
            self.metrics["workflows_by_type"][type_key] = {
                "count": 0,
                "success_count": 0,
                "average_time": 0.0
            }
        
        type_metrics = self.metrics["workflows_by_type"][type_key]
        type_metrics["count"] += 1
        
        if success:
            type_metrics["success_count"] += 1
        
        # Update type-specific average time
        type_count = type_metrics["count"]
        current_type_avg = type_metrics["average_time"]
        type_metrics["average_time"] = (
            (current_type_avg * (type_count - 1) + execution_time) / type_count
        )
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific workflow"""
        
        # Check active workflows
        if workflow_id in self.active_workflows:
            return {
                "status": "running",
                "workflow": self.active_workflows[workflow_id]
            }
        
        # Check completed workflows
        for result in self.completed_workflows:
            if result.workflow_id == workflow_id:
                return {
                    "status": result.status.value,
                    "result": result
                }
        
        return None
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        
        return {
            "workflow_metrics": self.metrics,
            "active_workflows": len(self.active_workflows),
            "completed_workflows": len(self.completed_workflows),
            "system_health": {
                "orchestrator_status": "healthy",
                "module_status": {
                    "music_controller": "healthy",
                    "emotion_engine": "healthy",
                    "track_operator": "healthy",
                    "music_analyzer": "healthy"
                }
            }
        }
    
    def register_custom_workflow(
        self, 
        workflow_type: WorkflowType, 
        steps: List[WorkflowStep]
    ) -> None:
        """Register a custom workflow definition"""
        
        # Validate workflow steps
        step_ids = {step.step_id for step in steps}
        for step in steps:
            for dep in step.dependencies:
                if dep not in step_ids:
                    raise WorkflowException(
                        f"Step {step.step_id} depends on non-existent step {dep}"
                    )
        
        self.workflow_definitions[workflow_type] = steps
        self.logger.info(f"Registered custom workflow: {workflow_type.value}")
    
    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel an active workflow"""
        
        if workflow_id in self.active_workflows:
            # Remove from active workflows
            request = self.active_workflows.pop(workflow_id)
            
            # Create cancelled result
            result = WorkflowResult(
                workflow_id=workflow_id,
                workflow_type=request.workflow_type,
                status=WorkflowStatus.CANCELLED,
                execution_time=0.0
            )
            
            self.completed_workflows.append(result)
            self.logger.info(f"Cancelled workflow {workflow_id}")
            return True
        
        return False
    
    def shutdown(self) -> None:
        """Shutdown workflow orchestrator and cleanup resources"""
        
        self.logger.info("Shutting down workflow orchestrator...")
        
        # Cancel all active workflows
        for workflow_id in list(self.active_workflows.keys()):
            self.cancel_workflow(workflow_id)
        
        self.logger.info("Workflow orchestrator shutdown complete")