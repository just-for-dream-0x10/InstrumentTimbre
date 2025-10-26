"""
Comprehensive tests for end-to-end workflow integration and exception handling
"""

import pytest
import asyncio
import numpy as np
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from InstrumentTimbre.core.workflow_integration import (
    WorkflowIntegrationEngine, WorkflowRequest, WorkflowResult, 
    WorkflowType, WorkflowStatus, WorkflowStep
)
from InstrumentTimbre.core.system_integration.exception_types import (
    AudioProcessingError, ModelError, SystemResourceError, ValidationException
)


class TestWorkflowIntegration:
    """Test suite for workflow integration engine"""
    
    @pytest.fixture
    def sample_audio_data(self):
        """Generate sample audio data for testing"""
        sr = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        # Simple sine wave
        audio = np.sin(2 * np.pi * 440 * t) * 0.5
        return audio, sr
    
    @pytest.fixture
    def workflow_engine(self):
        """Create workflow integration engine for testing"""
        config = {
            "num_workers": 2,
            "timeout": 60.0,
            "enable_monitoring": False  # Disable for testing
        }
        return WorkflowIntegrationEngine(config)
    
    @pytest.mark.asyncio
    async def test_complete_music_analysis_workflow(self, workflow_engine, sample_audio_data):
        """Test complete music analysis workflow execution"""
        
        audio_data, sr = sample_audio_data
        
        request = WorkflowRequest(
            workflow_type=WorkflowType.COMPLETE_MUSIC_ANALYSIS,
            input_data={
                "audio_data": audio_data,
                "sample_rate": sr
            },
            parameters={
                "analysis_depth": "comprehensive"
            }
        )
        
        # Mock the integration engine methods
        with patch.object(workflow_engine.integration_engine, 'execute_operation') as mock_execute:
            mock_execute.return_value = AsyncMock()
            mock_execute.return_value.success = True
            mock_execute.return_value.result_data = {
                "features": {"mfcc": [1, 2, 3], "spectral_centroid": 0.5},
                "quality_score": 0.8
            }
            mock_execute.return_value.processing_time = 1.0
            
            result = await workflow_engine.execute_workflow(request)
            
            assert result.status == WorkflowStatus.COMPLETED
            assert result.result_data is not None
            assert result.execution_time > 0
            assert len(result.step_results) > 0
    
    @pytest.mark.asyncio
    async def test_intelligent_track_generation_workflow(self, workflow_engine, sample_audio_data):
        """Test intelligent track generation workflow"""
        
        audio_data, sr = sample_audio_data
        
        request = WorkflowRequest(
            workflow_type=WorkflowType.INTELLIGENT_TRACK_GENERATION,
            input_data={
                "audio_data": audio_data,
                "sample_rate": sr
            },
            parameters={
                "target_role": "bass",
                "instrument": "bass_guitar",
                "volume": 0.7
            }
        )
        
        with patch.object(workflow_engine.integration_engine, 'execute_operation') as mock_execute:
            mock_execute.return_value = AsyncMock()
            mock_execute.return_value.success = True
            mock_execute.return_value.result_data = {
                "generated_track": np.random.random(len(audio_data)),
                "harmonic_consistency": 0.8,
                "quality_score": 0.75
            }
            mock_execute.return_value.processing_time = 2.0
            
            result = await workflow_engine.execute_workflow(request)
            
            assert result.status == WorkflowStatus.COMPLETED
            assert "generated_track" in str(result.result_data)
    
    @pytest.mark.asyncio
    async def test_professional_audio_enhancement_workflow(self, workflow_engine, sample_audio_data):
        """Test professional audio enhancement workflow"""
        
        audio_data, sr = sample_audio_data
        
        request = WorkflowRequest(
            workflow_type=WorkflowType.PROFESSIONAL_AUDIO_ENHANCEMENT,
            input_data={
                "audio_data": audio_data,
                "sample_rate": sr
            },
            parameters={
                "enhancement_level": "professional"
            }
        )
        
        with patch.object(workflow_engine.integration_engine, 'execute_operation') as mock_execute:
            mock_execute.return_value = AsyncMock()
            mock_execute.return_value.success = True
            mock_execute.return_value.result_data = {
                "enhanced_audio": audio_data * 1.1,  # Mock enhancement
                "mastering_quality": 0.9,
                "quality_score": 0.85
            }
            mock_execute.return_value.processing_time = 1.5
            
            result = await workflow_engine.execute_workflow(request)
            
            assert result.status == WorkflowStatus.COMPLETED
            assert result.quality_metrics.get("overall_quality", 0) > 0.8
    
    @pytest.mark.asyncio
    async def test_workflow_validation_error_handling(self, workflow_engine):
        """Test workflow input validation and error handling"""
        
        # Test missing input data
        request = WorkflowRequest(
            workflow_type=WorkflowType.COMPLETE_MUSIC_ANALYSIS,
            input_data={},  # Empty input data
            parameters={"analysis_depth": "comprehensive"}
        )
        
        result = await workflow_engine.execute_workflow(request)
        assert result.status == WorkflowStatus.FAILED
        assert "required" in str(result.error_info).lower()
        
        # Test missing required parameters
        request = WorkflowRequest(
            workflow_type=WorkflowType.INTELLIGENT_TRACK_GENERATION,
            input_data={"audio_data": np.random.random(1000)},
            parameters={}  # Missing required parameters
        )
        
        result = await workflow_engine.execute_workflow(request)
        assert result.status == WorkflowStatus.FAILED
    
    @pytest.mark.asyncio
    async def test_workflow_step_failure_recovery(self, workflow_engine, sample_audio_data):
        """Test workflow recovery from step failures"""
        
        audio_data, sr = sample_audio_data
        
        request = WorkflowRequest(
            workflow_type=WorkflowType.COMPLETE_MUSIC_ANALYSIS,
            input_data={
                "audio_data": audio_data,
                "sample_rate": sr
            },
            parameters={"analysis_depth": "comprehensive"}
        )
        
        # Mock failure on first call, success on retry
        call_count = 0
        def mock_execute_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_result = AsyncMock()
            if call_count == 1:
                # Simulate failure on first attempt
                mock_result.success = False
                mock_result.error_info = {"message": "Temporary failure"}
            else:
                # Success on retry
                mock_result.success = True
                mock_result.result_data = {"quality_score": 0.7}
                mock_result.processing_time = 1.0
            return mock_result
        
        with patch.object(workflow_engine.integration_engine, 'execute_operation', 
                         side_effect=mock_execute_side_effect):
            
            result = await workflow_engine.execute_workflow(request)
            
            # Should eventually succeed after retry
            assert call_count > 1  # Retry was attempted
    
    @pytest.mark.asyncio
    async def test_quality_gate_enforcement(self, workflow_engine, sample_audio_data):
        """Test quality gate enforcement in workflows"""
        
        audio_data, sr = sample_audio_data
        
        request = WorkflowRequest(
            workflow_type=WorkflowType.COMPLETE_MUSIC_ANALYSIS,
            input_data={
                "audio_data": audio_data,
                "sample_rate": sr
            },
            parameters={"analysis_depth": "comprehensive"},
            quality_requirements={"min_quality": 0.8}  # High quality threshold
        )
        
        # Mock low quality result
        with patch.object(workflow_engine.integration_engine, 'execute_operation') as mock_execute:
            mock_execute.return_value = AsyncMock()
            mock_execute.return_value.success = True
            mock_execute.return_value.result_data = {
                "quality_score": 0.5  # Below threshold
            }
            mock_execute.return_value.processing_time = 1.0
            
            result = await workflow_engine.execute_workflow(request)
            
            # Should fail due to quality gate
            assert result.status == WorkflowStatus.FAILED
            assert "quality below threshold" in str(result.error_info).lower()
    
    @pytest.mark.asyncio
    async def test_parallel_step_execution(self, workflow_engine, sample_audio_data):
        """Test parallel execution of workflow steps"""
        
        audio_data, sr = sample_audio_data
        
        request = WorkflowRequest(
            workflow_type=WorkflowType.COMPLETE_MUSIC_ANALYSIS,
            input_data={
                "audio_data": audio_data,
                "sample_rate": sr
            },
            parameters={"analysis_depth": "comprehensive"}
        )
        
        execution_times = []
        
        def mock_execute_with_delay(*args, **kwargs):
            # Simulate processing time
            start_time = time.time()
            # Record execution start time
            execution_times.append(start_time)
            
            mock_result = AsyncMock()
            mock_result.success = True
            mock_result.result_data = {"quality_score": 0.8}
            mock_result.processing_time = 0.5
            return mock_result
        
        with patch.object(workflow_engine.integration_engine, 'execute_operation',
                         side_effect=mock_execute_with_delay):
            
            start_time = time.time()
            result = await workflow_engine.execute_workflow(request)
            total_time = time.time() - start_time
            
            assert result.status == WorkflowStatus.COMPLETED
            # Parallel execution should be faster than sequential
            assert total_time < len(execution_times) * 0.5
    
    @pytest.mark.asyncio
    async def test_exception_handling_and_recovery(self, workflow_engine, sample_audio_data):
        """Test comprehensive exception handling and recovery"""
        
        audio_data, sr = sample_audio_data
        
        # Test AudioProcessingError recovery
        request = WorkflowRequest(
            workflow_type=WorkflowType.PROFESSIONAL_AUDIO_ENHANCEMENT,
            input_data={
                "audio_data": audio_data,
                "sample_rate": sr
            },
            parameters={"enhancement_level": "professional"}
        )
        
        # Mock AudioProcessingError followed by successful recovery
        call_count = 0
        def mock_execute_with_recovery(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise AudioProcessingError("Audio processing failed")
            else:
                mock_result = AsyncMock()
                mock_result.success = True
                mock_result.result_data = {"quality_score": 0.6}  # Degraded quality
                mock_result.processing_time = 1.0
                return mock_result
        
        with patch.object(workflow_engine.integration_engine, 'execute_operation',
                         side_effect=mock_execute_with_recovery):
            
            result = await workflow_engine.execute_workflow(request)
            
            # Should recover from the error
            assert len(result.recovery_actions) > 0
    
    def test_system_status_monitoring(self, workflow_engine):
        """Test system status and monitoring capabilities"""
        
        status = workflow_engine.get_system_status()
        
        assert "active_workflows" in status
        assert "total_workflows_executed" in status
        assert "success_rate" in status
        assert "system_health" in status
        assert "workflow_types_supported" in status
        
        # Verify workflow types are properly listed
        assert len(status["workflow_types_supported"]) > 0
        assert "complete_music_analysis" in status["workflow_types_supported"]
    
    @pytest.mark.asyncio
    async def test_workflow_metrics_tracking(self, workflow_engine, sample_audio_data):
        """Test workflow execution metrics tracking"""
        
        audio_data, sr = sample_audio_data
        
        initial_metrics = workflow_engine.metrics.copy()
        
        request = WorkflowRequest(
            workflow_type=WorkflowType.COMPLETE_MUSIC_ANALYSIS,
            input_data={
                "audio_data": audio_data,
                "sample_rate": sr
            },
            parameters={"analysis_depth": "comprehensive"}
        )
        
        with patch.object(workflow_engine.integration_engine, 'execute_operation') as mock_execute:
            mock_execute.return_value = AsyncMock()
            mock_execute.return_value.success = True
            mock_execute.return_value.result_data = {"quality_score": 0.8}
            mock_execute.return_value.processing_time = 1.0
            
            await workflow_engine.execute_workflow(request)
            
            # Verify metrics were updated
            assert workflow_engine.metrics["total_workflows"] > initial_metrics["total_workflows"]
            assert workflow_engine.metrics["successful_workflows"] > initial_metrics["successful_workflows"]
            assert workflow_engine.metrics["average_execution_time"] > 0
    
    @pytest.mark.asyncio
    async def test_workflow_shutdown_cleanup(self, workflow_engine):
        """Test proper cleanup during shutdown"""
        
        # Start a workflow
        request = WorkflowRequest(
            workflow_type=WorkflowType.COMPLETE_MUSIC_ANALYSIS,
            input_data={"audio_data": np.random.random(1000)},
            parameters={"analysis_depth": "basic"}
        )
        
        # Add to active workflows (simulate running workflow)
        workflow_engine.active_workflows["test_workflow"] = request
        
        # Test shutdown
        await workflow_engine.shutdown()
        
        # Verify cleanup
        assert len(workflow_engine.active_workflows) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])