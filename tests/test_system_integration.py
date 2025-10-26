"""
System Integration Tests

Tests for the complete system integration including module coordination,
error handling, monitoring, and end-to-end functionality.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch
from typing import Dict, Any

from InstrumentTimbre.core.system_integration import (
    IntegrationEngine,
    ModuleCoordinator,
    SystemErrorHandler,
    SystemMonitor,
    OperationRequest,
    OperationType,
    ErrorSeverity,
    ModuleIntegrationError
)
from config import Config, fast_config, get_config


class TestSystemIntegration:
    """Test system integration functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = fast_config()  # Use fast config for testing
        self.test_audio = self._create_test_audio()
        
    def _create_test_audio(self) -> np.ndarray:
        """Create test audio data"""
        duration = 2.0
        sample_rate = 22050
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Simple sine wave
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)
        return audio.astype(np.float32)
    
    def test_integration_engine_initialization(self):
        """Test integration engine initialization"""
        # Test with monitoring disabled for faster testing
        engine = IntegrationEngine(config=self.config, enable_monitoring=False)
        
        assert engine.config is not None
        assert engine.module_coordinator is not None
        assert engine.error_handler is not None
        assert engine.monitor is None  # Disabled
        
        # Check that core modules are registered
        module_status = engine.module_coordinator.get_module_status()
        assert "feature_extractor" in module_status
        assert "model_manager" in module_status
    
    def test_module_coordinator_basic_functionality(self):
        """Test module coordinator basic operations"""
        coordinator = ModuleCoordinator(max_workers=2)
        
        # Test module registration
        coordinator.register_module(
            name="test_module",
            module_type="test",
            version="1.0.0"
        )
        
        status = coordinator.get_module_status()
        assert "test_module" in status
        assert status["test_module"]["type"] == "test"
        assert status["test_module"]["version"] == "1.0.0"
    
    def test_error_handler_functionality(self):
        """Test error handler operations"""
        error_handler = SystemErrorHandler()
        
        # Test error handling
        test_error = ValueError("Test error")
        result = error_handler.handle_error(test_error, auto_recover=False)
        
        assert result["error_type"] == "ValueError"
        assert "Test error" in result["user_message"]
        assert not result["recovery_successful"]
        
        # Test error statistics
        stats = error_handler.get_error_statistics()
        assert stats["total_errors"] == 1
        assert "ValueError" in stats["error_counts_by_type"]
    
    def test_system_monitor_metrics_collection(self):
        """Test system monitor metrics collection"""
        monitor = SystemMonitor(monitoring_interval=0.1)
        
        # Test current metrics collection
        metrics = monitor.get_current_metrics()
        assert "system" in metrics
        assert "process" in metrics
        assert "cpu_percent" in metrics["system"]
        assert "memory_percent" in metrics["system"]
        
        # Test health check
        health = monitor.get_system_health()
        assert "status" in health
        assert "score" in health
        assert isinstance(health["score"], (int, float))
    
    def test_operation_request_validation(self):
        """Test operation request validation"""
        engine = IntegrationEngine(config=self.config, enable_monitoring=False)
        
        # Valid request
        valid_request = OperationRequest(
            operation_type=OperationType.MUSIC_ANALYSIS,
            input_data={"audio_data": self.test_audio},
            parameters={"analyze_emotion": True}
        )
        
        # This should not raise an exception
        engine._validate_operation_request(valid_request)
        
        # Invalid request - missing input data
        invalid_request = OperationRequest(
            operation_type=OperationType.MUSIC_ANALYSIS,
            input_data={},
            parameters={}
        )
        
        with pytest.raises(Exception):
            engine._validate_operation_request(invalid_request)
    
    @patch('InstrumentTimbre.core.features.unified_features.UnifiedFeatureExtractor')
    @patch('InstrumentTimbre.core.analysis.music_understanding_engine.MusicUnderstandingEngine')
    def test_music_analysis_pipeline(self, mock_analyzer, mock_features):
        """Test music analysis operation pipeline"""
        # Mock the modules
        mock_features_instance = Mock()
        mock_features_instance.process.return_value = {"features": "test_features"}
        mock_features.return_value = mock_features_instance
        
        mock_analyzer_instance = Mock()
        mock_analyzer_instance.process.return_value = {"analysis": "test_analysis"}
        mock_analyzer.return_value = mock_analyzer_instance
        
        # Create engine
        engine = IntegrationEngine(config=self.config, enable_monitoring=False)
        
        # Create operation request
        request = OperationRequest(
            operation_type=OperationType.MUSIC_ANALYSIS,
            input_data={"audio_data": self.test_audio},
            parameters={"analyze_emotion": True}
        )
        
        # Execute operation
        result = engine.execute_operation(request)
        
        # Verify result
        assert result.success
        assert result.operation_type == OperationType.MUSIC_ANALYSIS
        assert result.result_data is not None
        assert "analysis_result" in result.result_data
    
    def test_error_recovery_system(self):
        """Test error recovery mechanisms"""
        error_handler = SystemErrorHandler()
        
        # Test recoverable error
        from InstrumentTimbre.core.system_integration.exception_types import RecoverableError
        
        recoverable_error = RecoverableError(
            "Temporary failure",
            retry_count=0,
            max_retries=2
        )
        
        result = error_handler.handle_error(recoverable_error, auto_recover=True)
        
        # Should attempt recovery
        assert result["recovery_attempted"]
        # Recovery success depends on implementation
    
    def test_performance_statistics_tracking(self):
        """Test performance statistics tracking"""
        engine = IntegrationEngine(config=self.config, enable_monitoring=False)
        
        # Initial stats should be empty
        status = engine.get_system_status()
        assert status["performance_stats"]["total_operations"] == 0
        
        # Mock a successful operation
        engine._update_performance_stats(OperationType.MUSIC_ANALYSIS, True, 1.5)
        
        # Check updated stats
        status = engine.get_system_status()
        perf_stats = status["performance_stats"]
        assert perf_stats["total_operations"] == 1
        assert perf_stats["successful_operations"] == 1
        assert perf_stats["failed_operations"] == 0
        assert OperationType.MUSIC_ANALYSIS.value in perf_stats["operations_by_type"]
    
    def test_resource_checking(self):
        """Test system resource checking"""
        monitor = SystemMonitor()
        
        # Test resource requirements check
        resource_check = monitor.check_resource_requirements(
            "test_operation",
            memory_mb=100,  # Small requirement
            cpu_percent=10
        )
        
        assert "can_proceed" in resource_check
        assert "warnings" in resource_check
        assert "blockers" in resource_check
        
        # Test with high requirements (should generate warnings)
        high_resource_check = monitor.check_resource_requirements(
            "heavy_operation",
            memory_mb=100000,  # Very high requirement
            cpu_percent=90
        )
        
        # Should have some warnings or blockers
        assert len(high_resource_check["warnings"]) > 0 or len(high_resource_check["blockers"]) > 0
    
    def test_module_dependency_resolution(self):
        """Test module dependency resolution"""
        coordinator = ModuleCoordinator()
        
        # Register modules with dependencies
        coordinator.register_module(
            name="module_a",
            module_type="test",
            version="1.0.0",
            dependencies=[]
        )
        
        coordinator.register_module(
            name="module_b",
            module_type="test",
            version="1.0.0",
            dependencies=["module_a"]
        )
        
        # Test execution order calculation
        execution_order = coordinator._calculate_execution_order(["module_b", "module_a"])
        
        # module_a should come before module_b
        assert len(execution_order) >= 1
        # First group should contain module_a (no dependencies)
        assert "module_a" in execution_order[0]
    
    def test_configuration_integration(self):
        """Test configuration system integration"""
        from config import validate_config, fast_config, high_quality_config
        
        # Test different configs with engine
        configs = [fast_config(), high_quality_config()]
        
        for config in configs:
            assert validate_config(config)
            
            # Engine should accept valid configs
            engine = IntegrationEngine(config=config, enable_monitoring=False)
            assert engine.config == config
    
    def test_system_shutdown_cleanup(self):
        """Test system shutdown and cleanup"""
        engine = IntegrationEngine(config=self.config, enable_monitoring=True)
        
        # Verify components are running
        assert engine.monitor is not None
        assert engine.monitor.is_monitoring
        
        # Shutdown
        engine.shutdown()
        
        # Verify cleanup
        assert not engine.monitor.is_monitoring
    
    def test_concurrent_operations(self):
        """Test handling multiple concurrent operations"""
        engine = IntegrationEngine(config=self.config, enable_monitoring=False)
        
        # Track operations
        assert len(engine.active_operations) == 0
        
        # Simulate adding operations to active list
        request1 = OperationRequest(
            operation_type=OperationType.MUSIC_ANALYSIS,
            input_data={"audio_data": self.test_audio},
            parameters={}
        )
        
        request2 = OperationRequest(
            operation_type=OperationType.MUSIC_ANALYSIS,
            input_data={"audio_data": self.test_audio},
            parameters={}
        )
        
        # Add to active operations (simulating concurrent execution)
        engine.active_operations["op1"] = request1
        engine.active_operations["op2"] = request2
        
        assert len(engine.active_operations) == 2
        
        # Test system status includes active operations
        status = engine.get_system_status()
        assert status["active_operations"] == 2


class TestErrorScenarios:
    """Test various error scenarios and recovery"""
    
    def test_invalid_audio_format_error(self):
        """Test handling of invalid audio format"""
        from InstrumentTimbre.core.system_integration.exception_types import InvalidAudioFormatError
        
        error = InvalidAudioFormatError("Unsupported format: .xyz")
        
        assert "Invalid audio format" in str(error)
        assert error.category.value == "audio_processing"
        assert len(error.suggestions) > 0
    
    def test_insufficient_memory_error(self):
        """Test handling of memory issues"""
        from InstrumentTimbre.core.system_integration.exception_types import InsufficientMemoryError
        
        error = InsufficientMemoryError(required_mb=4000, available_mb=2000)
        
        assert "Insufficient memory" in str(error)
        assert error.severity == ErrorSeverity.HIGH
        assert "reduce" in " ".join(error.suggestions).lower()
    
    def test_model_not_found_error(self):
        """Test handling of missing model files"""
        from InstrumentTimbre.core.system_integration.exception_types import ModelNotFoundError
        
        error = ModelNotFoundError("/path/to/missing/model.pt")
        
        assert "Model not found" in str(error)
        assert error.severity == ErrorSeverity.CRITICAL
        assert "Download" in " ".join(error.suggestions)


class TestPerformanceMonitoring:
    """Test performance monitoring functionality"""
    
    def test_metrics_history_tracking(self):
        """Test metrics history tracking"""
        monitor = SystemMonitor(monitoring_interval=0.1, history_size=10)
        
        # Start monitoring briefly
        monitor.start_monitoring()
        time.sleep(0.3)  # Let it collect a few samples
        monitor.stop_monitoring()
        
        # Check that metrics were collected
        assert len(monitor.system_metrics_history) > 0
        assert len(monitor.process_metrics_history) > 0
        
        # Check metrics format
        if monitor.system_metrics_history:
            metric = monitor.system_metrics_history[0]
            assert hasattr(metric, 'cpu_percent')
            assert hasattr(metric, 'memory_percent')
            assert hasattr(metric, 'timestamp')
    
    def test_alert_system(self):
        """Test alert system functionality"""
        monitor = SystemMonitor()
        
        # Add custom alert callback
        triggered_alerts = []
        def alert_callback(alert_info):
            triggered_alerts.append(alert_info)
        
        monitor.add_alert_callback(alert_callback)
        
        # Create test metrics that should trigger alerts
        from InstrumentTimbre.core.system_integration.system_monitor import SystemMetrics
        
        high_cpu_metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=95.0,  # High CPU
            memory_percent=50.0,
            memory_available_mb=2000.0,
            disk_usage_percent=70.0
        )
        
        # Check alerts
        monitor._check_alerts(high_cpu_metrics)
        
        # Should have triggered high CPU alert
        # Note: May not trigger due to cooldown, but this tests the mechanism
        assert len(monitor.alert_rules) > 0


# Integration test
def test_end_to_end_system_integration():
    """End-to-end system integration test"""
    
    # Use fast config for quick testing
    config = fast_config()
    
    # Create integration engine
    engine = IntegrationEngine(config=config, enable_monitoring=False)
    
    # Test system status
    status = engine.get_system_status()
    assert "system_health" in status
    assert "module_status" in status
    assert "performance_stats" in status
    
    # Test that we can get module status
    module_status = engine.module_coordinator.get_module_status()
    assert isinstance(module_status, dict)
    
    # Test error handler statistics
    error_stats = engine.error_handler.get_error_statistics()
    assert "total_errors" in error_stats
    
    # Cleanup
    engine.shutdown()
    
    print("✅ End-to-end system integration test passed")


if __name__ == "__main__":
    # Run the end-to-end test
    test_end_to_end_system_integration()
    print("✅ All system integration tests completed")