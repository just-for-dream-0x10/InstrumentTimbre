"""

"""

import pytest
from unittest.mock import Mock, patch
from InstrumentTimbre.core.operations.operation_dispatcher import (
    OperationDispatcher, get_operation_dispatcher, intelligent_track_operation
)
from InstrumentTimbre.core.operations.data_structures import (
    OperationRequest, OperationType, TrackRole, TrackData,
    EmotionConstraints, MusicConstraints, EmotionType,
    create_empty_track
)


class TestOperationDispatcher:
    """Operation"""
    
    def setup_method(self):
        """"""
        self.dispatcher = OperationDispatcher()
        
        # 
        self.dispatcher.track_generator = Mock()
        self.dispatcher.track_replacer = Mock()
        self.dispatcher.track_repairer = Mock()
        self.dispatcher.orchestrator = Mock()
        self.dispatcher.conflict_detector = Mock()
        self.dispatcher._initialized = True
    
    def test_initialization(self):
        """"""
        dispatcher = OperationDispatcher()
        assert not dispatcher._initialized
        
        status = dispatcher.get_status()
        assert not status["initialized"]
    
    def test_parse_natural_language_request(self):
        """"""
        current_tracks = [create_empty_track("track1", "piano", TrackRole.MELODY)]
        
        # generate
        request = self.dispatcher.parse_natural_language_request(
            "", current_tracks
        )
        
        assert request is not None
        assert request.operation_type == OperationType.GENERATE
        assert request.target_instrument == "violin"
        assert request.target_role == TrackRole.HARMONY
    
    def test_parse_replacement_request(self):
        """"""
        current_tracks = [create_empty_track("track1", "piano", TrackRole.MELODY)]
        
        request = self.dispatcher.parse_natural_language_request(
            "", current_tracks
        )
        
        assert request is not None
        assert request.operation_type == OperationType.REPLACE
        assert request.target_instrument == "guitar"
    
    def test_parse_repair_request(self):
        """"""
        current_tracks = [create_empty_track("track1", "piano", TrackRole.MELODY)]
        
        request = self.dispatcher.parse_natural_language_request(
            "", current_tracks
        )
        
        assert request is not None
        assert request.operation_type == OperationType.REPAIR
    
    def test_parse_unknown_request(self):
        """"""
        current_tracks = [create_empty_track("track1", "piano", TrackRole.MELODY)]
        
        request = self.dispatcher.parse_natural_language_request(
            "", current_tracks
        )
        
        assert request is None
    
    def test_process_generation_request(self):
        """processgenerate"""
        # 
        mock_result = Mock()
        mock_result.success = True
        mock_result.generated_track = create_empty_track("new", "violin", TrackRole.HARMONY)
        self.dispatcher.track_generator.generate_track.return_value = mock_result
        self.dispatcher.conflict_detector.detect_conflicts.return_value = []
        
        request = OperationRequest(
            operation_type=OperationType.GENERATE,
            target_instrument="violin",
            target_role=TrackRole.HARMONY
        )
        
        current_tracks = [create_empty_track("track1", "piano", TrackRole.MELODY)]
        
        result = self.dispatcher.process_request(request, current_tracks)
        
        assert result.success
        assert self.dispatcher.track_generator.generate_track.called
    
    def test_process_replacement_request(self):
        """process"""
        # 
        mock_result = Mock()
        mock_result.success = True
        mock_result.generated_track = create_empty_track("replaced", "guitar", TrackRole.MELODY)
        self.dispatcher.track_replacer.replace_track.return_value = mock_result
        self.dispatcher.conflict_detector.detect_conflicts.return_value = []
        
        request = OperationRequest(
            operation_type=OperationType.REPLACE,
            target_instrument="guitar",
            target_role=TrackRole.MELODY,
            reference_track="track1"
        )
        
        current_tracks = [create_empty_track("track1", "piano", TrackRole.MELODY)]
        
        result = self.dispatcher.process_request(request, current_tracks)
        
        assert result.success
        assert self.dispatcher.track_replacer.replace_track.called
    
    def test_process_repair_request(self):
        """process"""
        # 
        mock_result = Mock()
        mock_result.success = True
        mock_result.generated_track = create_empty_track("repaired", "piano", TrackRole.MELODY)
        self.dispatcher.track_repairer.repair_track.return_value = mock_result
        self.dispatcher.conflict_detector.detect_conflicts.return_value = []
        
        request = OperationRequest(
            operation_type=OperationType.REPAIR,
            target_instrument="piano",
            target_role=TrackRole.MELODY,
            reference_track="track1"
        )
        
        current_tracks = [create_empty_track("track1", "piano", TrackRole.MELODY)]
        
        result = self.dispatcher.process_request(request, current_tracks)
        
        assert result.success
        assert self.dispatcher.track_repairer.repair_track.called
    
    def test_process_invalid_request(self):
        """process"""
        request = OperationRequest(
            operation_type=OperationType.GENERATE,
            target_instrument="violin",
            target_role=TrackRole.HARMONY,
            intensity=1.5  # 
        )
        
        current_tracks = [create_empty_track("track1", "piano", TrackRole.MELODY)]
        
        result = self.dispatcher.process_request(request, current_tracks)
        
        assert not result.success
        assert "" in result.warnings
    
    def test_missing_reference_track(self):
        """"""
        self.dispatcher.track_replacer.replace_track.return_value = Mock(success=False)
        
        request = OperationRequest(
            operation_type=OperationType.REPLACE,
            target_instrument="guitar",
            target_role=TrackRole.MELODY,
            reference_track="nonexistent"  # 
        )
        
        current_tracks = [create_empty_track("track1", "piano", TrackRole.MELODY)]
        
        result = self.dispatcher.process_request(request, current_tracks)
        
        assert not result.success
        assert "" in result.warnings


class TestGlobalFunctions:
    """"""
    
    @patch('InstrumentTimbre.core.operations.operation_dispatcher.OperationDispatcher')
    def test_get_operation_dispatcher_singleton(self, mock_dispatcher_class):
        """（）"""
        # 
        import InstrumentTimbre.core.operations.operation_dispatcher as dispatcher_module
        dispatcher_module._dispatcher_instance = None
        
        # ，
        dispatcher1 = get_operation_dispatcher()
        dispatcher2 = get_operation_dispatcher()
        
        assert dispatcher1 is dispatcher2
        assert mock_dispatcher_class.call_count == 1
    
    def test_intelligent_track_operation_with_string(self):
        """Operation"""
        current_tracks = [create_empty_track("track1", "piano", TrackRole.MELODY)]
        
        # 
        with patch('InstrumentTimbre.core.operations.operation_dispatcher.get_operation_dispatcher') as mock_get_dispatcher:
            mock_dispatcher = Mock()
            mock_dispatcher.parse_natural_language_request.return_value = OperationRequest(
                operation_type=OperationType.GENERATE,
                target_instrument="violin",
                target_role=TrackRole.HARMONY
            )
            mock_dispatcher.process_request.return_value = Mock(success=True)
            mock_get_dispatcher.return_value = mock_dispatcher
            
            result = intelligent_track_operation("", current_tracks)
            
            assert mock_dispatcher.parse_natural_language_request.called
            assert mock_dispatcher.process_request.called
    
    def test_intelligent_track_operation_with_request_object(self):
        """Operation"""
        current_tracks = [create_empty_track("track1", "piano", TrackRole.MELODY)]
        
        request = OperationRequest(
            operation_type=OperationType.GENERATE,
            target_instrument="violin",
            target_role=TrackRole.HARMONY
        )
        
        # 
        with patch('InstrumentTimbre.core.operations.operation_dispatcher.get_operation_dispatcher') as mock_get_dispatcher:
            mock_dispatcher = Mock()
            mock_dispatcher.process_request.return_value = Mock(success=True)
            mock_get_dispatcher.return_value = mock_dispatcher
            
            result = intelligent_track_operation(request, current_tracks)
            
            assert mock_dispatcher.process_request.called
            # 
            assert not mock_dispatcher.parse_natural_language_request.called
    
    def test_intelligent_track_operation_with_constraints(self):
        """Operation"""
        current_tracks = [create_empty_track("track1", "piano", TrackRole.MELODY)]
        
        request = OperationRequest(
            operation_type=OperationType.GENERATE,
            target_instrument="violin",
            target_role=TrackRole.HARMONY
        )
        
        emotion_constraints = EmotionConstraints(
            primary_emotion=EmotionType.HAPPY,
            intensity=0.8
        )
        
        music_constraints = MusicConstraints(
            key="C_major",
            time_signature="4/4",
            tempo=120
        )
        
        with patch('InstrumentTimbre.core.operations.operation_dispatcher.get_operation_dispatcher') as mock_get_dispatcher:
            mock_dispatcher = Mock()
            mock_dispatcher.process_request.return_value = Mock(success=True)
            mock_get_dispatcher.return_value = mock_dispatcher
            
            result = intelligent_track_operation(
                request, current_tracks, 
                emotion_constraints=emotion_constraints,
                music_constraints=music_constraints
            )
            
            # 
            call_args = mock_dispatcher.process_request.call_args
            processed_request = call_args[0][0]
            
            assert processed_request.emotion_constraints == emotion_constraints
            assert processed_request.music_constraints == music_constraints


if __name__ == "__main__":
    pytest.main([__file__])