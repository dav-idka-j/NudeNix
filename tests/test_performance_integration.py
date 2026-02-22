import pytest
import numpy as np
from unittest.mock import MagicMock, patch, ANY, call
from main import MainProcessor
from statistics import (
    PerformanceStatistics,
    PipelineSummaryModule,
    ReaderReportModule,
    OutputReportModule,
    HashingReportModule,
    DetectionReportModule,
    CensorReportModule
)
from preprocess_detector import ImageHashDetector
from detector import NsfwDetector
from censor import CensorProcessor
from censor_drawing_contexts.in_place_drawing_context import InPlaceDrawingContext
from detection_types import Detection

def test_main_processor_registers_modules():
    """Verify that MainProcessor registers its expected reporting modules."""
    mock_stats = MagicMock(spec=PerformanceStatistics)
    mock_stats.enabled = True

    MainProcessor(
        reader=MagicMock(),
        detector=MagicMock(),
        censor_processor=MagicMock(),
        writer=MagicMock(),
        config_loader=MagicMock(),
        stats=mock_stats
    )

    # Check for registration calls
    # Summary priority 0, Reader 10, Output 50
    mock_stats.register_module.assert_any_call(ANY, priority=0)
    mock_stats.register_module.assert_any_call(ANY, priority=10)
    mock_stats.register_module.assert_any_call(ANY, priority=50)

def test_imagehash_detector_registers_module():
    """Verify ImageHashDetector registers its reporting module."""
    mock_stats = MagicMock(spec=PerformanceStatistics)
    mock_stats.enabled = True

    ImageHashDetector(MagicMock(), {'enabled': True}, stats=mock_stats)

    mock_stats.register_module.assert_called_once_with(ANY, priority=20)

def test_nsfw_detector_registers_module():
    """Verify NsfwDetector registers its reporting module."""
    mock_stats = MagicMock(spec=PerformanceStatistics)
    mock_stats.enabled = True

    # Mock NudeDetector to avoid loading model
    with patch('detector.NudeDetector'):
        NsfwDetector(stats=mock_stats)

    mock_stats.register_module.assert_called_once_with(ANY, priority=30)

def test_censor_processor_registers_module():
    """Verify CensorProcessor registers its reporting module."""
    mock_stats = MagicMock(spec=PerformanceStatistics)
    mock_stats.enabled = True

    CensorProcessor(MagicMock(), MagicMock(), stats=mock_stats)

    mock_stats.register_module.assert_called_once_with(ANY, priority=40)

def test_main_processor_calls_stats_timers():
    """Verify that MainProcessor uses stats timers during its run."""
    mock_reader = MagicMock()
    mock_reader.open.return_value = True
    # Return one frame then stop
    mock_reader.read_frame.side_effect = [(np.zeros((10, 10, 3), np.uint8), None, 0.0), (None, None, None)]
    mock_reader.get_fps.return_value = 30.0

    mock_detector = MagicMock()
    mock_detector.detect.return_value = []

    mock_censor_processor = MagicMock()
    mock_censor_processor.apply_censorship.return_value = np.zeros((10, 10, 3), np.uint8)

    mock_writer = MagicMock()
    mock_writer.open.return_value = True
    mock_writer.get_fps.return_value = 30.0
    mock_writer.get_video_time_base.return_value = MagicMock()

    mock_stats = MagicMock(spec=PerformanceStatistics)
    mock_stats.enabled = True

    processor = MainProcessor(
        reader=mock_reader,
        detector=mock_detector,
        censor_processor=mock_censor_processor,
        writer=mock_writer,
        config_loader=MagicMock(),
        stats=mock_stats
    )

    processor.run()

    # Verify major timers were called
    mock_stats.timer.assert_any_call('reader')
    mock_stats.timer.assert_any_call('censors_total')
    mock_stats.timer.assert_any_call('output')

    timer_calls = [c[0][0] for c in mock_stats.timer.call_args_list]
    assert 'detection' not in timer_calls

    mock_stats.increment_counter.assert_any_call('frame_count')
    mock_stats.start.assert_called_once()
    mock_stats.stop.assert_called_once()
    mock_stats.print_report.assert_called_once()

def test_imagehash_detector_records_stats():
    """Verify ImageHashDetector records hashing and detection metrics."""
    mock_base_detector = MagicMock()
    mock_stats = MagicMock(spec=PerformanceStatistics)
    mock_stats.enabled = True

    detector = ImageHashDetector(mock_base_detector, {'enabled': True})

    detector.detect(np.zeros((10, 10, 3), np.uint8), stats=mock_stats)

    mock_stats.timer.assert_any_call('hashing')
    mock_stats.timer.assert_any_call('detection')

def test_nsfw_detector_records_stats():
    """Verify NsfwDetector records detection metrics."""
    # We need to mock the underlying NudeDetector to avoid loading the model
    mock_nude_detector = MagicMock()
    mock_nude_detector.detect.return_value = [{'class': 'FACE_FEMALE', 'score': 0.9, 'box': [0,0,10,10]}]

    mock_stats = MagicMock(spec=PerformanceStatistics)
    mock_stats.enabled = True

    detector = NsfwDetector(nude_detector_instance=mock_nude_detector)

    detector.detect(np.zeros((10, 10, 3), np.uint8), stats=mock_stats)

    mock_stats.increment_counter.assert_called_once_with('detection_runs')
    mock_stats.record_categorized_counter.assert_called_once_with('label_frequency', 'FACE_FEMALE')

def test_inplace_drawing_context_records_censor_time():
    """Verify InPlaceDrawingContext times individual censor applications."""
    mock_registry = MagicMock()
    mock_censor = MagicMock()
    mock_censor.get_type.return_value = 'test_censor'
    mock_registry.get_censor_instance.return_value = mock_censor

    mock_stats = MagicMock(spec=PerformanceStatistics)
    mock_stats.enabled = True

    ctx = InPlaceDrawingContext(mock_registry)

    detection = Detection({'class': 'test', 'score': 1.0, 'box': [0, 0, 10, 10]})

    ctx.apply_censor_effect(np.zeros((20, 20, 3), np.uint8), detection, 'test_censor', {}, stats=mock_stats)

    mock_stats.time_censor.assert_called_once_with('test_censor')
