import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from detector import NsfwDetector

@pytest.fixture
def mock_nude_detector():
    """Fixture for a mock NudeDetector instance."""
    mock = MagicMock()
    mock.detect.return_value = [
        {'box': [10, 10, 20, 20], 'score': 0.9, 'class': 'TEST_CLASS_1'},
        {'box': [30, 30, 40, 40], 'score': 0.7, 'class': 'TEST_CLASS_2'}
    ]
    return mock

def test_nsfw_detector_init_with_nude_detector_instance(mock_nude_detector):
    """
    Test that NsfwDetector uses the provided NudeDetector instance.
    """
    detector = NsfwDetector(nude_detector_instance=mock_nude_detector)
    assert detector.detector == mock_nude_detector
    # Ensure NudeDetector's constructor was NOT called
    mock_nude_detector.assert_not_called()

def test_nsfw_detector_init_creates_nude_detector():
    """
    Test that NsfwDetector creates a NudeDetector instance if none is provided.
    """
    with patch('detector.NudeDetector') as MockNudeDetector:
        detector = NsfwDetector(model_path='test_path', inference_resolution=640)
        MockNudeDetector.assert_called_once_with(model_path='test_path', inference_resolution=640)
        assert detector.detector == MockNudeDetector.return_value

def test_detect_returns_detections(mock_nude_detector):
    """
    Test that the detect method correctly returns detections from the NudeDetector.
    """
    detector = NsfwDetector(nude_detector_instance=mock_nude_detector)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    detections = detector.detect(frame)

    mock_nude_detector.detect.assert_called_once_with(frame)
    expected_detections = [
        {'box': [10, 10, 20, 20], 'score': 0.9, 'class': 'TEST_CLASS_1'},
        {'box': [30, 30, 40, 40], 'score': 0.7, 'class': 'TEST_CLASS_2'}
    ]
    assert detections == expected_detections

def test_detect_with_no_detections(mock_nude_detector):
    """
    Test that the detect method handles cases with no detections.
    """
    mock_nude_detector.detect.return_value = []
    detector = NsfwDetector(nude_detector_instance=mock_nude_detector)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    detections = detector.detect(frame)

    mock_nude_detector.detect.assert_called_once_with(frame)
    assert detections == []
