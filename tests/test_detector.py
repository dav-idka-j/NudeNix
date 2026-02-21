import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from detector import NsfwDetector
from detection_types import Detection # Import the Detection wrapper class

@pytest.fixture
def mock_nude_detector():
    """Fixture for a mock NudeDetector instance."""
    mock = MagicMock()
    # mock.detect.return_value should return raw dictionaries as NudeDetector does
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
    Test that the detect method correctly returns Detection objects.
    """
    detector = NsfwDetector(nude_detector_instance=mock_nude_detector)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    detections = detector.detect(frame)

    mock_nude_detector.detect.assert_called_once_with(frame)

    assert len(detections) == 2
    assert isinstance(detections[0], Detection)
    assert detections[0].x() == 10
    assert detections[0].y() == 10
    assert detections[0].width() == 20
    assert detections[0].height() == 20
    assert detections[0].score() == 0.9
    assert detections[0].label() == 'TEST_CLASS_1'

    assert isinstance(detections[1], Detection)
    assert detections[1].x() == 30
    assert detections[1].y() == 30
    assert detections[1].width() == 40
    assert detections[1].height() == 40
    assert detections[1].score() == 0.7
    assert detections[1].label() == 'TEST_CLASS_2'


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
