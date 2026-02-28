import pytest
from unittest.mock import MagicMock, patch, ANY
import numpy as np
import cv2 # For cv2.rectangle and cv2.resize in InPlaceDrawingContext tests
from censor import CensorProcessor
from config_loader import ConfigLoader
from interfaces.base_censor_drawing_context import BaseCensorDrawingContext
from censor_drawing_contexts.in_place_drawing_context import InPlaceDrawingContext
from censors_registry import CensorsRegistry
from detection_types import Detection # Import the Detection wrapper class

@pytest.fixture
def mock_config_loader():
    """Fixture for a mock ConfigLoader configured for TEST_LABEL with two censors.
    Returns a NEW dictionary each time get_label_settings is called to prevent test interference.
    """
    mock_loader = MagicMock(spec=ConfigLoader)

    def _get_label_settings_side_effect(label):
        if label == 'TEST_LABEL':
            # Ensure min_score is a float, not a mock
            return {
                'enabled': True,
                'min_score': 0.8,
                'censors': [
                    {
                        'type': 'box',
                        'settings': {'color': (0, 0, 255), 'thickness': -1}
                    },
                    {
                        'type': 'pixelation',
                        'settings': {'pixel_size': 16}
                    }
                ]
            }
        return {'enabled': False, 'min_score': 0.0, 'censors': []} # Default for other labels, ensure min_score is float

    mock_loader.get_label_settings.side_effect = _get_label_settings_side_effect
    return mock_loader

@pytest.fixture
def mock_drawing_context():
    """Fixture for a mock BaseCensorDrawingContext."""
    mock_context = MagicMock(spec=BaseCensorDrawingContext)
    return mock_context

def test_censor_processor_initialization(mock_config_loader, mock_drawing_context):
    """Test that CensorProcessor initializes correctly with a drawing context."""
    censor_processor = CensorProcessor(mock_config_loader, mock_drawing_context)
    assert censor_processor.config_loader == mock_config_loader
    assert censor_processor.drawing_context == mock_drawing_context

def test_apply_censorship_no_detections(mock_config_loader, mock_drawing_context):
    """Test that no censorship is applied when there are no detections."""
    censor_processor = CensorProcessor(mock_config_loader, mock_drawing_context)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = []

    processed_frame = censor_processor.apply_censorship(frame.copy(), detections)

    # The frame should be unchanged if no drawing calls are made
    assert np.array_equal(frame, processed_frame)
    mock_drawing_context.apply_censor_effect.assert_not_called()

def test_apply_censorship_disabled_label(mock_config_loader, mock_drawing_context):
    """Test that no censorship is applied for a disabled label."""
    # Configure mock to return disabled settings for TEST_LABEL
    mock_config_loader.get_label_settings.side_effect = lambda label: {
        'enabled': False, 'min_score': 0.8, 'censors': []
    } if label == 'TEST_LABEL' else {'enabled': False, 'min_score': 0.0, 'censors': []}

    censor_processor = CensorProcessor(mock_config_loader, mock_drawing_context)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = [Detection({'class': 'TEST_LABEL', 'score': 0.9, 'box': [10, 10, 10, 10]})]

    processed_frame = censor_processor.apply_censorship(frame.copy(), detections)

    assert np.array_equal(frame, processed_frame)
    mock_drawing_context.apply_censor_effect.assert_not_called()


def test_apply_censorship_low_score(mock_config_loader, mock_drawing_context):
    """Test that no censorship is applied for a detection with a low score."""
    # Configure mock to return settings with a high min_score for TEST_LABEL
    mock_config_loader.get_label_settings.side_effect = lambda label: {
        'enabled': True, 'min_score': 0.95, 'censors': [{'type': 'box', 'settings': {}}]
    } if label == 'TEST_LABEL' else {'enabled': False, 'min_score': 0.0, 'censors': []}


    censor_processor = CensorProcessor(mock_config_loader, mock_drawing_context)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = [Detection({'class': 'TEST_LABEL', 'score': 0.9, 'box': [10, 10, 10, 10]})] # score < 0.95

    processed_frame = censor_processor.apply_censorship(frame.copy(), detections)

    assert np.array_equal(frame, processed_frame)
    mock_drawing_context.apply_censor_effect.assert_not_called()


def test_apply_censorship_single_censor_single_detection(mock_config_loader, mock_drawing_context):
    """Test that a single censor is applied for a valid single detection."""
    # Configure mock_config_loader to only have 'box' censor
    settings = {
        'enabled': True,
        'min_score': 0.8,
        'censors': [{'type': 'box', 'settings': {'color': (0, 0, 255), 'thickness': -1}}]
    }
    mock_config_loader.get_label_settings.side_effect = lambda label: settings if label == 'TEST_LABEL' else {'enabled': False, 'min_score': 0.0, 'censors': []}

    censor_processor = CensorProcessor(mock_config_loader, mock_drawing_context)

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detection = Detection({'class': 'TEST_LABEL', 'score': 0.9, 'box': [10, 10, 10, 10]})

    censor_processor.apply_censorship(frame.copy(), [detection])

    mock_drawing_context.apply_censor_effect.assert_called_once_with(
        image_np=ANY, # Use image_np=ANY to match keyword arg
        detection=detection, # Pass the Detection object
        censor_type='box',
        settings={'color': (0, 0, 255), 'thickness': -1},
        stats=None,
        metadata=ANY
    )

def test_apply_censorship_multiple_censors_multiple_detections(mock_config_loader, mock_drawing_context):
    """
    Test that censorship is applied for multiple valid detections
    and that multiple censors are applied via the drawing context.
    """
    # Ensure mock_config_loader returns the default settings with two censors
    # The fixture's side_effect function should already handle this.

    censor_processor = CensorProcessor(mock_config_loader, mock_drawing_context)

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detection1 = Detection({'class': 'TEST_LABEL', 'score': 0.9, 'box': [10, 10, 10, 10]})
    detection2 = Detection({'class': 'TEST_LABEL', 'score': 0.9, 'box': [50, 50, 10, 10]})
    detections = [detection1, detection2]

    censor_processor.apply_censorship(frame.copy(), detections)

    assert mock_drawing_context.apply_censor_effect.call_count == 4 # 2 detections * 2 censors

    # Check calls for first detection
    mock_drawing_context.apply_censor_effect.assert_any_call(
        image_np=ANY, detection=detection1, censor_type='box', settings={'color': (0, 0, 255), 'thickness': -1}, stats=None, metadata=ANY
    )
    mock_drawing_context.apply_censor_effect.assert_any_call(
        image_np=ANY, detection=detection1, censor_type='pixelation', settings={'pixel_size': 16}, stats=None, metadata=ANY
    )

    # Check calls for second detection
    mock_drawing_context.apply_censor_effect.assert_any_call(
        image_np=ANY, detection=detection2, censor_type='box', settings={'color': (0, 0, 255), 'thickness': -1}, stats=None, metadata=ANY
    )
    mock_drawing_context.apply_censor_effect.assert_any_call(
        image_np=ANY, detection=detection2, censor_type='pixelation', settings={'pixel_size': 16}, stats=None, metadata=ANY
    )
