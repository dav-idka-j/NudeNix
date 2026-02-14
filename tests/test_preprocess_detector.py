import pytest
from unittest.mock import MagicMock, patch, call
from preprocess_detector import ImageHashDetector
from detector import NsfwDetector
import imagehash
from PIL import Image
import numpy as np
import cv2

# Mock NsfwDetector for ImageHashDetector
@pytest.fixture
def mock_nsfw_detector():
    mock = MagicMock(spec=NsfwDetector)
    # Configure the mock detect method to return some dummy detections
    mock.detect.return_value = [
        {'box': [10, 10, 20, 20], 'score': 0.9, 'label': 'SFW'},
        {'box': [30, 30, 40, 40], 'score': 0.8, 'label': 'NSFW'}
    ]
    return mock

# Mock imagehash algorithms, but don't set side_effect here
@pytest.fixture
def mock_imagehash_algorithms_individual():
    """Fixture for individual mock imagehash functions."""
    mock_ahash = MagicMock(spec=imagehash.average_hash)
    mock_phash = MagicMock(spec=imagehash.phash)
    mock_dhash = MagicMock(spec=imagehash.dhash)
    mock_whash = MagicMock(spec=imagehash.whash)

    return {
        'average_hash': mock_ahash,
        'phash': mock_phash,
        'dhash': mock_dhash,
        'whash': mock_whash
    }

# Patch imagehash functions for all tests using this fixture
@pytest.fixture(autouse=True)
def patch_imagehash_functions(mock_imagehash_algorithms_individual):
    """Patch the imagehash functions globally for tests in this module."""
    with (
        patch('preprocess_detector.imagehash.average_hash', new=mock_imagehash_algorithms_individual['average_hash']),
        patch('preprocess_detector.imagehash.phash', new=mock_imagehash_algorithms_individual['phash']),
        patch('preprocess_detector.imagehash.dhash', new=mock_imagehash_algorithms_individual['dhash']),
        patch('preprocess_detector.imagehash.whash', new=mock_imagehash_algorithms_individual['whash'])
    ):
        yield mock_imagehash_algorithms_individual

# Mock numpy array for image
@pytest.fixture
def dummy_image():
    return np.zeros((100, 100, 3), dtype=np.uint8)

# Define some consistent hash values for testing
HASH_INITIAL = imagehash.hex_to_hash("0000000000000000")
HASH_SIMILAR = imagehash.hex_to_hash("0000000000000000") # Same hash for 0 diff
HASH_DIFFERENT = imagehash.hex_to_hash("FFFFFFFFFFFFFFFF")

def test_imagehash_detector_init(mock_nsfw_detector):
    """Test ImageHashDetector initialization with default and custom configs."""
    # Default config
    config = {'enabled': True}
    detector = ImageHashDetector(mock_nsfw_detector, config)
    assert detector.nsfw_detector == mock_nsfw_detector
    assert detector.enabled is True
    assert detector.algorithm_name == 'average_hash'
    assert detector.hash_threshold == 11
    assert detector.max_skip_frames == 5
    assert detector.debug_overlay is False

    # Custom config
    custom_config = {
        'enabled': True,
        'algorithm': 'phash',
        'hash_threshold': 5,
        'max_skip_frames': 10,
        'debug_overlay': True
    }
    detector_custom = ImageHashDetector(mock_nsfw_detector, custom_config)
    assert detector_custom.algorithm_name == 'phash'
    assert detector_custom.hash_threshold == 5
    assert detector_custom.max_skip_frames == 10
    assert detector_custom.debug_overlay is True

def test_imagehash_detector_disabled(mock_nsfw_detector, dummy_image):
    """Test detect method when image hashing is disabled."""
    config = {'enabled': False}
    detector = ImageHashDetector(mock_nsfw_detector, config)

    result = detector.detect(dummy_image)
    mock_nsfw_detector.detect.assert_called_once_with(dummy_image)
    assert result == mock_nsfw_detector.detect.return_value

def test_imagehash_detector_first_frame_runs_detection(mock_nsfw_detector, patch_imagehash_functions, dummy_image):
    """Test first frame always runs detection and caches result."""
    config = {'enabled': True, 'algorithm': 'average_hash'}
    detector = ImageHashDetector(mock_nsfw_detector, config)

    # Set return value for the first hash calculation
    patch_imagehash_functions['average_hash'].return_value = HASH_INITIAL

    result = detector.detect(dummy_image)
    mock_nsfw_detector.detect.assert_called_once_with(dummy_image)
    assert result == mock_nsfw_detector.detect.return_value
    assert detector.last_known_hash == HASH_INITIAL
    assert detector.last_known_detections == mock_nsfw_detector.detect.return_value
    assert detector.current_frame_skip_count == 0

def test_imagehash_detector_max_skip_frames_reached_runs_detection(
    mock_nsfw_detector, patch_imagehash_functions, dummy_image):
    """Test detection runs when max_skip_frames is reached, even if hash is similar."""
    config = {'enabled': True, 'max_skip_frames': 1, 'hash_threshold': 1} # Threshold 1, so diff 0 is similar
    detector = ImageHashDetector(mock_nsfw_detector, config)

    # Frame 1: Initial detection (HASH_INITIAL)
    patch_imagehash_functions['average_hash'].return_value = HASH_INITIAL
    detector.detect(dummy_image)
    mock_nsfw_detector.detect.reset_mock() # Reset mock call count
    assert detector.current_frame_skip_count == 0

    # Frame 2: Hash is similar (HASH_SIMILAR, diff 0). Should skip.
    patch_imagehash_functions['average_hash'].return_value = HASH_SIMILAR
    detector.detect(dummy_image)
    mock_nsfw_detector.detect.assert_not_called()
    assert detector.current_frame_skip_count == 1 # Skipped once

    # Frame 3: Hash is similar (HASH_SIMILAR, diff 0). max_skip_frames (1) reached. Should run detection.
    patch_imagehash_functions['average_hash'].return_value = HASH_SIMILAR
    detector.detect(dummy_image)
    mock_nsfw_detector.detect.assert_called_once_with(dummy_image)
    assert detector.current_frame_skip_count == 0 # Reset after detection

def test_imagehash_detector_similar_hash_skips_detection(
    mock_nsfw_detector, patch_imagehash_functions, dummy_image):
    """Test detection is skipped when hashes are similar."""
    config = {'enabled': True, 'hash_threshold': 1} # Threshold 1, so diff 0 is similar
    detector = ImageHashDetector(mock_nsfw_detector, config)

    # First frame (runs detection)
    patch_imagehash_functions['average_hash'].return_value = HASH_INITIAL
    first_detections = detector.detect(dummy_image)
    mock_nsfw_detector.detect.reset_mock()
    assert detector.current_frame_skip_count == 0


    # Second frame (similar hash, should skip)
    patch_imagehash_functions['average_hash'].return_value = HASH_SIMILAR
    second_detections = detector.detect(dummy_image)

    mock_nsfw_detector.detect.assert_not_called()
    assert second_detections == first_detections
    assert detector.current_frame_skip_count == 1

def test_imagehash_detector_different_hash_runs_detection(
    mock_nsfw_detector, patch_imagehash_functions, dummy_image):
    """Test detection runs when hashes are different."""
    config = {'enabled': True, 'hash_threshold': 1} # Threshold 1, so diff 0 is similar
    detector = ImageHashDetector(mock_nsfw_detector, config)

    # First frame (runs detection)
    patch_imagehash_functions['average_hash'].return_value = HASH_INITIAL
    detector.detect(dummy_image)
    mock_nsfw_detector.detect.reset_mock()
    assert detector.current_frame_skip_count == 0


    # Second frame (different hash, should run detection)
    patch_imagehash_functions['average_hash'].return_value = HASH_DIFFERENT
    second_detections = detector.detect(dummy_image)

    mock_nsfw_detector.detect.assert_called_once_with(dummy_image)
    assert second_detections == mock_nsfw_detector.detect.return_value
    assert detector.current_frame_skip_count == 0

@patch('preprocess_detector.cv2.circle')
def test_imagehash_detector_debug_overlay_red_on_detect(
    mock_cv2_circle, mock_nsfw_detector, patch_imagehash_functions, dummy_image):
    """Test red debug overlay is drawn when detection runs."""
    config = {'enabled': True, 'debug_overlay': True, 'hash_threshold': 1}
    detector = ImageHashDetector(mock_nsfw_detector, config)

    # First frame (always runs detection)
    patch_imagehash_functions['average_hash'].return_value = HASH_INITIAL
    detector.detect(dummy_image)
    mock_cv2_circle.assert_called_with(dummy_image, (10, 10), 5, (0, 0, 255), -1) # Blue circle (RGB)
    mock_cv2_circle.reset_mock()

    # Different hash (runs detection)
    patch_imagehash_functions['average_hash'].return_value = HASH_DIFFERENT
    detector.detect(dummy_image)
    mock_cv2_circle.assert_called_with(dummy_image, (10, 10), 5, (255, 0, 0), -1) # Red circle (RGB)

@patch('preprocess_detector.cv2.circle')
def test_imagehash_detector_debug_overlay_green_on_skip(
    mock_cv2_circle, mock_nsfw_detector, patch_imagehash_functions, dummy_image):
    """Test green debug overlay is drawn when detection is skipped."""
    config = {'enabled': True, 'hash_threshold': 1, 'debug_overlay': True} # Threshold 1, so diff 0 is similar
    detector = ImageHashDetector(mock_nsfw_detector, config)

    # First frame (runs detection)
    patch_imagehash_functions['average_hash'].return_value = HASH_INITIAL
    detector.detect(dummy_image)
    mock_cv2_circle.reset_mock() # Clear calls from first detection

    # Similar hash (skips detection)
    patch_imagehash_functions['average_hash'].return_value = HASH_SIMILAR
    detector.detect(dummy_image)
    mock_cv2_circle.assert_called_with(dummy_image, (10, 10), 5, (0, 255, 0), -1) # Green circle

def test_imagehash_detector_no_debug_overlay_when_disabled(
    mock_nsfw_detector, patch_imagehash_functions, dummy_image):
    """Test no debug overlay is drawn when debug_overlay is False."""
    config = {'enabled': True, 'debug_overlay': False, 'hash_threshold': 1} # Threshold 1, so diff 0 is similar
    detector = ImageHashDetector(mock_nsfw_detector, config)

    with patch('preprocess_detector.cv2.circle') as mock_cv2_circle:
        # First frame
        patch_imagehash_functions['average_hash'].return_value = HASH_INITIAL
        detector.detect(dummy_image)
        # Second frame, similar hash
        patch_imagehash_functions['average_hash'].return_value = HASH_SIMILAR
        detector.detect(dummy_image)
        mock_cv2_circle.assert_not_called()
