import pytest
import numpy as np
import cv2
from unittest.mock import patch

from censor_drawing_contexts.in_place_drawing_context import InPlaceDrawingContext
from censors_registry import CensorsRegistry
from detection_types import Detection


class TestBoxCensor:
    def test_box_censor(self):
        """Test InPlaceDrawingContext with box censor."""
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        original_frame = frame.copy()

        # x_min, y_min, x_max, y_max
        bbox_coords = (10, 10, 20, 20)
        detection_obj = Detection({'class': 'test', 'score': 1.0, 'box': [bbox_coords[0], bbox_coords[1], bbox_coords[2] - bbox_coords[0], bbox_coords[3] - bbox_coords[1]]})

        settings = {
            'color': {'r': 255, 'g': 0, 'b': 0},
            'thickness': -1,
            'display_label': False,
            'display_score': False
        }

        censors_registry = CensorsRegistry()
        drawing_context = InPlaceDrawingContext(censors_registry)
        drawing_context.apply_censor_effect(frame, detection_obj, 'box', settings)

        assert np.all(frame[10:21, 10:21] == (0, 0, 255))
        assert not np.any(frame[0:10, :, :] != original_frame[0:10, :, :])
        assert not np.any(frame[21:, :, :] != original_frame[21:, :, :])
        assert not np.any(frame[:, 0:10, :] != original_frame[:, 0:10, :])
        assert not np.any(frame[:, 21:, :] != original_frame[:, 21:, :])

class TestPixelationCensor:
    def test_pixelation_censor_basic(self):
        """Test InPlaceDrawingContext with pixelation censor with default settings."""
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        for i in range(50):
            frame[i, :, 0] = i * 5 # Blue gradient
            frame[:, i, 1] = i * 3 # Green gradient
        original_frame = frame.copy()
        detection_obj = Detection({'class': 'test', 'score': 1.0, 'box': [10, 10, 30, 30]})
        settings = {'strength': 10, 'scale_with_detection_size': False}

        censors_registry = CensorsRegistry()
        drawing_context = InPlaceDrawingContext(censors_registry)
        drawing_context.apply_censor_effect(frame, detection_obj, 'pixelation', settings)

        assert not np.array_equal(frame[10:40, 10:40], original_frame[10:40, 10:40])

    def test_pixelation_censor_grayscale(self):
        """Test the grayscale option of the pixelation censor."""
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        for i in range(50):
            frame[i, :, 0] = i * 5
            frame[:, i, 1] = i * 3
        original_frame = frame.copy()
        detection_obj = Detection({'class': 'test', 'score': 1.0, 'box': [10, 10, 30, 30]})
        settings = {'strength': 5, 'grayscale': True, 'scale_with_detection_size': False}

        censors_registry = CensorsRegistry()
        drawing_context = InPlaceDrawingContext(censors_registry)
        drawing_context.apply_censor_effect(frame, detection_obj, 'pixelation', settings)

        roi = frame[10:40, 10:40]
        # Check if the region is now grayscale (R, G, and B channels are the same)
        assert np.all(roi[:,:,0] == roi[:,:,1]) and np.all(roi[:,:,1] == roi[:,:,2])
        assert not np.array_equal(roi, original_frame[10:40, 10:40])

    def test_pixelation_censor_scale_with_size(self):
        """Test the scale_with_detection_size option of the pixelation censor."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detection_obj = Detection({'class': 'test', 'score': 1.0, 'box': [10, 10, 80, 80]})
        # Strength is 50%, smaller dim is 80, so pixel_size should be 40
        settings = {'strength': 50, 'scale_with_detection_size': True}

        censors_registry = CensorsRegistry()
        drawing_context = InPlaceDrawingContext(censors_registry)
        drawing_context.apply_censor_effect(frame, detection_obj, 'pixelation', settings)

        roi = frame[10:90, 10:90]
        # With pixel size of 40 on an 80x80 ROI, there should be 4 blocks (2x2)
        # All pixels in a block should be the same color
        assert np.all(roi[0:40, 0:40] == roi[0, 0])
        assert np.all(roi[40:80, 0:40] == roi[40, 0])
        assert np.all(roi[0:40, 40:80] == roi[0, 40])
        assert np.all(roi[40:80, 40:80] == roi[40, 40])

    @patch('cv2.resize')
    def test_pixelation_censor_sampling_mode(self, mock_resize):
        """Test that sampling_mode correctly selects the OpenCV interpolation flag."""
        # Make resize return a dummy array of the correct shape to avoid crashing
        mock_resize.side_effect = lambda img, dsize, interpolation: np.zeros((dsize[1], dsize[0], 3), dtype=np.uint8)

        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        detection_obj = Detection({'class': 'test', 'score': 1.0, 'box': [10, 10, 30, 30]})

        # Test 'accurate'
        settings_accurate = {'strength': 10, 'sampling_mode': 'accurate', 'scale_with_detection_size': False}
        registry_accurate = CensorsRegistry()
        context_accurate = InPlaceDrawingContext(registry_accurate)
        context_accurate.apply_censor_effect(frame, detection_obj, 'pixelation', settings_accurate)

        # Check that cv2.resize was called with INTER_AREA for downscaling
        downscale_call = mock_resize.call_args_list[0]
        assert downscale_call.kwargs['interpolation'] == cv2.INTER_AREA

        mock_resize.reset_mock()

        # Re-apply side effect for the next part of the test
        mock_resize.side_effect = lambda img, dsize, interpolation: np.zeros((dsize[1], dsize[0], 3), dtype=np.uint8)

        # Test 'fast'
        settings_fast = {'strength': 10, 'sampling_mode': 'fast', 'scale_with_detection_size': False}
        registry_fast = CensorsRegistry()
        context_fast = InPlaceDrawingContext(registry_fast)
        context_fast.apply_censor_effect(frame, detection_obj, 'pixelation', settings_fast)

        # Check that cv2.resize was called with INTER_NEAREST for downscaling
        downscale_call_fast = mock_resize.call_args_list[0]
        assert downscale_call_fast.kwargs['interpolation'] == cv2.INTER_NEAREST

class TestGenericCensorBehavior:
    def test_unsupported_censor_type(self):
        """Test InPlaceDrawingContext with an unsupported censor type."""
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        original_frame = frame.copy()
        detection_obj = Detection({'class': 'test', 'score': 1.0, 'box': [10, 10, 30, 30]})
        settings = {}

        censors_registry = CensorsRegistry()
        drawing_context = InPlaceDrawingContext(censors_registry)

        with pytest.raises(ValueError, match="No censor registered for type: unsupported_type"):
            drawing_context.apply_censor_effect(frame, detection_obj, 'unsupported_type', settings)

        assert np.array_equal(frame, original_frame)
