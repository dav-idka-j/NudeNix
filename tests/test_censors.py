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

class TestBlurCensor:
    def test_blur_censor_basic(self):
        """Test the blur censor with basic settings."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Create a checkerboard pattern to make blurring noticeable
        frame[0:50, 0:50, :] = 255
        frame[50:100, 50:100, :] = 255
        original_frame = frame.copy()

        detection_obj = Detection({'class': 'test', 'score': 1.0, 'box': [20, 20, 60, 60]})
        settings = {'kernel_size': 15, 'scale_with_detection_size': False}

        censors_registry = CensorsRegistry()
        drawing_context = InPlaceDrawingContext(censors_registry)
        drawing_context.apply_censor_effect(frame, detection_obj, 'blur', settings)

        # The ROI should be changed
        roi = frame[20:80, 20:80]
        assert not np.array_equal(roi, original_frame[20:80, 20:80])
        # The edges of the pattern should be smoothed
        assert not np.all(np.logical_or(roi == 0, roi == 255))

    def test_blur_censor_scale_with_size(self):
        """Test the blur censor with kernel scaling enabled."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[25:75, 25:75, :] = 255
        original_frame = frame.copy()

        detection_obj = Detection({'class': 'test', 'score': 1.0, 'box': [10, 10, 80, 80]})
        # kernel_size 20% of 80 = 16, odd is 17
        settings = {'kernel_size': 20, 'scale_with_detection_size': True}

        censors_registry = CensorsRegistry()
        drawing_context = InPlaceDrawingContext(censors_registry)

        with patch('cv2.GaussianBlur') as mock_blur:
            # Mock GaussianBlur to return a placeholder and verify ksize
            mock_blur.side_effect = lambda src, ksize, sigma: np.zeros(src.shape, dtype=src.dtype)
            drawing_context.apply_censor_effect(frame, detection_obj, 'blur', settings)

            # Check ksize in call: (17, 17)
            args, kwargs = mock_blur.call_args
            assert args[1] == (17, 17)

class TestGrayscaleCensor:
    def test_grayscale_censor_full_intensity(self):
        """Test grayscale censor with 100% intensity."""
        # Create a colorful image
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, 0] = 255 # Blue

        detection_obj = Detection({'class': 'test', 'score': 1.0, 'box': [25, 25, 50, 50]})
        settings = {'intensity': 1.0}

        censors_registry = CensorsRegistry()
        drawing_context = InPlaceDrawingContext(censors_registry)
        drawing_context.apply_censor_effect(frame, detection_obj, 'grayscale', settings)

        roi = frame[25:75, 25:75]
        # In grayscale, B=G=R
        assert np.all(roi[:,:,0] == roi[:,:,1])
        assert np.all(roi[:,:,1] == roi[:,:,2])
        # Verify it's not all zeros or unchanged
        assert not np.all(roi == 255)
        assert np.any(roi > 0)

    def test_grayscale_censor_partial_intensity(self):
        """Test grayscale censor with 50% intensity (blending)."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, 0] = 255 # Solid Blue

        detection_obj = Detection({'class': 'test', 'score': 1.0, 'box': [0, 0, 100, 100]})
        settings = {'intensity': 0.5}

        censors_registry = CensorsRegistry()
        drawing_context = InPlaceDrawingContext(censors_registry)
        drawing_context.apply_censor_effect(frame, detection_obj, 'grayscale', settings)

        # Blue channel should be reduced but not as much as full grayscale
        assert 30 < frame[0,0,0] < 250
        # G and R channels should be increased from 0
        assert frame[0,0,1] > 0
        assert frame[0,0,2] > 0

class TestBlankingCensor:
    def test_blanking_censor(self):
        """Test that the blanking censor fills the entire frame."""
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 255 # White frame

        # Detection is only in a small part
        detection_obj = Detection({'class': 'test', 'score': 1.0, 'box': [40, 40, 20, 20]})
        settings = {'color': {'r': 255, 'g': 0, 'b': 0}} # Red in BGR is (0, 0, 255)

        censors_registry = CensorsRegistry()
        drawing_context = InPlaceDrawingContext(censors_registry)
        drawing_context.apply_censor_effect(frame, detection_obj, 'blanking', settings)

        # Check corners (outside the detection box) to ensure full frame blanking
        assert np.all(frame[0, 0] == [0, 0, 255])
        assert np.all(frame[99, 99] == [0, 0, 255])
        # Check inside detection box
        assert np.all(frame[50, 50] == [0, 0, 255])

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
