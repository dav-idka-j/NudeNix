import cv2
import numpy as np
from typing import Tuple, Dict, Any

from detection_types import Detection # Import Detection class

class PixelationCensor:
    """
    Censor implementation that pixelates the detected region with various options.
    """
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.strength = settings.get('strength', 10)
        self.scale_with_detection_size = settings.get('scale_with_detection_size', True)
        self.grayscale = settings.get('grayscale', False) # TODO: Move greyscale to its own censor

        sampling_mode = settings.get('sampling_mode', 'accurate')
        if sampling_mode == 'fast':
            self.interpolation_down = cv2.INTER_NEAREST
        else: # 'accurate'
            self.interpolation_down = cv2.INTER_AREA

        # For upscaling, INTER_NEAREST preserves the blocky pixelated effect
        self.interpolation_up = cv2.INTER_NEAREST

        if self.strength <= 0:
            self.strength = 1 # Prevent invalid values

    def apply(self, image_np: np.ndarray, detection: Detection) -> None:
        x_min, y_min, x_max, y_max = detection.x1(), detection.y1(), detection.x2(), detection.y2()

        # Ensure bounding box coordinates are within image dimensions
        h, w = image_np.shape[:2]
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)

        if x_max <= x_min or y_max <= y_min:
            return # Invalid or empty bounding box

        # Extract the region of interest
        roi = image_np[y_min:y_max, x_min:x_max]
        if roi.shape[0] == 0 or roi.shape[1] == 0:
            return

        # Calculate pixel size
        if self.scale_with_detection_size:
            # Strength is a percentage of the smaller dimension
            smaller_dim = min(roi.shape[0], roi.shape[1])
            pixel_size = int((self.strength / 100.0) * smaller_dim)
        else:
            # Strength is an absolute pixel size
            pixel_size = self.strength

        if pixel_size <= 0:
            pixel_size = 1

        # Handle grayscale
        if self.grayscale:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # Revert back to 3 channels for merging
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

        # Determine the size of the small, pixelated ROI
        target_w = max(1, roi.shape[1] // pixel_size)
        target_h = max(1, roi.shape[0] // pixel_size)

        # Downscale and then upscale to achieve pixelation
        small_roi = cv2.resize(roi, (target_w, target_h), interpolation=self.interpolation_down)
        pixelated_roi = cv2.resize(small_roi, (roi.shape[1], roi.shape[0]), interpolation=self.interpolation_up)

        # Place the pixelated ROI back into the original image
        image_np[y_min:y_max, x_min:x_max] = pixelated_roi

    def get_type(self) -> str:
        return "pixelation"
