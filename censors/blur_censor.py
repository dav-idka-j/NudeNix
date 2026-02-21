import cv2
import numpy as np
from typing import Dict, Any
from detection_types import Detection

class BlurCensor:
    """
    Censor implementation that applies a Gaussian blur to the detected region.
    """
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.kernel_size = settings.get('kernel_size', 15)
        self.sigma = settings.get('sigma', 0)
        self.scale_with_detection_size = settings.get('scale_with_detection_size', True)

        # Ensure kernel size is odd and at least 1
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
        if self.kernel_size < 1:
            self.kernel_size = 1

    def apply(self, image_np: np.ndarray, detection: Detection) -> None:
        x_min, y_min, x_max, y_max = detection.x1(), detection.y1(), detection.x2(), detection.y2()

        # Ensure bounding box coordinates are within image dimensions
        h, w = image_np.shape[:2]
        x_min = max(0, int(x_min))
        y_min = max(0, int(y_min))
        x_max = min(w, int(x_max))
        y_max = min(h, int(y_max))

        if x_max <= x_min or y_max <= y_min:
            return # Invalid or empty bounding box

        # Extract the region of interest
        roi = image_np[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            return

        # Calculate kernel size based on ROI size if requested
        k_size = self.kernel_size
        if self.scale_with_detection_size:
            smaller_dim = min(roi.shape[0], roi.shape[1])
            # Scale kernel_size as a percentage of the smaller dimension
            k_size = int((self.kernel_size / 100.0) * smaller_dim)
            if k_size % 2 == 0:
                k_size += 1
            if k_size < 1:
                k_size = 1

        # Apply Gaussian Blur
        blurred_roi = cv2.GaussianBlur(roi, (k_size, k_size), self.sigma)

        image_np[y_min:y_max, x_min:x_max] = blurred_roi
