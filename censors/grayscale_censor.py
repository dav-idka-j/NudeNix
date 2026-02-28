import cv2
import numpy as np
from typing import Dict, Any
from detection_types import Detection

class GrayscaleCensor:
    """
    Censor implementation that desaturates (grayscales) the detected region.
    """
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.intensity = settings.get('intensity', 1.0) # 1.0 = full grayscale, 0.0 = no change
        self.adaptive = settings.get('adaptive', False)

    def apply(self, image_np: np.ndarray, detection: Detection, metadata: Any = None) -> None:
        x_min, y_min, x_max, y_max = detection.x1(), detection.y1(), detection.x2(), detection.y2()

        h, w = image_np.shape[:2]
        x_min = max(0, int(x_min))
        y_min = max(0, int(y_min))
        x_max = min(w, int(x_max))
        y_max = min(h, int(y_max))

        if x_max <= x_min or y_max <= y_min:
            return

        roi = image_np[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            return

        # Calculate effective intensity
        actual_intensity = self.intensity
        if self.adaptive and metadata:
            # Scale intensity between 0 and self.intensity based on confidence ratio
            actual_intensity = self.intensity * metadata.confidence_ratio

        if actual_intensity <= 0:
            return

        # Convert ROI to grayscale
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_roi_bgr = cv2.cvtColor(gray_roi, cv2.COLOR_GRAY2BGR)

        if actual_intensity >= 1.0:
            image_np[y_min:y_max, x_min:x_max] = gray_roi_bgr
        else:
            # Blend original and grayscale based on intensity
            blended = cv2.addWeighted(gray_roi_bgr, actual_intensity, roi, 1.0 - actual_intensity, 0)
            image_np[y_min:y_max, x_min:x_max] = blended

    def get_type(self) -> str:
        return "grayscale"
