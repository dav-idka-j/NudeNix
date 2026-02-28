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

    def apply(self, image_np: np.ndarray, detection: Detection) -> None:
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

        # Convert ROI to grayscale
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Convert back to BGR so it can be merged/blended with original
        gray_roi_bgr = cv2.cvtColor(gray_roi, cv2.COLOR_GRAY2BGR)

        if self.intensity >= 1.0:
            image_np[y_min:y_max, x_min:x_max] = gray_roi_bgr
        elif self.intensity > 0.0:
            # Blend original and grayscale based on intensity
            blended = cv2.addWeighted(gray_roi_bgr, self.intensity, roi, 1.0 - self.intensity, 0)
            image_np[y_min:y_max, x_min:x_max] = blended

    def get_type(self) -> str:
        return "grayscale"
