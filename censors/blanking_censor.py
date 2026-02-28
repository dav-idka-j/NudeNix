import cv2
import numpy as np
from typing import Dict, Any
from detection_types import Detection

class BlankingCensor:
    """
    Censor implementation that blanks the entire frame with a solid color.
    """
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        color_dict = settings.get('color', {'r': 0, 'g': 0, 'b': 0})
        # OpenCV uses BGR
        self.color = (color_dict.get('b', 0), color_dict.get('g', 0), color_dict.get('r', 0))
        self.adaptive = settings.get('adaptive', False)

    def apply(self, image_np: np.ndarray, detection: Detection, metadata: Any = None) -> None:
        # Fill the entire image with the specified color
        image_np[:] = self.color

    def get_type(self) -> str:
        return "blanking"
