import sys
import numpy as np
import cv2
from typing import Dict, List, Any, TYPE_CHECKING
from interfaces.base_censor_drawing_context import BaseCensorDrawingContext
from detection_types import Detection

if TYPE_CHECKING:
    from censors_registry import CensorsRegistry

class OverlayDrawingContext(BaseCensorDrawingContext):
    """
    An implementation of BaseCensorDrawingContext that pre-renders censored patches
    with alpha masking, so only the actual censorship (and not the background) is visible.
    """
    def __init__(self, censors_registry: 'CensorsRegistry'):
        self.censors_registry = censors_registry
        self.commands: List[Dict[str, Any]] = []

    def apply_censor_effect(self, image_np: np.ndarray, detection: 'Detection', censor_type: str, settings: Dict, stats=None) -> None:
        """
        Extracts the ROI, applies the censor, calculates the difference to create an alpha mask,
        and stores the resulting RGBA patch.
        """
        x1, y1, x2, y2 = detection.bounding_box()
        h, w = image_np.shape[:2]

        # Use padding to capture text/labels that might be drawn outside the box
        padding = 40
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(w, x2 + padding)
        y2_pad = min(h, y2 + padding)

        if x2_pad <= x1_pad or y2_pad <= y1_pad:
            return

        roi_original = image_np[y1_pad:y2_pad, x1_pad:x2_pad].copy()
        roi_censored = roi_original.copy()

        # Apply the censor
        label = detection.label()
        try:
            censor_instance = self.censors_registry.get_censor_instance(label, censor_type, settings)

            # Temporary detection relative to the padded ROI
            roi_detection = Detection({
                'box': [x1 - x1_pad, y1 - y1_pad, x2 - x1, y2 - y1],
                'class': label,
                'score': detection.score()
            })

            if stats:
                with stats.time_censor(censor_type):
                    censor_instance.apply(roi_censored, roi_detection)
            else:
                censor_instance.apply(roi_censored, roi_detection)

            # Create Alpha Mask
            # Any pixel that changed in any channel (B, G, or R) is considered "censor content"
            diff = np.any(roi_original != roi_censored, axis=-1).astype(np.uint8) * 255

            # Dilate the mask slightly to handle anti-aliasing edges (especially for text)
            kernel = np.ones((3, 3), np.uint8)
            diff = cv2.dilate(diff, kernel, iterations=1)

            # Construct RGBA patch
            # Merge the BGR censored image with the calculated Alpha mask
            b, g, r = cv2.split(roi_censored)
            roi_rgba = cv2.merge([b, g, r, diff])

            self.commands.append({
                'box': (x1_pad, y1_pad, x2_pad - x1_pad, y2_pad - y1_pad),
                'patch_rgba': roi_rgba
            })
        except ValueError as e:
            print(f"Warning: {e}", file=sys.stderr)

    def get_commands(self) -> List[Dict[str, Any]]:
        return self.commands

    def clear_commands(self) -> None:
        self.commands = []
