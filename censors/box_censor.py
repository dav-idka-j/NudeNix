import cv2
import numpy as np
from typing import Tuple, Dict, Any

from detection_types import Detection

class BoxCensor:
    """
    Censor implementation that draws a colored box over the detected region,
    with configurable size, color, thickness, and optional label/score display.
    """
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.thickness = settings.get('thickness', 2)
        self.size_percent = settings.get('size_percent', 100)

        color_dict = settings.get('color', {'r': 0, 'g': 0, 'b': 255}) # Default blue
        self.color = (color_dict['b'], color_dict['g'], color_dict['r'])

        self.display_label = settings.get('display_label', True)
        self.display_score = settings.get('display_score', True)
        self.adaptive = settings.get('adaptive', False)

    def apply(self, image_np: np.ndarray, detection: Detection, metadata: Any = None) -> None:
        x_n, y_n, w_n, h_n = detection.x(), detection.y(), detection.width(), detection.height()

        # Adjust box size based on size_percent
        if self.size_percent != 100:
            scale = self.size_percent / 100.0
            new_w = w_n * scale
            new_h = h_n * scale
            x_n = x_n + (w_n - new_w) / 2
            y_n = y_n + (h_n - new_h) / 2
            w_n, h_n = new_w, new_h

        x1, y1 = int(x_n), int(y_n)
        x2, y2 = int(x_n + w_n), int(y_n + h_n)

        # Ensure coordinates are within frame bounds
        h, w, _ = image_np.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return # Invalid or empty bounding box after adjustments

        # Draw box
        cv2.rectangle(image_np, (x1, y1), (x2, y2), self.color, self.thickness)

        # Add text if enabled
        if self.display_label or self.display_score:
            label_text = detection.label()
            score_text = f"{detection.score():.2f}"

            text_parts = []
            if self.display_label:
                text_parts.append(label_text)
            if self.display_score:
                text_parts.append(score_text)
            text = ": ".join(text_parts)

            text_x = x1
            text_y = y1 - 10
            if text_y < 10: # Ensure text is not drawn off-screen upwards
                text_y = y1 + 20 # Draw below if not enough space above

            cv2.putText(image_np, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def get_type(self) -> str:
        return "box"
