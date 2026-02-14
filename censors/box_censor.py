import cv2
import numpy as np

class BoxCensor:
    def __init__(self, settings):
        self.thickness = settings.get('thickness', 2)
        self.size_percent = settings.get('size_percent', 100)
        color_dict = settings.get('color', {'r': 0, 'g': 0, 'b': 255}) # Default red
        self.color = (color_dict['b'], color_dict['g'], color_dict['r']) # Convert to BGR for OpenCV
        self.display_label = settings.get('display_label', True)
        self.display_score = settings.get('display_score', True)

    def apply(self, frame_ndarray: np.ndarray, detection: dict):
        """
        Applies a box censor effect to the given frame for a single detection.

        Args:
            frame_ndarray (np.ndarray): The video frame as a NumPy array (RGB), modified in-place.
            detection (dict): A single detection dictionary from NudeDetector.
        """
        x_n, y_n, w_n, h_n = detection['box']
        
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
        h, w, _ = frame_ndarray.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Draw box
        cv2.rectangle(frame_ndarray, (x1, y1), (x2, y2), self.color, self.thickness)

        # Add text if enabled
        if self.display_label or self.display_score:
            label = detection['class']
            score = detection['score']
            text_parts = []
            if self.display_label:
                text_parts.append(label)
            if self.display_score:
                text_parts.append(f"{score:.2f}")
            text = ": ".join(text_parts)

            text_x = x1
            text_y = y1 - 10
            if text_y < 0:
                text_y = y1 + 20

            cv2.putText(frame_ndarray, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

