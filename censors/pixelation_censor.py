import cv2
import numpy as np

class PixelationCensor:
    def __init__(self, settings):
        self.strength = settings.get('strength', 10) # Block size in pixels
        self.type = settings.get('type', 'mosaic') # For now, only 'mosaic'
        self.scale_with_detection_size = settings.get('scale_with_detection_size', True)
        self.grayscale = settings.get('grayscale', False)
        self.sampling_mode = settings.get('sampling_mode', 'accurate') # 'fast' or 'accurate'
        self.shape = settings.get('shape', 'square') # For now, only 'square'

        if self.type != 'mosaic':
            print(f"Warning: Pixelation type '{self.type}' not supported. Falling back to 'mosaic'.")
        if self.shape != 'square':
            print(f"Warning: Pixelation shape '{self.shape}' not supported. Falling back to 'square'.")

    def apply(self, frame_ndarray: np.ndarray, detection: dict):
        """
        Applies a pixelation censor effect to the given frame for a single detection.

        Args:
            frame_ndarray (np.ndarray): The video frame as a NumPy array (RGB), modified in-place.
            detection (dict): A single detection dictionary from NudeDetector.
        """
        x_n, y_n, w_n, h_n = detection['box']

        # Ensure coordinates are integers
        x1, y1, w, h = int(x_n), int(y_n), int(w_n), int(h_n)

        # Ensure bounding box is valid and within frame dimensions
        frame_h, frame_w, _ = frame_ndarray.shape
        x1 = max(0, x1)
        y1 = max(0, y1)
        w = min(w, frame_w - x1)
        h = min(h, frame_h - y1)

        if w <= 0 or h <= 0:
            return # Nothing to censor if box is invalid

        # Calculate effective strength (pixel block size)
        pixel_block_size = self.strength
        if self.scale_with_detection_size:
            # Scale strength relative to the smallest dimension of the detection
            # e.g., strength as a percentage of width/height
            min_dim = min(w, h)
            if min_dim > 0:
                pixel_block_size = max(1, int(min_dim * (self.strength / 100.0))) # Interpret strength as percentage

        # Ensure pixel_block_size is at least 1
        pixel_block_size = max(1, pixel_block_size)

        # Get the region of interest (ROI)
        roi = frame_ndarray[y1:y1+h, x1:x1+w]

        if roi.shape[0] <= 0 or roi.shape[1] <= 0:
            return # Invalid ROI

        # Apply grayscale if enabled
        if self.grayscale:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # Convert back to 3 channels if pixelation needs it
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

        # Calculate new dimensions for downscaling
        new_w = w // pixel_block_size
        new_h = h // pixel_block_size

        if new_w <= 0 or new_h <= 0:
            return # Cannot pixelate if downscaled dimensions are zero or negative

        # Downscale
        # INTER_AREA is generally good for downsampling, INTER_NEAREST for fast blocks on upscale
        interpolation_down = cv2.INTER_AREA # 'accurate' mosaic
        if self.sampling_mode == 'fast':
            interpolation_down = cv2.INTER_NEAREST # For 'fast' (center pixel, less accurate average)

        roi_small = cv2.resize(roi, (new_w, new_h), interpolation=interpolation_down)

        # Upscale to create pixelation effect
        pixelated_roi = cv2.resize(roi_small, (w, h), interpolation=cv2.INTER_NEAREST)

        # Place pixelated ROI back into the original frame
        frame_ndarray[y1:y1+h, x1:x1+w] = pixelated_roi
