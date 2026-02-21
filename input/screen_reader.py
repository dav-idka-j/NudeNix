import time
from typing import Tuple, Optional, Dict, Any

import numpy as np
from PIL import Image, ImageGrab
import av

from interfaces.base_frame_reader import BaseFrameReader


class ScreenCaptureReader(BaseFrameReader):
    """
    A frame reader that captures content from the screen.
    """

    def __init__(self, monitor: int = 1, fps: float = 30.0):
        """
        Initializes the screen capture reader.

        Args:
            monitor (int): The monitor number to capture (1-indexed).
            fps (float): The target frames per second.
        """
        self.monitor = monitor
        self.target_fps = fps
        self._frame_size = (0, 0)
        self._bbox = None
        self._last_frame_time = 0
        self._start_time = None

    def open(self) -> bool:
        """
        Opens the screen for capturing. Detects screen dimensions.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            img = ImageGrab.grab()
            self._frame_size = img.size

            self._bbox = (0, 0, self._frame_size[0], self._frame_size[1])
            self._start_time = None # Will be set on first read_frame
            return True
        except Exception:
            return False

    def read_frame(self) -> Tuple[Optional[np.ndarray], Optional[Any], Optional[float]]:
        """
        Reads a frame from the screen.

        Returns:
            A tuple containing the video frame (as a NumPy array), None for audio, and PTS.
        """
        now = time.monotonic()
        time_to_wait = (1 / self.target_fps) - (now - self._last_frame_time)
        if time_to_wait > 0:
            time.sleep(time_to_wait)

        current_time = time.monotonic()
        self._last_frame_time = current_time

        if self._start_time is None:
            self._start_time = current_time

        try:
            pil_image: Image.Image = ImageGrab.grab(bbox=self._bbox)
            # Convert RGB to BGR for OpenCV compatibility
            frame = np.array(pil_image)
            frame = frame[:, :, ::-1].copy() # RGB to BGR

            # Calculate PTS based on elapsed time (in seconds)
            pts_seconds = current_time - self._start_time

            return frame, None, pts_seconds
        except Exception:
            return None, None, None

    def close(self) -> None:
        """
        Closes the reader (no-op for screen capture).
        """
        pass

    def get_frame_size(self) -> Tuple[int, int]:
        """
        Returns the width and height of the captured screen.
        """
        return self._frame_size

    def get_fps(self) -> float:
        """
        Returns the target frames per second.
        """
        return self.target_fps

    def get_duration(self) -> float:
        """
        Returns the duration (infinite for screen capture).
        """
        return float('inf')

    def get_audio_layout(self) -> Optional[Any]:
        """
        Returns the audio layout (None for screen capture).
        """
        return None

    def get_audio_rate(self) -> Optional[int]:
        """
        Returns the audio sample rate (None for screen capture).
        """
        return None

    def get_video_stream(self) -> Optional[av.VideoStream]:
        """
        Returns the underlying PyAV video stream object (None for screen capture).
        """
        return None

    def get_audio_stream(self) -> Optional[av.AudioStream]:
        """
        Returns the underlying PyAV audio stream object (None for screen capture).
        """
        return None
