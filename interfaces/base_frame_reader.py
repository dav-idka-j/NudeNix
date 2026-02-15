from abc import ABC, abstractmethod
from typing import Tuple, Union, Optional, Dict, Any

import numpy as np
import av # For av.AudioFrame and av.AudioLayout type hints

class BaseFrameReader(ABC):
    """
    Abstract Base Class for frame readers.
    Defines the interface for any class that provides a stream of video frames.
    """

    @abstractmethod
    def open(self) -> bool:
        """
        Opens the frame source.

        Returns:
            bool: True if the source was successfully opened, False otherwise.
        """
        pass

    @abstractmethod
    def read_frame(self) -> Tuple[Optional[np.ndarray], Optional[av.AudioFrame]]:
        """
        Reads the next video frame and an optional audio frame from the source.

        Returns:
            Tuple[Optional[np.ndarray], Optional[av.AudioFrame]]:
                A tuple containing the video frame (as a NumPy array, or None)
                and an optional audio frame (or None).
        """
        pass


    @abstractmethod
    def close(self) -> None:
        """
        Closes the frame source and releases any resources.
        """
        pass


    @abstractmethod
    def get_frame_size(self) -> Tuple[int, int]:
        """
        Returns the width and height of the video frames.

        Returns:
            Tuple[int, int]: A tuple (width, height) of the video frames.
        """
        pass

    @abstractmethod
    def get_fps(self) -> float:
        """
        Returns the frames per second of the video stream.

        Returns:
            float: The frame rate.
        """
        pass

    @abstractmethod
    def get_duration(self) -> float:
        """
        Returns the duration of the video stream in seconds.

        Returns:
            float: The duration in seconds.
        """
        pass

    @abstractmethod
    def get_audio_layout(self) -> Optional[av.AudioLayout]:
        """
        Returns the audio layout of the audio stream, if available.

        Returns:
            Optional[av.AudioLayout]: The audio layout object, or None if no audio stream.
        """
        pass

    @abstractmethod
    def get_audio_rate(self) -> Optional[int]:
        """
        Returns the sample rate of the audio stream, if available.

        Returns:
            Optional[int]: The audio sample rate, or None if no audio stream.
        """
        pass
