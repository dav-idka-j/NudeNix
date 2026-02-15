from abc import ABC, abstractmethod
from typing import Optional, Union

import av

class BaseFrameWriter(ABC):
    """
    Abstract Base Class for frame writers.
    Defines the interface for any class that writes video and/or audio frames
    to an output destination.
    """

    @abstractmethod
    def open(self) -> bool:
        """
        Opens the output destination.

        Returns:
            bool: True if the destination was successfully opened, False otherwise.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Closes the output destination and releases any resources.
        """
        pass

    @abstractmethod
    def write_video_frame(self, frame: av.VideoFrame) -> None:
        """
        Writes a single video frame to the output.

        Args:
            frame (av.VideoFrame): The video frame to write.
        """
        pass

    @abstractmethod
    def write_audio_packet(self, packet: av.Packet) -> None:
        """
        Writes an audio packet to the output.

        Args:
            packet (av.Packet): The audio packet to write.
        """
        pass

    @abstractmethod
    def flush_encoders(self) -> None:
        """
        Flushes any pending frames/packets from the encoders to the output.
        Should be called before closing the writer.
        """
        pass
