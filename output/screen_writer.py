import sys
import os
from typing import Optional, Any, Tuple, List, Dict
from fractions import Fraction
import av
import numpy as np
import cv2
from interfaces.base_frame_writer import BaseFrameWriter
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QImage, QPixmap, QPen, QColor

class OverlayWindow(QMainWindow):
    """
    A transparent, click-through window for displaying censorship patches.
    """
    def __init__(self):
        super().__init__()
        # Set window flags for transparency and click-through
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint |
            Qt.FramelessWindowHint |
            Qt.WindowTransparentForInput |
            Qt.Tool  # Don't show in taskbar
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground, True)

        self.commands: List[Dict[str, Any]] = []

    def update_commands(self, commands: List[Dict[str, Any]]):
        """
        Updates the list of patches to draw and triggers a repaint.
        """
        self.commands = commands
        self.update()

    def _clear_overlay(self, painter):
        # Set composition mode to Source to replace everything with transparency
        painter.setCompositionMode(QPainter.CompositionMode_Source)
        painter.fillRect(self.rect(), Qt.transparent)
        # Switch back to SourceOver for drawing patches on top of transparency
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)

    def _draw_border(self, painter):
        painter.setPen(QPen(QColor(255, 0, 0, 255), 2))
        painter.drawRect(0, 0, self.width() - 1, self.height() - 1)

    def _draw_patches(self, painter):
        for cmd in self.commands:
            x, y, w, h = cmd['box']
            patch_rgba = cmd['patch_rgba']  # np.ndarray (BGRA)

            # Convert BGRA to RGBA for QImage (PyQt uses RGB order)
            # OpenCV uses BGRA by default if we merge B,G,R,A
            patch_rgba_rgb = cv2.cvtColor(patch_rgba, cv2.COLOR_BGRA2RGBA)

            height, width, channel = patch_rgba_rgb.shape
            bytes_per_line = 4 * width

            # Create QImage from the NumPy array (Format_RGBA8888 for 4 channels)
            q_img = QImage(patch_rgba_rgb.data, width, height, bytes_per_line, QImage.Format_RGBA8888)

            # QPixmap.fromImage makes a copy of the data, which is safe for the paint thread
            pixmap = QPixmap.fromImage(q_img)
            painter.drawPixmap(x, y, pixmap)

    def paintEvent(self, event):
        """
        Renders the censorship patches onto the transparent window.
        """
        painter = QPainter(self)
        self._clear_overlay(painter)
        self._draw_border(painter)
        self._draw_patches(painter)
        painter.end()

class ScreenWriter(BaseFrameWriter):
    """
    A writer that manages a PyQt-based transparent overlay for real-time censorship.
    """
    def __init__(self, *args, **kwargs):
        # We now prefer taking the reader instance to get dynamic values at open() time
        self.reader = kwargs.get('reader')

        # Fallback values for compatibility
        self._reader_fps = kwargs.get('reader_fps', 30.0)
        self._reader_frame_size = kwargs.get('reader_frame_size', (1920, 1080))
        self._reader_audio_layout = kwargs.get('reader_audio_layout')
        self._reader_audio_rate = kwargs.get('reader_audio_rate')

        self.overlay_context = kwargs.get('overlay_context')

        self.app = None
        self.window = None

    def open(self) -> bool:
        """
        Initializes the PyQt application and the overlay window.
        """

        # Update dimensions from reader if available
        if self.reader:
            self._reader_frame_size = self.reader.get_frame_size()
            self._reader_fps = self.reader.get_fps()

        if not QApplication.instance():
            self.app = QApplication(sys.argv)
        else:
            self.app = QApplication.instance()

        self.window = OverlayWindow()
        w, h = self._reader_frame_size
        self.window.setGeometry(0, 0, w, h)
        self.window.show()

        self.app.processEvents()
        print(f"ScreenWriter: Overlay window opened at {w}x{h}")
        return True

    def write_video_frame(self, frame: av.VideoFrame):

        # Get commands from the overlay context
        commands = self.overlay_context.get_commands()

        # Update the window with new patches
        # Pass a list copy to ensure thread safety (though current is sync)
        self.window.update_commands(list(commands))

        # Clear commands in the context for the next frame
        self.overlay_context.clear_commands()

        # Process events to keep the UI responsive and trigger the repaint
        if self.app:
            self.app.processEvents()

    def write_audio_packet(self, packet: av.Packet):
        pass

    def flush_encoders(self):
        pass

    def close(self):
        if self.window:
            self.window.close()
            self.window = None
            print("ScreenWriter: Overlay window closed.")

    def get_codec_context(self) -> Any:
        return None

    def get_stream(self) -> Any:
        return None

    def get_fps(self) -> float:
        return self._reader_fps

    def get_frame_size(self) -> Tuple[int, int]:
        return self._reader_frame_size

    def get_audio_layout(self) -> Optional[av.AudioLayout]:
        return self._reader_audio_layout

    def get_audio_rate(self) -> Optional[int]:
        return self._reader_audio_rate

    def get_video_time_base(self) -> Optional[Fraction]:
        """Returns the time base of the output video stream."""
        if self._reader_fps > 0:
            return Fraction(1, int(self._reader_fps))
        return None
