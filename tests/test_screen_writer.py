from fractions import Fraction
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import av
from screen_writer import ScreenWriter

class TestScreenWriter(unittest.TestCase):
    def setUp(self):
        self.mock_context = MagicMock()
        self.mock_reader = MagicMock()
        self.mock_reader.get_fps.return_value = 30.0
        self.mock_reader.get_frame_size.return_value = (1920, 1080)
        self.writer = ScreenWriter(
            reader=self.mock_reader,
            overlay_context=self.mock_context
        )

    @patch('screen_writer.QApplication')
    @patch('screen_writer.OverlayWindow')
    def test_open_success(self, MockWindow, MockApp):
        MockApp.instance.return_value = None
        self.writer.get_video_time_base = MagicMock(return_value=Fraction(1, 30))
        self.assertTrue(self.writer.open())
        self.assertIsNotNone(self.writer.window)
        self.assertIsNotNone(self.writer.app)
        self.writer.window.setGeometry.assert_called_with(0, 0, 1920, 1080)
        self.writer.window.show.assert_called_once()

    def test_write_video_frame_updates_window(self):
        # Setup mocks
        self.writer.window = MagicMock()
        self.writer.app = MagicMock()

        mock_commands = [{'box': (0, 0, 10, 10), 'patch_rgba': np.zeros((10, 10, 4), dtype=np.uint8)}]
        self.mock_context.get_commands.return_value = mock_commands

        # Call write_video_frame
        dummy_frame = MagicMock(spec=av.VideoFrame)
        self.writer.write_video_frame(dummy_frame)

        # Verify interactions
        self.mock_context.get_commands.assert_called_once()
        self.writer.window.update_commands.assert_called_with(mock_commands)
        self.mock_context.clear_commands.assert_called_once()
        self.writer.app.processEvents.assert_called_once()

    def test_close(self):
        mock_window = MagicMock()
        self.writer.window = mock_window
        self.writer.close()
        mock_window.close.assert_called_once()
        self.assertIsNone(self.writer.window)

if __name__ == '__main__':
    unittest.main()
