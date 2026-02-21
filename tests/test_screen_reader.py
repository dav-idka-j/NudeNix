import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image
from input.screen_reader import ScreenCaptureReader

class TestScreenCaptureReader(unittest.TestCase):

    @patch('PIL.ImageGrab.grab')
    def test_open_success(self, mock_grab):
        """Test that open() successfully determines frame size."""
        mock_image = Image.new('RGB', (1920, 1080))
        mock_grab.return_value = mock_image
        reader = ScreenCaptureReader(fps=30)

        result = reader.open()

        self.assertTrue(result)
        self.assertEqual(reader.get_frame_size(), (1920, 1080))
        self.assertIsNotNone(reader._bbox)
        self.assertEqual(reader._bbox, (0, 0, 1920, 1080))
        mock_grab.assert_called_once()

    @patch('PIL.ImageGrab.grab', side_effect=Exception("Screen grab failed"))
    def test_open_failure(self, mock_grab):
        """Test that open() returns False on exception."""
        reader = ScreenCaptureReader()

        result = reader.open()

        self.assertFalse(result)

    def test_get_fps(self):
        """Test that get_fps() returns the configured FPS."""
        reader = ScreenCaptureReader(fps=60)

        self.assertEqual(reader.get_fps(), 60)

    def test_get_duration(self):
        """Test that get_duration() returns infinity."""

        reader = ScreenCaptureReader()

        self.assertEqual(reader.get_duration(), float('inf'))

    def test_audio_methods(self):
        """Test that audio-related methods return None."""

        reader = ScreenCaptureReader()

        self.assertIsNone(reader.get_audio_layout())
        self.assertIsNone(reader.get_audio_rate())

    @patch('PIL.ImageGrab.grab')
    @patch('time.sleep') # Patch sleep to make the test run faster
    def test_read_frame_success(self, mock_sleep, mock_grab):
        """Test that read_frame() returns a valid numpy array."""

        mock_image = Image.new('RGB', (800, 600))
        mock_grab.return_value = mock_image

        reader = ScreenCaptureReader(fps=30)
        reader.open()

        frame, audio, pts = reader.read_frame()

        self.assertIsInstance(frame, np.ndarray)
        self.assertIsNone(audio)
        self.assertIsNotNone(pts)
        self.assertEqual(frame.shape, (600, 800, 3))
        self.assertEqual(frame.dtype, np.uint8)

        # Check if the color format is BGR by checking the first pixel
        # A red PIL image should be blue in BGR
        red_img = Image.new('RGB', (1, 1), color='red')
        mock_grab.return_value = red_img
        frame, _, _ = reader.read_frame()
        # In BGR, red (255, 0, 0) becomes (0, 0, 255)
        np.testing.assert_array_equal(frame[0, 0, :], [0, 0, 255])

        mock_grab.assert_called() # Called in open() and read_frame()
        # We don't assert sleep call count because timing is tricky

    @patch('PIL.ImageGrab.grab', side_effect=Exception("Screen grab failed"))
    @patch('time.sleep')
    def test_read_frame_failure(self, mock_sleep, mock_grab):
        """Test that read_frame() returns None, None on exception."""
        reader = ScreenCaptureReader(fps=30)
        reader.open()

        frame, audio, pts = reader.read_frame()

        self.assertIsNone(frame)
        self.assertIsNone(audio)
        self.assertIsNone(pts)


if __name__ == '__main__':
    unittest.main()
