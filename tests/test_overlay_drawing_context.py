import unittest
import numpy as np
from unittest.mock import MagicMock
from censor_drawing_contexts.overlay_drawing_context import OverlayDrawingContext
from detection_types import Detection

class TestOverlayDrawingContext(unittest.TestCase):
    def setUp(self):
        self.registry = MagicMock()
        self.ctx = OverlayDrawingContext(self.registry)

    def test_apply_censor_effect_stores_command(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        detection = Detection({
            'box': [10, 20, 30, 40], # x, y, w, h
            'class': 'test_label',
            'score': 0.9
        })

        mock_censor = MagicMock()
        self.registry.get_censor_instance.return_value = mock_censor

        self.ctx.apply_censor_effect(img, detection, 'box', {})

        commands = self.ctx.get_commands()
        self.assertEqual(len(commands), 1)

        cmd = commands[0]
        # x1_pad = max(0, 10 - 40) = 0
        # y1_pad = max(0, 20 - 40) = 0
        # x2_pad = min(100, 10 + 30 + 40) = 80
        # y2_pad = min(100, 20 + 40 + 40) = 100
        self.assertEqual(cmd['box'], (0, 0, 80, 100))
        self.assertIsInstance(cmd['patch_rgba'], np.ndarray)
        self.assertEqual(cmd['patch_rgba'].shape, (100, 80, 4))

        mock_censor.apply.assert_called_once()
        # The detection passed to apply should be relative to the padded ROI
        # roi_detection.box = [10 - 0, 20 - 0, 30, 40] = [10, 20, 30, 40]
        call_args = mock_censor.apply.call_args
        passed_roi = call_args[0][0]
        passed_detection = call_args[0][1]

        self.assertEqual(passed_detection.x(), 10)
        self.assertEqual(passed_detection.y(), 20)
        self.assertEqual(passed_detection.width(), 30)
        self.assertEqual(passed_detection.height(), 40)

    def test_clear_commands(self):
        self.ctx.commands = [{'dummy': 'data'}]
        self.ctx.clear_commands()
        self.assertEqual(len(self.ctx.get_commands()), 0)

if __name__ == '__main__':
    unittest.main()
