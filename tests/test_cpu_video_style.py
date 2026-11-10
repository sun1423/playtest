import unittest
import numpy as np
from cpu_video_style import STYLE_MODELS, _square_frame


class CpuVideoStyleTests(unittest.TestCase):
    def test_style_catalog_has_quality_and_fast_modes(self):
        self.assertEqual(STYLE_MODELS["anime_512"]["size"], 512)
        for name in ("candy", "mosaic", "rain_princess", "udnie"):
            self.assertEqual(STYLE_MODELS[name]["size"], 224)

    def test_portrait_letterbox_preserves_aspect_ratio(self):
        import cv2
        frame = np.zeros((1280, 720, 3), dtype=np.uint8)
        canvas, box = _square_frame(cv2, frame, 512)
        self.assertEqual(canvas.shape, (512, 512, 3))
        left, top, width, height, original_w, original_h = box
        self.assertEqual((original_w, original_h), (720, 1280))
        self.assertLess(width, 512)
        self.assertEqual(height, 512)
        self.assertEqual(left + width, 512 - left)

    def test_landscape_letterbox_preserves_aspect_ratio(self):
        import cv2
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        canvas, box = _square_frame(cv2, frame, 512)
        self.assertEqual(canvas.shape, (512, 512, 3))
        left, top, width, height, original_w, original_h = box
        self.assertEqual((original_w, original_h), (1280, 720))
        self.assertEqual(width, 512)
        self.assertLess(height, 512)
        self.assertEqual(top + height, 512 - top)


if __name__ == "__main__":
    unittest.main()
