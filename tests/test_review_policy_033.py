import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from batch_worm_roi import make_overlay, qc_status_for_count  # noqa: E402
from manual_head_annotation import enhanced_tiff_rgb  # noqa: E402


class ReviewPolicy033Tests(unittest.TestCase):
    def test_only_exact_count_passes(self):
        self.assertEqual(qc_status_for_count(10, 10), "PASS")
        self.assertEqual(qc_status_for_count(9, 10), "REVIEW_COUNT_MISMATCH")
        self.assertEqual(qc_status_for_count(11, 10), "REVIEW_COUNT_MISMATCH")

    def test_tiff_monitor_uses_equal_grayscale_channels(self):
        array = np.arange(16 * 12, dtype=np.uint16).reshape(12, 16) * 127
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            path = Path(folder) / "gray16.tif"
            Image.fromarray(array).save(path)
            displayed = np.asarray(enhanced_tiff_rgb(path))
        self.assertEqual(displayed.shape, (12, 16, 3))
        self.assertTrue(np.array_equal(displayed[..., 0], displayed[..., 1]))
        self.assertTrue(np.array_equal(displayed[..., 1], displayed[..., 2]))
        self.assertGreater(int(displayed.max()), int(displayed.min()))

    def test_ai_attention_draws_yellow_qc_border_without_review_status(self):
        normalized = np.full((40, 50), 80, dtype=np.uint8)
        instances = np.zeros((40, 50), dtype=np.uint16)
        background = np.zeros((40, 50), dtype=bool)
        rendered = np.asarray(make_overlay(
            normalized, instances, background, "PASS", attention=True))
        self.assertEqual(tuple(rendered[1, 1]), (255, 210, 32))
        self.assertEqual(qc_status_for_count(10, 10), "PASS")


if __name__ == "__main__":
    unittest.main()
