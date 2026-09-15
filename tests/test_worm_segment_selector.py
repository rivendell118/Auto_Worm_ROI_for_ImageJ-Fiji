import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from batch_worm_roi import process_image  # noqa: E402
from manual_head_annotation import (  # noqa: E402
    HeadDirection, match_head_directions, save_image_annotations)
from worm_segment_selector import select_instance_segments  # noqa: E402


def straight_worm():
    labels = np.zeros((140, 80), dtype=np.uint16)
    labels[10:130, 30:51] = 1
    return labels


class WormSegmentSelectorTests(unittest.TestCase):
    def test_head_to_tail_fraction_follows_arrow_direction(self):
        labels = straight_worm()
        top_matches, _ = match_head_directions(
            labels, [HeadDirection(40, 45, 40, 4)])
        top, reports = select_instance_segments(labels, top_matches, 0.0, 0.25)
        top_y = np.nonzero(top)[0]
        self.assertLess(float(top_y.mean()), 45.0)
        self.assertEqual(reports[0].start_fraction, 0.0)
        self.assertEqual(reports[0].end_fraction, 0.25)

        bottom_matches, _ = match_head_directions(
            labels, [HeadDirection(40, 95, 40, 136)])
        bottom, _ = select_instance_segments(labels, bottom_matches, 0.0, 0.25)
        bottom_y = np.nonzero(bottom)[0]
        self.assertGreater(float(bottom_y.mean()), 95.0)

    def test_middle_half_is_original_subset_and_preserves_label(self):
        labels = straight_worm()
        matches, _ = match_head_directions(
            labels, [HeadDirection(40, 45, 40, 4)])
        result, reports = select_instance_segments(labels, matches, 0.25, 0.75)
        self.assertEqual(set(np.unique(result)), {0, 1})
        self.assertFalse(np.any((result > 0) & (labels == 0)))
        self.assertGreater(reports[0].selected_area_fraction, 0.40)
        self.assertLess(reports[0].selected_area_fraction, 0.60)

    def test_full_range_reproduces_whole_instance(self):
        labels = straight_worm()
        matches, _ = match_head_directions(
            labels, [HeadDirection(40, 45, 40, 4)])
        result, _ = select_instance_segments(labels, matches, 0.0, 1.0)
        self.assertTrue(np.array_equal(result, labels))

    def test_curved_worm_uses_geodesic_centerline(self):
        image = Image.new("L", (180, 180), 0)
        draw = ImageDraw.Draw(image)
        points = [(35, 15), (60, 45), (105, 55), (135, 90), (115, 130), (75, 160)]
        draw.line(points, fill=1, width=17, joint="curve")
        for x, y in (points[0], points[-1]):
            draw.ellipse((x-8, y-8, x+8, y+8), fill=1)
        labels = np.asarray(image, dtype=np.uint16)
        matches, unmatched = match_head_directions(
            labels, [HeadDirection(52, 38, 31, 10)])
        self.assertEqual(unmatched, 0)
        result, reports = select_instance_segments(labels, matches, 0.50, 1.0)
        self.assertTrue(result.any())
        self.assertFalse(np.any((result > 0) & (labels == 0)))
        self.assertGreater(reports[0].centerline_length_px, 150.0)

    def test_missing_head_arrow_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "one matched head arrow"):
            select_instance_segments(straight_worm(), [], 0.25, 0.75)

    def test_invalid_range_is_rejected(self):
        labels = straight_worm()
        matches, _ = match_head_directions(
            labels, [HeadDirection(40, 45, 40, 4)])
        with self.assertRaisesRegex(ValueError, "0 <= start"):
            select_instance_segments(labels, matches, 0.75, 0.25)

    def test_process_image_exports_selected_segment_and_report(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            image_path = folder / "one.tif"
            output = folder / "out"
            output.mkdir()
            raw = np.full((140, 80), 10, dtype=np.uint8)
            raw[10:130, 30:51] = 100
            Image.fromarray(raw).save(image_path)
            save_image_annotations(
                image_path, (80, 140), [HeadDirection(40, 45, 40, 4)])
            labels = straight_worm()

            def fake_predict(*_args, **_kwargs):
                return labels.copy(), raw.copy(), None

            with patch("batch_worm_roi.predict_raw", side_effect=fake_predict):
                result = process_image(
                    image_path, output, model=None, device=None,
                    standard_count=1, segment_selection=True,
                    segment_start=0.25, segment_end=0.75)
            self.assertEqual(result[1:4], [1, 1, "PASS"])
            self.assertTrue(result[22])
            self.assertEqual(result[23:26], [0.25, 0.75, 1])
            self.assertTrue(Path(result[27]).is_file())
            measurement = (output / "measurements"
                           / "one_measurements.csv").read_text(encoding="utf-8-sig")
            self.assertIn("01_worm", measurement)
            self.assertTrue((output / "other" / "one_QC.png").is_file())


if __name__ == "__main__":
    unittest.main()
