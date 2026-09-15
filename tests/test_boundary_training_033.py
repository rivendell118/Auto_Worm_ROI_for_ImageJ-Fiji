import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train_worm_unet import (boundary_issue_weight_map, issue_from_stem,
                             semantic_target)


class BoundaryTraining033Tests(unittest.TestCase):
    def fixture(self):
        labels = np.zeros((80, 100), dtype=np.uint16)
        labels[10:70, 20:44] = 1
        labels[10:70, 48:72] = 2
        return labels, semantic_target(labels, boundary_radius=3)

    def test_issue_marker_is_encoded_in_supplement_stem(self):
        self.assertEqual(issue_from_stem("boundary3__merge__g1__sample"), "merge")
        self.assertEqual(issue_from_stem("boundary3__merge_split__g1__sample"), "merge_split")
        self.assertEqual(issue_from_stem("base__sample"), "normal")

    def test_merge_weight_includes_true_gap_but_not_remote_background(self):
        labels, target = self.fixture()
        weights = boundary_issue_weight_map(
            labels, target, issue="merge", contact_boost=3.0,
            contact_dilation=4)
        self.assertEqual(float(weights[40, 46]), 3.0)
        self.assertEqual(float(weights[2, 2]), 1.0)

    def test_split_weight_emphasizes_only_true_interior(self):
        labels, target = self.fixture()
        weights = boundary_issue_weight_map(
            labels, target, issue="split", continuity_boost=1.75)
        self.assertEqual(float(weights[40, 30]), 1.75)
        self.assertEqual(float(weights[40, 20]), 1.0)
        self.assertEqual(float(weights[2, 2]), 1.0)

    def test_normal_record_is_unweighted(self):
        labels, target = self.fixture()
        weights = boundary_issue_weight_map(
            labels, target, issue="normal", contact_boost=3.0,
            continuity_boost=1.75)
        self.assertTrue(np.array_equal(weights, np.ones_like(weights)))


if __name__ == "__main__":
    unittest.main()
