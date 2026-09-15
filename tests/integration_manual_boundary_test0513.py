"""Regression for 0513-0-1: correct already-counted tail ownership."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inspect_roi_dataset import load_rois, polygon_mask  # noqa: E402
from manual_head_annotation import (  # noqa: E402
    load_image_annotations,
    load_image_boundary_guides,
    split_instances_by_head_directions,
)


def label_map(zip_path):
    result = np.zeros((2048, 2048), dtype=np.uint16)
    for label, roi in enumerate(load_rois(zip_path)[:-1], 1):
        result[polygon_mask(result.shape, roi["points"]) > 0] = label
    return result


def dice(labels, reference, label):
    intersection = int(((labels == label) & (reference == label)).sum())
    return 2.0 * intersection / max(
        int((labels == label).sum() + (reference == label).sum()), 1)


def main():
    issue = WORKSPACE / "issues" / "0913"
    source = WORKSPACE / "test" / "test04" / "0513-0-1.tif"
    automatic = label_map(issue / "0513-0-1_RoiSet.zip")
    reference = label_map(issue / "RoiSet_0513-0-1(人工).zip")
    arrows = load_image_annotations(source, current_size=(2048, 2048))
    guides = load_image_boundary_guides(source, current_size=(2048, 2048))

    combined, combined_reports = split_instances_by_head_directions(
        automatic, arrows, max_instances=10, boundary_guides=guides)
    boundary_only, boundary_reports = split_instances_by_head_directions(
        automatic, [], max_instances=10, boundary_guides=guides)

    assert np.array_equal(combined > 0, automatic > 0)
    assert np.array_equal(boundary_only > 0, automatic > 0)
    assert int(combined.max()) == int(boundary_only.max()) == 10
    assert combined_reports[-1].method == "manual_boundary_refine"
    assert boundary_reports[-1].method == "manual_boundary_refine"
    for result in (combined, boundary_only):
        assert dice(result, reference, 1) > dice(automatic, reference, 1) + 0.04
        assert dice(result, reference, 2) > dice(automatic, reference, 2) + 0.04
    print("MANUAL_BOUNDARY_0513_OK: combined and boundary-only modes both repair the two left tails")


if __name__ == "__main__":
    main()
