"""N2-2 真实 TIFF 的人工分界曲线端到端验证。"""

import csv
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from batch_worm_roi import OTHER_DIRNAME, run_gui_batch  # noqa: E402
from inspect_roi_dataset import load_rois, polygon_mask  # noqa: E402
from manual_head_annotation import (  # noqa: E402
    BoundaryGuide,
    load_image_annotations,
    save_image_annotations,
)


def label_map(zip_path):
    result = np.zeros((2048, 2048), dtype=np.uint16)
    for label, roi in enumerate(load_rois(zip_path)[:-1], 1):
        result[polygon_mask((2048, 2048), roi["points"]) > 0] = label
    return result


def main():
    source = WORKSPACE / "test" / "test06" / "N2-2.tif"
    validation = ROOT / "validation_outputs" / "manual_boundary_test06"
    input_dir, output_dir = validation / "input", validation / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = input_dir / source.name
    shutil.copy2(source, target)

    reference = label_map(
        ROOT.parent / "0910 0.3.2 CUDA" / "validation_outputs" /
        "test06_manual_split_fix" / "N2-2_RoiSet.zip")
    points = []
    for y in range(reference.shape[0]):
        left = np.flatnonzero(reference[y] == 9)
        right = np.flatnonzero(reference[y] == 10)
        if len(left) and len(right):
            points.append((0.5 * (float(left.max()) + float(right.min())), float(y)))
    points = points[::12] + ([points[-1]] if points[-1] != points[::12][-1] else [])
    guide = BoundaryGuide(tuple(points))
    arrows = load_image_annotations(source, current_size=(2048, 2048))
    save_image_annotations(target, (2048, 2048), arrows, [guide])

    result = run_gui_batch(
        str(input_dir), str(output_dir),
        str(ROOT / "models" / "0.2.0" / "worm.pt"),
        str(ROOT / "models" / "0.2.0" / "tip.pt"),
        standard_count=10, allowed_count_min=10, allowed_count_max=10,
        ignore_filename_count=True, low_clarity_split=True,
        manual_head_annotation=True, on_status=lambda text: print(text, flush=True))[0]
    assert result[1] == 10 and result[15] is True, result
    assert result[3] == "PASS", result
    assert result[17] >= 1 and result[19] == 1, result
    with (output_dir / OTHER_DIRNAME / "N2-2_manual_split_qc.csv").open(
            "r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[-1]["method"] == "manual_boundary_refine", rows
    assert all(row["method"] == "manual_head_coarse" for row in rows[:-1]), rows
    assert (output_dir / OTHER_DIRNAME / "N2-2_QC.png").is_file()
    print("MANUAL_BOUNDARY_TEST06_OK: head coarse split then boundary refinement; 10/10 heads matched")


if __name__ == "__main__":
    main()
