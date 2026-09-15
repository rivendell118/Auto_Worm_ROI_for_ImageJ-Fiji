"""CUDA end-to-end regression for independent/manual-combined boundary handling."""

import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from batch_worm_roi import OTHER_DIRNAME, run_gui_batch  # noqa: E402
from inspect_roi_dataset import load_rois, polygon_mask  # noqa: E402


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
    source_dir = WORKSPACE / "test" / "test04"
    issue_dir = WORKSPACE / "issues" / "0913"
    reference = label_map(issue_dir / "RoiSet_0513-0-1(人工).zip")
    original = label_map(issue_dir / "0513-0-1_RoiSet.zip")
    with tempfile.TemporaryDirectory(dir=ROOT / "tests") as temporary:
        temporary = Path(temporary)
        input_dir, output_dir = temporary / "input", temporary / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        shutil.copy2(source_dir / "0513-0-1.tif", input_dir / "0513-0-1.tif")
        shutil.copy2(source_dir / "_manual_head_annotations.json",
                     input_dir / "_manual_head_annotations.json")
        result = run_gui_batch(
            str(input_dir), str(output_dir),
            str(ROOT / "models" / "0.2.0" / "worm.pt"),
            str(ROOT / "models" / "0.2.0" / "tip.pt"),
            standard_count=10, allowed_count_min=10, allowed_count_max=10,
            ignore_filename_count=True, low_clarity_split=True,
            manual_head_annotation=True, measurement_backend="imagej",
            on_status=lambda text: print(text, flush=True))[0]
        assert result[1] == 10 and result[3] == "PASS", result
        with (output_dir / OTHER_DIRNAME / "0513-0-1_manual_split_qc.csv").open(
                "r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        assert rows and rows[-1]["method"] == "manual_boundary_refine", rows
        prediction = label_map(output_dir / OTHER_DIRNAME / "0513-0-1_RoiSet.zip")
        assert dice(prediction, reference, 1) > dice(original, reference, 1) + 0.04
        assert dice(prediction, reference, 2) > dice(original, reference, 2) + 0.04
        # ROI polygon serialization can move a one-pixel contour band; the
        # in-memory unit tests above enforce exact foreground preservation.
        union_difference = float(np.logical_xor(
            prediction > 0, original > 0).sum()) / max(int((original > 0).sum()), 1)
        print("serialized_union_difference", union_difference)
        assert union_difference < 0.01

        # Remove every arrow and exercise the same fixture through the complete
        # batch pipeline. Boundary refinement must remain independently usable.
        annotation_path = input_dir / "_manual_head_annotations.json"
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        annotation["images"]["0513-0-1.tif"]["arrows"] = []
        annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
        boundary_output = temporary / "boundary_only_output"
        boundary_output.mkdir()
        boundary_result = run_gui_batch(
            str(input_dir), str(boundary_output),
            str(ROOT / "models" / "0.2.0" / "worm.pt"),
            str(ROOT / "models" / "0.2.0" / "tip.pt"),
            standard_count=10, allowed_count_min=10, allowed_count_max=10,
            ignore_filename_count=True, low_clarity_split=True,
            manual_head_annotation=True, measurement_backend="imagej",
            on_status=lambda text: print(text, flush=True))[0]
        assert boundary_result[1] == 10 and boundary_result[14] == 0
        boundary_prediction = label_map(boundary_output / OTHER_DIRNAME / "0513-0-1_RoiSet.zip")
        assert dice(boundary_prediction, reference, 1) > dice(original, reference, 1) + 0.04
        assert dice(boundary_prediction, reference, 2) > dice(original, reference, 2) + 0.04
    print("MANUAL_BOUNDARY_0513_E2E_OK")


if __name__ == "__main__":
    main()
