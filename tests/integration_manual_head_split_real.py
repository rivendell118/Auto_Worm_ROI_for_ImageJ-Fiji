"""真实高清粗结果中，以人工头部箭头拆开边界粘连实例。"""

import csv
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from batch_worm_roi import OTHER_DIRNAME, run_gui_batch  # noqa: E402
from evaluate_worm_unet import pair_instances  # noqa: E402
from inspect_roi_dataset import load_rois, polygon_mask  # noqa: E402


def label_map(zip_path):
    result = np.zeros((2048, 2048), dtype=np.uint16)
    for label, roi in enumerate(load_rois(zip_path)[:-1], 1):
        result[polygon_mask((2048, 2048), roi["points"]) > 0] = label
    return result


def main():
    input_dir = ROOT / "validation_outputs" / "manual_head_integration" / "input"
    output_dir = ROOT / "validation_outputs" / "manual_head_split_high" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = run_gui_batch(
        str(input_dir), str(output_dir),
        str(ROOT / "models" / "0.1.0" / "worm.pt"),
        str(ROOT / "models" / "0.1.0" / "tip.pt"),
        standard_count=10, allowed_count_min=10, allowed_count_max=10,
        ignore_filename_count=True, low_clarity_split=False,
        manual_head_annotation=True, on_status=lambda text: print(text, flush=True))
    result = results[0]
    print("RESULT", result)
    report = output_dir / OTHER_DIRNAME / "0513-10-1_manual_split_qc.csv"
    with report.open("r", newline="", encoding="utf-8-sig") as handle:
        split_rows = list(csv.DictReader(handle))
    head_report = output_dir / OTHER_DIRNAME / "0513-10-1_head_annotations.csv"
    with head_report.open("r", newline="", encoding="utf-8-sig") as handle:
        head_rows = list(csv.DictReader(handle))
    assert result[1] == 10, result
    assert result[3] == "PASS", result
    assert len(split_rows) == 2, split_rows
    assert all(row["status"] == "REVIEW_MANUAL_HEAD_SPLIT" for row in split_rows)
    assert len(head_rows) == 10 and all(row["status"] == "MATCHED" for row in head_rows)
    assert result[15] is True and result[17] == 2, result
    reference = label_map(
        ROOT / "validation_outputs" / "test04_gap_final" / "0513-10-1_RoiSet.zip")
    prediction = label_map(output_dir / OTHER_DIRNAME / "0513-10-1_RoiSet.zip")
    _pairs, dices = pair_instances(reference, prediction)
    mean_dice = float(np.mean(dices))
    assert len(dices) == 10 and mean_dice > 0.75, dices
    print("MANUAL_HEAD_SPLIT_REAL_OK: 8 coarse instances -> 10 worms; mean Dice %.4f" % mean_dice)


if __name__ == "__main__":
    main()
