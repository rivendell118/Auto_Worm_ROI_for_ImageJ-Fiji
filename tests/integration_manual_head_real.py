"""一张真实 TIFF 的人工头部标注完整链路验证（按需手动运行）。"""

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
from manual_head_annotation import HeadDirection, save_image_annotations  # noqa: E402


def main():
    source = WORKSPACE / "test" / "test04" / "0513-10-1.tif"
    roi_zip = ROOT / "validation_outputs" / "test04_gap_final" / "0513-10-1_RoiSet.zip"
    validation = ROOT / "validation_outputs" / "manual_head_integration"
    input_dir = validation / "input"
    output_dir = validation / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = input_dir / source.name
    shutil.copy2(source, target)

    arrows = []
    for roi in load_rois(roi_zip)[:-1]:
        mask = polygon_mask((2048, 2048), roi["points"]).astype(bool)
        yy, xx = np.nonzero(mask)
        y_min, y_max = int(yy.min()), int(yy.max())
        shaft_y = int(round(y_min + 0.28 * (y_max - y_min)))
        shaft_band = np.nonzero(mask[max(0, shaft_y - 3):shaft_y + 4])[1]
        head_band = np.nonzero(mask[y_min:min(y_min + 16, mask.shape[0])])[1]
        arrows.append(HeadDirection(
            float(np.median(shaft_band)), float(shaft_y),
            float(np.median(head_band)), float(max(0, y_min - 18))))
    save_image_annotations(target, (2048, 2048), arrows)

    results = run_gui_batch(
        str(input_dir), str(output_dir),
        str(ROOT / "models" / "0.2.0" / "worm.pt"),
        str(ROOT / "models" / "0.2.0" / "tip.pt"),
        standard_count=10, allowed_count_min=10, allowed_count_max=10,
        ignore_filename_count=True, low_clarity_split=True,
        manual_head_annotation=True, on_status=lambda text: print(text, flush=True))
    result = results[0]
    assert result[1] == 10, result
    assert result[3] == "PASS", result
    assert result[13] is True and result[14] == 10 and result[15] is True, result
    assert result[17] == 0, result
    report = Path(result[16])
    with report.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10 and all(row["status"] == "MATCHED" for row in rows)
    assert (output_dir / OTHER_DIRNAME / "0513-10-1_QC.png").is_file()
    print("MANUAL_HEAD_REAL_INTEGRATION_OK: 10/10 arrows matched")


if __name__ == "__main__":
    main()
