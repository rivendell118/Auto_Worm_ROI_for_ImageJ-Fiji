# 历史探查脚本，不是测试。2026-09-14 由 test_shape_refiner_032.py 改名而来：它只注册
# 0 个测试用例，名字却匹配 test_*，会被 `unittest discover` 连带导入并执行模块级代码。
# 改名后 discovery 不再导入它。仍按脚本单独运行：
# <构建 python> tests\shape_refiner_032_probe.py
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worm_shape_refiner import (  # noqa: E402
    _candidate_is_safe,
    _centerline_candidate,
    _local_morphology_candidate,
    _local_roughness,
    _probability_evidence,
    measure_shape,
    refine_instance_shapes,
)


def curved_worm(shape=(420, 420), offset_x=0):
    mask = np.zeros(shape, dtype=np.uint8)
    points = []
    for y in range(45, 375):
        x = int(round(170 + offset_x + 60 * np.sin((y - 45) / 92.0)))
        radius = max(3, int(round(12 * np.sin(np.pi * (y - 44) / 332.0) ** 0.35)))
        cv2.circle(mask, (x, y), radius, 1, -1)
        points.append((x, y))
    return mask.astype(bool)


def worm_with_local_defects(shape=(420, 420)):
    clean = np.zeros(shape, dtype=np.uint8)
    cv2.ellipse(clean, (210, 210), (125, 16), 12, 0, 360, 1, -1)
    damaged = clean.copy()
    # A short outward spur and inward notch on opposite body edges.
    cv2.circle(damaged, (214, 190), 5, 1, -1)
    cv2.circle(damaged, (207, 226), 4, 0, -1)
    return clean.astype(bool), damaged.astype(bool)


def main():
    original = curved_worm()
    candidate = _centerline_candidate(original)
    assert candidate is not None and candidate.any()
    safe, area_change, centroid_shift, iou = _candidate_is_safe(
        original, candidate, measure_shape(original))
    assert safe, (area_change, centroid_shift, iou)
    assert area_change <= 0.10 and centroid_shift <= 0.025 and iou >= 0.84

    clean, damaged = worm_with_local_defects()
    local_candidates = [
        _local_morphology_candidate(damaged, strength)
        for strength in (0.20, 0.32, 0.42)
    ]
    local_candidates = [value for value in local_candidates if value is not None]
    assert local_candidates
    before_defect = measure_shape(damaged)
    best_local = min(local_candidates, key=lambda value: _local_roughness(measure_shape(value)))
    local_safe, local_area, local_centroid, local_iou = _candidate_is_safe(
        damaged, best_local, before_defect, local_cleanup=True)
    damaged_boundary = damaged ^ ndimage.binary_erosion(damaged)
    repaired_boundary = best_local ^ ndimage.binary_erosion(best_local)
    boundary_shifts = np.concatenate((
        ndimage.distance_transform_edt(~repaired_boundary)[damaged_boundary],
        ndimage.distance_transform_edt(~damaged_boundary)[repaired_boundary],
    ))
    damaged_error = int(np.logical_xor(damaged, clean).sum())
    repaired_error = int(np.logical_xor(best_local, clean).sum())
    assert local_safe, (local_area, local_centroid, local_iou,
                        np.percentile(boundary_shifts, [95, 99]))
    assert repaired_error < damaged_error, (
        damaged_error, repaired_error,
        [int(np.logical_xor(value, clean).sum()) for value in local_candidates],
        [int(np.logical_xor(value, damaged).sum()) for value in local_candidates],
        [_local_roughness(measure_shape(value)) for value in local_candidates],
    )
    assert _local_roughness(measure_shape(best_local)) < _local_roughness(before_defect)

    damaged_instances = np.zeros(damaged.shape, dtype=np.uint16)
    damaged_instances[damaged] = 1
    clean_probability = np.where(clean, 0.93, 0.04).astype(np.float32)
    repaired_instances, defect_reports = refine_instance_shapes(
        damaged_instances, clean_probability)
    assert defect_reports[0].repaired, defect_reports[0]
    assert defect_reports[0].candidate_kind == "local_morph"
    pipeline_error = int(np.logical_xor(repaired_instances == 1, clean).sum())
    assert pipeline_error < damaged_error, (
        pipeline_error, defect_reports[0],
        [(_candidate_is_safe(damaged, value, before_defect, clean_probability,
                             local_cleanup=True),
          _probability_evidence(damaged, value, clean_probability),
          measure_shape(value)) for value in local_candidates],
    )
    assert _local_roughness(measure_shape(repaired_instances == 1)) < _local_roughness(before_defect)

    instances = np.zeros(original.shape, dtype=np.uint16)
    instances[original] = 1
    second = curved_worm(offset_x=145)
    instances[second] = 2
    probability = np.where(instances > 0, 0.92, 0.03).astype(np.float32)
    refined, reports = refine_instance_shapes(instances, probability)
    assert int(refined.max()) == 2
    assert not np.logical_and(refined == 1, refined == 2).any()
    assert [report.label for report in reports] == [1, 2]
    for report in reports:
        if report.repaired:
            assert report.status == "REVIEW_SHAPE_REFINED"
            assert report.needs_review
            assert report.candidate_kind != "original"
            assert report.candidate_iou >= 0.84
    print("SHAPE_REFINER_032_SYNTHETIC_OK")


if __name__ == "__main__":
    main()
