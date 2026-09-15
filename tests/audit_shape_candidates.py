from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_shape_refiner import _prediction_path, _region_metrics  # noqa: E402
from evaluate_worm_unet import pair_instances, predict_semantic  # noqa: E402
from train_worm_unet import IMAGE_SIZE, WormUNet  # noqa: E402
from worm_shape_refiner import (  # noqa: E402
    _candidate_is_safe,
    _centerline_candidate,
    _limit_additions,
    _local_morphology_candidate,
    _local_roughness,
    _probability_evidence,
    _reasons,
    _smooth_contour_candidate,
    measure_shape,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("predictions")
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--limit", type=int, default=4)
    args = parser.parse_args()
    dataset, predictions = Path(args.dataset), Path(args.predictions)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    image_size = int(checkpoint.get("image_size", IMAGE_SIZE))
    model = WormUNet(base=int(checkpoint.get("base", 16))).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    rows = []
    paths = sorted((dataset / "masks").glob("*_masks.png"))[:args.limit]
    for reference_path in paths:
        stem = reference_path.name[:-len("_masks.png")]
        reference = np.asarray(Image.open(reference_path), dtype=np.uint16)
        coarse = np.asarray(Image.open(_prediction_path(predictions, stem)), dtype=np.uint16)
        probabilities = predict_semantic(
            model, dataset / "images" / f"{stem}.png", device, image_size)
        probability = np.asarray(Image.fromarray(
            probabilities[1:].sum(axis=0).astype(np.float32), mode="F").resize(
                (reference.shape[1], reference.shape[0]), Image.Resampling.BILINEAR),
            dtype=np.float32)
        pairs, _ = pair_instances(coarse, reference)
        reference_for_label = {pred: ref for ref, pred, _ in pairs}
        for label in [int(value) for value in np.unique(coarse) if value]:
            original = coarse == label
            before_shape = measure_shape(original)
            reasons = _reasons(before_shape)
            if not reasons or label not in reference_for_label:
                continue
            options = [
                ("local_morph", _local_morphology_candidate(original)),
                ("curved_centerline", _centerline_candidate(original)),
                ("contour_weak", _smooth_contour_candidate(original, 0.0045)),
                ("contour_medium", _smooth_contour_candidate(original, 0.0070)),
                ("contour_strong", _smooth_contour_candidate(original, 0.0100)),
            ]
            gold = reference == reference_for_label[label]
            before_gold = _region_metrics(original, gold)
            for kind, option in options:
                if option is None:
                    continue
                option &= (coarse == 0) | (coarse == label)
                option = _limit_additions(original, option, before_shape.major_length, probability)
                is_local_cleanup = kind.startswith("local_")
                safe, area_change, centroid_shift, iou = _candidate_is_safe(
                    original, option, before_shape, probability,
                    local_cleanup=is_local_cleanup)
                after_shape = measure_shape(option)
                after_gold = _region_metrics(option, gold)
                evidence = _probability_evidence(original, option, probability)
                quality_gain = ((before_shape.quality - after_shape.quality) /
                                max(before_shape.quality, 1e-6))
                local_gain = ((_local_roughness(before_shape) - _local_roughness(after_shape)) /
                              max(_local_roughness(before_shape), 1e-6))
                score = (local_gain + 0.08 * evidence - 0.75 * area_change
                         - 3.0 * centroid_shift) if is_local_cleanup else (
                             quality_gain + 0.25 * evidence - 0.5 * area_change
                             - 2.0 * centroid_shift)
                rows.append({
                    "image": stem, "label": label, "reasons": "|".join(reasons),
                    "candidate": kind, "safe": safe, "quality_gain": quality_gain,
                    "local_gain": local_gain,
                    "probability_evidence": evidence, "score": score, "iou_to_coarse": iou,
                    "area_change": area_change, "centroid_shift": centroid_shift,
                    "delta_dice": after_gold["dice"] - before_gold["dice"],
                    "delta_boundary_f1": after_gold["boundary_f1"] - before_gold["boundary_f1"],
                    "delta_hd95": before_gold["hd95"] - after_gold["hd95"],
                    "delta_area_error": before_gold["area_error_fraction"] - after_gold["area_error_fraction"],
                })
        print(stem, flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Candidates:", len(rows))
    print("Output:", output)


if __name__ == "__main__":
    main()
