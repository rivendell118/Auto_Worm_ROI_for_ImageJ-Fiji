from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
import torch

from evaluate_worm_unet import pair_instances, predict_semantic
from train_worm_unet import IMAGE_SIZE, WormUNet
from worm_shape_refiner import (
    _longest_path_from_skeleton,
    _morphological_skeleton,
    refine_instance_shapes,
)


def _edge(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask)


def _surface_metrics(prediction: np.ndarray, reference: np.ndarray,
                     tolerance: float = 2.0) -> tuple[float, float, float]:
    pred_edge, ref_edge = _edge(prediction), _edge(reference)
    if not pred_edge.any() or not ref_edge.any():
        return 0.0, float("inf"), float("inf")
    distance_to_ref = ndimage.distance_transform_edt(~ref_edge)
    distance_to_pred = ndimage.distance_transform_edt(~pred_edge)
    pred_distance = distance_to_ref[pred_edge]
    ref_distance = distance_to_pred[ref_edge]
    precision = float(np.mean(pred_distance <= tolerance))
    recall = float(np.mean(ref_distance <= tolerance))
    boundary_f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
    distances = np.concatenate((pred_distance, ref_distance))
    return (boundary_f1, float(np.mean(distances)),
            float(np.percentile(distances, 95)))


def _skeleton_metrics(prediction: np.ndarray,
                      reference: np.ndarray) -> tuple[float, float]:
    # Skeleton metrics are normalized and can be evaluated at a bounded scale.
    # This keeps a 2048-pixel validation set practical without changing the
    # production refiner, which always operates at the original resolution.
    scale = min(1.0, 768.0 / max(prediction.shape))
    if scale < 1.0:
        size = (max(1, int(round(prediction.shape[1] * scale))),
                max(1, int(round(prediction.shape[0] * scale))))
        prediction = np.asarray(Image.fromarray(prediction).resize(
            size, Image.Resampling.NEAREST), dtype=bool)
        reference = np.asarray(Image.fromarray(reference).resize(
            size, Image.Resampling.NEAREST), dtype=bool)
    pred_skeleton = _morphological_skeleton(prediction)
    ref_skeleton = _morphological_skeleton(reference)
    if not pred_skeleton.any() or not ref_skeleton.any():
        return 0.0, float("nan")
    topology_precision = float(reference[pred_skeleton].mean())
    topology_sensitivity = float(prediction[ref_skeleton].mean())
    cldice = 2.0 * topology_precision * topology_sensitivity / max(
        topology_precision + topology_sensitivity, 1e-9)
    pred_path = _longest_path_from_skeleton(pred_skeleton)
    ref_path = _longest_path_from_skeleton(ref_skeleton)
    if pred_path is None or ref_path is None:
        return float(cldice), float("nan")
    direct = (np.linalg.norm(pred_path[0] - ref_path[0]) +
              np.linalg.norm(pred_path[-1] - ref_path[-1]))
    reversed_error = (np.linalg.norm(pred_path[0] - ref_path[-1]) +
                      np.linalg.norm(pred_path[-1] - ref_path[0]))
    ref_length = np.sqrt((np.diff(ref_path, axis=0) ** 2).sum(axis=1)).sum()
    endpoint_error = float(0.5 * min(direct, reversed_error) / max(ref_length, 1.0))
    return float(cldice), endpoint_error


def _region_metrics(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    intersection = int(np.logical_and(prediction, reference).sum())
    pred_area, ref_area = int(prediction.sum()), int(reference.sum())
    dice = 2.0 * intersection / max(pred_area + ref_area, 1)
    union = int(np.logical_or(prediction, reference).sum())
    iou = intersection / max(union, 1)
    boundary_f1, assd, hd95 = _surface_metrics(prediction, reference)
    cldice, endpoint_error = _skeleton_metrics(prediction, reference)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "boundary_f1": boundary_f1,
        "assd": assd,
        "hd95": hd95,
        "cldice": cldice,
        "endpoint_error_fraction": endpoint_error,
        "area_error_fraction": abs(pred_area - ref_area) / max(ref_area, 1),
    }


def _prediction_path(root: Path, stem: str) -> Path:
    candidates = [
        root / f"{stem}_predicted_instances.png",
        root / f"{stem}_masks.png",
        root / f"{stem}.png",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"No prediction found for {stem} under {root}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare shape refinement against independent manual masks.")
    parser.add_argument("dataset", help="dataset containing masks/*_masks.png")
    parser.add_argument("predictions", help="directory containing coarse instance masks")
    parser.add_argument("--probability-dir", default=None,
                        help="optional directory containing <stem>_foreground.npy")
    parser.add_argument("--checkpoint", default=None,
                        help="optional worm checkpoint used to recreate foreground probabilities")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=0,
                        help="evaluate only the first N images; 0 evaluates all")
    parser.add_argument("--sample", type=int, default=0,
                        help="evaluate N evenly spaced images across the dataset")
    parser.add_argument("--expected-count", type=int, default=10,
                        help="production count preset; repairs require n-1 through n+1")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    prediction_root = Path(args.predictions)
    output = Path(args.output) if args.output else prediction_root / "shape_refinement_evaluation.csv"
    model = device = None
    image_size = IMAGE_SIZE
    if args.checkpoint:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model = WormUNet(base=int(checkpoint.get("base", 16))).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        image_size = int(checkpoint.get("image_size", IMAGE_SIZE))
    rows: list[dict[str, object]] = []
    metric_names = [
        "dice", "iou", "boundary_f1", "assd", "hd95", "cldice",
        "endpoint_error_fraction", "area_error_fraction",
    ]

    reference_paths = sorted((dataset / "masks").glob("*_masks.png"))
    if args.sample and args.sample < len(reference_paths):
        indices = np.linspace(0, len(reference_paths) - 1, args.sample).round().astype(int)
        reference_paths = [reference_paths[int(index)] for index in indices]
    elif args.limit:
        reference_paths = reference_paths[:args.limit]
    for reference_path in reference_paths:
        stem = reference_path.name[:-len("_masks.png")]
        coarse_path = _prediction_path(prediction_root, stem)
        reference = np.asarray(Image.open(reference_path), dtype=np.uint16)
        coarse = np.asarray(Image.open(coarse_path), dtype=np.uint16)
        if coarse.shape != reference.shape:
            coarse = np.asarray(Image.fromarray(coarse).resize(
                (reference.shape[1], reference.shape[0]), Image.Resampling.NEAREST),
                dtype=np.uint16)
        probability = None
        if args.probability_dir:
            probability_path = Path(args.probability_dir) / f"{stem}_foreground.npy"
            if probability_path.is_file():
                probability = np.load(probability_path)
        if probability is None and model is not None and device is not None:
            normalized_path = dataset / "images" / f"{stem}.png"
            probabilities = predict_semantic(model, normalized_path, device, image_size)
            foreground_small = probabilities[1:].sum(axis=0).astype(np.float32)
            probability = np.asarray(Image.fromarray(foreground_small, mode="F").resize(
                (reference.shape[1], reference.shape[0]), Image.Resampling.BILINEAR),
                dtype=np.float32)
        count_allows_repair = abs(int(coarse.max()) - args.expected_count) <= 1
        refined, reports = refine_instance_shapes(
            coarse, probability if count_allows_repair else None)
        report_by_label = {report.label: report for report in reports}
        pairs, _ = pair_instances(coarse, reference)
        predicted_for_reference = {ref_id: pred_id for ref_id, pred_id, _ in pairs}

        for ref_id in [int(value) for value in np.unique(reference) if value]:
            pred_id = predicted_for_reference.get(ref_id, 0)
            reference_mask = reference == ref_id
            coarse_mask = coarse == pred_id if pred_id else np.zeros_like(reference_mask)
            refined_mask = refined == pred_id if pred_id else np.zeros_like(reference_mask)
            before = _region_metrics(coarse_mask, reference_mask)
            after = _region_metrics(refined_mask, reference_mask)
            report = report_by_label.get(pred_id)
            row: dict[str, object] = {
                "image": stem,
                "manual_label": ref_id,
                "predicted_label": pred_id,
                "repaired": bool(report.repaired) if report else False,
                "status": report.status if report else "MISSING",
                "candidate_kind": report.candidate_kind if report else "missing",
                "probability_evidence": report.probability_evidence if report else float("nan"),
            }
            for name in metric_names:
                row[f"before_{name}"] = before[name]
                row[f"after_{name}"] = after[name]
                # Positive delta always means improvement.
                if name in {"assd", "hd95", "endpoint_error_fraction", "area_error_fraction"}:
                    row[f"delta_{name}"] = before[name] - after[name]
                else:
                    row[f"delta_{name}"] = after[name] - before[name]
            rows.append(row)
        print(f"{stem}: coarse={int(coarse.max())}, refined={int(refined.max())}, "
              f"repaired={sum(int(report.repaired) for report in reports)}", flush=True)

    if not rows:
        raise FileNotFoundError(f"No reference masks found under {dataset / 'masks'}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    repaired = [row for row in rows if row["repaired"]]
    print(f"Instances: {len(rows)}; repaired: {len(repaired)}")
    if repaired:
        for name in metric_names:
            values = np.asarray([float(row[f"delta_{name}"]) for row in repaired], dtype=float)
            finite = values[np.isfinite(values)]
            print(f"delta_{name}: mean={float(finite.mean()) if len(finite) else float('nan'):.6f}; "
                  f"harm_rate={float(np.mean(finite < 0)) if len(finite) else float('nan'):.3f}")
    print("Results:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
