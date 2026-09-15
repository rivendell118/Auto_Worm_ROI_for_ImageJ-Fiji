from __future__ import print_function

import argparse
import csv
import os

import numpy as np
from PIL import Image
from scipy import ndimage
import torch

from evaluate_worm_unet import pair_instances, tip_metrics, colorize_labels
from train_tip_refiner import crop_with_padding, endpoint_crop_origin
from train_worm_unet import WormUNet, discover_records, split_records_from_manifest


def paste_patch(destination, patch, x0, y0, selector):
    size = patch.shape[0]
    x1, y1 = x0 + size, y0 + size
    dst_x0, dst_y0 = max(x0, 0), max(y0, 0)
    dst_x1, dst_y1 = min(x1, destination.shape[1]), min(y1, destination.shape[0])
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return
    src = (slice(dst_y0 - y0, dst_y1 - y0), slice(dst_x0 - x0, dst_x1 - x0))
    dst = (slice(dst_y0, dst_y1), slice(dst_x0, dst_x1))
    current = destination[dst]
    selected = selector[dst]
    current[selected] = patch[src][selected]


def keep_main_component(mask, guide):
    components, number = ndimage.label(mask)
    if number <= 1:
        return mask
    best_label, best_score = 0, -1
    for label in range(1, number + 1):
        component = components == label
        score = 10 * np.logical_and(component, guide).sum() + component.sum()
        if score > best_score:
            best_label, best_score = label, score
    return components == best_label


def refine_instances(model, image, coarse, device, patch_size=192,
                     probability_threshold=0.40, replace_fraction=0.14):
    masks = []
    original_masks = []
    for label in range(1, int(coarse.max()) + 1):
        guide = coarse == label
        yy, xx = np.nonzero(guide)
        if not len(yy):
            continue
        y_min, y_max = int(yy.min()), int(yy.max())
        worm_height = max(y_max - y_min + 1, 1)
        refined = guide.copy()
        for side in ("top", "bottom"):
            x0, y0 = endpoint_crop_origin(guide, side, patch_size)
            image_patch = crop_with_padding(image, x0, y0, patch_size, 0.0)
            guide_patch = crop_with_padding(guide, x0, y0, patch_size, False)
            inputs = torch.from_numpy(np.ascontiguousarray(np.stack(
                (image_patch, guide_patch.astype(np.float32)))[None], dtype=np.float32)).to(device)
            # Match production inference: use FP32 on both CPU and CUDA.
            with torch.no_grad():
                probability = torch.softmax(model(inputs), dim=1)[0, 1].float().cpu().numpy()
            prediction = probability >= probability_threshold
            y_grid = np.arange(refined.shape[0])[:, None]
            if side == "top":
                selector = y_grid <= y_min + int(round(replace_fraction * worm_height))
            else:
                selector = y_grid >= y_max - int(round(replace_fraction * worm_height))
            selector = np.broadcast_to(selector, refined.shape)
            paste_patch(refined, prediction, x0, y0, selector)
        refined = keep_main_component(refined, guide)
        masks.append(refined)
        original_masks.append(guide)

    if not masks:
        return np.zeros_like(coarse)
    stack = np.stack(masks)
    coverage = stack.sum(axis=0)
    output = np.zeros(coarse.shape, dtype=np.uint16)
    for index, mask in enumerate(masks, 1):
        output[mask & (coverage == 1)] = index
    overlap_y, overlap_x = np.nonzero(coverage > 1)
    if len(overlap_y):
        distance_maps = np.stack([
            ndimage.distance_transform_edt(~mask) for mask in original_masks
        ])
        for y, x in zip(overlap_y, overlap_x):
            candidates = np.flatnonzero(stack[:, y, x])
            winner = candidates[np.argmin(distance_maps[candidates, y, x])]
            output[y, x] = int(winner) + 1
    return output


def matched_metrics(prediction, reference):
    pairs, dices = pair_instances(prediction, reference)
    tip_dices, endpoint_errors = [], []
    for ref_id, pred_id, _ in pairs:
        tip_dice, _, _, endpoint_error = tip_metrics(
            reference == ref_id, prediction == pred_id)
        tip_dices.append(tip_dice)
        endpoint_errors.append(endpoint_error)
    missing = int(reference.max()) - len(pairs)
    tip_dices.extend([0.0] * max(missing, 0))
    return (float(np.mean(dices)) if dices else 0.0,
            float(np.mean(tip_dices)) if tip_dices else 0.0,
            float(np.mean(endpoint_errors)) if endpoint_errors else np.nan)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("guide_dir")
    parser.add_argument("checkpoint")
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--evaluate-all", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--replace-fraction", type=float, default=0.14)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    image_size = int(checkpoint.get("image_size", 768))
    patch_size = int(checkpoint.get("patch_size", 192))
    model = WormUNet(base=int(checkpoint.get("base", 12)), in_channels=2, out_channels=2).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    records = discover_records(args.dataset)
    if args.evaluate_all:
        validation = records
    elif args.split_manifest:
        _, validation = split_records_from_manifest(records, args.split_manifest)
    else:
        raise SystemExit("Provide --split-manifest, or use --evaluate-all.")
    rows = []
    for image_path, mask_path, expected_count, stem in validation:
        image = np.asarray(Image.open(image_path).convert("L").resize(
            (image_size, image_size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        reference = np.asarray(Image.open(mask_path).resize(
            (image_size, image_size), Image.Resampling.NEAREST), dtype=np.uint16)
        guide_path = os.path.join(args.guide_dir, stem + "_predicted_instances.png")
        coarse = np.asarray(Image.open(guide_path).resize(
            (image_size, image_size), Image.Resampling.NEAREST), dtype=np.uint16)
        refined = refine_instances(model, image, coarse, device, patch_size,
                                   args.threshold, args.replace_fraction)
        coarse_metrics = matched_metrics(coarse, reference)
        refined_metrics = matched_metrics(refined, reference)
        rows.append([stem, expected_count, int(coarse.max()), int(refined.max()),
                     *coarse_metrics, *refined_metrics])
        print("%-50s coarse_tip=%.3f refined_tip=%.3f coarse_end=%.3f refined_end=%.3f" %
              (stem, coarse_metrics[1], refined_metrics[1],
               coarse_metrics[2], refined_metrics[2]), flush=True)
    output_dir = os.path.dirname(args.checkpoint)
    output_name = ("whole_worm_refinement_evaluation_all.csv" if args.evaluate_all else
                   "whole_worm_refinement_evaluation.csv")
    output_path = os.path.join(output_dir, output_name)
    with open(output_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "expected_count", "coarse_count", "refined_count",
                         "coarse_dice", "coarse_tip_dice", "coarse_endpoint_error_fraction",
                         "refined_dice", "refined_tip_dice", "refined_endpoint_error_fraction"])
        writer.writerows(rows)
    print("Mean coarse tip Dice %.4f -> refined %.4f" %
          (np.mean([row[5] for row in rows]), np.mean([row[8] for row in rows])))
    print("Mean endpoint error %.4f -> %.4f" %
          (np.mean([row[6] for row in rows]), np.mean([row[9] for row in rows])))
    print("Results:", output_path)


if __name__ == "__main__":
    main()
