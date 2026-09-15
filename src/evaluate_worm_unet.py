from __future__ import print_function

import argparse
import csv
import glob
import os

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
import torch

from train_worm_unet import (IMAGE_SIZE, WormUNet, discover_records,
                             split_records, split_records_from_manifest)


def predict_semantic(model, image_path, device, image_size=IMAGE_SIZE):
    original = Image.open(image_path).convert("L")
    small = original.resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(small, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array[None, None]).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].float().cpu().numpy()
    return probabilities


def semantic_to_instances(probabilities, min_area_fraction=0.005, min_height_fraction=0.10,
                          interior_threshold=None, erosion_iterations=0,
                          max_instances=12):
    semantic = np.argmax(probabilities, axis=0)
    foreground = semantic > 0
    # A small closing prevents isolated one-pixel background errors from cutting a worm.
    foreground = ndimage.binary_closing(foreground, iterations=1)
    foreground_components, foreground_number = ndimage.label(foreground)
    if foreground_number:
        component_sizes = np.bincount(foreground_components.ravel())[1:]
        keep = np.flatnonzero(component_sizes >= 0.03 * component_sizes.max()) + 1
        foreground = np.isin(foreground_components, keep)
    interior = semantic == 1 if interior_threshold is None else probabilities[1] >= interior_threshold
    interior = np.logical_and(interior, foreground)
    if erosion_iterations:
        interior = ndimage.binary_erosion(interior, iterations=erosion_iterations)
    components, number = ndimage.label(interior)
    foreground_area = max(int(foreground.sum()), 1)
    ys, xs = np.nonzero(foreground)
    bundle_height = (int(ys.max()) - int(ys.min()) + 1) if len(ys) else IMAGE_SIZE
    candidates = []
    for label in range(1, number + 1):
        yy, xx = np.nonzero(components == label)
        if len(yy) == 0:
            continue
        area = len(yy)
        height = int(yy.max()) - int(yy.min()) + 1
        if area >= min_area_fraction * foreground_area and height >= min_height_fraction * bundle_height:
            candidates.append((label, area, height, float(xx.mean())))
    if max_instances and len(candidates) > max_instances:
        candidates = sorted(candidates, key=lambda item: item[1] * item[2], reverse=True)[:max_instances]
    candidates.sort(key=lambda item: item[3])
    seeds = np.zeros_like(components, dtype=np.uint16)
    for new_label, (old_label, _, _, _) in enumerate(candidates, 1):
        seeds[components == old_label] = new_label
    if not candidates:
        return seeds, semantic
    # Assign predicted boundary pixels to the nearest worm interior. This produces
    # mutually exclusive ImageJ-compatible ROIs without changing their outer union.
    nearest = ndimage.distance_transform_edt(seeds == 0, return_distances=False, return_indices=True)
    filled = seeds[tuple(nearest)]
    filled[np.logical_not(foreground)] = 0
    return filled.astype(np.uint16), semantic


def resize_labels(labels, size):
    return np.asarray(Image.fromarray(labels).resize(size, Image.Resampling.NEAREST), dtype=np.uint16)


def pair_instances(prediction, reference):
    pred_ids = [int(v) for v in np.unique(prediction) if v]
    ref_ids = [int(v) for v in np.unique(reference) if v]
    dice = np.zeros((len(ref_ids), len(pred_ids)), dtype=np.float64)
    for i, ref_id in enumerate(ref_ids):
        ref = reference == ref_id
        for j, pred_id in enumerate(pred_ids):
            pred = prediction == pred_id
            dice[i, j] = 2.0 * np.logical_and(ref, pred).sum() / (ref.sum() + pred.sum() + 1e-9)
    if dice.size:
        rows, cols = linear_sum_assignment(-dice)
        pairs = [(ref_ids[r], pred_ids[c], float(dice[r, c])) for r, c in zip(rows, cols)]
    else:
        pairs = []
    paired_refs = {item[0] for item in pairs}
    all_dices = [item[2] for item in pairs]
    all_dices.extend([0.0] * (len(ref_ids) - len(paired_refs)))
    return pairs, all_dices


def tip_metrics(reference_mask, prediction_mask, fraction=0.18):
    ref_y, ref_x = np.nonzero(reference_mask)
    pred_y, pred_x = np.nonzero(prediction_mask)
    if not len(ref_y) or not len(pred_y):
        return 0.0, np.nan, np.nan, np.nan
    y_min, y_max = int(ref_y.min()), int(ref_y.max())
    worm_length = max(y_max - y_min + 1, 1)
    cutoff = max(int(round(fraction * worm_length)), 2)
    y_grid = np.arange(reference_mask.shape[0])[:, None]
    zone = (y_grid <= y_min + cutoff) | (y_grid >= y_max - cutoff)
    ref_tip = reference_mask & zone
    pred_tip = prediction_mask & zone
    tip_dice = 2.0 * np.logical_and(ref_tip, pred_tip).sum() / (
        ref_tip.sum() + pred_tip.sum() + 1e-9)

    def endpoint(mask, top):
        yy, xx = np.nonzero(mask)
        extent = max(int(yy.max()) - int(yy.min()) + 1, 1)
        band = max(int(round(0.03 * extent)), 2)
        selected = yy <= yy.min() + band if top else yy >= yy.max() - band
        return np.asarray((float(xx[selected].mean()), float(yy[selected].mean())))

    top_error = float(np.linalg.norm(endpoint(reference_mask, True) - endpoint(prediction_mask, True)))
    bottom_error = float(np.linalg.norm(endpoint(reference_mask, False) - endpoint(prediction_mask, False)))
    normalized_error = 0.5 * (top_error + bottom_error) / worm_length
    return float(tip_dice), top_error, bottom_error, float(normalized_error)


def find_tiff(search_root, stem):
    if "__" in stem:
        stem = stem.split("__", 1)[1]
    for extension in (".tif", ".tiff", ".TIF", ".TIFF"):
        matches = glob.glob(os.path.join(search_root, "**", stem + extension), recursive=True)
        matches = [path for path in matches if "_inspection" not in path and "_segmentation_dataset" not in path]
        if matches:
            return matches[0]
    return None


def contour_overlay(image_path, reference, prediction, output_path):
    base = Image.open(image_path).convert("RGB")
    ref_edge = np.zeros(reference.shape, dtype=bool)
    pred_edge = np.zeros(prediction.shape, dtype=bool)
    for label in np.unique(reference):
        if label:
            mask = reference == label
            ref_edge |= np.logical_and(mask, np.logical_not(ndimage.binary_erosion(mask)))
    for label in np.unique(prediction):
        if label:
            mask = prediction == label
            pred_edge |= np.logical_and(mask, np.logical_not(ndimage.binary_erosion(mask)))
    rgb = np.asarray(base).copy()
    rgb[ref_edge] = (0, 255, 255)
    rgb[pred_edge] = (255, 0, 255)
    result = Image.fromarray(rgb)
    draw = ImageDraw.Draw(result)
    draw.rectangle((5, 5, 470, 34), fill=(0, 0, 0))
    draw.text((10, 10), "manual: cyan   automatic: magenta", fill=(255, 255, 255))
    result.save(output_path)


def colorize_labels(labels):
    colors = np.asarray([
        (0, 0, 0), (230, 25, 75), (60, 180, 75), (255, 225, 25),
        (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240),
        (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128),
        (220, 190, 255)], dtype=np.uint8)
    return colors[np.asarray(labels, dtype=np.int64) % len(colors)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split-manifest", default=None,
                        help="optional manifest defining the evaluation split")
    parser.add_argument("--only-split", choices=("training", "validation"), default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--tiff-root", default=None)
    parser.add_argument("--interior-threshold", type=float, default=None)
    parser.add_argument("--erosion", type=int, default=0)
    parser.add_argument("--min-area", type=float, default=0.005)
    parser.add_argument("--min-height", type=float, default=0.10)
    parser.add_argument("--max-instances", type=int, default=12,
                        help="0 disables the safety cap")
    args = parser.parse_args()
    checkpoint_path = args.checkpoint or os.path.join(args.dataset, "worm_unet", "best_model.pt")
    output_dir = args.output or os.path.join(
        os.path.dirname(os.path.abspath(checkpoint_path)), "evaluation")
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = WormUNet(base=int(checkpoint.get("base", 16))).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    image_size = int(checkpoint.get("image_size", IMAGE_SIZE))
    records = discover_records(args.dataset)
    if args.split_manifest:
        manifest_training, manifest_validation = split_records_from_manifest(
            records, args.split_manifest)
        validation_stems = {record[3] for record in manifest_validation}
    else:
        checkpoint_validation = checkpoint.get("validation_records")
        if checkpoint_validation:
            validation_stems = set(checkpoint_validation)
        else:
            _, validation_records = split_records(records)
            validation_stems = {record[3] for record in validation_records}
    if args.only_split:
        want_validation = args.only_split == "validation"
        records = [record for record in records
                   if (record[3] in validation_stems) == want_validation]
    summary_rows, worm_rows = [], []

    for image_path, mask_path, expected_count, stem in records:
        probabilities = predict_semantic(model, image_path, device, image_size=image_size)
        predicted_small, semantic = semantic_to_instances(
            probabilities, min_area_fraction=args.min_area,
            min_height_fraction=args.min_height,
            interior_threshold=args.interior_threshold,
            erosion_iterations=args.erosion,
            max_instances=args.max_instances or None)
        reference = np.asarray(Image.open(mask_path), dtype=np.uint16)
        predicted = resize_labels(predicted_small, (reference.shape[1], reference.shape[0]))
        pairs, dices = pair_instances(predicted, reference)
        tiff_path = find_tiff(args.tiff_root, stem) if args.tiff_root else None
        intensity = np.asarray(Image.open(tiff_path), dtype=np.float64) if tiff_path else None
        mean_errors, tip_dices, endpoint_errors = [], [], []
        for ref_id, pred_id, dice in pairs:
            tip_dice, top_error, bottom_error, normalized_endpoint_error = tip_metrics(
                reference == ref_id, predicted == pred_id)
            tip_dices.append(tip_dice)
            endpoint_errors.append(normalized_endpoint_error)
            ref_mean = float(intensity[reference == ref_id].mean()) if intensity is not None else np.nan
            pred_mean = float(intensity[predicted == pred_id].mean()) if intensity is not None else np.nan
            mean_error = abs(pred_mean - ref_mean) / max(abs(ref_mean), 1e-9) if intensity is not None else np.nan
            if np.isfinite(mean_error):
                mean_errors.append(mean_error)
            worm_rows.append([stem, "validation" if stem in validation_stems else "training",
                              ref_id, pred_id, dice, tip_dice, top_error, bottom_error,
                              normalized_endpoint_error, ref_mean, pred_mean, mean_error])
        predicted_count = int(predicted.max())
        summary_rows.append([
            stem, "validation" if stem in validation_stems else "training", expected_count,
            predicted_count, predicted_count == expected_count,
            float(np.mean(dices)) if dices else 0.0,
            float(np.min(dices)) if dices else 0.0,
            float(np.mean(np.asarray(dices) >= 0.5)) if dices else 0.0,
            float(np.mean(tip_dices)) if tip_dices else 0.0,
            float(np.mean(endpoint_errors)) if endpoint_errors else np.nan,
            float(np.mean(mean_errors)) if mean_errors else np.nan,
        ])
        Image.fromarray(predicted).save(os.path.join(output_dir, stem + "_predicted_instances.png"))
        Image.fromarray(colorize_labels(predicted)).save(
            os.path.join(output_dir, stem + "_predicted_instances_color.png"))
        contour_overlay(image_path, reference, predicted,
                        os.path.join(output_dir, stem + "_overlay.png"))
        print("%-45s expected=%2d predicted=%2d mean_dice=%.3f" %
              (stem, expected_count, predicted_count, np.mean(dices) if dices else 0.0), flush=True)

    with open(os.path.join(output_dir, "image_summary.csv"), "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "split", "expected_count", "predicted_count", "count_correct",
                         "mean_matched_dice", "minimum_matched_dice", "fraction_dice_ge_0.5",
                         "mean_tip_dice", "mean_endpoint_error_fraction",
                         "mean_fluorescence_mape"])
        writer.writerows(summary_rows)
    with open(os.path.join(output_dir, "worm_pairs.csv"), "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "split", "manual_id", "automatic_id", "dice", "tip_dice",
                         "top_endpoint_error_pixels", "bottom_endpoint_error_pixels",
                         "mean_endpoint_error_fraction",
                         "manual_mean", "automatic_mean", "mean_error_fraction"])
        writer.writerows(worm_rows)
    validation = [row for row in summary_rows if row[1] == "validation"]
    print("Validation count accuracy: %d/%d" % (sum(bool(row[4]) for row in validation), len(validation)))
    print("Validation mean matched Dice: %.4f" % np.mean([row[5] for row in validation]))
    print("Validation mean tip Dice: %.4f" % np.mean([row[8] for row in validation]))
    print("Validation mean normalized endpoint error: %.4f" % np.mean([row[9] for row in validation]))
    print("Results:", output_dir)


if __name__ == "__main__":
    main()
