from __future__ import print_function

import glob
import os
import sys

if sys.version_info[0] < 3:
    print("This utility requires Python 3.9+.")
    sys.exit(2)

import numpy as np
from PIL import Image

from inspect_roi_dataset import image_array


GRID_SIZE = 401


def load_label(path):
    return np.asarray(Image.open(path), dtype=np.uint16)


def bundle_bbox(labels):
    ys, xs = np.nonzero(labels)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def shape_profiles(labels, count=10):
    xmin, ymin, xmax, ymax = bundle_bbox(labels)
    width = float(max(1, xmax - xmin))
    height = float(max(1, ymax - ymin))
    grid = np.linspace(0.0, 1.0, GRID_SIZE)
    left_profiles = np.full((count, GRID_SIZE), np.nan, dtype=np.float32)
    right_profiles = np.full((count, GRID_SIZE), np.nan, dtype=np.float32)

    for label in range(1, count + 1):
        row_y = []
        row_left = []
        row_right = []
        for y in range(ymin, ymax + 1):
            xs = np.flatnonzero(labels[y] == label)
            if xs.size:
                row_y.append((y - ymin) / height)
                row_left.append((xs[0] - xmin) / width)
                row_right.append((xs[-1] - xmin) / width)
        row_y = np.asarray(row_y)
        valid = np.logical_and(grid >= row_y.min(), grid <= row_y.max())
        left_profiles[label - 1, valid] = np.interp(grid[valid], row_y, row_left)
        right_profiles[label - 1, valid] = np.interp(grid[valid], row_y, row_right)
    return left_profiles, right_profiles


def mean_template(profile_pairs):
    left = np.nanmean(np.stack([pair[0] for pair in profile_pairs]), axis=0)
    right = np.nanmean(np.stack([pair[1] for pair in profile_pairs]), axis=0)
    return left, right


def render_template(template, shape, bbox):
    left, right = template
    xmin, ymin, xmax, ymax = bbox
    width = float(max(1, xmax - xmin))
    height = float(max(1, ymax - ymin))
    labels = np.zeros(shape, dtype=np.uint16)
    distances = np.full(shape, np.inf, dtype=np.float32)

    for y in range(ymin, ymax + 1):
        u = (y - ymin) / height
        grid_index = min(GRID_SIZE - 1, int(round(u * (GRID_SIZE - 1))))
        for index in range(left.shape[0]):
            lo = left[index, grid_index]
            hi = right[index, grid_index]
            if np.isnan(lo) or np.isnan(hi):
                continue
            x0 = max(0, int(round(xmin + lo * width)))
            x1 = min(shape[1] - 1, int(round(xmin + hi * width)))
            if x1 < x0:
                continue
            center = 0.5 * (x0 + x1)
            xs = np.arange(x0, x1 + 1)
            current_distance = np.abs(xs - center)
            replace = current_distance < distances[y, x0:x1 + 1]
            labels[y, x0:x1 + 1][replace] = index + 1
            distances[y, x0:x1 + 1][replace] = current_distance[replace]
    return labels


def dice_per_label(prediction, reference, count=10):
    values = []
    for label in range(1, count + 1):
        pred = prediction == label
        ref = reference == label
        denominator = int(pred.sum()) + int(ref.sum())
        values.append(2.0 * int(np.logical_and(pred, ref).sum()) / max(1, denominator))
    return values


def main(dataset_root):
    mask_dir = os.path.join(dataset_root, "masks")
    output_dir = os.path.join(dataset_root, "template_evaluation")
    os.makedirs(output_dir, exist_ok=True)
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*_masks.png")))
    labels_all = [load_label(path) for path in mask_paths]
    profiles_all = [shape_profiles(labels) for labels in labels_all]
    source_root = os.path.dirname(dataset_root)

    all_dice = []
    for test_index, (path, reference) in enumerate(zip(mask_paths, labels_all)):
        training = [profiles for index, profiles in enumerate(profiles_all) if index != test_index]
        template = mean_template(training)
        prediction = render_template(template, reference.shape, bundle_bbox(reference))
        dice = dice_per_label(prediction, reference)
        image_name = os.path.basename(path).replace("_masks.png", ".tif")
        intensity = image_array(Image.open(os.path.join(source_root, image_name))).astype(np.float64)
        manual_means = [float(intensity[reference == label].mean()) for label in range(1, 11)]
        predicted_means = [float(intensity[prediction == label].mean()) for label in range(1, 11)]
        mean_errors = [abs(pred - ref) / max(abs(ref), 1e-12)
                       for pred, ref in zip(predicted_means, manual_means)]
        all_dice.extend(dice)
        base = os.path.basename(path).replace("_masks.png", "")
        Image.fromarray(prediction).save(os.path.join(output_dir, base + "_template_masks.png"))
        print("%s: mean Dice %.4f, min %.4f, fluorescence Mean MAPE %.2f%%, max %.2f%%" %
              (base, float(np.mean(dice)), min(dice), 100.0 * float(np.mean(mean_errors)),
               100.0 * max(mean_errors)))
    print("Overall: mean Dice %.4f, median %.4f, min %.4f" %
          (float(np.mean(all_dice)), float(np.median(all_dice)), min(all_dice)))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate_shape_template.py <segmentation-dataset-folder>")
        sys.exit(2)
    main(sys.argv[1])
