from __future__ import print_function

import argparse
import csv
import glob
import os
import sys

if sys.version_info[0] < 3:
    print("This utility requires Python 3.9+; do not run it with MGLTools Python 2.")
    sys.exit(2)

import numpy as np
from PIL import Image, ImageDraw

from inspect_roi_dataset import image_array, load_rois, polygon_mask


def normalize_for_segmentation(array, mode="legacy"):
    sample = array.ravel()[::16]
    low = float(np.percentile(sample, 2.0))
    high = float(np.percentile(sample, 99.8))
    if mode == "background_aware":
        height, width = array.shape[:2]
        # The acquisition layout deliberately leaves a blank strip on the left.
        # Its median is a robust black level for bright/noisy fluorescence images.
        background_strip = array[
            int(round(0.10 * height)):int(round(0.90 * height)),
            int(round(0.02 * width)):int(round(0.18 * width))]
        if background_strip.size:
            low = max(low, float(np.median(background_strip)))
    elif mode != "legacy":
        raise ValueError("Unknown normalization mode: " + str(mode))
    if high <= low:
        high = low + 1.0
    normalized = np.clip((array.astype(np.float32) - low) * (255.0 / (high - low)), 0, 255)
    # A mild square-root transform makes dim worm edges visible without
    # modifying the raw TIFF used later for fluorescence measurements.
    normalized = np.sqrt(normalized / 255.0) * 255.0
    return normalized.astype(np.uint8)


def resolve_overlaps(masks):
    stack = np.stack(masks, axis=0).astype(bool)
    coverage = stack.sum(axis=0)
    labels = np.zeros(coverage.shape, dtype=np.uint16)

    # Assign singly covered pixels directly.
    for index in range(stack.shape[0]):
        labels[np.logical_and(stack[index], coverage == 1)] = index + 1

    # Manual ImageJ polygons overlap slightly at shared edges and tips. For
    # label-image training, assign overlapping pixels to the nearest horizontal
    # centerline among the ROIs that contain that pixel.
    centers = np.full((stack.shape[0], stack.shape[1]), np.nan, dtype=np.float32)
    for index in range(stack.shape[0]):
        for y in range(stack.shape[1]):
            xs = np.flatnonzero(stack[index, y])
            if xs.size:
                centers[index, y] = 0.5 * (xs[0] + xs[-1])

    overlap_y, overlap_x = np.nonzero(coverage > 1)
    for y, x in zip(overlap_y, overlap_x):
        candidates = np.flatnonzero(stack[:, y, x])
        distances = np.abs(centers[candidates, y] - x)
        labels[y, x] = int(candidates[np.nanargmin(distances)]) + 1
    return labels, coverage


def label_overlay(normalized, labels):
    rgb = np.dstack((normalized // 4, normalized, normalized // 4)).astype(np.uint8)
    canvas = Image.fromarray(rgb, "RGB")
    draw = ImageDraw.Draw(canvas)
    colors = [
        (255, 64, 64), (255, 170, 0), (255, 255, 0), (100, 255, 80),
        (0, 255, 210), (0, 180, 255), (80, 100, 255), (180, 80, 255),
        (255, 80, 210), (255, 150, 170), (255, 255, 255), (150, 150, 150),
    ]
    for label in range(1, int(labels.max()) + 1):
        mask = labels == label
        edge = np.logical_and(mask, np.logical_not(
            np.logical_and.reduce((
                np.roll(mask, 1, axis=0), np.roll(mask, -1, axis=0),
                np.roll(mask, 1, axis=1), np.roll(mask, -1, axis=1),
            ))
        ))
        ys, xs = np.nonzero(edge)
        color = colors[(label - 1) % len(colors)]
        for x, y in zip(xs[::2], ys[::2]):
            draw.point((int(x), int(y)), fill=color)
    return canvas


def main(root, output_root=None, normalization_mode="legacy"):
    output_root = output_root or os.path.join(root, "_segmentation_dataset")
    input_dir = os.path.join(output_root, "images")
    mask_dir = os.path.join(output_root, "masks")
    overlay_dir = os.path.join(output_root, "overlays")
    for directory in (input_dir, mask_dir, overlay_dir):
        os.makedirs(directory, exist_ok=True)

    records = []
    for roi_zip in sorted(glob.glob(os.path.join(root, "RoiSet_*.zip"))):
        stem = os.path.basename(roi_zip)[len("RoiSet_"):-len(".zip")]
        image_path = os.path.join(root, stem + ".tif")
        image = Image.open(image_path)
        array = image_array(image)
        rois = load_rois(roi_zip)
        worm_rois = rois[:-1]
        masks = [polygon_mask(image.size, roi["points"]) for roi in worm_rois]
        labels, coverage = resolve_overlaps(masks)
        normalized = normalize_for_segmentation(array, mode=normalization_mode)

        Image.fromarray(normalized, "L").save(os.path.join(input_dir, stem + ".png"))
        Image.fromarray(labels).save(os.path.join(mask_dir, stem + "_masks.png"))
        label_overlay(normalized, labels).save(os.path.join(overlay_dir, stem + "_labels.png"))

        records.append({
            "image": stem + ".tif",
            "width": image.size[0],
            "height": image.size[1],
            "instances": int(labels.max()),
            "foreground_pixels": int((labels > 0).sum()),
            "overlap_pixels_resolved": int((coverage > 1).sum()),
            "normalization_mode": normalization_mode,
        })
        print("%s: %d instances, %d overlapping pixels resolved" %
              (stem, int(labels.max()), int((coverage > 1).sum())))

    manifest_path = os.path.join(output_root, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print("Dataset:", output_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--output", default=None)
    parser.add_argument("--normalization", choices=("legacy", "background_aware"),
                        default="legacy")
    options = parser.parse_args()
    main(options.root, options.output, options.normalization)
