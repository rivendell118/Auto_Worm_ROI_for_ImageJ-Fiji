from __future__ import print_function

import argparse
import glob
import os

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from cellpose import models


COLORS = [
    (255, 64, 64), (255, 170, 0), (255, 255, 0), (100, 255, 80),
    (0, 255, 210), (0, 180, 255), (80, 100, 255), (180, 80, 255),
    (255, 80, 210), (255, 150, 170), (255, 255, 255), (150, 150, 150),
]


def match_instances(predicted, reference):
    pred_ids = np.unique(predicted)
    pred_ids = pred_ids[pred_ids > 0]
    ref_ids = np.unique(reference)
    ref_ids = ref_ids[ref_ids > 0]
    scores = np.zeros((len(ref_ids), len(pred_ids)), dtype=np.float64)
    for i, ref_id in enumerate(ref_ids):
        ref = reference == ref_id
        ref_area = int(ref.sum())
        for j, pred_id in enumerate(pred_ids):
            pred = predicted == pred_id
            intersection = int(np.logical_and(ref, pred).sum())
            scores[i, j] = 2.0 * intersection / max(1, ref_area + int(pred.sum()))
    if scores.size == 0:
        return [], scores
    rows, cols = linear_sum_assignment(-scores)
    return [(int(ref_ids[r]), int(pred_ids[c]), float(scores[r, c])) for r, c in zip(rows, cols)], scores


def overlay(image, labels):
    rgb = np.dstack((image // 4, image, image // 4)).astype(np.uint8)
    canvas = Image.fromarray(rgb)
    draw = ImageDraw.Draw(canvas)
    for label in np.unique(labels):
        if label == 0:
            continue
        mask = labels == label
        edge = np.logical_and(mask, np.logical_not(ndimage.binary_erosion(mask)))
        ys, xs = np.nonzero(edge)
        color = COLORS[(int(label) - 1) % len(COLORS)]
        for x, y in zip(xs[::2], ys[::2]):
            draw.point((int(x), int(y)), fill=color)
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--diameter", type=float, default=50.0,
                        help="Diameter at 1024-pixel image width")
    parser.add_argument("--niter", type=int, default=1200)
    args = parser.parse_args()

    image_dir = os.path.join(args.dataset, "images")
    mask_dir = os.path.join(args.dataset, "masks")
    output_dir = os.path.join(args.dataset, "cellpose_baseline")
    os.makedirs(output_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    if args.limit:
        paths = paths[:args.limit]

    model = models.CellposeModel(gpu=True, pretrained_model="cpsam_v2")
    for image_path in paths:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        image = np.asarray(Image.open(image_path), dtype=np.uint8)
        reference = np.asarray(Image.open(os.path.join(mask_dir, stem + "_masks.png")), dtype=np.uint16)
        scale = image.shape[1] / 1024.0
        diameter = args.diameter * scale
        masks, flows, styles = model.eval(
            image,
            diameter=diameter,
            niter=int(round(args.niter * scale)),
            flow_threshold=0.4,
            cellprob_threshold=0.0,
            max_size_fraction=0.6,
            tile_overlap=0.2,
        )
        matches, _ = match_instances(masks, reference)
        matched_scores = [item[2] for item in matches]
        pred_count = int(masks.max())
        good = sum(score >= 0.5 for score in matched_scores)
        print("%s: expected=%d predicted=%d, matched Dice mean=%.3f, Dice>=0.5 %d/%d" %
              (stem, int(reference.max()), pred_count,
               float(np.mean(matched_scores)) if matched_scores else 0.0,
               good, int(reference.max())))
        Image.fromarray(masks.astype(np.uint16)).save(os.path.join(output_dir, stem + "_masks.png"))
        overlay(image, masks).save(os.path.join(output_dir, stem + "_overlay.png"))


if __name__ == "__main__":
    main()
