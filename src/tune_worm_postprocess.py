from __future__ import print_function

import argparse
import itertools
import os

import numpy as np
from scipy import ndimage
import torch

from train_worm_unet import (WormUNet, discover_records, split_records,
                             split_records_from_manifest)
from evaluate_worm_unet import predict_semantic, semantic_to_instances, resize_labels, pair_instances
from PIL import Image


def candidate_count(probabilities, area_fraction, height_fraction, threshold, erosion):
    semantic = np.argmax(probabilities, axis=0)
    foreground = semantic > 0
    foreground = ndimage.binary_closing(foreground, iterations=1)
    interior = semantic == 1 if threshold is None else probabilities[1] >= threshold
    if erosion:
        interior = ndimage.binary_erosion(interior, iterations=erosion)
    components, number = ndimage.label(interior)
    areas = np.bincount(components.ravel())
    boxes = ndimage.find_objects(components)
    foreground_area = max(int(foreground.sum()), 1)
    yy = np.nonzero(foreground)[0]
    bundle_height = int(yy.max() - yy.min() + 1) if len(yy) else semantic.shape[0]
    count = 0
    for label in range(1, number + 1):
        box = boxes[label - 1]
        component_height = box[0].stop - box[0].start if box is not None else 0
        if (areas[label] >= area_fraction * foreground_area and
                component_height >= height_fraction * bundle_height):
            count += 1
    return count


def candidate_features(probabilities, threshold, erosion):
    semantic = np.argmax(probabilities, axis=0)
    foreground = semantic > 0
    foreground = ndimage.binary_closing(foreground, iterations=1)
    foreground_components, foreground_number = ndimage.label(foreground)
    if foreground_number:
        component_sizes = np.bincount(foreground_components.ravel())[1:]
        keep = np.flatnonzero(component_sizes >= 0.03 * component_sizes.max()) + 1
        foreground = np.isin(foreground_components, keep)
    interior = semantic == 1 if threshold is None else probabilities[1] >= threshold
    interior &= foreground
    if erosion:
        interior = ndimage.binary_erosion(interior, iterations=erosion)
    components, number = ndimage.label(interior)
    areas = np.bincount(components.ravel())
    boxes = ndimage.find_objects(components)
    foreground_area = max(int(foreground.sum()), 1)
    yy = np.nonzero(foreground)[0]
    bundle_height = int(yy.max() - yy.min() + 1) if len(yy) else semantic.shape[0]
    features = []
    for label in range(1, number + 1):
        box = boxes[label - 1]
        if box is not None:
            features.append((areas[label] / foreground_area,
                             (box[0].stop - box[0].start) / bundle_height))
    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--counts-only", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.checkpoint or os.path.join(args.dataset, "worm_unet", "best_model.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = WormUNet(base=int(checkpoint.get("base", 16))).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    image_size = int(checkpoint.get("image_size", 512))
    records = discover_records(args.dataset)
    if args.split_manifest:
        _, validation = split_records_from_manifest(records, args.split_manifest)
        validation_stems = {record[3] for record in validation}
    else:
        validation_stems = set(checkpoint.get("validation_records", []))
    if not validation_stems:
        _, validation = split_records(records)
        validation_stems = {record[3] for record in validation}
    cached = [(record, predict_semantic(model, record[0], device, image_size=image_size))
              for record in records if record[3] in validation_stems]
    thresholds = [None, 0.25, 0.35, 0.45, 0.55, 0.65]
    erosions = range(0, 7)
    feature_cache = {}
    for record, probabilities in cached:
        for threshold, erosion in itertools.product(thresholds, erosions):
            feature_cache[(record[3], threshold, erosion)] = candidate_features(
                probabilities, threshold, erosion)
    settings = []
    for threshold, erosion, area, height in itertools.product(
            thresholds, erosions, [0.002, 0.003, 0.005, 0.008, 0.012],
            [0.05, 0.10, 0.16, 0.22]):
        results = []
        for record, _ in cached:
            features = feature_cache[(record[3], threshold, erosion)]
            count = sum(a >= area and h >= height for a, h in features)
            results.append((record[3], record[2], count == record[2]))
        val_n10 = sum(ok for stem, count, ok in results if stem in validation_stems and count == 10)
        val_anomaly = sum(ok for stem, count, ok in results if stem in validation_stems and count != 10)
        val_all = sum(ok for stem, _, ok in results if stem in validation_stems)
        all_correct = sum(ok for _, _, ok in results)
        settings.append((val_n10, val_anomaly, val_all, all_correct,
                         threshold, erosion, area, height))
    settings.sort(key=lambda item: (item[0], item[1], item[3]), reverse=True)
    print("Top count settings:")
    for setting in settings[:15]:
        print(setting)

    print("\nTop anomaly-recall settings:")
    anomaly_settings = sorted(settings, key=lambda item: (item[1], item[0], item[3]), reverse=True)
    for setting in anomaly_settings[:15]:
        print(setting)

    if args.counts_only:
        return

    print("\nInstance Dice for five best settings:")
    for setting in settings[:5]:
        _, _, _, _, threshold, erosion, area, height = setting
        rows = []
        for record, probabilities in cached:
            if record[3] not in validation_stems:
                continue
            predicted, _ = semantic_to_instances(
                probabilities, area, height, threshold, erosion, max_instances=None)
            reference = np.asarray(Image.open(record[1]), dtype=np.uint16)
            predicted = resize_labels(predicted, (reference.shape[1], reference.shape[0]))
            _, dices = pair_instances(predicted, reference)
            rows.append((record[3], int(predicted.max()), float(np.mean(dices))))
        print("setting", setting, "validation", rows,
              "mean", float(np.mean([row[2] for row in rows])))


if __name__ == "__main__":
    main()
