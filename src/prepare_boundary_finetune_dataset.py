from __future__ import print_function

import argparse
import csv
import os
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from inspect_roi_dataset import image_array, load_rois, polygon_mask
from prepare_segmentation_dataset import label_overlay, normalize_for_segmentation, resolve_overlaps


def classify_issue(stem):
    has_merge = "!" in stem or "！" in stem
    has_split = "=" in stem
    if has_merge and has_split:
        return "merge_split"
    if has_merge:
        return "merge"
    if has_split:
        return "split"
    return "normal"


def clean_source_stem(roi_stem):
    stem = roi_stem[len("RoiSet_"):] if roi_stem.startswith("RoiSet_") else roi_stem
    return stem


def matching_image(roi_zip):
    source_stem = clean_source_stem(roi_zip.stem)
    candidates = [source_stem]
    # One supplied archive is named RoiSet_Phsp4..., while its TIFF and CSV are
    # named hsp4.... Keep this compatibility narrow so real leading P names are
    # not silently changed when an exact image exists.
    if source_stem.startswith("P"):
        candidates.append(source_stem[1:])
    for stem in candidates:
        path = roi_zip.parent / (stem + ".tif")
        if path.is_file():
            return path, stem
    raise FileNotFoundError("No TIFF matches %s" % roi_zip)


def safe_name(value):
    value = re.sub(r"[!！=]+", " ", value)
    value = re.sub(r"\s+", "_", value.strip())
    value = re.sub(r"[^0-9A-Za-z_-]+", "_", value)
    return value.strip("_") or "sample"


def copy_base_dataset(base_root, image_dir, mask_dir):
    records = []
    for image_path in sorted((base_root / "images").glob("*.png")):
        mask_path = base_root / "masks" / (image_path.stem + "_masks.png")
        if not mask_path.is_file():
            raise FileNotFoundError("Missing base mask: %s" % mask_path)
        destination_image = image_dir / image_path.name
        destination_mask = mask_dir / mask_path.name
        shutil.copy2(str(image_path), str(destination_image))
        shutil.copy2(str(mask_path), str(destination_mask))
        labels = np.asarray(Image.open(mask_path), dtype=np.uint16)
        records.append({
            "Stem": image_path.stem,
            "Split": "training",
            "Source": "base_v020",
            "Issue": "normal",
            "Group": "base",
            "SourceImage": str(image_path),
            "SourceRoi": "",
            "Instances": int(labels.max()),
            "Width": int(labels.shape[1]),
            "Height": int(labels.shape[0]),
            "OverlapPixelsResolved": 0,
        })
    if not records:
        raise ValueError("No base PNGs found under %s" % base_root)
    return records


def build_supplement(supplement_root, image_dir, mask_dir, overlay_dir):
    roi_zips = sorted(
        path for path in supplement_root.rglob("RoiSet_*.zip")
        if "_auto_roi" not in path.name.lower())
    records = []
    if not roi_zips:
        raise ValueError("No manual RoiSet_*.zip archives found under %s" % supplement_root)

    for ordinal, roi_zip in enumerate(roi_zips, 1):
        image_path, source_stem = matching_image(roi_zip)
        rois = load_rois(str(roi_zip))
        if len(rois) < 2:
            raise ValueError("Expected worm ROI(s) followed by background ROI: %s" % roi_zip)
        image = Image.open(image_path)
        array = image_array(image)
        worm_rois = rois[:-1]
        masks = [polygon_mask(image.size, roi["points"]) for roi in worm_rois]
        labels, coverage = resolve_overlaps(masks)
        normalized = normalize_for_segmentation(array, mode="background_aware")

        issue = classify_issue(source_stem)
        group = roi_zip.parent.name
        output_stem = "boundary3__%s__g%s__%02d__%s" % (
            issue, safe_name(group), ordinal, safe_name(source_stem))
        Image.fromarray(normalized, "L").save(image_dir / (output_stem + ".png"))
        Image.fromarray(labels).save(mask_dir / (output_stem + "_masks.png"))
        label_overlay(normalized, labels).save(overlay_dir / (output_stem + "_labels.png"))

        match = re.search(r"-([12])$", source_stem.strip())
        if not match:
            raise ValueError("Supplement name must end in -1 or -2: %s" % source_stem)
        split = "training" if match.group(1) == "1" else "validation"
        records.append({
            "Stem": output_stem,
            "Split": split,
            "Source": "supplement3",
            "Issue": issue,
            "Group": group,
            "SourceImage": str(image_path),
            "SourceRoi": str(roi_zip),
            "Instances": int(labels.max()),
            "Width": image.size[0],
            "Height": image.size[1],
            "OverlapPixelsResolved": int((coverage > 1).sum()),
        })
        print("%s: %s, %s, %d instances" %
              (source_stem, split, issue, int(labels.max())))
    return records


def write_manifests(output_root, records):
    details_path = output_root / "dataset_manifest.csv"
    fields = list(records[0].keys())
    with open(details_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    split_path = output_root / "split_manifest.csv"
    with open(split_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Stem", "Split"])
        writer.writeheader()
        for record in records:
            writer.writerow({"Stem": record["Stem"], "Split": record["Split"]})
    return details_path, split_path


def main():
    parser = argparse.ArgumentParser(
        description="Build the low-clarity boundary fine-tuning dataset for 0.3.3")
    parser.add_argument("--base", required=True, help="existing normalized training dataset")
    parser.add_argument("--supplement", required=True, help="补充样例3 root")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base_root = Path(args.base).resolve()
    supplement_root = Path(args.supplement).resolve()
    output_root = Path(args.output).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("Output directory is not empty: %s" % output_root)
    image_dir = output_root / "images"
    mask_dir = output_root / "masks"
    overlay_dir = output_root / "overlays"
    for directory in (image_dir, mask_dir, overlay_dir):
        directory.mkdir(parents=True, exist_ok=True)

    records = copy_base_dataset(base_root, image_dir, mask_dir)
    records.extend(build_supplement(supplement_root, image_dir, mask_dir, overlay_dir))
    details_path, split_path = write_manifests(output_root, records)
    train_count = sum(record["Split"] == "training" for record in records)
    val_count = sum(record["Split"] == "validation" for record in records)
    print("Dataset: %s" % output_root)
    print("Records: %d training, %d validation" % (train_count, val_count))
    print("Details: %s" % details_path)
    print("Split: %s" % split_path)


if __name__ == "__main__":
    main()
