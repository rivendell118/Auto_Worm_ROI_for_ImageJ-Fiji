from __future__ import print_function

import argparse
import csv
import glob
import os
import re
import shutil

import numpy as np
from PIL import Image

from train_worm_unet import discover_records, split_records


def copy_record(record, prefix, destination_images, destination_masks, split, source, rows):
    image_path, mask_path, count, stem = record
    output_stem = prefix + "__" + stem
    image_name = output_stem + ".png"
    mask_name = output_stem + "_masks.png"
    shutil.copy2(image_path, os.path.join(destination_images, image_name))
    shutil.copy2(mask_path, os.path.join(destination_masks, mask_name))
    rows.append({
        "Stem": output_stem,
        "Image": image_name,
        "Mask": mask_name,
        "ExpectedCount": count,
        "PrimaryN10": count == 10,
        "Split": split,
        "Source": source,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("old_dataset")
    parser.add_argument("supplemental_dataset")
    parser.add_argument("output_dataset")
    args = parser.parse_args()
    output_images = os.path.join(args.output_dataset, "images")
    output_masks = os.path.join(args.output_dataset, "masks")
    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_masks, exist_ok=True)

    rows = []
    old_records = discover_records(args.old_dataset)
    old_training, old_validation = split_records(old_records)
    old_validation_stems = {record[3] for record in old_validation}
    for record in old_records:
        split = "validation" if record[3] in old_validation_stems else "training"
        copy_record(record, "base", output_images, output_masks, split, "existing", rows)

    supplemental_records = discover_records(args.supplemental_dataset)
    for record in supplemental_records:
        # Experimental replicate '-2' is held out; paired '-1' remains in training.
        # This is stricter than a random image split while retaining every imaging family.
        split = "validation" if re.search(r"-2(?:\s|$)", record[3]) else "training"
        copy_record(record, "supp", output_images, output_masks, split, "supplemental", rows)

    manifest_path = os.path.join(args.output_dataset, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    counts = {}
    splits = {}
    for row in rows:
        counts[int(row["ExpectedCount"])] = counts.get(int(row["ExpectedCount"]), 0) + 1
        key = (row["Split"], int(row["ExpectedCount"]))
        splits[key] = splits.get(key, 0) + 1
    print("Images:", len(rows))
    print("Count distribution:", counts)
    print("Split distribution:", splits)
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
