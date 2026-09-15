from __future__ import print_function

import argparse
import csv
import os
import re
import shutil

from train_worm_unet import discover_records


def read_splits(path):
    result = {}
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            result[row["Stem"]] = row["Split"].strip().lower()
    return result


def new_data_split(stem, count):
    if count == 20:
        # There is no paired -2 replicate for n=20. Keep one image untouched.
        return "validation" if "0101-50-1" in stem else "training"
    return "validation" if re.search(r"-2(?:-|\s|$)", stem) else "training"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dataset")
    parser.add_argument("--old-manifest", required=True)
    parser.add_argument(
        "--source", action="append", nargs=3, required=True,
        metavar=("PREFIX", "DATASET", "SPLIT_RULE"),
        help="SPLIT_RULE: old_manifest or new_replicates")
    args = parser.parse_args()

    if os.path.exists(args.output_dataset) and os.listdir(args.output_dataset):
        raise SystemExit("Output dataset already exists and is not empty: " + args.output_dataset)
    image_dir = os.path.join(args.output_dataset, "images")
    mask_dir = os.path.join(args.output_dataset, "masks")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    old_splits = read_splits(args.old_manifest)
    rows, seen = [], set()

    for prefix, dataset, rule in args.source:
        if rule not in ("old_manifest", "new_replicates"):
            raise ValueError("Unknown split rule: " + rule)
        for image_path, mask_path, count, stem in discover_records(dataset):
            output_stem = prefix + "__" + stem
            if output_stem in seen:
                raise ValueError("Duplicate output stem: " + output_stem)
            seen.add(output_stem)
            if rule == "old_manifest":
                if output_stem not in old_splits:
                    raise ValueError("Missing old split assignment: " + output_stem)
                split = old_splits[output_stem]
            else:
                split = new_data_split(stem, count)
            image_name = output_stem + ".png"
            mask_name = output_stem + "_masks.png"
            shutil.copy2(image_path, os.path.join(image_dir, image_name))
            shutil.copy2(mask_path, os.path.join(mask_dir, mask_name))
            rows.append({
                "Stem": output_stem,
                "Image": image_name,
                "Mask": mask_name,
                "ExpectedCount": count,
                "PrimaryN10": count == 10,
                "Split": split,
                "Source": prefix,
            })

    manifest = os.path.join(args.output_dataset, "manifest.csv")
    with open(manifest, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    distribution = {}
    for row in rows:
        key = (row["Split"], int(row["ExpectedCount"]))
        distribution[key] = distribution.get(key, 0) + 1
    print("Images:", len(rows))
    print("Split/count distribution:", distribution)
    print("Manifest:", manifest)


if __name__ == "__main__":
    main()
