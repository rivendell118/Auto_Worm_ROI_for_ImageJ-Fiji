from __future__ import print_function

"""把一批新的人工标注并进已有训练集，产出 0.4.2 用的数据集。

与 `prepare_boundary_finetune_dataset.py` 的两点不同，都是被补充样例4 的实际文件命名逼出来的：

- 图片名的匹配放宽了。那一批的 ROI 叫 `RoiSet_0913-0-1.zip`，TIFF 却叫
  `0913-0-1 green.tif`（多了通道后缀），而 0.3.3 的匹配只认 `<stem>.tif` 与
  「去掉开头 P 再试一次」，两边都对不上。
- 归一化模式、来源标签、前缀都改成开关。高清单用 legacy、低清单用 background_aware，
  同一批图在两个档位下要产出两份不同归一化的数据集。

基座数据集自己的划分会被保留（`split_manifest.csv` 在就照抄），这样 0.3.3 那次
补充样例3 的 10 张留出图不会被降级成训练图。
"""

import argparse
import csv
import os
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from inspect_roi_dataset import image_array, load_rois, polygon_mask
from prepare_boundary_finetune_dataset import classify_issue, safe_name
from prepare_segmentation_dataset import (
    label_overlay, normalize_for_segmentation, resolve_overlaps)


def matching_image(roi_zip):
    """找到与 RoiSet_xxx.zip 对应的 TIFF，容忍 `<stem> <后缀>.tif` 这种命名。

    只认唯一命中：同名前缀下出现两张图说明这批命名有歧义，宁可直接报错，
    也不要静默挑一张去训练。
    """
    stem = roi_zip.stem
    if stem.startswith("RoiSet_"):
        stem = stem[len("RoiSet_"):]
    exact = roi_zip.parent / (stem + ".tif")
    if exact.is_file():
        return exact
    candidates = sorted(roi_zip.parent.glob(stem + "*.tif"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            "需要唯一的 TIFF，实际找到 %d 个：%s" % (len(candidates), roi_zip))
    return candidates[0]


def read_csv_rows(path):
    if not path.is_file():
        return []
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def copy_base(base_root, image_dir, mask_dir):
    """照抄基座数据集的图片与掩膜，并带回它自己的划分。"""
    image_paths = sorted((base_root / "images").glob("*.png"))
    if not image_paths:
        raise ValueError("基座数据集里没有 PNG：%s" % base_root)
    splits = {row["Stem"]: row["Split"].strip().lower()
              for row in read_csv_rows(base_root / "split_manifest.csv")}
    # 基座自己的 dataset_manifest 里记着真来源与真毛病（`!` 合并 / `=` 断裂），
    # 而 Stem 里的毛病是用文字拼的（`boundary3__merge__…`），classify_issue 只认符号，
    # 按 Stem 重算会把 0.3.3 那批的来源和毛病全记成 normal。
    details = {row["Stem"]: row
               for row in read_csv_rows(base_root / "dataset_manifest.csv")}
    rows = []
    for image_path in image_paths:
        mask_path = base_root / "masks" / (image_path.stem + "_masks.png")
        if not mask_path.is_file():
            raise FileNotFoundError("基座缺掩膜：%s" % mask_path)
        shutil.copy2(str(image_path), str(image_dir / image_path.name))
        shutil.copy2(str(mask_path), str(mask_dir / mask_path.name))
        labels = np.asarray(Image.open(mask_path), dtype=np.uint16)
        source_row = details.get(image_path.stem, {})
        rows.append({
            "Stem": image_path.stem,
            "Split": splits.get(image_path.stem, "training"),
            "Source": source_row.get("Source") or "base",
            "Issue": source_row.get("Issue") or classify_issue(image_path.stem),
            "Instances": int(labels.max()),
            "Width": int(labels.shape[1]),
            "Height": int(labels.shape[0]),
        })
    print("基座：%d 张，其中留出 %d 张"
          % (len(rows), sum(row["Split"] == "validation" for row in rows)))
    return rows


def build_supplement(supplement_root, prefix, source, normalization, image_dir, mask_dir,
                     overlay_dir):
    roi_zips = sorted(path for path in supplement_root.rglob("RoiSet_*.zip")
                      if "_auto_roi" not in str(path).lower())
    if not roi_zips:
        raise ValueError("补充集里没有人工 RoiSet_*.zip：%s" % supplement_root)
    rows = []
    for roi_zip in roi_zips:
        image_path = matching_image(roi_zip)
        rois = load_rois(str(roi_zip))
        if len(rois) < 2:
            raise ValueError("ROI 应有虫体若干 + 末尾背景区：%s" % roi_zip)
        image = Image.open(image_path)
        array = image_array(image)
        masks = [polygon_mask(image.size, roi["points"]) for roi in rois[:-1]]
        labels, coverage = resolve_overlaps(masks)
        normalized = normalize_for_segmentation(array, mode=normalization)

        source_stem = roi_zip.stem[len("RoiSet_"):]
        match = re.search(r"-([12])$", source_stem.strip())
        if not match:
            raise ValueError("补充集文件名应以 -1 或 -2 结尾：%s" % source_stem)
        stem = "%s__%s" % (prefix, safe_name(source_stem))
        Image.fromarray(normalized, "L").save(str(image_dir / (stem + ".png")))
        Image.fromarray(labels).save(str(mask_dir / (stem + "_masks.png")))
        label_overlay(normalized, labels).save(str(overlay_dir / (stem + "_labels.png")))

        rows.append({
            "Stem": stem,
            "Split": "training" if match.group(1) == "1" else "validation",
            "Source": source,
            "Issue": classify_issue(source_stem),
            "Instances": int(labels.max()),
            "Width": int(image.size[0]),
            "Height": int(image.size[1]),
            "OverlapPixelsResolved": int((coverage > 1).sum()),
        })
        print("  %-28s %-10s %d 条虫  %s" %
              (source_stem, rows[-1]["Split"], rows[-1]["Instances"], image_path.name))
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Merge a supplementary annotation folder into an existing dataset")
    parser.add_argument("base", help="已有 images/ + masks/ 的数据集")
    parser.add_argument("supplement", help="补充样例根目录（含 RoiSet_*.zip 与 TIFF）")
    parser.add_argument("output")
    parser.add_argument("--prefix", default="supp", help="补充样本在 Stem 里的前缀")
    parser.add_argument("--source", default="supplement", help="manifest 里的来源标签")
    parser.add_argument("--normalization", choices=("legacy", "background_aware"),
                        default="legacy")
    args = parser.parse_args()

    base_root = Path(args.base).resolve()
    supplement_root = Path(args.supplement).resolve()
    output_root = Path(args.output).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("输出目录非空：%s" % output_root)
    image_dir, mask_dir = output_root / "images", output_root / "masks"
    overlay_dir = output_root / "overlays"
    for directory in (image_dir, mask_dir, overlay_dir):
        directory.mkdir(parents=True, exist_ok=True)

    records = copy_base(base_root, image_dir, mask_dir)
    records.extend(build_supplement(supplement_root, args.prefix, args.source,
                                    args.normalization, image_dir, mask_dir, overlay_dir))

    fields = ["Stem", "Split", "Source", "Issue", "Instances", "Width", "Height",
              "OverlapPixelsResolved"]
    for row in records:
        row.setdefault("OverlapPixelsResolved", 0)
    write_csv_rows(output_root / "dataset_manifest.csv", fields, records)
    write_csv_rows(output_root / "split_manifest.csv", ["Stem", "Split"],
                   [{"Stem": row["Stem"], "Split": row["Split"]} for row in records])

    train_count = sum(row["Split"] == "training" for row in records)
    val_count = len(records) - train_count
    print("数据集：%s" % output_root)
    print("共 %d 张：训练 %d、留出 %d" % (len(records), train_count, val_count))


if __name__ == "__main__":
    main()
