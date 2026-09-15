"""给高清 0.4.2 checkpoint 盖上产品参数与出处，写出 models/0.1.1/worm.pt。

与 finalize_low_model_042.py 同构，两处不同：

① 高清的留出指标记在 `holdout_*` 前缀下，而不是 `supplement_validation_*`。
   因为随包的高清模型是 `--train-all` 出来的（`trained_on_all_records: true`，54 张全
   进了训练），checkpoint 自带的 `val_*_at_best` 是**在训练集上**算的，不能当作泛化
   能力。真正没见过的只有另跑一次的 holdout 权重见过的那 4 张 —— 因此这里读的是那次
   holdout 评测的目录，指标名前缀也换掉，免得以后有人把两者混为一谈。

② tip_boost 等训练超参来自被盖戳的那个 checkpoint 本身，不在这里重复声明。

低清之所以不这样：低清随包权重本身就是按 90/14 划出来的（`trained_on_all_records`
为 false），它的验证集确实是留出的。
"""
from __future__ import print_function

import argparse
import csv
import os
import tempfile

import torch


def evaluation_metrics(directory):
    summary_path = os.path.join(directory, "image_summary.csv")
    with open(summary_path, encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "validation"]
    if not rows:
        raise ValueError("评测目录里没有 validation 行：%s" % summary_path)
    return {
        "holdout_count_accuracy": "%d/%d" % (
            sum(row["count_correct"] == "True" for row in rows), len(rows)),
        "holdout_mean_matched_dice": sum(
            float(row["mean_matched_dice"]) for row in rows) / len(rows),
        "holdout_mean_tip_dice": sum(
            float(row["mean_tip_dice"]) for row in rows) / len(rows),
        "holdout_mean_endpoint_error": sum(
            float(row["mean_endpoint_error_fraction"]) for row in rows) / len(rows),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Finalize the 0.4.2 high-clarity checkpoint")
    parser.add_argument("candidate")
    parser.add_argument("output")
    parser.add_argument("--evaluation", required=True,
                        help="**留出权重的**评测目录（不是随包权重自己的评测）")
    parser.add_argument("--training-dataset", required=True)
    parser.add_argument("--software-version", default="0.4.2")
    args = parser.parse_args()

    checkpoint = torch.load(args.candidate, map_location="cpu")
    if int(checkpoint.get("image_size", 0)) != 768:
        raise ValueError("意外的工作分辨率 %s，高清模型一律 768"
                         % checkpoint.get("image_size"))
    if not checkpoint.get("trained_on_all_records"):
        raise ValueError("随包高清模型应当是 --train-all 出来的，"
                         "否则它自带的 val_* 就是留出指标，不该走这个脚本")
    checkpoint.update({
        "postprocess_interior_threshold": None,
        "postprocess_erosion": 2,
        "postprocess_min_area": 0.012,
        "postprocess_min_height": 0.10,
        "postprocess_max_instances": 12,
        "normalization_mode": "legacy",
        "software_version": args.software_version,
        "training_purpose": "high_clarity_worm_and_boundary_segmentation",
        "training_dataset": args.training_dataset,
    })
    checkpoint.update(evaluation_metrics(args.evaluation))

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="worm_042_", suffix=".pt", dir=os.path.dirname(output))
    os.close(descriptor)
    try:
        torch.save(checkpoint, temporary)
        verified = torch.load(temporary, map_location="cpu")
        if "model_state" not in verified or \
                verified.get("software_version") != args.software_version:
            raise ValueError("Final checkpoint verification failed")
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print("Final model:", output)
    for key in sorted(k for k in checkpoint if k.startswith("holdout_")):
        print("  %s = %s" % (key, checkpoint[key]))


if __name__ == "__main__":
    main()
