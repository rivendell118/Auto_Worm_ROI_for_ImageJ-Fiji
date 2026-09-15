"""给低清 0.4.2 checkpoint 盖上产品参数与出处，写出 models/0.2.1/worm.pt。

与 finalize_low_model_033.py 的唯一区别：验证指标**从评测目录里读**，不在源码里写死。
0.3.3 那份把 `supplement_validation_count_accuracy` 等数字硬编码在脚本里，数字来路只能
靠人记住；这里改成 `--evaluation` 指向 evaluate_worm_unet.py 的输出目录，从
image_summary.csv 现算，checkpoint 里记的数字必然出自那次评测。

postprocess_* 与 normalization_mode 仍然是写死的：它们不是测量结果，是**产品决定**，
必须与推理侧的默认值一致，不该随某次评测浮动。
"""
from __future__ import print_function

import argparse
import csv
import os
import tempfile

import torch


def evaluation_metrics(directory):
    """从 evaluate_worm_unet.py 的输出目录里读出留出集的汇总指标。"""
    summary_path = os.path.join(directory, "image_summary.csv")
    with open(summary_path, encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "validation"]
    if not rows:
        raise ValueError("评测目录里没有 validation 行：%s" % summary_path)
    correct = sum(row["count_correct"] == "True" for row in rows)
    dice = sum(float(row["mean_matched_dice"]) for row in rows) / len(rows)
    return {
        "supplement_validation_count_accuracy": "%d/%d" % (correct, len(rows)),
        "supplement_validation_mean_matched_dice": dice,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Finalize the 0.4.2 low-clarity checkpoint")
    parser.add_argument("candidate")
    parser.add_argument("output")
    parser.add_argument("--evaluation", required=True,
                        help="evaluate_worm_unet.py 的输出目录（留出集）")
    parser.add_argument("--training-dataset", required=True,
                        help="训练数据的一句话描述，写进 checkpoint 供以后追溯")
    parser.add_argument("--software-version", default="0.4.2")
    args = parser.parse_args()

    checkpoint = torch.load(args.candidate, map_location="cpu")
    if int(checkpoint.get("image_size", 0)) != 768:
        raise ValueError("意外的工作分辨率 %s，2026-09-15 起低清模型一律 768"
                         % checkpoint.get("image_size"))
    metrics = evaluation_metrics(args.evaluation)
    checkpoint.update({
        "postprocess_interior_threshold": None,
        "postprocess_erosion": 3,
        "postprocess_min_area": 0.003,
        "postprocess_min_height": 0.10,
        "postprocess_max_instances": 12,
        "normalization_mode": "background_aware",
        "software_version": args.software_version,
        "training_purpose": "low_clarity_adjacent_boundary_recognition",
        "training_dataset": args.training_dataset,
    })
    checkpoint.update(metrics)

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="worm_042_", suffix=".pt", dir=os.path.dirname(output))
    os.close(descriptor)
    try:
        torch.save(checkpoint, temporary)
        # 回读确认再替换：序列化失败也不能把已经在用的模型弄坏。
        verified = torch.load(temporary, map_location="cpu")
        if "model_state" not in verified or \
                verified.get("software_version") != args.software_version:
            raise ValueError("Final checkpoint verification failed")
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print("Final model:", output)
    for key in sorted(metrics):
        print("  %s = %s" % (key, metrics[key]))


if __name__ == "__main__":
    main()
