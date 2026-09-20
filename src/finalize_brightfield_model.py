"""把明场那对权重盖上产品参数与出处，写出 models/brightfield-0.1.0/{worm,tip}.pt。

与 finalize_low_model_043.py 的三点不同，都是明场特有的：

1. **一次写两个文件**。明场模型是 worm + tip 一对，缺任何一个 `run_gui_batch` 都会
   `FileNotFoundError` 拒绝启动（`batch_worm_roi.py` 里那两行），所以让它们出自同一次
   调用，不会出现「换了 worm 忘了换 tip」的半新半旧状态。
2. **归一化必须是 `legacy`**。荧光那边低清用 `background_aware`，它的判据是「左侧空条带
   的中位数就是黑电平」——那是荧光采集布局的性质。明场样例1 的左侧条带是**亮的背景**
   （实测中位数 283，全图 2% 分位只有 100），照那个黑电平归一化会把虫体和大部分背景一起
   压成 0（实测归一化后全图均值 16.8）。这里把 `legacy` 写成硬校验：传进来的 checkpoint
   若记着别的模式，直接报错而不是悄悄盖上。
3. **`normalization_mode` 与 `postprocess_*` 仍然写死**，理由与 0.4.3 一致——它们是产品
   决定，不是某次测量的结果。注意 `run_gui_batch` 目前**只从主（荧光）checkpoint 读**
   `postprocess_*`，明场这一份的值在推理时不生效；仍写进去是为了这一对的出处完整，
   以及将来真要让明场用自己那套形态学阈值时不必回头补。
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
    count_error = sum(abs(int(row["predicted_count"]) - int(row["expected_count"]))
                      for row in rows)
    return {
        "validation_count_accuracy": "%d/%d" % (correct, len(rows)),
        "validation_mean_matched_dice": dice,
        "validation_total_count_error": count_error,
    }


def save_verified(checkpoint, output):
    """写盘的最后一个动作必须是「回读确认」，序列化失败不能弄坏已在用的权重。"""
    output = os.path.abspath(output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="bf_", suffix=".pt", dir=os.path.dirname(output))
    os.close(descriptor)
    try:
        torch.save(checkpoint, temporary)
        verified = torch.load(temporary, map_location="cpu")
        if "model_state" not in verified:
            raise ValueError("Final checkpoint verification failed: %s" % output)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print("Final model: %s (%d bytes)" % (output, os.path.getsize(output)))


def main():
    parser = argparse.ArgumentParser(
        description="Finalize the brightfield pair (worm + tip) for 0.5.0")
    parser.add_argument("candidate_worm")
    parser.add_argument("candidate_tip")
    parser.add_argument("output_dir", help="例如 models/brightfield-0.1.0")
    parser.add_argument("--evaluation", required=True,
                        help="evaluate_worm_unet.py 的输出目录（留出集）")
    parser.add_argument("--holdout-checkpoint", required=True,
                        help="上面那个评测目录用的 checkpoint 文件名。随包的是 train-all "
                             "那一版，与出留出数字的 4/2 版不是同一个文件——必须写清楚，"
                             "否则这几个数字看起来像随包权重的自评")
    parser.add_argument("--training-dataset", required=True,
                        help="训练数据的一句话描述，写进 checkpoint 供以后追溯")
    parser.add_argument("--tip-val-patch-dice", type=float, required=True,
                        help="tip 精修器训练日志里的 best_val_patch_dice，明场没有独立留出，"
                             "这个数是 train-all 下的训练集自评，写清楚来路")
    parser.add_argument("--software-version", default="0.5.0")
    args = parser.parse_args()

    worm = torch.load(args.candidate_worm, map_location="cpu")
    if int(worm.get("image_size", 0)) != 768:
        raise ValueError("意外的工作分辨率 %s，明场模型与低清模型一致，一律 768"
                         % worm.get("image_size"))
    if str(worm.get("normalization_mode", "")) != "legacy":
        raise ValueError(
            "明场必须用 legacy 归一化，候选里记的是 %r。background_aware 的左侧空条带判据"
            "对亮底明场图不成立，会把虫体压成 0。" % worm.get("normalization_mode"))
    tip = torch.load(args.candidate_tip, map_location="cpu")
    if int(tip.get("image_size", 0)) != 768 or int(tip.get("in_channels", 0)) != 2:
        raise ValueError("tip 精修器的形状不对：image_size=%s in_channels=%s"
                         % (tip.get("image_size"), tip.get("in_channels")))

    metrics = evaluation_metrics(args.evaluation)
    worm.update({
        "postprocess_interior_threshold": None,
        "postprocess_erosion": 3,
        "postprocess_min_area": 0.003,
        "postprocess_min_height": 0.10,
        "postprocess_max_instances": 32,
        "normalization_mode": "legacy",
        "software_version": args.software_version,
        "training_purpose": "brightfield_dark_worm_segmentation",
        "training_dataset": args.training_dataset,
    })
    worm.update(metrics)
    worm["validation_provenance"] = (
        "留出集数字来自 %s（4 训练 / 2 留出那一版）；随包的是 train-all 六张那一版，"
        "两者同配置不同数据划分" % args.holdout_checkpoint)

    tip.update({
        "software_version": args.software_version,
        "training_purpose": "brightfield_tip_refinement",
        "training_dataset": args.training_dataset,
        "brightfield_tip_val_patch_dice_provenance":
            "train-all 下的训练集自评，非独立留出",
        "brightfield_tip_val_patch_dice": args.tip_val_patch_dice,
    })

    save_verified(worm, os.path.join(args.output_dir, "worm.pt"))
    save_verified(tip, os.path.join(args.output_dir, "tip.pt"))
    for key in sorted(metrics):
        print("  %s = %s" % (key, metrics[key]))


if __name__ == "__main__":
    main()
