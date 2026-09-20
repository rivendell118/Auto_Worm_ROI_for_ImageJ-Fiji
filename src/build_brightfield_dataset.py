from __future__ import print_function

"""建明场模型的数据集：**只有明场图，不并任何荧光基座**。

与 `build_supplement_dataset.py` 的关系：那个脚本的第一个参数是**基座数据集**，它先把
基座的 image/mask 原样抄进新数据集，再把新标注补进去。明场与荧光不同域——明场是亮底
黑虫（样例1 的像素均值 228~271，虫体是最暗的一小撮），荧光是暗底亮虫。把两批图并进
同一个数据集，只会让模型同时学两套相反的对比度先验，所以这里**不抄基座**。

其余全部复用 `build_supplement_dataset` 里已经写好、有测试覆盖的部分：

- `matching_image()` 容忍 `<stem> <后缀>.tif` 的命名——明场样例1 正是这个形状
  （ROI 叫 `RoiSet_l4440-ctr-1 lay2 n=15.zip`，图叫 `l4440-ctr-1 lay2 n=15.tif`）。
- `resolve_overlaps()` 处理手工多边形在虫体相邻边缘的少量重叠。
- manifest 的列与 `split_manifest.csv` 的形状。

三个与荧光那边**故意不同**的地方，都在下面标了「明场专用」：

1. 归一化固定 `legacy`。`background_aware` 的判据是「左侧有一条空条带，取它的中位数
   当黑电平」——那是荧光采集布局的性质。明场的左侧条带是**亮的背景**（实测中位数
   283，而全图 2% 分位是 100），拿它当黑电平会把虫体和大部分背景一起压成 0：
   实测 `background_aware` 归一化后全图均值 16.8、98% 分位 157，虫体信息基本没了；
   `legacy` 下 2% 分位 0、98% 分位 225，虫体仍然是一块可分的暗区。
2. 划分不按文件名后缀猜。`supplement_split_for()` 的判据是文件名以 `-1`/`-2` 结尾，
   明场的文件以 `n=15` 这种形状结尾，对不上。这里改成显式点名留出图（`--holdout`），
   没点名的一律进训练集，与 `--supplement-split training` 同一条路。
3. `Issue` 列与 `--issue` 开关挂钩，并且**不再让 `classify_issue()` 说话**。理由见
   `normalise_issue()` 的注释。

用法：

    <venv python> src/build_brightfield_dataset.py "<明场样例1 目录>" training/brightfield_v050 \\
        --holdout "l4440-ctr-3 lay2 n=16" --holdout "l4440-ctr-6 lay2 n=11" --issue merge
"""

import argparse
from pathlib import Path

from build_supplement_dataset import DATASET_MANIFEST_FIELDS, build_supplement, \
    stem_for, write_csv_rows


def normalise_issue(rows, issue):
    """把 manifest 的 `Issue` 列改成训练侧**真正会用到**的值。

    `build_supplement_dataset` 用 `classify_issue()` 按文件名里的 `!` / `=` 判毛病，
    明场的文件名带一个 `=`（`n=15`），于是六张会**全被记成 `split`**——这正是 0.4.3 那轮
    补充样例5 踩过的坑：清单里那一列看着有值，实际没接线。训练脚本的 `issue_from_stem()`
    只认 `boundary3__` 开头的 Stem，所以清单里该记的就是「Stem 里真的带了什么」：
    给了 `--issue` 就是它，没给就是 `normal`。
    """
    for row in rows:
        row["Issue"] = issue or "normal"
    return rows


def apply_holdout(rows, holdout_names, prefix, issue, supplement_root):
    """把点名的源图挪到留出集；其余留作训练集。

    `build_supplement()` 出来的行全在训练集里（等价于 `--supplement-split training`）。
    点名用的是**源文件名**（`RoiSet_` 后面那一段），在写进 manifest 前先换算成
    训练脚本认得的 Stem，免得两处各写一遍命名规则而对不上。
    """
    if not holdout_names:
        raise ValueError("必须显式点名留出图：明场的文件名没有 -1/-2 后缀，"
                         "不能靠补充集那套规则自动划分")
    wanted = {stem_for(name, prefix, issue): name for name in holdout_names}
    known = {row["Stem"] for row in rows}
    missing = sorted(name for stem, name in wanted.items() if stem not in known)
    if missing:
        raise ValueError("留出图不在这批里：%s；目录 %s" % (", ".join(missing), supplement_root))
    for row in rows:
        row["Split"] = "validation" if row["Stem"] in wanted else "training"
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Build a brightfield-only dataset from one annotated sample folder")
    parser.add_argument("supplement", help="含 RoiSet_*.zip 与明场 TIFF 的目录")
    parser.add_argument("output")
    parser.add_argument("--prefix", default="bf", help="Stem 前缀")
    parser.add_argument("--source", default="brightfield", help="manifest 里的来源标签")
    parser.add_argument("--holdout", action="append", default=[],
                        help="留出图的源文件名（RoiSet_ 后面那一段），可重复给；"
                             "没点名的一律进训练集")
    parser.add_argument("--issue", choices=("", "merge", "split", "merge_split"),
                        default="",
                        help="训练侧的毛病标签。非空时 Stem 写成 boundary3__<标签>__…，"
                             "训练脚本的 --hard-repeat / --contact-boost / "
                             "--continuity-boost 才会对这批图生效；留空则一律 normal、"
                             "那些开关全是空转")
    args = parser.parse_args()
    if not args.holdout:
        parser.error("至少要一张留出图（--holdout）：没有留出集就得不到任何可比的数字")

    # 明场专用：固定 legacy。理由见模块开头的第 1 条。
    normalization = "legacy"
    issue = args.issue
    supplement_root = Path(args.supplement).resolve()
    output_root = Path(args.output).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("输出目录非空：%s" % output_root)
    image_dir, mask_dir = output_root / "images", output_root / "masks"
    overlay_dir = output_root / "overlays"
    for directory in (image_dir, mask_dir, overlay_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows = build_supplement(supplement_root, args.prefix, args.source, normalization,
                            image_dir, mask_dir, overlay_dir,
                            supplement_split="training", issue=issue)
    normalise_issue(rows, issue)
    apply_holdout(rows, args.holdout, args.prefix, issue, supplement_root)

    for row in rows:
        row.setdefault("OverlapPixelsResolved", 0)
    write_csv_rows(output_root / "dataset_manifest.csv",
                   DATASET_MANIFEST_FIELDS, rows)
    write_csv_rows(output_root / "split_manifest.csv", ["Stem", "Split"],
                   [{"Stem": row["Stem"], "Split": row["Split"]} for row in rows])

    train_count = sum(row["Split"] == "training" for row in rows)
    print("数据集：%s" % output_root)
    print("归一化：%s（明场固定值）" % normalization)
    print("共 %d 张：训练 %d、留出 %d" % (len(rows), train_count, len(rows) - train_count))


if __name__ == "__main__":
    main()
