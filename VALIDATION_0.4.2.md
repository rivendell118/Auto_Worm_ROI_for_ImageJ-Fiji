# 自动圈虫 0.4.2 CUDA for ImageJ 验证记录

验证日期：2026-09-16

0.4.2 有两件事。一是**模型更新**：换掉随包的四个权重文件，逻辑代码与 0.4.1 相同，验证的
重点是模型——新旧权重在**留出图**和**真实批次**上的对照（本文档第 1～6 节）。二是**输出
目录结构与测量表列序的改动**：输出目录分成 `measurements/` 与 `other/`，测量表的七个数值
列移到最前（第 7 节）。任何一处结论都是实际跑出来的，命令与原始输出一并写下。

## 结果

- Python 回归测试：**140/140 通过**，8 个模块全部 `OK`（本机无 pytest，逐文件直接运行）：

  | 模块 | 用例数 |
  |---|---|
  | `test_boundary_training_033.py` | 4 |
  | `test_imagej_integration.py` | 60 |
  | `test_low_clarity_splitter.py` | 6 |
  | `test_make_release.py` | 17 |
  | `test_manual_exclusion.py` | 18 |
  | `test_manual_head_annotation.py` | 25 |
  | `test_review_policy_033.py` | 3 |
  | `test_worm_segment_selector.py` | 7 |

- 高清留出集（4 张，模型没见过）：虫数全对由 **0/4 升到 4/4**，平均 Dice 0.6719 → 0.9261。
- 低清留出集（14 张）：虫数全对 9/14 → **10/14**，平均 Dice 0.8705 → 0.8877。**有 2 张变差**，
  逐张列在下面，没有隐去。
- 端到端回归（46 张真实图，走完整条流水线）：**6 张虫数变化，六张全是改对，零变差**。
- 低清指纹判据实测：新旧两版低清权重都判为低清，两版高清权重都不命中。
- 权重自带的验证指标与训练日志逐一核对一致；`image_size` 均为 768、`boundary_radius` 均为 6。
- Java 回归测试：`tests\AutoWormMeasurementTest.java` **14/14 通过**（0.4.1 时 12 条，本次
  新增「列序」与「测量文件夹不存在时自动建好」两条）。
- 发布文件已重建：`dist/Auto_Worm_ROI.jar`、`dist/AutoWormImageJ/AutoWormGUI.exe`，
  `dist/AutoWormImageJ/models/` 与仓库 `models/` 逐文件 SHA256 相同。

## 训练数据的构成与留出划分

来自两个数据集的 `split_manifest.csv`，不是推算：

| 数据集 | 记录数 | 划分 | 来源前缀分布 |
|---|---|---|---|
| `training/high_v041` | 54 | 训练 50 / 留出 4 | `base` 12、`supp` 34、`supp4` 8 |
| `training/low_v041` | 104 | 训练 90 / 留出 14 | `base` 12、`boundary3` 20、`new` 30、`supp` 34、`supp4` 8 |

`images/` 与 `masks/` 目录里的文件数与清单条数逐一对齐（54/54、104/104）。

留出的是 `supp4__0913-*-2` 这 4 张（高清、低清共用同一批），以及低清另外 10 张
`boundary3__*-2`。这 4 张就是用户报「虫数不对」的那批 0913 采集的另一种子。

## 留出集对照（命令与原始数字）

评测脚本 `eval042/eval_042.py`，对每份权重跑：

```
python -X utf8 src/evaluate_worm_unet.py <dataset> \
    --checkpoint <权重> \
    --split-manifest <dataset>/split_manifest.csv \
    --only-split validation \
    --output <输出目录> \
    --erosion <N> --min-area <X> --max-instances 12
```

> 后处理参数**必须显式传**：`evaluate_worm_unet.py` **不读** checkpoint 里的
> `postprocess_*`，它的默认值是 `erosion=0 / min_area=0.005`，与出模型时的设置不符。
> 高清用 `--erosion 2 --min-area 0.012`，低清用 `--erosion 3 --min-area 0.003`，两者都取自
> 各自 checkpoint 里的实际值。

### 高清 · 4 张留出图

| 图 | 旧 `0.1.0` 虫数 | 新 虫数 | 旧 Dice | 新 Dice |
|---|---|---|---|---|
| `supp4__0913-0-2` | 7 | **10** | 0.5623 | 0.9254 |
| `supp4__0913-10-2` | 8 | **10** | 0.6679 | 0.9214 |
| `supp4__0913-100-2` | 9 | **10** | 0.7852 | 0.9289 |
| `supp4__0913-1k-2` | 8 | **10** | 0.6721 | 0.9286 |
| **虫数全对** | **0/4** | **4/4** | | |
| 平均 Dice | | | 0.6719 | **0.9261** |
| 平均 tip Dice | | | 0.7623 | **0.8662** |
| 平均归一化端点误差 | | | 0.0315 | **0.0194** |

旧模型 4 张的 `fraction_dice_ge_0.5` 分别是 0.7 / 0.8 / 0.9 / 0.8，`minimum_matched_dice`
全部为 **0.0**——即每张都有一条虫完全没圈到。新模型 4 张全部为 1.0，最小值 0.88~0.91。

新权重还另跑了一遍**留出权重**（`worm_unet_high041_holdout`，50 训练 / 4 留出）用于对照：
虫数同样 4/4，平均 Dice 0.9261，与随包的 train-all 权重（0.9307）相差 0.005 以内。

### 低清 · 14 张留出图

| | 旧 `0.2.0` | 新 `0.2.1` |
|---|---|---|
| 虫数全对 | 9/14 | **10/14** |
| 平均 Dice | 0.8705 | **0.8877** |
| 平均 tip Dice | 0.7857 | **0.8110** |
| 平均归一化端点误差 | 0.0395 | **0.0335** |

逐张（只列虫数有变化的 5 张，其余 9 张虫数与判定完全不变）：

| 图 | 旧 | 新 | 旧 Dice | 新 Dice |
|---|---|---|---|---|
| `boundary3__merge__g4__14__hsp4_020901_wdr5-2` | 10 ✓ | 9 | 0.8367 | 0.7259 |
| `boundary3__normal__g1__04__hsp4_250605_wdr5-2` | 11 | 10 ✓ | 0.8696 | 0.8841 |
| `boundary3__normal__g3__12__hsp4_250619_wdr5-2` | 10 ✓ | 11 | 0.8868 | 0.8985 |
| `supp4__0913-10-2` | 9 | 10 ✓ | 0.7892 | 0.9115 |
| `supp4__0913-100-2` | 9 | 10 ✓ | 0.7921 | 0.9143 |

**变差的两张（`g4__14`、`g3__12`）都是 `boundary3`，也就是旧低清模型的训练域；变好的三张
里两张来自新加入的 `supp4`。** 把新域加进训练集，代价是在旧域上让出一部分，这一点如实
记在这里：低清这一版的留出表现是「净 +1」，不是全面提升。

一个附带的观察：`g4__14` 变差后 Dice 掉到 0.7259、`minimum_matched_dice` 为 0.0，说明少的
那一条是被并进了相邻的虫（不是漏检）——与 `merge` 这一类的毛病相符，也和旧模型在同一张图
上 tip Dice 只有 0.583 相印证。未做进一步处理。

## 端到端回归（真实批次，新旧各跑一遍）

比 Dice 更能说明问题的是**端到端虫数**——用户报的就是虫数。脚本 `regress041/regress_041.py`
把三批真实图分别用新旧模型跑完整的 `run_gui_batch`（形状细化、人工头向标注、Python 测量
后端全开），逐张对照 `batch_summary.csv` 的 `worm_count` / `qc_status`。

| 批次 | 张数 | 模型档 | 虫数有变化的图 |
|---|---|---|---|
| 0913（报 bug 的那批，带人工头向标注） | 8 | 高清 | 0/8 |
| 0913 | 8 | 低清 | **2/8，全部变好** |
| 样例1（0716） | 8 | 高清 | **4/8，全部变好** |
| 补充样例2 | 30 | 低清 | 0/30 |

**合计 46 张，6 张有变化，六张全是改对，没有一张变差。** 逐张：

```
A_0913 高清   8 张全部不变，其中 0913-1k-2 两版都是 11 条 REVIEW_COUNT_MISMATCH
A_0913 低清   0913-10-2   旧 9 REVIEW_COUNT_MISMATCH -> 新 10 PASS
              0913-100-2  旧 9 REVIEW_COUNT_MISMATCH -> 新 10 PASS
              其余 6 张不变（含 0913-1k-2 两版都是 11 条 REVIEW_COUNT_MISMATCH）
B_样例1 高清  0716-0-2    旧 9 REVIEW_COUNT_MISMATCH -> 新 10 PASS
              0716-10-1   旧 9 REVIEW_COUNT_MISMATCH -> 新 10 PASS
              0716-10-2   旧 9 REVIEW_COUNT_MISMATCH -> 新 10 PASS
              0716-1k-2   旧 9 REVIEW_COUNT_MISMATCH -> 新 10 PASS
              其余 4 张不变
C_补充样例2 低清  30 张全部不变（其中 2 张 n=20 的图两版都是 12 条 REVIEW_COUNT_MISMATCH）
```

样例1（0716）那四张值得单独说：它们是高清模型**原本的训练域**。旧模型在这里数出 9 条被判
`REVIEW_COUNT_MISMATCH`，新模型四张全部改成 10 条 `PASS`——说明这次微调没有只顾新域而丢掉
老本行。这一点是选新权重而非只加新数据重训的主要依据。

`0913-1k-2` 在**两个模型、两个档位下都是 11 条**，是 0.4.1 就存在的既有现象，本次既没有
改好也没有改坏；`补充样例2` 里那 2 张 n=20 的图同理（预期 20 条、两版都数出 12 条，
属另一类问题）。这两处都不在本次范围内，列出来是为了说明「零变差」不等于「全对」。

## 两个 tip 精修器

整虫模型换了，精修器的训练输入（guide）必须跟着重出，否则它学的是**旧模型犯的错**。
`tip_042.py` 先用新整虫模型对各自数据集的全量记录跑 `evaluate_worm_unet.py`（不加
`--only-split`，因此训练集和验证集都有），断言每张都产出了 `_predicted_instances.png`
（精修器的 `TipDataset` 就是按这个名字找 guide 的），再以它为输入训练。

超参不传任何值，用脚本默认（epochs 50 / batch 16 / lr 3e-4 / patch 192 /
`probability_threshold` 0.40 / `replace_fraction` 0.14），与随包 tip 的元数据逐项相同；
只传 `--train-all --initial-checkpoint --output-name`。

| | 旧 | 新 | 日志末行 |
|---|---|---|---|
| 高清 tip | 0.9247 | **0.9380** | `Best validation patch Dice 0.9380` |
| 低清 tip | 0.9266 | **0.9347** | `Best validation patch Dice 0.9347` |

两个都是 50 轮跑满，过程中 `val_patch_dice` 单调爬升、末轮略回落，取的是最优点。高清的
起点是 0.9192（第 1 轮）、低清是 0.9113，都低于各自的最终值。

## 权重元数据核对

逐键读出新旧四个权重（`0.4.1` 的树对比 `0.4.2` 的树），确认没有「改了一个没注意到的数」：

| 键 | 旧 `0.1.0` | 新 `0.1.1` | 旧 `0.2.0` | 新 `0.2.1` |
|---|---|---|---|---|
| `image_size` | 768 | 768 | 768 | 768 |
| `boundary_radius` | 6 | 6 | 6 | 6 |
| `normalization_mode` | （缺） | `legacy` | `background_aware` | `background_aware` |
| `postprocess_erosion` | 2 | 2 | 3 | 3 |
| `postprocess_min_area` | 0.012 | 0.012 | 0.003 | 0.003 |
| `postprocess_min_height` | 0.1 | 0.1 | 0.1 | 0.1 |
| `postprocess_max_instances` | （缺） | 12 | 12 | 12 |
| `standard_count` / `allowed_counts` | 10 / [9,10,11] | 同 | — | — |
| `tip_boost` / `tip_fraction` | 4.0 / 0.18 | 4.0 / 0.18 | 1.0 / 0.18 | 1.0 / 0.18 |
| `hard_repeat` / `contact_boost` / `continuity_boost` | （缺） | 1 / 1.0 / 1.0 | 4 / 3.0 / 1.75 | 4 / 3.0 / 1.75 |
| `base` | 16 | 16 | 16 | 16 |
| `best_selection_score` | 0.8845 | 0.8866 | 0.7940 | 0.8046 |

三处需要说明，都是**无行为差异**的：

- 旧 `0.1.0` 里没有 `postprocess_max_instances`。推理侧
  `batch_worm_roi.py:1186-1187` 读的是 `int(checkpoint.get("postprocess_max_instances", 12))`
  ——**缺省值就是 12**，新权重显式写上的也是 12，取值不变。
- 新 `0.1.1` 里多了 `hard_repeat=1` / `contact_boost=1.0` / `continuity_boost=1.0`。这三个是
  低清档专用（`low_clarity_split` 为真时才生效），高清档不读；且 1 / 1.0 / 1.0 本身就是
  「不增强」的取值。是训练脚本把默认值一并写进 checkpoint 的结果。
- 新 `0.1.1` 的 `val_*_at_best`（0.9686 / 0.9141 / 0.8028 / 0.9428）是 **train-all 在训练集
  上**算的，比旧的 0.9686 / 0.9110 / 0.7992 / 0.9425 略高属正常，**不能当泛化能力读**。真正
  可比的数字是上面的留出集表，因此 `finalize_high_model_042.py` 把留出指标另起
  `holdout_*` 前缀写入，两个来源在权重里不会混。

### 训练日志（选点依据）

```
高清 train-all   Best validation selection score 0.8866
                 epoch 20/40 val_fg=0.9686 val_interior=0.9141 val_boundary=0.8028
                             val_tip=0.9428 score=0.8866   （第 20 轮即最优点，跑满 40 轮）
                 54 unique, 0 marked, 54 effective

低清 768 重跑    Best validation selection score 0.8046（旧 0.2.0 为 0.7940）
                 epoch 30/40 val_fg=0.9486 val_interior=0.8825 val_boundary=0.7586
                             val_tip=0.9486 score=0.8019
                 90 unique, 5 marked, 105 effective
```

低清第一次启动时漏了 `--image-size 768`，checkpoint 记成了 512 / `boundary_radius=4`。
**那一版没有进入任何评测，也没有打包**，发现后删除输出目录、找到进程
（`wmic process where "name='python.exe'" get CommandLine` → `taskkill /PID <pid> /T /F`）
并带 `--image-size 768 --tip-boost 1.0` 重跑。重跑后读回的 `image_size: 768`、
`boundary_radius: 6` 与旧 `0.2.0` 一致，即上表。之所以要盯这个数：推理侧
`batch_worm_roi.py:1176-1177` 是从 checkpoint 读 `image_size` 的，512 会静默改变所有低清用户
的工作分辨率。`finalize_low_model_042.py` 现在把 `image_size != 768` 当作致命错误直接退出。

## 低清指纹判据

`batch_worm_roi.py` 的 `LOW_CLARITY_MODEL_SHA256S` 实测（导入的是本仓库 `src/`，不是别处）：

```
常数： ('afb80dc4bde0cc644ab4269bfc73463d7f20eace9592a21857ce155b8f4eed4b',
        '5ce04d283a8dbc4e7b8ae460f6310627ecdfbef5feca23ace7bdbe51ef84ff3a')
SOFTWARE_VERSION = 0.4.2-beta-imagej

0.4.2 新低清 models/0.2.1/worm.pt   afb80dc4bde0cc64...  判定为低清 = True
0.4.1 旧低清 models/0.2.0/worm.pt   5ce04d283a8dbc4e...  判定为低清 = True
0.4.2 新高清 models/0.1.1/worm.pt   6315cfaf21594c93...  判定为低清 = False
0.4.1 旧高清 models/0.1.0/worm.pt   f4ac790f963cd0c7...  判定为低清 = False
```

四个方向全部符合预期：两版低清都认得出，两版高清都不命中。这条只在**命令行**路径上生效
（界面每次都把 `low_clarity_split` 显式传进来），因此它要防的是「拿 0.4.1 散出去的低清权重
走命令行，被静默当成高清」。

## 权重出处字段

新权重里原本记的是开发机绝对路径，已改写为仓库内相对路径。改写脚本只动
`initialized_from` 一个键，写回前回读校验、失败不替换。实测结果：

```
0.1.1/worm.pt   'models/0.1.0/worm.pt'
0.1.1/tip.pt    'models/0.1.0/tip.pt'
0.2.1/worm.pt   'models/0.2.0/worm.pt'
0.2.1/tip.pt    'models/0.2.0/tip.pt'
```

注意这四条指向的是**仓库里已经不再存在的目录**（0.4.2 的树里只有 `0.1.1` / `0.2.1`）。
这是有意的：它是「从哪一版权重微调来的」的历史指针，不是运行时路径，代码里没有任何地方
读它。改写**改变了文件的字节**，因此 `models/0.2.1/worm.pt` 的 SHA256 是在改写之后重新算的
——先算后改会让 `LOW_CLARITY_MODEL_SHA256S` 里那个值对不上，实测也确实对不上过
（改写前 `041c5cdd…`，改写后 `afb80dc4…`）。

## 输出目录结构与测量表列序

改动本身见 `CHANGELOG_0.4.2.md`；这一节只记验证。

**Python 侧 140/140 全过**（上表列出的 8 个模块，逐文件直接运行，本机无 pytest）。
其中 `test_imagej_integration.py` 的 60 条里，路径断言全部改成新布局，并新增了「ROW
按列名取、表头重排不会写错列」的用例。

**Java 侧 14/14 全过**：

```
javac --release 8 -encoding UTF-8 -cp "lib\ij.jar" -d build\plugin_classes ^
    plugin_src\Auto_Worm_ROI.java plugin_src\Auto_Worm_Annotations.java
javac --release 8 -encoding UTF-8 -cp "lib\ij.jar;build\plugin_classes" ^
    -d build\measurement_test_classes tests\AutoWormMeasurementTest.java
java -cp "build\measurement_test_classes;lib\ij.jar;build\plugin_classes" ^
    AutoWormMeasurementTest
→ AUTOWORM_MEASUREMENT_OK
```

> 类路径上要放**类目录**而不是 `dist\AutoWorm_ROI.jar`：插件在**无名包**里，javac 不会
> 从 jar 里读无名包的类型，用 jar 时每处 `Auto_Worm_ROI.xxx` 都报「找不到符号」（22 条），
> 看着像插件没编出来。运行时两种都可以。这一点已写进测试文件头注释。

新增的两条：`theNumbersThatAreReadFirstComeFirst`（两张表的表头都以
`Area,Mean,Min,Max,IntDen,Median,RawIntDen,` 开头，且每一行的列数与表头一致、
第 9 列是图名——行与表头错位会被抓住，因为写错列的 CSV 仍然是格式正确的 CSV）；
`theMeasurementFolderIsMadeIfItIsNotThere`（把 `measurements\` 删掉再测量，
`saveTable` 自行建目录并把表写进去）。

**真实数据的端到端**（`0513-0-1.tif`，10 条虫，随包 `models/0.2.1`）：

```
Python 圈虫（measurement_backend=imagej）
  → other\0513-0-1_RoiSet.zip、other\0513-0-1_QC.png、other\0513-0-1_split_qc.csv、
    other\batch_summary.csv；measurements\ 存在但为空
真实插件测量同一个目录（Auto_Worm_ROI.measureOutputFolder）
  → measurements\0513-0-1_measurements.csv（11 行 = 10 虫 + 背景）
    measurements\ImageJ_measurements_all.csv（同 11 行）
```

两张表的表头逐字：

```
Area,Mean,Min,Max,IntDen,Median,RawIntDen,Index,Image,Label,PixelArea,
BackgroundCorrectedMean,CTCF,RawCTCF,Type,MeasurementBackend,
PixelWidth,PixelHeight,SpatialUnit,ValueUnit,IntensityFunction
```

第 1 行（`01_worm`、`Index=1`、`Image=0513-0-1.tif`）：

```
108168,257.965,94,730,27903590.000,251,27903590.000,1,0513-0-1.tif,01_worm,108168,
155.937,16867346.961,16867346.961,worm,ImageJ 1.54p,1,1,pixel,Gray Value,None
```

同一张图换 `measurement_backend=python` 再跑一遍，表头是 Python 侧的那一套
（没有 `Index`/`Image`/`PixelArea`/`RawCTCF` 与末尾五列），前七列位置与上表一致：

```
Area,Mean,Min,Max,IntDen,Median,RawIntDen,,Label,BackgroundCorrectedMean,CTCF,Type
108168,257.9652947267214,94.0,730.0,27903590.0,251.0,27903590.0,1,01_worm,
155.93657052651503,16867346.960712075,worm
```

两条路的 `Area` 都是 `108168`，`Mean` 一致（Python 写全精度，ImageJ 按 `ResultsTable`
的惯例写 3 位小数）——**重排只搬列，没有动数**。

未覆盖的：**旧布局（扁平目录）无法测量**这一点没有做成用例，属设计决定（`other\` 里
找不到 `<图名>_RoiSet.zip` 就报错并跳过该图，报错信息里带完整路径）；0.4.1 及更早的
`_auto_roi` 目录需要手工搬进两个子文件夹或重跑。

## 发布文件

```
dist/Auto_Worm_ROI.jar                            23534 字节
  sha256 48b84e984236d0d93c70d92d285edd83a96ee17bd4b7a0ada67bfe5c9a288a9a
dist/AutoWormImageJ/AutoWormGUI.exe               47960362 字节
  sha256 498b4d05354ce4d824189674e06f8a93a726c088f3b41a2c8ba8bca10b971d98

models/0.1.1/worm.pt  6315cfaf21594c93ebd4c4963c7929b4561e9a049b5656703c4bd5554bc1160e
models/0.1.1/tip.pt   4c9443010b5730c6f87c84c07c446303ffd3ea6b4a37d589980f74ffff4e046a
models/0.2.1/worm.pt  afb80dc4bde0cc644ab4269bfc73463d7f20eace9592a21857ce155b8f4eed4b
models/0.2.1/tip.pt   6dd86bfc467d58c2471cc61691d1bbd1bf607555f53a9e80e907e5879b1969db
```

`dist/AutoWormImageJ/models/` 下同名四个文件的 SHA256 与上表**逐个相同**（`postbuild.py`
整棵 `models/` 复制）。EXE 在 `src/batch_worm_roi.py` 的指纹改动**之后**重新构建过一次
——第一次构建早于那次改动，产物是旧的，已作废。上面这两个大小与哈希是**输出结构改动之后**
再重建的一版：jar 改了 `Auto_Worm_ROI.java`（子目录与列序），EXE 改了
`batch_worm_roi.py` 与 `worm_roi_gui.py`。发布包由 `make_release.py` 在对
`dist\` 与实盘 `src\` 做 `code_surface` 指纹比对之后打出，EXE 若与源码不符会直接失败。

构建中撞到一处真实问题并已修复：`build_0.4.2.bat` 第 1 步直接往 `dist\Auto_Worm_ROI.jar`
写 jar，而 0.4.2 的树里**没有 `dist\`**（被 `.gitignore` 排除，从 0.4.1 复制时没带过来），
`jar` 报的是 `java.nio.file.NoSuchFileException: …\Temp\Auto_Worm_ROI.jar… -> dist\Auto_Worm_ROI.jar`
——一条看不出所以然的 Java 堆栈。脚本现已加 `if not exist "dist" mkdir "dist"`，
位置与建 `build\plugin_classes` 的那行一致。

## 未做自动验证的部分

- **人工界面的验收**：界面代码只动了一处（`_load_qc_status_summary` 改从 `other/` 读
  `batch_summary.csv`，见第 7 节），其余沿用 0.4.1 的验收结论；但换了随包权重，界面选
  「高清晰度图像 / 低清晰度图像」后跑通一条真实数据，仍建议人工过一遍。
- **`补充样例4` 的 8 张里，进入留出的只有 4 张**（`*-2` 那批，与旧低清的留出集共用）。
  另外 4 张（`*-1`）在训练集里，因此它们上的表现不能作为依据。
- `0913-1k-2` 数出 11 条、`补充样例2` 两张 n=20 数出 12 条，两处均为既有现象，本次未处理。
- 低清留出集上 2 张变差的图（`g4__14`、`g3__12`）未做进一步处理或补偿。

## 尚需人工界面验收

本次新增一项（列在下面）。此外，0.4.1 那一轮遗留的界面验收事项**至今仍有效**，因为本次
没有逐条重跑；它们涉及的是界面在各版本间没变过的行为，一并列在这里，跑一次即可同时覆盖：
安装/首次启动能正常打开、选完输入输出目录后能开跑、跑完弹出的结果目录正确、界面上的
「高清晰度图像 / 低清晰度图像」两个选项各自能跑到出结果、批处理中途点取消能停下、以及
输出目录里不再残留临时文件。这些在 0.4.1 的验收里已逐条过过一遍，本次沿用其结论而未重跑
（`VALIDATION_0.4.1.md` 已从发行包移除，全文只保留在仓库历史里）。

本次新增的一项：

- 在界面上分别选「高清晰度图像」与「低清晰度图像」各跑一张真实图，确认两次都正常出结果。
- 用这一版的界面跑完一批后，把该输出目录添加进监视器面板（或直接打开监视器），确认缩略图
  照常出现、`PASS` / 「需人工复核」状态显示正确。QC 缩略图用的是递归查找，路径改动不影响
  它；但 `batch_summary.csv` 的位置是写死的 `other\`，读不到时**不报错、只是不显示汇总**，
  这类「静默不显示」只能靠眼睛发现。

  > 这里**不能**靠界面文案来确认用的是哪一版权重：`MODEL_CONFIGS` 里的 `version` 字段
  > （`0.1.1` / `0.2.1`）**没有任何代码读它**，界面上不会显示。0.4.1 里同样是死字段，
  > 本次只是把值跟着改了。要确认版本，看 `dist\AutoWormImageJ\models\` 下实际存在哪个目录，
  > 或对权重文件算 SHA256 与本文档的字表对照。函数 `_validate_runtime`
  > （`src/worm_roi_gui.py:1465`）只检查 `checkpoint` / `tip_checkpoint` 两个文件**存不存在**，
  > 不校验版本。
