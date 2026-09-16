# 0.4.2 CUDA for ImageJ

## 0.4.2（2026-09-16）

这一版有两件事，彼此无关。

一是**模型更新**：高清与低清两个整虫模型、以及它们各自的 tip 精修器，全部用新的
人工标注 `补充样例4` 重新训练过，随包权重随之替换。推理与界面逻辑与 0.4.1 相同，
只有两条为配合新权重而必须动的地方（模型目录名、命令行判定低清的那条判据）。

二是**输出目录结构改了**：输出目录下分成 `measurements/` 与 `other/` 两个子文件夹，
测量表内部的列序也重排成「要用的七个数值在最前」。这件事只动文件放在哪里、列排在第几，
不动任何数值；但**升级后旧目录里的结果不会再被自动找到**，见
[输出目录分成两个子文件夹](#输出目录分成两个子文件夹测量表列序重排)。

### 为什么重训

用户报「0913 那批图的虫数不对」。0.4.1 已经修掉了这批图里 `0913-100-2 green.tif` 的
崩溃（`predict_raw` 里 PIL 只读数组被写入），但把那批图连同人工标注一起量，**模型本身的
分割仍然是错的**：新标注的 4 张留出图，随包高清模型一张都没数对。

| 留出图（补充样例4 的 4 张） | 旧 `0.1.0` | 新（留出权重） |
|---|---|---|
| `0913-0-2` | 数出 7 | 10 ✓ |
| `0913-10-2` | 数出 8 | 10 ✓ |
| `0913-100-2` | 数出 9 | 10 ✓ |
| `0913-1k-2` | 数出 8 | 10 ✓ |
| **虫数全对** | **0/4** | **4/4** |
| 平均 Dice | 0.6719 | 0.9261 |
| 平均 tip Dice | 0.7623 | 0.8662 |
| 平均归一化端点误差 | 0.0315 | 0.0194 |

### 训练数据

`补充样例4`（开发机上的一批内部样例，不在发行包里）= 8 张 TIFF + 8 个 `RoiSet_*.zip`
+ 8 份 `Results_*.csv`，全部来自 0913 这批采集。这里的 ROI 是**人工标注**，不是插件的
输出：ROI 名是 ImageJ 自动编号（`0479-0320.roi`），而插件写出的是 `01_worm.roi`；
`0913-0-2` 的 CSV 列了 10 条虫，而同一张图插件自己跑出来是 8 条。因此可以直接当 ground
truth 用。

新的 `src/build_supplement_dataset.py` 把这份补充样例转成训练用的图像与掩膜。它与既有的
`prepare_boundary_finetune_dataset.py` 有两处不同，都是这份样例的命名逼出来的：

- 样例里 TIFF 叫 `0913-0-1 green.tif`、RoiSet 叫 `RoiSet_0913-0-1.zip`，而
  `prepare_segmentation_dataset.py` 是按**完全相等**的 `<stem>.tif` 找图的，配不上。
  新增 `matching_image()` 容忍 `<stem> <后缀>.tif`，但**要求唯一匹配**，找到 0 个或 2 个
  都直接报错，不会随便挑一张。
- 样例的来源与毛病（`!` 合并、`=` 断裂）记在**基座自己的** `dataset_manifest.csv` 里，
  而按文件名重新推是推不出来的（`classify_issue()` 找的是 `!`/`=`，对
  `boundary3__merge__…` 这样的名字一律返回 `normal`）。因此 `copy_base()` 从基座的
  清单里**照抄** `Source` 与 `Issue`，而不是重算。

两个新数据集的构成与归一化方式（各自与它要微调的那个 checkpoint 保持一致）：

| 数据集 | 构成 | 划分 | `normalization_mode` |
|---|---|---|---|
| `training/high_v041` | 46（原 `_combined_training_n10`）+ 8（补充样例4） | 50 训练 / 4 留出 | `legacy` |
| `training/low_v041` | 76（`_combined_training_v020`）+ 20（补充样例3）+ 8（补充样例4） | 90 训练 / 14 留出 | `background_aware` |

训练都是**从现有 checkpoint 微调**（`--initial-checkpoint`），不是从零重训。超参与各自
的上一版逐项相同（高清 `--tip-boost 4.0`；低清 `--hard-repeat 4 --contact-boost 3.0
--continuity-boost 1.75`），两个 tip 精修器用 `--train-all` 全量重训，训练输入是**新整虫
模型自己吐出来的** guide（换了整虫模型就必须重跑 guide，否则精修器学的是旧模型犯的错）。

> 低清第一次启动时漏了 `--image-size 768`，checkpoint 记成了 512（`boundary_radius`
> 随之变成 4）。**这一版没有发出去**，发现后删掉重跑。之所以要盯这个数：推理侧
> `batch_worm_roi.py` 是从 checkpoint 读 `image_size` 的，512 会静默改变所有低清用户的
> 工作分辨率。`finalize_low_model_042.py` 现在把 `image_size != 768` 直接当致命错误，
> 免得再犯。

### 留出集上的对照

低清 14 张留出图（`boundary3` 的 10 张 + 补充样例4 的 4 张）：

| | 旧 `0.2.0` | 新 `0.2.1` |
|---|---|---|
| 虫数全对 | 9/14 | **10/14** |
| 平均 Dice | 0.8705 | **0.8877** |
| 平均 tip Dice | 0.7857 | **0.8110** |
| 平均归一化端点误差 | 0.0395 | **0.0335** |

**这 14 张里有 4 张的虫数变了，两好两坏，净 +1**，逐张列出来：

| 图 | 旧 | 新 | |
|---|---|---|---|
| `boundary3__merge__g4__14__…` | 10 ✓ | 9 | **变差** |
| `boundary3__normal__g1__04__…` | 11 | 10 ✓ | 变好 |
| `boundary3__normal__g3__12__…` | 10 ✓ | 11 | **变差** |
| `supp4__0913-10-2` | 9 | 10 ✓ | 变好 |
| `supp4__0913-100-2` | 9 | 10 ✓ | 变好 |

变差的两张都来自 `boundary3`（旧低清模型的特长域），变好的三张里两张来自新加入的
补充样例4。把新域的 8 张加进训练集，代价就是在旧域上让出一部分——**没有免费的提升**，
这一点如实写在这里，用之前请按自己的数据判断值不值。

### 端到端回归（三批真实图，走完整条流水线）

比 Dice 更能说明问题的是**端到端虫数**：用户报的就是虫数。三批图分别用新旧模型各跑一遍
`run_gui_batch`，逐张对照 `batch_summary.csv` 里的 `worm_count`：

| 批次 | 张数 | 用的模型 | 虫数有变化的图 |
|---|---|---|---|
| 0913（用户报 bug 的那批） | 8 | 高清 | 0/8 |
| 0913 | 8 | 低清 | 2/8 —— 全部**变好** |
| 样例1（0716） | 8 | 高清 | 4/8 —— 全部**变好** |
| 补充样例2 | 30 | 低清 | 0/30 |

**46 张图共 6 张虫数变化，六张全是改对，没有一张变差。** 逐张：

- 高清 · 样例1：`0716-0-2`、`0716-10-1`、`0716-10-2`、`0716-1k-2` 由 9 条
  `REVIEW_COUNT_MISMATCH` 变为 10 条 `PASS`。这四张是高清模型**原本的训练域**，
  说明这次微调没有只顾新域而丢掉老本行。
- 低清 · 0913：`0913-10-2`、`0913-100-2` 由 9 条变为 10 条 `PASS`。
- 0913 八张在两个模型下都**没有**变差的图；其中 `0913-1k-2` 两版都数出 11 条
  `REVIEW_COUNT_MISMATCH`，这是 0.4.1 就有的既有现象，本次没有触及。

### 随包权重的目录名跟着换了

`models/0.1.0` → **`models/0.1.1`**，`models/0.2.0` → **`models/0.2.1`**，各自含
`worm.pt` + `tip.pt`。旧的两份**不再随包**（0.4.1 的发行包里仍然有）。这样目录名与权重
本身对得上，不会出现「`0.1.0` 里装的其实是第二代权重」这种要翻文档才知道的事。
界面 `MODEL_CONFIGS` 指到新目录，`postbuild.py` 照旧整棵 `models/` 复制 —— 因此随包体积
与 0.4.1 持平，没有变大。

新权重另外补了出处字段（旧的高清权重里一个都没有）：`software_version`、
`training_dataset`，以及

- 高清：`holdout_count_accuracy` / `holdout_mean_matched_dice` /
  `holdout_mean_tip_dice` / `holdout_mean_endpoint_error`
- 低清：`supplement_validation_count_accuracy` /
  `supplement_validation_mean_matched_dice`

**高清那几个数为什么叫 `holdout_` 而不叫 `val_`**：随包高清权重是 `--train-all` 出来的
（`trained_on_all_records: true`，54 张全进了训练），它自带的 `val_*_at_best` 是**在训练
集上**算的，不能当泛化能力读。真正没见过的只有留出那一版权重见过的那 4 张，所以
`finalize_high_model_042.py` 强制要求 `--evaluation` 指向留出权重的评测目录，并给指标换
了前缀。低清不需要这样：它本身就是按 90/14 划出来的，验证集确实是留出的。

权重里 `initialized_from` 原本记的是开发机上的绝对路径（形如
`…\备份\0915 0.4.2 CUDA for ImageJ(Java8)\models\0.1.0\tip.pt`），已改写
成仓库内的相对路径 `models/0.1.0/tip.pt`。功能上无影响，但发行包里不该有开发机的目录结构。

### 命令行判定低清的那条判据，从单个指纹改成一组

`batch_worm_roi.py` 里 `LOW_CLARITY_MODEL_SHA256`（单个值）改成
`LOW_CLARITY_MODEL_SHA256S`（元组），同时收录 **0.2.1 与 0.2.0** 两版权重。

这不是顺手扩的：界面每次都把 `low_clarity_split` 显式传进来，但**命令行**不传，靠这个
指纹自动判断。只留新指纹的话，0.4.1 那批已经散出去的低清权重到了 0.4.2 上会被静默当成
高清走另一条分支 —— 正是这条判据当初要防的事。现在两版都认得出，高清权重仍然不命中
（已用三个 SHA 实测）。

### 输出目录分成两个子文件夹，测量表列序重排

输出目录（默认 `_auto_roi`）下面现在固定分两层：

```text
_auto_roi/
  measurements/   <图名>_measurements.csv、ImageJ_measurements_all.csv
  other/          <图名>_RoiSet.zip、<图名>_QC.png、batch_summary.csv、
                  <图名>_shape_qc.csv / _split_qc.csv / _head_annotations.csv /
                  _manual_split_qc.csv / _segment_qc.csv
```

分的是「要拿去统计的」和「看一眼就行的」：测量表常被整批拷走做下游分析，此前它和预览图、
QC 明细混在同一个目录里，拷贝时得按后缀一张张挑。`batch_summary.csv` 是**这次运行的工作
记录**（识别数量、`qc_status`、耗时、`run_id`、失败原因），不是 ROI 的数值，因此与 QC 图
一起放进 `other`，不算测量表。

测量表的**列序**同时按「先给要用的数」重排：`Area`、`Mean`、`Min`、`Max`、`IntDen`、
`Median`、`RawIntDen` 移到第 1 至 7 列，`Index`、`Image`、`Label`、`PixelArea`、
`BackgroundCorrectedMean`、`CTCF`、`RawCTCF`、`Type`、`MeasurementBackend` 依次排在其后。
记录校准的五列（`PixelWidth`、`PixelHeight`、`SpatialUnit`、`ValueUnit`、
`IntensityFunction`）仍然压在行尾 —— `tests/AutoWormMeasurementTest.java` 是按「倒数第
五列起」读它们的，位置一动那个用例就会失败，这是故意的，免得顺序被悄悄改掉。
**数值一个没动**：重排只搬运列，不动算式。本轮拿 `0513-0-1.tif` 走完整条流水线，
Python 自测与 ImageJ 插件两条路量出的同一条虫数值一致（`Area` 都是 `108168`；`Mean`
Python 侧写全精度 `257.9652947267214`，ImageJ 那张表按 `ResultsTable` 的惯例写成
`257.965`）。

Python 自测与 ImageJ 插件两种后端写出的列序一致，但**表头并不完全一样**：Python 侧没有
校准概念，`IntDen` 与 `RawIntDen` 是同一个数，也没有 `Index` / `Image` / `PixelArea` /
`RawCTCF` 与末尾五列校准列。两边共有的部分（前七个数值列）位置相同。

**不兼容旧布局，也不做兼容。** 插件只从 `other/` 里找 `<图名>_RoiSet.zip`，找不到就报
「ImageJ 无法读取 ROI ZIP」并跳过该图（报错信息里会写出它找过的那个完整路径）；测量表
一律写进 `measurements/`，该文件夹不存在时由 `saveTable` 在写之前建好 —— 这样把 ImageJ
指向一个旧版本的扁平目录时是**只读**的，不会在别人的文件夹里凭空造出两个子目录。旧目录
要接着测，就把文件按上表搬进两个子文件夹，或者重跑一遍。GUI 读取 QC 图用的是递归查找
（`rglob`），因此监视器面板在新旧布局下都能找到缩略图；`batch_summary.csv` 的读取路径是
写死的 `other/`，找不到就只是不显示汇总，不报错。

**两个名字有两份。** `MEASUREMENTS_DIRNAME` / `OTHER_DIRNAME`（`batch_worm_roi.py`）与
`MEASUREMENTS_DIR` / `OTHER_DIR`（`Auto_Worm_ROI.java`）是同两个目录名的两份声明，
改一边必须改另一边，没有东西会替你检查。之所以不合并：桥接协议传给 Java 的仍然是**输出
根目录**，两个子目录名由 Java 侧自己拼，Python 不需要知道插件打算写什么。

### 构建脚本：新克隆的仓库里 `dist\` 不存在

`build_0.4.2.bat` 第 1 步直接往 `dist\Auto_Worm_ROI.jar` 写 jar。`jar` 是先写临时文件
再改名过去的，目标目录不存在时抛的是 `NoSuchFileException` 加一串 Java 堆栈，看不出
是「目录没建」。`dist/` 被 `.gitignore` 排除，新克隆的树里没有，于是第一次构建必挂。
现在脚本先 `mkdir dist`，与它建 `build\plugin_classes` 的写法一致。本机这次重建就是
撞上它才发现的。

### 新增的脚本

- `src/build_supplement_dataset.py`：把补充样例（TIFF + RoiSet + Results CSV）转成训练集。
- `src/finalize_low_model_042.py`、`src/finalize_high_model_042.py`：给训练产物盖上产品
  参数与出处，写出随包权重。与 0.3.3 那份 `finalize_low_model_033.py` 的差别是：
  验证指标从**评测目录里现读**（`--evaluation`），不再硬编码在源码里 —— checkpoint 里记
  的数字必然出自那次评测，而不是谁抄进去的。两处会在写盘前校验并回读确认再替换。

### 没有改的东西

- Python 推理路径、分割与 ROI 几何、界面行为：与 0.4.1 逐字相同，只有上面写明的两处
  （模型目录名、低清指纹集合）。
- **测量出来的数值**：`Area`、`Mean`、`Min`、`Max`、`IntDen`、`Median`、`RawIntDen`、
  `CTCF`、`RawCTCF`、`BackgroundCorrectedMean` 的计算式一个没动，ImageJ 测量本身也
  没动。变的是这些数写在**哪张表的第几列**，以及那张表放在**哪个文件夹**。
- 低清拆分（`low_clarity_split`）、形状细化、人工头向标注、人工排除区、部分圈画跳过：
  未动。
- 0.4.1 引入的半像素边界约定与那批 P1/P2/P3 修复：未动。

## 0.4.1

0.4.1 相对 0.4.0 只有一处实质改动：**导出的 ROI 改按像素边界画，ImageJ 量到的像素与圈虫统计的像素自此逐个相同。** 下面是这一版对用户有影响的几点。

- **修复了 ImageJ 比 Python 少算一圈边界像素**：ImageJ 判定像素是否属于 ROI，看的是像素**中心**落没落在多边形里，而导出的轮廓顶点原本落在像素**编号**上，两者差半格，于是每个方向少一行一列（实测同一只虫：Python 侧 45272 像素、ImageJ 侧 44325 像素，约 2.1%）。现在轮廓画在像素边界上，两边取到的像素集合逐像素相同。
- **数值与 0.4.0 及更早版本对不上，这是本次改动的目的**：用旧版导出的 ROI 比 Python 统计的区域小一圈，因此旧版跑、ImageJ 后端测出的面积、均值与 CTCF 会偏低（面积约差 2%；边界与内部亮度差得越多，均值差异越大）。**若要新旧数据混用，请以同一版本处理整批。**
- **代价**：ROI 不再抽稀，顶点数变多。同一批真实数据实测每张图的 ROI 总顶点数 1448 → 4690，`*_RoiSet.zip` 约 8.5 KB → 约 59 KB。
- **修复「窗口中临时设的校准被静默忽略」**：在 ImageJ 里用 `Analyze > Calibrate...` 或 `Analyze > Set Scale...` 设了校准、还没有存回文件时，原先的提醒不会出现（`ImagePlus.changes` 在这条路径上不会置位），而圈画与测量用的都是磁盘文件里的旧校准——屏幕是一张图，算的是另一套数。现在调用插件时会先比对窗口与文件的校准，逐项列出并说明采用文件上的那一份。
- **测量输出新增五列校准信息**：`PixelWidth`、`PixelHeight`、`SpatialUnit`、`ValueUnit`、`IntensityFunction`，附在每张图的 `*_measurements.csv` 与合并表末尾，逐行记录该图是用什么校准量出来的。
- **两项与手工测量有意不同的地方**（自 0.4.0 起如此，写在这里备查）：**不继承 `Limit to Threshold`**（插件统计整个 ROI），**不继承 `Redirect to`**（插件直接统计原图）。要与手工 `Analyze > Measure` 对上数，请先关掉这两项。

更早的 0.4.0 及其各轮复审见仓库历史中的 `CHANGELOG_0.4.1.md`（该文件已从发行包移除，全文只保留在仓库历史里）。
