# Auto Worm ROI —— ImageJ / Fiji (Java 8) Plugin

>**An English translation follows the Chinese original — see [English Version](#auto-worm-roi--imagej--fiji-java-8-plugin-english) below.**
>
>该插件用于批量自动圈画线虫，生成ROI；荧光统计与 CTCF 仍交给 ImageJ/Fiji 进行。插件不改变原有的荧光统计与 CTCF 处理过程，只是将这一步自动化。
>
>当前版本（0.4.3 CUDA for ImageJ（Java 8））是为使用NVIDIA系列显卡，且  ImageJ 内置 Java 版本为 Java 8 的用户开发的测试版。具体详见 [测试版说明](#测试版说明)
>
>本插件使用 deepseek-V4.1-flash 和 ChatGPT 5.6 sol 协助开发，由 ChatGPT 5.6 sol 和 ChatGPT 6 Astra 审查。

## 这个插件是干什么的

读入一张灰度 TIFF，把一批线虫逐条分割、圈画出来，输出成 ImageJ 的 ROI。

## 测量是怎么实现的

ROI 生成后会交回 ImageJ：插件用 ImageJ 本体的 `ImageStatistics` 统计每个 ROI 的像素，按 ImageJ 的空间与强度校准得到 `Area`、`Mean`、`IntDen` 等值，再用 `ResultsTable` 写出测量表，`CTCF` 就由这些值算出（细节见 [ImageJ 测量与 CTCF](#imagej-测量与-ctcf)）。

这里**不经过 `Analyzer`**：它读的是 `Analyze > Set Measurements` 里的全局设置（尤其是 `Redirect to`），会出现结果行的图名与实际取值的图不是同一张的情况，详见[与手工测量对照时](#与手工测量对照时)。

## 测试版说明

0.4.3 CUDA for ImageJ（Java 8）在开发机上跑通了大部分功能。但是，部分异常情况因缺乏数据仅进行了代码审查，未能进行真实数据测试。此外，可能还存在未能够考虑到的异常情况，需要用户进行反馈。

另外，本插件中的模型使用 NVIDIA GeForce RTX 4060 Laptop 进行训练，暂未在其他型号的 NVIDIA GPU 上进行过测试。如遇硬件兼容问题，请一并反馈。

反馈请发至我的个人邮箱：rivendell118@gmail.com
                  或是 3024732774@qq.com
也可以在GitHub上提交issue。

## 安装
硬件要求：计算机需要带有 NVIDIA GPU
软件要求：需要预先安装 Fiji 或 ImageJ 1.x（1.54p 及以后）

1. 把 .zip 文件整个解压到 ImageJ/Fiji 的根目录，即`ImageJ-win64.exe`（Fiji 里是 `fiji-win64.exe`）所在的那个文件夹。
2. 重启 ImageJ/Fiji。

解压后应当是这样：

```text
<Fiji 根目录>/
  ImageJ-win64.exe
  plugins/
    Auto_Worm_ROI.jar        插件本体
    AutoWormImageJ/
      AutoWormGUI.exe        GUI界面
      models/                分割模型，外置便于单独替换
      _internal/             运行库（Python、PyTorch、CUDA、cuDNN…）
      licenses/              第三方组件许可全文
  README_0.4.3_ImageJ_Java8.md     本文件
  CHANGELOG_0.4.3.md
  VALIDATION_0.4.3.md
  LICENSE
```

- `AutoWormImageJ` 里的 `models`、`_internal`、`licenses` **缺一不可!!!** 不要只复制其中一部分。
- `licenses/` 不要删，理由见[许可与第三方组件](#许可与第三方组件)。
- 四份文档（README / CHANGELOG / VALIDATION / LICENSE）放在哪里都行，不影响运行。
- 不愿意解压到根目录的话，把 `plugins/` 下的两个条目手工复制到 `<Fiji 根目录>/plugins/` 下效果相同。

**最终用户不需要安装 Python，也不需要安装 CUDA Toolkit**。PyTorch、CUDA 13.2、cuDNN、OpenCV、SciPy 和 MSVC 运行库都已经放进 `AutoWormImageJ`。但用户的电脑需要安装 **580 或更高版本** 的 NVIDIA 驱动。GUI 界面里的「设置 > CUDA 与系统信息」可以查看 GPU、驱动、PyTorch、CUDA、cuDNN 状态；开始处理前插件也会自动检查一遍。

## 快速上手

1. 打开 ImageJ。
2. 运行插件：`Plugins > Auto Worm ROI` 子菜单里的第一项，名称形如 `Auto Worm ROI <版本号>`。菜单文字用纯英文 ASCII，以兼容 Fiji 1.54p 读取 `plugins.config` 时的系统编码行为；启动后的界面仍然可以选中文或英文。
3. 在界面里选好当前文件夹、图像类型和预设虫数，按需要勾选「平滑修复」/「手动标注」/「部分圈画」，然后点「开始处理」。
4. 处理完成后界面保持打开，ImageJ 自动接收并测量本批结果。可以接着选别的文件夹继续处理。
5. 关闭界面即结束插件。

结果默认写在 **当前文件夹下的 `_auto_roi` 子目录** 里，可以在设置界面里改。这个目录下面固定分成 `measurements`（测量表）和 `other`（其余结果）两个子文件夹，见 [结果文件说明](#结果文件说明)。

## 当前图像有未保存的修改时

调用插件后、界面打开之前，如果 ImageJ 里当前这张图有过未保存的改动（例如刚做过滤波、裁剪、反色），会先弹一个提示：**本次处理的是磁盘上保存过的版本**。圈虫和测量都从磁盘文件读像素，屏幕上未保存的改动不参与；否则 ROI 与测量结果会指向看着一样、实际不同的两份图像。

- 点「确定」继续：按磁盘上的版本处理这一批，未保存的改动被忽略（图本身不会被改动，也不会被保存）。
- 点「取消」：什么都不做，状态栏提示未处理。

要把改动一起处理，先在 ImageJ 里保存图像，再重新调用插件。

## 界面上的参数

| 参数 | 作用 |
| --- | --- |
| 当前文件夹 | 要处理的目录。批处理范围就是该目录下的 TIFF |
| 图像类型 | `高清晰度图像` 或 `低清晰度图像`，两套在不同采集条件下训练的分割模型 |
| 预设虫数 n | 每张图里应有的虫数。识别数量必须等于它，不匹配的图需人工复核 |
| 平滑修复 | 独立的轮廓质量检测与保守形状修复，作用在模型分好的结果上 |
| 手动标注（头向/分界/排除区） | 启用后读取你标注的头向箭头、人工分界和背景排除区 |
| 部分圈画（需手动标注头向） | 只圈虫体的一段，而不是整条 |

## 功能详解

### 图像类型与模型

两套模型分别在不同采集条件下训练：`高清晰度图像` 与 `低清晰度图像`。选错不会报错，只是分割质量下降，所以换采集设备或换放大倍率时值得各试一遍，用每张图的 `*_QC.png` 对比。模型是 `AutoWormImageJ\models\` 下的普通文件，可以单独替换。

### 预设虫数 n

程序要求每张图识别出的虫体数量**与 n 完全相等**。数量一致的图在 `batch_summary.csv` 里 `qc_status` 为 `PASS`；不一致的记为 `REVIEW_COUNT_MISMATCH`，需要人工复核——注意这个判定只影响 QC 标记，分割本身不会被强行凑成 n 条。同一个文件夹里的图虫数不同时，按文件夹分次处理。

### 平滑修复

对分割结果做轮廓质量检测和有安全门控的形状修复：修复只在不会明显改变面积与质心的前提下进行，改动幅度超限就不动它。适合边界毛糙、形状不规则的图。开启后会为每张图写出 `*_shape_qc.csv`，其中 `status`、`repaired`、`needs_review` 三个字段说明该图的轮廓质量和是否被改过。有虫体被改动或被标记为轮廓可疑时，该图在监视器里会显示黄框，下一节讲边框颜色的判定。

### 监视器上的边框颜色

监视器里每张 QC 图外面都有一圈边框，颜色说明这张图值不值得点开细看。一张图只显示优先级最高的那一种：**红 > 黄 > 蓝**。

| 边框 | 触发条件 |
| --- | --- |
| 红框 | 该图识别出的虫体数量与「预设虫数 n」不相等，也就是 `batch_summary.csv` 里该图 `qc_status` 为 `REVIEW_COUNT_MISMATCH`。界面上这张图的左上角会额外显示一行「需人工复核 ｜ <原图名>」 |
| 黄框 | 数量正常，但分割结果被程序改写过：`shape_refined_count`（平滑修复实际修了形状的虫体数）、`shape_review_count`（平滑修复标记为轮廓可疑的虫体数）、`low_clarity_split_count`（低清晰度模式下的粘连拆分次数）三者任一大于 0 |
| 蓝框 | 当前选中的那一张，与结果好坏无关 |

红框说明**数量**不对，必须处理：核对「预设虫数 n」，或者用「手动标注」补头向箭头、人工分界。黄框只说明**这张图不是模型的原始输出**——轮廓被修过，或者低清晰度模式下有一个粘连块被拆成了两条。它不代表结果一定错了（修复本身有安全门控，改动幅度超限就不会动手），但这类图是人工抽看时该先看的一批。

黄框的依据是逐条虫体的记录，两份表的列不一样：

- `<图名>_shape_qc.csv`：`status` 列是 `REVIEW_SHAPE_REFINED`（真的改过形状）或 `REVIEW_SHAPE`（只标记有问题、没动手）。`repaired` 为真表示轮廓确实被改过；`needs_review` 在这两种状态下都为真，所以它标记的范围比 `repaired` 宽。另有修复前后的面积与质心偏移、各项质量指标和 `reasons` 列写明的判定原因。
- `<图名>_split_qc.csv`：每行是一次粘连拆分，`status` 为 `REVIEW_LOW_CLARITY_SPLIT`，另有拆分阈值、头部缝隙宽度、分界支持度和两个子块面积比，可据此判断这次拆得靠不靠得住。

红框和黄框是**直接画在 `*_QC.png` 文件里**的：用别的看图软件打开它、在文件管理器里预览、或者把图发给别人，边框都还在。蓝框只存在于界面里，是当前选中项的高亮。

「手动标注」（头向箭头 / 人工分界 / 背景排除区）和「部分圈画」**不会**单独触发黄框——前者是你自己指定的改动，后者只改截取的区段、不改变虫体划分。这类图仍按红、蓝两色判定。

处理失败的图不会出现在监视器里：它的结果文件（含 `*_QC.png`）在失败时就一并删掉了，见[批处理、失败与汇总](#批处理失败与汇总)。

### 手动标注

三种标注：**头向箭头**、**人工分界**、**背景排除区**。

- `head_1`、`head_2`…… 直线或箭头，从虫体内部画向头部；每条虫一支。
- `boundary_1`…… 线或自由线，用来提示相连虫体的人工分界。
- `exclude_1`…… 闭合面积 ROI，内部区域不参与识别，适合背景较脏的图。

头向箭头与人工分界是 **两个平级、可独立工作** 的算法：只有头向箭头时，程序按头向对粘连虫体做粗分；只有人工分界时，程序直接依据分界线拆分或重划相邻 ROI；两者同时存在时，固定先执行头向粗分、再执行人工分界精修。人工分界精修允许在虫数已经正确时重新分配相邻 ROI 的像素归属，但不会改变原有的前景像素总集合。

**人工分界线是硬约束**：画了分界线的地方，相邻两段虫体的交界就落在你画的这条线上，不再由模型自己决定落在哪里。压线只在分界线周围一条窄带内进行（约为图像短边的 4%，1024 像素的图约 40 像素）：若实际缝隙离你画的线比这还远，程序会维持模型给出的划分——硬压过去会从带外切出一条不属于任何虫体的碎块，那比接缝偏一点更糟。

**推荐直接在 GUI 界面里标注**（勾选「手动标注」）

三种标注在界面里的用法：

- **头向箭头**：从虫体内部按住左键拖向头部后松开，箭头尖端就是头部。
- **人工分界**：沿两条粘连虫之间的缝隙，从头端到尾端**左击依次点出节点**，鼠标移动时会从最后一个节点拉出一条跟随光标的虚线；**右击结束**这一条，按 Esc 放弃正在画的这一条。折线的拐点就保留在你点下的位置，不会被抹圆或抽稀；保存后显示为青色折线，它同时就是程序实际使用的那条边界。
- **背景排除区**：沿脏背景或其他无需识别区域的外围按住左键画一圈，松开后自动闭合。

也可以在 ImageJ 的 ROI Manager 里准备 ROI，用上面那套前缀命名，然后运行 `Plugins > Auto Worm ROI > Save Current Annotation`，插件会在 TIFF 旁生成 `<原文件名>.autoworm.json`。逐图 sidecar 不会影响同一文件夹里其他图像的标注；旧版的文件夹级 `_manual_head_annotations.json` 仍然兼容。

同一张图同时存在两种记录时，**按标注类型分别合并**：sidecar 里非空的一类覆盖文件夹记录，为空的一类沿用文件夹记录，因此只画了头向箭头的 sidecar 不会丢掉该图在界面里画的人工分界。在界面里保存某张图时，如果该图已有 sidecar，同一份结果会一并写回 sidecar，所以界面里的删除和改动都会立即生效。

⚠️ 同一张图的标注请不要在两个界面之间来回维护。以下情形目前无法从 ImageJ 一侧完成：

- 该图的分界只存在于 GUI 界面写出的 `_manual_head_annotations.json` 里，而你在 ROI Manager 中删掉 `boundary_` 后重新 `Save Current Annotation`——此时 sidecar 的 `boundaries` 是空数组，合并规则会回落到文件夹记录，那条分界仍然生效。
- 要真正删掉它，请在 GUI 界面里删除后保存，或者直接删除该图的 `<原文件名>.autoworm.json`。
- 反方向不受影响：只要 sidecar 里存在非空的分界，它就会覆盖文件夹记录里同一图的分界，而在界面中保存会把改动写回 sidecar。

*所以，为了方便起见，请各位尽量在 GUI 界面中进行改动*

### 部分圈画

勾选「部分圈画（需手动标注头向）」后会强制读取头向标注，只圈虫体的一段。双滑块共用一条有方向的轨道：**左端为头（0），右端为尾（1）**。例如中段 `0.25–0.75`、后四分之一 `0.75–1.00`、前半段 `0.00–0.50`。

引擎先识别完整虫体和中轴线，再按头尾方向截取选定区段。**背景 ROI 仍以完整虫体为依据**，避免被截掉的虫体区域混进背景里。

### 批处理、失败与汇总

每次处理都在输出文件夹的 `other` 子目录写入 `batch_summary.csv`，每处理完一张刷新一次，所以中途取消或出错也能拿到已完成的部分。

- **单张失败不终止整批。** 某张图出错时程序记下原因、跳到下一张，其余图像照常处理并交给 ImageJ 测量。失败的那张在日志里显示为 `Failed_<原文件名>: ERROR <原因>`，在 `batch_summary.csv` 里同样记为 `Failed_<原文件名>`，`qc_status` 为 `FAILED`，原因写在最后一列 `error`（成功的行该列留空）。**磁盘上的原文件不会被改名**，`Failed_` 只是日志和汇总表里的记法。
- **失败的那张不会留下任何结果文件。** 程序在画 QC 图之前就已经写好了 `*_RoiSet.zip`，如果中途失败，这个 zip 会被删掉，连同该图已写出的测量表和各类中间报告。这一点很重要：ImageJ 只测量「存在 `<图名>_RoiSet.zip`」的图，残留没清掉的话，一张刚被报为失败的图反而会被测进结果表。极少数情况下（例如 QC 图正被看图软件打开而删不掉）会删不掉，此时日志里补一行 `Could not remove partial results for ...`，请照它手动删除。
- **进度。** 每处理完一张输出一行 `Progress: 3/12 (25%)`，失败的那张也计入进度，在日志窗口里就能判断还剩多少。
- **结果重名会被拒绝。** 结果文件按「去掉扩展名的文件名」命名，因此同一文件夹下的 `a.tif` 和 `a.tiff` 会写到同一组文件名上互相覆盖。检测到这种情况会在处理开始前拒绝整批并列出冲突的文件名，不做静默覆盖。
- **失败的图在 ImageJ 一侧不会静默消失。** 测量结束后，如果本批有失败图像，插件会弹出「部分图像未处理」并逐条列出 `Failed_<原文件名>` 与原因，同样写进 ImageJ 的日志窗口。

## 支持的图像格式

只处理**单通道、单层、单时间点**的灰度 TIFF：8 位和 16 位均可，**16 位大端字节序（`I;16B`）同样支持**。

彩色 TIFF（RGB、RGBA、调色板、带 alpha 的灰度、CMYK）和多页 TIFF（多通道、Z 堆栈、时间序列，含 ImageJ 超栈和 OME-TIFF）会被拒绝；改名成 `.tif` 的 PNG / BMP 等非 TIFF 文件也会被拒绝。

**例外：多余平面全是 0 的图会放行。** MetaMorph 等采集软件会给每张图附加一个空的 overlay 平面，这类文件真正的数据仍在第一页，附加平面没有任何信号、不可能量出错误的数值，因此不予拒绝，只在日志里说明，例如 `Phsp16.2 0101-70-1.tif: 1 empty extra plane(s); treated as overlay`。需要注意的是 ImageJ **不会**跳过这个平面，它会把该图显示成 2 个切片，测量前请确认窗口中停在的是第 1 个切片。

拒绝的原因是程序只读取文件的**第一页**。批处理里除当前在 ImageJ 中打开的那一张之外，其余图像都由 ImageJ 从磁盘重新打开并测量第 1 个平面，与程序一致；但**当前打开的那一张**（也就是插件传给原界面的那一张），ImageJ 测的是窗口中当前显示的通道、Z 层和时间点。只要它停在别的平面上，圈出的 ROI 与实际测量值就会指向不同平面，而且**全程不报任何错**。因此检查在处理开始前进行，覆盖整批：只要有一张不符合，就按原因分组列出文件名并拒绝整批，不会先跑一部分再中断。一次列出的文件名有上限，超过时会补一句「共 N 张不合格」，不会漏报数量。

多通道 / 堆栈数据请先用 ImageJ 的 `Image > Stacks > Stack to Images` 拆成单页 TIFF，只保留要测量的通道，再重新处理。

二值（1 位）图像不在上述格式之列，但**不会被拒绝**：程序在开始前提示「本模型未针对二值图片进行训练」，确认后照常处理。两个模型都是在 8 位和 16 位采集图上训练的，二值图的分割结果可能不准确，所以提示保留，只是不挡住流程。

## 文件与文件夹命名

结果文件都按「输入文件名去掉扩展名」生成（见下一节），因此输入的名字直接决定输出的名字。有两条限制：

**文件名与文件夹名里都不要用方括号 `[` `]`。** 开始处理前程序会把输入文件夹整个列一遍，这一步用的是通配匹配，而方括号在通配语法里是**字符集**：**文件夹名**叫 `[batch2]` 时，它被理解成「b、a、t、c、h、2 这六个字符里的任意一个」，于是一张 TIFF 都匹配不到，整批被拒绝并提示 `No TIFF images found in: ...`——可界面上明明已经把图片列出来了（界面用的是另一种列目录方式），现象和信息对不上，很容易被当成程序坏了。图像**文件名**里带方括号目前不会被拦下，但建议一并不用，免得以后再踩到同一类问题。

**文件名开头和结尾不要带空格。** 处理本身能正常跑完，但测量环节是拿文件名字符串与批处理报上来的名字逐字比较的，首尾多一个空格就匹配不上：那张图会被**静默跳过**——既没有测量结果，也不会有任何提示。名字**中间**的空格没有问题（`0716-0-1 green.tif` 就是这么命名的）。

## 结果文件说明

`<图名>` 指输入文件去掉扩展名的名字。输出目录下固定分成两个子文件夹：**测量表放 `measurements`，其余结果放 `other`**，这样要拿去做统计的那几张表始终待在一起、不会被预览图和 QC 明细混在一起。

| 文件 | 写出方 | 位置 | 内容 |
| --- | --- | --- | --- |
| `<图名>_measurements.csv` | ImageJ | `measurements` | 该图的测量表 |
| `ImageJ_measurements_all.csv` | ImageJ | `measurements` | 本批所有图的合并测量表 |
| `batch_summary.csv` | 圈虫 | `other` | 每张图一行的总表：识别数量、`qc_status`、耗时、失败原因（`error` 列） |
| `<图名>_RoiSet.zip` | 圈虫 | `other` | 该图的 ROI 集合，ImageJ 读它做测量 |
| `<图名>_QC.png` | 圈虫 | `other` | 圈画结果预览图，叠出分割结果并写上 QC 状态，建议抽看 |
| `<图名>_head_annotations.csv` | 圈虫 | `other` | 头向标注的匹配结果 |
| `<图名>_manual_split_qc.csv` | 圈虫 | `other` | 人工分界精修明细，仅在该图的分界真正改变了像素归属时写出 |
| `<图名>_shape_qc.csv` | 圈虫 | `other` | 平滑修复前后的轮廓质量 |
| `<图名>_split_qc.csv` | 圈虫 | `other` | 低清晰度模式下的粘连拆分明细 |
| `<图名>_segment_qc.csv` | 圈虫 | `other` | 部分圈画的选段明细 |

**0.4.2 之前的版本把上面这些文件全平铺在输出文件夹里。** 这一版只认新布局：调用插件测量时，ROI 集合只从 `other` 里找，没找到就报「找不到 ROI ZIP」并跳过该图，测量表也一律写进 `measurements`（该文件夹不存在时会自动建好）。拿旧版本留下的文件夹去测量，需要先把文件按上表搬进两个子文件夹，或干脆重跑一遍。

`<图名>_manual_split_qc.csv`、`<图名>_shape_qc.csv`、`<图名>_split_qc.csv`、`<图名>_segment_qc.csv` 这四张 QC 明细表**只在与它对应的功能开启时才写出**；未开启时会顺带删掉上一次运行留下的同名文件，免得看到过期数据。

注意别把 `<图名>_head_annotations.csv` 和标注输入文件搞混：`_manual_head_annotations.json`（在**图像文件夹**里）和 `<图名>.autoworm.json`（在 TIFF 旁边）是标注数据本身，不是结果。

## ImageJ 测量与 CTCF

每个 ROI 由 ImageJ 本体测量 `Area`、`Mean`、`Min`、`Max`、`IntDen`、`Median` 和 `RawIntDen`。插件另外输出：

```text
BackgroundCorrectedMean = worm Mean - background Mean
CTCF = worm IntDen - worm Area × background Mean
RawCTCF = worm RawIntDen - worm PixelArea × background raw Mean
```

`CTCF` 遵从 ImageJ 的空间和强度校准；`RawCTCF` 使用原始像素值（`RawCTCF` 是更老的开发版本中使用的算法，在发行版接入 ImageJ 后废弃。但是我还是把它保留下来，因为不想删了）。测量表里的 `MeasurementBackend` 会记录实际使用的 ImageJ 版本。

**列的顺序固定为：`Area`、`Mean`、`Min`、`Max`、`IntDen`、`Median`、`RawIntDen` 在第 1 至 7 列**，其后的 `Index`、`Image`、`Label`、`PixelArea`、`BackgroundCorrectedMean`、`CTCF`、`RawCTCF`、`Type`、`MeasurementBackend` 依次排在后面。要直接拿去做统计的七个数值都在最前面，导入 Excel、R 或 pandas 时不必再按列名挑。`<图名>_measurements.csv` 与 `ImageJ_measurements_all.csv` 用同一套列序。图上的校准信息不参与这个排序，仍然固定压在整行的末尾（见下）。

表格里的 `RawIntDen` 与 `RawCTCF` 是同一套原始数值：插件取的是 `getRawStatistics()`，即**真正的原始像素之和**，这一列只服务于 `RawCTCF` 这个参考量，别处不用它。因此它与 ImageJ 1.54p 的**同名列**在「有强度校准」的图上数值不同——ImageJ 那一列算的是「像素数 × 校准后的均值」；图上没有强度校准时两者一致（本轮在 8 位、16 位与 32 位浮点图上对照过，结论限于 ImageJ 1.54p）。要拿 `RawIntDen` 与 ImageJ 逐位对照，先确认该图没有强度校准；`CTCF` 一项两种情况下都与 ImageJ 一致。

测量表每行的末尾还有五列，记录**这一行是用什么校准量出来的**：`PixelWidth`、`PixelHeight`、`SpatialUnit`、`ValueUnit`、`IntensityFunction`（例如 `0.500,0.500,micron,Gray,Straight Line: y = 10 + 2*x`）。同一批图在不同机器或不同会话里算出不同的数值时，先对照这五列。`CTCF` 是校准后的 `Area × Mean` 减去背景项，所以像素尺寸或强度校准一变它就跟着变——例如只把像素尺寸改成原来的一半，同一张图的 `CTCF` 会变成四分之一。

***每张测量表（`<图名>_measurements.csv`）的最后一行固定为这张图的背景值,圈的位置不一定好，建议人工核对一遍再决定要不要采用***

### 与手工测量对照时

在 ImageJ 里手工 `Analyze > Measure` 时，插件的测量条件有两点与它**故意不同**，想在两边对上数，请先关掉这两项：

| `Set Measurements` 里的设置 | 插件的做法 | 对照前请 |
| --- | --- | --- |
| `Limit to Threshold` | **不继承**，统计整个 ROI，不看阈值 | 关闭该项 |
| `Redirect to` | **不继承**，直接测量这一行所写的原图 | 设为 `None` |

`Limit to Threshold` 开着时，手工测量只统计 ROI 内落在阈值区间里的像素，而插件统计整个 ROI，`Area`、`Mean`、`IntDen`、背景均值与 `CTCF` 都可能不同。按完整 ROI 测量是本插件一贯的约定，不是错误；将来若要按阈值测量，会作为独立的模式加入，而不是跟着这个全局开关走。`Redirect to` 则是 0.4.0 起刻意绕开的：它会让结果行的图名与实际取值的图不是同一张，而同一行里的 `RawIntDen` 又不受它影响，界面上没有任何提示。要测别的图，请直接打开那张图再测。

**测量用的是文件上的校准。** 圈虫和测量都从磁盘文件读像素，校准也一样取自该文件，因此在 ImageJ 里用 `Analyze > Calibrate...` 或 `Analyze > Set Scale...` 临时改过校准、还没写回文件时，调用插件会先弹出提示，把窗口与文件两套校准都列出来并说明将采用文件的那一份；点「取消」则什么都不做。要把新校准用于测量，先 `File > Save` 存回文件再重新调用。例外是 `Analyze > Set Scale...` 里的 **Global** 选项：它一旦勾选，所有图都按它算、与各图文件里的校准无关，此时日志里会写明「全局标定生效，覆盖各图自己的校准」。每批测量开始时，ImageJ 日志里还会写一行本次实际使用的校准。

## 常见问题

**菜单里没有 Auto Worm ROI。**
确认 `Auto_Worm_ROI.jar` 在 `<Fiji 根目录>/plugins/` 下，然后重启 Fiji——新装的插件需要重启才会出现在菜单里。

**提示「本模型未针对二值图片进行训练」。**
这是提示不是错误，确认后照常处理。二值图能用，只是分割精度没有保证。

**提示「部分图像未处理」，或者结果表里少了一张图。**
少的那张处理失败了。到输出文件夹 `other` 子目录下的 `batch_summary.csv` 里找 `qc_status` 为 `FAILED` 的行，`error` 列就是原因。

**整批被拒绝，说图像不合格。**
按弹窗里列出的文件名和原因处理：多通道 / 堆栈用 `Image > Stacks > Stack to Images` 拆开，损坏或改名的非 TIFF 文件转成正常的单页灰度 TIFF。

**日志里出现 `Could not remove partial results for ...`。**
有一张失败图的结果文件没删掉（通常是 QC 图正被看图软件打开）。关掉看图软件，照日志里的文件名手动删除。

**图在 ImageJ 里显示成多个切片。**
这是带空 overlay 平面的图，属于放行的情况。测量前确认窗口停在**第 1 个切片**。

**虫数识别不对。**
先确认「图像类型」选的是不是匹配采集条件的模型，再核对「预设虫数 n」。边界粘连严重的图可以试「平滑修复」。都不行就用「手动标注」画头向箭头或人工分界。

**找不到结果。**
默认在**输入图像文件夹下的 `_auto_roi` 子目录**里，也可以在界面里改成别的路径。测量表在它下面的 `measurements`，其余结果在 `other`。

**处理完界面没关，是不是卡住了。**
不是。界面保持打开是设计如此：一批处理完就自动交给 ImageJ 测量，可以接着选下一个文件夹，关闭界面才结束插件。

**总弹「ImageJ 正在测量上一批结果」，可 ImageJ 里其实什么都没在跑。**
桥接目录里还留着上一次的通知文件。正常情况下这些文件由 ImageJ 一侧在测量完成后删掉；如果在测量途中（大批量要跑好几分钟）直接把 ImageJ 关掉，它们就留在磁盘上了，此后每次点「开始处理」都会被这一条挡住，而界面上没有任何办法清除。处理办法：**关掉这个界面窗口，重新启动 ImageJ，再从 Fiji 里运行一次插件**——每次运行插件都会新建一个桥接目录，新的那一个一定是干净的。不想重启 Fiji 的话，也可以手工删掉 `%TEMP%\autoworm-imagej-*\` 里残留的 `*.properties` 文件。

**点了 ✕ 窗口没关，状态栏停在「正在等待当前图片和汇总表安全写出…」。**
再点一次 ✕ 就会正常关闭，**不会丢数据**。这是退出确认对话框正好赶上批次结束时的一个时序问题：那一次点击只记下了「要退出」，而负责收尾的流程当时已经跑完，没有谁再来看这个记号。已经写出的结果都在输出文件夹里，可以放心关。

## 许可与第三方组件

本软件自身的代码与文档按 **MIT 许可**发布，全文见发行包根目录的 `LICENSE`。这层许可只覆盖本项目自己的代码和文档；随软件分发的第三方组件仍按各自的许可分发，不受 MIT 影响。

`AutoWormImageJ\licenses\` 里是随软件分发的第三方组件许可，共五份：

```text
THIRD-PARTY-NOTICES.txt   组件清单、各自许可与许可全文在包内的位置（先读这一份）
NVIDIA-CUDA-EULA.txt      NVIDIA CUDA Toolkit 最终用户许可协议
NVIDIA-cuDNN-SLA.txt      NVIDIA cuDNN 软件许可协议
Pillow-LICENSE.txt        Pillow 许可
roifile-LICENSE.txt       roifile 许可
```

运行包里含 PyTorch、NumPy、SciPy、OpenCV、Pillow、roifile，以及 NVIDIA 的 CUDA 运行库、cuBLAS、cuDNN、cuFFT、cuRAND、cuSOLVER、cuSPARSE、NVRTC、nvJitLink 和 CUPTI 动态库（都在 `_internal\torch\lib\`）。它们各自的许可全文位置见 `THIRD-PARTY-NOTICES.txt`。

NVIDIA 未对本软件作任何形式的认可或担保。使用随软件分发的 NVIDIA 组件时，以下限制适用：不得对它们做反向工程或反编译，不得移除其中的版权声明，不得把它们当独立产品分发，也不得单独再分发这些 DLL，也不得把它们用于使自身受制于要求开放源代码的开源许可。这些库按「现状」提供且不附带担保，NVIDIA 明确声明未针对医疗、生命支持等关键应用做过测试或认证。

**本软件为科研工具，测量结果不得用于临床诊断或任何与生命安全相关的用途。**

## 附录：构建与打包（开发者）

要求 JDK，以及装有 PyInstaller、PyTorch、OpenCV、SciPy、Pillow、roifile 的 Python 环境：

```bat
build_0.4.3.bat "完整路径\python.exe"
```

构建最后一步会把 `licenses\` 整个复制进 `dist\AutoWormImageJ\licenses\`；该目录缺失时构建报错退出，不会静默产出一个没有许可声明的包。

**改过 `src\*.py` 之后必须重新构建**：`AutoWormGUI.exe` 里嵌的是 `src\` 的副本，不重打包的话用户拿到的还是旧代码。打包发布 zip 再跑一次：

```bat
python make_release.py --previous-exe "<上一版的 AutoWormGUI.exe>"
```

它会先核对版本号在 `plugins.config`、`APP_VERSION`、`SOFTWARE_VERSION` 三处一致，再把 EXE 里的每个模块与当前 `src\` 逐个比对（不一致就拒绝打包），然后把四份文档刷新到 `dist\`，最后按 `plugins/` 结构打包成 `release\AutoWorm-<版本>-ImageJ.zip`。给了 `--previous-exe` 时还会要求**旧包必须与当前源码不一致**，否则这个比对说明不了任何问题。

测试样例可用 `python diagnostics\make_test_samples.py` 生成，它会写出覆盖各类不合格图像的合成 TIFF，并用程序自己的预检逐张复核，判定与预期不符即报错退出。

编译所用 `lib/ij.jar` 为 ImageJ 1.54p 的 `ij.jar`。插件只使用 ImageJ 1.x 的公开 API，`javac --release 8` 即可产出可在 ImageJ 1.54p 及 Fiji 同名版本上运行的 class 文件。模型保持为外置文件，便于后续单独替换。

---

# Auto Worm ROI — ImageJ / Fiji (Java 8) Plugin (English)

>This is the English translation of the document above. The Chinese original is authoritative: if the two disagree, follow the Chinese text.
>
>This plug-in batch-draws ROIs around nematodes automatically. Fluorescence quantification and CTCF are still performed by ImageJ/Fiji. The plug-in does not change how fluorescence is measured or how CTCF is computed; it only automates the ROI-drawing step.
>
>The current version (0.4.3 CUDA for ImageJ (Java 8)) is a beta build for users with an NVIDIA GPU whose ImageJ runs on Java 8. See [Beta Notice](#beta-notice) for details.
>
>This plug-in was developed with the assistance of deepseek-V4.1-flash and ChatGPT 5.6 sol, and reviewed by ChatGPT 5.6 sol and ChatGPT 6 Astra.

## What This Plug-in Does

It reads a grayscale TIFF, segments and outlines each nematode in the batch, and outputs them as ImageJ ROIs.

## How Measurement Is Performed

Once the ROIs are generated they are handed back to ImageJ: the plug-in uses ImageJ's own `ImageStatistics` to measure the pixels of each ROI, applies ImageJ's spatial and intensity calibration to obtain `Area`, `Mean`, `IntDen` and so on, then writes the measurement table with `ResultsTable`. `CTCF` is computed from those values (see [ImageJ Measurement and CTCF](#imagej-measurement-and-ctcf) for details).

This path **does not go through `Analyzer`**: `Analyzer` reads the global settings under `Analyze > Set Measurements` (especially `Redirect to`), which can make the image named in a result row differ from the image the values actually came from. See [When Comparing Against Manual Measurement](#when-comparing-against-manual-measurement).

## Beta Notice

0.4.3 CUDA for ImageJ (Java 8) has most of its functionality working on the development machine. Some error paths, however, have only been code-reviewed because of a lack of data, and were not tested on real data. There may also be error conditions I have not thought of; user feedback is welcome.

In addition, the models in this plug-in were trained on an NVIDIA GeForce RTX 4060 Laptop and have not been tested on other NVIDIA GPU models. If you hit hardware compatibility problems, please report them as well.

Please send feedback to my personal email: rivendell118@gmail.com
                    or 3024732774@qq.com
You can also open an issue on GitHub.

## Installation

Hardware requirement: a computer with an NVIDIA GPU.
Software requirement: Fiji or ImageJ 1.x (1.54p or later) must already be installed.

1. Extract the entire .zip into the root folder of your ImageJ/Fiji — the folder that contains `ImageJ-win64.exe` (in Fiji, `fiji-win64.exe`).
2. Restart ImageJ/Fiji.

After extraction it should look like this:

```text
<Fiji root>/
  ImageJ-win64.exe
  plugins/
    Auto_Worm_ROI.jar        the plug-in itself
    AutoWormImageJ/
      AutoWormGUI.exe        the GUI
      models/                segmentation models, external so they can be swapped individually
      _internal/             runtime libraries (Python, PyTorch, CUDA, cuDNN…)
      licenses/              full text of third-party licenses
  README_0.4.3_ImageJ_Java8.md     this file
  CHANGELOG_0.4.3.md
  VALIDATION_0.4.3.md
  LICENSE
```

- Inside `AutoWormImageJ`, `models`, `_internal` and `licenses` are **all required!!!** Do not copy only some of them.
- Do not delete `licenses/`; the reason is in [License and Third-Party Components](#license-and-third-party-components).
- The four documents (README / CHANGELOG / VALIDATION / LICENSE) can go anywhere; their location does not affect operation.
- If you would rather not extract into the root folder, manually copying the two entries under `plugins/` into `<Fiji root>/plugins/` has the same effect.

**End users do not need to install Python, and do not need to install the CUDA Toolkit.** PyTorch, CUDA 13.2, cuDNN, OpenCV, SciPy and the MSVC runtime are all bundled inside `AutoWormImageJ`. The user's machine does, however, need an NVIDIA driver of **version 580 or later**. "Settings > CUDA and system info" in the GUI shows the GPU, driver, PyTorch, CUDA and cuDNN status; the plug-in also runs the same check automatically before it starts processing.

## Quick Start

1. Open ImageJ.
2. Run the plug-in: the first item in the `Plugins > Auto Worm ROI` submenu, named something like `Auto Worm ROI <version>`. The menu text is plain English ASCII, to stay compatible with how Fiji 1.54p reads `plugins.config` under the system encoding; the window itself still offers Chinese or English.
3. In the window, choose the current folder, the image type and the expected worm count, tick "smooth repair" / "manual annotation" / "partial outlining" as needed, then click "Start processing".
4. When processing finishes the window stays open and ImageJ automatically receives and measures this batch's results. You can then pick another folder and keep going.
5. Closing the window ends the plug-in.

Results are written by default into **the `_auto_roi` subfolder of the current folder**, and that location can be changed in the settings window. Under it the files are always split into two subfolders, `measurements` (measurement tables) and `other` (everything else); see [Result Files](#result-files).

## When the Current Image Has Unsaved Changes

After the plug-in is invoked and before the window opens, if the image currently open in ImageJ has unsaved changes (for example you just applied a filter, a crop or an inversion), a prompt appears first: **this run will process the version saved on disk.** Both outlining and measurement read pixels from the file on disk, so unsaved on-screen changes play no part; otherwise the ROIs and the measurement results would point at two images that look the same but are not.

- Click "OK" to continue: this batch is processed from the on-disk version and the unsaved changes are ignored (the image itself is neither modified nor saved).
- Click "Cancel" to do nothing, and the status bar reports that nothing was processed.

To process the changes as well, save the image in ImageJ first and then invoke the plug-in again.

## Parameters in the Window

| Parameter | Purpose |
| --- | --- |
| Current folder | The directory to process. The batch scope is the TIFFs directly under it |
| Image type | `High-clarity image` or `low-clarity image` — two segmentation models trained on different acquisition conditions |
| Expected worm count n | How many worms each image should contain. The detected count must equal it; images that do not match need manual review |
| Smooth repair | Independent contour-quality checking plus conservative shape repair, applied on top of the model's output |
| Manual annotation (head direction / boundary / exclusion) | When enabled, reads the head-direction arrows, manual boundaries and background exclusion regions you drew |
| Partial outlining (requires manual head direction) | Outlines only one segment of the worm body rather than the whole worm |

## Feature Details

### Image Type and Models

The two models were trained under different acquisition conditions: `high-clarity image` and `low-clarity image`. Choosing the wrong one does not raise an error; segmentation quality simply drops. So when you change acquisition hardware or magnification it is worth trying both and comparing each image's `*_QC.png`. The models are ordinary files under `AutoWormImageJ\models\` and can be replaced individually.

### Expected Worm Count n

The program requires that the number of worms detected in each image be **exactly equal to n**. Images whose count matches get `qc_status` `PASS` in `batch_summary.csv`; those that do not are recorded as `REVIEW_COUNT_MISMATCH` and need manual review — note that this verdict only affects the QC flag, and segmentation is never forced to produce n worms. When images in the same folder have different worm counts, process them folder by folder.

### Smooth Repair

This performs contour-quality checking plus safety-gated shape repair on the segmentation result: repair happens only when it will not noticeably change the area and centroid, and if the change would exceed the limit nothing is touched. It suits images with rough or irregular boundaries. When enabled, a `*_shape_qc.csv` is written for each image, whose `status`, `repaired` and `needs_review` fields describe that image's contour quality and whether anything was changed. If any worm was changed or flagged as a questionable contour, that image is shown with a yellow frame in the monitor; the next section explains how frame colours are decided.

### Frame Colours in the Monitor

Every QC image in the monitor has a frame around it, and the colour tells you whether the image is worth opening for a closer look. Only the highest-priority colour is shown: **red > yellow > blue**.

| Frame | Trigger |
| --- | --- |
| Red | The number of worms detected in that image differs from the expected worm count n, i.e. that image's `qc_status` in `batch_summary.csv` is `REVIEW_COUNT_MISMATCH`. The window additionally shows a line "needs manual review \| <original name>" in the top-left corner of that image |
| Yellow | The count is fine, but the segmentation result was rewritten by the program: any of `shape_refined_count` (worms whose shape smooth repair actually changed), `shape_review_count` (worms smooth repair flagged as questionable contours) or `low_clarity_split_count` (number of touching-worm splits in low-clarity mode) is greater than 0 |
| Blue | The currently selected image, unrelated to result quality |

A red frame means the **count** is wrong and must be dealt with: check the expected worm count n, or use manual annotation to add head-direction arrows or manual boundaries. A yellow frame only means **this image is not the model's raw output** — the contour was repaired, or in low-clarity mode a touching block was split into two. It does not mean the result is necessarily wrong (the repair itself is safety-gated and will not act if the change exceeds the limit), but these are the images to look at first when spot-checking by hand.

The yellow frame is based on per-worm records, and the two tables do not have the same columns:

- `<image>_shape_qc.csv`: the `status` column is `REVIEW_SHAPE_REFINED` (the shape really was changed) or `REVIEW_SHAPE` (only flagged, nothing was done). A true `repaired` means the contour really was changed; `needs_review` is true in both states, so it flags a wider set than `repaired`. It also carries the area and centroid shift before and after repair, various quality metrics, and the reasons for the verdict spelled out in the `reasons` column.
- `<image>_split_qc.csv`: each row is one touching-worm split, with `status` `REVIEW_LOW_CLARITY_SPLIT`, plus the split threshold, the head gap width, the boundary support and the area ratio of the two sub-blocks, from which you can judge how trustworthy that split is.

Red and yellow frames are **drawn directly into the `*_QC.png` file**: open it in another image viewer, preview it in a file manager, or send it to someone else, and the frame is still there. The blue frame exists only inside the window, as a highlight of the current selection.

Manual annotation (head-direction arrows / manual boundaries / background exclusion regions) and partial outlining do **not** trigger a yellow frame on their own — the former is a change you specified yourself, and the latter only changes the extracted segment, not how worms are divided. Such images are still judged by the red and blue colours alone.

Images that failed processing do not appear in the monitor: their result files (including `*_QC.png`) are deleted along with the failure, see [Batch Processing, Failures and Summary](#batch-processing-failures-and-summary).

### Manual Annotation

Three kinds of annotation: **head-direction arrows**, **manual boundaries** and **background exclusion regions**.

- `head_1`, `head_2`… straight lines or arrows drawn from inside the worm body towards the head; one per worm.
- `boundary_1`… lines or freehand lines, used to indicate a manual boundary between touching worms.
- `exclude_1`… closed-area ROIs; the interior does not take part in detection, which suits images with dirty background.

Head-direction arrows and manual boundaries are **two peer algorithms that work independently**: with only head-direction arrows, the program roughly splits touching worms by head direction; with only manual boundaries, the program splits or redraws adjacent ROIs directly from the boundary lines; when both are present, head-direction rough splitting always runs first, followed by manual-boundary refinement. Manual-boundary refinement is allowed to redistribute the pixel ownership of adjacent ROIs even when the worm count is already correct, but it never changes the original total set of foreground pixels.

**A manual boundary line is a hard constraint**: where you drew a boundary, the interface between the two adjacent worm segments lands on that line, no longer decided by the model. Snapping happens only within a narrow band around the boundary line (about 4% of the image's short side, roughly 40 pixels on a 1024-pixel image): if the actual gap is further from your line than that, the program keeps the model's division — forcing it would cut a fragment belonging to no worm out of the band, which is worse than a seam that is slightly off.

**Annotating directly in the GUI is recommended** (tick "manual annotation").

How the three annotations work in the window:

- **Head-direction arrow**: press the left button inside the worm body, drag towards the head and release; the tip of the arrow is the head.
- **Manual boundary**: along the gap between two touching worms, from the head end to the tail end, **left-click to place the nodes one by one**; as the mouse moves, a dashed line follows the cursor from the last node; **right-click ends** that boundary, and Esc abandons the one being drawn. The polyline's corners stay exactly where you clicked them and are not rounded off or thinned; once saved it is shown as a cyan polyline, and it is at the same time the boundary the program actually uses.
- **Background exclusion region**: press the left button and draw a loop around the dirty background or other region that needs no detection; releasing closes it automatically.

You can also prepare ROIs in ImageJ's ROI Manager using the prefixes above and run `Plugins > Auto Worm ROI > Save Current Annotation`; the plug-in then writes `<original filename>.autoworm.json` next to the TIFF. A per-image sidecar does not affect the annotations of other images in the same folder; the older folder-level `_manual_head_annotations.json` remains supported.

When both records exist for the same image, they are **merged separately per annotation type**: a non-empty type in the sidecar overrides the folder record, while an empty type falls back to the folder record, so a sidecar that only has head-direction arrows will not lose the manual boundaries drawn for that image in the window. When you save an image in the window, if that image already has a sidecar the same result is written back to the sidecar too, so deletions and changes made in the window take effect immediately.

⚠️ Please do not maintain the annotations of the same image across both interfaces. The following cases cannot currently be done from the ImageJ side:

- The image's boundary exists only in the `_manual_head_annotations.json` written by the GUI, and you delete `boundary_` in the ROI Manager and run `Save Current Annotation` again — the sidecar's `boundaries` is then an empty array, the merge rule falls back to the folder record, and that boundary still takes effect.
- To really delete it, delete it in the GUI and save, or simply delete that image's `<original filename>.autoworm.json`.
- The reverse direction is unaffected: as long as the sidecar contains a non-empty boundary it overrides the boundary for the same image in the folder record, and saving in the window writes the change back to the sidecar.

*So, for convenience, please make your changes in the GUI whenever you can.*

### Partial Outlining

Ticking "partial outlining (requires manual head direction)" forces head-direction annotations to be read and outlines only one segment of the worm body. The two sliders share one directed track: **the left end is the head (0), the right end is the tail (1)**. For example the middle section `0.25–0.75`, the last quarter `0.75–1.00`, the front half `0.00–0.50`.

The engine first detects the complete worm body and its centreline, then extracts the selected segment along the head-to-tail direction. **The background ROI is still based on the complete worm body**, so that the part of the body that was cut away does not leak into the background.

### Batch Processing, Failures and Summary

Every run writes `batch_summary.csv` into the output folder's `other` subfolder, refreshing it after each image, so you still have the completed part if the run is cancelled midway or hits an error.

- **A single failure does not stop the batch.** When an image errors out, the program records the reason and moves to the next one; the remaining images are processed as usual and handed to ImageJ for measurement. The failed one appears in the log as `Failed_<original filename>: ERROR <reason>`, and in `batch_summary.csv` as `Failed_<original filename>` with `qc_status` `FAILED` and the reason in the last column `error` (that column is empty for successful rows). **The original file on disk is not renamed**; `Failed_` is only a notation used in the log and the summary table.
- **A failed image leaves no result files behind.** The program writes `*_RoiSet.zip` before it draws the QC image, so if it fails midway that zip is deleted, along with any measurement table and intermediate reports already written for that image. This matters: ImageJ only measures images that have a `<image>_RoiSet.zip`, so if leftovers were not cleaned up, an image just reported as failed would instead be measured into the result table. In rare cases (for example the QC image is open in an image viewer and cannot be deleted) the deletion fails; the log then adds a line `Could not remove partial results for ...` and you should delete it by hand as it says.
- **Progress.** Each finished image prints a line `Progress: 3/12 (25%)`. The failed image counts towards progress too, so you can tell how much is left from the log window.
- **Duplicate result names are rejected.** Result files are named after the input filename with the extension removed, so `a.tif` and `a.tiff` in the same folder would write to the same set of filenames and overwrite each other. When this is detected the whole batch is rejected before processing starts, with the conflicting filenames listed, rather than silently overwriting.
- **Failed images do not vanish silently on the ImageJ side.** After measurement finishes, if the batch contained failed images the plug-in pops up "some images were not processed" and lists each `Failed_<original filename>` with its reason, and writes the same to ImageJ's log window.

## Supported Image Formats

Only **single-channel, single-plane, single-time-point** grayscale TIFFs are processed: 8-bit and 16-bit are both fine, and **16-bit big-endian (`I;16B`) is supported as well**.

Colour TIFFs (RGB, RGBA, palette, grayscale with alpha, CMYK) and multi-page TIFFs (multi-channel, Z-stacks, time series, including ImageJ hyperstacks and OME-TIFF) are rejected; so are non-TIFF files such as PNG or BMP renamed to `.tif`.

**Exception: images whose extra planes are entirely 0 are allowed through.** Acquisition software such as MetaMorph attaches an empty overlay plane to each image; the real data in such files is still on the first page, the extra plane carries no signal and cannot produce a wrong value, so it is not rejected — the log simply says so, for example `Phsp16.2 0101-70-1.tif: 1 empty extra plane(s); treated as overlay`. Note that ImageJ does **not** skip this plane: it displays the image as 2 slices, so before measuring make sure the window is on slice 1.

The reason for rejection is that the program reads only the **first page** of the file. In a batch, apart from the one image currently open in ImageJ, every image is reopened from disk by ImageJ and its first plane measured, consistent with the program; but **the one currently open** (the one the plug-in handed to the original interface) is measured by ImageJ at whatever channel, Z slice and time point the window is currently showing. As long as it is sitting on another plane, the ROIs drawn and the values actually measured will point at different planes, and **no error is raised anywhere**. The check therefore runs before processing starts and covers the whole batch: if even one image fails to qualify, the filenames are listed grouped by reason and the whole batch is rejected, rather than running part of it and then stopping. There is a cap on how many filenames are listed at once; above it a line "N images in total failed" is added, so the count is never under-reported.

For multi-channel or stack data, first split it into single-page TIFFs with ImageJ's `Image > Stacks > Stack to Images`, keep only the channel you want to measure, and process again.

Binary (1-bit) images are not among the formats above, but they are **not rejected**: before starting, the program points out that "this model was not trained on binary images" and, once confirmed, processes them as usual. Both models were trained on 8-bit and 16-bit acquisitions, so segmentation of binary images may be inaccurate; the notice stays, it just does not block the flow.

## File and Folder Naming

Result files are all generated from the input filename with its extension removed (see the next section), so the input name directly determines the output name. There are two restrictions:

**Do not use square brackets `[` `]` in either filenames or folder names.** Before processing starts the program lists the input folder in full, and that step uses wildcard matching, in which square brackets are a **character set**: when a **folder name** is `[batch2]`, it is understood as "any one of the six characters b, a, t, c, h, 2", so not a single TIFF matches and the whole batch is rejected with `No TIFF images found in: ...` — even though the window has clearly already listed the images (the window uses a different directory-listing method), so the symptom and the message do not match and it is very easy to mistake for a broken program. Square brackets in image **filenames** are not currently blocked, but it is best to avoid them as well, so the same class of problem does not come back later.

**Do not put spaces at the start or end of a filename.** Processing itself runs through fine, but the measurement step compares the filename string against the name reported by the batch character by character, and one extra space at either end means no match: that image is **silently skipped** — no measurement result and no warning at all. Spaces in the **middle** of the name are fine (`0716-0-1 green.tif` is named that way).

## Result Files

`<image>` means the input filename with its extension removed. The output directory is always split into two subfolders: **measurement tables go in `measurements`, everything else in `other`**, so the few tables you take away for statistics always stay together and are not mixed up with preview images and QC details.

| File | Written by | Location | Contents |
| --- | --- | --- | --- |
| `<image>_measurements.csv` | ImageJ | `measurements` | That image's measurement table |
| `ImageJ_measurements_all.csv` | ImageJ | `measurements` | Merged measurement table for the whole batch |
| `batch_summary.csv` | Outlining | `other` | One row per image: detected count, `qc_status`, elapsed time, failure reason (the `error` column) |
| `<image>_RoiSet.zip` | Outlining | `other` | That image's ROI set; ImageJ reads it to measure |
| `<image>_QC.png` | Outlining | `other` | Preview of the outlining result, with the segmentation overlaid and the QC status written on it; worth spot-checking |
| `<image>_head_annotations.csv` | Outlining | `other` | Matching result of the head-direction annotations |
| `<image>_manual_split_qc.csv` | Outlining | `other` | Details of manual-boundary refinement, written only when that image's boundary actually changed pixel ownership |
| `<image>_shape_qc.csv` | Outlining | `other` | Contour quality before and after smooth repair |
| `<image>_split_qc.csv` | Outlining | `other` | Details of touching-worm splits in low-clarity mode |
| `<image>_segment_qc.csv` | Outlining | `other` | Segment details for partial outlining |

**Versions before 0.4.2 laid all of the files above flat in the output folder.** This version understands only the new layout: when you invoke the plug-in to measure, ROI sets are looked for only under `other`; if none is found it reports "ROI ZIP not found" and skips that image, and measurement tables always go into `measurements` (created automatically if the folder does not exist). To measure a folder left over from an older version, first move the files into the two subfolders as in the table above, or simply run the whole thing again.

The four QC detail tables `<image>_manual_split_qc.csv`, `<image>_shape_qc.csv`, `<image>_split_qc.csv` and `<image>_segment_qc.csv` are **written only when the corresponding feature is enabled**; when it is not, any file of the same name left by a previous run is deleted along the way, so you do not end up looking at stale data.

Do not confuse `<image>_head_annotations.csv` with the annotation input files: `_manual_head_annotations.json` (in the **image folder**) and `<image>.autoworm.json` (next to the TIFF) are the annotation data itself, not results.

## ImageJ Measurement and CTCF

Each ROI is measured by ImageJ itself for `Area`, `Mean`, `Min`, `Max`, `IntDen`, `Median` and `RawIntDen`. The plug-in additionally outputs:

```text
BackgroundCorrectedMean = worm Mean - background Mean
CTCF = worm IntDen - worm Area × background Mean
RawCTCF = worm RawIntDen - worm PixelArea × background raw Mean
```

`CTCF` respects ImageJ's spatial and intensity calibration; `RawCTCF` uses raw pixel values (`RawCTCF` is the algorithm used by older development versions and was abandoned once the released build was wired into ImageJ. I kept it anyway because I did not want to delete it). `MeasurementBackend` in the measurement table records the ImageJ version actually used.

**The column order is fixed: `Area`, `Mean`, `Min`, `Max`, `IntDen`, `Median`, `RawIntDen` occupy columns 1 to 7**, followed by `Index`, `Image`, `Label`, `PixelArea`, `BackgroundCorrectedMean`, `CTCF`, `RawCTCF`, `Type` and `MeasurementBackend` in that order. The seven values you take straight into a statistical analysis are all at the front, so importing into Excel, R or pandas needs no picking by column name. `<image>_measurements.csv` and `ImageJ_measurements_all.csv` use the same column order. The image's calibration information does not take part in this ordering and is still pinned to the very end of the row (see below).

`RawIntDen` and `RawCTCF` in the table come from the same set of raw values: the plug-in takes `getRawStatistics()`, i.e. the **true sum of raw pixels**, and this column serves only the reference quantity `RawCTCF` and is used nowhere else. It therefore differs from ImageJ 1.54p's **column of the same name** on images that have an intensity calibration — ImageJ's column computes "pixel count × calibrated mean"; on images without an intensity calibration the two agree (this was cross-checked on 8-bit, 16-bit and 32-bit float images this round, and the conclusion is limited to ImageJ 1.54p). To compare `RawIntDen` against ImageJ value by value, first confirm the image has no intensity calibration; `CTCF` agrees with ImageJ in both cases.

Each row of the measurement table also ends with five columns recording **what calibration this row was measured with**: `PixelWidth`, `PixelHeight`, `SpatialUnit`, `ValueUnit`, `IntensityFunction` (for example `0.500,0.500,micron,Gray,Straight Line: y = 10 + 2*x`). When the same batch of images yields different values on a different machine or in a different session, compare these five columns first. `CTCF` is the calibrated `Area × Mean` minus the background term, so it follows any change in pixel size or intensity calibration — for example halving the pixel size alone turns the `CTCF` of the same image into a quarter of what it was.

***The last row of each measurement table (`<image>_measurements.csv`) is always that image's background value. The circle placement there is not necessarily good, so please check it by hand before deciding whether to use it.***

### When Comparing Against Manual Measurement

When you measure by hand with `Analyze > Measure` in ImageJ, the plug-in's measurement conditions **deliberately differ** in two respects; to get the two to agree, turn both off first:

| Setting under `Set Measurements` | What the plug-in does | Before comparing |
| --- | --- | --- |
| `Limit to Threshold` | **Not inherited**; measures the whole ROI regardless of threshold | Turn it off |
| `Redirect to` | **Not inherited**; measures directly the original image written in that row | Set it to `None` |

With `Limit to Threshold` on, a manual measurement counts only the pixels inside the ROI that fall within the threshold range, while the plug-in counts the whole ROI, so `Area`, `Mean`, `IntDen`, the background mean and `CTCF` can all differ. Measuring the complete ROI is this plug-in's consistent convention, not an error; if threshold-based measurement is wanted later it will be added as a separate mode rather than following this global switch. `Redirect to` was deliberately avoided from 0.4.0 onwards: it makes the image named in a result row differ from the image the values came from, while `RawIntDen` in the same row is unaffected by it, and nothing in the interface tells you. To measure a different image, open that image and measure it directly.

**Measurement uses the calibration on the file.** Both outlining and measurement read pixels from the file on disk, and the calibration comes from that same file; so if you have temporarily changed the calibration in ImageJ with `Analyze > Calibrate...` or `Analyze > Set Scale...` and have not written it back to the file, invoking the plug-in first pops up a notice listing both the window's and the file's calibration and stating that the file's will be used; clicking "Cancel" does nothing. To apply the new calibration to measurement, do `File > Save` to write it back to the file and invoke the plug-in again. The exception is the **Global** option in `Analyze > Set Scale...`: once it is ticked, every image is computed by it regardless of the calibration in each file, and the log then states that "the global calibration is in effect and overrides each image's own calibration". At the start of each measurement run ImageJ's log also records the calibration actually used.

## Frequently Asked Questions

**There is no Auto Worm ROI in the menu.**
Check that `Auto_Worm_ROI.jar` is under `<Fiji root>/plugins/`, then restart Fiji — a newly installed plug-in only appears in the menu after a restart.

**It says "this model was not trained on binary images".**
That is a notice, not an error; confirm it and processing continues as usual. Binary images can be used, only the segmentation accuracy is not guaranteed.

**It says "some images were not processed", or the result table is missing an image.**
The missing one failed. Look in `batch_summary.csv` under the output folder's `other` subfolder for the row whose `qc_status` is `FAILED`; the `error` column gives the reason.

**The whole batch was rejected, saying the images do not qualify.**
Deal with the filenames and reasons listed in the dialog: split multi-channel or stack data with `Image > Stacks > Stack to Images`, and convert damaged or renamed non-TIFF files into proper single-page grayscale TIFFs.

**The log shows `Could not remove partial results for ...`.**
One failed image's result files could not be deleted (usually the QC image is open in an image viewer). Close the viewer and delete them by hand using the filename in the log.

**The image shows as several slices in ImageJ.**
This is an image with an empty overlay plane, which is one of the allowed cases. Before measuring, make sure the window is on **slice 1**.

**The worm count is wrong.**
First check that the "image type" matches the model for your acquisition conditions, then check the expected worm count n. For images with heavy boundary adhesion, try smooth repair. If none of that works, use manual annotation to draw head-direction arrows or manual boundaries.

**I cannot find the results.**
By default they are in **the `_auto_roi` subfolder of the input image folder**; you can also change it to another path in the window. Measurement tables are in `measurements` under it, and the rest in `other`.

**The window stays open after processing — is it stuck?**
No. The window staying open is by design: once a batch is done it is handed to ImageJ for measurement automatically, and you can pick the next folder; the plug-in ends when you close the window.

**It keeps saying "ImageJ is measuring the previous batch", but nothing is actually running in ImageJ.**
A leftover notification file from the previous run is still in the bridge directory. Normally these files are deleted by the ImageJ side once measurement finishes; if you close ImageJ outright during measurement (a large batch takes several minutes), they stay on disk, and from then on every click on "Start processing" is blocked by this message with no way to clear it from the window. The fix: **close this window, restart ImageJ, and run the plug-in again from Fiji** — each run of the plug-in creates a fresh bridge directory, and the new one is guaranteed to be clean. If you would rather not restart Fiji, you can also delete the leftover `*.properties` files in `%TEMP%\autoworm-imagej-*\` by hand.

**I clicked ✕ and the window did not close; the status bar is stuck on "waiting for the current image and summary table to be written safely…".**
Click ✕ once more and it closes normally, **with no data loss**. This is a timing issue where the exit-confirmation dialog happens to coincide with the end of a batch: that click only recorded "please exit", while the finishing routine had already completed and nobody came back to look at the flag. Everything already written is in the output folder, so it is safe to close.

## License and Third-Party Components

This software's own code and documentation are released under the **MIT License**, the full text of which is in `LICENSE` in the root of the distribution package. That license covers only this project's own code and documentation; third-party components distributed with the software are still distributed under their own licenses and are unaffected by MIT.

`AutoWormImageJ\licenses\` holds the third-party component licenses distributed with the software, five in total:

```text
THIRD-PARTY-NOTICES.txt   component list, each one's license, and where its full text sits in the package (read this one first)
NVIDIA-CUDA-EULA.txt      NVIDIA CUDA Toolkit End User License Agreement
NVIDIA-cuDNN-SLA.txt      NVIDIA cuDNN Software License Agreement
Pillow-LICENSE.txt        Pillow license
roifile-LICENSE.txt       roifile license
```

The runtime bundle contains PyTorch, NumPy, SciPy, OpenCV, Pillow and roifile, plus NVIDIA's CUDA runtime, cuBLAS, cuDNN, cuFFT, cuRAND, cuSOLVER, cuSPARSE, NVRTC, nvJitLink and CUPTI dynamic libraries (all under `_internal\torch\lib\`). Where each one's full license text lives is given in `THIRD-PARTY-NOTICES.txt`.

NVIDIA does not endorse or warrant this software in any way. When using the NVIDIA components distributed with the software, the following restrictions apply: they must not be reverse-engineered or decompiled, their copyright notices must not be removed, they must not be distributed as a standalone product, these DLLs must not be redistributed separately, and they must not be used in a way that subjects you to an open-source license requiring you to open your source. These libraries are provided "as is" and without warranty, and NVIDIA expressly states that they have not been tested or certified for medical, life-support or other critical applications.

**This software is a research tool; its measurement results must not be used for clinical diagnosis or any life-safety-related purpose.**

## Appendix: Building and Packaging (Developers)

Requires a JDK, and a Python environment with PyInstaller, PyTorch, OpenCV, SciPy, Pillow and roifile installed:

```bat
build_0.4.3.bat "full path\python.exe"
```

The last build step copies the whole `licenses\` directory into `dist\AutoWormImageJ\licenses\`; if that directory is missing the build fails with an error rather than silently producing a package with no license notices.

**After changing `src\*.py` you must rebuild**: `AutoWormGUI.exe` embeds a copy of `src\`, so without repackaging the user still gets the old code. To package the release zip, run:

```bat
python make_release.py --previous-exe "<previous version's AutoWormGUI.exe>"
```

It first checks that the version number agrees in `plugins.config`, `APP_VERSION` and `SOFTWARE_VERSION`, then compares every module inside the EXE against the current `src\` one by one (refusing to package on any mismatch), then refreshes the four documents into `dist\`, and finally packages everything into `release\AutoWorm-<version>-ImageJ.zip` in the `plugins/` layout. When `--previous-exe` is given it additionally requires that **the old package differ from the current source**, otherwise the comparison proves nothing.

Test samples can be generated with `python diagnostics\make_test_samples.py`, which writes synthetic TIFFs covering every kind of non-qualifying image and re-checks each one with the program's own pre-check, failing with an error if a verdict does not match expectations.

The `lib/ij.jar` used for compilation is ImageJ 1.54p's `ij.jar`. The plug-in uses only ImageJ 1.x's public API, and `javac --release 8` is enough to produce class files that run on ImageJ 1.54p and Fiji versions of the same name. The models stay as external files so they can be swapped individually later.
