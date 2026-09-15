# 自动圈虫 0.4.1 CUDA for ImageJ 验证记录

验证日期：2026-09-14（2026-09-15 三轮复审修复后重建，见「第一轮 / 第二轮 / 第三轮复审修复的验证」；同日再修 0.4.1 的取样约定差异，见「0.4.1：ROI 改按像素边界导出」）

## 结果

- Python 回归测试：98/98 通过（首轮新增 12 项，见下；第二轮、第三轮修复后增到 110/110，0.4.1 的取样约定修复再增 2 项 = 112，综合复审两轮修复后增到 131，只读数组修复后 133，人工分界线压线后 **136/136**，见各自小节）。
- 套件完整性验证通过：确认 `batch_worm_roi`、`inspect_roi_dataset`、`worm_roi_gui` 确实从本仓库 `src\` 导入。此前**不是**这样：`tests\test_erosion_0101.py` 名字匹配 `test_*`，被 discovery 导入，并在模块级把旧工作区（`备份\0910 0.3.2 CUDA\src`）插到 `sys.path` 最前面后 `import inspect_roi_dataset`，使 0.3.2 的那份占据了 `sys.modules`。实测该进程里 `inspect_roi_dataset.__file__` 指向旧工作区的 `备份\0910 0.3.2 CUDA\src\inspect_roi_dataset.py`，而此前整套测试一直是全绿——也就是说，「全部通过」这个结论当时并不成立，测的不是本仓库的代码。该文件已改名为 `erosion_0101_probe.py`（保留内容并加说明注释），`test_shape_refiner_032.py` 同样只注册 0 个用例，一并改名为 `shape_refiner_032_probe.py`；新增 `SuiteIntegrityTests` 断言模块来源，防止再次静默发生。
- 图像格式预检验证通过：多页 TIFF 判为 `stack`、RGB 判为 `colour`、改名的非 TIFF 判为 `format`、损坏文件判为 `unreadable`，均被拒绝；单层灰度 `I;16`（小端）、`I;16B`（大端）、8 位与二值图全部放行，不误杀。
- 空 overlay 平面例外验证通过：多余平面全为 0 的多页 TIFF 被判为可用并在日志里写明空平面数；第 2 页或更靠后的页只要含非零像素就仍判为 `stack`，逐页检查而非只看第 2 页。
- 预检按真实读取路径判定：会实际解码一次像素，因此被截断的半拷贝 TIFF（文件头正常、解码时才失败）能在处理开始前被拦下，不会跑到一半才中断。
- 整批拒绝验证通过：含多页 TIFF 的目录会抛出 `ValueError` 并点名文件，且在拒绝时连输出目录都不会创建。
- 用本机 37 个真实样例 TIFF（22 个 `I;16` + 15 个 `I;16B`）回归预检，误杀 0 个；全解码耗时约 52 ms/张。
- 用 `补充样例2` 的 30 个真实 TIFF 回归预检，30/30 放行，其中 6 个 MetaMorph 文件按空 overlay 平面例外放行并各记录 1 条说明。
- RTX 4060 Laptop GPU 上跑通三例端到端：真实单层文件夹 2 张全部 `PASS`；同一文件夹加入 MetaMorph 空 overlay 平面文件后 3 张全部 `PASS`（该文件此前会被拒绝，现在能正常跑出结果）；放入真正的 2 页堆栈时在处理开始前被拒，且输出目录未被创建。
- 单张失败不终止整批验证通过（RTX 4060 Laptop GPU、真实权重、3 张真实 TIFF）：强制让第 2 张失败后，第 3 张照常跑完并 `PASS`；日志依次出现 `Failed_b.tif: ERROR RuntimeError: forced mid-batch failure`、`Progress: 1/3 (33%)`、`2/3 (66%)`、`3/3 (100%)` 与 `Failed images (1): Failed_b.tif`，失败图不生成 ROI ZIP。
- 汇总表列对齐验证通过：`batch_summary.csv` 共 36 列；失败行 `image` 为 `Failed_b.tif`、`qc_status` 为 `FAILED`、`error` 为失败原因；成功行 `error` 为空，`run_id`、`device`、`software_version`、`measurement_backend` 均落在各自列名下。该批的 `process_image` 是真实返回值（29 项），因此这不是桩数据的巧合。
- 结果重名拒绝验证通过：同一文件夹下 `a.tif` 与 `a.tiff` 会在处理开始前被整批拒绝并列出冲突文件名，不静默覆盖。
- 二值图像提示验证通过：1 位 TIFF 通过预检进入处理，同时带 `binary` 提示，后续流程不受影响。
- 二值图像崩溃修复验证通过：修复前，1 位 TIFF 能通过预检并在带提示的情况下进入处理，随后以 `TypeError: numpy boolean subtract, the '-' operator, is not supported` 失败 —— 也就是说预检放行了它，批处理却处理不了它。修复后同一张图在真实 CUDA 流水线中跑完并出结果。新增回归测试断言 `image_array()` 对 1 位图返回 `uint8` 且取值集合为 `{0, 255}`；该测试的第一版曾因样例全零而失败（`[0] != [0, 255]`），说明它确实在检查展宽而不是在空转。这条测试后来还意外充当了套件完整性问题的探针：它在完整套件里失败、单独运行却通过（见下），因为完整套件里它拿到的 `image_array` 不是本仓库的那一份。
- 中途失败图不再被 ImageJ 误测，验证通过：样例文件夹里放一张正常图和一张预检放行、处理中途才失败的畸形尺寸图（1×512 / 512×1 / 1×1 会触发 `ValueError: x1 must be greater than or equal to x0`，属于真实可复现的中途失败，不是造出来的桩异常）。修复前，失败图已写出的 `RoiSet.zip` 与 `measurements.csv` 会留在输出文件夹里，而 ImageJ 只测量存在 `<图名>_RoiSet.zip` 的图，于是它会把这张刚被报为 `FAILED` 的图测进结果表。修复后输出文件夹里只剩正常图的那一份结果。新增单元测试断言失败图的全部 8 种结果后缀都被删除，而与该图同名的无关文件（`bad_notes.txt`）和名称前缀更长、不属于它的文件（`other_QC.png`）保持不动。
- 畸形尺寸图像的处理方式本身是合适的，未做修改：1×1、1×512、512×1 这类图通过预检后会在处理中途失败，被记为失败图而不影响同批其余图像。这几张图因此成为「中途失败」这条路径可复现的样例来源。
- 汇总表行宽与表头逐行核对：新增断言要求每一行的字段数等于表头列数，并断言成功行的 `error` 列为空、`software_version` 列确为版本号；此前若某行少一列，CSV 仍能被正常解析，只是错位，这类错误不会被长度检查之外的任何方式发现。
- 人工标注算法顺序验证通过：人工头向与人工分界可分别独立运行；同时启用时固定先执行头向粗分，再执行人工分界精修。
- 人工标注来源合并验证通过：同一张图同时存在 ImageJ sidecar 与 GUI 文件夹标注时按标注类型合并，只含头向箭头的 sidecar 不再丢弃该图在 GUI 中保存的人工分界；sidecar 中非空的一类仍优先。
- GUI 写回 sidecar 验证通过：GUI 删除分界后 sidecar 同步清空、GUI 改动分界后 sidecar 同步更新；`boundaries` 缺省保存不会清掉只存在于 sidecar 的分界；同名多图格式文件不会被写回覆盖。
- 已知限制（见 README“手动标注”）：在 ImageJ 侧删除只存在于 GUI 文件夹记录中的分界不会生效，需在 GUI 中删除或直接删除该图的 `.autoworm.json`。
- 人工分界精修报告验证通过：分界已被当前划分满足时不再写入名不副实的 `manual_boundary_refine` 记录，输出与输入逐像素相同。
- `0513-0-1 green` 问题样例回归通过：组合模式与仅人工分界模式均保持 10 条虫和前景像素集合不变，最左侧两条虫的 ROI Dice 分别由约 0.861/0.821 提升至 0.917/0.886。
- `N2-2` 真实样例回归通过：头向粗分后人工分界精修，10/10 个头向标注匹配。
- 新增 ImageJ 集成测试：逐图 `.autoworm.json` 读取通过；ImageJ 后端只导出 ROI、不生成 Python 测量 CSV，通过。
- Java 插件：使用官方 `ij.jar` 编译成功，目标 class 版本 52（Java 8 兼容）。
- 插件侧部分失败提示已写入源码并通过编译：测量完成后若桥接清单的 `message_b64` 非空，插件弹出「部分图像未处理」并列出各失败图与原因；该路径依赖运行中的原界面，未做自动化测试，见下文人工界面验收。
- 使用 ImageJ 1.54p 的 `ij.jar`（Java 8）重新编译并验证；插件能从 Fiji 成功启动原 GUI。
- `plugins.config` 已验证为纯 ASCII，消除 Fiji 默认字符集造成的菜单乱码。
- 原 GUI ImageJ 桥接自检：成功写出完成清单，推理摘要为 `measurement_backend=imagej`，Python 测量 CSV 不存在。
- 终端面板布局验证通过：将“保存日志”按钮收进终端的全宽标题栏后，标题栏、终端正文、滚动条和监视器面板右缘对齐；隐藏 GUI 结构检查通过，并已对实际渲染界面截图复核。
- CUDA 环境检测：RTX 4060 Laptop GPU、驱动 596.49、PyTorch 2.14.0+cu132、CUDA 13.2、cuDNN 92400，检查通过。
- ImageJ 无界面测量冒烟测试：成功读取 ROI ZIP，得到 11 行结果（10 条虫 + 1 个背景），包含 `CTCF` 和 `MeasurementBackend`。
- 本次重建后的打包 EXE 已在隐藏界面模式下运行真实样例 `0710-0-1 green.tif`：进程退出码为 0，检出 10 条且 `PASS`，桥接清单状态为 `complete`；输出仅含 ROI ZIP、QC PNG、分割报告与批次摘要，没有 Python `*_measurements.csv`。随后使用同一批输出重新编译并运行 Java 测量冒烟测试，成功得到 11 行结果（10 条虫 + 1 个背景）。MSVC CRT 同步完成。
- RTX 4060 Laptop GPU 样例端到端：`0716-0-1 green.tif` 检出 10 条，预设 10 条，`PASS`。
- ROI-only 边界确认：引擎输出目录只有 ROI ZIP、QC PNG 和批次摘要；ImageJ 测量前不存在 `*_measurements.csv`。
- 最终发布文件存在：`dist/Auto_Worm_ROI.jar` 与 `dist/AutoWormImageJ/AutoWormGUI.exe`。
- 插件 jar 因状态栏文案改动（6 处 `IJ.showStatus` 改为英文，见 CHANGELOG）重新编译，并逐项核对过改动范围：新旧 jar 的条目集合相同（7 项），除 `Auto_Worm_ROI.class` 外全部逐字节相同；两份 class 的常量池按**与下标无关**的方式解析后（每条引用解析成类名/方法名/字段名/描述符/数值）比对，差异只有那 6 处字符串字面量，被引用的类、方法、字段、描述符与数值常量完全一致——这一点必须这样验，因为字符串条数变化会让常量池下标整体平移，按下标逐条比对会报出大量假差异。另做了一次**可复现性证明**：用未改动过的源码（`git show HEAD:plugin_src/*.java`）重新编译，产物与本发布包内的 4 个 class **逐字节相同**，说明本机 javac 是确定性的，因此「只差字符串」这一结论成立，而不是「恰好没看出别的差异」。编译命令与 `build_0.4.1.bat` 第 1 步一致：`javac --release 8 -encoding UTF-8 -cp lib/ij.jar`，class 文件主版本仍为 52（Java 8），可在 ImageJ 1.54p 与 Fiji 同名版本上运行。本轮没有改 Python 代码，`AutoWormGUI.exe` 未重建、哈希不变。
- 第三方许可已随包分发并逐份核对：`dist\AutoWormImageJ\licenses\` 内 5 个文件的 SHA256 与仓库 `licenses\` 逐一相同（见“最终文件校验”）。
- NVIDIA 两份许可文本是官方网站原文，不是转述：从 `docs.nvidia.com` 取回后只去掉网页导航与页脚。抽取脚本对结果逐条断言——正文首行、末行必须是协议的标题行与签署版本行，正文里必须能找到关键条款（CUDA 侧的 `2.6. Attachment A` 与其中的 `cudart.dll`、`nvrtc.dll, nvrtc-builtins.dll`、`cupti.dll`；cuDNN 侧的补充条款与其「the runtime files .so and .dll」），且不得含 `Skip to main content`、`Choose version`、`Privacy Policy`、`Table of Contents` 等任何网页字样。断言不通过就报错退出，不会写出文件。
- 许可复制步骤经实际运行验证：当前 `dist\AutoWormImageJ\licenses\` 由 `postbuild.py` 生成，不是手工拷贝；该步骤在源目录缺失时报错退出而不是跳过。
- 许可覆盖范围按包内实际内容核定，不是照抄依赖清单：逐个列出 `_internal\torch\lib\` 下 23 个 NVIDIA 动态库并与 CUDA Toolkit EULA 的 Attachment A 对照，确认全部属于可再发行项（该附件允许文件名带版本号与架构信息）；确认 cuDNN 不在该附件内、改按 cuDNN 许可协议补充条款第 2 节处理；并检查全包其余位置的 dll，确认没有漏在 `torch\lib` 之外的 NVIDIA 运行库。
- 合成测试样例生成器自校验通过，并且**证明了它不是空转的**：`diagnostics/make_test_samples.py` 生成 10 个文件夹共 28 张合成 TIFF，覆盖正常（8 位 / 16 位小端 / 16 位大端 / 32 位整型 / 32 位浮点）、含信号的多页栈、5 种彩色模式、截断与损坏与空文件、改名的非 TIFF、空 overlay 平面、二值、同名不同扩展名、混合目录、畸形尺寸。生成后逐张用程序自己的 `check_image()` 复核，判定与预期不符即报错退出。为了确认这套复核真的会失败，特意破坏了文件内容、`note` 与 `warning` 三处预期，每次都确实报错退出后才恢复。
- 上述样例被逐文件夹喂给真实的 `run_gui_batch`（RTX 4060 Laptop GPU、真实权重），而不是只调判定函数。两个 bug 都是这一步才暴露的：二值图崩溃只在处理路径上出现，失败图的残留 `RoiSet.zip` 也只有在真实跑完一批之后才会留在文件夹里。
- Java 桥接新增单元测试（该段此后由「清单」改为「通知队列」，测试一并改写并扩充，见「本轮复审修复的验证」）：所有字段经 Base64 往返后仍是原 UTF-8 文本（含中文与多行消息），且写出的通知文件本身可被 ASCII 解码；写入过程不残留 `.tmp`；未设桥接环境变量时按独立模式运行、不写任何文件；路径不可写时报告错误而不是抛出异常；失败清单的 10 条上限与末尾「共 N 张」计数。
- 发布打包脚本的比对**经证伪后**才被采用：本轮重建前的 EXE 与当前源码比对，得到 12 个模块相同、1 个不同，差异恰好是本轮修改的 `worm_roi_gui`；新 EXE 则为 13 个模块全部相同。若旧包也全部相同，这个比对就说明不了任何事，因此 `--previous-exe` 参数被设计成在该情形下直接报错退出。
- 版本号一致性检查经实测有效：`plugins.config`、`APP_VERSION`、`SOFTWARE_VERSION` 三处任一不同即拒绝打包，不会出现菜单显示一个版本、汇总 CSV 记录另一个版本的情况。

## 第一轮复审修复的验证（2026-09-15）

本轮按一份外部复审意见改动，五处修复逐条先核实、再实现、再验证。以下证据都是**实际执行**得到的，不是代码走查结论。

- **改写测量路径不改变数值**（对应 `Analyzer` → `getStatistics`）：在一个带界面的探针程序里对同一张合成 16 位图、同一组 ROI 分别走旧路径（`new Analyzer(image, MEASURES, table).measure()`）与新路径（`image.getStatistics(MEASURES)`），逐项打印。四种 ROI 形状——矩形 `Roi`、`OvalRoi`、`PolygonRoi`、`ShapeRoi`——在未标定与已标定两种情况下，Area / Mean / Min / Max / Median / IntDen **六项全部逐位相同**。未标定例：`Area=2700.0000 Mean=2021.337037 Min=1024.0 Max=4095.0 Med=1986.0 IntDen=5457610.0000`；标定后：`Area=337.5000 Mean=2021.337037 IntDen=682201.2500`。因此这次改写是纯粹的行为收敛，不会让历史数据与新数据对不上。
- **全局重定向确实会污染旧路径**（这是上一条存在的原因）：在同一个探针里对另一张图执行 `Analyzer.setRedirectImage(...)` 后再测原图，旧路径的 `Mean` 从 `2021.337037` 变成 `2039.955556`（即另一张图的统计量），而 `getStatistics` 仍是 `2021.337037`；同时 `getRawStatistics()` 不受重定向影响，因此旧代码写出的同一行里 `Mean` 与 `RawIntDen` 来自两张不同的图。修复后该情形不再可能发生。探针同时确认 `Analyzer.setRedirectImage(null)` 能清掉三个相关静态量，但修复不依赖用户或本插件去清——直接从源头绕开。
- **`ImagePlus.changes` 可作为「有未保存改动」的判据**（这是新增提示框的依据）：带界面运行时实测——刚打开的图 `false`；`IJ.run(image, "Invert", "")` 与 `IJ.run(image, "Subtract Background...", "rolling=50")` 之后 `true`；`IJ.save()` 之后回到 `false`；而直接 `getProcessor().putPixel(...)` + `updateAndDraw()` **不会**置位。也就是说它反映的正是 ImageJ 命令层造成的改动，是需要提示的那一类；这也说明**不能**用 `putPixel` 来伪造该提示。另核实 ImageJ 1.54p 的 `ImagePlus` 没有 `isModified()` 方法（`javap` 确认），因此只能用这个公开字段。无界面模式（`-Djava.awt.headless=true`）下该标志对任何操作都为 `false`，探针第一次就是这样跑出错误结论的，改回带界面后结论才成立——这一点记在这里，免得以后有人又用 headless 去试。
- **桥接通知队列**：新增 7 项单元测试——第二批通知不会覆盖第一批（目录里两个文件按名排序为写入顺序）；`.properties.tmp` 不会被当作通知读走；`_imagej_bridge_pending()` 仅在目录里还有未读通知时为真；`ok_count` / `ok_b64` 能够往返且逐名一致；空清单仍写成 `ok_count=0` 而不是不写；不传清单时**不写出**这两个键（旧规则得以保留）；`_successful_images()` 正确排除失败图、在拿不到清单时返回 `None`。
- **完成清单的键名与文件名字节一致**：`ok_b64` 里是换行分隔的**输入文件夹原始文件名**，插件按同一文件夹列出的名字原样比对，不做大小写折叠、不比较去扩展名的短名，因此不存在「同名不同扩展名」被错配的可能（`A.tif` 与 `A.tiff` 会在处理开始前被拒绝，见上文）。
- **发布包指纹加强**（对应「EXE 与 `src/` 一致性」检查）：旧指纹只取 `co_name` / `co_names` / `co_varnames` 与字符串常量，因此对**数字**改动完全失明。四个实验：① `x + 1` → `x - 99`，旧指纹判为相同、新指纹判为不同；② `CUDA_MIN_DRIVER_MAJOR = 580` → `579`，同样如此（这一改动的名字与字符串均未变，字节码也逐字节相同，只有常量值变了）；③ 22 个 `src\*.py` 编译后经 `marshal` 往返（即 EXE 内实际存放的形态）指纹**全部一致**，说明加强没有引入假阳性；④ 同一段代码反复执行 20 万次（触发 3.13 的字节码特化）后指纹仍与首次编译时一致，且再次 `marshal` 往返仍一致。加强后还暴露并修掉了一个**真实的假阳性**：`repr(frozenset)` 的打印顺序取决于内部散列表的填充顺序，编译时按源码顺序、从 `.pyc` 读回时按存储顺序，`evaluate_shape_refiner.py` 里的四项 `frozenset` 因此被判为「与源码不同」；现在集合元素先各自渲染成字符串再排序，与顺序无关（元组仍按顺序比较）。这条也已写成单元测试。
- **加强后的指纹在真实 EXE 上复跑**：现存（本轮重建前）EXE 与当前 `src/` 比对——**12 个未改动模块全部判为相同，只有 `worm_roi_gui` 判为不同**，与「本轮只改了一个 Python 文件」逐一吻合；若加强引入了假阳性，未改动的模块会一起报 `differ`。
- **本轮重建后的完整打包校验**（`make_release.py --previous-exe`，一次性跑完）：新 EXE **13 个模块全部与 `src/` 相同**；上一轮 EXE 12 个相同、1 个不同（差异恰为 `worm_roi_gui`），因此「相同」这个结论不是空的；版本号三处一致；四份文档刷新到 `dist\`。任何一项不通过都会在打包前退出。
- **重建后的 EXE 跑通真实样例**（不是只跑单元测试）：把桥接目录指向一个空文件夹，按插件的调用方式设 `AUTOWORM_IMAGEJ_MODE=1` 与 `AUTOWORM_IMAGEJ_BRIDGE_DIR`，对真实样例 `0710-0-1 green.tif` 执行 `--headless-run`。结果：进程退出码 0；桥接目录里出现**恰好一个** `00000000000000000000.properties`（名字即队列约定的 20 位零填充序号），内容 `status=complete`、`input_b64`/`output_b64` 解析回正确的输入与输出目录、`message_b64` 为空；输出目录只有 `*_RoiSet.zip`、`*_QC.png`、`batch_summary.csv`，**没有** Python `*_measurements.csv`（ROI-only 边界仍然成立）。该次运行不带 `ok_count`/`ok_b64`，正是文档里说的「旧规则」情形——诊断用的无界面路径不产出完成清单，插件侧因此按 ROI ZIP 测量。
- **随包 README 的改动**（用户指定的位置与内容）：在「快速上手」之后新增「当前图像有未保存的修改时」一节，写明弹窗的两种选择、处理的是磁盘版本、以及「要一起处理就先保存」。文案与源码逐句核对过：提示确实出现在界面打开之前（`execute()` 里先 `confirmUnsavedChanges` 再 `locateGui()`/启动进程）；点「取消」确实什么都不做、只在状态栏留一句 `Nothing was processed: the current image has unsaved changes.`；除测量外代码不改写也不保存该图。
- **产物**：`build_0.4.1.bat` 四步全部成功（`javac --release 8` + `jar` → PyInstaller → `postbuild.py` → 产物校验），jar 与 EXE 均为本轮新构建，哈希见下。
- `Auto_Worm_Annotations.java` 与 `src/batch_worm_roi.py` 本轮**没有改动**：前者不含桥接与测量代码，后者的行为不受这五处修复影响。改动的四个文件是 `plugin_src/Auto_Worm_ROI.java`、`src/worm_roi_gui.py`、`make_release.py` 与 `tests/test_imagej_integration.py`（新增 `tests/test_make_release.py`）。

## 第二轮复审修复的验证（2026-09-15）

第二轮复审针对上一轮的提交 `ee68201`，确认前三处测量问题已修复，另外提出三项新问题并建议暂缓发布。三项都先逐条对源码核实（结论：**全部成立**），再实现、再验证。前两项都在 Java 轮询循环里，Python 侧无法触及，因此本轮新增了一个直接驱动**已构建 jar** 的 Java 测试。

- **第一项（关窗丢弃待测批次）确实存在**：源码核实——`pendingNotifications()` 每轮只在循环开头取一次快照，测量在快照的循环体内执行；A 测量期间入队的 B 与关闭通知都不在快照里，A 测完后 `if (!process.isAlive()) break;` 因进程已退出而跳出，随后 `removeBridge()` 删除整个目录。这与复审给出的现象（`A_MEASURED=true`、`B_MEASURED=false`）一致。
- **第二项（闸门放行过早）确实存在**：源码核实——通知在**读取时**（`finally` 里）就被删除，而测量在其后开始；GUI 的 `_imagej_bridge_pending()` 只判断「目录里还有没有 `.properties`」，于是毫秒级的读操作就解除了本应持续整个测量过程的闸门。上一轮我自己写的那段注释把它说成「all read」，正是把「已读」当成了「已测」。
- **第三项（指纹丢失顺序）用探针复现**：`a = 1; b = 2` 与 `a = 2; b = 1` 的**字节码逐字节相同**（`950053007200530172016702`），差异只在 `co_consts`（`(1, 2, None)` 对 `(2, 1, None)`），按集合存放的指纹因此判为相同；字典换值、两个函数体互换、嵌套函数与常量换位同样全部漏判（探针四条全部报 MISSED）。修正后四条全部判为不同。
- **修好之后，两个方向都用「改回旧行为就该失败」证明过**，不是只跑通就算：把「退出后排空」改回「进程一没就 break」，`tests\AutoWormBridgeQueueTest.java` 在第 75 行失败——`batches measured: expected 2, got 1`，正是复审报告里那一批结果消失的现象；把「处理完再删」改回「读到就删」，同一测试在第 110 行失败——`in the queue while being measured: expected [true], got [false]`。两次都把改回旧行为的源码放在临时目录里编译运行，未改动仓库文件；改回新实现后测试通过。
- **新 Java 测试驱动的是已构建的 jar，不是重新编译的源码**：`Auto_Worm_ROI.pumpNotifications` 与 `Handled` / `BatchHandler` 与测试同包可见，测试直接调用它，并用 `--sleep` 另起一个进程冒充 GUI（「进程什么时候消失」必须是真的，不能是假设）。四种情况：① GUI 在测量途中退出、且待测批次已入队 → 两批都被测到、队列清空；② 测量期间该批的通知仍在队列里（这正是 GUI 闸门读的那个文件）；③ 通知删不掉（`dos:readonly`）→ 只测一次、循环照常结束；④ 通知读不了（非 ASCII 字节）→ 被丢弃、不进处理器、不留残留。运行在 **Fiji 自带的 Java 8 运行时**上（与用户实际运行环境一致），编译用 JDK 21 加 `--release 8`。
- **造「删不掉」这个用例走过的弯路值得记下**：先用打开的文件句柄（`FileChannel`）并没有挡住删除——JDK 是以共享删除方式打开文件的，实测打印 `false`（文件其实已被删掉）。改用 Windows 的只读属性才是真的删不掉（`Files.delete` 对只读文件抛 `AccessDeniedException`），测试因此会断言「删除确实失败了」，并在属性不可用（非 Windows）时按「本机无法覆盖此情形」报告而不是假装覆盖了。
- **指纹改为有序之后没有引入误报**：`src\*.py` 全部 22 个模块经 `marshal` 往返（EXE 内实际存放的形态）指纹**全部一致**，包括带 `frozenset` 的 `evaluate_shape_refiner.py`（上一轮的假阳性修复仍然成立，单个无序容器依旧先渲染元素再排序，元组仍按顺序比较）。用本轮重建前的 EXE 比对当前源码：**12 个模块相同、只有 `worm_roi_gui` 不同**——与「本轮只改了一个 Python 文件」逐一吻合；若有序化带来假阳性，未改动的模块会一起报 `differ`。
- **测试数 98 → 103**（Python），另有新增的 Java 队列测试 1 个文件 4 个情形。`tests\test_make_release.py` 新增四项「换序必须判为不同」与一项「换序判据经 `marshal` 往返后仍然成立」。
- **本轮同时改了 Python 与 Java，因此 jar 与 EXE 都必须重建**：Python 侧的改动是 `_imagej_bridge_pending()` 的注释与那句「等待 ImageJ」提示（提示语即使只改一个字，指纹比对也会要求重打包）。
- **产物**：`build_0.4.1.bat` 四步全部成功，jar 与 EXE 均为本轮新构建，哈希见下。（该批处理文件在本机的调用需注意两点：Windows 环境变量 `NoDefaultCurrentDirectoryInExePath` 已置位，`call build_0.4.1.bat` 这种不带路径的写法会「找不到文件」，要写成 `call "%CD%\build_0.4.1.bat"`；仓库路径含非 ASCII 字符，经中间层转发命令行时容易被改写，直接从图形界面或 cmd 交互运行时无此问题。）

## 第三轮复审修复的验证（2026-09-15）

第三轮复审针对提交 `97edd94`，提出五项问题（两项 P1、三项 P2），本次全部修复。五项都先逐条对源码核实（结论：**全部成立**），再实现、再验证。前两项把「导出的 ROI 就是被统计的区域」变成强制不变量，用**改前 / 改后对照探针**验证（旧实现取自 `git show HEAD:src/batch_worm_roi.py`，与新实现并排跑同样的输入）；后三项在 Java 侧，用直接驱动**已构建 jar** 的测试验证。

- **发现 1（背景 ROI 可能压在虫体或排除区上）确实存在，且能用探针复现**：源码核实——背景矩形由「虫体左边界减去图像宽度的 2.5%」算出，且 `x1 = max(x0 + 5, …)` 强制至少 5 像素宽，虫体贴近左边缘时右边缘必然落进虫体；排除区存在时只在「排除区是整块矩形」的路径上才尝试重找。400×400 合成图上改前 / 改后对照：

  | 情形 | 改前 mask / 导出 ROI | 改前 ROI 内的脏像素 | 改后 mask / 导出 ROI | 改后脏像素 |
  | --- | --- | --- | --- | --- |
  | 虫体贴左边缘（`x 5..60`） | 0 px / 660 px | 660 个虫体像素 | 36190 px / 36190 px | 0 |
  | 排除区盖住全部候选位置 | 0 px / 19910 px | 19910 个已排除像素 | 13100 px / 13100 px | 0 |

  改前的 `mask` 与导出的 ROI 是两块完全不同的地方（`array_equal` 为 False），改后逐像素相同。新的判定规则：候选矩形必须是**完全干净**的区域，依次尝试虫体左 / 右 / 上 / 下方取面积最大者；一处都找不到、或最大者小于 200 像素且不足整图面积的 0.1% 时返回「不给背景」，由 `process_image` 按单图失败路径拒绝该图（消息含 `找不到干净背景区域`），批处理继续下一张。
- **发现 2（虫体内部的人工排除区被填回）确实存在**：源码核实——虫体 ROI 一律走 `ImageJRoi.frompoints` + `roitype=POLYGON`，普通多边形无法表达「这块不算」。400×400 合成图、虫体实心块内挖掉 1050 像素（灰 400 而周围 100）改前 / 改后对照：

  | | 统计面积 | 统计均值 | 导出 ROI 面积 | 导出 ROI 均值 | 被排除像素重新进入 |
  | --- | --- | --- | --- | --- | --- |
  | 改前 | 13250 | 100.0000 | 14300 | 122.0280 | 1050 |
  | 改后 | 13250 | 100.0000 | 13250 | 100.0000 | 0 |

- **复合 ROI 的格式是**读**出来的，不是猜的**：先用 `ShapeProbe` 让 ImageJ 自己写一个带孔 `ShapeRoi` 并打印序列化路径，得到三元组 `(操作, x, y)`：0=起点、1=连线、4=闭合子路径，坐标为绝对坐标，孔洞子路径绕行方向与外轮廓相反（非零填充规则，没有任何标记位）；文件头仍是矩形类型（`roitype=1`）、`shape_roi_size` 非零、版本 228、大端。再用 `roifile` 写出同一布局的文件交 `DecodeProbe` 用 ImageJ 读回并测量：`area 300.0 / pixelCount 300 / mean 100.0`（图里环为 100、洞为 900），**洞被承认**；不承认时应当是 400 / 150。端到端再跑一次真实路径：`process_image`（`measurement_backend="imagej"`）对 200×200 图导出的 ROI，用 `MeasureZipProbe` 交给 ImageJ——ROI 解码为 `ij.gui.ShapeRoi`、两条子路径、被排除的 630 像素不再进入统计，虫体均值与 Python 侧同为 `100.0000`（面积一项的差异见本节末尾）。
- **发现 3（GUI 退出瞬间写入的通知被丢弃）确实存在**：源码核实——`pumpNotifications` 每轮先扫队列、**之后**才 `process.isAlive()`，GUI「写最后一个通知 + 退出」落在两者之间时这一轮什么也没扫到，随即满足「进程已退出且本轮无进展」而结束。窗口只有毫秒级，无法稳定地赛跑，因此把时序**建模**下来：一个 `Process` 在被询问 `isAlive()` 的那一刻写出通知再返回 false（这正是循环对 `Process` 的全部要求）。新用例 `aNotificationWrittenAsTheGuiExitsIsStillMeasured` 在本轮 jar 上 `measured == 1`；把同样的测试对**改前源码**（`git show HEAD:plugin_src/Auto_Worm_ROI.java`，仅把 `measureOutputFolder` 的可见性改成包内可见，其余不动）编译运行，报 `batches measured: expected 1, got 0`，与复审报告的现象一致。
- **发现 4（上一批的合并测量表残留）确实存在**：`measureOutputFolder` 开头新增 `removeStaleCombinedTable`。探针 `StaleTableProbe`（输出目录里预置一张上一批的 `ImageJ_measurements_all.csv`，本批一张也没测成）：**改前** `previous batch's table still here: true`（旧表原样留下，事后无法分辨它属于哪一批），**改后** `false` 并在日志中写明已删除。守卫也验证过：输出目录里放一个同名但首行不是我们的 CSV，批处理结束后文件**内容逐字节未变**（既不删也不覆盖）。
- **发现 5（单张测量失败中断整批）确实存在**：逐图循环体原先只有 `finally`。`tests\AutoWormMeasurementTest.java` 用「第一张的 `*_RoiSet.zip` 是坏文件、第二张完好」的夹具驱动真实方法：**改前**异常从 `measureOutputFolder` 逃逸（`loadRois` → `IOException`），第二张从未被测量、合并表也没生成；**改后**两张都走完流程，`bbb_good_measurements.csv` 与合并表都生成且合并表里只有好的那张，跳过的那张在日志中点名、完成日志里写明「跳过 1 张」。全部都没测成时仍报失败，但消息改为「没有一张图像测量成功，已跳过 N 张」。
- **两处 P1 又在真实数据上端到端跑了一遍**（不只是合成图与单元测试）：用本轮重建出的 `AutoWormGUI.exe` 以无界面模式跑 `test01` 的三张真实 TIFF，三张全部 `PASS`、各导出 10 条虫 + 1 个背景（即新的背景判定没有把正常图拒之门外），再把导出的 ROI 逐像素复核——**背景与虫体零重叠**（背景 172203 / 176595 / 169235 px，其中虫体像素 0 个；改前是整块压在虫体上）。孔洞一侧同样在真实图上验：复制一张真实图，按 GUI 写 sidecar 的格式（`<图名>.autoworm.json` 的 `exclusions`）在某条虫内部写入一块 9×9 的排除区，带 `--manual-head-annotation` 重新跑批，该虫导出为复合 ROI——ImageJ 解码为 `ij.gui.ShapeRoi`、两条子路径；把同一只虫分别以「带洞的复合 ROI」和「只含外轮廓的普通多边形」交给 ImageJ 测量，得到 `pixelCount` **44253 对 44325**、`area` **84406.0 对 84543.3**：差出的 72 px 正是那个洞，说明洞被真正挖掉而不是填回。这两项跑的是**构建产物**，不是源码。
- **测试数 103 → 110**（Python，八份文件：`test_imagej_integration.py` 44、`test_manual_head_annotation.py` 18、`test_make_release.py` 17、`test_manual_exclusion.py` 11、`test_worm_segment_selector.py` 7、`test_low_clarity_splitter.py` 6、`test_boundary_training_033.py` 4、`test_review_policy_033.py` 3）。Java 侧：`AutoWormBridgeQueueTest.java` 4 → 5 项，新增 `AutoWormMeasurementTest.java` 3 项（该文件需要带界面运行——被测方法结尾会像对用户那样弹出 ImageJ 结果窗口，因此测试结束时 `System.exit(0)` 收尾，否则 AWT 事件线程会让进程留着不退出）。
- **Java 测试跑在已构建的 jar 上**，编译用 JDK 21 加 `--release 8`，运行用 Fiji 自带的 Java 8 运行时；`measureOutputFolder` 的可见性由 `private` 改为包内可见，与 `pumpNotifications` 同样的理由（让测试能驱动**已构建的产物**而不是重新编译的源码）。
- **本轮同时改了 Python 与 Java，因此 jar 与 EXE 都已重建**；`src\batch_worm_roi.py` 的改动会进入 EXE 内嵌的副本。
- **产物**：`build_0.4.1.bat` 四步全部成功，jar 与 EXE 均为本轮新构建，哈希见下。

### 第三轮复审时新发现的一处差异（0.4.1 已修，见下一节）

做端到端验证时发现：同一块矩形，Python（`cv2.fillPoly`）认为覆盖 6360 像素，而 ImageJ 从导出的 ROI 量到 6188——**每个方向少一行一列**。这不是洞的问题，也不是第三轮改动引入的：普通多边形 ROI 从 0.4.0 第一个版本起就是这个行为。用同一块矩形试了四种写法（200×200 图，目标区间 `x 145..197, y 40..159`）：

| 导出的写法 | ImageJ 量到的像素 |
| --- | --- |
| 多边形 ROI，角点 `(145,40)-(197,159)` | 6188 |
| 复合 ROI，同样角点 | 6188 |
| 复合 ROI，角点整体平移 `+0.5` | 6188 |
| 复合 ROI，角点各自向外扩 `0.5` | **6360** |

原因是取样约定不同：`cv2.fillPoly` 以像素坐标本身为取样点，ImageJ 以**像素中心**为取样点，因此边界那一圈像素（中心恰好落在多边形边上）被判在外。端到端那次也一致：虫体外轮廓 Python 侧 81×120、ImageJ 侧 80×119（9090 对 8890），而洞的 630 像素两边相同。

对用户的影响：同一只虫，ImageJ 后端量到的面积比 Python 后端少一圈边界像素（本次实测 9090 对 8890，约 2%；上面那次真实数据端到端也一样——同一只虫导出 ROI 的轮廓 Python 侧 45272 px、ImageJ 侧 44325 px，约 2.1%），均值也因少了这些像素而不同（整只虫亮度一致时相同，边界与内部差得越多差异越大）；`CTCF = IntDen − Area × 背景均值` 会跟着偏。第三轮当时未改（不在该轮授权的五项之内，且修正会改变所有导出 ROI 的边界约定），留待单独决定——用户随后要求修，即为 0.4.1。

## 0.4.1：ROI 改按像素边界导出（2026-09-15）

上一节那处差异的修法：**导出的轮廓改画在像素边界上，而不是画在像素编号上**。第 c 列的左右边界在 `c-0.5` / `c+0.5`，第 r 行的上下边界在 `r-0.5` / `r+0.5`，顶点因此落在半整数上；ImageJ 按像素中心取样，取到的像素集合与 Python 掩膜逐像素相同。0.4.0 的做法是把 `cv2.findContours` 的轮廓（顶点落在像素编号上）直接导出，并先用 `approxPolyDP` 按容差 1.0 抽稀。

- **实现**：`src\batch_worm_roi.py` 的 `instance_contours` 不再用 OpenCV 找轮廓，改为自己沿像素边界描一圈（`_boundary_edges` 收集每个前景像素朝向空邻域的有向边，`_trace_loops` 在岔口挑最靠右的出边接成闭环，`_drop_collinear` 去掉直线上的中间点）。孔洞就是反向绕行的另一条回路，仍是复合 ROI。背景矩形同理改为 `(x0-0.5, y0-0.5) – (x1+0.5, y1+0.5)`。`tolerance` / `approxPolyDP` 参数随之取消——不抽稀才是这套约定的前提。`compound_roi` 的包围盒由 `int()` 截断改为向外 `floor` / `ceil`，因为顶点现在在半整数上。`batch_worm_roi.py` 唯一用到 `cv2` 的就是这段轮廓代码，`import cv2` 一并删除（打包仍带 cv2，`gui_imagej.spec` 里在 `hiddenimports` 中显式列出）。
- **正确性用 ImageJ 的规则反过来验**：`tests\test_manual_exclusion.py` 里原先用 `cv2.fillPoly` 栅格化多边形，现在改为按 ImageJ 的规则采样——像素中心落在形状内即算选中，多条子路径按非零填充规则累加绕数（这同时把「孔洞靠反向绕行表达」这件事也纳入了断言）。两个新用例 `test_exported_roi_samples_back_to_exactly_the_masked_pixels` / `..._without_holes_...` 断言「导出的 ROI 采样回来 == Python 掩膜」，其中一个用例带 2 个孔洞（内部整块 + 贴着下缘的 1 像素细缝），并且把 ROI 写出去、`ImagejRoi.frombytes` 读回来再采样一遍，覆盖子路径顺序、绕行方向与包围盒三处。
- **形状样本逐个量过**：矩形、圆盘、L 形、圆环（带洞）、对角相切的两块（「8」字）、贴边块、单像素、带缺口块、带 2 个孔的虫形块，共 9 例，Python 侧掩膜像素数与「按 ImageJ 规则采样导出的 ROI」逐像素相同。其中 4 例另存成 TIFF + ROI ZIP 交给 ImageJ 1.54p 实测：`1600 / 2821 / 2380 / 800` 全部与掩膜一致，均值均为 `100.0000`；圆环解码为 `ij.gui.ShapeRoi`，洞被承认。
- **真实数据端到端**：用本轮构建的 `AutoWormGUI.exe` 以 `--headless-run` 跑 `test01` 的三张真实 TIFF（各 10 条虫），再把每张图的 `*_RoiSet.zip` 交给 ImageJ 逐条测量，与 Python 后端写进 `*_measurements.csv` 的 `Area` 对照——**44 个 ROI（每图 10 条虫 + 1 个背景，共 4 组）全部相等**，最大者 172203（背景）与 47554（虫体）也无一位偏差。同一批 ROI 另按 ImageJ 规则栅格化复核重叠：背景与虫体、虫体彼此**重叠均为 0 像素**，第三轮的 F1 不变量在改动后仍然成立。
- **人工排除区一并复核**：复制其中一张真实图，按 GUI 写 sidecar 的格式在某条虫内部写入 9×9 排除区，带 `--manual-head-annotation` 重新跑批，该虫仍导出为复合 ROI（ImageJ 解码 `ShapeRoi`），Python 侧 45097 px、ImageJ 侧 45097 px，**逐像素相等**——洞既没有被填回，边界也不再少一圈。
- **代价（如实记录）**：不抽稀的代价是顶点数变多。同一次真实数据端到端里，每张图的 ROI 总顶点数由 1448 涨到 4690（约 3.2 倍），`*_RoiSet.zip` 由约 8.5 KB 涨到约 59 KB（约 7 倍，实测三张：8478→58966、8490→58774、8186→57262）。单张 59 KB 对 Fiji 不构成负担，但这是个只有改动前后对比才看得出来的变化。
- **数值会对不上历史数据**：这是本次改动的目的，不是副作用——0.4.0 及更早版本导出的 ROI 少一圈边界像素，0.4.1 起两边一致。用新旧两版跑同一批图，ImageJ 后端量到的面积/均值/CTCF 会变（面积约差 2%），Python 后端的输出不变（`cv2` 侧本来就是这个约定）。发布说明里要写明这一点。

## 外部定量审查后的修复（2026-09-15）

一份外部审查（`0.4.1_荧光定量与CTCF审查.txt`）把本版与 ImageJ 1.54p 的 `Analyzer` / `ImageStatistics` 逐项对照后给出三条发现，其中前两条本次处理、第三条只需写进文档。审查的条件写得很明确——「同一份磁盘图像、同一套 ROI、同一套有效校准、不限制阈值、不重定向到别的图像」——这个前提条件本身也说明后面两条差异来自测量条件不同，而不是算错。

- **审查对本版 ROI 边界修复的独立确认**：审查方从当前 Python 源码导出七类 ROI（矩形、椭圆、带孔、贴边、单像素、对角接触、背景），用原版 `RoiDecoder` 读取，并按测量处理器实际使用的 ROI/mask 比较每个像素，**每类 missing=0、extra=0**，梯度图均值也相等；真实样例 10 条虫 + 1 个背景的七项测量与 CTCF 全部一致。这是 0.4.1 的取样约定修复在本机之外得到的第二份证据（第一份见上一节）。
- **发现 1（`[P2]` 临时强度校准被静默忽略）确实存在，已修**：审查给出的复现（16 位图、虫体 100 像素每像素 100、背景每像素 20、窗口设 y = 10 + 2x、不保存）在**本轮构建的 jar** 上重跑，数值逐项相同——`window.changes = false`、窗口 CTCF `16000.0`、插件按磁盘文件 `8000.0`。修好后同一个探针得到 `calibrationDifference = 强度校准：窗口 Straight Line（Gray）；文件 未校准（Gray Value）`，也就是提醒与日志现在有东西可说了。探针放在仓库之外的一个临时目录里（不随包发布），它调用的正是 `dist\Auto_Worm_ROI.jar` 里那个方法，不是另抄一份判断逻辑。
- **发现 2（`RawIntDen` 与 ImageJ 同名列不同）经复核成立，选择保留插件的语义**：审查的判断是「不宜直接判为算错」，与 `javap` 看到的实现一致——`Analyzer` 用 `stats.pixelCount × stats.umean`，而 1.54p 的 `ShortStatistics` 在有强度校准时把 `umean` 赋成了校准后的均值；插件用的是 `getRawStatistics()`（以空校准取统计），得到的是真正的原始和。两个后果都被接受：`RawCTCF` 建立在这个和上，Python 侧的数据校验也按原始和核对。因此**没有**为了列名一致而改算式，改为在 CHANGELOG 与 README 里写明这一列是「真正的原始像素和」，并说明它与 ImageJ 1.54p 同名列在有强度校准时数值不同。**这一条与下面的新列合起来看才有意义**：两个表达不同含义的列，只有把校准一并写进表里，事后才分得清哪张图的 `RawIntDen` 能与 ImageJ 直接对照。补齐这两处文档前先核了一遍这一列的全部读者：插件里 `rawIntDen` 与 `rawMean` 只进 `RawCTCF` 的算式（`Auto_Worm_ROI.java:666`）与 `RawIntDen` 列本身，没有任何 QC 列、筛选用途或下游脚本读它；`src\inspect_roi_dataset.py:154` 那个同名读者读的是手工 ImageJ 的 `Results_*.csv`（配 `RoiSet_*.zip`），与插件输出无关。据此在 README「ImageJ 测量与 CTCF」一节补了一句：`RawIntDen` 只服务于 `RawCTCF` 这个参考量，与 ImageJ 同名列在有强度校准的图上不同、无强度校准时一致（**该处 README 改动经用户逐句确认后写入**，2026-09-15）。
- **新增的五列在真实产物上验过**：探针跑出的整行是 `1,review.tif,01_worm,100,100,100,100,100,10000,100,10000,80,8000,8000,worm,ImageJ 1.54p,1,1,pixel,Gray Value,None`，末五列即 `PixelWidth,PixelHeight,SpatialUnit,ValueUnit,IntensityFunction`。`tests\AutoWormMeasurementTest.java` 另用一个带空间与强度校准的 TIFF（0.5 × 0.5 micron、直线 y = 10 + 2x）断言**每一行**都有这五列且值正确——包括背景那一行，因为背景均值同样受校准影响。数值列按 `ResultsTable` 的数字格式写出（`0.500` 而不是 `0.5`）：这是它与表内其余数字列一致的做法，测试因此按数值比较而不是按字符串比较。
- **新增两项 Java 测试，并且都先确认过「旧判据在这个夹具下确实是瞎的」**：`theCalibrationOfTheImageIsRecordedInEveryRow` 与 `aCalibrationSetInTheWindowIsNoticed`。后者在设置强度校准之后断言 `window.changes` 仍为 `false`——若哪天 ImageJ 改了行为、这个夹具不再复现静默情形，该断言会失败并提醒换夹具，而不是让测试继续「通过」却什么都没测到。两个 Java 测试文件都在**已构建的 jar** 上运行（编译用 JDK 21 `--release 8`，运行用 Fiji 自带的 Java 8 运行时），本轮结果：`AUTOWORM_MEASUREMENT_OK`（5 个情形）与 `AUTOWORM_BRIDGE_QUEUE_OK`（5 个情形）。测试的 `main` 改为失败时 `System.exit(1)`、成功时 `System.exit(0)`：被测方法结尾会像对用户那样弹出 ImageJ 结果窗口，AWT 事件线程不是守护线程，失败而不退出会让进程一直挂着，看起来像还在跑。
- **Python 套件 112/112 通过**（本轮未改 Python 代码，属回归确认；用本项目的构建用 Python 以 `-B -m unittest discover` 从 `tests\` 目录运行）。
- **发现 3（不继承 `Limit to Threshold` 与 `Redirect To`）按文档处理，未改代码**：两条都核对了实现——`MEASURES` 常量不含 `Measurements.LIMIT`；测量走 `image.getStatistics(...)`，不经过 `Analyzer` 的全局重定向（`Redirect To` 的污染路径在 0.4.0 已修掉，见 CHANGELOG）。对照手工测量时请关闭这两项；`Redirect To` 一侧尤其不要为了「与手工一致」而撤销这个保护，那会回到「结果行标原图名、实际量到另一张图」。
- **随包 README 的三处补写**（经用户确认后写入，2026-09-15）：审查指出 README 里与本次修复不符的三处，逐条核对现状后改掉。① 「测量是怎么实现的」一节原写「由 ImageJ 自己的 `Analyzer`、`ImageStatistics` 和 `ResultsTable` 在原始像素上测量」——`Analyzer` 从 0.4.0 起就不再参与（正是为了避免 `Redirect to` 污染），且「在原始像素上」对 `CTCF` 不成立（它用的是校准后的 Area 与 Mean），改为按实际路径描述并指明不经过 `Analyzer`。② 「ImageJ 测量与 CTCF」一节补上新增的五列，并给了一句可核对的换算关系（像素尺寸减半时 `CTCF` 变为四分之一，与审查报告里「全局 0.5 × 0.5 使 CTCF 由 8000 变 2000」一致）。③ 新增子节「与手工测量对照时」，用表格写明 `Limit to Threshold` 与 `Redirect to` 不继承、对照前要关掉，并写明测量以**文件上的校准**为准、临时改的校准会先弹提示、`Global` 选项是例外、日志里会写本次实际使用的校准。三处改动都只涉及文档，`Auto_Worm_ROI.jar` 与 `AutoWormGUI.exe` 的 SHA256 不变；zip 的字节内容因此改变，发布仍以 `make_release.py` 当次输出的 SHA256 为准。
- **产物**：本轮只改了 Java 源码与测试，因此**只重建了 jar**，`AutoWormGUI.exe` 未重建、哈希不变（Python 源码一个字节没动，`make_release.py` 的模块一致性检查照常通过）。重建后的 jar：jar 内 10 个条目，`Auto_Worm_ROI.class` 含新增的 `confirmSameCalibration` / `calibrationDifference` 与各辅助方法（`javap` 已确认在位），`Auto_Worm_Annotations.class` 未变。

```text
Auto_Worm_ROI.jar（外部审查修复后重建）
SHA256 98D6C06BBC7D93035ED798B85AA6C88FFCC612082023A1D1AB16FE8592F26633
       （2026-09-15：新增窗口/磁盘校准比对与五列校准信息；
         此前 0.4.1 的 jar 为 2B076236…）
```

```text
AutoWormGUI.exe
SHA256 DCC7D1280A61B398DEE3BBAD5D5103C8A2E8E66E5361CFFB31F784C08628A4F
       （2026-09-15 0.4.1：ROI 改按像素边界导出、背景矩形同步改；
         另含 0.4.0 → 0.4.1 的版本号；上一版为 C17855BB…）
       （外部审查修复一轮未改 Python 代码，本次重建没有覆盖它）
```

取样约定修复时的那一版 jar（`2B076236…`）已被上面这一版取代，两者只差校准比对与五列校准信息；`AutoWormGUI.exe` 的哈希在外部审查修复前后**相同**，因为这一轮没有改 Python 代码。

两个哈希都是**重编后重新计算**的：jar 由 `[1/4]` 步骤用 JDK 21 `--release 8` 重编（`javap -constants` 读回 `VERSION = "0.4.1"`，`javap -v` 显示 major version 仍为 52，即 Java 8 目标；第三轮新增的 `pumpNotifications` / `readNotification` / `removeNotification` / `removeStaleCombinedTable` 与放宽为包私有的 `measureOutputFolder` 全部在位），EXE 由 `[2/4]` 步骤重新构建，`[3/4]`、`[4/4]` 分别完成运行库/模型/许可同步与输出校验。重建后的 jar 另跑了一次 `tests\AutoWormBridgeQueueTest.java`（5 个情形全过，`AUTOWORM_BRIDGE_QUEUE_OK`）与 `tests\AutoWormMeasurementTest.java`（外部审查修复前 3 个情形、修复后 5 个情形，全过，`AUTOWORM_MEASUREMENT_OK`；其日志开头打印的版本已是「自动圈虫 0.4.1」），整套 Python 测试 112 项全过——**测试跑的是刚构建出来的产物**，不是手工编译的临时 class。

`make_release.py --previous-exe` 用**上一版的 EXE** 作对照，再一次证伪了「比对本身失效」这个可能：新 EXE 13 个模块与 `src\` 全部相同、0 个不同；上一版 EXE 则有 2 个不同，且正好是本轮改过的两个文件——`batch_worm_roi`（取样约定）与 `worm_roi_gui`（版本号）。上一版 EXE 取自上轮发布包，不参与本次打包。

两份被解压安装的文件都由 `build_0.4.1.bat` 重新构建。打包的 GUI 里带的是不是当前 `src/`，没有靠时间戳判断，而是把 PyInstaller 归档里每个模块的**代码对象**取出来，与现场编译当前 `src\*.py` 得到的代码对象逐模块比对。指纹取自：常量的渲染（字符串、数字、bytes、None、元组，集合先渲染再排序）、`co_argcount` / `co_posonlyargcount` / `co_kwonlyargcount` / `co_flags`、自由变量与闭包变量、以及**字节码本身**（`co_code`）；行号与注释有意排除在外，因为它们只能反映格式。这样不需要人工维护一组标记字符串，改一个变量名、一句提示语或**一个数字**都会被算作不同。

指纹里的各条**按编译器产生的先后顺序排列**，而不是装进集合再比对——这是第二轮复审后改的，因为集合看不见顺序，而顺序本身承载信息。`a = 1; b = 2` 与 `a = 2; b = 1` 的字节码逐字节相同（差异只在常量表：`(1, 2, None)` 对 `(2, 1, None)`），两条常量一旦被排序进集合，就再没有东西区分这两个程序；同理，嵌套代码对象现在展开在它在 `co_consts` 里所占的位置上，常量与名字因此各自与所属的代码对象绑定，而不是被拍平成一张全局表——两个函数互换函数体、或一个嵌套函数在常量表里换位，都会改变指纹。有序化没有引入误报：`src\*.py` 全部 22 个模块经 `marshal` 往返（EXE 里实际存放的形态）指纹全部一致，包括带 `frozenset` 的 `evaluate_shape_refiner.py`（单个无序容器仍先渲染元素再排序，元组仍按顺序比较）；用本轮重建前的 EXE 比对当前源码，12 个模块相同、只有本轮改过的 `worm_roi_gui` 不同（未改动的模块必须仍报相同——若它们也一起报 `differ`，说明指纹本身有问题，上一轮加强指纹时就真的遇到过这种情况，见上一节的 `frozenset` 一条）。旧包一侧是这次核对的必要条件：若它也全部相同，那么「新包相同」就无从判断是打包成功还是比对本身失效。比对的实现与两侧结果都固化在 `make_release.py` 里，会随每次打包重跑，两个方向的行为另有 `tests\test_make_release.py` 的 17 项测试钉住（其中 5 项专门钉顺序：换序必须判为不同，且该判据经 `marshal` 往返后仍然成立）。

随包分发的第三方许可（`dist\AutoWormImageJ\licenses\`，与仓库 `licenses\` 逐字节相同）：

```text
THIRD-PARTY-NOTICES.txt   B22E4517545F2EC68D8765F08C30988EA191A2705BB39DC930F220B8B024228A
NVIDIA-CUDA-EULA.txt      5438AE5B7F535F3A22032F7052BD431AD6BA087902AFEE471FFBA56464D1C6BA
NVIDIA-cuDNN-SLA.txt      DFC6F5A266CDA6076C5CC355956EA1120418CEBD6F96008BA8A044A419C98949
Pillow-LICENSE.txt        4F7866A74802C6326F81FAFF59A56546B6AEC2B10B91973E0E9308DE95E79857
roifile-LICENSE.txt       2D306DAE704B06B117E1C15B39FE5728883B641A12769E21A85870D25213307C
```

两份 NVIDIA 文本的出处与抓取日期记在 `THIRD-PARTY-NOTICES.txt` 第 3 节，可据此在联网环境下重新取回并比对 SHA256。这几份文件不随 `AutoWormGUI.exe` 一起变化，因此**加入它们不需要重新打包 EXE**。

发布 zip 顶层是 ImageJ 惯例的 `plugins/`，解压到 Fiji 根目录即可安装。打包由 `make_release.py` 完成，ZIP 总计 14,761 个文件；运行包本体 3.11 GiB，**按 deflate 压缩后 zip 为 2.02 GiB**（约 35%）。

0.4.1 的 `dist\AutoWormImageJ\` 下有 14,756 个文件，比 0.4.0 少 3 个，差异全部是 `_internal\api-ms-win-core-*.dll` 这类由操作系统提供的 API set 转发 DLL（`api-ms-win-core-fibers-l1-1-1`、`api-ms-win-core-kernel32-legacy-l1-1-1`、`api-ms-win-core-sysinfo-l1-2-0`），并未主动删除任何依赖。**这 3 个文件的缺失不影响运行**：本轮的 0.4.1 EXE 正是用这份 `dist\` 跑完真实数据端到端（3 张真实 TIFF + 1 张带人工排除区，44 个 ROI 全部与 Python 掩膜逐像素相等），启动与推理均正常。

此前这一轮用的是 `--store` 不压缩，理由是「内容以 DLL 和模型为主，压缩收益有限」，打出的 zip 同为 3.11 GiB。该理由经实测不成立：同一份 3.11 GiB 运行包用 deflate 压到 2.02 GiB，下载量少一个 GiB。因此本轮发布包改为默认 deflate，`AGENT.md` 第 4 节的说法一并改正。**zip 的字节内容因此与上一轮候选包不同**，发布时请以本次输出的 SHA256 为准。

发布 ZIP 的 SHA256 在每次打包完成后由 `make_release.py` 输出，应作为发布页或下载说明中的外部校验值提供；不把 ZIP 自身的哈希写进 ZIP 内部文档，以免形成无法自洽的自引用（本文件随 ZIP 分发，所以上面一节只列 `Auto_Worm_ROI.jar` 与 `AutoWormGUI.exe` 两项，ZIP 的哈希另记在仓库的 `AGENT.md` 与发布说明里）。

zip 的结构经逐条核对，并与 README「安装」一节画出的目录树一致：顶层只有 `plugins/` 与四份文档，没有 `dist/` 之类只对开发机有意义的前缀；`plugins/Auto_Worm_ROI.jar`、`plugins/AutoWormImageJ/AutoWormGUI.exe`、两套模型的四个 `.pt`、`licenses/THIRD-PARTY-NOTICES.txt` 均在位；14,746 个 `_internal/` 条目全部通过 CRC 校验，可完整读回。

随包的 README 已改写为 Markdown，并最终定为面向 Java 8 用户的版本（`README_0.4.1_ImageJ_Java8.md`，取代最初的 `.txt`），文档仍是四份、打包内容与文件总数不变，因此上一段的结构核对结论继续成立；`make_release.py` 的 `DOC_NAMES` 与模块注释里的文档清单同步为这个名字（不改会导致打包时报文档缺失）。README 的这几轮改动只涉及文档，`Auto_Worm_ROI.jar` 与 `AutoWormGUI.exe` 的 SHA256 与上面「最终文件校验」一节相同，未重新构建。

README 中关于监视器边框的表述另做过一次**依据源码的核对**：原文写「调用平滑修复后对应图像边框会标黄」，逐条比对 `worm_roi_gui.py` 的边框绘制（`status.startswith("REVIEW")` → 红、`qc_attention` → 黄、当前选中 → 蓝）与 `batch_worm_roi.py` 的汇总列，确认黄框只由 `shape_refined_count` / `shape_review_count` / `low_clarity_split_count` 决定，与「是否开过平滑修复」无关；同时确认红黄两色由 `make_overlay()` 直接画进 `*_QC.png`，蓝框只在界面里。该处已改正（错误句子删除，新增「监视器上的边框颜色」一节），另修 5 处 `NVDIA` 拼写。**这次改动同样只涉及文档**，两个产物的 SHA256 仍未变；但 zip 的字节内容因文档更新而改变，发布仍以 `make_release.py` 当次输出的 SHA256 为准。

按 README 的安装步骤曾对更早的候选包**实际装过一次**（2026-09-14，装在本机 Fiji 上）：压缩包目录结构、文件数量、许可文件以及安装后的 JAR/EXE 哈希均核对一致，证明「解压到 Fiji 根目录」这条安装路径可用。此后的几轮重建包都没有自动覆盖该安装——得到明确要求前不改动本机的 Fiji。各次包本身另按上文完成结构、源码一致性、真实样例与 Java 测量验证。

**本机 Fiji 当前的状态（2026-09-15 实测）**：`plugins\Auto_Worm_ROI.jar` 与 `plugins\AutoWormImageJ\AutoWormGUI.exe` 都是 2026-09-15 10:52 写入的一对，SHA256 分别为 `9850F95E…` 与 `867AF380…`——正是 0.4.0 第二轮修复的那对构建，两件配套、互相匹配。**0.4.1 的构建没有装入**（用户要求不干扰本机原有 Fiji），因此本机 Fiji 里跑的仍是 0.4.0 的代码；0.4.1 的全部验证都在 `dist\` 的产物上完成。要换成 0.4.1，须把 jar 与 EXE **两件一起**覆盖：0.4.1 与更早版本之间的桥接协议本身没有变化，但 jar 与 EXE 若各换一半（例如新 jar 配老 EXE），仍应避免——版本号来自 jar 一侧，混装会让菜单显示的版本与汇总 CSV 里记录的 `software_version` 对不上。换完需重启 Fiji 生效。

发布前若在本机 Fiji 里试装过，记得 `plugins\AutoWormImageJ\licenses\` 也要一并覆盖，否则第三方许可会停留在旧版。

GUI/CUDA 发布目录约 3.11 GiB，包含 PyTorch/CUDA、cuDNN、OpenCV、SciPy、MSVC 运行库和两套模型；最终用户无需另装 Python 或 CUDA Toolkit。

图像格式允许列表依赖 Pillow 对各 TIFF 编码所报的像素模式，故一并记录构建环境版本：Pillow 12.3.0、NumPy 2.5.3（构建用 Python 3.13.2 虚拟环境）。仓库没有 pin 住 Pillow 版本，如果将来在别的环境重新打包，需重新确认 `I;16B` / `I;16L` 等模式名称未变。

## 综合 Bug 复审后的修复（P1 三条，2026-09-15）

这一轮的输入是两份互不相关的复审：一份外部的 `0.4.1_第二轮Bug复审.txt`，一份本机自查，合并记录在 `0.4.1_综合Bug复审.md`（13 条发现：P1 四条、P2 四条、P3 五条，每条注明来源是外部、本机还是双方独立复现）。合并时先做了一件事——确认两份复审看的是**同一个构建**：外部复审记录的 jar 哈希 `98D6C06B…F26633` 与当时的 `dist\Auto_Worm_ROI.jar` 逐位相同，因此两份意见可以直接合并，不必先排除「版本不同各说各话」。逐条与 0.4.0 的同名文件比对后确认：**13 条里没有一条是 0.4.1 引入的回归**，全部是自 0.4.0 或更早就在的历史行为。本机复审还有一条把「不是缺陷」误报成缺陷（`refine_instance_shapes` 的重叠归属），逐行复核后已排除，并作为一条「不要改」记进 `AGENT.md` 第 9 节——往交接文档里写一条假缺陷，下一个人会照着去改一段正确的代码。

本轮只改 P1 里的三条（用户确定的范围），余下 1 条 P1 与 P2/P3 共十条留待后续。三条的共同点是**都属于「出错了但没人知道」**：不崩、不报错，只是结果少了一块或界面不动了。

- **P1-1（Java）单图测量表写失败时，它的行仍然进了合并表**：测量是逐图做的，每图有自己的 `*_measurements.csv`，全部测完后再把所有行汇成一个 `ImageJ_measurements_all.csv`。原来的写法是**边测边往合并表里加行**，单图那张表则在循环里单独保存。于是单图那张表写失败（磁盘满、权限、文件被占用）时，日志把这张图记成「已跳过」，可它的行半个字都没少地留在了合并表里——合并表里那几行的数值是真的，这正是它危险的地方：输出目录里再没有第二样东西能与之矛盾，而旁边那张**不存在**的单图表，很容易被读成「这张图不需要单独的表」。改法是把顺序倒过来：先把该图的行攒在一个 `MeasuredImage` 里，**单图表写成功落盘之后**再折进合并表；写失败则该图按跳过处理，合表里没有它，日志与目录内容一致。回归测试 `anImageWhoseTableCannotBeWrittenIsNotInTheCombinedTable` 用目录占位（在 `aaa_blocked_measurements.csv` 处建一个包含 `keep.txt` 的目录，于是测量本身成功、只有写失败）触发，断言三条：占位目录及其中的文件原封不动（不会被「替换」掉）、输出目录里没有残留的临时文件、合并表里有 `bbb_good.tif` 而没有 `aaa_blocked.tif`。**这个缺陷另用一份独立探针在改动前后各跑了一次**，因为「加一条测试」本身不能证明被测行为真的坏过。探针只做这一件事（造两张图、两张 ROI、把 `aaa_blocked_measurements.csv` 的位置占成目录，然后调 `measureOutputFolder`），对**冻结的 0.4.0 jar**（`2DCF21C0…`，P1-1 是 0.4.0 起就有的行为，该处代码两版相同）与**本次重建的 jar** 各跑一遍，输出：

```text
===== 0.4.0 的 jar（改动前） =====        ===== 本次重建的 jar（改动后） =====
RESULT  throw=none                          RESULT  throw=none
RESULT  blockedRowInCombined=true           RESULT  blockedRowInCombined=false
RESULT  goodRowInCombined=true              RESULT  goodRowInCombined=true
RESULT  placeholderSurvived=true            RESULT  placeholderSurvived=true
```

改动前那一列就是缺陷本身：写不出单图表的 `aaa_blocked.tif` **进了合并表**。还有一点是这次才注意到的——`throw=none`，也就是整批**并没有报错**，日志里那句「已跳过」和合并表里那几行并存，两者都不引人注意。改动后同一行变成 `false`，而另一张图照旧在表里。

- **P1-1 附带的一个发现：`ResultsTable.saveAs()` 是按扩展名选分隔符的**。原子写临时文件的命名一开始写成 `x_measurements.csv.tmp`（和 Python 侧 `_atomic_write` 的 `"%s.%s.tmp%s"` 长得像，但那个写法扩展名在最后），结果这批测量表**全部变成制表符分隔**。这是 `theCalibrationOfTheImageIsRecordedInEveryRow` 报出来的——它断言末五列是 `PixelWidth,PixelHeight,SpatialUnit,ValueUnit,IntensityFunction`，而报错信息打出的表头是制表符分隔的。反汇编 `ij/measure/ResultsTable.class` 看到判据：文件名以 `.csv` 或 `.CSV` 结尾 → 逗号，否则 → TAB。修法是把临时文件的扩展名放回最后（`x_measurements.tmp.csv`），与 Python 侧保持一致；测试里的 `noTemporaries` 辅助函数随之改为查找 `.tmp.`（标记在扩展名**之前**）而不是 `.tmp` 结尾，注释里写明原因，免得下一个人再「顺手改回去」。**这张表是实验数据本身**，用 Excel 打开时制表符与逗号看不出区别，只有下游按逗号解析的脚本会整列错位——属于「不改就会安静地产出坏数据」的那一类。
- **P1-3（GUI）Excel 动过的 QC 汇总表会让窗口再也打不开**：`_load_qc_status_summary` 读的是 `_qc_status_summary.csv`，也就是质检工作流**自己写出来、README 又明确让实验员去看**的那个文件。它的异常处理捕的是 `(OSError, csv.Error)`，而 `UnicodeDecodeError` 是 `ValueError` 的子类、**不是** `OSError`，且是在 `for row in csv.DictReader(handle)` 迭代时才抛（打开文件本身不会失败）。所以在中文机器上用 Excel 打开再保存一次（Excel 会写成 GBK）之后，这个异常直接逃出去；而**文件夹路径是被记住的**，于是此后每次启动窗口都在恢复上次文件夹时倒下——用户看到的是「程序一开就崩」，而要恢复只能去改文件名。同项目 `batch_worm_roi.py` 里那个同类读取器一直是捕了 `UnicodeDecodeError` 的，只有 GUI 这一处漏了。修法是补上这个异常类型（放在元组里而不是写成 `except Exception`，免得把真正的 bug 也一起吞掉），行为回到与其他读取器一致：读不出来就当没有汇总表，界面照常打开。测试 `test_a_summary_csv_in_another_encoding_does_not_stop_the_window` 直接把这个文件按 GBK 写出来复现；`test_a_bad_summary_file_does_not_leave_the_batch_half_finished` 另确认读不出汇总表不会让一批处理停在中途。
- **P1-4（GUI）事件泵一旦抛异常就永久停摆**：`_poll_events` 靠每次处理完队列后重新挂一个 `after(100, …)` 来继续跑。原来这次重新挂**在 `try` 之外、函数末尾**，于是只要处理队列里的某一项时抛出任何异常，异常逃出 `after` 回调（Tk 只把它打到 stderr），**重新挂接那一步也就没执行**——事件泵从此再也不会被唤醒。表现出来就是「跑完了但界面不动」：进度不更新、按钮不回弹、状态栏停在最后一句，而后台的活儿可能已经干完了。修法是在进入 `try` 之前置 `keep_polling = True`，两处正常的提前 `return` 改成先置 `False` 再返回，并在 `finally` 里按这个标志重新挂接；同时新增一条 `except Exception` 走 `_report_pump_failure`（在终端面板写一行中文提示 + `类型: 消息`，再 `traceback.print_exc()`），把「安静地停摆」变成「界面照常运行、日志里有话」。这里有个**不能用简单 `finally` 就完事**的地方：整批处理结束时 `_processing_finished` 会 `root.destroy()`，而它是靠返回 `True` 来表示「窗口已经没了」，窗口销毁之后再挂 `after` 会报错——因此返回值仍然决定要不要重新挂接，标志只是把「正常提前返回」和「窗口已销毁」这两种都归成不挂。同一轮还给收尾那一步新加了 `_scan_qc_images_safely`：`_processing_finished` 里扫描输出目录重新列图时若抛异常，同样会顺着 `after` 回调逃掉并带走事件泵。测试 `test_the_event_pump_survives_a_failure_and_keeps_running` 用一个桩对象驱动，断言三件事：抛异常后**仍然重新挂接**、窗口已销毁时**不再挂接**、异常没有逃出 `after` 回调。
- **测试数**：Python **112 → 115**（`tests\test_imagej_integration.py` 新增三项，对应上面 P1-3 的两条与 P1-4 的一条）；Java `tests\AutoWormMeasurementTest.java` **5 → 6**（新增 P1-1 的用例）。两个 Java 测试文件都跑在**刚构建的 jar** 上，`ImageJMeasurementSmoke` 因 `measureWithImageJ` 的签名变化（多返回一份行数据）同步改为按反射读回 `rows` / `table` 两个字段。全套结果：`AUTOWORM_MEASUREMENT_OK`（6 个情形）、`AUTOWORM_BRIDGE_QUEUE_OK`、`IMAGEJ_MEASUREMENT_OK rows=11`（真实样例 `0716-0-1 green.tif`，10 条虫 + 1 个背景，走的是 `loadRois` + `measureWithImageJ` 的真实路径），Python `Ran 115 tests … OK`。三条修复各自都有「改之前会失败」的证据：P1-1 有在旧 jar 上的实跑，P1-3 有 GBK 复现，P1-4 有桩对象上「不重新挂接」的对照。
- **这一轮 Java 与 Python 都改了**，因此 jar 与 `AutoWormGUI.exe` **两件都要重建**（`build_0.4.1.bat` 的 `[1/4]` 与 `[2/4]`），与上一轮「只重建 jar、EXE 哈希不变」不同。

```text
Auto_Worm_ROI.jar（P1 三条修复后重建）
SHA256 64CDF7319EAC444A3368CC4D25FE210F65269975330B0585BF7AF79932FC14F3
       （2026-09-15：单图表写成功后再进合并表、两张表改原子写；
         此前 0.4.1 的 jar 为 98D6C06B…F26633，再往前为 2B076236…）
```

```text
AutoWormGUI.exe（P1 三条修复后重建）
SHA256 8429D655682220ED1FC2AA96500218A5D25EA6EBCEF75D9D64223E665259E972
       （2026-09-15：QC 汇总表补捕 UnicodeDecodeError、事件泵出错后仍重新挂接；
         此前 0.4.1 的 EXE 为 DCC7D128…A4F——该版自 0.4.1 定版起未变，
         本轮是它第一次因 P1-3/P1-4 重建）
```

重建后的 jar 里多出两个条目 `Auto_Worm_ROI$MeasuredImage.class` 与 `Auto_Worm_ROI$Row.class`（`jar --list` 已确认在位），正是本轮「先攒行、单图表落盘后再折进合并表」用到的两个新内部类——这是「构建产物里确实是新代码」的最直接证据。**这两个哈希与上面「外部定量审查后的修复」一节里记的那一对都不同**，发布一律以本节为准；上一节的两个哈希自此作废。

P1-1 的回归证据留在了这次的实际运行输出里，可以直接看到修复后的行为：`aaa_blocked.tif` 那一步报的是 `java.nio.file.AccessDeniedException: …\aaa_blocked_measurements.tmp.csv -> …\aaa_blocked_measurements.csv`，紧接着该图被记成「无法处理，已跳过」，而同批 `bbb_good.tif` 照常测完、合并表里只有 1 张图。临时文件的名字正是 `aaa_blocked_measurements.tmp.csv`——扩展名在最后，也就是上面那条分隔符教训修好之后的形态。

## 综合 Bug 复审后的修复（P1-2、P2-5 ~ P2-8、P3-10、P3-11，2026-09-15）

上一节处理了同一份复审（`0.4.1_综合Bug复审.md`，13 条）里的 P1 三条，本轮处理**余下十条里能改代码的七条**；剩下三条（P3-9、P3-12、P3-13）用户明确只要写进 README，代码未动，见本节末。

- **P1-2（`src\inspect_roi_dataset.py` 的 `polygon_mask`）离线校验工具在正确的数据上报出偏差，方向还是反的**：0.4.1 改了**导出**侧的取样约定（ROI 顶点画在像素边界上，ImageJ 数到的就是 Python 掩膜里那些像素），但把 ROI 读回来画成掩膜的这个工具还在按像素**格**填充——`ImageDraw.polygon` 与 `cv2.fillPoly` 是同一套约定，一个 `n × n` 的方块会填出 `(n+1) × (n+1)` 个像素，面积因此虚高约 6%。这一点格外要紧：这个工具的用途正是「证明 ROI 和掩膜对得上」，它在数据正确时报出偏差，等于把操作者引向相反的方向。改为对每个像素中心求多边形绕数、按非零填充规则取值——与 `tests\test_manual_exclusion.py` 里那份独立实现（`_winding` / `shape_pixels`）是同一套判据，孔洞靠子路径反向绕行扣除，将来要读复合 ROI 也成立。三条新测试走**生产导出 → 生产读取**的完整路径（`instance_contours` → `compound_roi` → `roiwrite` → `inspect_roi_dataset.load_rois` → `polygon_mask`），对 6 种形状（单像素、2×2 方块、40×60 矩形、L 形、对角细线、12×9 满幅）逐个断言读回的掩膜与原掩膜逐像素相同、并与那份独立实现一致；另有一条把最小的用例钉死：**1 个像素的 ROI 读回来必须正好 1 个像素**（按格填充会给 4）。
- **P1-2 附带修掉的一个 bug（同一轮新写的测试当场抓出来的）**：改写时一开始把累加器写成 `np.zeros(size)`，而 `size` 是 PIL 的 `(宽, 高)`，下面按行列索引的部分是 `(高, 宽)`——**两者只在正方形图上相等**。非正方形图直接 `ValueError: operands could not be broadcast together with shapes (9,12) (12,9)`，40×60 与 12×9 两个用例就是为此加的。**没有进入任何已发布的构建**（发现时还未重建产物），已改为按行列数建累加器并在注释里写明原因。
- **P2-8（`src\inspect_roi_dataset.py`）0 只虫的图会让整次校验就此结束**：模型没找到虫的图仍会追加一个背景 ROI（`batch_worm_roi` 无条件追加），于是它的 ROI 恰好 1 条、虫列表为空，`min()` 抛 `min() iterable argument is empty`。这个调用在 `inspect_pair` 里没有护栏，异常一路穿出 `main()` 的循环——**这一张图之后的所有图都不会再有报告**，而先打印出来的那几张看起来一切正常。空列表改为记 `nan`（`min_or_nan` / `max_or_nan` / `mean_or_nan`），并把每张图的 `inspect_pair` 单独兜底：打印这一张的原因后继续下一张。新测试 `DatasetInspectionTests` 造一个两张图的真实数据集（`one.tif` 有一条虫、`two.tif` 只有背景），跑完 `main()` 后断言 `two.tif` **出现在汇总里**且 `worm_count == 0`、`min_pixel_area == "nan"`，而 `one.tif` 的 `min_pixel_area == 420`、`mean_mape_percent < 1e-6`。
- **P2-5（GUI + 桥接）用户点停止后，已经跑完的图一个荧光值都没有**：停止时一律发 `cancelled`，而插件对 `cancelled` 只更新状态栏、不做任何测量。于是已经完成的那些图，ROI 躺在输出目录里、测量表却是空的，用户失去了本来已经算好的那一部分。改为发 `complete`，并把成功清单收窄到**确实完成**的图（`completed_stems`，由进度日志行收集、按小写词干比较）；一张都没跑完时才仍发 `cancelled`。`_successful_images` 因此新增 `only_finished` 参数。三条新测试覆盖：完成清单只含已完成的图、一张都没完成时仍走 `cancelled`、以及 `only_finished` 收窄后的清单内容。
- **P2-6（Java + Python）单图测量表从来不删，失败图的表会冒充本轮结果**：合并表一直是测量前先删的，单图 `<图名>_measurements.csv` 则从来没删过。本轮该图测量失败（ROI ZIP 被截断、文件不再是可读的 TIFF）时，上一轮那张表原地不动——日志说「已跳过」，文件夹里却有一张写着这张图名字的表，**数值是真的，只是来自已经被重新分割过的数据**。Java 侧新增 `removeStaleImageTable`，调用点在 `finishedImages` 与 ROI ZIP 两道 `continue` **之后**：只有本轮真会测量或真会报失败的图才删，本轮没走到的图保留自己的表（单图路径只传一个收窄的集合，无条件删除会毁掉别的图的结果）；识别方式照抄 `removeStaleCombinedTable`——看两个列名 `CTCF` 与 `MeasurementBackend`，**不比对整行表头**，这样 0.4.0 以来任何一版（含 0.4.1 新增的五列）都认得出来，而同名的他人文件不动。Python 侧新增 `_remove_stale_imagej_report`，在 `measurement_backend == "imagej"` 分支里与 `_remove_stale_report(csv_path, PYTHON_MEASUREMENT_HEADER)` 一起调用。**两种后端的表靠 `MeasurementBackend` 这一列区分**：两个表头**都含 `CTCF`**，只看它分不开。Java 侧三条用例（旧表被删、只像我们的文件不动、本轮没走到的图保留其表），Python 侧五条（插件表被删、外来表头不动、Python 表被删、两种表头互不误删等）。
- **P2-7（GUI）列输入文件夹是开始按钮路径上唯一没有护栏的文件系统调用**：`_start_processing` 里这次 `iterdir` 排在整套预检**之后**，文件夹被删、盘被拔、共享权限被收回时异常直接逃出回调；带窗口的发行版没有 stderr，按钮看起来就像坏的——既不提示、预检也白跑一遍。改为捕 `OSError` 并弹窗提示重新选文件夹。三条新测试分别覆盖文件夹不可列出时的提示、这条失败不影响其余路径、以及正常路径照旧。
- **P3-10（Java）空间校准用 double 精确相等比较，TIFF 一旦量化就永远报「不一致」**：`calibrationDifference` 原来逐项 `!=` 比较。而 TIFF 保存空间校准是**量化**的——探针实测窗口里的 `0.123456789`（`Analyze > Set Scale` 留在内存里的那个 double）存盘读回变成 `0.1234567901234568`：

  ```text
  written width  = 0.123456789
  reloaded width = 0.1234567901234568
  reloaded unit  = micron
  exact equal    = false
  ```

  于是每一张这样的图都误报「窗口与文件不同」，用户只能每次点确定，很快就学会无视这个提示——那时候**真的**不一致也就不会被看见了。改为像素宽高按**相对容差 1e-6** 比较（新辅助函数 `samePixelSize`），并**不再比较 `pixelDepth`**：它是切片间距，而这里所有测量都是单个 2-D 平面上的 Area / Mean / CTCF，没有一个看得见它，拿它触发只是多一个没法处置的弹窗。单位仍按原样比较。两条新用例：`aQuantizedPixelSizeIsNotADifference` 覆盖「量化值 / 超容差报 / 容差内不报 / 只有 `pixelDepth` 不同不报」四种情况，`aRealTiffRoundTripIsNotADifference` 用真实的 `IJ.saveAsTiff` ↔ `IJ.openImage` 往返复现。
- **P3-11（Python）`_split_qc.csv` 缺陈旧清理分支**：平滑修复、人工标注、部分圈画三个同类都有「本轮没开这个功能就删掉上一轮的报告」，只有低清拆分没有——于是一张图上会留着一份描述**本轮没有做过**的步骤的报告，而文件里没有任何字段说明它属于哪一轮。补上 `else` 分支，字段定义直接用 `SplitReport.__dataclass_fields__`，与读取时识别陈旧文件的判据是同一份。

**P2-6 与 P3-10 的改前/改后对照**（探针 `P2610Probe.java`，不随包发布；对**上一版 jar** `64CDF731…F26633` 与本次重建的 jar 各跑一遍同一份夹具）：

```text
===== 上一版 jar（改动前） =====                   ===== 本次重建的 jar（改动后） =====
P3-10 window=0.123456789                           P3-10 window=0.123456789
      disk=0.1234567901234568                            disk=0.1234567901234568
      -> difference=REPORTED                             -> difference=none
[Auto Worm] 本次测量使用的校准（首张图 aaa_bad.tif）…  [Auto Worm] 已删除上一轮留下的单图测量表，本轮将重新测量：
                                                          aaa_bad_measurements.csv
[Auto Worm] 无法测量，已跳过：aaa_bad.tif（…）      [Auto Worm] 无法测量，已跳过：aaa_bad.tif（…）
pass reported: 没有一张图像测量成功，已跳过 1 张…    pass reported: 没有一张图像测量成功，已跳过 1 张…
P2-6 stale table still there = true                P2-6 stale table still there = false
```

改动前那一列就是两处缺陷本身：同一张图上「窗口的值」与「文件的值」明明只是量化差异，却被判成不同；而那个本轮**没能测量**的 `aaa_bad.tif`，上一轮留下的 `aaa_bad_measurements.csv` 原样留在输出目录里。改动后同一行为 `false`，并且日志里多出一句明确的「已删除上一轮留下的单图测量表」。**P3-10 的夹具必须让窗口持未量化的 double、文件持读回值**：第一次写探针时两边都从文件打开，得到的是同一个量化值，在旧 jar 上也报 `difference=none`——对照不出来，那一遍是白跑的。

- **测试数**：Python **115 → 131**（+16：`test_imagej_integration.py` 新增 P2-6 Python 侧 5 项、P3-11 3 项、P2-8 2 项、P2-7 3 项，`test_manual_exclusion.py` 新增 P1-2 的 3 项）；Java `tests\AutoWormMeasurementTest.java` **6 → 12**（+6：P2-6 三条、P3-10 两条，另有一条把「本轮没走到的图保留其表」钉住）。全套结果：

  ```text
  Python   Ran 131 tests in 0.867s   OK
  Java     AUTOWORM_MEASUREMENT_OK        （12 个情形）
           AUTOWORM_BRIDGE_QUEUE_OK
           IMAGEJ_MEASUREMENT_OK rows=11  （真实样例 0716-0-1 green.tif，
                                           10 条虫 + 1 个背景）
  ```

  三个 Java 测试与 Python 套件都是在**本次重建之后的产物上**重跑的一遍，不是拿改动前的运行结果充数。
- **这一轮 Java 与 Python 都改了**，因此 jar 与 `AutoWormGUI.exe` **两件都要重建**。新哈希（取代上面 P1 三条那一节的 `64CDF731…` / `8429D655…`）：

```text
Auto_Worm_ROI.jar（P2-6 与 P3-10 修复后重建）
SHA256 A25142A5086FA13426C08576067B850BEAB6A920A237B8ACC8307C6B333AA2D5
       （2026-09-15：单图测量表在测量前删除、空间校准改按相对容差 1e-6 比较
         且不再比较 pixelDepth；此前 0.4.1 的 jar 为 64CDF731…FC14F3）
```

```text
AutoWormGUI.exe（P2-5 / P2-6 / P2-7 / P3-11 修复后重建）
SHA256 AC47982353B3A3E733669C512319E70227D078737DA9F62FB898590F5B6DA8EB
       （2026-09-15：停止后已完成的图仍交给 ImageJ 测量、输入文件夹不可列出时弹窗、
         插件测量表陈旧清理、_split_qc.csv 陈旧清理；
         此前 0.4.1 的 EXE 为 8429D655…E972）
```

- **本轮只写进 README、代码未动的三条**（用户指定范围）：**P3-9** 文件夹名含 `[` `]` 时 `glob` 把它当字符集，整批报 `No TIFF images found`，而界面用 `iterdir()` 明明列着图；**P3-13** 文件名首尾带空格会被 `str(name).strip()` 那一侧静默跳过（不测量、不报错）；**P3-12** 桥接残留的恢复办法（关掉界面窗口、重启 ImageJ 后再运行插件；或手工删 `%TEMP%\autoworm-imagej-*\*.properties`），以及「退出请求被模态框吞掉时再点一次 ✕」。README 新增「文件与文件夹命名」一节与「常见问题」两条。**P3-9 的一行修法（`glob.escape`）与 P3-13 的协议改动本轮都没有做。**

## 真实数据实测发现的一处崩溃（只读数组，2026-09-15）

这一条不是复审出来的，是 0.4.1 装到本机 Fiji 之后**用户拿真实数据跑出来的**。触发它的那张图在批处理汇总里记着：

```text
image                          worm_count  expected_count  qc_status  error
Failed_0913-100-2 green.tif                            FAILED     ValueError: assignment destination is read-only
```

- **现象**：用户那批真实数据（同一文件夹下 8 张 `0913-* green.tif`），7 张正常，只有 `0913-100-2 green.tif` 失败，其余全部计数正确（`0913-1k-2 green.tif` 为 11 条，本来就是 `REVIEW_COUNT_MISMATCH`，与本次无关）。
- **两个条件同时成立才会崩**，少一个都不触发：

  | 条件 | 代码里的判据 | 那张图的实际情况 |
  | --- | --- | --- |
  | 「形状细化」开着 | `return_foreground_probability=shape_refinement` | 该用户配置里是**开**的（同批每张图都写出了 `*_shape_qc.csv`） |
  | 图上有**人工排除区** | `full_exclusion is not None` | 8 张里**只有这一张**有（1 个排除区、0 箭头、0 边界线） |

  同批的 `0913-0-2 green.tif` 有 10 个箭头 + 18 条边界线却**没有**排除区，因此绕过了那行赋值、正常跑完——这也是为什么它看起来「手动标注没问题，只有排除区出事」。
- **根因**（`src\batch_worm_roi.py` 的 `predict_raw`）：前景概率数组是经 PIL 缩放拿回来的，而 `np.asarray()` 会把 PIL 缓冲区的**只读标志一并带出**，紧接着的排除区清零又往这个数组里写：

  ```python
  foreground_probability = np.asarray(
      Image.fromarray(foreground_probability, mode="F").resize(
          (raw.shape[1], raw.shape[0]), Image.Resampling.BILINEAR),
      dtype=np.float32)                      # ← writeable=False
  if full_exclusion is not None:
      foreground_probability[full_exclusion] = 0.0   # ← ValueError
  ```

  单独验证这一条判据：

  ```text
  >>> a = np.arange(16, dtype=np.float32).reshape(4, 4)
  >>> b = np.asarray(Image.fromarray(a, mode="F").resize((8, 8), Image.Resampling.BILINEAR), dtype=np.float32)
  >>> b.flags.writeable
  False
  ```

  同一个函数里另外两处 `np.asarray(Image...)` 不受影响：一处的后面跟着 `> 0`，另一处跟着 `astype()`/`/ 255.0`，都会产生新的可写数组。修法是给这一个数组补 `.copy()`（`dtype=np.float32).copy()`）。
- **不是 0.4.1 引入的**：拿冻结的 0.4.0 跑同一组输入（同一个 `predict_raw`、同一个假模型、同样的排除区与 `return_foreground_probability=True`）：

  ```text
  0.4.0: ValueError: assignment destination is read-only
  ```

  0.4.0 的 `src\batch_worm_roi.py:724` 同样把 `manual_exclusion_mask` 传进 `predict_raw`、第 720 行同样有 `return_foreground_probability=shape_refinement`，那一行赋值也一字不差。**自 0.4.0 起就带着这个 bug**，0.4.1 没有让它更容易触发，也没有修掉它。
- **复现**：把该图与 `_manual_head_annotations.json` 复制到临时目录（源目录只读），按界面在「高清晰度图像」模式下实际传的参数调用 **`run_gui_batch`**（不是 `process_image` —— `run_gui_batch` 才会从 checkpoint 里读 `image_size`、腐蚀次数、各阈值与 tip 参数，直接调 `process_image` 会用它自己的默认值、分割结果明显更差，那样复现出来的东西说明不了问题）。修复前的 traceback 逐字如下，与批处理汇总里记的错误一致：

  ```text
  File "src\batch_worm_roi.py", line 456, in predict_raw
      foreground_probability[full_exclusion] = 0.0
  ValueError: assignment destination is read-only
  ```

- **修复后同一批图重跑**（本机 RTX 4060 Laptop GPU，FP32，与用户那次同一台机器）：**8 张全部处理成功，`failed: 0`**。

  ```text
  0913-0-1 green.tif       worms=10  expected=10  PASS                     heads=0/10 boundaries=0  exclusions=0
  0913-0-2 green.tif       worms=10  expected=10  PASS                     heads=10/10 boundaries=18 exclusions=0
  0913-10-1 green.tif      worms=10  expected=10  PASS                     heads=0/10 boundaries=0  exclusions=0
  0913-10-2 green.tif      worms=10  expected=10  PASS                     heads=0/10 boundaries=0  exclusions=0
  0913-100-1 green.tif     worms=10  expected=10  PASS                     heads=0/10 boundaries=0  exclusions=0
  0913-100-2 green.tif     worms=10  expected=10  PASS                     heads=0/10 boundaries=0  exclusions=1   ← 原为 FAILED
  0913-1k-1 green.tif      worms=10  expected=10  PASS                     heads=0/10 boundaries=0  exclusions=0
  0913-1k-2 green.tif      worms=11  expected=10  REVIEW_COUNT_MISMATCH    heads=0/11 boundaries=0  exclusions=0
  failed: 0
  ```

- **与原批处理逐字段对照**：把重跑写出的 `batch_summary.csv` 与用户那份原表按图名对齐、逐列比较（路径列、`run_id`、以及 `segment_start_fraction` 这类本轮未复制的界面偏好除外）。结果：**除 `0913-100-2` 那一行之外，其余 7 行逐字段完全相同**——`worm_count`、`expected_count`、`qc_status`、`shape_refined_count`、`shape_review_count`、`low_clarity_split_count`、`head_annotation_count`、`head_annotation_complete`、`manual_head_split_count`、`manual_boundary_guide_count`、`manual_exclusion_region_count`、`manual_exclusion_area_px`、`measurement_backend`、`inference_precision`、`worm_model_sha256`、`tip_model_sha256` 全部一致；`0913-100-2` 那一行由「全空 + FAILED」变成 `worms=10 expected=10 PASS`（排除区 1 个、3335 像素）。这既说明修复没有动别的东西，也说明这次复现确实复刻了用户那一次。
- **测试数 131 → 133**：`tests\test_manual_exclusion.py` 新增两项，都**不打桩 `predict_raw`**、用一个返回常量 logits 的假模型驱动真函数——这个文件里原有的排除区测试一律把 `predict_raw` 打了桩，只验证它前后的接线，**正因如此这条 bug 才一直没被发现**。第一项复现原崩溃（带排除区 + 要求返回前景概率），断言返回的概率数组可写、排除区内为 0、排除区外大于 0；第二项断言没有排除区时返回的数组同样可写。两项在**临时撤掉 `.copy()`** 后确实失败（`AssertionError: False is not true` 与 `ValueError: assignment destination is read-only` 各一），补回后通过。全套：

  ```text
  Python   Ran 133 tests in 1.588s   OK
  ```

- **只重建 `AutoWormGUI.exe`，jar 未动**：这一轮只改了 Python。`plugin_src\` 未改，因此没有重跑 `javac`/`jar`，`dist\Auto_Worm_ROI.jar` 与上一轮**逐字节相同**（重建前后都是 `A25142A5…AA2D5`）——不是「重建后恰好一样」，是根本没有重建。

```text
AutoWormGUI.exe（只读数组崩溃修复后重建）
SHA256 EEFD05A6D5E4A79CA11281DFCEF308C0AB06DE13C001AB364398541DBC46A35A
       （2026-09-15：predict_raw 的前景概率数组补 .copy()，带人工排除区且开启
         形状细化的图不再以「assignment destination is read-only」失败；
         此前 0.4.1 的 EXE 为 AC479823…6DA8EB）

Auto_Worm_ROI.jar（本轮未重建，与上一轮相同）
SHA256 A25142A5086FA13426C08576067B850BEAB6A920A237B8ACC8307C6B333AA2D5
```

## 人工分界线改为多边形取点、并严格按线拆分（2026-09-15）

用户提出两条：分界线的画法要像 ImageJ 的 Polygon Selections；人工分界线存在时必须严格按线拆。

- **画法验证通过**：界面隐藏自检（`tests\check_manual_head_gui.py`）断言左击落节点后草稿逐点累积、`<Motion>` 会画出橡皮筋、**分界线模式下松开左键不结束**、右击后该条线进入标注列表且草稿清空、只有两个节点且长度不足时拒收、Esc 清空草稿、撤销能退回上一步。自检把保存标注的函数换成空实现，不会在测试图片旁写出 `_manual_head_annotations.json`。实机操作时的光标为十字，帮助文案写明「左击依次点出节点，右击结束；Esc 放弃这一条」。
- **画法与算法的一致性验证**：界面中保存下来的分界线不再以样条曲线显示（此前 `smooth=True`），屏幕上看到的折线就是算法里画墙用的折线；节点也不再抽稀（只去掉相距不足 1 像素的重复点，双击会落在同一处），折线拐点与实验员点出的位置逐个相同。
- **严格按线拆的量化验证**：在带人工标注的真实样例（`0913-0-2 green.tif`，1024×1024，10 条虫、18 条人工分界线）上，沿每条分界线取 220 个采样点，从采样点沿法向往两侧找第一个属于其他虫体的像素（上限 21 像素），两侧都超过 5 像素即判定「接缝没落在线上」：

  | 版本 | 落在线上 | 中位距离 |
  | --- | --- | --- |
  | 0.4.0 的划分（未做压线） | 192/220（87.3%） | 1.0 像素 |
  | 0.4.1 既有的软惩罚实现 | 202/220（91.8%） | 1.0 像素 |
  | 本轮压线实现 | **205/220（93.2%）** | 1.0 像素 |

  同一次运行里虫数与精修报告数均未改变（10 条虫、2 条 `manual_boundary_refine`），也就是说提升来自接缝位置本身，不是靠多拆或少拆。
- **两条被否掉的思路（有实测数据，不要再走）**：把墙做成完全不可穿只让距离从 1.90 像素变成 1.87 像素（前沿本来就没穿墙，是从线的自由端绕过去的）；把墙延长到图像边缘反而从 87.3% 掉到 68.6%，精修报告由 2 条降到 1 条（延长后的墙把别的虫体切开了）。
- **压线的安全边界验证通过**：改动只发生在分界线周围按图像尺寸算出的窄带内（实测取短边的 4%），且只交换「本来属于对面那一侧」的像素；每次交换后要求每个标签仍然非空且连通（≥99.5% 的像素在同一个连通块里），否则整条放弃。合成用例覆盖三种情形：接缝在窄带之内会被压到线上、偏离超出窄带时**保持原样且不切碎虫体**、线两侧本来就是同一只虫时**一个像素都不改**。压线前后前景像素集合逐像素相同。
- 单元测试数 133 → 136，全套：

  ```text
  Python   Ran 136 tests in 1.095s   OK
  ```

- 真实数据集成脚本 `integration_manual_boundary_test0513`（组合模式与仅分界模式均修好左侧两条尾部分割）与 `integration_manual_boundary_test06`（头向粗分后分界精修，10/10 头向匹配）在本轮改动后重跑仍通过。
- **只重建 `AutoWormGUI.exe`，jar 未动**：这一轮只改了 Python。`plugin_src\` 里的两个 `.java` 与 `plugins.config` 都比现存的 jar 旧，重跑 `javac`/`jar` 只会因时间戳让包字节变化，因此没有重建（重建前后都是 `A25142A5…AA2D5`）。

```text
AutoWormGUI.exe（人工分界线改为多边形取点 + 严格按线拆分后重建）
SHA256 2C6290E9BD1AFB33E379189B911EC86DCA7ED16E5455DCC52696AE73218C23E0
       （2026-09-15：分界线改为左击落点、右击结束；接缝按人工分界线压回，
         带标注样例上的遵守率由 87.3% 提升到 93.2%；
         此前一版的 EXE 为 EEFD05A6…46A35A）

Auto_Worm_ROI.jar（本轮未重建，与上一轮相同）
SHA256 A25142A5086FA13426C08576067B850BEAB6A920A237B8ACC8307C6B333AA2D5
```

## 三条并排的虫画了箭头也拆不开（用户实测报告，2026-09-15）

用户在真实数据上报：`0913-0-2 green.tif` 上「从左往右第 2、3、4 条虫子判成一条，即使在我已经进行了人工分割的情况下也是如此」。该图带有 10 支头向箭头、4 条人工分界线、1 个排除区。

**修复前的实测**（当前代码在保存下来的修复前标签图上重算，逐条打印）：

```text
修复前的标签数: 8
  标签 1 面积= 35756      标签 2 面积= 95104   ← 三条虫连成一块
  标签 3 面积= 34252      标签 4 面积= 32828
  标签 5 面积= 36113      标签 6 面积= 34708
  标签 7 面积= 32908      标签 8 面积= 36817
有多支箭头的块: [2]  {1: 1, 2: 3, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1}
这块的面积: 95104  箭头数: 3

轴芯候选: None
分数候选: 3
尾部投票: [1, 1, 3]                       ← 两支箭头投到了同一块，安全检查据此否掉整份分区
候选子块面积: [32133, 31180, 31791]  纵向跨度比: [0.971, 0.988, 1.0]
连通块数: [1, 1, 1]
覆盖完整: True
安全检查(修复后的代码): 通过               ← 同一个候选分区，改后接受
按纵轴归属的兜底方案面积: [4030, 9820, 81254]   ← 另一条候选路径，等于没分
```

即：候选分区本身是完整、连通、三块面积相当的「教科书解」，唯一的问题是**投票认不出箭头属于哪一块**，而它当时是唯一的判据。

**修复后的实测**（同一次运行里接着跑整条标注流水线）：

```text
修复后（整条标注流水线）标签数: 10
报告: [(2, 'REVIEW_MANUAL_HEAD_SPLIT', '2|3|4')]
```

**交付产物上的复核**：不拿中途的数组，而是把输出目录里的 `0913-0-2 green_RoiSet.zip` 读回来重建标签图（按 ROI 名取标签），再沿 4 条人工分界线逐像素采样，从采样点沿法向往两侧找第一个属于其他虫体的像素（上限 21 像素），两侧都超过 5 像素即判定「接缝没落在线上」——与本节上方「人工分界线改为多边形取点」用的是同一套判据：

| 版本 | 虫数 | 采样数 | 落在线上（5 像素内） |
| --- | --- | --- | --- |
| 修复前 | 8 | 3744 | 1913（51.1%） |
| **修复后** | **10** | 3744 | **3037（81.1%）** |

分界线贴合度由 51.1% 升到 81.1%，说明用户的四条分界线**跟着一起恢复了作用**——它们在修复前同样什么都没做（见下一段）。

**根因二的现场**：粗分没做成，那个 95104 的块仍然在，四条分界线于是被合并成一组、标签集为 {1,2,3,4}，走的是「每个标签一个核心」的精修路径；而标签 2 自己就是三条虫，长出来的归属退化成 `[69968, 36035, 94792, 32791, 34318]`，指标算不出来，四条线一并失效。这是根因一的连锁，不是两处独立的缺陷。

**整批回归**：同批另外 7 张真实图跑完整流水线后逐字段对照 `batch_summary.csv`，除被修的这张之外**一行都没变**，包括 `0913-1k-2 green.tif` 原本就是 `worms=11 expected=10 REVIEW_COUNT_MISMATCH`（与用户手上的原表一致，不在本轮范围内）。8 张里 7 张 `PASS`：

```text
0913-0-1 green.tif       worms=10  expected=10  PASS
0913-0-2 green.tif       worms=10  expected=10  PASS     ← 修复前为 worms=8
0913-10-1 green.tif      worms=10  expected=10  PASS
0913-10-2 green.tif      worms=10  expected=10  PASS
0913-100-1 green.tif     worms=10  expected=10  PASS
0913-100-2 green.tif     worms=10  expected=10  PASS
0913-1k-1 green.tif      worms=10  expected=10  PASS
0913-1k-2 green.tif      worms=11  expected=10  REVIEW_COUNT_MISMATCH（原本如此）
```

**头部信息表**：修复后 `0913-0-2 green_head_annotations.csv` 十行**全部** `MATCHED`，其中 `03_worm` 一行为 `match_method=ordered`、`match_distance_px=10.88`（箭头尾端到第 3 条虫最近像素的距离），其余九行为 `tail`。修复前该行是 `MISSING`。

**单元测试 136 → 140**，全套：

```text
Python   Ran 140 tests in 1.459s   OK
```

两条新用例在**关掉新回退**的实现上确实为红（三支箭头压在同一条虫上时只剩 1 块；报告侧第 3 条为 `MISSING`、`unmatched=1`），补回后通过。

**真实数据集成脚本**：`integration_manual_boundary_test0513`（组合模式与仅分界模式均修好左侧两条尾部）与 `integration_manual_boundary_test06`（头向粗分后边界精修，10/10 头向匹配）在本轮改动后重跑仍通过；界面隐藏自检 `tests\check_manual_head_gui.py` 通过（`MANUAL_ANNOTATION_GUI_OK`）。`integration_manual_head_split_real`、`integration_manual_boundary_e2e_0513`、`integration_manual_head_real` 三项的失败与本轮改动无关，且**逐字与改动前相同**：前者是既有缺陷（`0513-10-1.tif` 只数出 8 条，该脚本无任何人工标注，`AGENT.md` 已记）；后两者的失败发生在读取工作区里已有的 ROI 夹具时（`Unsupported ROI type 1 in 03_worm.roi`、工作区里缺 `test04_gap_final\0513-10-1_RoiSet.zip`），压根没跑到本轮改动的代码。

**产物**：本轮只改了 Python。顺带重跑过一次 `javac`/`jar` 后逐条目比对确认 12 个条目**内容逐字节相同**（只差 zip 时间戳），因此 jar 按上一版**原样放回**——交付的 jar 保持不变，只需替换 EXE。

```text
AutoWormGUI.exe（三条并排虫按次序配对后重建）
SHA256 6D15B6132301C2DA637722B998CD23FA694785D7CA1D2BE788C5826713861DF4
       （2026-09-15：箭头投票认不出归属时改按横排次序配对；
         同一张图 8 → 10 条虫，人工分界线贴合度 51.1% → 81.1%；
         此前一版的 EXE 为 2C6290E9…23E0）

Auto_Worm_ROI.jar（本轮未重建，与上一轮逐字节相同）
SHA256 A25142A5086FA13426C08576067B850BEAB6A920A237B8ACC8307C6B333AA2D5
```

**打包核对**（`make_release.py --previous-exe …`，上一版 EXE 取自上一版发行包里的 `2C6290E9…`）：

```text
new EXE: 13 module(s) match src/, 0 differ
previous EXE: 12 module(s) match src/, 1 differ
    differ: manual_head_annotation        ← 恰好是本轮改的那一个模块
ok: the previous EXE differs from src/, so the comparison above is not vacuous
```

即：新包里的每一个模块都与现场编译的 `src\*.py` 一致，而上一版包**只有** `manual_head_annotation`
一个模块不同——既证明改动进了包，也说明这个比对不是空转。四份随包文档刷新到 `dist\` 后逐份与仓库根
的同名文件 SHA256 相同（`README` 与 `LICENSE` `unchanged`，`CHANGELOG` 与 `VALIDATION` `updated`）。
发布包本身的哈希记在 `AGENT.md`（它不进包，因此不会自我指涉）。

本轮**未**改动本机 Fiji（0.4.1 的安装需用户明确要求后才做，见 `AGENT.md` 第 7 节）。

## 尚需人工界面验收

自动测试已覆盖计算链路，也已按 README 的安装步骤把插件放入 Fiji 的 `plugins` 文件夹并确认菜单项出现。仍建议人工操作一次完整实验流程，重点检查双滑块拖动、人工标注是否符合实验习惯，以及本批出现失败图像时「部分图像未处理」提示是否容易看到。

人工验收应把本次生成的测试包解压到一份独立 Fiji 副本后进行；不要把当前本机 Fiji 中上一轮候选包的表现误认为本次终端布局构建的结果。

原界面的失败原因通过桥接通知传给插件（原先只传出「原界面处理失败，请查看 Auto Worm 日志。」）。桥接通知本身的读写已补上单元测试（编码往返、无 `.tmp` 残留、独立模式不写文件、路径不可写时报错、清单条数上限、两批不互相覆盖、完成清单往返），但**跨进程的那一段**仍未做自动化测试，需人工确认三件事：

1. **失败提示**：在一批图中让某张失败，检查 Fiji 弹出的提示是否给出真实原因与 `Failed_<原文件名>`，而不是笼统的一句话。
2. **未保存改动提示**（本轮新增）：在 ImageJ 里打开一张图、用 `Process > Filters > Gaussian Blur...` 之类改一下之后不保存，直接调用插件，应当弹出中文提示说明「处理的是磁盘上的版本」；点「确定」继续后，结果表里的数值应当与磁盘文件一致（与保存后再跑一次的结果相同），点「取消」则什么都不做且状态栏写明未处理。这条**必须**用命令改图来触发——直接写像素不会置位该标志，弹不出提示是正常的。
3. **连续两批**（队列存在的原因）：一批跑完后**不要**关闭 GUI，在 ImageJ 还没测量完（或看到结果表之后）再处理第二个文件夹，两批的结果都应各自出现在一个结果表里（第一批的 `ImageJ_measurements_all.csv` 与第二批各写各的输出目录），不应出现某一批被静默跳过。测量期间 GUI 里点「开始处理」会被拒绝并弹出「等待 ImageJ」对话框（正文为「ImageJ 正在测量上一批结果，测量完成后才能开始下一批。」，英文 `ImageJ is still measuring the previous batch...`），这是预期行为；**这句提示只有在测量真的还在进行时才该出现**——上一批测完后闸门必须立刻放行，不能因为残留文件而一直卡住。

4. **虫体内部有人工排除区时，导出的 ROI 在 Fiji 里要显示成「挖了洞」**（第三轮新增）：在一张图上给某条虫的内部画一块人工排除区（该区块在分割结果里被挖空），跑完一批后在输出目录里打开该图的 `*_RoiSet.zip`，逐个查看 ROI 显示方式——带排除区的那条虫应当显示为**一个 ROI**、中间那块是空的（`ShapeRoi`，ImageJ 的 ROI Manager 里显示为一个条目），而不是两个分开的 ROI、也不是把洞补上的实心块。同一张图上没有排除区的虫仍是普通多边形 ROI。**这一条必须在带界面的 Fiji 里看**：序列化格式已用 ImageJ 自己解码验证过（见上文探针结果），但 ROI Manager 的显示方式与「面积按挖洞后算」的直观确认只能在界面上做。
5. **找不到干净背景区域时应当拒绝该图、而不是硬给一块背景**（第三轮新增）：构造一张虫体与人工排除区铺满四周留白的图（例如把排除区画大，覆盖图像四边的空白带），处理后该图应当被记为失败并在日志里给出含「找不到干净背景区域」的原因，同批其余图像照常跑完；**不应**出现一个压在虫体上的背景 ROI。这一条同时确认单图失败不再中断整批。
6. **ROI 的轮廓要正好贴住虫体的最外圈像素**（0.4.1 新增）：在带界面的 Fiji 里打开一张已处理的图，从 ROI Manager 里把某条虫的 ROI 叠加显示（`More >> Labels` 或直接看 ROI 的虚线轮廓），确认虚线**压在虫体最外圈像素的外沿**上，虫体最外侧那一圈像素完整地落在 ROI 里——既没有被切掉，也没有多含一圈背景。同时按 `Measure`（或 `Analyze > Measure`）看结果表里的 `Area`：它应当与输出目录里该图的 `*_measurements.csv` 中同一标签的 `Area` **相等**（这是本次改动的全部目的，已用 44 个真实 ROI 自动核对过，此处只是让操作者自己看一眼）。**这一条只看 0.4.1 的包**：0.4.0 及更早版本必然对不上（少一圈边界像素），看到差异是预期的，不是安装出了问题。

7. **窗口里临时设的校准应当被认出来**（外部审查修复后新增）：打开一张图，用 `Analyze > Calibrate...` 设一个强度校准（例如 `Straight Line`、系数 10 与 2）**但不要保存**，然后直接调用插件——应当弹出中文提示，里面同时列出「窗口」与「文件」两套校准，并说明测量用的是文件上的那一份；点「取消」什么都不做，状态栏写 `Nothing was processed: the calibration on screen differs from the saved file.`；若改用 `File > Save` 把校准存进 TIFF 再调用，则**不应**弹这个提示。这一步**不能**靠改像素来试（那会走另一条 `changes` 提示），必须真的用 `Analyze > Calibrate` 或 `Analyze > Set Scale` 改校准；探针已用本机 `ij.jar` 的 `Calibrator` 实现复现过数值（窗口 CTCF 16000 对插件 8000），此处是让操作者确认弹窗文字与按钮行为符合直觉。
8. **测量表末尾的五列要看得懂**（外部审查修复后新增）：跑完一批后打开某张图的 `*_measurements.csv`，确认末五列是 `PixelWidth`、`PixelHeight`、`SpatialUnit`、`ValueUnit`、`IntensityFunction`，且与这张图在 ImageJ 里 `Image > Properties...`（或 `Analyze > Set Scale...`）显示的像素尺寸、单位、强度校准一致；没有强度校准的图应当显示 `Gray Value` 与 `None`。若在 ImageJ 里勾过 `Analyze > Set Scale` 的 Global 选项，日志里「本次测量使用的校准」那行应当带一句「全局标定生效，覆盖各图自己的校准」。

9. **处理到一半点「停止」时，已经跑完的图仍然要出测量结果**（综合复审修复后新增）：准备一批五六张图，等监视器上第一张出现 `PASS` 之后立刻点「停止」（或直接关掉界面窗口）。预期：这次仍会写出一条**完成**通知，Fiji 侧对**已经做完的那些图**照常测量并出结果表；已经做完但没有测量值的图，在 0.4.1 之前是「ROI 在、表是空的」。若一张都还没跑完就停下，则仍按取消处理、不触发测量。**这一条只能在界面上做**：`cancelled` 与 `complete` 的差别在插件侧的行为上，进程内的单元测试够不着跨进程的那一段。
10. **分界线的画法要顺手**（人工分界线改造后新增）：选「人工分界线」模式，沿两条粘连虫之间的缝隙**左击落几个节点**（节点之间应当出现青色虚线折线，鼠标移动时从最后一个节点拉出一条跟随光标的虚线），**右击结束**——该条线应当变成青色实线折线，计数里「分界 +1」。中途按 Esc 应当整条丢掉、不影响已经保存的线。再确认三件事：折线的拐点就在点下去的位置（没有被抹圆或抽稀）、显示出来的青色线与实际生效的边界一致（屏幕上看到的折线就是算法里用的那条）、以及画得明显太短的一条（两个挨在一起的节点）应当被拒收并响一声。排除区仍是按住左键画一圈、松开闭合。
11. **人工分界线确实是硬约束**（人工分界线改造后新增）：挑一张有粘连虫、且你在缝隙处画了分界线的图，跑完之后在监视器里放大看分割结果——两条虫的交界应当**落在你画的那条线上**，而不是在它附近摆动。已知边界：如果实际缝隙离你画的线超过图像短边的约 4%（1024 像素的图约 40 像素），程序不会硬压（避免切出一条不属于任何虫体的碎块），此时仍按模型给出的划分走。
12. **三条并排的虫画了箭头也要能拆开**（本轮新增，直接对应你报的那张图）：把你最初报的那张图（`0913-0-2 green.tif`，10 支箭头 + 4 条分界线）**原样再跑一遍**，预期：监视器里数出 **10** 条虫（此前是 8 条），从左往右第 2、3、4 条是三条分开的虫；`*_manual_split_qc.csv` 里有一行 `REVIEW_MANUAL_HEAD_SPLIT 2 -> 2|3|4`；`*_head_annotations.csv` **十行全部 `MATCHED`**，其中 `03_worm` 一行的 `match_method` 是 `ordered`（其余九行是 `tail`）。这一条最好也顺手看一眼第 3 条虫的头部位置（`head_x` / `head_y`）是否落在它自己的头上——那是按次序配对补出来的，自动测试只能验到「配给了哪一条」，验不到「肉眼看着对不对」。

预检弹窗的两种提示也需人工过目一次：一次列出超过 10 张不合格图像时，末尾是否出现「共 N 张不合格」；以及递交二值图像时「本模型未针对二值图片进行训练」的提示是否在读起来像「可以继续」而不是「已被拒绝」。

「保存日志」按钮需人工点一次：确认默认文件名带时间戳、另存为对话框可取消、取消后不留下空文件，以及保存出的 txt 与日志窗口内容一致。这条路径要弹出系统对话框，自动化测试无法覆盖。

多通道 / Z 堆栈 / 时间序列图像不在本版本范围内：程序只读取每个文件的第一页，而 ImageJ 测量当前显示的平面，两者无法对应，因此这类图像会在处理开始前被整批拒绝（见 README“支持的图像格式”）。唯一例外是多余平面全为 0 的采集软件 overlay 层，这类文件按上文的空平面例外放行。对这类文件，ImageJ 仍会把它显示成多个切片，人工界面验收时可顺带确认测量前窗口中停在的是第 1 个切片。
