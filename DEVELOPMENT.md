# 开发说明

面向改这个仓库的人。用户读的文档是 `README_0.4.2_ImageJ_Java8.md`，本文件只管开发侧。

**完整的交接文档不在本仓库里**，在工作区中与仓库同级目录往上一层的 `devdocs\`：

```text
..\..\devdocs\AGENT-0915-0.4.2.md
```

那份文档有项目背景、三层架构与数据流、逐轮改动记录、已知限制、环境坑，以及**为什么某些
东西故意不在仓库里**。这个文件只做一件事：在没有任何本机路径的前提下，把**能跨机器照抄的
命令**和**入口**留下。

> 注意：从 GitHub 克隆下来的副本**没有** `devdocs\`，那是开发机上的东西。协作者需要向
> 维护者索取。

## 这个仓库是什么

「自动圈虫」的 ImageJ/Fiji 插件版：Java 插件启动打包好的 GUI（Python + CUDA），用户在
GUI 里选文件夹、调参数、必要时做人工标注。

**核心分工（改动时必须守住）**：GUI 只生成 ROI，**不产出任何荧光数值**；荧光统计与 CTCF
由 ImageJ 的 `ImageStatistics` / `ResultsTable` 完成。Python 侧一旦出现 `_measurements.csv`
的数值测量结果即为违约。

| 层 | 位置 | 职责 |
| --- | --- | --- |
| Java 插件 | `plugin_src/` | 菜单入口、启动 GUI、读桥接通知队列、用 ImageJ 测量并出表 |
| Python GUI | `src/worm_roi_gui.py` | 界面、预检、配置、把每批结果写进桥接目录 |
| Python 批处理 | `src/batch_worm_roi.py` | 逐图推理、后处理、写 ROI ZIP / QC 图 / `batch_summary.csv` |
| 模型 | `models/0.1.1`（高清）、`models/0.2.1`（低清） | 整虫分割与头尾细化 |

## 跑测试（改完必跑）

从仓库根执行：

```bash
<venv python> -m unittest discover -s tests -p "test_*.py"
```

需要一个带 **torch + Pillow** 的虚拟环境（跑测试不需要 CUDA）。

**用项目自己的虚拟环境，不要用系统自带的 `python`** —— 这台开发机上的系统 `python` 是
Python 2.7，连 `pathlib` 都没有，报错会以各种莫名其妙的方式出现。别的机器上未必是 2.7，
但道理一样：本项目的依赖只装在虚拟环境里。

## 构建

```bat
set PATH=<JDK 21>\bin;%PATH%
build_0.4.2.bat "<venv python>"
```

四步：`javac` + `jar` → PyInstaller（`gui_imagej.spec`）→ `postbuild.py`（同步 MSVC 运行库、
拷贝 `models\`）→ 校验产物存在。整轮约 3 分钟，PyInstaller 占大头。

`--clean --noconfirm` 会先清掉上一次的 `dist\AutoWormImageJ`，所以中途失败会留下不完整的包。
`dist\` 不入库，重建即可恢复。

**改 `src\*.py` 要重建 EXE；改 `plugin_src\*.java` 要重建 JAR。** 两件事互不代替。

## 打包发布 zip

构建之后跑：

```bash
<venv python> make_release.py --previous-exe "<上一版的 AutoWormGUI.exe>"
```

会核对版本号与包内代码，不通过就拒绝打包。默认 deflate 压缩，zip 写到 `release\`（不入库）。

## 跑插件的 Java 测试

类路径放**类目录**，不要放 `dist\AutoWorm_ROI.jar`：

```bash
set PATH=<JDK 21>\bin;%PATH%

javac --release 8 -encoding UTF-8 -cp "lib\ij.jar" -d "build\plugin_classes" ^
    plugin_src\Auto_Worm_ROI.java plugin_src\Auto_Worm_Annotations.java

javac --release 8 -encoding UTF-8 -cp "lib\ij.jar;build\plugin_classes" ^
    -d "build\measurement_test_classes" tests\AutoWormMeasurementTest.java

java -cp "build\measurement_test_classes;lib\ij.jar;build\plugin_classes" AutoWormMeasurementTest
```

跑完会打印 `AUTOWORM_MEASUREMENT_OK`；任一用例失败则打印堆栈并以 1 退出。当前 16 条。

用例在 `main()` 里逐个直接调用，**不走 JUnit**，所以 `lib\` 里没有测试框架是正常的。

## 生成合成测试样例

```bash
<venv python> diagnostics\make_test_samples.py
```

10 个文件夹 / 32 个文件，覆盖 `check_image()` 能返回的每一种判定。写完会拿 `check_image()`
重跑一遍自检，不符即以 1 退出。输出到 `diagnostics\test_samples\`（已忽略，不入库）；
目标目录非空时拒绝写入，要覆盖得加 `--force`。

## 仓库里**故意没有**的东西

改之前先看一眼 `.gitignore` 末尾那几段说明。以下都是**搬走**而不是删掉，理由写在注释里：

| 不在仓库里 | 在哪 | 为什么 |
| --- | --- | --- |
| 交接文档 `AGENT.md` | 工作区 `devdocs\` | 通篇是本机目录布局 |
| 29 个历史探针脚本 | 工作区 `probes\` | 带本机路径；但它们是某些实验现象的唯一记录 |
| 真实实验图像 | 工作区 `testdata\` | 图像进了 git 历史就撤不回来 |
| 界面偏好 `.worm_roi_gui.json` | `%LOCALAPPDATA%\AutoWorm\settings.json` | 本机状态，且会把开发机的文件夹带给别人 |

> 仓库里 `CHANGELOG_*`、`VALIDATION_*`、`review*/` 和 `LLM_API接入可行性论证_0.4.2.md`
> 中还有若干「见 `AGENT.md` 第 N 节」的引用。**那些没改**——它们是当时写下的记录，
> 引用在写下时是准的，改了就抹掉了当时的事实。要查它们指向的内容，按上面那张表去
> `devdocs\`。

这些名字都写进了 `.gitignore`。**要从历史里彻底清除某样东西，`git rm` 是不够的**——
旧提交里仍能按 blob 原样取出来，得改写历史。怎么改见交接文档的落地记录。
