import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from roifile import ImagejRoi, ROI_TYPE, roiwrite

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from batch_worm_roi import (  # noqa: E402
    _apply_exclusion_mask_to_instances,
    _signed_area,
    background_mask_and_polygon,
    compound_roi,
    instance_contours,
    predict_raw,
    process_image,
)
from inspect_roi_dataset import load_rois  # noqa: E402
from manual_head_annotation import (  # noqa: E402
    ExclusionRegion,
    exclusion_mask,
    save_image_annotations,
)


def _winding(points, columns, rows):
    """一条子路径在每个像素中心处的绕数（非零填充规则用的就是它）。"""
    winding = np.zeros(columns.shape, dtype=np.int64)
    for (x1, y1), (x2, y2) in zip(points, np.roll(points, -1, axis=0)):
        if y1 == y2:
            continue
        up = (y1 <= rows) & (rows < y2)
        down = (y2 <= rows) & (rows < y1)
        side = (x2 - x1) * (rows - y1) - (columns - x1) * (y2 - y1)
        winding += (up & (side > 0)).astype(np.int64)
        winding -= (down & (side < 0)).astype(np.int64)
    return winding


def shape_pixels(subpaths, shape):
    """ImageJ 眼里这组子路径覆盖哪些像素。

    ImageJ 判定一个像素属不属于 ROI，看的是像素**中心**落没落在形状里（像素 (c, r)
    的中心就在 (c, r)）；孔洞靠子路径反向绕行、按非零填充规则扣掉。所以 ROI 的顶点
    必须画在像素边界上（第 c 列的两条竖边在 c-0.5 / c+0.5），画在像素编号上就会差
    一圈行列 —— 这正是下面几个测试要盯住的地方。
    """
    rows, columns = np.mgrid[0:shape[0], 0:shape[1]]
    total = np.zeros(shape, dtype=np.int64)
    for points in subpaths:
        total += _winding(np.asarray(points, dtype=np.float64), columns, rows)
    return total != 0


def polygon_pixels(polygon, shape):
    return shape_pixels([polygon], shape)


def roi_pixels(roi, shape):
    """把导出的 ROI 读回来，再按 ImageJ 的规则采样。"""
    if roi.composite:
        return shape_pixels(roi.coordinates(multi=True), shape)
    return polygon_pixels(roi.coordinates(), shape)


def _block(height, width, row, column, row_end=None, column_end=None):
    mask = np.zeros((height, width), dtype=bool)
    mask[row:row_end or row + 1, column:column_end or column + 1] = True
    return mask


def _l_shape():
    mask = np.zeros((40, 40), dtype=bool)
    mask[5:30, 8:16] = True
    mask[22:30, 16:32] = True
    return mask


def _diagonal():
    # 只有对角相邻的一条细线：行列边界怎么算，在这里差别最大。
    mask = np.zeros((30, 30), dtype=bool)
    rows, columns = np.mgrid[0:30, 0:30]
    mask[columns == rows] = True
    mask[5, 6] = True
    return mask


class RoiMaskReaderTests(unittest.TestCase):
    """离线校验工具读回来的掩膜，必须是 ImageJ 数的那一批像素。

    inspect_roi_dataset 是把导出的 ROI 画成掩膜、再照着掩膜算面积和荧光的，
    所以它的光栅化约定一旦和导出侧不一致，它就会在数据其实正确时报出偏差 ——
    一个专门用来"证明 ROI 和掩膜对得上"的工具，报错的方向还正好是反的。

    0.4.1 把导出侧的约定改成了像素中心采样（顶点画在像素边界上），
    但读的一侧当时还在按像素格填充：一个 n x n 的方块会填出 (n+1) x (n+1) 个像素，
    面积因此虚高约 6%。下面这些用例就是钉住这条：生产导出 -> 生产读取 -> 掩膜，
    每一步都走真正的代码路径，不用测试自己搭的模型。
    """

    # 边界情形：最容易被"按格填充"和"按中心采样"这两套约定分开的形状。
    SHAPES = {
        "single pixel": lambda: _block(8, 8, 2, 2),
        "2x2 block": lambda: _block(8, 8, 2, 2, 4, 4),
        "rectangle": lambda: _block(40, 60, 5, 7, 35, 47),
        "L shape": _l_shape,
        "diagonal": _diagonal,
        "full frame": lambda: np.ones((12, 9), dtype=bool),
    }

    def _round_trip(self, mask):
        """生产路径导出再读回，返回 (掩膜, 读到的多边形顶点)。"""
        outer, holes = instance_contours(mask)
        self.assertEqual(holes, [], "这个用例不该有孔洞：读取侧不接受复合 ROI")
        roi = compound_roi(outer, holes, "01_worm")
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            archive = Path(folder) / "RoiSet_one.zip"
            roiwrite(str(archive), [roi], mode="w")
            rois = load_rois(str(archive))
        self.assertEqual(len(rois), 1)
        return rois[0]["points"]

    def test_the_reader_reproduces_the_mask_for_every_shape(self):
        from inspect_roi_dataset import polygon_mask

        for name, build in self.SHAPES.items():
            with self.subTest(name):
                mask = build()
                points = self._round_trip(mask)
                read = polygon_mask(mask.shape[::-1], points).astype(bool)
                self.assertTrue(
                    np.array_equal(read, mask),
                    "%s: 读回来的掩膜 %d 像素，原掩膜 %d 像素"
                    % (name, read.sum(), mask.sum()))

    def test_the_reader_agrees_with_this_files_independent_oracle(self):
        # 两套实现是各写各的：这边按绕数逐边累加，那边在像素中心上采样。
        # 它们一致才有意义 —— 只看"和生产代码一致"的话，两边一起错就看不出来。
        from inspect_roi_dataset import polygon_mask

        for name, build in self.SHAPES.items():
            with self.subTest(name):
                mask = build()
                points = self._round_trip(mask)
                read = polygon_mask(mask.shape[::-1], points).astype(bool)
                self.assertTrue(np.array_equal(read, shape_pixels([points], mask.shape)))

    def test_a_single_pixel_roi_is_one_pixel(self):
        # 最小、也最尖锐的用例：按格填充会给出 4。这条同时也是文档里那句
        # "顶点画在像素边界上" 的最小可执行说明。
        from inspect_roi_dataset import polygon_mask

        mask = _block(8, 8, 2, 2)
        points = self._round_trip(mask)
        self.assertEqual(int(polygon_mask(mask.shape[::-1], points).sum()), 1)


class ManualExclusionTests(unittest.TestCase):
    def test_exclusion_removes_instance_and_compacts_labels(self):
        labels = np.zeros((80, 100), dtype=np.uint16)
        labels[10:70, 10:21] = 1
        labels[10:70, 45:56] = 2
        labels[10:70, 80:91] = 3
        forbidden = np.zeros(labels.shape, dtype=bool)
        forbidden[5:75, 40:62] = True
        result = _apply_exclusion_mask_to_instances(labels, forbidden)
        self.assertEqual(int(result.max()), 2)
        self.assertFalse(result[forbidden].any())
        self.assertTrue((result[10:70, 10:21] == 1).all())
        self.assertTrue((result[10:70, 80:91] == 2).all())

    def test_exclusion_keeps_only_largest_remainder_of_cut_instance(self):
        labels = np.zeros((50, 80), dtype=np.uint16)
        labels[10:40, 5:75] = 1
        forbidden = np.zeros(labels.shape, dtype=bool)
        forbidden[:, 50:60] = True
        result = _apply_exclusion_mask_to_instances(labels, forbidden)
        self.assertEqual(int(result.max()), 1)
        self.assertTrue(result[20, 20])
        self.assertFalse(result[20, 65])

    def test_background_roi_avoids_excluded_pixels(self):
        labels = np.zeros((100, 100), dtype=np.uint16)
        labels[10:90, 60:75] = 1
        forbidden = np.zeros(labels.shape, dtype=bool)
        forbidden[35:65, 10:25] = True
        background, polygon = background_mask_and_polygon(labels, forbidden)
        self.assertTrue(background.any())
        self.assertFalse((background & forbidden).any())
        self.assertTrue(np.array_equal(background, polygon_pixels(polygon, labels.shape)))

    def test_background_roi_never_covers_worm_or_excluded_pixels(self):
        """背景矩形碰到虫体时不能硬塞：那是把虫体像素当成背景来平均。

        虫体贴着左边缘，左侧留不出余量。旧版会把矩形的右边缘钳到虫体身上，
        于是「统计的区域」和「导出的 ROI」变成两块不同的地方。
        """
        labels = np.zeros((200, 200), dtype=np.uint16)
        labels[60:140, 5:60] = 1
        background, polygon = background_mask_and_polygon(labels)
        self.assertTrue(background.any())
        covered = polygon_pixels(polygon, labels.shape)
        self.assertEqual(int((covered & (labels > 0)).sum()), 0)
        self.assertTrue(np.array_equal(background, covered))

    def test_background_mask_matches_exported_roi_when_pixels_are_excluded(self):
        labels = np.zeros((200, 200), dtype=np.uint16)
        labels[60:140, 120:175] = 1
        forbidden = np.zeros(labels.shape, dtype=bool)
        forbidden[60:140, 0:120] = True          # 左边整块被排除掉
        background, polygon = background_mask_and_polygon(labels, forbidden)
        self.assertTrue(background.any())
        covered = polygon_pixels(polygon, labels.shape)
        self.assertEqual(int((covered & (labels > 0)).sum()), 0)
        self.assertEqual(int((covered & forbidden).sum()), 0)
        self.assertTrue(np.array_equal(background, covered))

    def test_background_is_rejected_when_no_clear_area_is_left(self):
        """一块干净地方都没有时不报背景，由调用方拒绝这张图。

        与其拿被虫体污染的均值去算 CTCF（QC 那一列还照样 PASS），不如明确失败。
        """
        labels = np.ones((100, 100), dtype=np.uint16)     # 整张图都是虫体
        background, polygon = background_mask_and_polygon(labels)
        self.assertIsNone(background)
        self.assertIsNone(polygon)

    def test_background_is_rejected_when_nothing_clear_is_large_enough(self):
        # 虫体占满中间，只在上方剩一条 1 像素高的缝：不够当背景。
        labels = np.zeros((100, 100), dtype=np.uint16)
        labels[2:100, :] = 1
        background, polygon = background_mask_and_polygon(labels)
        self.assertIsNone(background)
        self.assertIsNone(polygon)

    def test_exported_worm_roi_keeps_the_exclusion_as_a_hole(self):
        """排除区在虫体内部时必须导出成「洞」。

        普通多边形 ROI 没有「这块不算」的表达方式，ImageJ 会把洞填回去，面积和均值
        都跟着变 —— 而 Python 后端报的是挖掉洞之后的值。两边对不上，用户没法判断
        该信哪个。孔洞靠子路径绕行方向与外轮廓相反来表达（非零填充规则）。
        """
        labels = np.zeros((120, 160), dtype=np.uint16)
        labels[20:100, 30:131] = 1
        forbidden = np.zeros(labels.shape, dtype=bool)
        forbidden[50:70, 60:81] = True            # 20 x 21，整个落在虫体内部
        instances = _apply_exclusion_mask_to_instances(labels, forbidden)

        outer, holes = instance_contours(instances == 1)
        self.assertEqual(len(holes), 1)
        # 洞正好是被排除掉的那些像素，不多不少。
        self.assertTrue(np.array_equal(polygon_pixels(holes[0], labels.shape),
                                       forbidden & (labels > 0)))
        # 外轮廓仍然覆盖整只虫（含被挖掉的部分），洞由子路径减掉。
        self.assertTrue(np.array_equal(polygon_pixels(outer, labels.shape), labels > 0))

        roi = compound_roi(outer, holes, "01_worm")
        back = ImagejRoi.frombytes(roi.tobytes())
        self.assertTrue(back.composite)
        subpaths = back.coordinates(multi=True)
        self.assertEqual(len(subpaths), 2)
        # 两条子路径必须反向，否则非零填充规则会把洞当成又一块实心。
        self.assertLess(_signed_area(np.asarray(subpaths[0], dtype=np.float64)) *
                        _signed_area(np.asarray(subpaths[1], dtype=np.float64)), 0)
        # 顶点画在像素边界上：洞占第 60..80 列、第 50..69 行，所以边在 59.5 / 80.5。
        hole = np.asarray(subpaths[1], dtype=np.float64)
        self.assertEqual((hole[:, 0].min(), hole[:, 1].min(),
                          hole[:, 0].max(), hole[:, 1].max()),
                         (59.5, 49.5, 80.5, 69.5))

    def test_exported_roi_samples_back_to_exactly_the_masked_pixels(self):
        """导出的 ROI 用 ImageJ 的规则采样回来，必须与 Python 统计的掩膜逐像素相同。

        旧版把 cv2.findContours 的轮廓（顶点落在像素编号上）直接导出，而 ImageJ 看的
        是像素中心落在哪儿，于是同一只虫 ImageJ 会少算最外面那一圈行列（实测 45272
        对 44325 像素，2.1%）—— 两种测量后端给出两个数，QC 那一列还照样 PASS。
        """
        labels = np.zeros((120, 160), dtype=np.uint16)
        labels[20:100, 30:131] = 1
        forbidden = np.zeros(labels.shape, dtype=bool)
        forbidden[50:70, 60:81] = True        # 整块落在虫体内部
        forbidden[90, 100:120] = True         # 贴着下缘的一条细缝
        mask = _apply_exclusion_mask_to_instances(labels, forbidden) == 1

        outer, holes = instance_contours(mask)
        self.assertEqual(len(holes), 2)
        self.assertTrue(np.array_equal(shape_pixels([outer] + holes, mask.shape), mask))
        # 写出去再读回来也要一样：绕行方向、包围盒、子路径都得对得上。
        back = ImagejRoi.frombytes(compound_roi(outer, holes, "01_worm").tobytes())
        self.assertTrue(np.array_equal(roi_pixels(back, mask.shape), mask))

    def test_exported_roi_without_holes_samples_back_to_exactly_the_masked_pixels(self):
        labels = np.zeros((120, 160), dtype=np.uint16)
        rows, columns = np.mgrid[0:120, 0:160]
        labels[((rows - 60.0) ** 2 / 30.0 ** 2 + (columns - 80.0) ** 2 / 10.0 ** 2)
               <= 1.0] = 1
        mask = labels == 1

        outer, holes = instance_contours(mask)
        self.assertEqual(holes, [])
        self.assertTrue(np.array_equal(polygon_pixels(outer, mask.shape), mask))
        back = ImagejRoi.frombytes(compound_roi(outer, holes, "01_worm").tobytes())
        self.assertEqual(back.roitype, ROI_TYPE.POLYGON)
        self.assertTrue(np.array_equal(roi_pixels(back, mask.shape), mask))

    def test_worm_roi_without_exclusions_stays_a_plain_polygon(self):
        labels = np.zeros((120, 160), dtype=np.uint16)
        labels[20:100, 30:131] = 1
        outer, holes = instance_contours(labels == 1)
        self.assertEqual(holes, [])
        back = ImagejRoi.frombytes(compound_roi(outer, holes, "01_worm").tobytes())
        self.assertFalse(back.composite)
        self.assertEqual(back.roitype, ROI_TYPE.POLYGON)

    def test_process_image_fails_when_no_clean_background_is_left(self):
        """一张图没有干净背景时按单图失败处理，消息里要说清是背景的问题。"""
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            image_path = folder / "crowded.tif"
            output = folder / "out"
            output.mkdir()
            Image.fromarray(np.full((100, 100), 80, dtype=np.uint8)).save(image_path)
            # 四条边框全被排除，只剩中间那块虫体。
            save_image_annotations(image_path, (100, 100), [], exclusions=[
                ExclusionRegion(((0, 0), (100, 0), (100, 20), (0, 20))),
                ExclusionRegion(((0, 80), (100, 80), (100, 100), (0, 100))),
                ExclusionRegion(((0, 0), (20, 0), (20, 100), (0, 100))),
                ExclusionRegion(((80, 0), (100, 0), (100, 100), (80, 100))),
            ])
            labels = np.zeros((100, 100), dtype=np.uint16)
            labels[20:80, 20:80] = 1

            with patch("batch_worm_roi.predict_raw",
                       return_value=(labels.copy(), np.full(labels.shape, 80, dtype=np.uint8),
                                     None)):
                with self.assertRaises(ValueError) as caught:
                    process_image(image_path, output, model=None, device=None,
                                  standard_count=1, manual_head_annotation=True)
            self.assertIn("背景", str(caught.exception))
            self.assertIn("crowded.tif", str(caught.exception))

    def test_process_image_applies_saved_exclusion_and_reports_it(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            image_path = folder / "dirty.tif"
            output = folder / "out"
            output.mkdir()
            Image.fromarray(np.full((100, 100), 80, dtype=np.uint8)).save(image_path)
            region = ExclusionRegion(((65, 10), (90, 10), (90, 90), (65, 90)))
            save_image_annotations(image_path, (100, 100), [], exclusions=[region])
            labels = np.zeros((100, 100), dtype=np.uint16)
            labels[10:90, 40:51] = 1
            labels[20:80, 70:81] = 2

            def fake_predict(*_args, **kwargs):
                self.assertIsNotNone(kwargs.get("exclusion_mask"))
                return labels.copy(), np.full(labels.shape, 80, dtype=np.uint8), None

            with patch("batch_worm_roi.predict_raw", side_effect=fake_predict):
                result = process_image(
                    image_path, output, model=None, device=None,
                    standard_count=1, manual_head_annotation=True)
            self.assertEqual(result[1:4], [1, 1, "PASS"])
            self.assertEqual(result[20], 1)
            self.assertGreater(result[21], 1000)
            self.assertTrue((output / "other" / "dirty_QC.png").is_file())


class PredictRawExclusionTests(unittest.TestCase):
    """排除区必须能走完**真实**的 predict_raw。

    上面那些测试一律把 predict_raw 打了桩，只验证它前后的接线。打桩的好处是快、
    不依赖权重，代价是这条函数体从来没被排除区走到过 —— 0.4.1 的「只读数组」崩溃
    就是这么漏出去的。这里的测试用假模型驱动真函数。
    """

    class _FlatForeground(torch.nn.Module):
        """假模型：整幅图都给六成前景概率，不加载任何权重。"""

        def forward(self, tensor):
            batch, _, height, width = tensor.shape
            logits = torch.zeros(batch, 2, height, width)
            logits[:, 1] = 0.5
            return logits

    def test_foreground_probability_survives_an_exclusion_region(self):
        """排除区 + 形状细化时，不能往只读数组里写 0。

        形状细化要求 predict_raw 返回前景概率，而概率数组是经 PIL 缩放拿回来的：
        np.asarray 会把 PIL 缓冲区的只读标志一并带出来。图上没有排除区时那行赋值
        根本不执行，所以只有「开着形状细化 + 图上有排除区」才炸 —— 这正是
        0913-100-2 green.tif 那一张的情况。
        """
        image = np.full((64, 64), 120, dtype=np.uint8)
        mask = exclusion_mask(
            image.shape, [ExclusionRegion(((10, 10), (30, 10), (30, 30), (10, 30)))])

        instances, _normalized, _semantic, probability = predict_raw(
            self._FlatForeground(), image, torch.device("cpu"),
            return_foreground_probability=True, exclusion_mask=mask)

        self.assertEqual(instances.shape, image.shape)
        self.assertEqual(probability.shape, image.shape)
        self.assertTrue(probability.flags.writeable)
        self.assertEqual(float(probability[20, 20]), 0.0)
        self.assertGreater(float(probability[50, 50]), 0.0)

    def test_foreground_probability_is_also_writable_without_exclusions(self):
        """没有排除区时同样返回一个可写数组。"""
        image = np.full((64, 64), 120, dtype=np.uint8)

        _instances, _normalized, _semantic, probability = predict_raw(
            self._FlatForeground(), image, torch.device("cpu"),
            return_foreground_probability=True)

        self.assertEqual(probability.shape, image.shape)
        self.assertTrue(probability.flags.writeable)


if __name__ == "__main__":
    unittest.main()
