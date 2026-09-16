"""背景矩形搜索：快速路径会漏，精确回退要能兜住。

`_largest_clear_rectangle` 只找「整列都空」的矩形。某一列只要有一个被挡的像素，整列
就出局，于是横穿条带的障碍**上方或下方**明明还留着一块干净矩形，它也找不到，整张图
被判「找不到干净背景区域」。这里钉住三件事：

1. 精确搜索本身是对的（与暴力枚举逐个比对，随机形状）；
2. 快速路径漏掉的情形，端到端确实被救回来了；
3. 返回的 mask 与导出给 ImageJ 的 polygon 是同一块像素区域。
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import batch_worm_roi as bw
from batch_worm_roi import (_largest_clear_rectangle, _largest_clear_rectangle_exact,
                            background_mask_and_polygon)


def _brute_force_area(blocked, x0, x1, y0, y1):
    """区域内最大全空矩形的面积，直接枚举，作为对照。"""
    region = blocked[y0:y1 + 1, x0:x1 + 1]
    height, width = region.shape
    best = 0
    for row_start in range(height):
        for row_end in range(row_start, height):
            for column_start in range(width):
                for column_end in range(column_start, width):
                    if not region[row_start:row_end + 1, column_start:column_end + 1].any():
                        best = max(best, (row_end - row_start + 1) * (column_end - column_start + 1))
    return best


class ExactRectangleTests(unittest.TestCase):
    def test_matches_brute_force_on_random_masks(self):
        # 固定种子：失败可复现，也不会因为运气好而偶尔放过去。
        rng = np.random.default_rng(20260916)
        for trial in range(200):
            height = int(rng.integers(1, 9))
            width = int(rng.integers(1, 9))
            blocked = rng.random((height, width)) < 0.25
            with self.subTest(trial=trial, shape=(height, width)):
                found = _largest_clear_rectangle_exact(blocked, 0, width - 1, 0, height - 1)
                self.assertEqual(found[0] if found else 0,
                                 _brute_force_area(blocked, 0, width - 1, 0, height - 1))

    def test_the_returned_rectangle_is_actually_clear(self):
        rng = np.random.default_rng(4242)
        for trial in range(200):
            height = int(rng.integers(1, 12))
            width = int(rng.integers(1, 12))
            blocked = rng.random((height, width)) < 0.3
            found = _largest_clear_rectangle_exact(blocked, 0, width - 1, 0, height - 1)
            if found is None:
                continue
            area, x0, y0, x1, y1 = found
            with self.subTest(trial=trial):
                self.assertFalse(blocked[y0:y1 + 1, x0:x1 + 1].any(),
                                 "the rectangle offered as clear is not clear")
                self.assertEqual(area, (x1 - x0 + 1) * (y1 - y0 + 1))

    def test_the_exact_search_never_returns_less_than_the_fast_path(self):
        # 精确搜索是快速路径的超集，所以不会把原来找得到的图弄丢。
        rng = np.random.default_rng(77)
        for trial in range(200):
            height = int(rng.integers(1, 10))
            width = int(rng.integers(1, 10))
            blocked = rng.random((height, width)) < 0.25
            fast = _largest_clear_rectangle(blocked, 0, width - 1, 0, height - 1)
            exact = _largest_clear_rectangle_exact(blocked, 0, width - 1, 0, height - 1)
            with self.subTest(trial=trial):
                self.assertGreaterEqual(exact[0] if exact else 0, fast[0] if fast else 0)

    def test_an_obstacle_row_does_not_hide_the_rectangle_beside_it(self):
        # 一整行障碍横穿条带：快速路径返回 None，精确搜索应给出半块。
        blocked = np.zeros((60, 100), dtype=bool)
        blocked[30, :] = True
        self.assertIsNone(_largest_clear_rectangle(blocked, 0, 99, 0, 59))
        found = _largest_clear_rectangle_exact(blocked, 0, 99, 0, 59)
        self.assertIsNotNone(found)
        self.assertEqual(found[0], 3000)  # 30 行 × 100 列


class BackgroundSearchFallbackTests(unittest.TestCase):
    """端到端：误判为「没有干净背景」的图要被救回来。"""

    def _scenario(self):
        """虫体在上下边缘、中间横穿一条人工排除带。"""
        instances = np.zeros((200, 200), dtype=np.int32)
        instances[10, 150] = 1
        instances[190, 150] = 1
        forbidden = np.zeros((200, 200), dtype=bool)
        forbidden[100, :] = True
        return instances, forbidden

    def test_a_band_crossing_the_strip_no_longer_reports_no_background(self):
        instances, forbidden = self._scenario()
        # 没有排除带时本来就能找到背景：说明这套输入本身是合理的，下面返回 None
        # 确实是那条带子造成的，不是用例本身构造得没法测。
        plain, _ = background_mask_and_polygon(instances)
        self.assertIsNotNone(plain, "the fixture is unusable: no background even without the band")

        mask, polygon = background_mask_and_polygon(instances, forbidden)
        self.assertIsNotNone(
            mask,
            "a clean rectangle far larger than the minimum was available above the "
            "exclusion band, but the image was reported as having no background")

    def test_the_background_never_touches_a_blocked_pixel(self):
        instances, forbidden = self._scenario()
        mask, _ = background_mask_and_polygon(instances, forbidden)
        self.assertIsNotNone(mask)
        self.assertFalse((mask & (instances > 0)).any(),
                         "the background rectangle covers a worm pixel")
        self.assertFalse((mask & forbidden).any(),
                         "the background rectangle covers an excluded pixel")

    def test_the_exported_polygon_is_the_exported_mask(self):
        # 两个测量后端要给出同一个背景值，靠的就是这两者始终是同一块区域。
        instances, forbidden = self._scenario()
        mask, polygon = background_mask_and_polygon(instances, forbidden)
        self.assertIsNotNone(mask)
        self.assertEqual(polygon.shape, (4, 2))
        xs, ys = polygon[:, 0], polygon[:, 1]
        # 导出的矩形画在像素边界上：每个被选中的像素中心都落在多边形内，紧邻其外的
        # 四个像素中心都落在多边形外。
        inside = ((np.arange(mask.shape[1])[None, :] > xs.min())
                  & (np.arange(mask.shape[1])[None, :] < xs.max()))
        inside_rows = ((np.arange(mask.shape[0])[:, None] > ys.min())
                       & (np.arange(mask.shape[0])[:, None] < ys.max()))
        covered = inside_rows & inside
        self.assertTrue(np.array_equal(covered, mask),
                        "mask and polygon cover different pixels")

    def test_a_truly_backgroundless_image_is_still_refused(self):
        # 回退不是「一律放行」：确实没有干净矩形时仍要拒绝。
        instances = np.ones((40, 40), dtype=np.int32)   # 整幅图都是虫体
        mask, polygon = background_mask_and_polygon(instances)
        self.assertIsNone(mask)
        self.assertIsNone(polygon)


if __name__ == "__main__":
    unittest.main()
