import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from manual_head_annotation import (  # noqa: E402
    ANNOTATION_FILENAME,
    BoundaryGuide,
    ExclusionRegion,
    HeadDirection,
    draw_exclusion_regions,
    draw_head_arrows,
    exclusion_mask,
    load_image_annotations,
    load_image_boundary_guides,
    load_image_exclusion_regions,
    match_head_directions,
    save_image_annotations,
    split_instances_by_head_directions,
)
from manual_head_annotation import (  # noqa: E402
    _ordered_arrow_assignment,
    _owners_survive,
    _partition_safety,
    _snap_seam_to_guides,
)


class ManualHeadAnnotationTests(unittest.TestCase):
    def test_sidecar_roundtrip_and_coordinate_scaling(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            image_path = Path(folder) / "sample.tif"
            arrow = HeadDirection(10, 20, 12, 4)
            destination = save_image_annotations(image_path, (100, 200), [arrow])
            self.assertEqual(destination.name, ANNOTATION_FILENAME)
            stored = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(stored["schema_version"], 3)
            loaded = load_image_annotations(image_path)
            self.assertEqual(loaded, [arrow])
            scaled = load_image_annotations(image_path, current_size=(200, 100))[0]
            self.assertEqual((scaled.tail_x, scaled.tail_y), (20, 10))
            self.assertEqual((scaled.head_x, scaled.head_y), (24, 2))

    def test_boundary_guide_roundtrip_scaling_and_backward_compatible_arrow_save(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            image_path = Path(folder) / "sample.tif"
            arrow = HeadDirection(10, 20, 12, 4)
            boundary = BoundaryGuide(((30, 5), (32, 40), (36, 80)))
            save_image_annotations(image_path, (100, 100), [arrow], [boundary])
            loaded = load_image_boundary_guides(image_path, current_size=(200, 50))
            self.assertEqual(loaded[0].points, ((60, 2.5), (64, 20), (72, 40)))
            # 旧调用只保存箭头时，不应抹掉已经存在的人工分界线。
            save_image_annotations(image_path, (100, 100), [arrow])
            self.assertEqual(load_image_boundary_guides(image_path), [boundary])

    def test_exclusion_roundtrip_scaling_mask_and_backward_compatible_save(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            image_path = Path(folder) / "dirty.tif"
            region = ExclusionRegion(((10, 20), (40, 20), (40, 60), (10, 60)))
            save_image_annotations(image_path, (100, 100), [], exclusions=[region])
            scaled = load_image_exclusion_regions(image_path, current_size=(200, 50))
            self.assertEqual(
                scaled[0].points, ((20, 10), (80, 10), (80, 30), (20, 30)))
            mask = exclusion_mask((50, 200), scaled)
            self.assertTrue(mask[20, 40])
            self.assertFalse(mask[5, 5])
            # 旧调用不传 exclusions 时必须保留已经圈好的排除区域。
            save_image_annotations(image_path, (100, 100), [])
            self.assertEqual(load_image_exclusion_regions(image_path), [region])

    def test_qc_exclusion_is_drawn_without_mutating_source(self):
        source = Image.new("RGB", (80, 80), (0, 0, 0))
        region = ExclusionRegion(((15, 15), (60, 15), (60, 60), (15, 60)))
        rendered = draw_exclusion_regions(source, [region])
        self.assertEqual(source.getpixel((15, 15)), (0, 0, 0))
        self.assertGreater(rendered.getpixel((15, 15))[0], 200)

    def test_arrows_match_unique_worms_and_locate_head_pixels(self):
        labels = np.zeros((100, 70), dtype=np.uint16)
        labels[10:90, 10:21] = 1
        labels[10:90, 40:51] = 2
        arrows = [
            HeadDirection(15, 55, 15, 5),
            HeadDirection(45, 45, 45, 95),
        ]
        matches, unmatched = match_head_directions(labels, arrows)
        self.assertEqual(unmatched, 0)
        self.assertEqual([item.status for item in matches], ["MATCHED", "MATCHED"])
        self.assertEqual((matches[0].head_x, matches[0].head_y), (15, 10))
        self.assertEqual((matches[1].head_x, matches[1].head_y), (45, 89))
        self.assertLess(matches[0].direction_y, 0)
        self.assertGreater(matches[1].direction_y, 0)

    def test_arrow_pressed_on_a_neighbour_is_matched_by_row_order(self):
        labels = np.zeros((100, 70), dtype=np.uint16)
        labels[10:90, 10:30] = 1
        labels[10:90, 30:50] = 2
        labels[10:90, 50:70] = 3
        arrows = [
            HeadDirection(18, 80, 18, 20),
            HeadDirection(36, 80, 36, 20),
            # 画在第 3 条头上的那支，尾端压进了第 2 条：杆子前半段整段都在 2 里。
            HeadDirection(34, 84, 56, 24),
        ]
        matches, unmatched = match_head_directions(labels, arrows)
        self.assertEqual(unmatched, 0)
        self.assertEqual([item.status for item in matches], ["MATCHED"] * 3)
        self.assertEqual(matches[2].match_method, "ordered")
        self.assertEqual((matches[2].arrow_tail_x, matches[2].arrow_tail_y), (34, 84))
        self.assertAlmostEqual(matches[2].match_distance_px, 16.0, delta=1.0)
        self.assertEqual((matches[1].match_method, matches[2].match_method),
                         ("tail", "ordered"))

    def test_spare_arrow_is_not_forced_onto_an_already_matched_worm(self):
        labels = np.zeros((100, 70), dtype=np.uint16)
        labels[10:90, 10:30] = 1
        labels[10:90, 30:50] = 2
        arrows = [
            HeadDirection(36, 80, 36, 20),
            HeadDirection(38, 84, 58, 24),
            HeadDirection(16, 80, 16, 20),
        ]
        matches, unmatched = match_head_directions(labels, arrows)
        self.assertEqual(unmatched, 1)
        self.assertEqual([item.status for item in matches], ["MATCHED", "MATCHED"])
        self.assertEqual([item.match_method for item in matches], ["tail", "tail"])

    def test_duplicate_and_far_background_arrows_are_not_forced(self):
        labels = np.zeros((100, 100), dtype=np.uint16)
        labels[20:80, 45:56] = 1
        arrows = [
            HeadDirection(50, 50, 50, 15),
            HeadDirection(50, 60, 50, 85),
            HeadDirection(1, 1, 1, 15),
        ]
        matches, unmatched = match_head_directions(labels, arrows)
        self.assertEqual(matches[0].status, "MATCHED")
        self.assertEqual(unmatched, 2)

    def test_qc_arrow_is_drawn_without_mutating_source(self):
        labels = np.zeros((80, 80), dtype=np.uint16)
        labels[10:70, 35:46] = 1
        matches, _ = match_head_directions(
            labels, [HeadDirection(40, 55, 40, 5)])
        source = Image.new("RGB", (80, 80), (0, 0, 0))
        rendered = draw_head_arrows(source, matches)
        self.assertEqual(source.getpixel((40, 30)), (0, 0, 0))
        self.assertGreater(rendered.getpixel((40, 30))[0], 200)

    def test_two_arrows_split_one_boundary_joined_instance(self):
        labels = np.zeros((180, 120), dtype=np.uint16)
        labels[10:170, 25:86] = 1
        image = np.zeros(labels.shape, dtype=np.float32)
        image[labels > 0] = 0.8
        image[10:170, 54:58] = 0.12
        arrows = [
            HeadDirection(39, 55, 36, 5),
            HeadDirection(72, 55, 75, 5),
        ]
        result, reports = split_instances_by_head_directions(
            labels, arrows, image=image, max_instances=3)
        self.assertEqual(int(result.max()), 2)
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0].status, "REVIEW_MANUAL_HEAD_SPLIT")
        self.assertTrue(np.array_equal(result > 0, labels > 0))
        for label in (1, 2):
            components, count = ndimage.label(result == label)
            self.assertEqual(count, 1, components.max())

    def test_arrows_pressed_on_the_head_seam_still_split_three_bodies(self):
        """箭头压在虫头那儿的缝上时，投票会把两支箭头算进同一段。

        0913-0-2 实测：三段并排的虫体被模型连成一块，实验员一段一支箭头，但
        虫头一端两段只隔 40 像素、箭头又短，前两支的杆子落在了邻段那一侧，
        投票结果是 [1,1,3]，整块于是被原样输出（用户看到的仍是「2、3、4 条
        虫判成一条」）。这时改按箭头与子块的左右次序配对，配出来的正是实验员
        想要的那一份。
        """
        labels = np.zeros((460, 220), dtype=np.uint16)
        for offset in (0, 34, 68):
            for step in range(380):
                centre = 70 + offset + 0.3 * step
                half = 22 if 6 < step < 374 else 12
                labels[40 + step, int(centre - half):int(centre + half)] = 1
        # 前两支箭头的尾端都压在所在虫体左侧的缝上，第三支画在虫身中段。
        arrows = [
            HeadDirection(57, 110, 87, 46),
            HeadDirection(91, 110, 121, 46),
            HeadDirection(150, 110, 155, 46),
        ]
        result, reports = split_instances_by_head_directions(
            labels, arrows, max_instances=3)
        self.assertEqual(int(result.max()), 3)
        self.assertEqual([report.method for report in reports], ["manual_head_coarse"])
        self.assertTrue(np.array_equal(result > 0, labels > 0))
        fractions = []
        for label in (1, 2, 3):
            components, count = ndimage.label(result == label)
            self.assertEqual(count, 1, components.max())
            fractions.append((result == label).sum() / (labels > 0).sum())
        self.assertGreater(min(fractions), 0.30)

    def test_arrows_all_pressed_into_one_body_are_not_split(self):
        """全部箭头都落在同一段上时不能硬拆：次序配对与投票整体对不上。"""
        partition = np.zeros((120, 120), dtype=np.uint16)
        partition[:, 0:40], partition[:, 40:80], partition[:, 80:120] = 1, 2, 3
        mask = partition > 0
        pressed = [HeadDirection(58, 100, 60, 20),
                   HeadDirection(60, 100, 62, 20),
                   HeadDirection(62, 100, 64, 20)]
        self.assertIsNone(_partition_safety(partition, mask, pressed, 120))
        self.assertEqual(_ordered_arrow_assignment(partition, pressed), [1, 2, 3])
        ordered = [HeadDirection(20, 100, 22, 20),
                   HeadDirection(60, 100, 62, 20),
                   HeadDirection(100, 100, 102, 20)]
        self.assertIsNotNone(_partition_safety(partition, mask, ordered, 120))

    def test_separate_instances_and_duplicate_arrows_are_unchanged(self):
        labels = np.zeros((120, 100), dtype=np.uint16)
        labels[10:110, 15:31] = 1
        labels[10:110, 60:76] = 2
        separate = [HeadDirection(23, 40, 23, 5), HeadDirection(68, 40, 68, 5)]
        result, reports = split_instances_by_head_directions(labels, separate)
        self.assertEqual(reports, [])
        self.assertTrue(np.array_equal(result, labels))

    def test_short_diagonal_arrows_can_split_curved_or_converging_bodies(self):
        labels = np.zeros((190, 150), dtype=np.uint16)
        labels[10:180, 20:121] = 1
        image = np.zeros(labels.shape, dtype=np.float32)
        image[labels > 0] = 0.8
        image[10:180, 68:73] = 0.08
        # 头部的短箭头均向左上；旧的无限直线外推会在尾端跑到虫体右侧。
        arrows = [
            HeadDirection(48, 52, 28, 7),
            HeadDirection(91, 52, 71, 7),
        ]
        result, reports = split_instances_by_head_directions(
            labels, arrows, image=image, max_instances=2)
        self.assertEqual(int(result.max()), 2)
        self.assertEqual(len(reports), 1)
        self.assertTrue(np.array_equal(result > 0, labels > 0))
        fractions = [(result == label).sum() / (labels > 0).sum() for label in (1, 2)]
        self.assertGreater(min(fractions), 0.40)
        matches, unmatched = match_head_directions(result, arrows)
        self.assertEqual(unmatched, 0)
        self.assertTrue(all(match.status == "MATCHED" for match in matches))

    def test_head_coarse_partition_runs_before_boundary_refinement(self):
        labels = np.zeros((190, 150), dtype=np.uint16)
        labels[10:180, 18:132] = 1
        image = np.ones(labels.shape, dtype=np.float32) * 0.7
        arrows = [HeadDirection(46, 55, 43, 7), HeadDirection(104, 55, 107, 7)]
        boundary = BoundaryGuide(((75, 18), (72, 65), (78, 115), (75, 170)))
        result, reports = split_instances_by_head_directions(
            labels, arrows, image=image, max_instances=2,
            boundary_guides=[boundary])
        self.assertEqual(int(result.max()), 2)
        self.assertEqual(
            [report.method for report in reports],
            ["manual_head_coarse", "manual_boundary_refine"])
        self.assertTrue(np.array_equal(result > 0, labels > 0))
        matches, unmatched = match_head_directions(result, arrows)
        self.assertEqual(unmatched, 0)
        self.assertTrue(all(match.status == "MATCHED" for match in matches))
        duplicate = [HeadDirection(23, 40, 23, 5), HeadDirection(24, 42, 24, 7)]
        result, reports = split_instances_by_head_directions(labels, duplicate)
        self.assertEqual(reports, [])
        self.assertTrue(np.array_equal(result, labels))

    def test_boundary_guide_can_split_without_head_arrows(self):
        labels = np.zeros((180, 120), dtype=np.uint16)
        labels[10:170, 20:101] = 1
        boundary = BoundaryGuide(((60, 10), (58, 65), (62, 120), (60, 169)))
        result, reports = split_instances_by_head_directions(
            labels, [], max_instances=2, boundary_guides=[boundary])
        self.assertEqual(int(result.max()), 2)
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0].method, "manual_boundary_refine")
        self.assertEqual(reports[0].arrow_count, 0)
        self.assertTrue(np.array_equal(result > 0, labels > 0))

    def test_boundary_contours_refine_existing_labels_without_head_arrows(self):
        labels = np.zeros((200, 110), dtype=np.uint16)
        labels[10:120, 20:50] = 1
        labels[10:105, 50:80] = 2
        # The lower part of the right worm is incorrectly owned by label 1.
        labels[105:185, 50:80] = 1
        desired = np.zeros_like(labels)
        desired[10:120, 20:50] = 1
        desired[10:185, 50:80] = 2
        guides = [
            BoundaryGuide(((20, 95), (20, 120), (49, 120), (49, 95))),
            BoundaryGuide(((50, 95), (50, 184), (79, 184), (79, 95))),
        ]
        result, reports = split_instances_by_head_directions(
            labels, [], max_instances=2, boundary_guides=guides)
        before = int((labels == desired).sum())
        after = int((result == desired).sum())
        self.assertGreater(after, before)
        self.assertEqual(int(result.max()), 2)
        self.assertEqual(reports[-1].status, "REVIEW_MANUAL_BOUNDARY_REFINE")
        self.assertTrue(np.array_equal(result > 0, labels > 0))


class SeamSnapTests(unittest.TestCase):
    """人工分界线是硬约束：接缝飘到一边时，要按实验员画的那条线压回来。"""

    def _scene(self, seam=105):
        """一块粘连虫体、两个芯，归属给出的接缝在画的线右边 seam-100 像素。"""
        shape = (200, 200)
        mask = np.zeros(shape, dtype=bool)
        mask[40:160, 20:180] = True
        owners = np.zeros(shape, dtype=np.uint8)
        owners[40:160, 20:seam] = 1
        owners[40:160, seam:180] = 2
        cores = []
        for left, right in ((45, 55), (152, 162)):
            core = np.zeros(shape, dtype=bool)
            core[60:140, left:right] = True
            cores.append(core)
        guide = BoundaryGuide(((100.0, 45.0), (100.0, 155.0)))
        return mask, owners, cores, guide

    def test_seam_moves_onto_the_drawn_line(self):
        # 100 像素的小图，带子半径 max(6, 200*0.04) = 8，接缝只飘了 5 像素。
        mask, owners, cores, guide = self._scene(seam=105)
        snapped = _snap_seam_to_guides(owners, cores, mask, [guide], 3)
        row = 100
        self.assertEqual(int(snapped[row, 99]), 1)
        self.assertEqual(int(snapped[row, 100]), 1)
        self.assertEqual(int(snapped[row, 101]), 2)
        self.assertTrue(np.array_equal(snapped > 0, mask))
        self.assertTrue(_owners_survive(owners, snapped))
        # 修正只在线的带子里，远处的归属不动。
        self.assertEqual(int(snapped[row, 40]), 1)
        self.assertEqual(int(snapped[row, 170]), 2)

    def test_seam_beyond_the_band_is_left_alone_without_breaking_a_worm(self):
        """飘得比带子还远时不硬压：压过去会把带外那条残条切下来。"""
        mask, owners, cores, guide = self._scene(seam=112)
        snapped = _snap_seam_to_guides(owners, cores, mask, [guide], 3)
        self.assertTrue(np.array_equal(snapped, owners))
        for label in (1, 2):
            _, count = ndimage.label(snapped == label)
            self.assertEqual(count, 1)

    def test_untouched_when_both_sides_are_one_worm(self):
        mask, owners, cores, _ = self._scene()
        owners = np.ones((200, 200), dtype=np.uint8)
        owners[~mask] = 0
        guide = BoundaryGuide(((100.0, 45.0), (100.0, 155.0)))
        snapped = _snap_seam_to_guides(owners, cores, mask, [guide], 3)
        self.assertTrue(np.array_equal(snapped, owners))


class SidecarMergeTests(unittest.TestCase):
    """ImageJ sidecar 与 GUI 文件夹记录必须按类型合并，且 GUI 保存能改到 sidecar。"""

    def _sidecar(self, image_path, arrows=(), boundaries=(), exclusions=()):
        sibling = image_path.with_name(image_path.name + ".autoworm.json")
        sibling.write_text(json.dumps({
            "schema_version": 3, "source": "ImageJ ROI Manager",
            "width": 2048, "height": 2048,
            "arrows": [asdict(arrow) for arrow in arrows],
            "boundaries": [asdict(guide) for guide in boundaries],
            "exclusions": [asdict(region) for region in exclusions],
        }), encoding="utf-8")
        return sibling

    def test_arrows_only_sidecar_keeps_gui_boundaries(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            image_path = Path(folder) / "sample.tif"
            arrow = HeadDirection(40, 50, 20, 10)
            guide = BoundaryGuide(((60, 5), (58, 60), (62, 115)))
            save_image_annotations(image_path, (2048, 2048), [arrow], [guide])
            # ImageJ 只画了头向箭头，sidecar 里 boundaries 是空数组。
            self._sidecar(image_path, arrows=[HeadDirection(41, 51, 21, 11)])
            self.assertEqual(load_image_boundary_guides(image_path, (2048, 2048)), [guide])
            loaded = load_image_annotations(image_path, (2048, 2048))
            self.assertEqual(loaded, [HeadDirection(41, 51, 21, 11)])

    def test_non_empty_sidecar_list_still_overrides_the_folder_store(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            image_path = Path(folder) / "sample.tif"
            save_image_annotations(
                image_path, (2048, 2048), [HeadDirection(40, 50, 20, 10)],
                [BoundaryGuide(((60, 5), (58, 60), (62, 115)))])
            sidecar_guide = BoundaryGuide(((10, 10), (20, 20), (30, 30)))
            self._sidecar(image_path, boundaries=[sidecar_guide])
            self.assertEqual(
                load_image_boundary_guides(image_path, (2048, 2048)), [sidecar_guide])

    def test_gui_save_writes_back_to_an_existing_sidecar(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            image_path = Path(folder) / "sample.tif"
            sidecar_guide = BoundaryGuide(((30, 5), (28, 60), (32, 115)))
            self._sidecar(image_path, boundaries=[sidecar_guide])
            # 1. GUI 删掉全部分界：sidecar 也要被清空，否则删除会被合并规则挡住。
            save_image_annotations(image_path, (2048, 2048), [], [])
            self.assertEqual(load_image_boundary_guides(image_path, (2048, 2048)), [])
            # 2. GUI 画一条新分界：sidecar 里也要是新的这条。
            gui_guide = BoundaryGuide(((60, 5), (58, 60), (62, 115)))
            save_image_annotations(image_path, (2048, 2048), [], [gui_guide])
            self.assertEqual(
                load_image_boundary_guides(image_path, (2048, 2048)), [gui_guide])

    def test_saving_without_boundaries_keeps_sidecar_only_guides(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            image_path = Path(folder) / "sample.tif"
            arrow = HeadDirection(40, 50, 20, 10)
            sidecar_guide = BoundaryGuide(((30, 5), (28, 60), (32, 115)))
            self._sidecar(image_path, arrows=[arrow], boundaries=[sidecar_guide])
            # boundaries 缺省表示"不改动"，这里不能因为文件夹记录是空的就清掉它。
            save_image_annotations(image_path, (2048, 2048), [arrow])
            self.assertEqual(
                load_image_boundary_guides(image_path, (2048, 2048)), [sidecar_guide])

    def test_multi_image_file_is_not_clobbered_by_sidecar_write_back(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            image_path = Path(folder) / "sample.tif"
            sibling = image_path.with_name(image_path.name + ".autoworm.json")
            foreign = {"schema_version": 3, "images": {"other.tif": {"arrows": []}}}
            sibling.write_text(json.dumps(foreign), encoding="utf-8")
            save_image_annotations(image_path, (2048, 2048), [HeadDirection(1, 2, 3, 4)])
            self.assertEqual(json.loads(sibling.read_text(encoding="utf-8")), foreign)


if __name__ == "__main__":
    unittest.main()
