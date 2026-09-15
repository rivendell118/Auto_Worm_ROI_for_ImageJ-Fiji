"""按人工头向和中轴线弧长选择每条线虫的定量区段。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from worm_shape_refiner import _longest_skeleton_path


@dataclass
class SegmentSelectionReport:
    worm_label: int
    status: str
    start_fraction: float
    end_fraction: float
    head_x: float
    head_y: float
    centerline_length_px: float
    original_area: int
    selected_area: int
    selected_area_fraction: float


def _matched_heads(head_matches) -> dict[int, object]:
    return {
        int(match.worm_label): match
        for match in head_matches
        if getattr(match, "status", "") == "MATCHED"
        and getattr(match, "head_x", None) is not None
        and getattr(match, "head_y", None) is not None
    }


def _oriented_centerline(mask: np.ndarray, head_x: float, head_y: float):
    """返回从头到尾排列的局部坐标中轴线及其归一化弧长。"""
    path = _longest_skeleton_path(mask)
    if path is None or len(path) < 2:
        return None, None
    first_distance = (path[0, 0] - head_x) ** 2 + (path[0, 1] - head_y) ** 2
    last_distance = (path[-1, 0] - head_x) ** 2 + (path[-1, 1] - head_y) ** 2
    if last_distance < first_distance:
        path = path[::-1].copy()
    delta = np.diff(path, axis=0)
    cumulative = np.concatenate((
        [0.0], np.cumsum(np.sqrt((delta * delta).sum(axis=1)))))
    total = float(cumulative[-1])
    if total < 7.0:
        return None, None
    return path, cumulative / total


def _local_instance(instances: np.ndarray, label: int):
    yy, xx = np.nonzero(instances == label)
    if not len(xx):
        return None
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    return y0, y1, x0, x1, instances[y0:y1, x0:x1] == label


def _fraction_map(mask: np.ndarray, path: np.ndarray,
                  fractions: np.ndarray) -> np.ndarray:
    centerline = np.zeros(mask.shape, dtype=bool)
    values = np.zeros(mask.shape, dtype=np.float32)
    xi = np.clip(np.rint(path[:, 0]).astype(int), 0, mask.shape[1] - 1)
    yi = np.clip(np.rint(path[:, 1]).astype(int), 0, mask.shape[0] - 1)
    centerline[yi, xi] = True
    values[yi, xi] = fractions.astype(np.float32)
    nearest = ndimage.distance_transform_edt(
        ~centerline, return_distances=False, return_indices=True)
    return values[tuple(nearest)]


def select_instance_segments(instances: np.ndarray, head_matches,
                             start_fraction: float, end_fraction: float):
    """沿每条虫从头到尾的中轴线弧长保留指定闭区间。

    输出始终是原实例的子集，不新增像素。若任一实例缺少唯一有效头向，或无法
    得到连续中轴线，则拒绝生成可能方向错误的科研 ROI。
    """
    source = np.asarray(instances)
    if source.ndim != 2:
        raise ValueError("instances must be a two-dimensional label map")
    start = float(start_fraction)
    end = float(end_fraction)
    if not (0.0 <= start < end <= 1.0):
        raise ValueError("segment range must satisfy 0 <= start < end <= 1")
    labels = [int(label) for label in np.unique(source) if label > 0]
    heads = _matched_heads(head_matches)
    missing = [label for label in labels if label not in heads]
    if missing:
        raise ValueError(
            "segment selection requires one matched head arrow for every worm; "
            "missing labels: " + ",".join(map(str, missing)))

    output = np.zeros(source.shape, dtype=source.dtype)
    reports = []
    for label in labels:
        local = _local_instance(source, label)
        if local is None:
            continue
        y0, y1, x0, x1, mask = local
        match = heads[label]
        local_head_x = float(match.head_x) - x0
        local_head_y = float(match.head_y) - y0
        path, fractions = _oriented_centerline(mask, local_head_x, local_head_y)
        if path is None:
            raise ValueError("unable to determine a continuous centerline for worm %d" % label)
        original_area = int(mask.sum())
        if start <= 0.0 and end >= 1.0:
            selected = mask.copy()
        else:
            position = _fraction_map(mask, path, fractions)
            selected = mask & (position >= start) & (position <= end)
            components, count = ndimage.label(
                selected, structure=np.ones((3, 3), dtype=np.uint8))
            if count:
                sizes = np.bincount(components.ravel())
                sizes[0] = 0
                selected = components == int(np.argmax(sizes))
        selected_area = int(selected.sum())
        if selected_area <= 0:
            raise ValueError("selected centerline range is empty for worm %d" % label)
        output[y0:y1, x0:x1][selected] = label
        centerline_length = float(np.sqrt((np.diff(path, axis=0) ** 2).sum(axis=1)).sum())
        reports.append(SegmentSelectionReport(
            worm_label=label,
            status="SELECTED",
            start_fraction=start,
            end_fraction=end,
            head_x=float(match.head_x),
            head_y=float(match.head_y),
            centerline_length_px=centerline_length,
            original_area=original_area,
            selected_area=selected_area,
            selected_area_fraction=selected_area / max(original_area, 1),
        ))
    if set(np.unique(output)) != set(np.unique(source)):
        raise RuntimeError("segment selection changed the instance count")
    if np.any((output > 0) & (source == 0)):
        raise RuntimeError("segment selection added pixels outside the original instances")
    return output, reports


def draw_segment_guides(image: Image.Image, whole_instances: np.ndarray, head_matches,
                        start_fraction: float, end_fraction: float) -> Image.Image:
    """在 QC 图上画蓝色整条中轴线及橙色选中区段。"""
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    heads = _matched_heads(head_matches)
    width = max(2, int(round(min(image.size) / 550.0)))
    for label in [int(value) for value in np.unique(whole_instances) if value > 0]:
        if label not in heads:
            continue
        local = _local_instance(whole_instances, label)
        if local is None:
            continue
        y0, _y1, x0, _x1, mask = local
        match = heads[label]
        path, fractions = _oriented_centerline(
            mask, float(match.head_x) - x0, float(match.head_y) - y0)
        if path is None:
            continue
        points = [(float(x + x0), float(y + y0)) for x, y in path]
        draw.line(points, fill=(65, 170, 255), width=width, joint="curve")
        selected = (fractions >= float(start_fraction)) & (fractions <= float(end_fraction))
        selected_points = [point for point, keep in zip(points, selected) if keep]
        if len(selected_points) >= 2:
            draw.line(selected_points, fill=(255, 145, 20), width=width + 2, joint="curve")
            radius = max(3, width + 1)
            for x, y in (selected_points[0], selected_points[-1]):
                draw.ellipse((x-radius, y-radius, x+radius, y+radius),
                             fill=(255, 145, 20), outline=(255, 255, 255), width=1)
    return canvas


SEGMENT_REPORT_FIELDS = list(SegmentSelectionReport.__dataclass_fields__)
