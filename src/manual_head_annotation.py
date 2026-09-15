"""人工头向、分界线和排除区域的持久化、匹配与结果绘制。

标注以独立 JSON sidecar 保存，不修改原始 TIFF。实验员从虫体内部向头部拖动
箭头；推理完成后，本模块根据箭头起点/箭杆经过的实例区域，将箭头关联到虫体。
头向粗分与人工分界精修是两个可独立工作的平级算法；二者同时存在时固定先粗分、
再精修。闭合排除区域用于屏蔽不需要识别的脏背景。
"""

from __future__ import annotations

import json
import math
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from inspect_roi_dataset import image_array


ANNOTATION_FILENAME = "_manual_head_annotations.json"
IMAGEJ_SIDECAR_SUFFIX = ".autoworm.json"
SCHEMA_VERSION = 3
ANNOTATION_KEYS = ("arrows", "boundaries", "exclusions")

# Radius of the band around a drawn boundary inside which the seam is pushed
# back onto it, as a fraction of the shorter image side (see
# _snap_seam_to_guides). Wider bands reach further misplaced seams but also
# sample more of the neighbouring territories; 0.04 was the best value measured
# on the annotated 0913-0-2 image (205/220 samples split, vs 192/220 unmodified).
SNAP_BAND_FRACTION = 0.04

# Offsets, as fractions of the band radius, at which each side of a guide is
# probed, innermost first. Sampling only the outer part of the band is what makes
# a seam that drifted towards one side visible: the probe lands beyond it and
# reports the territory that should own that side. Both offsets are read, and the
# outer one wins wherever it lands on a worm, so a side whose inner probe went
# astray still gets a target.
SNAP_PROBE_OFFSETS = (0.85, 1.0)


@dataclass(frozen=True)
class HeadDirection:
    tail_x: float
    tail_y: float
    head_x: float
    head_y: float


@dataclass(frozen=True)
class BoundaryGuide:
    points: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class ExclusionRegion:
    points: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class HeadMatch:
    worm_label: int
    roi_name: str
    status: str
    head_x: float | None
    head_y: float | None
    arrow_tail_x: float | None
    arrow_tail_y: float | None
    arrow_head_x: float | None
    arrow_head_y: float | None
    direction_x: float | None
    direction_y: float | None
    arrow_length_px: float | None
    match_distance_px: float | None
    match_method: str


@dataclass
class ManualSplitReport:
    original_label: int
    arrow_count: int
    status: str
    min_child_area_fraction: float
    min_child_span_fraction: float
    arrow_separation_px: float
    method: str = "auto_image_seam"
    resulting_labels: str = ""


def annotation_path(folder: str | os.PathLike[str]) -> Path:
    return Path(folder) / ANNOTATION_FILENAME


def _valid_arrow(value) -> HeadDirection | None:
    if not isinstance(value, dict):
        return None
    try:
        numbers = [float(value[key]) for key in ("tail_x", "tail_y", "head_x", "head_y")]
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(number) for number in numbers):
        return None
    arrow = HeadDirection(*numbers)
    if math.hypot(arrow.head_x - arrow.tail_x, arrow.head_y - arrow.tail_y) < 1.0:
        return None
    return arrow


def _valid_boundary(value) -> BoundaryGuide | None:
    if isinstance(value, BoundaryGuide):
        value = asdict(value)
    if not isinstance(value, dict) or not isinstance(value.get("points"), (list, tuple)):
        return None
    points = []
    for item in value["points"]:
        try:
            if isinstance(item, dict):
                point = float(item["x"]), float(item["y"])
            else:
                point = float(item[0]), float(item[1])
        except (KeyError, IndexError, TypeError, ValueError):
            return None
        if not all(math.isfinite(number) for number in point):
            return None
        if not points or math.hypot(point[0] - points[-1][0], point[1] - points[-1][1]) >= 0.5:
            points.append(point)
    if len(points) < 2:
        return None
    length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))
    return BoundaryGuide(tuple(points)) if length >= 3.0 else None


def _valid_exclusion(value) -> ExclusionRegion | None:
    if isinstance(value, ExclusionRegion):
        value = asdict(value)
    if not isinstance(value, dict) or not isinstance(value.get("points"), (list, tuple)):
        return None
    points = []
    for item in value["points"]:
        try:
            if isinstance(item, dict):
                point = float(item["x"]), float(item["y"])
            else:
                point = float(item[0]), float(item[1])
        except (KeyError, IndexError, TypeError, ValueError):
            return None
        if not all(math.isfinite(number) for number in point):
            return None
        if not points or math.hypot(point[0] - points[-1][0], point[1] - points[-1][1]) >= 0.5:
            points.append(point)
    if len(points) >= 2 and math.hypot(
            points[0][0] - points[-1][0], points[0][1] - points[-1][1]) < 0.5:
        points.pop()
    if len(points) < 3:
        return None
    area_twice = abs(sum(
        first[0] * second[1] - second[0] * first[1]
        for first, second in zip(points, points[1:] + points[:1])))
    return ExclusionRegion(tuple(points)) if area_twice >= 8.0 else None


def _read_store(folder: str | os.PathLike[str]) -> dict:
    path = annotation_path(folder)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": SCHEMA_VERSION, "images": {}}
    if not isinstance(data, dict) or not isinstance(data.get("images"), dict):
        return {"schema_version": SCHEMA_VERSION, "images": {}}
    return data


def _image_entry(store: dict, image_name: str) -> dict | None:
    images = store.get("images", {})
    direct = images.get(image_name)
    if isinstance(direct, dict):
        return direct
    folded = image_name.casefold()
    for key, value in images.items():
        if str(key).casefold() == folded and isinstance(value, dict):
            return value
    return None


def _sidecar_entry(path: Path) -> dict | None:
    """Read the ImageJ per-image sidecar written by Auto_Worm_Annotations."""
    sidecar = path.with_name(path.name + IMAGEJ_SIDECAR_SUFFIX)
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if isinstance(data.get("images"), dict):
        return _image_entry(data, path.name)
    if any(key in data for key in ANNOTATION_KEYS):
        return data
    return None


def _read_image_entry(path: Path) -> dict | None:
    """Merge the ImageJ sidecar with the GUI folder store, type by type.

    The ImageJ plug-in writes ``<image-name>.autoworm.json`` so saving an
    annotation for one image never rewrites annotations belonging to the other
    images in the same acquisition folder.  That sidecar is always written with
    all three lists, so an empty list means "the ROI Manager held nothing of
    that kind" rather than "drop the annotation of that kind".  Merging per
    type keeps a sidecar holding only head arrows from silently discarding the
    boundaries the GUI stored for the same image.
    """
    sidecar = _sidecar_entry(path)
    legacy = _image_entry(_read_store(path.parent), path.name)
    if sidecar is None:
        return legacy
    if legacy is None:
        return sidecar
    merged = dict(legacy)
    for key, value in sidecar.items():
        if key in ANNOTATION_KEYS and not (
                isinstance(value, (list, tuple)) and len(value)):
            continue
        merged[key] = value
    return merged


def load_image_annotations(
        image_path: str | os.PathLike[str], current_size: tuple[int, int] | None = None
) -> list[HeadDirection]:
    """读取一张 TIFF 的箭头；若图像尺寸发生变化，则按比例恢复坐标。"""
    path = Path(image_path)
    entry = _read_image_entry(path)
    if not entry:
        return []
    arrows = [arrow for arrow in (_valid_arrow(value) for value in entry.get("arrows", [])) if arrow]
    if not current_size:
        return arrows
    try:
        saved_width = float(entry["width"])
        saved_height = float(entry["height"])
    except (KeyError, TypeError, ValueError):
        return arrows
    width, height = current_size
    if saved_width <= 0 or saved_height <= 0 or (saved_width == width and saved_height == height):
        return arrows
    sx, sy = width / saved_width, height / saved_height
    return [HeadDirection(
        arrow.tail_x * sx, arrow.tail_y * sy,
        arrow.head_x * sx, arrow.head_y * sy) for arrow in arrows]


def load_image_boundary_guides(
        image_path: str | os.PathLike[str], current_size: tuple[int, int] | None = None
) -> list[BoundaryGuide]:
    """读取人工分界曲线，并在图像尺寸变化时同步缩放。"""
    path = Path(image_path)
    entry = _read_image_entry(path)
    if not entry:
        return []
    guides = [guide for guide in (_valid_boundary(value)
                                   for value in entry.get("boundaries", [])) if guide]
    if not current_size:
        return guides
    try:
        saved_width = float(entry["width"])
        saved_height = float(entry["height"])
    except (KeyError, TypeError, ValueError):
        return guides
    width, height = current_size
    if saved_width <= 0 or saved_height <= 0 or (saved_width == width and saved_height == height):
        return guides
    sx, sy = width / saved_width, height / saved_height
    return [BoundaryGuide(tuple((x * sx, y * sy) for x, y in guide.points))
            for guide in guides]


def load_image_exclusion_regions(
        image_path: str | os.PathLike[str], current_size: tuple[int, int] | None = None
) -> list[ExclusionRegion]:
    """读取闭合排除区域，并在图像尺寸变化时同步缩放。"""
    path = Path(image_path)
    entry = _read_image_entry(path)
    if not entry:
        return []
    regions = [region for region in (_valid_exclusion(value)
                                      for value in entry.get("exclusions", [])) if region]
    if not current_size:
        return regions
    try:
        saved_width = float(entry["width"])
        saved_height = float(entry["height"])
    except (KeyError, TypeError, ValueError):
        return regions
    width, height = current_size
    if saved_width <= 0 or saved_height <= 0 or (saved_width == width and saved_height == height):
        return regions
    sx, sy = width / saved_width, height / saved_height
    return [ExclusionRegion(tuple((x * sx, y * sy) for x, y in region.points))
            for region in regions]


def _write_json_atomic(destination: Path, payload: dict) -> None:
    temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def save_image_annotations(
        image_path: str | os.PathLike[str], image_size: tuple[int, int], arrows,
        boundaries=None, exclusions=None
) -> Path:
    """原子写入文件夹标注；未传入的标注类型沿用该图原有数据。

    若该图已存在 ImageJ sidecar，则把同一份结果同步写回 sidecar。GUI 读取的是
    sidecar 与文件夹记录按类型合并后的结果，所以写回不会丢内容，而能让最近一次
    保存的一侧成为权威——否则 GUI 对分界的删除或改动会被非空的 sidecar 列表压掉。
    """
    path = Path(image_path)
    width, height = (int(image_size[0]), int(image_size[1]))
    if width <= 0 or height <= 0:
        raise ValueError("image_size must contain positive width and height")
    valid = [arrow for arrow in (_valid_arrow(
        asdict(value) if isinstance(value, HeadDirection) else value) for value in arrows) if arrow]
    store = _read_store(path.parent)
    # 合并后的视图，这样 boundaries=None 时也不会漏掉只存在于 sidecar 的标注。
    previous = _read_image_entry(path) or {}
    if boundaries is None:
        valid_boundaries = [guide for guide in (_valid_boundary(value)
                                                for value in previous.get("boundaries", [])) if guide]
    else:
        valid_boundaries = [guide for guide in (_valid_boundary(value) for value in boundaries) if guide]
    if exclusions is None:
        valid_exclusions = [region for region in (_valid_exclusion(value)
                                                   for value in previous.get("exclusions", [])) if region]
    else:
        valid_exclusions = [region for region in (_valid_exclusion(value)
                                                   for value in exclusions) if region]
    store["schema_version"] = SCHEMA_VERSION
    images = store.setdefault("images", {})
    # 清除仅大小写不同的旧键，避免 Windows 上一张图出现两份记录。
    for key in list(images):
        if str(key).casefold() == path.name.casefold() and key != path.name:
            del images[key]
    entry = {
        "width": width,
        "height": height,
        "arrows": [asdict(arrow) for arrow in valid],
        "boundaries": [asdict(guide) for guide in valid_boundaries],
        "exclusions": [asdict(region) for region in valid_exclusions],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    images[path.name] = entry
    destination = annotation_path(path.parent)
    _write_json_atomic(destination, store)

    sidecar = path.with_name(path.name + IMAGEJ_SIDECAR_SUFFIX)
    if sidecar.is_file():
        try:
            existing = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        # 逐图 sidecar 是单图格式；若该名下的文件是多图格式，不去覆盖它。
        if not isinstance(existing, dict) or not isinstance(existing.get("images"), dict):
            payload = {"schema_version": SCHEMA_VERSION, "source": "AutoWorm GUI"}
            payload.update({key: entry[key] for key in
                            ("width", "height", *ANNOTATION_KEYS)})
            _write_json_atomic(sidecar, payload)
    return destination


def enhanced_tiff_rgb(image_path: str | os.PathLike[str]) -> Image.Image:
    """将常见 8/16 位 TIFF 拉伸为适合监视器标注的黑白灰度图。"""
    # image_array, not np.asarray: plain asarray turns a 16-bit big-endian TIFF
    # into the wrong numbers, and the annotation view must show the same pixels
    # the engine segments.
    with Image.open(image_path) as opened:
        array = image_array(opened)
    if array.ndim == 3:
        if array.shape[2] >= 3:
            rgb = np.asarray(array[..., :3], dtype=np.float32)
            array = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] +
                     0.114 * rgb[..., 2])
        else:
            array = array[..., 0]
    finite = np.asarray(array)[np.isfinite(array)]
    positive = finite[finite > 0]
    sampled = positive[::16]
    high = float(np.percentile(sampled, 99.8)) if sampled.size else 1.0
    gray = np.clip(np.asarray(array, dtype=np.float32) *
                   (255.0 / max(high, 1.0)), 0, 255).astype(np.uint8)
    return Image.fromarray(gray, "L").convert("RGB")


def exclusion_mask(image_shape: tuple[int, int], regions) -> np.ndarray:
    """把排除区域转换为与原图一致的布尔掩膜；True 表示禁止识别。"""
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("image_shape must contain positive height and width")
    mask_image = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(mask_image)
    valid = [region for region in (
        value if isinstance(value, ExclusionRegion) else _valid_exclusion(value)
        for value in (regions or [])) if region]
    for region in valid:
        points = [(min(max(float(x), 0.0), width - 1.0),
                   min(max(float(y), 0.0), height - 1.0)) for x, y in region.points]
        draw.polygon(points, fill=1)
    return np.asarray(mask_image, dtype=bool)


def _labels_along_arrow(instances: np.ndarray, arrow: HeadDirection,
                        nearest_info=None) -> tuple[int, str, float]:
    """优先使用起点附近和箭杆前半段的像素投票关联实例。"""
    height, width = instances.shape
    xs = np.linspace(arrow.tail_x, arrow.tail_x + 0.45 * (arrow.head_x - arrow.tail_x), 25)
    ys = np.linspace(arrow.tail_y, arrow.tail_y + 0.45 * (arrow.head_y - arrow.tail_y), 25)
    xi = np.clip(np.rint(xs).astype(int), 0, width - 1)
    yi = np.clip(np.rint(ys).astype(int), 0, height - 1)
    labels = instances[yi, xi]
    positive = labels[labels > 0]
    if positive.size:
        counts = np.bincount(positive.astype(np.int64))
        label = int(np.argmax(counts))
        yy, xx = np.nonzero(instances == label)
        distance = float(np.sqrt(np.min((xx - arrow.tail_x) ** 2 + (yy - arrow.tail_y) ** 2)))
        method = "tail" if int(instances[yi[0], xi[0]]) == label else "shaft"
        return label, method, distance
    if nearest_info is None:
        return 0, "unmatched", float("inf")
    distances, indices = nearest_info
    tx = int(np.clip(round(arrow.tail_x), 0, width - 1))
    ty = int(np.clip(round(arrow.tail_y), 0, height - 1))
    nearest_y, nearest_x = int(indices[0, ty, tx]), int(indices[1, ty, tx])
    return int(instances[nearest_y, nearest_x]), "nearest", float(distances[ty, tx])


def match_head_directions(instances: np.ndarray, arrows) -> tuple[list[HeadMatch], int]:
    """把人工箭头唯一匹配到分割实例，并返回每条虫一行的头部信息。"""
    if instances.ndim != 2:
        raise ValueError("instances must be a two-dimensional label map")
    valid = [arrow for arrow in (
        value if isinstance(value, HeadDirection) else _valid_arrow(value) for value in arrows) if arrow]
    candidates = []
    diagonal = math.hypot(instances.shape[1], instances.shape[0])
    foreground = instances > 0
    nearest_info = (ndimage.distance_transform_edt(
        ~foreground, return_indices=True) if foreground.any() else None)
    for index, arrow in enumerate(valid):
        label, method, distance = _labels_along_arrow(instances, arrow, nearest_info)
        # 防止画在背景上的箭头被强行关联到很远的虫体。
        if method == "nearest" and distance > max(12.0, diagonal * 0.035):
            label, method = 0, "unmatched"
        candidates.append((distance, index, label, method, arrow))
    candidates.sort(key=lambda value: (value[0], value[1]))
    assigned: dict[int, tuple[str, float, HeadDirection]] = {}
    duplicate: list[HeadDirection] = []
    unmatched = 0
    for distance, _index, label, method, arrow in candidates:
        if label <= 0:
            # 画在背景上的箭头不强行关联到虫体。
            unmatched += 1
            continue
        if label in assigned:
            # 两支箭头压在同一条虫上：先记下来，等分完再看有没有虫没人认领。
            duplicate.append(arrow)
            continue
        assigned[label] = (method, distance, arrow)
    unmatched += len(_assign_spare_arrows_by_order(instances, assigned, duplicate, valid))

    results = []
    for label in range(1, int(instances.max()) + 1):
        if label not in assigned:
            results.append(HeadMatch(
                label, "%02d_worm" % label, "MISSING", None, None,
                None, None, None, None, None, None, None, None, ""))
            continue
        method, distance, arrow = assigned[label]
        dx, dy = arrow.head_x - arrow.tail_x, arrow.head_y - arrow.tail_y
        length = math.hypot(dx, dy)
        yy, xx = np.nonzero(instances == label)
        nearest = int(np.argmin((xx - arrow.head_x) ** 2 + (yy - arrow.head_y) ** 2))
        results.append(HeadMatch(
            label, "%02d_worm" % label, "MATCHED",
            float(xx[nearest]), float(yy[nearest]),
            arrow.tail_x, arrow.tail_y, arrow.head_x, arrow.head_y,
            dx / length, dy / length, length, distance, method))
    return results, unmatched


def _arrow_groups(instances: np.ndarray, arrows: list[HeadDirection]):
    foreground = instances > 0
    nearest_info = (ndimage.distance_transform_edt(
        ~foreground, return_indices=True) if foreground.any() else None)
    diagonal = math.hypot(instances.shape[1], instances.shape[0])
    groups: dict[int, list[HeadDirection]] = {}
    for arrow in arrows:
        label, method, distance = _labels_along_arrow(instances, arrow, nearest_info)
        if label <= 0 or (method == "nearest" and distance > max(12.0, diagonal * 0.035)):
            continue
        groups.setdefault(label, []).append(arrow)
    return groups


def _axis_owner_partition(mask: np.ndarray, arrows: list[HeadDirection]):
    """按到各人工纵轴的距离，为原前景中的每个像素建立初始归属。"""
    yy, xx = np.indices(mask.shape, dtype=np.float32)
    directions = []
    for arrow in arrows:
        dx, dy = arrow.head_x - arrow.tail_x, arrow.head_y - arrow.tail_y
        length = math.hypot(dx, dy)
        if length < 1.0:
            return None
        directions.append(np.asarray((dx / length, dy / length), dtype=np.float64))
    # 纵轴没有正负方向；先把所有方向与第一条轴对齐，再使用共同方向。
    # 这能防止两根短箭头的轻微手抖被无限外推后在虫体中段交叉。
    reference = directions[0]
    aligned = [direction if float(np.dot(direction, reference)) >= 0 else -direction
               for direction in directions]
    common = np.mean(aligned, axis=0)
    common_length = float(np.linalg.norm(common))
    if common_length < 0.25:
        return None
    ux, uy = common / common_length
    costs = []
    for arrow in arrows:
        # 到箭头所指虫体纵轴的垂直距离。纵轴向头部及虫身后方无限延伸，
        # 因而短箭头也能为整条长虫提供稳定的左右归属先验。
        costs.append(np.abs((xx - arrow.tail_x) * uy - (yy - arrow.tail_y) * ux))
    owner = np.argmin(np.stack(costs), axis=0)
    return np.where(mask, owner + 1, 0).astype(np.uint16)


def _axis_partition_cores(mask: np.ndarray, arrows: list[HeadDirection]):
    """由人工箭头轴线建立互斥的长核心，供有序缝隙追踪使用。"""
    partition = _axis_owner_partition(mask, arrows)
    if partition is None:
        return None
    cores = []
    parent_y = np.nonzero(mask)[0]
    parent_span = max(int(np.ptp(parent_y)) + 1, 1)
    parent_area = int(mask.sum())
    for index in range(len(arrows)):
        core = partition == index + 1
        core = ndimage.binary_erosion(core, iterations=2)
        components, number = ndimage.label(
            core, structure=np.ones((3, 3), dtype=np.uint8))
        if number:
            sizes = np.bincount(components.ravel())
            sizes[0] = 0
            core = components == int(np.argmax(sizes))
        cy = np.nonzero(core)[0]
        if (int(core.sum()) < max(20, 0.06 * parent_area / len(arrows))
                or not len(cy) or int(np.ptp(cy)) + 1 < 0.42 * parent_span):
            return None
        cores.append(core)
    return cores


def _fraction_partition_cores(mask: np.ndarray, count: int):
    """建立随虫体外轮廓弯曲的横向分位核心。

    人工箭头通常只覆盖头部。若把短箭头直线无限外推，弯曲虫体的轴线可能在
    尾端跑出实例。这里让备用核心逐行跟随原实例的左右包络，再由图像暗缝决定
    最终边界；核心只提供各虫从左到右的拓扑顺序。
    """
    if count < 2 or not mask.any():
        return None
    active_rows = np.flatnonzero(mask.any(axis=1))
    left = np.full(mask.shape[0], np.nan, dtype=np.float64)
    right = np.full(mask.shape[0], np.nan, dtype=np.float64)
    for row in active_rows:
        values = np.flatnonzero(mask[row])
        left[row], right[row] = values[0], values[-1]
    span = right - left
    x_grid = np.arange(mask.shape[1], dtype=np.float64)[None, :]
    cores = []
    parent_span = max(int(np.ptp(active_rows)) + 1, 1)
    parent_area = int(mask.sum())
    connectivity = np.ones((3, 3), dtype=np.uint8)
    for index in range(count):
        # 每个预期子虫取自身横向区间中央 64% 作为核心，避开外边界和暗缝。
        lo = (index + 0.18) / count
        hi = (index + 0.82) / count
        core = (mask
                & (x_grid >= (left + lo * span)[:, None])
                & (x_grid <= (left + hi * span)[:, None]))
        core = ndimage.binary_erosion(core, iterations=1)
        components, number = ndimage.label(core, structure=connectivity)
        if not number:
            return None
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        core = components == int(np.argmax(sizes))
        cy = np.nonzero(core)[0]
        if (int(core.sum()) < max(20, 0.05 * parent_area / count)
                or not len(cy) or int(np.ptp(cy)) + 1 < 0.55 * parent_span):
            return None
        cores.append(core)
    return cores


def _across_axis(arrows):
    """并排的几条虫"从左往右"是哪个方向：与共同头向垂直的单位向量。

    共同头向不稳定（箭头太少或互相垂直）时返回 None，调用方据此放弃排序。
    """
    stable = _common_arrow_direction(arrows)
    if stable is None:
        return None
    return np.asarray((-stable[1], stable[0]), dtype=np.float64)


def _ordered_arrow_assignment(partition: np.ndarray, arrows: list[HeadDirection]):
    """把箭头和子块各自沿虫体横排方向排序后一一配对。

    虫头一端几条虫并排挨得最近（实测两条虫头之间只隔 40 像素），箭头又短，
    杆子很容易压在邻虫那一侧。这时投票认不出哪支箭头属于哪一块，但"实验员
    一只虫画一支箭头"仍然成立：箭头的左右次序与子块的左右次序一致。返回
    每支箭头对应的子块编号，块数与箭头数不等或出现空块时返回 None。
    """
    across = _across_axis(arrows)
    if across is None:
        return None
    rows, columns = np.indices(partition.shape)
    centres = []
    for index in range(1, len(arrows) + 1):
        region = partition == index
        if not region.any():
            return None
        centres.append(float(columns[region].mean() * across[0]
                             + rows[region].mean() * across[1]))
    order = np.argsort([arrow.tail_x * across[0] + arrow.tail_y * across[1]
                        for arrow in arrows])
    children = [index + 1 for index in np.argsort(centres)]
    assigned = [0] * len(arrows)
    for rank, index in enumerate(order):
        assigned[int(index)] = children[rank]
    return assigned


def _assign_spare_arrows_by_order(instances: np.ndarray,
                                  assigned: dict[int, tuple],
                                  spare: list[HeadDirection],
                                  arrows: list[HeadDirection]) -> list[HeadDirection]:
    """投票落空的箭头，按横排次序补配给没人认领的虫；返回仍没配上的箭头。

    箭头画在虫头一端时，整根都可能压在邻虫身上（0913-0-2 实测第 3 支从头到尾
    都落在第 2 条的像素上），投票认不出来。但"从左往右一只虫画一支箭头"仍然
    成立：落空的箭头与没人认领的虫按横排次序一一配对。多一支少一支都不猜。
    """
    missing = [label for label in range(1, int(instances.max()) + 1)
               if label not in assigned]
    if not missing or len(missing) != len(spare) or not spare:
        return spare
    across = _across_axis(arrows)
    if across is None:
        return spare
    rows, columns = np.indices(instances.shape)
    regions = {}
    centres = []
    for label in missing:
        regions[label] = np.nonzero(instances == label)
        yy, xx = regions[label]
        centres.append(float(xx.mean() * across[0] + yy.mean() * across[1]))
    order = np.argsort([arrow.tail_x * across[0] + arrow.tail_y * across[1]
                        for arrow in spare])
    ranked = [missing[int(position)] for position in np.argsort(centres)]
    for rank, position in enumerate(order):
        arrow = spare[int(position)]
        label = ranked[rank]
        yy, xx = regions[label]
        distance = float(np.sqrt(np.min((xx - arrow.tail_x) ** 2
                                        + (yy - arrow.tail_y) ** 2)))
        assigned[label] = ("ordered", distance, arrow)
    return []


def _partition_safety(partition: np.ndarray | None, mask: np.ndarray,
                      arrows: list[HeadDirection], parent_height: int):
    """确认候选分区完整、连通，并且每支箭头各自对应一块子虫。"""
    if partition is None:
        return None
    count = len(arrows)
    arrow_labels = [_labels_along_arrow(partition, arrow)[0] for arrow in arrows]
    if len(set(arrow_labels)) != count or any(label <= 0 for label in arrow_labels):
        # 投票把两支箭头算进了同一块。箭头画在虫身中段时投票是准的，压在虫头
        # 的缝上时不准（0913-0-2 实测三支箭头投出 [1,1,3]）。退到按次序配对，
        # 并只在多数箭头仍与配对结果吻合时才接受 —— 全都对不上说明这份分区
        # 与箭头无关。
        assignment = _ordered_arrow_assignment(partition, arrows)
        if assignment is None:
            return None
        agreeing = sum(1 for label, target in zip(arrow_labels, assignment)
                       if label == target)
        if agreeing * 2 <= count:
            return None
    child_areas = np.asarray([(partition == index).sum()
                              for index in range(1, count + 1)], dtype=np.float64)
    child_spans = []
    connectivity = np.ones((3, 3), dtype=np.uint8)
    for index in range(1, count + 1):
        region = partition == index
        child_y = np.nonzero(region)[0]
        child_spans.append(
            (int(np.ptp(child_y)) + 1) / parent_height if len(child_y) else 0.0)
        components, number = ndimage.label(region, structure=connectivity)
        if not number:
            return None
        sizes = np.bincount(components.ravel())[1:]
        if sizes.max() < 0.998 * region.sum():
            return None
    if not np.array_equal(partition > 0, mask):
        return None
    area_fractions = child_areas / max(float(mask.sum()), 1.0)
    if area_fractions.min() < max(0.10, 0.34 / count) or min(child_spans) < 0.52:
        return None
    return area_fractions, child_spans


def _repair_axis_partition(mask: np.ndarray, partition: np.ndarray,
                           arrows: list[HeadDirection]):
    """保留各箭头所在主连通域，并把零散像素归还最近的有效子虫。"""
    retained = partition.copy()
    connectivity = np.ones((3, 3), dtype=np.uint8)
    for _pass in range(4):
        cleaned = np.zeros(partition.shape, dtype=np.uint16)
        for index, arrow in enumerate(arrows, 1):
            region = retained == index
            components, number = ndimage.label(region, structure=connectivity)
            if not number:
                return None
            tx = int(np.clip(round(arrow.tail_x), 0, mask.shape[1] - 1))
            ty = int(np.clip(round(arrow.tail_y), 0, mask.shape[0] - 1))
            component = int(components[ty, tx])
            if component == 0:
                sizes = np.bincount(components.ravel())
                sizes[0] = 0
                component = int(np.argmax(sizes))
            cleaned[components == component] = index
        missing = mask & (cleaned == 0)
        if missing.any():
            if not (cleaned > 0).any():
                return None
            nearest = ndimage.distance_transform_edt(
                cleaned == 0, return_distances=False, return_indices=True)
            cleaned[missing] = cleaned[tuple(nearest)][missing]
        if np.array_equal(cleaned, retained):
            retained = cleaned
            break
        retained = cleaned
    for index in range(1, len(arrows) + 1):
        region = retained == index
        components, number = ndimage.label(region, structure=connectivity)
        if not number:
            return None
        sizes = np.bincount(components.ravel())[1:]
        if sizes.max() < 0.998 * region.sum():
            return None
    return retained


def _coarse_partition_by_head_directions(original: np.ndarray, arrows,
                                         image, max_instances):
    """第一阶段：只使用头向箭头粗分粘连实例，不读取人工分界线。"""
    groups = _arrow_groups(original, arrows)
    if not any(len(values) > 1 for values in groups.values()):
        return original.copy(), []

    from low_clarity_splitter import _ordered_partition

    output = original.copy()
    reports = []
    next_label = int(original.max()) + 1
    current_count = len([value for value in np.unique(original) if value])
    for parent_label in sorted(groups):
        group = groups[parent_label]
        if len(group) < 2:
            continue
        extra = len(group) - 1
        if max_instances and current_count + extra > max_instances:
            continue
        yy, xx = np.nonzero(original == parent_label)
        if not len(yy):
            continue
        y0, y1 = max(0, int(yy.min()) - 3), min(original.shape[0], int(yy.max()) + 4)
        x0, x1 = max(0, int(xx.min()) - 3), min(original.shape[1], int(xx.max()) + 4)
        mask = original[y0:y1, x0:x1] == parent_label
        local_arrows = [HeadDirection(
            arrow.tail_x - x0, arrow.tail_y - y0,
            arrow.head_x - x0, arrow.head_y - y0) for arrow in group]

        separations = [math.hypot(first.tail_x - second.tail_x,
                                  first.tail_y - second.tail_y)
                       for index, first in enumerate(local_arrows)
                       for second in local_arrows[index + 1:]]
        min_separation = min(separations)
        parent_height = max(int(np.ptp(yy)) + 1, 1)
        estimated_child_width = len(yy) / parent_height / len(group)
        if min_separation < max(4.0, 0.28 * estimated_child_width):
            continue

        if image is None:
            gray = np.ones(mask.shape, dtype=np.float32) * 0.5
        else:
            gray = image[y0:y1, x0:x1].astype(np.float32)
            if gray.max() > 1.0:
                gray /= 255.0
            gray = np.clip(gray, 0.0, 1.0)
        probability = np.zeros((3, *mask.shape), dtype=np.float32)
        probability[1] = np.where(mask, 0.15 + 0.85 * gray, 0.0)
        probability[0] = 1.0 - probability[1]
        partition = None
        safety = None
        core_candidates = []
        axis_cores = _axis_partition_cores(mask, local_arrows)
        if axis_cores is not None:
            core_candidates.append(axis_cores)
        fraction_cores = _fraction_partition_cores(mask, len(local_arrows))
        if fraction_cores is not None:
            core_candidates.append(fraction_cores)
        for cores in core_candidates:
            candidate = _ordered_partition(mask, cores, probability)
            checked = _partition_safety(candidate, mask, local_arrows, parent_height)
            if checked is not None:
                partition, safety = candidate, checked
                break
        if partition is None:
            candidate = _axis_owner_partition(mask, local_arrows)
            if candidate is not None:
                candidate = _repair_axis_partition(mask, candidate, local_arrows)
            checked = _partition_safety(candidate, mask, local_arrows, parent_height)
            if checked is not None:
                partition, safety = candidate, checked
        if partition is None or safety is None:
            continue
        area_fractions, child_spans = safety
        child_ids = [parent_label] + list(range(next_label, next_label + extra))
        next_label += extra
        local_output = output[y0:y1, x0:x1]
        for index, child_label in enumerate(child_ids, 1):
            local_output[partition == index] = child_label
        current_count += extra
        reports.append(ManualSplitReport(
            original_label=parent_label,
            arrow_count=len(group),
            status="REVIEW_MANUAL_HEAD_SPLIT",
            min_child_area_fraction=float(area_fractions.min()),
            min_child_span_fraction=float(min(child_spans)),
            arrow_separation_px=float(min_separation),
            method="manual_head_coarse",
            resulting_labels="|".join(str(label) for label in child_ids)))
    return output, reports


def _draw_boundary_barrier(shape, guides, width: int, extend: bool = False):
    canvas = Image.new("1", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(canvas)
    diagonal = math.hypot(shape[1], shape[0]) + 4.0
    for guide in guides:
        points = np.asarray(guide.points, dtype=np.float64)
        drawing_points = [tuple(point) for point in points]
        if extend and len(points) >= 2:
            def extended(point, neighbour):
                direction = point - neighbour
                length = float(np.linalg.norm(direction))
                return point if length < 1.0 else point + direction * (diagonal / length)
            drawing_points.insert(0, tuple(extended(points[0], points[1])))
            drawing_points.append(tuple(extended(points[-1], points[-2])))
        draw.line(drawing_points, fill=1, width=width, joint="curve")
    return np.asarray(canvas, dtype=bool)


def _labels_near_boundary(instances: np.ndarray, guide: BoundaryGuide):
    radius = max(6, int(round(min(instances.shape) * 0.006)))
    stroke = _draw_boundary_barrier(instances.shape, [guide], 3)
    neighbourhood = ndimage.binary_dilation(stroke, iterations=radius)
    values, counts = np.unique(instances[neighbourhood & (instances > 0)], return_counts=True)
    if not len(values):
        return set()
    threshold = max(3, int(round(0.01 * counts.sum())))
    selected = {int(value) for value, count in zip(values, counts) if count >= threshold}
    if not selected:
        selected = {int(values[int(np.argmax(counts))])}
    return selected


def _merge_boundary_groups(instances: np.ndarray, guides):
    pending = [(set(_labels_near_boundary(instances, guide)), [guide]) for guide in guides]
    pending = [(labels, items) for labels, items in pending if labels]
    merged = []
    while pending:
        labels, items = pending.pop(0)
        changed = True
        while changed:
            changed = False
            remaining = []
            for other_labels, other_items in pending:
                if labels & other_labels:
                    labels |= other_labels
                    items.extend(other_items)
                    changed = True
                else:
                    remaining.append((other_labels, other_items))
            pending = remaining
        merged.append((labels, items))
    return merged


def _common_arrow_direction(arrows):
    directions = []
    for arrow in arrows:
        direction = np.asarray((arrow.head_x - arrow.tail_x,
                                arrow.head_y - arrow.tail_y), dtype=np.float64)
        length = float(np.linalg.norm(direction))
        if length < 1.0:
            continue
        direction /= length
        if directions and float(np.dot(direction, directions[0])) < 0:
            direction = -direction
        directions.append(direction)
    if not directions:
        return None
    common = np.mean(directions, axis=0)
    length = float(np.linalg.norm(common))
    return common / length if length >= 0.25 else None


def _boundary_stable_direction(guides):
    """For U-shaped tail guides, return the direction from the tip back to the body."""
    directions = []
    for guide in guides:
        points = np.asarray(guide.points, dtype=np.float64)
        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        path_length = float(segment_lengths.sum())
        endpoint_distance = float(np.linalg.norm(points[-1] - points[0]))
        if path_length < 1.8 * max(endpoint_distance, 1.0):
            continue
        mouth = 0.5 * (points[0] + points[-1])
        tip = points[int(np.argmax(np.linalg.norm(points - mouth, axis=1)))]
        direction = mouth - tip
        length = float(np.linalg.norm(direction))
        if length < 3.0:
            continue
        direction /= length
        if directions and float(np.dot(direction, directions[0])) < 0:
            direction = -direction
        directions.append(direction)
    if not directions:
        return None
    common = np.mean(directions, axis=0)
    length = float(np.linalg.norm(common))
    return common / length if length >= 0.25 else None


def _partition_metrics(partition: np.ndarray, mask: np.ndarray, count: int):
    if partition is None or not np.array_equal(partition > 0, mask):
        return None
    parent_y, parent_x = np.nonzero(mask)
    parent_span_x = max(int(np.ptp(parent_x)) + 1, 1)
    parent_span_y = max(int(np.ptp(parent_y)) + 1, 1)
    areas, spans = [], []
    connectivity = np.ones((3, 3), dtype=np.uint8)
    for index in range(1, count + 1):
        region = partition == index
        area = int(region.sum())
        if not area:
            return None
        components, number = ndimage.label(region, structure=connectivity)
        sizes = np.bincount(components.ravel())[1:] if number else np.asarray([])
        if not len(sizes) or sizes.max() < 0.995 * area:
            return None
        yy, xx = np.nonzero(region)
        spans.append(max((int(np.ptp(xx)) + 1) / parent_span_x,
                         (int(np.ptp(yy)) + 1) / parent_span_y))
        areas.append(area)
    fractions = np.asarray(areas, dtype=np.float64) / max(float(mask.sum()), 1.0)
    if fractions.min() < max(0.025, 0.20 / count) or min(spans) < 0.35:
        return None
    return fractions, spans


def _core_owner_map(cores, shape):
    """Owner index of the nearest core, for every pixel.

    The cores are the seeds the growth starts from, so they say which worm a
    pixel belongs to without depending on the partition being corrected.
    """
    best = np.full(shape, np.inf)
    owner = np.zeros(shape, dtype=np.uint8)
    for index, core in enumerate(cores, 1):
        if not core.any():
            continue
        distance = ndimage.distance_transform_edt(~core)
        closer = distance < best
        best = np.where(closer, distance, best)
        owner[closer] = index
    return owner


def _guide_sample_points(guide):
    """Points spaced along a guide, and the local normal at each of them."""
    points = np.asarray(guide.points, dtype=np.float64)
    bases, normals = [], []
    for start, end in zip(points, points[1:]):
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length < 1.0:
            continue
        normal = np.asarray((-direction[1], direction[0])) / length
        for fraction in np.linspace(0.0, 1.0, max(2, int(length // 2))):
            bases.append(start + direction * fraction)
            normals.append(normal)
    if not bases:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.asarray(bases), np.asarray(normals)


def _guide_side_targets(bases, normals, radius, owner_map, mask):
    """Owner the nearest core gives to each side of every sample point.

    The outermost probe that lands on a worm wins: the further a probe sits from
    the line, the likelier it is to be beyond a seam that drifted outwards, which
    is exactly the territory that side should own.
    """
    targets = []
    for sign in (1, -1):
        probes = []
        for scale in SNAP_PROBE_OFFSETS:
            offset = bases + normals * (sign * scale * radius)
            rows = np.clip(np.rint(offset[:, 1]).astype(int), 0, mask.shape[0] - 1)
            columns = np.clip(np.rint(offset[:, 0]).astype(int), 0, mask.shape[1] - 1)
            probes.append(np.where(mask[rows, columns], owner_map[rows, columns], 0))
        targets.append(np.where(probes[-1] > 0, probes[-1], probes[0]).astype(np.uint8))
    return targets[0], targets[1]


def _nearest_sample(bases, shape):
    """Index of the nearest sample point, for every pixel.

    Points beyond the ends of the line keep the end's normal, which extends the
    line's own half-planes instead of wrapping a side around its tip: a side read
    from "whichever probe pixel is closer" flips over near the ends and cuts the
    territories into fragments there.
    """
    index_map = np.zeros(shape, dtype=np.int32)
    rows = np.clip(np.rint(bases[:, 1]).astype(int), 0, shape[0] - 1)
    columns = np.clip(np.rint(bases[:, 0]).astype(int), 0, shape[1] - 1)
    index_map[rows, columns] = np.arange(1, len(bases) + 1)
    _, index = ndimage.distance_transform_edt(index_map == 0, return_indices=True)
    return index_map[index[0], index[1]] - 1


def _owners_survive(before: np.ndarray, after: np.ndarray) -> bool:
    """Every owner stays non-empty and in one piece after the correction."""
    connectivity = np.ones((3, 3), dtype=np.uint8)
    for value in (int(item) for item in np.unique(before) if item):
        region = after == value
        area = int(region.sum())
        if not area:
            return False
        components, number = ndimage.label(region, structure=connectivity)
        sizes = np.bincount(components.ravel())[1:] if number else np.asarray([])
        if not len(sizes) or sizes.max() < 0.995 * area:
            return False
    return True


def _snap_seam_to_guides(owners: np.ndarray, cores, mask: np.ndarray, guides, width: int):
    """Push the seam back onto every drawn boundary line.

    The barrier the growth crosses is only expensive, not impassable, so a front
    can still wrap around the end of a line and leave the seam up to ~18 px away
    from it (measured on the annotated 0913-0-2 image). Pixels are therefore
    re-owned afterwards, one guide at a time, inside a band around that guide.

    Which owner a side of the line should end up with is read from the nearest
    core, not from the partition being corrected: the cores are the growth seeds,
    so they still say which worm a pixel belongs to where the partition is wrong.
    Probes sit in the outer part of the band so that a seam which drifted towards
    one side lands inside them and the territory behind it is sampled.

    Only pixels currently owned by the *other* side's owner are exchanged, so no
    third worm is touched, and each exchange must pass _owners_survive or it is
    dropped and the rest of the guides are applied as usual.
    """
    if not (owners > 0).any() or not guides or not cores:
        return owners
    snapped = owners.copy()
    band_radius = max(6, int(round(min(mask.shape) * SNAP_BAND_FRACTION)))
    core_owner = _core_owner_map(cores, mask.shape)
    rows_grid, columns_grid = np.indices(mask.shape)
    for guide in guides:
        stroke = _draw_boundary_barrier(mask.shape, [guide], width)
        band = ndimage.binary_dilation(stroke, iterations=band_radius) & mask
        if not band.any():
            continue
        bases, normals = _guide_sample_points(guide)
        if not len(bases):
            continue
        nearest = _nearest_sample(bases, mask.shape)
        left_target, right_target = _guide_side_targets(
            bases, normals, band_radius, core_owner, mask)
        left = left_target[nearest]
        right = right_target[nearest]
        side = ((columns_grid - bases[nearest, 0]) * normals[nearest, 0]
                + (rows_grid - bases[nearest, 1]) * normals[nearest, 1]) >= 0
        usable = band & (left > 0) & (right > 0) & (left != right)
        candidate = snapped.copy()
        flip_left = usable & side & (snapped == right)
        flip_right = usable & ~side & (snapped == left)
        candidate[flip_left] = left[flip_left]
        candidate[flip_right] = right[flip_right]
        if not (candidate != snapped).any():
            continue
        if _owners_survive(snapped, candidate):
            snapped = candidate
    return snapped


def _refine_multiple_labels_with_boundaries(instances: np.ndarray, labels, guides, arrows):
    """Repartition an existing group without changing its foreground or label count."""
    from low_clarity_splitter import _grow_cores

    ordered_labels = sorted(labels, key=lambda label: float(np.nonzero(instances == label)[1].mean()))
    yy, xx = np.nonzero(np.isin(instances, ordered_labels))
    if not len(yy):
        return None
    margin = max(8, int(round(min(instances.shape) * 0.008)))
    y0, y1 = max(0, int(yy.min()) - margin), min(instances.shape[0], int(yy.max()) + margin + 1)
    x0, x1 = max(0, int(xx.min()) - margin), min(instances.shape[1], int(xx.max()) + margin + 1)
    local = instances[y0:y1, x0:x1]
    mask = np.isin(local, ordered_labels)
    local_guides = [BoundaryGuide(tuple((x - x0, y - y0) for x, y in guide.points))
                    for guide in guides]
    width = max(3, int(round(min(instances.shape) * 0.003)))
    barrier = _draw_boundary_barrier(mask.shape, local_guides, width)

    arrow_groups = _arrow_groups(instances, arrows)
    group_arrows = [arrow for label in ordered_labels for arrow in arrow_groups.get(label, [])]
    stable = _common_arrow_direction(group_arrows)
    if stable is None:
        stable = _boundary_stable_direction(local_guides)
    rows, columns = np.indices(mask.shape, dtype=np.float32)
    cores = []
    for label in ordered_labels:
        region = local == label
        core = None
        if stable is not None:
            projection = columns * stable[0] + rows * stable[1]
            guide_projection = [x * stable[0] + y * stable[1]
                                for guide in local_guides for x, y in guide.points]
            threshold = max(guide_projection) + max(4.0, 2.0 * width)
            candidate = region & (projection >= threshold)
            if candidate.sum() >= max(20, 0.05 * region.sum()):
                core = candidate
        if core is None:
            distance = ndimage.distance_transform_edt(~barrier)
            core = region & (distance >= max(3.0, 2.0 * width))
        core = ndimage.binary_erosion(core, iterations=2)
        if core.sum() < max(12, 0.02 * region.sum()):
            return None
        cores.append(core)

    probability = np.zeros((3, *mask.shape), dtype=np.float32)
    probability[2, barrier] = 100.0
    owners = _snap_seam_to_guides(_grow_cores(mask, cores, probability),
                                  cores, mask, local_guides, width)
    metrics = _partition_metrics(owners, mask, len(ordered_labels))
    if metrics is None:
        return None
    result = instances.copy()
    local_result = result[y0:y1, x0:x1]
    for owner, label in enumerate(ordered_labels, 1):
        local_result[owners == owner] = label
    return result, metrics, ordered_labels, len(group_arrows)


def _split_one_label_with_boundaries(instances: np.ndarray, label: int, guides,
                                     max_instances):
    """Let boundary guides split one merged instance even when no arrows exist."""
    from low_clarity_splitter import _grow_cores

    current_count = len([value for value in np.unique(instances) if value])
    wanted = len(guides) + 1
    if max_instances:
        wanted = min(wanted, max_instances - current_count + 1)
    if wanted < 2:
        return None
    yy, xx = np.nonzero(instances == label)
    if not len(yy):
        return None
    margin = 4
    y0, y1 = max(0, int(yy.min()) - margin), min(instances.shape[0], int(yy.max()) + margin + 1)
    x0, x1 = max(0, int(xx.min()) - margin), min(instances.shape[1], int(xx.max()) + margin + 1)
    mask = instances[y0:y1, x0:x1] == label
    local_guides = [BoundaryGuide(tuple((x - x0, y - y0) for x, y in guide.points))
                    for guide in guides]
    width = max(3, int(round(min(instances.shape) * 0.003)))
    barrier = _draw_boundary_barrier(mask.shape, local_guides, width, extend=True)
    components, number = ndimage.label(mask & ~barrier, structure=np.ones((3, 3), dtype=np.uint8))
    if number < wanted:
        return None
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    component_ids = np.argsort(sizes)[::-1][:wanted]
    if sizes[component_ids[-1]] < max(20, 0.04 * mask.sum()):
        return None
    cores = [components == component for component in component_ids]
    probability = np.zeros((3, *mask.shape), dtype=np.float32)
    probability[2, barrier] = 100.0
    owners = _snap_seam_to_guides(_grow_cores(mask, cores, probability),
                                  cores, mask, local_guides, width)
    metrics = _partition_metrics(owners, mask, wanted)
    if metrics is None:
        return None
    child_ids = [label]
    next_label = int(instances.max()) + 1
    child_ids.extend(range(next_label, next_label + wanted - 1))
    result = instances.copy()
    local_result = result[y0:y1, x0:x1]
    for owner, child_label in enumerate(child_ids, 1):
        local_result[owners == owner] = child_label
    return result, metrics, child_ids


def _refine_with_boundary_guides(instances: np.ndarray, guides, arrows, max_instances):
    output = instances.copy()
    reports = []
    for labels, group_guides in _merge_boundary_groups(output, guides):
        arrow_groups_before = _arrow_groups(output, arrows)
        relevant_arrows = [arrow for label in labels
                           for arrow in arrow_groups_before.get(label, [])]
        if len(labels) == 1:
            label = next(iter(labels))
            refined = _split_one_label_with_boundaries(
                output, label, group_guides, max_instances)
            if refined is None:
                continue
            candidate, metrics, resulting = refined
            arrow_count = len(relevant_arrows)
            status = "REVIEW_MANUAL_BOUNDARY_SPLIT"
        else:
            refined = _refine_multiple_labels_with_boundaries(
                output, labels, group_guides, arrows)
            if refined is None:
                continue
            candidate, metrics, resulting, arrow_count = refined
            label = min(labels)
            status = "REVIEW_MANUAL_BOUNDARY_REFINE"
        # A guide the current partition already honours reproduces it exactly;
        # reporting that as a manual refinement would send the reviewer looking
        # for a change that never happened.
        if np.array_equal(candidate, output):
            continue
        output = candidate
        fractions, spans = metrics
        separations = [math.hypot(first.tail_x - second.tail_x,
                                  first.tail_y - second.tail_y)
                       for index, first in enumerate(relevant_arrows)
                       for second in relevant_arrows[index + 1:]]
        reports.append(ManualSplitReport(
            original_label=label,
            arrow_count=arrow_count,
            status=status,
            min_child_area_fraction=float(fractions.min()),
            min_child_span_fraction=float(min(spans)),
            arrow_separation_px=float(min(separations)) if separations else 0.0,
            method="manual_boundary_refine",
            resulting_labels="|".join(str(value) for value in resulting)))
    return output, reports


def apply_manual_annotation_partitions(instances: np.ndarray, arrows=None, image=None,
                                       max_instances: int | None = None,
                                       boundary_guides=None):
    """Apply independent manual algorithms in a deterministic order.

    Head arrows first produce a coarse partition. Boundary guides then refine
    that result and may also operate without arrows. Both stages preserve the
    original foreground union; a rejected safety check leaves the input exact.
    """
    original = np.asarray(instances)
    if original.ndim != 2:
        raise ValueError("instances must be a two-dimensional label map")
    if image is not None:
        image = np.asarray(image)
        if image.shape != original.shape or not np.isfinite(image).all():
            raise ValueError("image dimensions/values do not match instances")
    valid_arrows = [arrow for arrow in (
        value if isinstance(value, HeadDirection) else _valid_arrow(value)
        for value in (arrows or [])) if arrow]
    valid_guides = [guide for guide in (
        value if isinstance(value, BoundaryGuide) else _valid_boundary(value)
        for value in (boundary_guides or [])) if guide]

    output, head_reports = _coarse_partition_by_head_directions(
        original, valid_arrows, image, max_instances)
    output, boundary_reports = _refine_with_boundary_guides(
        output, valid_guides, valid_arrows, max_instances)
    reports = head_reports + boundary_reports
    if not np.array_equal(output > 0, original > 0):
        raise RuntimeError("manual annotation changed the foreground union")
    if not reports:
        return original.copy(), []

    labels = [int(value) for value in np.unique(output) if value]
    ordered = sorted(labels, key=lambda label: float(np.nonzero(output == label)[1].mean()))
    mapping = {old: new for new, old in enumerate(ordered, 1)}
    relabeled = np.zeros(output.shape, dtype=np.uint16)
    for old, new in mapping.items():
        relabeled[output == old] = new
    for report in reports:
        values = [int(value) for value in report.resulting_labels.split("|") if value]
        report.resulting_labels = "|".join(str(mapping.get(value, value)) for value in values)
    return relabeled, reports


def split_instances_by_head_directions(instances: np.ndarray, arrows, image=None,
                                       max_instances: int | None = None,
                                       boundary_guides=None):
    """Backward-compatible name for the two-stage manual partition pipeline."""
    return apply_manual_annotation_partitions(
        instances, arrows=arrows, image=image, max_instances=max_instances,
        boundary_guides=boundary_guides)


def draw_head_arrows(image: Image.Image, matches: list[HeadMatch]) -> Image.Image:
    """在 QC 图副本上画红色人工箭头和虫号。"""
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for match in matches:
        if match.status != "MATCHED":
            continue
        start = (float(match.arrow_tail_x), float(match.arrow_tail_y))
        end = (float(match.arrow_head_x), float(match.arrow_head_y))
        length = max(float(match.arrow_length_px), 1.0)
        ux = (end[0] - start[0]) / length
        uy = (end[1] - start[1]) / length
        head_size = max(8.0, min(22.0, length * 0.24))
        base_x, base_y = end[0] - ux * head_size, end[1] - uy * head_size
        perp_x, perp_y = -uy, ux
        left = (base_x + perp_x * head_size * 0.48, base_y + perp_y * head_size * 0.48)
        right = (base_x - perp_x * head_size * 0.48, base_y - perp_y * head_size * 0.48)
        width = max(2, int(round(min(image.size) / 450.0)))
        draw.line((start, end), fill=(255, 24, 24), width=width)
        draw.polygon((end, left, right), fill=(255, 24, 24))
        draw.text((end[0] + 5, end[1] + 3), str(match.worm_label), fill=(255, 255, 255),
                  stroke_width=2, stroke_fill=(130, 0, 0))
    return canvas


def draw_boundary_guides(image: Image.Image, guides: list[BoundaryGuide]) -> Image.Image:
    """在 QC 图上叠加青色人工分界曲线。"""
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    width = max(2, int(round(min(image.size) / 500.0)))
    for index, guide in enumerate(guides, 1):
        if len(guide.points) < 2:
            continue
        draw.line(guide.points, fill=(0, 255, 255), width=width, joint="curve")
        x, y = guide.points[0]
        radius = max(3, width + 1)
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=(0, 255, 255))
        draw.text((x + radius + 2, y + radius + 2), "B%d" % index,
                  fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 90, 100))
    return canvas


def draw_exclusion_regions(image: Image.Image, regions: list[ExclusionRegion]) -> Image.Image:
    """在 QC 图上以半透明紫红色叠加人工排除区域。"""
    canvas = image.convert("RGBA")
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    width = max(2, int(round(min(image.size) / 420.0)))
    for index, region in enumerate(regions, 1):
        if len(region.points) < 3:
            continue
        draw.polygon(region.points, fill=(255, 40, 190, 48),
                     outline=(255, 40, 190, 255), width=width)
        x, y = region.points[0]
        draw.text((x + 5, y + 5), "X%d" % index,
                  fill=(255, 255, 255, 255), stroke_width=2,
                  stroke_fill=(120, 0, 75, 255))
    return Image.alpha_composite(canvas, layer).convert("RGB")


HEAD_REPORT_FIELDS = list(HeadMatch.__dataclass_fields__)
MANUAL_SPLIT_REPORT_FIELDS = list(ManualSplitReport.__dataclass_fields__)
