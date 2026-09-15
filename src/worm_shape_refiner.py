#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立的线虫轮廓质量检测与保守形状修复。

本模块不参与 U-Net 或头尾网络推理，也不修改模型权重。
输入是模型已生成的实例标签图，输出是经过安全门控的可选修复结果和质量报告。
"""

from __future__ import annotations

import csv
import heapq
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage


@dataclass
class ShapeReport:
    label: int
    status: str
    repaired: bool
    needs_review: bool
    reasons: str
    quality_before: float
    quality_after: float
    contour_residual: float
    curvature_p95: float
    width_roughness: float
    centerline_roughness: float
    tip_bluntness: float
    solidity: float
    area_before: int
    area_after: int
    area_change_fraction: float
    centroid_shift_fraction: float
    candidate_kind: str
    probability_evidence: float
    candidate_iou: float


@dataclass
class ShapeMetrics:
    quality: float
    contour_residual: float
    curvature_p95: float
    width_roughness: float
    centerline_roughness: float
    tip_bluntness: float
    solidity: float
    major_length: float
    area: int
    centroid_x: float
    centroid_y: float


# 无量纲阈值。使用人工 ROI 校准时只需要替换此处数值，不影响主模型。
QUALITY_LIMITS = {
    "contour_residual": 0.055,
    "curvature_p95": 0.34,
    "width_roughness": 0.22,
    "centerline_roughness": 0.20,
    "tip_bluntness": 0.42,
}

# Calibrated conservatively on manual-mask candidate audits.  The retained
# candidate at -0.116 improved Dice/boundary/area metrics; a candidate at -0.131
# degraded boundary metrics, so the production gate is placed between them.
MIN_PROBABILITY_EVIDENCE = -0.12

# Local contour cleanup is deliberately allowed to disagree slightly with the
# coarse network at the one- or two-pixel boundary band.  Its much tighter
# geometry gates below prevent this relaxed probability threshold from
# reshaping the worm body.
MIN_LOCAL_PROBABILITY_EVIDENCE = -0.55
MIN_LOCAL_ROUGHNESS_GAIN = 0.06


def _main_component(mask: np.ndarray) -> np.ndarray:
    labels, number = ndimage.label(mask)
    if number <= 1:
        return ndimage.binary_fill_holes(mask)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return ndimage.binary_fill_holes(labels == int(np.argmax(sizes)))


def _contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    points = max(contours, key=cv2.contourArea)[:, 0, :].astype(np.float64)
    return points if len(points) >= 8 else None


def _resample_closed(points: np.ndarray, samples: int = 256) -> np.ndarray | None:
    closed = np.vstack((points, points[0]))
    delta = np.diff(closed, axis=0)
    segment = np.sqrt((delta * delta).sum(axis=1))
    perimeter = float(segment.sum())
    if perimeter <= 1e-6:
        return None
    cumulative = np.concatenate(([0.0], np.cumsum(segment)))
    distances = np.linspace(0.0, perimeter, samples, endpoint=False)
    index = np.searchsorted(cumulative, distances, side="right") - 1
    index = np.clip(index, 0, len(segment) - 1)
    fraction = (distances - cumulative[index]) / np.maximum(segment[index], 1e-9)
    return closed[index] + fraction[:, None] * delta[index]


def _profile(mask: np.ndarray) -> dict | None:
    yy, xx = np.nonzero(mask)
    if len(xx) < 20:
        return None
    xy = np.column_stack((xx, yy)).astype(np.float64)
    origin = xy.mean(axis=0)
    covariance = np.cov(xy, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    major = vectors[:, int(np.argmax(values))]
    # 统一方向，使报告和头尾外推可重复。
    if major[1] < 0 or (abs(major[1]) < 1e-8 and major[0] < 0):
        major = -major
    normal = np.asarray((-major[1], major[0]), dtype=np.float64)
    relative = xy - origin
    along = relative @ major
    across = relative @ normal
    lower = float(np.floor(along.min()))
    upper = float(np.ceil(along.max()))
    bins = max(8, int(round(upper - lower)) + 1)
    bin_index = np.clip(np.rint(along - lower).astype(int), 0, bins - 1)
    minimum = np.full(bins, np.inf)
    maximum = np.full(bins, -np.inf)
    np.minimum.at(minimum, bin_index, across)
    np.maximum.at(maximum, bin_index, across)
    valid = np.isfinite(minimum) & np.isfinite(maximum)
    if valid.sum() < max(8, int(0.55 * bins)):
        return None
    positions = np.arange(bins, dtype=np.float64)
    valid_positions = positions[valid]
    center_values = (minimum[valid] + maximum[valid]) * 0.5
    width_values = (maximum[valid] - minimum[valid] + 1.0) * 0.5
    center = np.interp(positions, valid_positions, center_values)
    width = np.interp(positions, valid_positions, width_values)
    sigma = min(12.0, max(1.25, bins * 0.012))
    smooth_center = ndimage.gaussian_filter1d(center, sigma=sigma, mode="nearest")
    smooth_width = ndimage.gaussian_filter1d(width, sigma=sigma, mode="nearest")
    body_slice = slice(max(1, int(0.25 * bins)), max(2, int(0.75 * bins)))
    body_width = max(float(np.median(smooth_width[body_slice])), 1.0)
    tip_bins = max(2, int(round(0.025 * bins)))
    start_bluntness = float(np.mean(width[:tip_bins]) / body_width)
    end_bluntness = float(np.mean(width[-tip_bins:]) / body_width)
    return {
        "origin": origin,
        "major": major,
        "normal": normal,
        "lower": lower,
        "center": center,
        "width": width,
        "smooth_center": smooth_center,
        "smooth_width": smooth_width,
        "body_width": body_width,
        "start_bluntness": start_bluntness,
        "end_bluntness": end_bluntness,
        "major_length": max(upper - lower, 1.0),
    }


def measure_shape(mask: np.ndarray) -> ShapeMetrics:
    mask = _main_component(mask)
    area = int(mask.sum())
    yy, xx = np.nonzero(mask)
    centroid_x = float(xx.mean()) if len(xx) else 0.0
    centroid_y = float(yy.mean()) if len(yy) else 0.0
    profile = _profile(mask)
    points = _contour(mask)

    contour_residual = curvature_p95 = 0.0
    half_width = 1.0
    if profile is not None:
        half_width = max(float(profile["body_width"]), 1.0)
    if points is not None:
        sampled = _resample_closed(points)
        if sampled is not None:
            smooth = np.column_stack((
                ndimage.gaussian_filter1d(sampled[:, 0], sigma=2.2, mode="wrap"),
                ndimage.gaussian_filter1d(sampled[:, 1], sigma=2.2, mode="wrap"),
            ))
            residual = np.sqrt(((sampled - smooth) ** 2).sum(axis=1))
            contour_residual = float(np.percentile(residual, 90) / half_width)
            incoming = smooth - np.roll(smooth, 1, axis=0)
            outgoing = np.roll(smooth, -1, axis=0) - smooth
            cross = incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
            dot = (incoming * outgoing).sum(axis=1)
            angles = np.abs(np.arctan2(cross, dot))
            curvature_p95 = float(np.percentile(angles, 95))

    width_roughness = centerline_roughness = tip_bluntness = 0.0
    major_length = 1.0
    if profile is not None:
        body_width = max(float(profile["body_width"]), 1.0)
        width_roughness = float(np.percentile(
            np.abs(profile["width"] - profile["smooth_width"]), 90) / body_width)
        centerline_roughness = float(np.percentile(
            np.abs(profile["center"] - profile["smooth_center"]), 90) / body_width)
        tip_bluntness = max(float(profile["start_bluntness"]), float(profile["end_bluntness"]))
        major_length = float(profile["major_length"])

    solidity = 1.0
    if points is not None:
        contour32 = points.astype(np.float32).reshape((-1, 1, 2))
        hull = cv2.convexHull(contour32)
        hull_area = float(cv2.contourArea(hull))
        if hull_area > 0:
            solidity = float(min(1.0, area / hull_area))

    ratios = [
        contour_residual / QUALITY_LIMITS["contour_residual"],
        curvature_p95 / QUALITY_LIMITS["curvature_p95"],
        width_roughness / QUALITY_LIMITS["width_roughness"],
        centerline_roughness / QUALITY_LIMITS["centerline_roughness"],
        tip_bluntness / QUALITY_LIMITS["tip_bluntness"],
    ]
    quality = float(max(ratios))
    return ShapeMetrics(
        quality, contour_residual, curvature_p95, width_roughness,
        centerline_roughness, tip_bluntness, solidity, major_length,
        area, centroid_x, centroid_y)


def _smooth_contour_candidate(mask: np.ndarray,
                              sigma_fraction: float = 0.007) -> np.ndarray | None:
    points = _contour(mask)
    if points is None:
        return None
    sampled = _resample_closed(points, samples=max(128, min(512, len(points))))
    if sampled is None:
        return None
    sigma = max(0.8, len(sampled) * sigma_fraction)
    smooth = np.column_stack((
        ndimage.gaussian_filter1d(sampled[:, 0], sigma=sigma, mode="wrap"),
        ndimage.gaussian_filter1d(sampled[:, 1], sigma=sigma, mode="wrap"),
    ))
    polygon = np.rint(smooth).astype(np.int32)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, mask.shape[1] - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, mask.shape[0] - 1)
    candidate = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillPoly(candidate, [polygon.reshape((-1, 1, 2))], 1)
    return _main_component(candidate.astype(bool))


def _local_morphology_candidate(mask: np.ndarray,
                                radius_fraction: float = 0.42) -> np.ndarray | None:
    """Remove a bump then fill a notch with a width-adaptive round kernel."""
    yy, xx = np.nonzero(mask)
    if len(xx) < 20:
        return None
    before = measure_shape(mask)
    mean_half_width = max(1.0, before.area / max(2.0 * before.major_length, 2.0))
    radius = int(np.clip(round(radius_fraction * mean_half_width), 1, 7))
    margin = radius + 3
    y0 = max(0, int(yy.min()) - margin)
    y1 = min(mask.shape[0], int(yy.max()) + margin + 1)
    x0 = max(0, int(xx.min()) - margin)
    x1 = min(mask.shape[1], int(xx.max()) + margin + 1)
    local = mask[y0:y1, x0:x1].astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    smooth_reference = cv2.morphologyEx(
        cv2.morphologyEx(local, cv2.MORPH_OPEN, kernel),
        cv2.MORPH_CLOSE, kernel).astype(bool)
    # Never let a local body cleanup shorten or round the anatomical tips.
    # The end zones are copied verbatim from the network mask.
    profile = _profile(mask)
    if profile is not None:
        grid_y, grid_x = np.indices(local.shape, dtype=np.float64)
        global_x = grid_x + x0
        global_y = grid_y + y0
        origin = np.asarray(profile["origin"], dtype=np.float64)
        major = np.asarray(profile["major"], dtype=np.float64)
        along = ((global_x - origin[0]) * major[0]
                 + (global_y - origin[1]) * major[1])
        original_along = ((xx - origin[0]) * major[0]
                          + (yy - origin[1]) * major[1])
        lower, upper = float(original_along.min()), float(original_along.max())
        end_margin = max(2.0, 0.08 * (upper - lower))
        end_zone = (along <= lower + end_margin) | (along >= upper - end_margin)
        smooth_reference[end_zone] = local.astype(bool)[end_zone]
    candidate = np.zeros_like(mask, dtype=bool)
    candidate[y0:y1, x0:x1] = _main_component(smooth_reference)
    return candidate


def _morphological_skeleton(mask: np.ndarray) -> np.ndarray:
    """Return a topology-preserving skeleton using Zhang-Suen thinning."""
    yy, xx = np.nonzero(mask)
    result = np.zeros_like(mask, dtype=bool)
    if not len(yy):
        return result
    margin = 3
    y0, y1 = max(0, int(yy.min()) - margin), min(mask.shape[0], int(yy.max()) + margin + 1)
    x0, x1 = max(0, int(xx.min()) - margin), min(mask.shape[1], int(xx.max()) + margin + 1)
    working = np.pad(mask[y0:y1, x0:x1].astype(bool), 1)
    # Vectorized Zhang-Suen iterations preserve connectivity through bends,
    # unlike a morphological skeleton whose diagonal fragments can disconnect.
    while True:
        removed = 0
        for first_step in (True, False):
            center = working[1:-1, 1:-1]
            p2 = working[:-2, 1:-1]
            p3 = working[:-2, 2:]
            p4 = working[1:-1, 2:]
            p5 = working[2:, 2:]
            p6 = working[2:, 1:-1]
            p7 = working[2:, :-2]
            p8 = working[1:-1, :-2]
            p9 = working[:-2, :-2]
            neighbours = (p2.astype(np.uint8) + p3 + p4 + p5 + p6 + p7 + p8 + p9)
            transitions = ((~p2 & p3).astype(np.uint8) + (~p3 & p4) +
                           (~p4 & p5) + (~p5 & p6) + (~p6 & p7) +
                           (~p7 & p8) + (~p8 & p9) + (~p9 & p2))
            if first_step:
                connectivity = ~(p2 & p4 & p6) & ~(p4 & p6 & p8)
            else:
                connectivity = ~(p2 & p4 & p8) & ~(p2 & p6 & p8)
            deletion = (center & (neighbours >= 2) & (neighbours <= 6) &
                        (transitions == 1) & connectivity)
            count = int(deletion.sum())
            if count:
                center[deletion] = False
                removed += count
        if not removed:
            break
    skeleton = working[1:-1, 1:-1]
    components, number = ndimage.label(skeleton, structure=np.ones((3, 3), dtype=np.uint8))
    if number:
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        skeleton = components == int(np.argmax(sizes))
    result[y0:y1, x0:x1] = skeleton > 0
    return result


def _longest_path_from_skeleton(skeleton: np.ndarray) -> np.ndarray | None:
    """Approximate a skeleton's geodesic diameter as ordered (x, y) points."""
    coordinates = np.argwhere(skeleton)
    if len(coordinates) < 8:
        return None
    height, width = skeleton.shape
    flat_to_index = {
        int(y) * width + int(x): index
        for index, (y, x) in enumerate(coordinates)
    }
    neighbours: list[list[tuple[int, float]]] = [[] for _ in range(len(coordinates))]
    for index, (y, x) in enumerate(coordinates):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if not (dx or dy):
                    continue
                other = flat_to_index.get((int(y) + dy) * width + int(x) + dx)
                if other is not None:
                    neighbours[index].append((other, 1.41421356237 if dx and dy else 1.0))
    endpoints = [index for index, adjacent in enumerate(neighbours) if len(adjacent) == 1]
    eligible = endpoints if len(endpoints) >= 2 else list(range(len(coordinates)))

    def farthest(start: int, candidates: list[int]) -> tuple[int, list[int], np.ndarray]:
        distance = np.full(len(coordinates), np.inf, dtype=np.float64)
        parent = [-1] * len(coordinates)
        distance[start] = 0.0
        queue = [(0.0, start)]
        while queue:
            value, current = heapq.heappop(queue)
            if value != distance[current]:
                continue
            for other, weight in neighbours[current]:
                proposed = value + weight
                if proposed < distance[other]:
                    distance[other] = proposed
                    parent[other] = current
                    heapq.heappush(queue, (proposed, other))
        valid = [index for index in candidates if np.isfinite(distance[index])]
        target = max(valid, key=lambda index: distance[index])
        return target, parent, distance

    first, _, _ = farthest(eligible[0], eligible)
    second, parent, _ = farthest(first, eligible)
    path = [second]
    while path[-1] != first and parent[path[-1]] >= 0:
        path.append(parent[path[-1]])
    if path[-1] != first or len(path) < 8:
        return None
    ordered_yx = coordinates[np.asarray(path[::-1], dtype=int)].astype(np.float64)
    return ordered_yx[:, ::-1]


def _longest_skeleton_path(mask: np.ndarray) -> np.ndarray | None:
    """Skeletonize a mask and return its ordered geodesic centerline."""
    return _longest_path_from_skeleton(_morphological_skeleton(mask))


def _resample_open(points: np.ndarray, spacing: float = 1.0) -> np.ndarray | None:
    delta = np.diff(points, axis=0)
    segment = np.sqrt((delta * delta).sum(axis=1))
    length = float(segment.sum())
    if length < 7.0:
        return None
    cumulative = np.concatenate(([0.0], np.cumsum(segment)))
    samples = max(8, int(round(length / max(spacing, 0.25))) + 1)
    positions = np.linspace(0.0, length, samples)
    x = np.interp(positions, cumulative, points[:, 0])
    y = np.interp(positions, cumulative, points[:, 1])
    return np.column_stack((x, y))


def _centerline_candidate(mask: np.ndarray) -> np.ndarray | None:
    """Rebuild a curved worm from its geodesic centerline and local half-width."""
    path = _longest_skeleton_path(mask)
    if path is None:
        return None
    center = _resample_open(path)
    if center is None or len(center) < 8:
        return None
    sigma = min(10.0, max(1.0, 0.012 * len(center)))
    smooth = np.column_stack((
        ndimage.gaussian_filter1d(center[:, 0], sigma=sigma, mode="nearest"),
        ndimage.gaussian_filter1d(center[:, 1], sigma=sigma, mode="nearest"),
    ))
    smooth[0], smooth[-1] = center[0], center[-1]
    tangent = np.gradient(smooth, axis=0)
    tangent_length = np.sqrt((tangent * tangent).sum(axis=1))
    if np.any(tangent_length < 1e-6):
        return None
    tangent /= tangent_length[:, None]
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    distance = ndimage.distance_transform_edt(mask)
    radius = ndimage.map_coordinates(
        distance, [smooth[:, 1], smooth[:, 0]], order=1, mode="nearest")
    if float(np.median(radius)) < 1.0:
        return None
    radius_sigma = min(14.0, max(1.2, 0.020 * len(radius)))
    radius = ndimage.gaussian_filter1d(radius, sigma=radius_sigma, mode="nearest")
    radius = np.maximum(radius, 0.75)

    body = radius[max(1, len(radius) // 4):max(2, 3 * len(radius) // 4)]
    body_radius = max(float(np.median(body)), 1.0)

    def extend_start(points, tangents, radii, at_start):
        endpoint_radius = float(radii[0] if at_start else radii[-1])
        if endpoint_radius / body_radius <= QUALITY_LIMITS["tip_bluntness"]:
            return points, tangents, radii
        count = int(round(min(0.06 * len(points), max(2.0, 1.25 * endpoint_radius))))
        if count <= 0:
            return points, tangents, radii
        if at_start:
            steps = np.arange(count, 0, -1, dtype=np.float64)
            extra_points = points[0] - steps[:, None] * tangents[0]
            extra_tangents = np.repeat(tangents[0][None], count, axis=0)
            extra_radii = np.linspace(0.0, endpoint_radius, count + 1)[:-1]
            return (np.vstack((extra_points, points)), np.vstack((extra_tangents, tangents)),
                    np.concatenate((extra_radii, radii)))
        steps = np.arange(1, count + 1, dtype=np.float64)
        extra_points = points[-1] + steps[:, None] * tangents[-1]
        extra_tangents = np.repeat(tangents[-1][None], count, axis=0)
        extra_radii = np.linspace(endpoint_radius, 0.0, count + 1)[1:]
        return (np.vstack((points, extra_points)), np.vstack((tangents, extra_tangents)),
                np.concatenate((radii, extra_radii)))

    smooth, tangent, radius = extend_start(smooth, tangent, radius, True)
    smooth, tangent, radius = extend_start(smooth, tangent, radius, False)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    polygon = np.vstack((smooth - radius[:, None] * normal,
                         (smooth + radius[:, None] * normal)[::-1]))
    polygon = np.rint(polygon).astype(np.int32)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, mask.shape[1] - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, mask.shape[0] - 1)
    candidate = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillPoly(candidate, [polygon.reshape((-1, 1, 2))], 1)
    return _main_component(candidate.astype(bool))


def _profile_candidate(mask: np.ndarray, profile: dict) -> np.ndarray | None:
    width = np.asarray(profile["smooth_width"], dtype=np.float64)
    center = np.asarray(profile["smooth_center"], dtype=np.float64)
    bins = len(width)
    if bins < 8:
        return None

    def extension(bluntness: float, endpoint_width: float) -> int:
        if bluntness <= QUALITY_LIMITS["tip_bluntness"]:
            return 0
        return int(round(min(0.07 * bins, max(2.0, 1.35 * endpoint_width))))

    start_extension = extension(float(profile["start_bluntness"]), float(width[0]))
    end_extension = extension(float(profile["end_bluntness"]), float(width[-1]))
    positions = np.arange(-start_extension, bins + end_extension, dtype=np.float64)

    extended_center = np.interp(positions, np.arange(bins), center)
    slope_window = max(2, int(0.04 * bins))
    start_slope = float(np.median(np.diff(center[:slope_window + 1])))
    end_slope = float(np.median(np.diff(center[-slope_window - 1:])))
    before = positions < 0
    after = positions > bins - 1
    extended_center[before] = center[0] + positions[before] * start_slope
    extended_center[after] = center[-1] + (positions[after] - bins + 1) * end_slope

    extended_width = np.interp(positions, np.arange(bins), width)
    if start_extension:
        extended_width[:start_extension] = np.linspace(0.0, width[0], start_extension + 1)[:-1]
    if end_extension:
        extended_width[-end_extension:] = np.linspace(width[-1], 0.0, end_extension + 1)[1:]
    extended_width = np.maximum(extended_width, 0.0)

    along = float(profile["lower"]) + positions
    origin = np.asarray(profile["origin"], dtype=np.float64)
    major = np.asarray(profile["major"], dtype=np.float64)
    normal = np.asarray(profile["normal"], dtype=np.float64)
    center_xy = origin + along[:, None] * major + extended_center[:, None] * normal
    left = center_xy - extended_width[:, None] * normal
    right = center_xy + extended_width[:, None] * normal
    polygon = np.vstack((left, right[::-1]))
    polygon = np.rint(polygon).astype(np.int32)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, mask.shape[1] - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, mask.shape[0] - 1)
    candidate = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillPoly(candidate, [polygon.reshape((-1, 1, 2))], 1)
    return _main_component(candidate.astype(bool))


def _reasons(metrics: ShapeMetrics) -> list[str]:
    reasons = []
    if metrics.contour_residual > QUALITY_LIMITS["contour_residual"]:
        reasons.append("ROUGH_CONTOUR")
    if metrics.curvature_p95 > QUALITY_LIMITS["curvature_p95"]:
        reasons.append("CURVATURE_SPIKE")
    if metrics.width_roughness > QUALITY_LIMITS["width_roughness"]:
        reasons.append("WIDTH_PROFILE")
    if metrics.centerline_roughness > QUALITY_LIMITS["centerline_roughness"]:
        reasons.append("CENTERLINE")
    if metrics.tip_bluntness > QUALITY_LIMITS["tip_bluntness"]:
        reasons.append("BLUNT_TIP")
    return reasons


def _candidate_is_safe(original: np.ndarray, candidate: np.ndarray,
                       before: ShapeMetrics,
                       foreground_probability: np.ndarray | None = None,
                       local_cleanup: bool = False,
                       ) -> tuple[bool, float, float, float]:
    area_after = int(candidate.sum())
    if not area_after or not before.area:
        return False, 1.0, 1.0, 0.0
    area_change = abs(area_after - before.area) / max(before.area, 1)
    yy, xx = np.nonzero(candidate)
    centroid_shift = np.hypot(
        float(xx.mean()) - before.centroid_x,
        float(yy.mean()) - before.centroid_y) / max(before.major_length, 1.0)
    intersection = int(np.logical_and(original, candidate).sum())
    union = int(np.logical_or(original, candidate).sum())
    iou = intersection / max(union, 1)
    # A geometry-only candidate must remain close to the neural-network mask.
    # In particular, do not erase pixels for which the network has strong
    # foreground evidence merely to make the outline look smoother.
    supported_removal = 0.0
    if foreground_probability is not None and foreground_probability.shape == original.shape:
        removed = original & ~candidate
        supported_removal = float(
            np.logical_and(removed, foreground_probability >= 0.50).sum()
        ) / max(before.area, 1)
    if local_cleanup:
        changed_fraction = float(np.logical_xor(original, candidate).sum()) / max(before.area, 1)
        original_boundary = original ^ ndimage.binary_erosion(original)
        candidate_boundary = candidate ^ ndimage.binary_erosion(candidate)
        if original_boundary.any() and candidate_boundary.any():
            to_candidate = ndimage.distance_transform_edt(~candidate_boundary)[original_boundary]
            to_original = ndimage.distance_transform_edt(~original_boundary)[candidate_boundary]
            shifts = np.concatenate((to_candidate, to_original))
            shift_p95 = float(np.percentile(shifts, 95))
            shift_p99 = float(np.percentile(shifts, 99))
        else:
            shift_p95 = shift_p99 = np.inf
        mean_half_width = max(1.0, before.area / max(2.0 * before.major_length, 2.0))
        safe = (
            changed_fraction <= 0.045
            and area_change <= 0.025
            and centroid_shift <= 0.006
            and iou >= 0.95
            and supported_removal <= 0.012
            and shift_p95 <= max(2.5, 0.30 * mean_half_width)
            and shift_p99 <= max(3.5, 0.50 * mean_half_width)
        )
    else:
        safe = (area_change <= 0.10 and centroid_shift <= 0.025 and iou >= 0.84
                and supported_removal <= 0.01)
    return safe, float(area_change), float(centroid_shift), float(iou)


def _local_roughness(metrics: ShapeMetrics) -> float:
    """Score only boundary defects; exclude large-scale bend and tip shape."""
    return float(max(
        metrics.contour_residual / QUALITY_LIMITS["contour_residual"],
        metrics.curvature_p95 / QUALITY_LIMITS["curvature_p95"],
        metrics.width_roughness / QUALITY_LIMITS["width_roughness"],
    ))


def _probability_evidence(original: np.ndarray, candidate: np.ndarray,
                          foreground_probability: np.ndarray | None) -> float:
    """Positive values mean changed pixels agree with the neural probability."""
    if foreground_probability is None or foreground_probability.shape != original.shape:
        return 0.0
    changed = original ^ candidate
    if not changed.any():
        return 0.0
    probability = np.clip(foreground_probability[changed].astype(np.float64), 0.0, 1.0)
    added = candidate[changed]
    evidence = np.where(added, 2.0 * probability - 1.0, 1.0 - 2.0 * probability)
    return float(np.mean(evidence))


def _limit_additions(original: np.ndarray, candidate: np.ndarray,
                     major_length: float, foreground_probability: np.ndarray | None) -> np.ndarray:
    region_y, region_x = np.nonzero(original | candidate)
    if not len(region_y):
        return original
    max_distance = max(3.0, 0.075 * major_length)
    margin = int(np.ceil(max_distance)) + 2
    y0 = max(0, int(region_y.min()) - margin)
    y1 = min(original.shape[0], int(region_y.max()) + margin + 1)
    x0 = max(0, int(region_x.min()) - margin)
    x1 = min(original.shape[1], int(region_x.max()) + margin + 1)
    local_original = original[y0:y1, x0:x1]
    local_candidate = candidate[y0:y1, x0:x1]
    outside_distance = ndimage.distance_transform_edt(~local_original)
    close_support = outside_distance <= max(2.0, 0.035 * major_length)
    allowed = local_original | close_support
    if foreground_probability is not None and foreground_probability.shape == original.shape:
        local_probability = foreground_probability[y0:y1, x0:x1]
        allowed |= (local_probability >= 0.14) & (outside_distance <= max_distance)
    limited = np.zeros_like(original)
    limited[y0:y1, x0:x1] = _main_component(local_candidate & allowed)
    return limited


def refine_instance_shapes(instances: np.ndarray,
                           foreground_probability: np.ndarray | None = None
                           ) -> tuple[np.ndarray, list[ShapeReport]]:
    """检测并保守修复每个实例；实例数量和标签顺序保持不变。"""
    instances = np.asarray(instances, dtype=np.uint16)
    candidates: list[tuple[int, np.ndarray]] = []
    reports: list[ShapeReport] = []
    for label in [int(value) for value in np.unique(instances) if value]:
        original = _main_component(instances == label)
        before = measure_shape(original)
        reasons = _reasons(before)
        chosen = original
        after = before
        repaired = False
        area_change = centroid_shift = 0.0
        candidate_kind = "original"
        probability_evidence = 0.0
        candidate_iou = 1.0

        if reasons:
            profile_reasons = {"WIDTH_PROFILE", "CENTERLINE", "BLUNT_TIP"}
            local_defect_reasons = {"ROUGH_CONTOUR", "CURVATURE_SPIKE", "WIDTH_PROFILE"}
            # Generate several conservative alternatives and retain the original
            # explicitly.  Candidate selection uses probability evidence as well
            # as shape quality, instead of accepting the first smoother outline.
            options: list[tuple[str, np.ndarray | None]] = [
                ("contour_weak", _smooth_contour_candidate(original, 0.0045)),
                ("contour_medium", _smooth_contour_candidate(original, 0.0070)),
                ("contour_strong", _smooth_contour_candidate(original, 0.0100)),
            ]
            if local_defect_reasons.intersection(reasons):
                options[0:0] = [
                    ("local_morph", _local_morphology_candidate(original)),
                ]
            if profile_reasons.intersection(reasons):
                options.insert(0, ("curved_centerline", _centerline_candidate(original)))
            # The historical straight-PCA profile candidate is intentionally not
            # auto-applied.  On the first manual-mask audit it reduced Dice in
            # 42.9% and boundary F1 in 64.3% of its repairs.  Keep the helper only
            # for reproducibility while curved-centerline candidates are evaluated.

            best_score = 0.0
            before_local_roughness = _local_roughness(before)
            seen: list[np.ndarray] = []
            for kind, option in options:
                if option is None:
                    continue
                if any(np.array_equal(option, previous) for previous in seen):
                    continue
                seen.append(option)
                # 不允许修复结果侵入其他虫的原始区域。
                option = option & ((instances == 0) | (instances == label))
                option = _limit_additions(
                    original, option, before.major_length, foreground_probability)
                is_local_cleanup = kind.startswith("local_")
                safe, candidate_area_change, candidate_centroid_shift, option_iou = _candidate_is_safe(
                    original, option, before, foreground_probability,
                    local_cleanup=is_local_cleanup)
                if not safe:
                    continue
                candidate_metrics = measure_shape(option)
                evidence = _probability_evidence(
                    original, option, foreground_probability)
                quality_gain = ((before.quality - candidate_metrics.quality) /
                                max(before.quality, 1e-6))
                local_gain = ((before_local_roughness - _local_roughness(candidate_metrics)) /
                              max(before_local_roughness, 1e-6))
                if is_local_cleanup:
                    score = (local_gain + 0.08 * evidence
                             - 0.75 * candidate_area_change
                             - 3.00 * candidate_centroid_shift)
                else:
                    score = (quality_gain + 0.25 * evidence
                             - 0.50 * candidate_area_change
                             - 2.00 * candidate_centroid_shift)
                probability_ok = (
                    foreground_probability is not None
                    and foreground_probability.shape == original.shape
                    and evidence >= (MIN_LOCAL_PROBABILITY_EVIDENCE if is_local_cleanup
                                     else MIN_PROBABILITY_EVIDENCE)
                )
                geometry_ok = (
                    local_gain >= MIN_LOCAL_ROUGHNESS_GAIN
                    and candidate_metrics.centerline_roughness
                    <= max(before.centerline_roughness * 1.05, before.centerline_roughness + 0.002)
                    and candidate_metrics.tip_bluntness
                    <= max(before.tip_bluntness * 1.05, before.tip_bluntness + 0.01)
                ) if is_local_cleanup else candidate_metrics.quality <= before.quality * 0.995
                if (geometry_ok and probability_ok and score > best_score + 0.005):
                    chosen = option
                    after = candidate_metrics
                    area_change = candidate_area_change
                    centroid_shift = candidate_centroid_shift
                    candidate_kind = kind
                    probability_evidence = evidence
                    candidate_iou = option_iou
                    repaired = True
                    best_score = score

        remaining = _reasons(after)
        if not reasons:
            status = "PASS_SHAPE"
            needs_review = False
        elif repaired:
            # Shape repair changes the quantitative ROI and must never silently
            # upgrade itself to PASS using the same heuristic that generated it.
            status = "REVIEW_SHAPE_REFINED"
            needs_review = True
        else:
            status = "REVIEW_SHAPE"
            needs_review = True
        candidates.append((label, chosen))
        reports.append(ShapeReport(
            label=label, status=status, repaired=repaired, needs_review=needs_review,
            reasons="|".join(remaining or reasons or ["NONE"]),
            quality_before=before.quality, quality_after=after.quality,
            contour_residual=after.contour_residual,
            curvature_p95=after.curvature_p95,
            width_roughness=after.width_roughness,
            centerline_roughness=after.centerline_roughness,
            tip_bluntness=after.tip_bluntness, solidity=after.solidity,
            area_before=before.area, area_after=after.area,
            area_change_fraction=area_change,
            centroid_shift_fraction=centroid_shift,
            candidate_kind=candidate_kind,
            probability_evidence=probability_evidence,
            candidate_iou=candidate_iou,
        ))

    if not candidates:
        return np.zeros_like(instances), reports

    output = np.zeros_like(instances)
    stack = np.stack([mask for _, mask in candidates])
    coverage = stack.sum(axis=0)
    for index, (label, mask) in enumerate(candidates):
        output[mask & (coverage == 1)] = label
    overlap_y, overlap_x = np.nonzero(coverage > 1)
    if len(overlap_y):
        original_owners = instances[overlap_y, overlap_x]
        best_distance = np.full(len(overlap_y), np.inf)
        best_label = np.zeros(len(overlap_y), dtype=np.uint16)
        # 逐条计算，避免 n=20 时同时在内存中保留 20 张 float64 距离图。
        for index, (label, _) in enumerate(candidates):
            eligible = stack[index, overlap_y, overlap_x]
            owner = original_owners == label
            best_label[owner & eligible] = label
            best_distance[owner & eligible] = -1.0
            unresolved = eligible & (best_distance >= 0)
            if not unresolved.any():
                continue
            distance = ndimage.distance_transform_edt(~(instances == label))
            values = distance[overlap_y, overlap_x]
            update = unresolved & (values < best_distance)
            best_distance[update] = values[update]
            best_label[update] = label
        output[overlap_y, overlap_x] = best_label
    return output, reports


def write_shape_report(path: str | Path, reports: list[ShapeReport]) -> None:
    path = Path(path)
    fields = list(ShapeReport.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            writer.writerow(asdict(report))
