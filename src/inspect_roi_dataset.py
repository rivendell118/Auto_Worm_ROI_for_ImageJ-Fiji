from __future__ import print_function

import sys

# MGLTools bundles a legacy Python 2 / NumPy stack that can terminate with a
# Windows access violation on large microscopy arrays. Refuse to run there
# instead of risking another native crash dialog.
if sys.version_info[0] < 3:
    print("This utility requires Python 3.9+; do not run it with MGLTools Python 2.")
    sys.exit(2)

import csv
import glob
import os
import struct
import zipfile

import numpy as np
from PIL import Image, ImageDraw


def be_u16(data, offset):
    return struct.unpack(">H", data[offset:offset + 2])[0]


def byte_value(value):
    return value if isinstance(value, int) else ord(value)


def parse_imagej_roi(data, name):
    if data[:4] != b"Iout":
        raise ValueError("Not an ImageJ ROI: " + name)

    roi_type = byte_value(data[6])
    top = be_u16(data, 8)
    left = be_u16(data, 10)
    bottom = be_u16(data, 12)
    right = be_u16(data, 14)
    count = be_u16(data, 16)
    options = be_u16(data, 50)

    # Polygon, freehand and traced ROIs all store coordinate arrays at byte 64.
    if roi_type not in (0, 7, 8):
        raise ValueError("Unsupported ROI type %d in %s" % (roi_type, name))

    x_start = 64
    y_start = x_start + 2 * count
    xs = [left + be_u16(data, x_start + 2 * i) for i in range(count)]
    ys = [top + be_u16(data, y_start + 2 * i) for i in range(count)]

    # ImageJ may append floating-point coordinates for sub-pixel ROIs.
    if options & 128:
        float_x_start = y_start + 2 * count
        float_y_start = float_x_start + 4 * count
        if len(data) >= float_y_start + 4 * count:
            xs = [struct.unpack(">f", data[float_x_start + 4 * i:float_x_start + 4 * i + 4])[0]
                  for i in range(count)]
            ys = [struct.unpack(">f", data[float_y_start + 4 * i:float_y_start + 4 * i + 4])[0]
                  for i in range(count)]

    return {
        "name": name,
        "type": roi_type,
        "bounds": (left, top, right, bottom),
        "points": list(zip(xs, ys)),
    }


def load_rois(zip_path):
    rois = []
    with zipfile.ZipFile(zip_path, "r") as archive:
        for name in archive.namelist():
            if name.lower().endswith(".roi"):
                rois.append(parse_imagej_roi(archive.read(name), name))
    return rois


def load_csv(csv_path):
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def image_array(image):
    if image.mode == "I;16B":
        return np.frombuffer(image.tobytes(), dtype=">u2").reshape((image.size[1], image.size[0]))
    if image.mode in ("I;16", "I;16L"):
        return np.frombuffer(image.tobytes(), dtype="<u2").reshape((image.size[1], image.size[0]))
    if image.mode == "1":
        # np.asarray on a 1-bit image gives a *bool* array, and the first thing
        # the pipeline does with it is a background subtraction -- "-" on bools
        # is a TypeError, so a binary image used to die mid-batch with a message
        # about bitwise_xor. Widen it to the same 0/255 the "L" path yields, so
        # the rest of the program sees an ordinary 8-bit image.
        return np.where(np.asarray(image), 255, 0).astype(np.uint8)
    return np.asarray(image)


def enhanced_rgb(array):
    positive = array[array > 0]
    # Sampling is visually indistinguishable here and is much faster with the
    # older NumPy bundled in some microscopy installations.
    sampled = positive[::16]
    high = float(np.percentile(sampled, 99.8)) if sampled.size else 1.0
    if high <= 0:
        high = 1.0
    green = np.clip(array.astype(np.float32) * (255.0 / high), 0, 255).astype(np.uint8)
    zero = np.zeros(green.shape, dtype=np.uint8)
    return Image.fromarray(np.dstack((zero, green, zero)), "RGB")


def polygon_mask(size, points):
    """The pixels ImageJ counts for this polygon.

    ImageJ decides a pixel belongs to a ROI by whether the pixel's *centre*
    falls inside it, and pixel (c, r) has its centre at (c, r). PIL's
    ImageDraw.polygon fills by pixel cells instead -- the convention
    cv2.fillPoly used, which 0.4.1 removed from the export side and this reader
    never followed. Rasterising an n x n block that way gives (n+1) x (n+1)
    pixels, so this tool reported about 6% more area than the measurement table
    on data that was in fact correct -- the opposite of what a tool whose job is
    "show that the ROIs match the masks" must do.

    Same rule as tests/test_manual_exclusion.py, which pins it from the other
    direction: the winding number of the polygon at each pixel centre, kept by
    the non-zero fill rule. Holes are subpaths wound the other way, so this
    stays correct if composite ROIs are ever read here.
    """
    # Half-open in y (y1 <= row < y2). That is what keeps a vertex from being
    # counted twice and the top edge of a polygon from spilling into the row
    # above it -- the boundary cases no fill-by-cell rule can get right.
    #
    # size is PIL's, i.e. (width, height), while everything below is indexed by
    # row and column. The two agree only on a square image, which is why the
    # accumulator is built from the row and column counts rather than from size
    # directly -- np.zeros(size) would give a (width, height) array and the
    # comparisons below would fail to broadcast on any other shape.
    rows = np.arange(size[1], dtype=np.float64)[:, None]
    columns = np.arange(size[0], dtype=np.float64)[None, :]
    vertices = np.asarray(points, dtype=np.float64)
    winding = np.zeros((size[1], size[0]), dtype=np.int64)
    for (x1, y1), (x2, y2) in zip(vertices, np.roll(vertices, -1, axis=0)):
        if y1 == y2:
            continue
        up = (y1 <= rows) & (rows < y2)
        down = (y2 <= rows) & (rows < y1)
        side = (x2 - x1) * (rows - y1) - (columns - x1) * (y2 - y1)
        winding += (up & (side > 0)).astype(np.int64)
        winding -= (down & (side < 0)).astype(np.int64)
    return (winding != 0).astype(np.uint8)


# An image the model found no worm in still gets a background ROI appended
# (batch_worm_roi appends one unconditionally), so it has exactly one ROI and
# the worm list is empty. min()/max() on that used to end the whole scan with
# "min() iterable argument is empty", taking every later image's report with it;
# np.mean([]) would only have warned and written a nan. Report nan for all three
# so such an image is summarised like any other.
def min_or_nan(values):
    return min(values) if values else float("nan")


def max_or_nan(values):
    return max(values) if values else float("nan")


def mean_or_nan(values):
    return float(np.mean(values)) if values else float("nan")


COLORS = [
    (255, 64, 64), (255, 170, 0), (255, 255, 0), (100, 255, 80),
    (0, 255, 210), (0, 180, 255), (80, 100, 255), (180, 80, 255),
    (255, 80, 210), (255, 150, 170), (255, 255, 255), (150, 150, 150),
]


def inspect_pair(image_path, roi_zip, csv_path, output_dir):
    image = Image.open(image_path)
    array = image_array(image)
    rois = load_rois(roi_zip)
    rows = load_csv(csv_path)
    canvas = enhanced_rgb(array)
    draw = ImageDraw.Draw(canvas)
    line_width = max(2, image.size[0] // 512)

    masks = []
    for index, roi in enumerate(rois):
        color = (0, 255, 255) if index == len(rois) - 1 else COLORS[index % len(COLORS)]
        points = roi["points"]
        draw.line(points + [points[0]], fill=color, width=line_width)
        center_x = int(sum(p[0] for p in points) / float(len(points)))
        center_y = int(sum(p[1] for p in points) / float(len(points)))
        draw.text((center_x, center_y), str(index + 1), fill=color)
        masks.append(polygon_mask(image.size, points))

    base = os.path.splitext(os.path.basename(image_path))[0]
    overlay_path = os.path.join(output_dir, base + "_roi_overlay.png")
    canvas.save(overlay_path)

    worm_masks = masks[:-1]
    pixel_areas = [int(mask.sum()) for mask in worm_masks]
    csv_areas = [float(row["Area"]) for row in rows[:-1]]
    measured_means = [float(array[mask.astype(bool)].mean()) for mask in worm_masks]
    measured_sums = [float(array[mask.astype(bool)].sum()) for mask in worm_masks]
    csv_means = [float(row["Mean"]) for row in rows[:-1]]
    csv_sums = [float(row["RawIntDen"]) for row in rows[:-1]]
    mean_relative_errors = [abs(value - reference) / max(abs(reference), 1e-12)
                            for value, reference in zip(measured_means, csv_means)]
    sum_relative_errors = [abs(value - reference) / max(abs(reference), 1e-12)
                           for value, reference in zip(measured_sums, csv_sums)]
    area_scales = [csv_area / pixel_area for csv_area, pixel_area in zip(csv_areas, pixel_areas)]
    area_scale = float(np.median(area_scales)) if area_scales else float("nan")
    coverage = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
    for mask in worm_masks:
        coverage += mask
    # At a pixel covered by k ROIs, the number of overlapping ROI pairs is
    # k*(k-1)/2. This avoids scanning the image once per pair.
    coverage_signed = coverage.astype(np.int16)
    pair_overlap = int(((coverage_signed * (coverage_signed - 1)) // 2).sum())

    return {
        "image": os.path.basename(image_path),
        "width": image.size[0],
        "height": image.size[1],
        "mode": image.mode,
        "roi_count": len(rois),
        "worm_count": max(0, len(rois) - 1),
        "csv_rows": len(rows),
        "area_scale": area_scale,
        "min_pixel_area": min_or_nan(pixel_areas),
        "max_pixel_area": max_or_nan(pixel_areas),
        "pair_overlap_pixels": pair_overlap,
        "mean_mape_percent": 100.0 * mean_or_nan(mean_relative_errors),
        "mean_max_error_percent": 100.0 * max_or_nan(mean_relative_errors),
        "raw_sum_mape_percent": 100.0 * mean_or_nan(sum_relative_errors),
        "raw_sum_max_error_percent": 100.0 * max_or_nan(sum_relative_errors),
        "overlay": overlay_path,
    }


def main(root):
    output_dir = os.path.join(root, "_inspection")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    summaries = []
    for roi_zip in sorted(glob.glob(os.path.join(root, "RoiSet_*.zip"))):
        stem = os.path.basename(roi_zip)[len("RoiSet_"):-len(".zip")]
        image_path = os.path.join(root, stem + ".tif")
        csv_path = os.path.join(root, "Results_" + stem + ".csv")
        if not os.path.isfile(csv_path):
            csv_path = os.path.join(root, "Results" + stem + ".csv")
        if not os.path.isfile(image_path) or not os.path.isfile(csv_path):
            print("Missing matching files for", roi_zip)
            continue
        try:
            summary = inspect_pair(image_path, roi_zip, csv_path, output_dir)
        except Exception as error:
            # This runs over a whole dataset, and one image with an unreadable
            # ROI set or a mismatched CSV must not cost every later image its
            # report -- the per-image message is exactly what the operator needs
            # to see, and a scan that stops at the first bad file hides how many
            # others were fine.
            print("Failed to inspect %s: %s: %s"
                  % (os.path.basename(roi_zip), type(error).__name__, error))
            continue
        summaries.append(summary)
        print("{image}: {width}x{height} {mode}, {worm_count} worms + background, "
              "ROI area {min_pixel_area}-{max_pixel_area} px, area scale {area_scale:.6f}, "
              "pair overlap {pair_overlap_pixels} px, Mean MAPE {mean_mape_percent:.3f}%, "
              "RawIntDen MAPE {raw_sum_mape_percent:.3f}%".format(**summary))

    summary_path = os.path.join(output_dir, "roi_dataset_summary.csv")
    if summaries:
        keys = ["image", "width", "height", "mode", "roi_count", "worm_count", "csv_rows",
                "area_scale", "min_pixel_area", "max_pixel_area", "pair_overlap_pixels", "overlay"]
        keys[11:11] = ["mean_mape_percent", "mean_max_error_percent",
                       "raw_sum_mape_percent", "raw_sum_max_error_percent"]
        with open(summary_path, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(summaries)
    print("Summary:", summary_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_roi_dataset.py <dataset-folder>")
        sys.exit(2)
    main(sys.argv[1])
