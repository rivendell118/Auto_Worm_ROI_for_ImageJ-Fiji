from __future__ import print_function

import argparse
import csv
import glob
import hashlib
import os
import re
import uuid
from collections import namedtuple
from dataclasses import asdict

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import torch
from roifile import ImagejRoi, ROI_TYPE, roiwrite

from inspect_roi_dataset import image_array
from prepare_segmentation_dataset import normalize_for_segmentation
from train_worm_unet import IMAGE_SIZE, WormUNet
from evaluate_worm_unet import semantic_to_instances, resize_labels, colorize_labels
from evaluate_tip_refiner import refine_instances


# Written into every batch_summary.csv row, so a test build can be told apart
# from a release build in a bug report without asking which download was used.
SOFTWARE_VERSION = "0.4.2-beta-imagej"

# 每一版**发行过**的低清模型指纹，新的在前。做成集合而不是单个值：0.4.2 随包带的是
# models/0.2.1，但 0.4.1 那批已经散出去的权重（models/0.2.0）还在用，两者都得认出来。
# 只认一个的话，拿旧权重走命令行的人会被静默当成高清，正是这条判据要防的事。界面不受
# 影响 —— 它每次都把 low_clarity_split 显式传进来，这里只服务于命令行。
LOW_CLARITY_MODEL_SHA256S = (
    "afb80dc4bde0cc644ab4269bfc73463d7f20eace9592a21857ce155b8f4eed4b",  # 0.2.1（0.4.2）
    "5ce04d283a8dbc4e7b8ae460f6310627ecdfbef5feca23ace7bdbe51ef84ff3a",  # 0.2.0（0.3.3~0.4.1）
)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(final_path, writer):
    """Write a complete sibling file, then atomically replace the destination."""
    final_path = os.path.abspath(final_path)
    root, extension = os.path.splitext(final_path)
    temporary = "%s.%s.tmp%s" % (root, uuid.uuid4().hex, extension)
    try:
        writer(temporary)
        os.replace(temporary, final_path)
    finally:
        if os.path.exists(temporary):
            try:
                os.unlink(temporary)
            except OSError:
                pass


# The two subfolders every run writes into, under the output folder. They are
# declared here rather than inline because the Java plug-in carries the same two
# names: it is handed the output folder and has to look for the ROI sets, and
# write the measurement tables, in the same places this program uses. Renaming
# one of these means renaming it on both sides.
MEASUREMENTS_DIRNAME = "measurements"
OTHER_DIRNAME = "other"


def output_subdirs(output_dir):
    """The (measurements, other) subfolders of an output folder. Does not create them."""
    return (os.path.join(output_dir, MEASUREMENTS_DIRNAME),
            os.path.join(output_dir, OTHER_DIRNAME))


def ensure_output_layout(output_dir):
    """Create the output folder and both subfolders; return (measurements, other)."""
    measurements_dir, other_dir = output_subdirs(output_dir)
    for path in (output_dir, measurements_dir, other_dir):
        os.makedirs(path, exist_ok=True)
    return measurements_dir, other_dir


# Header of the Python measurement table. Also used to recognise a table an
# earlier run left behind, so it must stay in one place.
#
# The seven numbers an experimenter actually reads come first, in the order
# ImageJ's own Results window shows them; every other column follows. Two
# columns here need a word of explanation:
#   * the unnamed one carries the ROI index, the way ImageJ exports it;
#   * IntDen and RawIntDen hold the same number -- the Python backend has no
#     calibration, so there is nothing for a raw value to differ from. The
#     plug-in's backend does distinguish them.
PYTHON_MEASUREMENT_HEADER = ["Area", "Mean", "Min", "Max", "IntDen", "Median",
                             "RawIntDen", "", "Label", "BackgroundCorrectedMean",
                             "CTCF", "Type"]


def measurement_row(values):
    """One measurement-table row, in PYTHON_MEASUREMENT_HEADER order.

    Built from a dict keyed by column name rather than as a bare list, so that
    reordering the header above cannot silently write values under the wrong
    names -- a mislabelled row is still a perfectly well-formed CSV. A row whose
    keys do not match the header exactly raises here instead.
    """
    if set(values) != set(PYTHON_MEASUREMENT_HEADER):
        raise ValueError("measurement row does not match the header; only in one: %s"
                         % sorted(set(values) ^ set(PYTHON_MEASUREMENT_HEADER)))
    return [values[column] for column in PYTHON_MEASUREMENT_HEADER]


def _write_csv_atomic(path, header, rows):
    def write(temporary):
        with open(temporary, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
    _atomic_write(path, write)


def _same_directory(left, right):
    """True when two paths name the same directory."""
    return os.path.normcase(os.path.realpath(left)) == os.path.normcase(
        os.path.realpath(right))


def _reject_output_equal_to_input(input_folder, output_dir):
    """Refuse to write results into the folder being segmented.

    The batch clears same-named leftovers from earlier runs out of output_dir:
    stale measurement tables and QC reports are unlinked, and the batch summary
    is rewritten. Those files belong to this program only when output_dir is a
    dedicated results folder. Pointed at the source folder, they are the
    experimenter's own files, so the run has to stop before anything is removed.
    """
    if _same_directory(input_folder, output_dir):
        raise ValueError(
            "The output folder must differ from the input folder being "
            "processed (%s). Processing removes same-named results left by "
            "earlier runs, which would delete files in the source folder. "
            "Choose a separate output folder; the default is the '_auto_roi' "
            "subfolder." % output_dir)


# PIL modes that carry a single greyscale intensity per pixel, i.e. the images
# for which this program and ImageJ describe the same thing. Everything else is
# refused. This is a policy of this program, not a restatement of image_array():
# anything not listed here reaches numpy as-is, which for "P" would silently be
# palette *indices* rather than intensities. "I;16B" is big-endian 16-bit grey,
# not colour, and is a large share of the TIFFs in this lab -- dropping it from
# the list would reject valid data.
SUPPORTED_TIFF_MODES = ("1", "L", "I", "F", "I;16", "I;16L", "I;16B")

# Logged by the batch, and shown as a dialog by the GUI, for a 1-bit image. It
# is processed normally: the guard only refuses what would silently describe a
# different plane from the one ImageJ measures, and a binary image does not.
BINARY_WARNING = ("1-bit (binary) image; the model was not trained on binary "
                  "images, so the segmentation may be inaccurate")

# Result of check_image(). "code" is None when the image can be processed;
# otherwise it is the reason it cannot, and "value" carries the frame count,
# the PIL mode or the error text. On an image that *is* usable, "note" is a
# remark worth logging (its extra planes turned out to be empty) and "warning"
# is the short code of a caveat the user should be told about more visibly --
# the GUI turns it into a dialog, since a log line is easy to miss.
#
#   code          value
#   "unreadable"  str - the file or its pixel data could not be read
#                       (corrupt, or truncated by an interrupted copy)
#   "format"      str - readable, but not a TIFF; ImageJ cannot open it
#   "stack"       int - this many planes, at least one of them non-empty
#   "colour"      str - a PIL mode outside SUPPORTED_TIFF_MODES
#
#   warning       meaning
#   "binary"      a 1-bit image, which the model was never trained on
ImageCheck = namedtuple("ImageCheck", "code value note warning")


def check_image(path):
    """Inspect one file, returning an ImageCheck.

    Accepted only if the read process_image performs actually succeeds. Checking
    just the header is not enough: a half-copied TIFF opens cleanly and reports a
    mode, then raises when its pixels are decoded -- it would pass a header check
    and kill the batch in the middle of the run.

    Kept as a pure function so the GUI pre-check and the batch guard cannot
    disagree about what is acceptable.
    """
    try:
        with Image.open(path) as image:
            if image.format != "TIFF":
                return ImageCheck("format", image.format or "unknown", "", "")
            try:
                frames = image.n_frames
            except Exception:
                # A chain Pillow cannot walk past page 0 still hands page 0 to
                # both this program and ImageJ, which is all either one reads.
                frames = 1
            note = ""
            if frames > 1:
                # Writers such as MetaMorph append overlay planes as extra
                # pages, and ImageJ presents them as extra slices rather than
                # skipping them. A plane that is entirely zero is still not a
                # rejection: the mismatch this guard exists to prevent is a
                # *plausible-looking* wrong measurement, and a plane with no
                # signal cannot produce one. Anything with real data in it is a
                # genuine channel/Z/timepoint and is refused below.
                for index in range(1, frames):
                    image.seek(index)
                    if image_array(image).any():
                        return ImageCheck("stack", frames, "", "")
                image.seek(0)
                note = "%d empty extra plane(s); treated as overlay" % (frames - 1)
            if image.mode not in SUPPORTED_TIFF_MODES:
                return ImageCheck("colour", image.mode, "", "")
            # Readable and greyscale, so it is processed -- but the two models
            # this program ships were trained on 8- and 16-bit acquisitions
            # only, and a 1-bit image is a different kind of picture.
            warning = "binary" if image.mode == "1" else ""
            image_array(image)
            return ImageCheck(None, "", note, warning)
    except Exception as exc:  # PIL raises a wide range of errors for bad files
        return ImageCheck("unreadable", str(exc) or exc.__class__.__name__, "", "")


def _reject_unsupported_images(paths):
    """Stop the batch before it starts if any image would be mis-segmented.

    Refusing the whole batch up front is deliberate: a run that silently
    segmented plane 0 of a stack would hand ImageJ ROIs that do not match the
    plane it then measures, and nothing downstream would flag it.

    Returns the remarks to log about images that were accepted anyway.
    """
    # Fixed order so the message reads the same way every run.
    order = [("unreadable", "cannot be read: %s"),
             ("format", "is not a TIFF (%s); ImageJ cannot open it"),
             ("stack", "has %s planes"),
             ("colour", "is not single-channel greyscale (PIL mode %s)")]
    problems = {code: [] for code, _ in order}
    notes = []
    for path in paths:
        found = check_image(path)
        if found.code is None:
            name = os.path.basename(path)
            if found.note:
                notes.append("%s: %s" % (name, found.note))
            if found.warning == "binary":
                notes.append("%s: WARNING %s" % (name, BINARY_WARNING))
            continue
        problems[found.code].append((os.path.basename(path), found.value))
    lines = ["  %s %s" % (name, template % value)
             for code, template in order
             for name, value in problems[code]]
    if not lines:
        return notes
    raise ValueError(
        "These images cannot be processed:\n%s\n"
        "This program reads only the first plane and only single-channel "
        "greyscale, while ImageJ measures the channel, Z-slice and timepoint "
        "currently selected in its window; on multi-plane or colour files the "
        "ROIs and the measurements would describe different planes. Convert the "
        "files to single-channel, single-plane TIFFs (Image > Stacks > Stack to "
        "Images, then keep the channel you want) and run again." % "\n".join(lines))


def find_stem_collisions(paths):
    """Group paths whose results would be written under one set of file names.

    process_image names everything it writes after the file name with the
    extension removed, so "A.tif" and "A.tiff" in the same folder both produce
    "A_RoiSet.zip" and "A_QC.png". The second one silently overwrites the first,
    and the summary lists two rows under one image name -- there is no error and
    nothing downstream can tell. Case-insensitive, because on Windows those two
    spellings are the same file anyway.

    Returns [(stem, [file names])] in first-seen order, empty when there is none.
    """
    grouped = {}
    for path in paths:
        name = os.path.basename(path)
        grouped.setdefault(os.path.splitext(name)[0].lower(), []).append(name)
    return [(stem, names) for stem, names in grouped.items() if len(names) > 1]


def _reject_output_name_collisions(paths):
    """Stop the batch before it starts if two images would overwrite each other."""
    collisions = find_stem_collisions(paths)
    if not collisions:
        return
    lines = ["  " + " / ".join(names) for _, names in collisions]
    raise ValueError(
        "These images would write their results to the same file names:\n%s\n"
        "Results are named after the file name without its extension, so a .tif "
        "and a .tiff with the same name collide and the second one silently "
        "overwrites the first. Keep only one of each pair." % "\n".join(lines))


# The tail of batch_summary.csv: run metadata shared by every row, rather than
# per-image results. Naming the columns once keeps the header, _success_row and
# _failure_row from drifting apart, which is the failure mode worth guarding
# here: a row one field short still writes a perfectly well formed CSV, with
# every value after the gap quietly sitting under the wrong column name.
RUN_COLUMNS = ("run_id", "device", "inference_precision",
               "worm_model_sha256", "tip_model_sha256", "software_version",
               "error")


def _failure_row(header, name, detail, run_id, device_name):
    """One batch_summary.csv row for an image this run could not process.

    Recorded under "Failed_<name>" so the image missing from the results is
    identifiable in the table itself, not only in the log. Everything the run
    never got to stays empty; the reason goes in the error column, which no
    successful row fills in.
    """
    values = {
        "image": "Failed_" + name,
        "qc_status": "FAILED",
        "error": detail,
        "run_id": run_id,
        "device": device_name,
        "software_version": SOFTWARE_VERSION,
    }
    return [values.get(field, "") for field in header]


def _success_row(header, result, run_values):
    """One batch_summary.csv row for an image this run processed.

    Built by column name like _failure_row. process_image returns its values
    positionally, in data-column order, so the length is checked rather than
    trusted: a mismatch is a programming error, and aborting with that message
    beats writing a table whose columns silently mean something else.
    """
    data_columns = len(header) - len(RUN_COLUMNS)
    if len(result) != data_columns:
        raise ValueError(
            "process_image returned %d values for %d result columns"
            % (len(result), data_columns))
    values = dict(zip(header, result))
    values.update(dict(zip(RUN_COLUMNS, run_values)))
    return [values[field] for field in header]


def _first_row(path):
    """The header row of a CSV, or None if there is nothing readable there."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", newline="", encoding="utf-8-sig") as handle:
            return next(csv.reader(handle), None)
    except (OSError, UnicodeDecodeError, csv.Error):
        return None


def _unlink_quietly(path):
    try:
        os.unlink(path)
    except OSError:
        pass


def _remove_stale_report(path, expected_header):
    """Delete a report from an earlier run, but only if it really is ours.

    output_dir can legitimately hold files the experimenter put there, so
    sharing a generated name is not proof that this program wrote the file.
    Compare the header row first; anything else is left untouched.
    """
    if _first_row(path) == list(expected_header):
        _unlink_quietly(path)


# The plug-in writes <stem>_measurements.csv as well, with a schema of its own
# that grows as columns are added (0.4.1 added five calibration columns), so its
# table cannot be recognised by comparing a whole header written here. Two column
# names identify it instead -- the same rule the plug-in applies to its own
# combined table.
IMAGEJ_MEASUREMENT_COLUMNS = ("CTCF", "MeasurementBackend")


def _remove_stale_imagej_report(path):
    """Delete the plug-in's measurement table for this image, if that is what it is.

    Both backends write <stem>_measurements.csv. A run using the ImageJ backend
    does not write that file itself; the plug-in writes it afterwards. So a table
    left there by an earlier ImageJ run sits beside this run's ROIs under the same
    name and from a different schema, and once this run is over it reads as this
    run's measurement -- the ROIs next to it are this run's, which is what makes
    it convincing. Nothing in either file distinguishes the two afterwards.
    """
    first_row = _first_row(path)
    if first_row and all(name in first_row for name in IMAGEJ_MEASUREMENT_COLUMNS):
        _unlink_quietly(path)


# Every file process_image can write for one image: the suffix it is named with,
# and which subfolder of the output folder it belongs in. Listing them together
# keeps the failure cleanup below from drifting away from what process_image
# actually produces -- and now the two have to agree across two directories.
#
# Only the measurement table is a result; it goes in measurements/. The ROI sets
# ImageJ reads, the QC overlay and the four review tables are working material
# and go in other/.
IMAGE_OUTPUT_FILES = {
    "_RoiSet.zip": OTHER_DIRNAME,
    "_measurements.csv": MEASUREMENTS_DIRNAME,
    "_QC.png": OTHER_DIRNAME,
    "_shape_qc.csv": OTHER_DIRNAME,
    "_split_qc.csv": OTHER_DIRNAME,
    "_head_annotations.csv": OTHER_DIRNAME,
    "_manual_split_qc.csv": OTHER_DIRNAME,
    "_segment_qc.csv": OTHER_DIRNAME,
}


def _discard_partial_outputs(output_dir, stem):
    """Delete whatever a failed image already wrote. Returns what would not go.

    process_image writes the ROI set and the measurement table before it draws
    the overlay, so a failure late in the pipeline leaves a complete, valid
    looking RoiSet.zip behind. That matters because the ImageJ side measures
    every input that has a <stem>_RoiSet.zip -- so without this the batch
    reports an image as FAILED and Fiji then measures it anyway, using ROIs
    from a pipeline that never finished, and nothing anywhere says so.

    Deleting is the safe direction rather than keeping the files: the image is
    already recorded in batch_summary.csv, and a missing ROI set is exactly
    what the failure message tells the user to expect. The names are the
    program's own output names, which a successful run would have overwritten
    in any case.
    """
    stuck = []
    for suffix, subfolder in IMAGE_OUTPUT_FILES.items():
        path = os.path.join(output_dir, subfolder, stem + suffix)
        if not os.path.exists(path):
            continue
        try:
            os.unlink(path)
        except OSError as exc:
            stuck.append("%s (%s)" % (os.path.basename(path), exc))
    return stuck


def expected_count_from_filename(stem):
    match = re.search(r"(?:^|[\s_-])n\s*=\s*(\d+)(?:[\s_.-]|$)", stem,
                      flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def predict_raw(model, raw, device, image_size=IMAGE_SIZE,
                interior_threshold=None, erosion_iterations=0,
                min_area_fraction=0.005, min_height_fraction=0.10, max_instances=12,
                tip_model=None, tip_patch_size=192,
                tip_probability_threshold=0.40, tip_replace_fraction=0.14,
                normalization_mode="legacy", return_foreground_probability=False,
                low_clarity_split=False, split_reports=None, split_count_limit=None,
                exclusion_mask=None):
    normalized = normalize_for_segmentation(raw, mode=normalization_mode)
    model_input = normalized.copy()
    full_exclusion = None
    if exclusion_mask is not None:
        full_exclusion = np.asarray(exclusion_mask, dtype=bool)
        if full_exclusion.shape != normalized.shape:
            raise ValueError("exclusion_mask must match the source image shape")
        model_input[full_exclusion] = 0
    small = Image.fromarray(model_input).resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(small, dtype=np.float32) / 255.0
    small_exclusion = None
    if full_exclusion is not None:
        small_exclusion = np.asarray(
            Image.fromarray(full_exclusion.astype(np.uint8) * 255).resize(
                (image_size, image_size), Image.Resampling.NEAREST), dtype=np.uint8) > 0
        array[small_exclusion] = 0.0
    tensor = torch.from_numpy(array[None, None]).to(device)
    # Scientific output must be comparable between the CPU and CUDA packages.
    # Keep inference in FP32 instead of silently enabling CUDA FP16 autocast.
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0].float().cpu().numpy()
    instances, semantic = semantic_to_instances(
        probabilities, min_area_fraction=min_area_fraction,
        min_height_fraction=min_height_fraction,
        interior_threshold=interior_threshold,
        erosion_iterations=erosion_iterations,
        max_instances=max_instances)
    if small_exclusion is not None:
        instances = _apply_exclusion_mask_to_instances(instances, small_exclusion)
    if low_clarity_split:
        from low_clarity_splitter import split_low_clarity_instances
        split_limit = min(v for v in (max_instances, split_count_limit) if v) if (
            max_instances or split_count_limit) else None
        instances, reports = split_low_clarity_instances(
            instances, probabilities, array, max_instances=split_limit)
        if split_reports is not None:
            split_reports.extend(reports)
    if tip_model is not None:
        instances = refine_instances(
            tip_model, array, instances, device,
            patch_size=tip_patch_size,
            probability_threshold=tip_probability_threshold,
            replace_fraction=tip_replace_fraction)
    if small_exclusion is not None:
        instances = _apply_exclusion_mask_to_instances(instances, small_exclusion)
    resized_instances = resize_labels(instances, (raw.shape[1], raw.shape[0]))
    if full_exclusion is not None:
        resized_instances = _apply_exclusion_mask_to_instances(
            resized_instances, full_exclusion)
    if return_foreground_probability:
        foreground_probability = probabilities[1:].sum(axis=0).astype(np.float32)
        # .copy() is not redundant: np.asarray of a PIL image inherits PIL's
        # read-only buffer flag, and the exclusion below writes into the result.
        # Without the copy an image that has an exclusion region and also has
        # shape refinement enabled dies with "assignment destination is
        # read-only" -- after the model has already run.
        foreground_probability = np.asarray(
            Image.fromarray(foreground_probability, mode="F").resize(
                (raw.shape[1], raw.shape[0]), Image.Resampling.BILINEAR),
            dtype=np.float32).copy()
        if full_exclusion is not None:
            foreground_probability[full_exclusion] = 0.0
        return resized_instances, normalized, semantic, foreground_probability
    return resized_instances, normalized, semantic


def _apply_exclusion_mask_to_instances(instances, forbidden):
    """清除排除区像素，保留每个实例最大的连通部分并连续编号。"""
    source = np.asarray(instances)
    blocked = np.asarray(forbidden, dtype=bool)
    if source.shape != blocked.shape:
        raise ValueError("forbidden mask must match instances")
    cleared = source.copy()
    cleared[blocked] = 0
    result = np.zeros(source.shape, dtype=source.dtype)
    next_label = 1
    for label in np.unique(cleared):
        if label <= 0:
            continue
        components, component_count = ndimage.label(cleared == label)
        if component_count <= 0:
            continue
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        largest = int(np.argmax(sizes))
        if sizes[largest] <= 0:
            continue
        result[components == largest] = next_label
        next_label += 1
    return result


# 导出给 ImageJ 的轮廓沿**像素边界**走，不是沿像素编号走。
#
# ImageJ 判定「这个像素属不属于 ROI」看的是像素**中心**落没落在多边形里，而像素 (c, r)
# 的中心就在 (c, r)；OpenCV 的 findContours 给出的顶点却落在像素编号上。照搬过去，
# ImageJ 采样到的像素会比 Python 掩膜少最外面那一圈行列 —— 实测同一只虫 Python 侧
# 45272 像素、ImageJ 侧 44325 像素，差 2.1%，而 QC 那一列照样 PASS。
#
# 改成描像素边界之后两边的像素集合逐像素相同：第 c 列的左右边界在 c-0.5 / c+0.5，
# 第 r 行的上下边界在 r-0.5 / r+0.5，顶点落在半整数上。背景 ROI 同理。见
# VALIDATION_0.4.2.md。


def _boundary_edges(mask):
    """掩膜里每个 True 像素朝向空邻域的那几条边，返回 {起点: [终点, ...]}。

    顶点用**双倍角点坐标**表示（第 c 列的两条竖边在 2c-1 与 2c+1）：坐标乘 2 之后
    半整数全是整数，绕环时不必做浮点比较。
    """
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    edges = {}
    for row, column in zip(*np.nonzero(mask)):
        up = row == 0 or not mask[row - 1, column]
        down = row + 1 == height or not mask[row + 1, column]
        left = column == 0 or not mask[row, column - 1]
        right = column + 1 == width or not mask[row, column + 1]
        west, east = 2 * column - 1, 2 * column + 1
        north, south = 2 * row - 1, 2 * row + 1
        if up:
            edges.setdefault((west, north), []).append((east, north))
        if right:
            edges.setdefault((east, north), []).append((east, south))
        if down:
            edges.setdefault((east, south), []).append((west, south))
        if left:
            edges.setdefault((west, south), []).append((west, north))
    return edges


def _turn_rank(incoming, outgoing):
    """从入边转到出边要转多少：0 = 右转、1 = 直行、2 = 左转、3 = 掉头。"""
    cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
    dot = incoming[0] * outgoing[0] + incoming[1] * outgoing[1]
    if cross < 0:
        return 0
    if cross == 0 and dot > 0:
        return 1
    if cross > 0:
        return 2
    return 3


def _trace_loops(edges):
    """把有向边接成一条条闭合回路。

    两条回路在同一个角点相遇时（对角相切的虫体像「8」字那样），每到一个岔口都挑**最靠
    右**的那条边走：这样才贴着像素走，不会跨过空白把两块实体连起来。
    """
    loops = []
    used = set()
    for start in sorted(edges):
        for end in edges[start]:
            if (start, end) in used:
                continue
            loop = [start]
            current, incoming = start, None
            while True:
                candidates = [c for c in edges.get(current, ()) if (current, c) not in used]
                if not candidates:
                    break
                if incoming is None or len(candidates) == 1:
                    following = candidates[0]
                else:
                    following = min(candidates, key=lambda c: (
                        _turn_rank(incoming, (c[0] - current[0], c[1] - current[1])), c))
                used.add((current, following))
                incoming = (following[0] - current[0], following[1] - current[1])
                current = following
                if current == start:
                    break
                loop.append(current)
            if current == start and len(loop) >= 4:
                loops.append(loop)
    return loops


def _drop_collinear(points):
    """去掉夹在邻居直线上、不改变形状的中间顶点（长直边只留两端）。"""
    if len(points) < 3:
        return points
    count = len(points)
    keep = []
    for index in range(count):
        previous = points[index - 1]
        current = points[index]
        following = points[(index + 1) % count]
        first = current - previous
        second = following - current
        if abs(first[0] * second[1] - first[1] * second[0]) > 1e-9:
            keep.append(current)
    return np.asarray(keep) if len(keep) >= 3 else points


def instance_contours(mask):
    """一个实例的外轮廓与它内部的孔洞，返回 (outer, holes)。坐标带 .5。

    mask 为空时返回 (None, [])。孔洞就是人工排除区在虫体内部留下的缺口，必须以
    「洞」的形式送进 ImageJ：普通多边形 ROI 没有「这块不算」的表达方式，被排除的
    像素会被重新计入面积与均值，与 Python 后端报出的数对不上。

    mask 是单连通的一块（调用方传进来的就是 `instances == label`），所以描出来的
    回路里面积最大的那条是外轮廓，其余都是洞。
    """
    occupancy = np.asarray(mask, dtype=bool)
    if not occupancy.any():
        return None, []
    loops = [points for points in
             (_drop_collinear(np.asarray(loop, dtype=np.float64) / 2.0)
              for loop in _trace_loops(_boundary_edges(occupancy)))
             if len(points) >= 3]
    if not loops:
        return None, []
    areas = [abs(_signed_area(points)) for points in loops]
    outer_index = int(np.argmax(areas))
    holes = [points for index, points in enumerate(loops) if index != outer_index]
    return loops[outer_index], holes


def largest_polygon(mask):
    """外轮廓。需要在 ImageJ 里保留孔洞时用 instance_contours()。"""
    return instance_contours(mask)[0]


def polygon_roi(points, name):
    roi = ImagejRoi.frompoints(points, name=name)
    roi.roitype = ROI_TYPE.POLYGON
    return roi


# ImageJ 的复合 ROI（ShapeRoi）把形状存成一条路径：三元组 (操作, x, y) 平铺，其中
# 0 = 起点、1 = 连线、4 = 闭合当前子路径；坐标是绝对坐标。孔洞靠**绕行方向与外轮廓
# 相反**来表达（非零填充规则），没有任何标记位。这几个数值是拿 ImageJ 自己写出的带孔
# ROI 读回来的，也验证过 ImageJ 能读回本函数写出的文件，见 VALIDATION_0.4.2.md。
_COMPOSITE_MOVETO = 0.0
_COMPOSITE_LINETO = 1.0
_COMPOSITE_CLOSE = 4.0
# ij.jar 1.54p 的 RoiEncoder 写出的版本号。
_COMPOSITE_VERSION = 228


def _signed_area(points):
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _subpath(values, points):
    for index, (x, y) in enumerate(points):
        values += [_COMPOSITE_MOVETO if index == 0 else _COMPOSITE_LINETO, float(x), float(y)]
    values.append(_COMPOSITE_CLOSE)


def compound_roi(outer, holes, name):
    """外轮廓加孔洞的 ROI；没有孔洞时就是普通多边形 ROI。"""
    if not len(holes):
        return polygon_roi(outer, name)
    outer_points = np.asarray(outer, dtype=np.float64)
    values = []
    _subpath(values, outer_points)
    outer_area = _signed_area(outer_points)
    for hole in holes:
        points = np.asarray(hole, dtype=np.float64)
        if _signed_area(points) * outer_area > 0:
            points = points[::-1]      # 与外轮廓同向就不是洞，是又一圈实心
        _subpath(values, points)
    roi = ImagejRoi()
    roi.version = _COMPOSITE_VERSION
    roi.byteorder = ">"          # roifile 默认小端，ImageJ 写的是大端
    roi.roitype = ROI_TYPE.RECT  # 复合 ROI 的文件头就是矩形，形状另存为附加数据
    roi.name = name
    # 顶点在半整数上，包围盒要往外取整才是真正覆盖到的整数像素范围。
    roi.left = int(np.floor(outer_points[:, 0].min()))
    roi.top = int(np.floor(outer_points[:, 1].min()))
    roi.right = int(np.ceil(outer_points[:, 0].max()))
    roi.bottom = int(np.ceil(outer_points[:, 1].max()))
    roi.multi_coordinates = np.asarray(values, dtype=">f4")
    roi.shape_roi_size = int(roi.multi_coordinates.size)
    return roi


# 背景 ROI 不能是「随便找块空地」：矩形里混进一个虫体像素，或者一块人工排除掉的脏
# 区域，背景均值就被污染，CTCF 能算出 0 甚至负数，而 QC 那一列照样是 PASS。所以候选
# 矩形只允许落在完全干净的位置，一块都找不到时宁可不给背景，由调用方拒绝这张图。
BACKGROUND_MIN_AREA_PIXELS = 200
BACKGROUND_MIN_AREA_FRACTION = 0.001


def _clear_runs(flags):
    """True 的连续段，返回 [(起, 止), ...]（含两端）。"""
    padded = np.r_[False, np.asarray(flags, dtype=bool), False]
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2].tolist(), (edges[1::2] - 1).tolist()))


def _no_rectangle_reaches(region, min_area):
    """区域内确定没有面积达到 min_area 的空矩形时返回 True。

    对每一列求「最长的连续空行段」，取所有列的最大值 tall。任何空矩形的高都不可能
    超过 tall（每一列都得自己空出那么多行），所以 area <= 宽 × tall 是硬上界；连
    宽 × tall 都不够 min_area 时，这一条带确实不必再搜。

    早先这里写的是「空列数 × 空行数」，看着更简单，但一条横穿整幅图的障碍会让空行数
    变成 0，上界跟着变成 0——而障碍上方明明还留着大片干净区域。那条上界虽然成立，
    却粗到会把该救回来的图挡在门外。

    只在算完每列的连续空行段之后才判断，这一步是 O(区域像素)；建直方图的代价已经在
    `_largest_clear_rectangle_exact` 里付过了，所以这里不额外增加复杂度量级。
    """
    clear = ~np.asarray(region, dtype=bool)
    if region.size == 0:
        return True
    tallest = 0
    for column in range(clear.shape[1]):
        for run in _clear_runs(clear[:, column]):
            tallest = max(tallest, run[1] - run[0] + 1)
            if tallest >= min_area:  # 宽至少 1，再算下去也不可能更小
                return False
    return clear.shape[1] * tallest < min_area


def _largest_clear_rectangle(blocked, x0, x1, y0, y1):
    """区域内面积最大的全空矩形，返回 (面积, x0, y0, x1, y1)，没有则 None。

    先把区域按「整列都空」切成若干连续列段，再在每段里找整行都空的连续行段：这样得到
    的矩形一定不含任何被挡下的像素，可以直接当作统计区域。

    注意这只是**一部分**空矩形，不是全部：某一列只要有一个被挡的像素，整列就出局，
    于是横穿条带的障碍上方或下方明明还留着一块干净矩形，这里也找不到。所以它只是快速
    路径，找不出合格结果时由 `_largest_clear_rectangle_exact` 兜底，两个函数返回同一种
    四元组。
    """
    region = blocked[y0:y1 + 1, x0:x1 + 1]
    best = None
    for column_start, column_end in _clear_runs(~region.any(axis=0)):
        rows = region[:, column_start:column_end + 1].any(axis=1)
        for row_start, row_end in _clear_runs(~rows):
            area = (column_end - column_start + 1) * (row_end - row_start + 1)
            if best is None or area > best[0]:
                best = (area, x0 + column_start, y0 + row_start,
                        x0 + column_end, y0 + row_end)
    return best


def _largest_clear_rectangle_exact(blocked, x0, x1, y0, y1):
    """区域内面积最大的全空矩形，返回 (面积, x0, y0, x1, y1)，没有则 None。

    同样是「面积最大的全空矩形」，但这次真的是：逐行累加每列连续空了几行，把这一行
    的高度当成直方图，用单调栈求直方图里的最大矩形。这是标准解法，能找出横穿条带的
    障碍**上方或下方**留下的那块干净矩形，而那正是快速路径漏掉的情形。

    比 `_largest_clear_rectangle` 慢（每行都要过一遍栈），所以只在快速路径拿不出合格
    面积时才跑，见 `background_mask_and_polygon`。
    """
    region = np.asarray(blocked[y0:y1 + 1, x0:x1 + 1])
    if region.size == 0:
        return None
    clear = ~region
    width = clear.shape[1]
    heights = np.zeros(width, dtype=np.int64)
    best = None
    for row in range(clear.shape[0]):
        heights = np.where(clear[row], heights + 1, 0)
        # 单调栈求直方图最大矩形：栈里存 (列下标, 高度)，高度递增；遇到更矮的就把
        # 栈顶弹出来结算，结算宽度是「到当前列为止」。
        stack = []
        for column in range(width + 1):
            height = int(heights[column]) if column < width else 0
            start = column
            while stack and stack[-1][1] >= height:
                left, popped = stack.pop()
                area = popped * (column - left)
                if best is None or area > best[0]:
                    best = (area, x0 + left, y0 + row - popped + 1,
                            x0 + column - 1, y0 + row)
                start = left
            if height:
                stack.append((start, height))
    return best


def background_mask_and_polygon(instances, forbidden_mask=None):
    """背景区域，返回 (mask, polygon)。

    mask 与 polygon 是**同一块区域**：polygon 是导出给 ImageJ 的 ROI，mask 是 Python
    侧统计用的像素。两者一旦不一致，同一张图在两种测量后端下就会得到不同的背景值，
    而 CTCF 是拿背景均值算的。这里靠「候选矩形必须完全干净」来保证一致，而不是先在
    矩形上算统计、事后再把脏像素从 mask 里挖掉。

    虫体左侧放不下就试右侧、上方、下方，取可行面积最大的那块；都放不下或干净面积小
    到没有参考价值时返回 (None, None)。
    """
    foreground = np.asarray(instances) > 0
    blocked = foreground
    if forbidden_mask is not None:
        forbidden = np.asarray(forbidden_mask, dtype=bool)
        if forbidden.shape != foreground.shape:
            raise ValueError("forbidden_mask must match instances")
        blocked = foreground | forbidden
    height, width = blocked.shape
    yy, xx = np.nonzero(foreground)
    if len(xx):
        left, right = int(xx.min()), int(xx.max())
        top, bottom = int(yy.min()), int(yy.max())
    else:
        left, right = int(width * 0.3), int(width * 0.7)
        top, bottom = int(height * 0.1), int(height * 0.9)
    margin_x = max(2, int(width * 0.025))
    margin_y = max(2, int(height * 0.025))
    candidates = [
        (max(2, margin_x), left - margin_x, max(2, top), min(height - 3, bottom)),
        (right + margin_x, width - 3, max(2, top), min(height - 3, bottom)),
        (max(2, left), min(width - 3, right), max(2, margin_y), top - margin_y),
        (max(2, left), min(width - 3, right), bottom + margin_y, height - 3),
    ]
    min_area = max(BACKGROUND_MIN_AREA_PIXELS, int(blocked.size * BACKGROUND_MIN_AREA_FRACTION))
    strips = [(sx0, sx1, sy0, sy1) for sx0, sx1, sy0, sy1 in candidates
              if sx1 >= sx0 and sy1 >= sy0]
    best = None
    for strip_x0, strip_x1, strip_y0, strip_y1 in strips:
        found = _largest_clear_rectangle(blocked, strip_x0, strip_x1, strip_y0, strip_y1)
        if found is not None and (best is None or found[0] > best[0]):
            best = found
    # 快速路径只找「整列都空」的矩形，横穿条带的障碍上方留下的那块干净区域它看不见，
    # 于是明明还有合格背景却报「找不到干净背景区域」，整张图被判失败。找不到合格面积
    # 时再跑一次精确搜索兜底——能过的图照样走原路，被误判的图这才有机会救回来。
    if best is None or best[0] < min_area:
        for strip_x0, strip_x1, strip_y0, strip_y1 in strips:
            region = blocked[strip_y0:strip_y1 + 1, strip_x0:strip_x1 + 1]
            # 确定达不到 min_area 就不必跑了：精确搜索每行过一次栈，比快速路径慢。
            if _no_rectangle_reaches(region, min_area):
                continue
            found = _largest_clear_rectangle_exact(blocked, strip_x0, strip_x1, strip_y0, strip_y1)
            if found is not None and (best is None or found[0] > best[0]):
                best = found
    if best is None:
        return None, None
    area, x0, y0, x1, y1 = best
    if area < min_area:
        return None, None
    mask = np.zeros(blocked.shape, dtype=bool)
    mask[y0:y1 + 1, x0:x1 + 1] = True
    # 与虫体 ROI 同一套规则：导出的矩形画在像素**边界**上（x0-0.5 到 x1+0.5），
    # ImageJ 采样到的才是 mask 里那 x0..x1 / y0..y1 这些像素，不多不少。
    polygon = np.asarray([(x0 - 0.5, y0 - 0.5), (x1 + 0.5, y0 - 0.5),
                          (x1 + 0.5, y1 + 0.5), (x0 - 0.5, y1 + 0.5)], dtype=np.float64)
    return mask, polygon


def statistics(raw, mask):
    values = raw[mask]
    if not len(values):
        return dict(area=0, mean=np.nan, minimum=np.nan, maximum=np.nan,
                    median=np.nan, raw_sum=np.nan)
    return dict(area=int(len(values)), mean=float(values.mean()), minimum=float(values.min()),
                maximum=float(values.max()), median=float(np.median(values)),
                raw_sum=float(values.astype(np.float64).sum()))


def make_overlay(normalized, instances, background_mask, qc_text,
                 attention=False, needs_review=False):
    rgb = np.dstack((normalized, normalized, normalized)).astype(np.uint8)
    colored = colorize_labels(instances)
    foreground = instances > 0
    rgb[foreground] = (0.55 * rgb[foreground] + 0.45 * colored[foreground]).astype(np.uint8)
    edges = np.zeros(instances.shape, dtype=bool)
    for label in np.unique(instances):
        if label:
            mask = instances == label
            edges |= mask & ~ndimage.binary_erosion(mask)
    rgb[edges] = (255, 255, 255)
    bg_edge = background_mask & ~ndimage.binary_erosion(background_mask)
    rgb[bg_edge] = (255, 0, 0)
    canvas = Image.fromarray(rgb)
    draw = ImageDraw.Draw(canvas)
    if needs_review:
        draw.rectangle((1, 1, canvas.width - 2, canvas.height - 2),
                       outline=(255, 72, 72), width=max(4, canvas.width // 512))
    elif attention:
        draw.rectangle((1, 1, canvas.width - 2, canvas.height - 2),
                       outline=(255, 210, 32), width=max(4, canvas.width // 512))
    draw.rectangle((4, 4, min(canvas.width - 4, 620), 35), fill=(0, 0, 0))
    draw.text((10, 10), qc_text, fill=(255, 255, 255))
    return canvas


def qc_status_for_count(count, expected_count):
    """Only a final worm-count mismatch is a manual-review condition."""
    return "PASS" if int(count) == int(expected_count) else "REVIEW_COUNT_MISMATCH"


def process_image(tiff_path, output_dir, model, device, image_size=IMAGE_SIZE,
                  interior_threshold=None, erosion_iterations=0,
                  min_area_fraction=0.005, min_height_fraction=0.10, max_instances=12,
                  standard_count=10, allowed_count_min=9, allowed_count_max=11,
                  tip_model=None, tip_patch_size=192,
                  tip_probability_threshold=0.40, tip_replace_fraction=0.14,
                  normalization_mode="legacy", use_filename_count=True,
                  shape_refinement=False, low_clarity_split=False,
                  manual_head_annotation=False, segment_selection=False,
                  segment_start=0.0, segment_end=1.0,
                  measurement_backend="python"):
    if measurement_backend not in ("python", "imagej"):
        raise ValueError("measurement_backend must be 'python' or 'imagej'")
    manual_head_annotation = bool(manual_head_annotation or segment_selection)
    image = Image.open(tiff_path)
    raw = image_array(image)
    stem = os.path.splitext(os.path.basename(tiff_path))[0]
    # Everything this function writes lands in one of these two. Created here as
    # well as in run_gui_batch because process_image is also called directly,
    # by the tests and by whoever wants a single image.
    measurements_dir, other_dir = ensure_output_layout(output_dir)
    marked_count = expected_count_from_filename(stem) if use_filename_count else None
    image_max_instances = marked_count if marked_count is not None else max_instances
    arrows = []
    boundary_guides = []
    exclusion_regions = []
    manual_exclusion_mask = None
    if manual_head_annotation:
        from manual_head_annotation import (
            exclusion_mask, load_image_annotations, load_image_boundary_guides,
            load_image_exclusion_regions)
        current_size = (raw.shape[1], raw.shape[0])
        arrows = load_image_annotations(tiff_path, current_size=current_size)
        boundary_guides = load_image_boundary_guides(tiff_path, current_size=current_size)
        exclusion_regions = load_image_exclusion_regions(tiff_path, current_size=current_size)
        if exclusion_regions:
            manual_exclusion_mask = exclusion_mask(raw.shape[:2], exclusion_regions)
    split_reports = []
    prediction = predict_raw(
        model, raw, device, image_size=image_size,
        interior_threshold=interior_threshold,
        erosion_iterations=erosion_iterations,
        min_area_fraction=min_area_fraction,
        min_height_fraction=min_height_fraction, max_instances=image_max_instances,
        tip_model=tip_model, tip_patch_size=tip_patch_size,
        tip_probability_threshold=tip_probability_threshold,
        tip_replace_fraction=tip_replace_fraction,
        normalization_mode=normalization_mode,
        return_foreground_probability=shape_refinement,
        low_clarity_split=low_clarity_split, split_reports=split_reports,
        split_count_limit=marked_count if marked_count is not None else standard_count,
        exclusion_mask=manual_exclusion_mask)
    if shape_refinement:
        instances, normalized, _, foreground_probability = prediction
    else:
        instances, normalized, _ = prediction

    manual_split_reports = []
    manual_split_report_path = ""
    if manual_head_annotation:
        from manual_head_annotation import apply_manual_annotation_partitions
        # Manual arrows and boundary guides are independent stages. The helper
        # always applies head-based coarse partitioning before boundary refinement.
        instances, manual_split_reports = apply_manual_annotation_partitions(
            instances, arrows, image=normalized, max_instances=max_instances,
            boundary_guides=boundary_guides)
        manual_split_report_path = os.path.join(
            other_dir, stem + "_manual_split_qc.csv")

    shape_reports = []
    shape_report_path = ""
    if shape_refinement:
        # 独立后处理模块：不参与神经网络推理，不修改模型权重。
        from worm_shape_refiner import refine_instance_shapes, write_shape_report
        coarse_count = int(instances.max())
        expected_for_repair = marked_count if marked_count is not None else standard_count
        count_allows_repair = coarse_count == expected_for_repair
        # A gross count error means instance identities are unreliable.  Keep
        # shape diagnostics, but omit probability evidence so the refiner is
        # detector-only and cannot alter quantitative ROIs.
        instances, shape_reports = refine_instance_shapes(
            instances,
            foreground_probability=(foreground_probability if count_allows_repair else None))
        shape_report_path = os.path.join(other_dir, stem + "_shape_qc.csv")
    if manual_exclusion_mask is not None:
        # Shape repair and guided splitting must never grow an ROI back into a
        # region that the experimenter explicitly excluded.
        instances = _apply_exclusion_mask_to_instances(instances, manual_exclusion_mask)
    count = int(instances.max())
    head_matches = []
    unmatched_head_arrows = 0
    matched_head_count = 0
    head_annotation_complete = False
    head_report_path = ""
    if manual_head_annotation:
        from manual_head_annotation import match_head_directions
        head_matches, unmatched_head_arrows = match_head_directions(instances, arrows)
        matched_head_count = sum(match.status == "MATCHED" for match in head_matches)
        head_annotation_complete = (
            count > 0 and matched_head_count == count and unmatched_head_arrows == 0)
        head_report_path = os.path.join(other_dir, stem + "_head_annotations.csv")
    whole_instances = instances.copy()
    segment_reports = []
    segment_report_path = ""
    if segment_selection:
        if not head_annotation_complete:
            raise ValueError(
                "%s: centerline segment selection requires exactly one valid head arrow "
                "for every detected worm (matched=%d, worms=%d, unmatched_arrows=%d)" %
                (os.path.basename(tiff_path), matched_head_count, count, unmatched_head_arrows))
        from worm_segment_selector import select_instance_segments
        try:
            instances, segment_reports = select_instance_segments(
                whole_instances, head_matches, segment_start, segment_end)
        except (ValueError, RuntimeError) as exc:
            raise ValueError("%s: %s" % (os.path.basename(tiff_path), exc)) from exc
        segment_report_path = os.path.join(other_dir, stem + "_segment_qc.csv")
    background_mask, background_polygon = background_mask_and_polygon(
        whole_instances, forbidden_mask=manual_exclusion_mask)
    if background_mask is None:
        # 没有一块干净的地方可以当背景。与其报一个被虫体污染的均值（CTCF 会跟着错，
        # 而 QC 列看不出来），不如把这张图判为失败，让用户自己看排除区画得对不对。
        raise ValueError(
            "%s: 找不到干净背景区域（候选位置都被虫体或人工排除区占满，或剩下不足 %d 像素）"
            "；请检查人工排除区，或改用虫体周围留有余地的图像"
            % (os.path.basename(tiff_path), BACKGROUND_MIN_AREA_PIXELS))
    background = (statistics(raw, background_mask)
                  if measurement_backend == "python" else None)
    rois, rows, areas = [], [], []
    for label in range(1, count + 1):
        mask = instances == label
        polygon, holes = instance_contours(mask)
        if polygon is None:
            continue
        name = "%02d_worm" % label
        rois.append(compound_roi(polygon, holes, name))
        if measurement_backend == "python":
            values = statistics(raw, mask)
            areas.append(values["area"])
            corrected_mean = values["mean"] - background["mean"]
            corrected_total = values["raw_sum"] - values["area"] * background["mean"]
            rows.append(measurement_row({
                "": label, "Label": name,
                "Area": values["area"], "Mean": values["mean"],
                "Min": values["minimum"], "Max": values["maximum"],
                "IntDen": values["raw_sum"], "Median": values["median"],
                "RawIntDen": values["raw_sum"],
                "BackgroundCorrectedMean": corrected_mean,
                "CTCF": corrected_total, "Type": "worm"}))
    rois.append(polygon_roi(background_polygon, "99_background"))
    if measurement_backend == "python":
        rows.append(measurement_row({
            "": len(rows) + 1, "Label": "99_background",
            "Area": background["area"], "Mean": background["mean"],
            "Min": background["minimum"], "Max": background["maximum"],
            "IntDen": background["raw_sum"], "Median": background["median"],
            "RawIntDen": background["raw_sum"],
            "BackgroundCorrectedMean": 0.0, "CTCF": 0.0, "Type": "background"}))
    expected_count = marked_count if marked_count is not None else standard_count
    qc = qc_status_for_count(count, expected_count)
    shape_refined_count = sum(int(report.repaired) for report in shape_reports)
    shape_review_count = sum(int(report.needs_review) for report in shape_reports)
    roi_path = os.path.join(other_dir, stem + "_RoiSet.zip")
    _atomic_write(roi_path, lambda temporary: roiwrite(temporary, rois, mode="w"))
    csv_path = os.path.join(measurements_dir, stem + "_measurements.csv")
    if measurement_backend == "python":
        _write_csv_atomic(csv_path, PYTHON_MEASUREMENT_HEADER, rows)
    else:
        # Never leave a stale measurement table next to this run's ROIs: the
        # Python backend's from an earlier run of it, and the plug-in's from an
        # earlier ImageJ run -- the latter is written after this function
        # returns, so if the plug-in never gets to measure this image, whatever
        # is here is all anyone will find beside these ROIs.
        _remove_stale_report(csv_path, PYTHON_MEASUREMENT_HEADER)
        _remove_stale_imagej_report(csv_path)
    shape_text = (" shape_fixed=%d shape_review=%d" %
                  (shape_refined_count, shape_review_count)) if shape_refinement else ""
    if split_reports:
        shape_text += " split=%d" % len(split_reports)
    if manual_head_annotation:
        shape_text += " heads=%d/%d" % (matched_head_count, count)
        if boundary_guides:
            shape_text += " boundaries=%d" % len(boundary_guides)
        if exclusion_regions:
            shape_text += " exclusions=%d" % len(exclusion_regions)
    if manual_split_reports:
        shape_text += " manual_split=%d" % len(manual_split_reports)
    if segment_selection:
        shape_text += " segment=%.2f-%.2f" % (segment_start, segment_end)
    ai_attention = bool(shape_review_count or shape_refined_count or split_reports)
    overlay = make_overlay(
        normalized, instances, background_mask,
        "%s   worms=%d expected=%d%s" %
        (qc, count, expected_count, shape_text),
        attention=ai_attention, needs_review=qc.startswith("REVIEW"))
    if segment_selection:
        from worm_segment_selector import draw_segment_guides
        overlay = draw_segment_guides(
            overlay, whole_instances, head_matches, segment_start, segment_end)
    if manual_head_annotation:
        from manual_head_annotation import (
            draw_boundary_guides, draw_exclusion_regions, draw_head_arrows)
        overlay = draw_head_arrows(overlay, head_matches)
        overlay = draw_boundary_guides(overlay, boundary_guides)
        overlay = draw_exclusion_regions(overlay, exclusion_regions)
    qc_path = os.path.join(other_dir, stem + "_QC.png")
    _atomic_write(qc_path, lambda temporary: overlay.save(temporary))
    if shape_refinement:
        _atomic_write(
            shape_report_path,
            lambda temporary: write_shape_report(temporary, shape_reports))
    else:
        # Do not leave a report from an older run that used shape refinement.
        from worm_shape_refiner import ShapeReport
        _remove_stale_report(os.path.join(other_dir, stem + "_shape_qc.csv"),
                             ShapeReport.__dataclass_fields__)
    split_report_path = ""
    if low_clarity_split:
        from low_clarity_splitter import SplitReport
        fields = list(SplitReport.__dataclass_fields__)
        split_report_path = os.path.join(other_dir, stem + "_split_qc.csv")
        _write_csv_atomic(split_report_path, fields,
                          [[asdict(report)[field] for field in fields] for report in split_reports])
    else:
        # Do not leave a split report from a run that had low-clarity splitting
        # turned on. Its siblings -- shape, head annotation, segment selection --
        # all clear theirs; this one kept the stale file, and a per-image report
        # that describes a step this run did not perform is worse than absent,
        # because nothing in it says which run wrote it.
        from low_clarity_splitter import SplitReport
        _remove_stale_report(os.path.join(other_dir, stem + "_split_qc.csv"),
                             list(SplitReport.__dataclass_fields__))
    if manual_head_annotation:
        from manual_head_annotation import HEAD_REPORT_FIELDS, MANUAL_SPLIT_REPORT_FIELDS
        _write_csv_atomic(
            head_report_path, HEAD_REPORT_FIELDS,
            [[asdict(match)[field] for field in HEAD_REPORT_FIELDS] for match in head_matches])
        _write_csv_atomic(
            manual_split_report_path, MANUAL_SPLIT_REPORT_FIELDS,
            [[asdict(report)[field] for field in MANUAL_SPLIT_REPORT_FIELDS]
             for report in manual_split_reports])
    else:
        from manual_head_annotation import HEAD_REPORT_FIELDS, MANUAL_SPLIT_REPORT_FIELDS
        for stale_name, stale_header in (
                (stem + "_head_annotations.csv", HEAD_REPORT_FIELDS),
                (stem + "_manual_split_qc.csv", MANUAL_SPLIT_REPORT_FIELDS)):
            _remove_stale_report(os.path.join(other_dir, stale_name), stale_header)
    if segment_selection:
        from worm_segment_selector import SEGMENT_REPORT_FIELDS
        _write_csv_atomic(
            segment_report_path, SEGMENT_REPORT_FIELDS,
            [[asdict(report)[field] for field in SEGMENT_REPORT_FIELDS]
             for report in segment_reports])
    else:
        from worm_segment_selector import SEGMENT_REPORT_FIELDS
        _remove_stale_report(os.path.join(other_dir, stem + "_segment_qc.csv"),
                             SEGMENT_REPORT_FIELDS)
    mean_segment_fraction = (float(np.mean(
        [report.selected_area_fraction for report in segment_reports]))
        if segment_reports else (1.0 if segment_selection else 0.0))
    return [os.path.basename(tiff_path), count, expected_count, qc,
            background["mean"] if background is not None else "", roi_path, csv_path,
            shape_refined_count, shape_review_count, shape_report_path,
            len(split_reports), split_report_path, bool(low_clarity_split),
            bool(manual_head_annotation), matched_head_count,
            bool(head_annotation_complete), head_report_path,
            len(manual_split_reports), manual_split_report_path,
            len(boundary_guides), len(exclusion_regions),
            int(manual_exclusion_mask.sum()) if manual_exclusion_mask is not None else 0,
            bool(segment_selection), float(segment_start), float(segment_end),
            len(segment_reports), mean_segment_fraction, segment_report_path,
            measurement_backend]


def run_gui_batch(input_folder, output_dir, checkpoint_path, tip_checkpoint_path,
                  standard_count=10, allowed_count_min=9, allowed_count_max=11,
                  ignore_filename_count=True, shape_refinement=False,
                  disable_tip_refinement=False, tip_threshold=None, tip_fraction=None,
                  interior_threshold=None, erosion=None, min_area=None, min_height=None,
                  max_instances=None, on_status=None, should_cancel=None,
                  low_clarity_split=None, manual_head_annotation=False,
                  segment_selection=False, segment_start=0.0, segment_end=1.0,
                  input_paths=None, measurement_backend="python",
                  on_image_failed=None):
    """进程内批量处理,供 GUI 直接调用(不依赖外部 python 或子进程)。

    加载整虫与头尾模型,遍历 input_folder 下的 TIFF,逐张调用 process_image,
    把每张图的结果以 "<图名>: worms=N, expected=N, <QC>" 传给 on_status 回调,
    最后写入 batch_summary.csv 并返回成功处理的结果列表。

    单张图失败不会中断整批:该图以 "Failed_<原文件名>" 记入日志与
    batch_summary.csv,并通过 on_image_failed(图名, 原因) 通知调用方,其余图照常
    处理。若整批都无法开始(格式不支持、输出重名等),则在处理前直接抛异常。

    与命令行的 main() 共享同一套推理与后处理细节;所有可选参数为 None 时，
    采用与 checkpoint 中记录一致的默认值。
    """
    def report(text, *args):
        if on_status is not None:
            on_status(text % args if args else text)

    input_folder = os.path.abspath(input_folder)
    segment_start, segment_end = float(segment_start), float(segment_end)
    if segment_selection and not (0.0 <= segment_start < segment_end <= 1.0):
        raise ValueError("segment range must satisfy 0 <= start < end <= 1")
    manual_head_annotation = bool(manual_head_annotation or segment_selection)
    output_dir = os.path.abspath(output_dir or os.path.join(input_folder, "_auto_roi"))
    _reject_output_equal_to_input(input_folder, output_dir)
    # Collect the work list before touching the filesystem, so a folder that
    # cannot be processed at all is refused without creating even an empty
    # results directory. Neither branch below reads output_dir.
    if input_paths is None:
        paths = []
        for pattern in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
            paths.extend(glob.glob(os.path.join(input_folder, pattern)))
        paths = sorted(set(paths))
    else:
        paths = sorted(set(os.path.abspath(path) for path in input_paths))
        invalid = [path for path in paths if (not os.path.isfile(path) or
                   os.path.splitext(path)[1].lower() not in (".tif", ".tiff"))]
        if invalid:
            raise FileNotFoundError("Invalid TIFF input: " + invalid[0])
    if not paths:
        raise FileNotFoundError("No TIFF images found in: " + input_folder)
    format_notes = _reject_unsupported_images(paths)
    _reject_output_name_collisions(paths)
    # Both subfolders are created up front, not on first use: the ImageJ side is
    # handed the output folder and looks for the ROI sets and the QC overlays in
    # other/ straight away, and it is much easier to see an empty folder than to
    # work out why a file was never written.
    ensure_output_layout(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device_name = torch.cuda.get_device_name(device)
    else:
        device_name = "CPU"
    precision = "FP32"
    run_id = uuid.uuid4().hex
    worm_model_sha256 = _sha256_file(checkpoint_path)
    # GUI selects this explicitly. CLI defaults to the known low-mode model
    # fingerprints, so the high-mode model never enters the new branch.
    if low_clarity_split is None:
        low_clarity_split = worm_model_sha256 in LOW_CLARITY_MODEL_SHA256S
    tip_model_sha256 = ("" if disable_tip_refinement else
                        _sha256_file(tip_checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = WormUNet(base=int(checkpoint.get("base", 16))).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    tip_model = None
    tip_patch_size = 192
    tip_probability_threshold = 0.40
    tip_replace_fraction = 0.14
    if not disable_tip_refinement:
        if not os.path.isfile(tip_checkpoint_path):
            raise FileNotFoundError("Tip refiner checkpoint not found: " + tip_checkpoint_path)
        tip_checkpoint = torch.load(tip_checkpoint_path, map_location=device)
        tip_model = WormUNet(
            base=int(tip_checkpoint.get("base", 12)),
            in_channels=int(tip_checkpoint.get("in_channels", 2)),
            out_channels=int(tip_checkpoint.get("out_channels", 2))).to(device)
        tip_model.load_state_dict(tip_checkpoint["model_state"])
        tip_model.eval()
        tip_patch_size = int(tip_checkpoint.get("patch_size", 192))
        tip_probability_threshold = (
            tip_threshold if tip_threshold is not None else
            float(tip_checkpoint.get("probability_threshold", 0.40)))
        tip_replace_fraction = (
            tip_fraction if tip_fraction is not None else
            float(tip_checkpoint.get("replace_fraction", 0.14)))
    image_size = int(checkpoint.get("image_size", IMAGE_SIZE))
    normalization_mode = str(checkpoint.get("normalization_mode", "legacy"))
    checkpoint_has_threshold = "postprocess_interior_threshold" in checkpoint
    interior_threshold = (interior_threshold if interior_threshold is not None else
                          checkpoint.get("postprocess_interior_threshold"))
    erosion_iterations = (erosion if erosion is not None else
                          checkpoint.get("postprocess_erosion"))
    min_area_fraction = (min_area if min_area is not None else
                         float(checkpoint.get("postprocess_min_area", 0.005)))
    min_height_fraction = (min_height if min_height is not None else
                           float(checkpoint.get("postprocess_min_height", 0.10)))
    max_instances = (max_instances if max_instances is not None else
                     int(checkpoint.get("postprocess_max_instances", 12)))
    standard_count = (standard_count if standard_count is not None else
                      int(checkpoint.get("standard_count", 10)))
    allowed_counts = checkpoint.get("allowed_counts", [standard_count - 1, standard_count, standard_count + 1])
    allowed_count_min = (allowed_count_min if allowed_count_min is not None else
                         int(min(allowed_counts)))
    allowed_count_max = (allowed_count_max if allowed_count_max is not None else
                         int(max(allowed_counts)))
    if image_size >= 768:
        if interior_threshold is None and not checkpoint_has_threshold:
            interior_threshold = 0.35
        erosion_iterations = 3 if erosion_iterations is None else erosion_iterations
    else:
        erosion_iterations = 0 if erosion_iterations is None else erosion_iterations
    summary_path = os.path.join(output_dir, OTHER_DIRNAME, "batch_summary.csv")
    summary_header = [
        "image", "worm_count", "expected_count", "qc_status",
        "background_mean", "roi_zip", "measurements_csv",
        "shape_refined_count", "shape_review_count", "shape_qc_csv",
        "low_clarity_split_count", "split_qc_csv", "low_clarity_split_enabled",
        "manual_head_annotation_enabled", "head_annotation_count",
        "head_annotation_complete", "head_annotation_csv",
        "manual_head_split_count", "manual_head_split_qc_csv",
        "manual_boundary_guide_count",
        "manual_exclusion_region_count", "manual_exclusion_area_px",
        "segment_selection_enabled", "segment_start_fraction", "segment_end_fraction",
        "segment_selection_count", "segment_mean_area_fraction", "segment_qc_csv",
        "measurement_backend",
    ]
    # The run columns come last so a reader keyed by column name (the GUI, the
    # export check) is unaffected by a new one. "error" is the very last of them,
    # and only a failed image ever fills it in.
    summary_header = summary_header + list(RUN_COLUMNS)
    # Replace a summary from an older run before exposing any current QC files.
    # It is refreshed after every completed image, so a later error still leaves
    # an internally consistent partial manifest.
    _write_csv_atomic(summary_path, summary_header, [])
    report("Processing: %s", input_folder)
    for note in format_notes:
        report("%s", note)
    report("Device: %s (%s)", device_name, precision)
    if low_clarity_split:
        report("Low-clarity head-gap separation: enabled")
    if manual_head_annotation:
        report("Manual arrows, boundaries and exclusion regions: enabled "
               "(reading per-image .autoworm.json or %s)",
               os.path.join(input_folder, "_manual_head_annotations.json"))
    if segment_selection:
        report("Centerline segment selection: enabled (head=0, tail=1, range %.2f-%.2f)",
               segment_start, segment_end)
    report("Run ID: %s", run_id)
    results = []
    # Everything that goes into batch_summary.csv, failures included; results is
    # only what succeeded. Keeping them apart means a failure stays in the table
    # when the next image rewrites it.
    rows = []
    failures = []
    cancelled = False
    total = len(paths)
    for index, path in enumerate(paths, 1):
        if should_cancel is not None and should_cancel():
            cancelled = True
            break
        name = os.path.basename(path)
        try:
            result = process_image(
                path, output_dir, model, device, image_size=image_size,
                interior_threshold=interior_threshold,
                erosion_iterations=erosion_iterations,
                min_area_fraction=min_area_fraction,
                min_height_fraction=min_height_fraction, max_instances=max_instances or None,
                standard_count=standard_count,
                allowed_count_min=allowed_count_min,
                allowed_count_max=allowed_count_max,
                tip_model=tip_model, tip_patch_size=tip_patch_size,
                tip_probability_threshold=tip_probability_threshold,
                tip_replace_fraction=tip_replace_fraction,
                normalization_mode=normalization_mode,
                use_filename_count=not ignore_filename_count,
                shape_refinement=shape_refinement,
                low_clarity_split=low_clarity_split,
                manual_head_annotation=manual_head_annotation,
                segment_selection=segment_selection,
                segment_start=segment_start, segment_end=segment_end,
                measurement_backend=measurement_backend)
        except Exception as exc:
            # One pathological image must not throw away the work already done on
            # the rest of the folder. The image is named "Failed_<name>" in the
            # log and in batch_summary.csv so it stays identifiable, and it gets
            # no ROI to measure -- including the ROI set it had already written
            # before it failed, which is removed here so the two statements stay
            # true together.
            detail = "%s: %s" % (type(exc).__name__, exc)
            failures.append((name, detail))
            rows.append(_failure_row(summary_header, name, detail, run_id, device_name))
            report("Failed_%s: ERROR %s", name, detail)
            stuck = _discard_partial_outputs(output_dir, os.path.splitext(name)[0])
            if stuck:
                # Locked by a viewer, most likely. Worth saying out loud: these
                # files will still be measured by ImageJ.
                report("  Could not remove partial results for %s: %s",
                       name, "; ".join(stuck))
            if on_image_failed is not None:
                on_image_failed(name, detail)
        else:
            # The empty last entry is the error column, which a success leaves blank.
            run_values = (run_id, device_name, precision, worm_model_sha256,
                          tip_model_sha256, SOFTWARE_VERSION, "")
            rows.append(_success_row(summary_header, result, run_values))
            # results is what callers expect from before the summary existed:
            # process_image's own values followed by the run metadata.
            results.append(result + list(run_values[:-1]))
            report("%s: worms=%d, expected=%d, %s",
                   result[0], result[1], result[2], result[3])
        _write_csv_atomic(summary_path, summary_header, rows)
        report("Progress: %d/%d (%d%%)", index, total, index * 100 // total)
    _write_csv_atomic(summary_path, summary_header, rows)
    if failures:
        report("Failed images (%d): %s", len(failures),
               ", ".join("Failed_" + name for name, _ in failures))
    if cancelled:
        report("Cancelled: completed %d of %d images", len(rows), total)
    report("Output: %s", output_dir)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Automatically segment packed C. elegans TIFF images and export ImageJ ROIs.")
    parser.add_argument("input_folder")
    parser.add_argument("--checkpoint", default=os.path.join(
        os.path.dirname(__file__), "样例1", "_combined_training_v020",
        "worm_unet_multidomain_final", "best_model.pt"))
    parser.add_argument("--tip-checkpoint", default=os.path.join(
        os.path.dirname(__file__), "样例1", "_combined_training_v020",
        "tip_refiner_multidomain_final", "best_model.pt"))
    parser.add_argument("--disable-tip-refinement", action="store_true")
    parser.add_argument("--tip-threshold", type=float, default=None)
    parser.add_argument("--tip-fraction", type=float, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--interior-threshold", type=float, default=None)
    parser.add_argument("--erosion", type=int, default=None)
    parser.add_argument("--min-area", type=float, default=None)
    parser.add_argument("--min-height", type=float, default=None)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--standard-count", type=int, default=None,
                        help="normal worm count for QC; segmentation itself is not forced to this count")
    parser.add_argument("--allowed-count-min", type=int, default=None)
    parser.add_argument("--allowed-count-max", type=int, default=None)
    parser.add_argument(
        "--ignore-filename-count", action="store_true",
        help="use standard-count and its allowed range even when the filename contains a count marker")
    parser.add_argument(
        "--enable-shape-refinement", action="store_true",
        help="run the independent contour quality and conservative shape-refinement postprocessor")
    parser.add_argument(
        "--disable-low-clarity-split", action="store_true",
        help="disable head-gap separation for low-mode baseline comparisons")
    parser.add_argument(
        "--enable-manual-head-annotation", action="store_true",
        help="read per-image manual arrows, boundaries and exclusions from the annotation JSON")
    parser.add_argument(
        "--enable-segment-selection", action="store_true",
        help="limit each ROI to a centerline range; requires one manual head arrow per worm")
    parser.add_argument("--segment-start", type=float, default=0.0,
                        help="selected start on the head-to-tail centerline, from 0 to 1")
    parser.add_argument("--segment-end", type=float, default=1.0,
                        help="selected end on the head-to-tail centerline, from 0 to 1")
    args = parser.parse_args()
    output_dir = os.path.abspath(args.output or os.path.join(os.path.abspath(args.input_folder), "_auto_roi"))
    run_gui_batch(
        os.path.abspath(args.input_folder), output_dir,
        args.checkpoint, args.tip_checkpoint,
        standard_count=args.standard_count,
        allowed_count_min=args.allowed_count_min,
        allowed_count_max=args.allowed_count_max,
        ignore_filename_count=args.ignore_filename_count,
        shape_refinement=args.enable_shape_refinement,
        low_clarity_split=False if args.disable_low_clarity_split else None,
        manual_head_annotation=args.enable_manual_head_annotation,
        segment_selection=args.enable_segment_selection,
        segment_start=args.segment_start,
        segment_end=args.segment_end,
        disable_tip_refinement=args.disable_tip_refinement,
        tip_threshold=args.tip_threshold,
        tip_fraction=args.tip_fraction,
        interior_threshold=args.interior_threshold,
        erosion=args.erosion,
        min_area=args.min_area,
        min_height=args.min_height,
        max_instances=args.max_instances,
        on_status=print)


if __name__ == "__main__":
    main()
