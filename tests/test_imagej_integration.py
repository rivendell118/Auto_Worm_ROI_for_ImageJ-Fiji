import base64
import contextlib
import csv
import io
import json
import os
import queue
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from batch_worm_roi import (IMAGEJ_MEASUREMENT_COLUMNS, IMAGE_OUTPUT_FILES,
                            MEASUREMENTS_DIRNAME, OTHER_DIRNAME,
                            PYTHON_MEASUREMENT_HEADER, RUN_COLUMNS, SOFTWARE_VERSION,
                            BrightfieldRoi, _reject_unsupported_images,
                            _remove_stale_imagej_report, _remove_stale_report,
                            _success_row, _write_csv_atomic, check_image,
                            find_stem_collisions, measurement_row, process_image,
                            run_gui_batch)
import worm_roi_gui
from inspect_roi_dataset import image_array
from manual_head_annotation import (HEAD_REPORT_FIELDS, MANUAL_SPLIT_REPORT_FIELDS,
                                    enhanced_tiff_rgb, load_image_annotations,
                                    load_image_boundary_guides,
                                    load_image_exclusion_regions)
from worm_segment_selector import SEGMENT_REPORT_FIELDS
from worm_shape_refiner import ShapeReport, write_shape_report


class _FakeVar:
    """Stand-in for a tk variable in the GUI stubs: only .get() is ever called."""

    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


def _synthetic_case(folder: Path):
    """A source image plus the labels process_image would have predicted."""
    image = folder / "one.tif"
    raw = np.full((64, 64), 10, dtype=np.uint16)
    raw[15:50, 25:40] = 100
    Image.fromarray(raw).save(image)
    labels = np.zeros((64, 64), np.uint16)
    labels[15:50, 25:40] = 1
    return image, raw, labels


class ImageJIntegrationTests(unittest.TestCase):
    def test_per_image_imagej_sidecar_is_loaded(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            image = folder / "one.tif"
            Image.fromarray(np.zeros((20, 30), np.uint8)).save(image)
            sidecar = {
                "schema_version": 3, "width": 30, "height": 20,
                "arrows": [{"tail_x": 10, "tail_y": 12, "head_x": 10, "head_y": 2}],
                "boundaries": [{"points": [[2, 2], [8, 8]]}],
                "exclusions": [{"points": [[20, 2], [28, 2], [28, 10], [20, 10]]}],
            }
            Path(str(image) + ".autoworm.json").write_text(
                json.dumps(sidecar), encoding="utf-8")
            self.assertEqual(len(load_image_annotations(image)), 1)
            self.assertEqual(len(load_image_boundary_guides(image)), 1)
            self.assertEqual(len(load_image_exclusion_regions(image)), 1)

    def test_imagej_backend_exports_roi_without_python_measurements(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            image, raw, labels = _synthetic_case(folder)
            output = folder / "out"
            output.mkdir()

            with patch("batch_worm_roi.predict_raw",
                       return_value=(labels, raw.astype(np.uint8), None)):
                result = process_image(
                    image, output, model=None, device=None, standard_count=1,
                    measurement_backend="imagej")
            # measurement_backend is the last data column; the three brightfield
            # ones sit after it, so it is counted from the end.
            self.assertEqual(result[-4], "imagej")
            self.assertTrue((output / OTHER_DIRNAME / "one_RoiSet.zip").is_file())
            self.assertFalse((output / MEASUREMENTS_DIRNAME / "one_measurements.csv").exists())


class OutputFolderSafetyTests(unittest.TestCase):
    """Results must never be written into the folder that holds the sources."""

    def test_output_folder_equal_to_input_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            with self.assertRaises(ValueError) as caught:
                run_gui_batch(str(folder), str(folder), "worm.pt", "tip.pt")
            self.assertIn("must differ from the input folder", str(caught.exception))

    def test_equal_folders_are_detected_through_case_and_relative_spelling(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            nested = folder / "images"
            nested.mkdir()
            for spelled in (str(nested).upper(),
                            str(nested / ".." / "images"),
                            str(nested) + os.sep):
                with self.assertRaises(ValueError):
                    run_gui_batch(str(nested), spelled, "worm.pt", "tip.pt")

    def test_default_output_folder_is_still_accepted(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            # The default (_auto_roi inside the image folder) must not trip the
            # guard; it fails later on the missing checkpoint, which proves the
            # guard itself let it through.
            with self.assertRaises(FileNotFoundError):
                run_gui_batch(str(folder), None, "worm.pt", "tip.pt")

    def test_stale_measurement_table_is_removed_when_switching_backends(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            image, raw, labels = _synthetic_case(folder)
            output = folder / "out"
            output.mkdir()
            table = output / MEASUREMENTS_DIRNAME / "one_measurements.csv"

            with patch("batch_worm_roi.predict_raw",
                       return_value=(labels, raw.astype(np.uint8), None)):
                process_image(image, output, model=None, device=None,
                              standard_count=1, measurement_backend="python")
                self.assertTrue(table.is_file())
                process_image(image, output, model=None, device=None,
                              standard_count=1, measurement_backend="imagej")
            self.assertFalse(table.exists())

    def test_foreign_files_sharing_generated_names_are_kept(self):
        # The output folder can legitimately hold the experimenter's own files,
        # including tables exported from ImageJ that share a generated name.
        # None of them may be deleted just because the name matches.
        foreign = {
            "one_measurements.csv": "Label\tArea\tMean\n1\t10\t2.5\n",
            "one_shape_qc.csv": "Label\tRepaired\n1\t0\n",
            "one_head_annotations.csv": "Label\tStatus\n1\tMATCHED\n",
            "one_manual_split_qc.csv": "Label\tMethod\n1\tmanual_head_coarse\n",
            "one_segment_qc.csv": "Label\tFraction\n1\t0.5\n",
        }
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            image, raw, labels = _synthetic_case(folder)
            output = folder / "out"
            output.mkdir()
            # Placed in whichever subfolder the program would write that name to,
            # taken from the program's own mapping -- a lookup by hand here would
            # let this test keep passing against a layout nothing writes.
            def foreign_path(name):
                return output / IMAGE_OUTPUT_FILES[name[len("one"):]] / name
            for name, text in foreign.items():
                foreign_path(name).parent.mkdir(parents=True, exist_ok=True)
                foreign_path(name).write_text(text, encoding="utf-8")

            with patch("batch_worm_roi.predict_raw",
                       return_value=(labels, raw.astype(np.uint8), None)):
                process_image(image, output, model=None, device=None,
                              standard_count=1, measurement_backend="imagej")

            for name, text in foreign.items():
                path = foreign_path(name)
                self.assertTrue(path.is_file(), "%s was deleted" % name)
                self.assertEqual(path.read_text(encoding="utf-8"), text)


class UnsupportedImageFormatTests(unittest.TestCase):
    """Multi-plane and colour images must be refused, greyscale must not be.

    process_image reads only plane 0 of a file, while ImageJ measures whichever
    channel/Z-slice/timepoint is selected in its window. On a stack or an RGB
    file the two silently describe different planes, so the batch refuses them.
    The allow-list side matters just as much: a needlessly strict list would
    reject the 16-bit greyscale the lab actually acquires.

    Extra planes that are entirely zero are the one exception. Acquirers such as
    MetaMorph append an empty overlay plane to every file, and refusing those
    would turn away data that cannot produce a wrong measurement: a plane with
    no signal in it can only ever yield zeros.
    """

    def _write(self, folder, name, image):
        path = folder / name
        image.save(path)
        return path

    def test_single_plane_greyscale_modes_are_accepted(self):
        # "I;16" and "I;16B" are what PIL reports for the 16-bit TIFFs this lab
        # produces; rejecting either would refuse most real data.
        cases = {
            "eight_bit.tif": Image.fromarray(np.zeros((8, 8), np.uint8)),
            "sixteen_bit.tif": Image.fromarray(np.zeros((8, 8), np.uint16)),
            "binary.tif": Image.fromarray(np.zeros((8, 8), bool)),
        }
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            for name, image in cases.items():
                with self.subTest(name):
                    path = self._write(folder, name, image)
                    self.assertIsNone(check_image(path).code,
                                      "%s was rejected but is plain greyscale" % name)

    def test_big_endian_sixteen_bit_is_not_mistaken_for_colour(self):
        # PIL reports big-endian 16-bit grey as "I;16B", which is not colour.
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            path = folder / "big_endian.tif"
            raw = np.arange(64, dtype=">u2").reshape(8, 8)
            Image.fromarray(raw).save(path)
            with Image.open(path) as saved:
                self.assertEqual(saved.mode, "I;16B")
            self.assertIsNone(check_image(path).code)

    def _write_stack(self, folder, name, planes):
        path = folder / name
        images = [Image.fromarray(plane) for plane in planes]
        images[0].save(path, save_all=True, append_images=images[1:])
        return path

    def test_multi_plane_image_is_rejected_as_a_stack(self):
        # Real data on plane 1 is what makes this a genuine stack: the operator
        # could measure it in ImageJ while the model segmented plane 0.
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            path = self._write_stack(folder, "stack.tif", [
                np.zeros((8, 8), np.uint8),
                np.full((8, 8), 200, np.uint8),
            ])
            found = check_image(path)
            self.assertEqual(found.code, "stack")
            self.assertEqual(found.value, 2)  # the plane count, not a formatted string

    def test_empty_extra_planes_are_accepted_and_reported_as_a_note(self):
        # MetaMorph writes every acquisition with a second, all-zero overlay
        # plane. Six such files were produced by this very program before the
        # guard existed; refusing them would throw away valid data.
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            path = self._write_stack(folder, "overlay.tif", [
                np.full((8, 8), 7, np.uint8),
                np.zeros((8, 8), np.uint8),
            ])
            found = check_image(path)
            self.assertIsNone(found.code)
            self.assertIn("overlay.tif", _reject_unsupported_images([str(path)])[0])

    def test_every_extra_plane_is_checked_not_just_the_second(self):
        # Three planes where the data sits in the last one: stopping after the
        # first extra plane would wave this through as an overlay.
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            path = self._write_stack(folder, "mixed.tif", [
                np.zeros((8, 8), np.uint8),
                np.zeros((8, 8), np.uint8),
                np.full((8, 8), 3, np.uint8),
            ])
            found = check_image(path)
            self.assertEqual(found.code, "stack")
            self.assertEqual(found.value, 3)

    def test_colour_image_is_rejected_as_colour(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            path = self._write(folder, "rgb.tif",
                               Image.fromarray(np.zeros((8, 8, 3), np.uint8), "RGB"))
            found = check_image(path)
            self.assertEqual(found.code, "colour")
            self.assertIn("RGB", found.value)

    def test_unreadable_file_is_reported_instead_of_raising(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            path = folder / "broken.tif"
            path.write_bytes(b"\xff\xfe not a tiff at all")
            self.assertEqual(check_image(path).code, "unreadable")

    def test_truncated_file_is_rejected_even_though_its_header_reads_cleanly(self):
        # A half-copied TIFF opens fine and reports a mode; only decoding the
        # pixels fails. A header-only check would pass it and then abort the
        # batch part-way through, after CUDA init and model load.
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            whole = folder / "whole.tif"
            Image.fromarray(np.zeros((256, 256), np.uint16)).save(whole)
            truncated = folder / "half.tif"
            data = whole.read_bytes()
            truncated.write_bytes(data[:len(data) // 2])

            with Image.open(truncated) as opened:  # header alone looks fine
                self.assertEqual(opened.mode, "I;16")
                self.assertEqual(opened.n_frames, 1)
            self.assertEqual(check_image(truncated).code, "unreadable")

    def test_a_png_named_tif_is_rejected_because_imagej_cannot_open_it(self):
        # Pillow sniffs content, so this passes a format-agnostic read and then
        # silently yields no measurements when ImageJ tries to open it.
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            path = folder / "actually_a_png.tif"
            Image.fromarray(np.zeros((8, 8), np.uint8)).save(path, format="PNG")
            found = check_image(path)
            self.assertEqual(found.code, "format")
            self.assertEqual(found.value, "PNG")

    def test_batch_refuses_a_folder_of_stacks_without_writing_anything(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            self._write_stack(folder, "stack.tif", [
                np.zeros((8, 8), np.uint8),
                np.full((8, 8), 200, np.uint8),
            ])
            output = folder / "out"

            with self.assertRaises(ValueError) as caught:
                run_gui_batch(str(folder), str(output), "worm.pt", "tip.pt")
            self.assertIn("stack.tif", str(caught.exception))
            # Refused before touching the filesystem, so not even the output
            # directory is created, let alone a batch summary.
            self.assertFalse(output.exists())

    def test_batch_accepts_greyscale_and_fails_later_on_the_missing_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            Image.fromarray(np.zeros((8, 8), np.uint16)).save(folder / "ok.tif")
            # The missing checkpoint proves the format guard let the folder
            # through instead of rejecting it.
            with self.assertRaises(FileNotFoundError):
                run_gui_batch(str(folder), str(folder / "out"), "worm.pt", "tip.pt")

    def test_binary_image_is_usable_but_flagged_for_a_warning(self):
        # 1-bit is greyscale, so it is not refused -- but the shipped models have
        # only seen 8- and 16-bit acquisitions, and the GUI turns this code into
        # the dialog that says so.
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            path = self._write(folder, "binary.tif",
                               Image.fromarray(np.zeros((8, 8), bool)))
            found = check_image(path)
            self.assertIsNone(found.code)
            self.assertEqual(found.warning, "binary")
            self.assertIn("binary", _reject_unsupported_images([str(path)])[0])

    def test_binary_image_reaches_the_model_as_an_eight_bit_array(self):
        # The warning above promises a 1-bit image is usable, and it was not:
        # np.asarray on mode "1" gives a *bool* array, so the first background
        # subtraction in process_image raised "numpy boolean subtract ... use
        # bitwise_xor" and the image came out as Failed_binary_1bit.tif. Checking
        # the verdict alone could not catch that -- this pins the dtype that
        # makes the verdict true.
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            raw = np.zeros((8, 8), bool)
            raw[2:6, 2:6] = True  # both values present, so the dtype is visible
            path = self._write(folder, "binary.tif", Image.fromarray(raw))
            with Image.open(path) as opened:
                self.assertEqual(opened.mode, "1")
                array = image_array(opened)
            self.assertEqual(array.dtype, np.uint8)
            self.assertEqual(sorted(np.unique(array).tolist()), [0, 255])

    def test_non_binary_modes_carry_no_warning(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            for name, raw in (("eight.tif", np.zeros((8, 8), np.uint8)),
                              ("sixteen.tif", np.zeros((8, 8), np.uint16))):
                with self.subTest(name):
                    path = self._write(folder, name, Image.fromarray(raw))
                    self.assertEqual(check_image(path).warning, "")


class DuplicateResultNameTests(unittest.TestCase):
    """Two images must not write their results to the same names.

    process_image names its outputs after the file name without the extension,
    so A.tif and A.tiff in one folder both produce A_RoiSet.zip and A_QC.png.
    The second one used to silently overwrite the first, leaving two rows under
    one image name in the summary and nothing anywhere to say so.
    """

    def test_same_stem_with_two_extensions_is_refused_before_anything_is_written(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            for name in ("A.tif", "A.tiff"):
                Image.fromarray(np.zeros((8, 8), np.uint16)).save(folder / name)
            output = folder / "out"

            with self.assertRaises(ValueError) as caught:
                run_gui_batch(str(folder), str(output), "worm.pt", "tip.pt")
            message = str(caught.exception)
            self.assertIn("A.tif", message)
            self.assertIn("A.tiff", message)
            self.assertFalse(output.exists())

    def test_distinct_stems_are_not_flagged(self):
        names = [os.path.join("x", name) for name in ("A.tif", "B.tiff", "C.tif")]
        self.assertEqual(find_stem_collisions(names), [])

    def test_names_differing_only_in_case_collide_as_well(self):
        # On the case-insensitive filesystems this runs on, A.tif and a.tiff
        # write the same file, so they must be caught together.
        found = find_stem_collisions([os.path.join("x", "A.tif"),
                                      os.path.join("x", "a.tiff")])
        self.assertEqual([stem for stem, _ in found], ["a"])
        self.assertEqual(found[0][1], ["A.tif", "a.tiff"])


class BatchFailureHandlingTests(unittest.TestCase):
    """One unprocessable image must not discard the rest of the batch.

    The engine is stubbed here -- a real checkpoint is not needed to make
    process_image raise, and the loop's behaviour around the failure is what is
    under test. The end-to-end CUDA run covers the same path with real weights.
    """

    def _run(self, folder, output, *fail_on):
        def fake_process_image(path, *args, **kwargs):
            name = os.path.basename(str(path))
            if name in fail_on:
                raise RuntimeError("boom in " + name)
            # 32 fields, the real process_image return value. _success_row refuses
            # any other length and _summary_rows checks every row against the
            # header, so this cannot drift unnoticed.
            return [name, 10, 10, "PASS"] + [""] * 28

        for name in ("worm.pt", "tip.pt"):
            (folder / name).write_bytes(b"stub checkpoint")
        lines, failures = [], []
        with patch("batch_worm_roi.torch.load", return_value={"model_state": {}}), \
                patch("batch_worm_roi.WormUNet"), \
                patch("batch_worm_roi.process_image", side_effect=fake_process_image):
            results = run_gui_batch(
                str(folder), str(output), str(folder / "worm.pt"),
                str(folder / "tip.pt"), disable_tip_refinement=True,
                on_status=lines.append,
                on_image_failed=lambda name, detail: failures.append((name, detail)))
        return results, lines, failures

    def _summary_rows(self, output):
        """The summary keyed by column, after checking no row is short.

        A row with fewer fields than the header silently shifts every column
        after it once read by name, which is how a failure row could end up
        looking like a success.
        """
        with (output / OTHER_DIRNAME / "batch_summary.csv").open(
                "r", newline="", encoding="utf-8-sig") as handle:
            parsed = list(csv.reader(handle))
        header, rows = parsed[0], parsed[1:]
        for row in rows:
            self.assertEqual(len(row), len(header),
                             "ragged summary row: %d fields against a %d-column "
                             "header" % (len(row), len(header)))
        return [dict(zip(header, row)) for row in rows]

    def test_failed_image_is_skipped_named_and_recorded(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            for name in ("ok1.tif", "bad.tif", "ok2.tif"):
                Image.fromarray(np.zeros((8, 8), np.uint16)).save(folder / name)
            output = folder / "out"

            results, lines, failures = self._run(folder, output, "bad.tif")

            # The other two were still processed: that is the whole point.
            self.assertEqual(len(results), 2)
            self.assertNotIn("bad.tif", [row[0] for row in results])
            self.assertEqual(failures, [("bad.tif", "RuntimeError: boom in bad.tif")])
            self.assertTrue(any(line.startswith("Failed_bad.tif: ERROR")
                                for line in lines), lines)

            rows = self._summary_rows(output)
            self.assertEqual([row["image"] for row in rows],
                             ["Failed_bad.tif", "ok1.tif", "ok2.tif"])
            failed = rows[0]
            self.assertEqual(failed["qc_status"], "FAILED")
            self.assertEqual(failed["error"], "RuntimeError: boom in bad.tif")
            self.assertEqual(failed["worm_count"], "")
            # A successful row's values must land under their own column names.
            # Checking the length alone is not enough: a row one field short still
            # parses, with every value after the gap in the wrong column.
            self.assertEqual(rows[1]["worm_count"], "10")
            self.assertEqual(rows[1]["qc_status"], "PASS")
            self.assertEqual(rows[1]["software_version"], SOFTWARE_VERSION)
            # The successes keep an empty error column, so the column means one
            # thing only.
            self.assertEqual(rows[1]["error"], "")

    def test_partial_results_of_a_failed_image_are_removed(self):
        """A failure after the ROI set was written must not leave it behind.

        process_image writes <stem>_RoiSet.zip before it draws the overlay, so a
        failure in the drawing step used to leave a valid-looking ROI set in the
        output folder. The ImageJ side measures every input that has one, so a
        run would report an image as FAILED and Fiji would then measure it
        anyway, with ROIs from a pipeline that never finished.
        """
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            Image.fromarray(np.zeros((8, 8), np.uint16)).save(folder / "bad.tif")
            output = folder / "out"
            output.mkdir()
            # Leftovers shaped exactly like the ones process_image writes, each in
            # the subfolder it belongs in, plus two files that are not one of its
            # outputs and must survive -- placed in other/ so that "the cleanup
            # only removes the names it owns" is what is being tested, not "the
            # cleanup empties the whole folder".
            for suffix, subfolder in IMAGE_OUTPUT_FILES.items():
                leftover = output / subfolder / ("bad" + suffix)
                leftover.parent.mkdir(parents=True, exist_ok=True)
                leftover.write_bytes(b"partial")
            (output / OTHER_DIRNAME / "bad_notes.txt").write_text("mine", encoding="utf-8")
            (output / OTHER_DIRNAME / "other_QC.png").write_bytes(b"not this image's")

            results, _, failures = self._run(folder, output, "bad.tif")

            self.assertEqual(results, [])
            self.assertEqual(len(failures), 1)
            for suffix, subfolder in IMAGE_OUTPUT_FILES.items():
                self.assertFalse((output / subfolder / ("bad" + suffix)).exists(),
                                 "partial %s survived a failure" % suffix)
            self.assertTrue((output / OTHER_DIRNAME / "bad_notes.txt").is_file())
            self.assertTrue((output / OTHER_DIRNAME / "other_QC.png").is_file())

    def test_a_row_builder_refuses_a_result_the_header_cannot_hold(self):
        """A short row is the silent failure mode, so the row builder rejects one."""
        header = ["image", "worm_count"] + list(RUN_COLUMNS)
        run_values = ("run", "cpu", "FP32", "worm-sha", "bf-sha", "tip-sha",
                      "0.4.0-imagej", "")

        self.assertEqual(_success_row(header, ["ok.tif", 10], run_values),
                         ["ok.tif", 10, "run", "cpu", "FP32", "worm-sha", "bf-sha",
                          "tip-sha", "0.4.0-imagej", ""])
        for wrong in (["ok.tif"], ["ok.tif", 10, "extra"]):
            with self.assertRaises(ValueError):
                _success_row(header, wrong, run_values)

    def test_progress_is_reported_after_every_image_including_failures(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            for name in ("ok1.tif", "bad.tif", "ok2.tif"):
                Image.fromarray(np.zeros((8, 8), np.uint16)).save(folder / name)

            _, lines, _ = self._run(folder, folder / "out", "bad.tif")

            self.assertEqual([line for line in lines if line.startswith("Progress:")],
                             ["Progress: 1/3 (33%)", "Progress: 2/3 (66%)",
                              "Progress: 3/3 (100%)"])

    def test_a_batch_where_every_image_fails_still_names_them_all(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            for name in ("a.tif", "b.tif"):
                Image.fromarray(np.zeros((8, 8), np.uint16)).save(folder / name)
            output = folder / "out"

            results, lines, failures = self._run(folder, output, "a.tif", "b.tif")

            self.assertEqual(results, [])
            self.assertEqual(len(failures), 2)
            self.assertTrue(any("Failed images (2):" in line for line in lines), lines)
            self.assertEqual([row["image"] for row in self._summary_rows(output)],
                             ["Failed_a.tif", "Failed_b.tif"])


class ImageJBridgeNotificationTests(unittest.TestCase):
    """The only channel back to the Java plug-in.

    Fiji starts the UI and then has nothing to read but these files: whether the
    batch finished, where the ROIs went, which images were skipped, and which
    ones to measure. Every field is checked here because a field that stops
    being written is not an error anywhere -- Fiji would simply measure nothing,
    or measure a partial set, and report neither.
    """

    def _written(self, folder, *args, **kwargs):
        with patch("worm_roi_gui.IMAGEJ_BRIDGE_DIR", str(folder)):
            error = worm_roi_gui._write_imagej_bridge_file(*args, **kwargs)
        return Path(folder), error

    def _notifications(self, directory):
        """Written notifications, in the order the Java side would read them."""
        return sorted(Path(directory).glob("*.properties"))

    def _fields(self, path):
        text = path.read_bytes().decode("ascii")  # Properties.load reads latin-1
        return dict(line.split("=", 1) for line in text.strip().splitlines())

    def test_every_field_round_trips_as_utf8_through_base64(self):
        # Non-ASCII goes through base64 on purpose: the file has to stay ASCII
        # so the Java side reads it the same way whatever the console encoding
        # is, and the message is Chinese in one of the two languages.
        message = "2 张图像处理失败，已跳过。"
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            directory, error = self._written(
                folder, "complete", r"C:\输入\图片", r"D:\结果", message)
            self.assertIsNone(error)
            files = self._notifications(directory)
            self.assertEqual(len(files), 1)
            fields = self._fields(files[0])

            self.assertEqual(fields["status"], "complete")
            for key, expected in (("input_b64", r"C:\输入\图片"),
                                  ("output_b64", r"D:\结果"),
                                  ("message_b64", message)):
                self.assertEqual(base64.b64decode(fields[key]).decode("utf-8"),
                                 expected, key)
            # The temporary file _write_imagej_bridge_file renames from must not
            # be left in Fiji's temp directory for the next run to find.
            self.assertEqual(sorted(p.name for p in directory.iterdir()),
                             [files[0].name])

    def test_no_bridge_variable_means_standalone_and_writes_nothing(self):
        # The normal, non-ImageJ launch. Writing a file here would put a stray
        # bridge notification in every user's temp directory.
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            with patch("worm_roi_gui.IMAGEJ_BRIDGE_DIR", ""):
                self.assertIsNone(worm_roi_gui._write_imagej_bridge_file(
                    "complete", "in", "out"))
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_an_unwritable_path_is_reported_instead_of_raised(self):
        # Called from the shutdown path, where an exception would replace the
        # real result with a traceback.
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            blocker = Path(folder) / "blocker"
            blocker.write_text("not a directory", encoding="utf-8")
            with patch("worm_roi_gui.IMAGEJ_BRIDGE_DIR", str(blocker)):
                error = worm_roi_gui._write_imagej_bridge_file("complete", "in", "out")
            self.assertIsInstance(error, str)
            self.assertTrue(error)

    def test_a_second_notification_does_not_replace_an_unread_first_one(self):
        # Fiji may still be measuring when the next batch finishes. With one
        # fixed path the earlier notification would be gone and that batch would
        # be measured by nobody; one name per notification lets both wait.
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            with patch("worm_roi_gui.IMAGEJ_BRIDGE_DIR", str(folder)):
                self.assertIsNone(worm_roi_gui._write_imagej_bridge_file(
                    "complete", "in", "out"))
                self.assertIsNone(worm_roi_gui._write_imagej_bridge_file("cancelled"))

            files = self._notifications(folder)
            self.assertEqual([self._fields(path)["status"] for path in files],
                             ["complete", "cancelled"])

    def test_pending_is_true_until_the_plug_in_has_finished_with_a_notification(self):
        # This is what stops a second batch from being started on top of ROIs
        # that are still being measured. The plug-in deletes each file once it
        # has handled it -- not when it reads it, which is the part that used to
        # open this gate minutes too early -- so an empty directory is exactly
        # "everything measured". The Java half of that is pinned by
        # tests/AutoWormBridgeQueueTest.java, which drives the queue itself.
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            with patch("worm_roi_gui.IMAGEJ_BRIDGE_DIR", str(folder)):
                self.assertFalse(worm_roi_gui._imagej_bridge_pending())
                worm_roi_gui._write_imagej_bridge_file("complete", "in", "out")
                self.assertTrue(worm_roi_gui._imagej_bridge_pending())
                self._notifications(folder)[0].unlink()  # what the plug-in does
                self.assertFalse(worm_roi_gui._imagej_bridge_pending())

    def test_a_half_written_notification_is_never_picked_up(self):
        # The writer renames into place, so an in-progress file never ends in
        # .properties and the plug-in can never read half a notification.
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            (folder / "00000000000000000009.properties.tmp").write_text(
                "status=complete", encoding="ascii")
            with patch("worm_roi_gui.IMAGEJ_BRIDGE_DIR", str(folder)):
                self.assertFalse(worm_roi_gui._imagej_bridge_pending())

    def test_the_images_that_finished_are_named_for_the_plug_in(self):
        # Fiji decides what to measure from this list. A ROI ZIP is written
        # before an image is fully processed, so the list is the only record of
        # what actually succeeded.
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            directory, error = self._written(
                folder, "complete", "in", "out", "",
                successful_images=["b.tif", "a.tif"])
            self.assertIsNone(error)
            fields = self._fields(self._notifications(directory)[0])
            self.assertEqual(fields["ok_count"], "2")
            self.assertEqual(base64.b64decode(fields["ok_b64"]).decode().split("\n"),
                             ["a.tif", "b.tif"])

    def test_an_empty_list_is_still_a_list(self):
        # Every image failed. Sending nothing would read as "no list", and Fiji
        # would fall back to measuring by ROI ZIP -- the very images it has just
        # been told failed.
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            directory, _ = self._written(
                folder, "complete", "in", "out", "", successful_images=[])
            fields = self._fields(self._notifications(directory)[0])
            self.assertEqual(fields["ok_count"], "0")
            self.assertEqual(base64.b64decode(fields["ok_b64"]), b"")

    def test_no_success_list_leaves_the_receiver_on_its_older_rule(self):
        # Only the diagnostic --headless-run path sends no list.
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            directory, _ = self._written(folder, "complete", "in", "out")
            fields = self._fields(self._notifications(directory)[0])
            self.assertNotIn("ok_count", fields)
            self.assertNotIn("ok_b64", fields)

    def test_successful_images_is_the_input_list_without_the_failures(self):
        # The GUI side of the same contract: what goes into ok_b64.
        class Stub:
            pass

        job = {"source_names": {"a.tif", "b.tif", "c.tif"}}
        self.assertEqual(
            worm_roi_gui.WormRoiGui._successful_images(Stub(), job, [("b.tif", "boom")]),
            ["a.tif", "c.tif"])
        self.assertEqual(worm_roi_gui.WormRoiGui._successful_images(Stub(), job, []),
                         ["a.tif", "b.tif", "c.tif"])
        # Nothing to report leaves the receiver on its older rule, which is
        # safer than sending an empty list that would measure nothing at all.
        self.assertIsNone(worm_roi_gui.WormRoiGui._successful_images(Stub(), {}, []))
        self.assertIsNone(worm_roi_gui.WormRoiGui._successful_images(Stub(), None, []))

    def test_a_summary_csv_in_another_encoding_does_not_stop_the_window(self):
        # batch_summary.csv is the QC workflow's own output and the README sends
        # the experimenter to it, so an Excel round-trip on a zh-CN machine
        # rewrites it as GBK. The reader has to treat that like any other
        # unreadable summary: the folder path is persisted, so an exception here
        # came back at every later launch and the window could not be opened.
        class Stub:
            qc_status = {}
            qc_attention = {}
            qc_planes = {}

        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            output = Path(folder)
            (output / OTHER_DIRNAME).mkdir()
            (output / OTHER_DIRNAME / "batch_summary.csv").write_bytes(
                "image,qc_status,shape_refined_count\n".encode("gbk")
                + "虫1.tif,ok,0\n".encode("gbk"))
            self.assertIsNone(
                worm_roi_gui.WormRoiGui._load_qc_status_summary(Stub(), output),
                "a summary this encoding cannot be read as should be ignored")

    def test_the_event_pump_survives_a_failure_and_keeps_running(self):
        # The pump is the only consumer of the worker's queue, and _on_close
        # waits for it to write the batch's last notification before the window
        # may go. If a failure in the drain stopped the after() chain the window
        # would stay open, look alive, and refuse to close.
        class FakeRoot:
            def __init__(self):
                self.calls = []

            def after(self, milliseconds, callback):
                self.calls.append(callback.__name__)

        class Stub:
            _poll_events = worm_roi_gui.WormRoiGui._poll_events
            _report_pump_failure = worm_roi_gui.WormRoiGui._report_pump_failure

            def __init__(self):
                self.root = FakeRoot()
                self.event_queue = queue.Queue()
                self.said = []

            def _tr(self, chinese, english):
                return chinese

            def _terminal_write(self, text, tag):
                self.said.append(tag)

            def _draw_previews(self):
                pass

            def _record_qc_status_from_line(self, line):
                return False

            def _scan_qc_images(self, reset_page=False):
                raise UnicodeDecodeError("utf-8", b"\xb3", 0, 1, "invalid start byte")

            def _processing_finished(self, return_code):
                return True

        # The failure is meant to happen, so its traceback is expected noise
        # rather than a result; keep it out of the test runner's output.
        stub = Stub()
        stub.event_queue.put(("refresh_qc", None))
        with contextlib.redirect_stderr(io.StringIO()):
            stub._poll_events()
        self.assertEqual(stub.root.calls, ["_poll_events"],
                         "the pump did not re-arm itself after the failure")
        self.assertIn("error", stub.said, "the failure was not reported to the user")

        # The one exit that must not re-arm: _processing_finished returns True
        # only after it has destroyed the window.
        stub = Stub()
        stub.event_queue.put(("done", 0))
        with contextlib.redirect_stderr(io.StringIO()):
            stub._poll_events()
        self.assertEqual(stub.root.calls, [], "the pump re-armed onto a destroyed window")

    def test_a_bad_summary_file_does_not_leave_the_batch_half_finished(self):
        # The scan runs at the top of _processing_finished, before the job is
        # cleared. Letting it throw left the controls disabled and self.process
        # pointing at a thread that had already ended -- and _on_close waits for
        # that same call to arrive, so the window could not be closed either.
        class Stub:
            _processing_finished = worm_roi_gui.WormRoiGui._processing_finished
            _scan_qc_images_safely = worm_roi_gui.WormRoiGui._scan_qc_images_safely
            _report_pump_failure = worm_roi_gui.WormRoiGui._report_pump_failure

            def __init__(self):
                self.said = []
                self.qc_status = {}
                self.qc_attention = {}
                self.active_job = {"failures": [], "output_dir": "out"}
                self.process = "a thread that has already ended"
                self.stop_requested = False
                self.close_when_stopped = False
                self.pending_error_text = ""

            def _tr(self, chinese, english):
                return chinese

            def _terminal_write(self, text, tag):
                self.said.append(text)

            def _set_processing_controls(self, busy):
                self.said.append("controls=%s" % busy)

            def _scan_qc_images(self, reset_page=False):
                raise UnicodeDecodeError("utf-8", b"\xb3", 0, 1, "invalid start byte")

            def _effective_output_dir(self):
                return Path("out")

            def _save_config(self):
                pass

        stub = Stub()
        with patch.object(worm_roi_gui, "IMAGEJ_BRIDGE_DIR", ""):
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertFalse(stub._processing_finished(0))

        self.assertIsNone(stub.process, "the finished batch was never cleared")
        self.assertIsNone(stub.active_job, "the finished batch was never cleared")
        self.assertIn("controls=False", stub.said, "the controls were left disabled")
        self.assertIn("处理完成", "\n".join(stub.said),
                      "the batch was not reported as finished")

    def test_a_folder_that_cannot_be_listed_is_reported_not_ignored(self):
        # The folder was readable when it was chosen and is not any more: deleted,
        # renamed, on a drive that was unplugged. _start_processing reached the
        # list of TIFFs only after the whole pre-flight pass, and a windowed build
        # has nowhere to print a traceback -- so the button did nothing at all,
        # with no message, and looked broken rather than being wrong about
        # anything the user could see.
        class Stub:
            _start_processing = worm_roi_gui.WormRoiGui._start_processing

            def __init__(self):
                self.process = None
                self.root = None
                self.annotation_path = None
                self.last_valid_worm_count = None
                self.current_folder = Path(tempfile.gettempdir()) / "aw-gone-folder"
                self.brightfield_roi_var = _FakeVar(False)
                self.said = []

            def _tr(self, chinese, english):
                return chinese

            def _validated_worm_count(self, show_error=False):
                return 3

            def _close_head_annotation(self):
                pass

            def _validate_runtime(self):
                return []

            def _terminal_write(self, text, tag):
                self.said.append(text)

        stub = Stub()
        self.assertFalse(stub.current_folder.exists(), "the fixture folder must not exist")
        with patch.object(worm_roi_gui, "IMAGEJ_BRIDGE_DIR", ""), \
                patch.object(worm_roi_gui.messagebox, "showerror") as showerror:
            stub._start_processing()

        showerror.assert_called_once()
        self.assertIn("无法读取文件夹", showerror.call_args[0][0])
        self.assertIn("aw-gone-folder", showerror.call_args[0][1])
        self.assertEqual(stub.said, [], "processing started anyway")

    def test_a_stopped_batch_still_hands_over_the_images_that_finished(self):
        # A stopped batch keeps the ROI sets and summary rows of the images that
        # did finish -- the README says so. Sending "cancelled" threw them away:
        # the plug-in measures nothing at all for a cancelled batch, so those
        # images ended up with no fluorescence and no CTCF anywhere, with their
        # ROI sets sitting in the output folder looking complete.
        class Stub:
            _processing_finished = worm_roi_gui.WormRoiGui._processing_finished
            _scan_qc_images_safely = worm_roi_gui.WormRoiGui._scan_qc_images_safely
            _report_pump_failure = worm_roi_gui.WormRoiGui._report_pump_failure
            _successful_images = worm_roi_gui.WormRoiGui._successful_images
            _failure_message = worm_roi_gui.WormRoiGui._failure_message

            def __init__(self, completed_stems):
                self.said = []
                self.sent = []
                self.qc_status = {}
                self.qc_attention = {}
                self.active_job = {
                    "source_names": {"a.tif", "b.tif", "c.tif"},
                    "completed_stems": completed_stems,
                    "failures": [],
                    "input_dir": "in",
                    "output_dir": "out",
                }
                self.process = "running"
                self.stop_requested = True
                self.close_when_stopped = False
                self.pending_error_text = ""

            def _tr(self, chinese, english):
                return chinese

            def _terminal_write(self, text, tag):
                self.said.append(text)

            def _set_processing_controls(self, busy):
                pass

            def _scan_qc_images(self, reset_page=False):
                pass

            def _effective_output_dir(self):
                return Path("out")

            def _save_config(self):
                pass

            def _notify_imagej(self, status, input_dir, output_dir, message="",
                               successful_images=None, planes=(1, 1)):
                self.sent.append((status, message, successful_images, planes))

        # Two of the three finished before the stop: those two are what Fiji can
        # measure, and the third must not be in the list.
        stub = Stub({"a", "b"})
        with patch.object(worm_roi_gui, "IMAGEJ_BRIDGE_DIR", "bridge"):
            self.assertFalse(stub._processing_finished(-1))
        self.assertEqual(len(stub.sent), 1, stub.sent)
        status, message, images, planes = stub.sent[0]
        self.assertEqual(status, "complete")
        self.assertEqual(images, ["a.tif", "b.tif"])
        self.assertIn("2", message)
        # Brightfield ROI off: the plug-in is told to measure plane 1, which is
        # what it measured before this field existed.
        self.assertEqual(planes, (1, 1))

        # A batch stopped before anything finished has nothing to measure, and
        # says so the way the plug-in has always understood.
        stub = Stub(set())
        with patch.object(worm_roi_gui, "IMAGEJ_BRIDGE_DIR", "bridge"):
            self.assertFalse(stub._processing_finished(-1))
        self.assertEqual(stub.sent, [("cancelled", "", None, (1, 1))])

    def test_only_finished_narrows_the_list_to_what_was_completed(self):
        # The list the GUI can vouch for: a stem whose progress line it has seen,
        # and which is not among the failures. Stems are collected lowercased.
        class Stub:
            pass

        job = {"source_names": {"ONE.tif", "two.tif", "three.tif"},
               "completed_stems": {"one", "two"}}
        self.assertEqual(
            worm_roi_gui.WormRoiGui._successful_images(Stub(), job, [], only_finished=True),
            ["ONE.tif", "two.tif"])
        self.assertEqual(
            worm_roi_gui.WormRoiGui._successful_images(
                Stub(), job, [("two.tif", "boom")], only_finished=True),
            ["ONE.tif"])
        # Without the set there is nothing to narrow by, and the caller falls
        # back to the older rule rather than guessing.
        self.assertIsNone(
            worm_roi_gui.WormRoiGui._successful_images(
                Stub(), {"source_names": {"a.tif"}}, [], only_finished=True))

    def test_the_failure_message_names_the_images_and_caps_the_list(self):
        # Fiji has no other way to learn why an image is missing from the
        # measurement table, so the skipped names have to be in this field even
        # though the batch itself succeeded.
        class Stub:
            _tr = staticmethod(lambda chinese, english: english)

        self.assertEqual(worm_roi_gui.WormRoiGui._failure_message(Stub(), []), "")

        eleven = [("img%d.tif" % index, "RuntimeError: boom") for index in range(11)]
        text = worm_roi_gui.WormRoiGui._failure_message(Stub(), eleven)
        self.assertIn("Failed_img0.tif: RuntimeError: boom", text)
        self.assertIn("11 image(s) failed", text)
        # Capped, so one broken folder cannot fill the dialog Fiji shows.
        self.assertNotIn("Failed_img10.tif", text)
        self.assertIn("11 failed in total", text)


class StaleReportCleanupTests(unittest.TestCase):
    """Every stale report must be recognised through the writer that made it.

    _remove_stale_report only deletes a file whose header matches the fields
    the writer emits. If a field list is ever renamed or reordered on one side
    only, cleanup would silently stop working -- these tests pin both ends.
    """

    def test_each_report_written_by_its_own_writer_is_recognised_as_stale(self):
        cases = [
            ("head.csv", lambda path: _write_csv_atomic(path, HEAD_REPORT_FIELDS, []),
             HEAD_REPORT_FIELDS),
            ("manual_split.csv",
             lambda path: _write_csv_atomic(path, MANUAL_SPLIT_REPORT_FIELDS, []),
             MANUAL_SPLIT_REPORT_FIELDS),
            ("segment.csv",
             lambda path: _write_csv_atomic(path, SEGMENT_REPORT_FIELDS, []),
             SEGMENT_REPORT_FIELDS),
            ("shape.csv", lambda path: write_shape_report(path, []),
             ShapeReport.__dataclass_fields__),
            ("measurements.csv",
             lambda path: _write_csv_atomic(path, PYTHON_MEASUREMENT_HEADER, []),
             PYTHON_MEASUREMENT_HEADER),
        ]
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            for name, write, header in cases:
                path = folder / name
                write(path)
                self.assertTrue(path.is_file(), "%s was not written" % name)
                _remove_stale_report(path, header)
                self.assertFalse(path.exists(), "%s was not cleaned up" % name)

    def test_a_header_only_file_that_is_not_ours_survives(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            path = folder / "head.csv"
            path.write_text("Slices\tCount\n1\t10\n", encoding="utf-8")
            _remove_stale_report(path, HEAD_REPORT_FIELDS)
            self.assertTrue(path.is_file())

    def test_undecodable_file_is_left_alone_instead_of_raising(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            path = folder / "junk.csv"
            path.write_bytes(b"\xff\xfe\x00\x01binary\x00")
            _remove_stale_report(path, HEAD_REPORT_FIELDS)
            self.assertTrue(path.is_file())


# A per-image table as the plug-in writes it. Two columns identify it, and they
# survive the plug-in gaining columns -- 0.4.1 added five calibration ones.
JAVA_MEASUREMENT_HEADER = ("Area,Mean,Min,Max,IntDen,Median,RawIntDen,Index,Image,Label,"
                           "PixelArea,BackgroundCorrectedMean,CTCF,RawCTCF,Type,"
                           "MeasurementBackend,PixelWidth,PixelHeight,SpatialUnit,"
                           "ValueUnit,IntensityFunction")
JAVA_MEASUREMENT_ROW = ("10,1.0,1,1,10,1,10,1,taken-last-week.tif,01_worm,10,0.0,999.0,"
                        "0,worm,ImageJ,1,1,pixel,Gray Value,None")


class PluginTableCleanupTests(unittest.TestCase):
    """The plug-in's per-image table must not survive a run that did not make it.

    Both backends write <stem>_measurements.csv, and only the Python backend
    writes that file here -- with the ImageJ backend the plug-in writes it after
    this process has exited. So a table left by an earlier ImageJ run sat under
    the same name, from a different schema, and after the run it read as this
    run's measurement. The ROIs it went with were this run's, which is what made
    it convincing. (The two now live in different subfolders -- the table in
    measurements/, the ROIs in other/ -- which does not help: both are this
    run's, and the names still line up.)
    """

    def _image(self, folder, name="one.tif"):
        """The image, and a stand-in for the model's prediction of it."""
        raw = np.full((100, 100), 80, dtype=np.uint8)
        raw[10:90, 40:51] = 200
        image_path = folder / name
        Image.fromarray(raw).save(image_path)
        labels = np.zeros((100, 100), dtype=np.uint16)
        labels[10:90, 40:51] = 1

        def fake_predict(*_args, **_kwargs):
            return labels.copy(), raw.copy(), None

        return image_path, fake_predict

    def _process(self, image_path, output, fake_predict, **kwargs):
        # measurement_backend="imagej" is what makes this the case under test:
        # with the Python backend the table is written here and simply replaces
        # whatever was there, so no stale file survives and nothing is proved.
        kwargs.setdefault("measurement_backend", "imagej")
        with patch("batch_worm_roi.predict_raw", side_effect=fake_predict):
            return process_image(image_path, output, model=None, device=None,
                                 standard_count=1, **kwargs)

    def test_the_plugins_table_is_removed_for_an_image_this_run_reprocesses(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            output = folder / "out"
            output.mkdir()
            image_path, fake_predict = self._image(folder)
            stale = output / MEASUREMENTS_DIRNAME / "one_measurements.csv"
            stale.parent.mkdir(parents=True)
            stale.write_text(JAVA_MEASUREMENT_HEADER + "\n" + JAVA_MEASUREMENT_ROW + "\n",
                             encoding="utf-8")

            self._process(image_path, output, fake_predict)

            self.assertFalse(stale.exists(),
                             "the earlier ImageJ run's table is still next to this run's ROIs")

    def test_a_foreign_file_with_the_generated_name_survives(self):
        # The output folder belongs to the experimenter. For an image this run
        # does not write a table for, a file that only shares the generated name
        # must be left exactly as it was.
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            output = folder / "out"
            output.mkdir()
            image_path, fake_predict = self._image(folder)
            foreign = output / MEASUREMENTS_DIRNAME / "one_measurements.csv"
            foreign.parent.mkdir(parents=True)
            foreign.write_text("image,count\nmine,3\n", encoding="utf-8")

            self._process(image_path, output, fake_predict)

            self.assertEqual(foreign.read_text(encoding="utf-8"), "image,count\nmine,3\n")

    def test_the_plugin_table_is_recognised_whatever_the_plugin_version(self):
        # A table written by an older or newer build is still ours: the check is
        # for the two identifying columns, not for a column list pinned here that
        # would go stale the moment the plug-in gains one.
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            for name, header in (
                    ("older.csv", "Image,Label,Area,CTCF,MeasurementBackend"),
                    ("full.csv", JAVA_MEASUREMENT_HEADER),
                    ("reordered.csv",
                     "MeasurementBackend,CTCF,Area,Image,PixelWidth,PixelHeight")):
                path = folder / name
                path.write_text(header + "\n", encoding="utf-8")
                _remove_stale_imagej_report(path)
                self.assertFalse(path.exists(), "%s was not recognised as ours" % name)

    def test_a_foreign_header_is_not_deleted_by_the_plugin_rule(self):
        # Either identifying column alone is not enough -- both are needed, so an
        # experimenter's own table that happens to count CTCF sites survives.
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            for name, content in (
                    ("counts.csv", "image,CTCF\nmine,3\n".encode("utf-8")),
                    ("python.csv", (",".join(PYTHON_MEASUREMENT_HEADER) + "\n").encode("utf-8")),
                    ("junk.csv", b"\xff\xfe\x00\x01binary\x00"),
                    ("empty.csv", b"")):
                path = folder / name
                path.write_bytes(content)
                _remove_stale_imagej_report(path)
                self.assertTrue(path.exists(), "%s was deleted" % name)

    def test_both_rules_of_a_generated_name_are_disjoint(self):
        # Neither rule may eat the other's table: the Python one is matched by its
        # whole header, the plug-in's by two of its columns.
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            java = folder / "java.csv"
            java.write_text(JAVA_MEASUREMENT_HEADER + "\n", encoding="utf-8")
            _remove_stale_report(java, PYTHON_MEASUREMENT_HEADER)
            self.assertTrue(java.exists(), "the Python rule deleted the plug-in's table")

            python = folder / "python.csv"
            pyth = ",".join(PYTHON_MEASUREMENT_HEADER)
            python.write_text(pyth + "\n", encoding="utf-8")
            _remove_stale_imagej_report(python)
            self.assertTrue(python.exists(), "the plug-in rule deleted the Python table")
            # Both headers count CTCF sites, so that column cannot be the
            # discriminator: it is the second identifying column that separates
            # them, and it is only in the plug-in's table.
            self.assertIn("CTCF", pyth)
            self.assertNotIn("MeasurementBackend", pyth)


class SplitReportCleanupTests(unittest.TestCase):
    """Turning low-clarity splitting off must not leave last run's split report.

    Its three siblings -- shape refinement, head annotation, segment selection --
    all clear theirs when their feature is off. This one did not, so a report
    describing a step the run did not perform sat in the output folder under this
    run's image name, with nothing in it saying which run wrote it.
    """

    def _image(self, folder):
        raw = np.full((100, 100), 80, dtype=np.uint8)
        raw[10:90, 40:51] = 200
        image_path = folder / "one.tif"
        Image.fromarray(raw).save(image_path)
        labels = np.zeros((100, 100), dtype=np.uint16)
        labels[10:90, 40:51] = 1

        def fake_predict(*_args, **_kwargs):
            return labels.copy(), raw.copy(), None

        return image_path, fake_predict

    def _process(self, image_path, output, fake_predict, **kwargs):
        with patch("batch_worm_roi.predict_raw", side_effect=fake_predict):
            return process_image(image_path, output, model=None, device=None,
                                 standard_count=1, **kwargs)

    def test_a_stale_split_report_is_removed_when_splitting_is_off(self):
        from low_clarity_splitter import SplitReport

        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            output = folder / "out"
            output.mkdir()
            image_path, fake_predict = self._image(folder)
            stale = output / OTHER_DIRNAME / "one_split_qc.csv"
            stale.parent.mkdir(parents=True)
            _write_csv_atomic(stale, list(SplitReport.__dataclass_fields__), [])

            self._process(image_path, output, fake_predict, low_clarity_split=False)

            self.assertFalse(stale.exists(),
                             "a split report from a run that split survived one that did not")

    def test_a_foreign_file_with_the_generated_name_survives(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            output = folder / "out"
            output.mkdir()
            image_path, fake_predict = self._image(folder)
            foreign = output / OTHER_DIRNAME / "one_split_qc.csv"
            foreign.parent.mkdir(parents=True)
            foreign.write_text("image,count\nmine,3\n", encoding="utf-8")

            self._process(image_path, output, fake_predict, low_clarity_split=False)

            self.assertEqual(foreign.read_text(encoding="utf-8"), "image,count\nmine,3\n")

    def test_splitting_on_still_writes_its_report(self):
        from low_clarity_splitter import SplitReport

        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            output = folder / "out"
            output.mkdir()
            image_path, fake_predict = self._image(folder)

            self._process(image_path, output, fake_predict, low_clarity_split=True)

            written = (output / OTHER_DIRNAME
                       / "one_split_qc.csv").read_text(encoding="utf-8-sig")
            self.assertEqual(written.splitlines()[0].split(","),
                             list(SplitReport.__dataclass_fields__))


class DatasetInspectionTests(unittest.TestCase):
    """The offline check-up must get through a dataset, not stop at the first oddity.

    It is run over the whole output folder from the command line, and its report
    is what says whether the ROIs and the masks agree. An image the model found no
    worm in still gets the background ROI the pipeline appends to every image, so
    its worm list is empty -- min() on that raised, and because the call was not
    guarded it ended the scan, taking every later image's report with it. The one
    image that was actually unusual was the one that stopped the check-up.
    """

    def _dataset(self, folder):
        """Two images: one with a worm, one with only the background ROI."""
        from batch_worm_roi import compound_roi, instance_contours
        from inspect_roi_dataset import polygon_mask
        from roifile import roiwrite

        for stem, worm in (("one", True), ("two", False)):
            rows, columns = np.mgrid[0:40, 0:60]
            raw = np.full((40, 60), 30, dtype=np.uint8)
            if worm:
                raw[10:30, 20:41] = 180
            Image.fromarray(raw).save(folder / (stem + ".tif"))

            rois = []
            if worm:
                mask = np.zeros((40, 60), dtype=bool)
                mask[10:30, 20:41] = True
                outer, holes = instance_contours(mask)
                rois.append(compound_roi(outer, holes, "01_worm"))
            background = np.zeros((40, 60), dtype=bool)
            background[0:5, 0:5] = True
            outer, holes = instance_contours(background)
            rois.append(compound_roi(outer, holes, "99_background"))
            roiwrite(str(folder / ("RoiSet_" + stem + ".zip")), rois, mode="w")

            # The measurement table the pipeline writes, with the same number of
            # rows as there are ROIs: no worm means one row, the background's.
            header = ",".join(PYTHON_MEASUREMENT_HEADER)
            body = []
            for index, roi in enumerate(rois, start=1):
                pixels = polygon_mask((60, 40), roi.coordinates())
                values = np.asarray(raw)[pixels.astype(bool)]
                # Written through measurement_row so the fixture cannot drift out
                # of step with the header above it.
                body.append(",".join(str(cell) for cell in measurement_row({
                    "": "%d" % index, "Label": roi.name,
                    "Area": "%d" % int(values.size),
                    "Mean": "%.4f" % values.mean(),
                    "Min": "%.4f" % values.min(), "Max": "%.4f" % values.max(),
                    "IntDen": "%.4f" % float(values.sum()),
                    "Median": "%.4f" % float(np.median(values)),
                    "RawIntDen": "%.4f" % float(values.sum()),
                    "BackgroundCorrectedMean": "0.0000", "CTCF": "0.0000",
                    "Type": "background" if index == len(rois) else "worm"})))
            (folder / ("Results_" + stem + ".csv")).write_text(
                header + "\n" + "\n".join(body) + "\n", encoding="utf-8")

    def test_an_image_with_no_worm_does_not_end_the_scan(self):
        import inspect_roi_dataset

        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            self._dataset(folder)

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                inspect_roi_dataset.main(str(folder))

            printed = output.getvalue()
            self.assertNotIn("Failed to inspect", printed)
            self.assertIn("two.tif", printed, "the scan never reached the second image")

            summary = folder / "_inspection" / "roi_dataset_summary.csv"
            with summary.open("r", newline="", encoding="utf-8-sig") as handle:
                rows = {row["image"]: row for row in csv.DictReader(handle)}
            self.assertEqual(sorted(rows), ["one.tif", "two.tif"])
            # Reported, not silently dropped: the columns are empty for an image
            # with no worm to describe, and the two images are still comparable.
            self.assertEqual(rows["two.tif"]["worm_count"], "0")
            self.assertEqual(rows["two.tif"]["min_pixel_area"], "nan")
            self.assertEqual(rows["two.tif"]["max_pixel_area"], "nan")
            self.assertEqual(rows["one.tif"]["worm_count"], "1")
            self.assertEqual(rows["one.tif"]["min_pixel_area"], "420")
            # And the numbers agree with the table, which is the tool's whole job.
            self.assertLess(float(rows["one.tif"]["mean_mape_percent"]), 1e-6)
            self.assertTrue((folder / "_inspection" / "one_roi_overlay.png").is_file())

    def test_a_single_unreadable_image_keeps_the_others_in_the_report(self):
        import inspect_roi_dataset

        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            self._dataset(folder)
            # An image whose ROI set cannot be read: what a killed batch leaves.
            (folder / "RoiSet_two.zip").write_bytes(b"this is not a zip file")

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                inspect_roi_dataset.main(str(folder))

            printed = output.getvalue()
            self.assertIn("Failed to inspect RoiSet_two.zip", printed)
            with (folder / "_inspection" / "roi_dataset_summary.csv").open(
                    "r", newline="", encoding="utf-8-sig") as handle:
                images = [row["image"] for row in csv.DictReader(handle)]
            self.assertEqual(images, ["one.tif"],
                             "one bad image cost the others their report")


class BrightfieldRoiTests(unittest.TestCase):
    """0.5.0: the brightfield plane makes the ROIs, the fluorescence one the numbers.

    Every test here is about the pairing staying intact. The failure this feature
    could introduce is not a crash but a plausible number measured from the wrong
    plane, which nothing downstream can detect -- so the two planes used in the
    fixtures differ in both base level and location of their bright block, and a
    test that swapped them would produce visibly different pixels.
    """

    def _stack(self, folder, name, planes):
        path = folder / name
        images = [Image.fromarray(plane) for plane in planes]
        images[0].save(path, save_all=True, append_images=images[1:])
        return path

    def _two_planes(self, folder, name="two_planes.tif"):
        """(path, fluorescence, brightfield) for a 2-plane file.

        Plane 1 is the fluorescence image: bright base, bright block top-left.
        Plane 2 is the brightfield one: dark base, bright block bottom-right.
        """
        fluorescence = np.full((64, 64), 1000, np.uint16)
        fluorescence[2:12, 2:12] = 4000
        brightfield = np.full((64, 64), 10, np.uint16)
        brightfield[52:62, 52:62] = 4000
        return self._stack(folder, name, [fluorescence, brightfield]), fluorescence, brightfield

    @staticmethod
    def _brightfield_roi(model="brightfield weights"):
        return BrightfieldRoi(
            model=model, tip_model=None, image_size=256,
            normalization_mode="background_aware", tip_patch_size=192,
            tip_probability_threshold=0.40, tip_replace_fraction=0.14)

    @staticmethod
    def _fake_predict(seen, labels):
        """A predict_raw stand-in that records the pixels it was handed."""
        def fake_predict(model, raw, device, **kwargs):
            seen.append((model, np.array(raw)))
            return (labels,
                    (raw.astype(np.float32) / float(raw.max()) * 255).astype(np.uint8),
                    None)
        return fake_predict

    @staticmethod
    def _qc_pixels(path):
        """A QC image with its text banner cut off.

        The banner is drawn from the QC status and the worm counts, not from the
        picture underneath, so comparing it would add a second way for these
        assertions to break without saying anything about the plane.
        """
        with Image.open(path) as opened:
            return np.asarray(opened.convert("RGB"))[40:].copy()

    def test_brightfield_off_refuses_a_stack_and_names_the_feature(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            self._two_planes(folder)
            output = folder / "out"

            with self.assertRaises(ValueError) as caught:
                run_gui_batch(str(folder), str(output), "worm.pt", "tip.pt")

            text = str(caught.exception)
            self.assertIn("two_planes.tif", text)
            self.assertIn("Brightfield", text)
            self.assertFalse(output.exists(),
                             "output was written for a batch that was refused")

    def test_brightfield_segments_the_brightfield_plane_and_measures_the_fluorescence_plane(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            stack, fluorescence, brightfield = self._two_planes(folder)
            output = folder / "out"
            output.mkdir()
            labels = np.zeros((64, 64), np.uint16)
            labels[15:50, 25:40] = 1
            seen = []

            with patch("batch_worm_roi.predict_raw",
                       side_effect=self._fake_predict(seen, labels)):
                result = process_image(
                    stack, output, model="fluorescence weights", device=None,
                    standard_count=1, fluorescence_plane=1, brightfield_plane=2,
                    brightfield_roi=self._brightfield_roi())

            # The segmentation ran on the brightfield weights and on plane 2's
            # pixels, compared against the array that was written to plane 2
            # rather than against a summary of it -- the two planes differ only in
            # where their bright block sits, so a base-level check alone could not
            # tell them apart.
            self.assertEqual(seen[0][0], "brightfield weights")
            self.assertTrue(np.array_equal(seen[0][1], brightfield),
                            "the segmentation input was not the brightfield plane")

            # The numbers came from plane 1. The ROI sits in plane 1's bright base.
            with (output / MEASUREMENTS_DIRNAME /
                  "two_planes_measurements.csv").open(
                      newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            worm = [row for row in rows if row["Type"] == "worm"]
            self.assertEqual(len(worm), 1, rows)
            self.assertEqual(float(worm[0]["Mean"]), 1000.0,
                             "the numbers did not come from the fluorescence plane")

            # The QC is the brightfield plane. Proved by construction rather than
            # by eyeballing: the same labels over a file holding only plane 2 give
            # a pixel-identical picture, and over a file holding only plane 1 a
            # different one.
            controls = {}
            for name, plane in (("brightfield_only", brightfield),
                                ("fluorescence_only", fluorescence)):
                source = folder / (name + ".tif")
                Image.fromarray(plane).save(source)
                control_output = folder / ("out_" + name)
                control_output.mkdir()
                with patch("batch_worm_roi.predict_raw",
                           side_effect=self._fake_predict([], labels)):
                    process_image(source, control_output, model="fluorescence weights",
                                  device=None, standard_count=1)
                controls[name] = self._qc_pixels(
                    control_output / OTHER_DIRNAME / (name + "_QC.png"))

            from_stack = self._qc_pixels(output / OTHER_DIRNAME / "two_planes_QC.png")
            self.assertTrue(
                np.array_equal(from_stack, controls["brightfield_only"]),
                "the QC was not drawn on the brightfield plane")
            self.assertFalse(
                np.array_equal(from_stack, controls["fluorescence_only"]),
                "the QC came from the fluorescence plane, which requirement 2 forbids")

            # The last three fields say which plane each half of the run used, and
            # that the feature was on. The monitor and the summary both read them.
            self.assertEqual(result[-3:], [1, 2, 1])

    def test_a_single_plane_tiff_is_still_processed_when_brightfield_is_on(self):
        # Decision 2: a folder holding both kinds is normal, and a single-plane
        # file has nothing to choose between -- it runs on the main model, takes
        # both its ROI and its numbers from its only plane, and says so.
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            stack, _, _ = self._two_planes(folder)
            single = folder / "single.tif"
            Image.fromarray(np.full((64, 64), 700, np.uint16)).save(single)
            output = folder / "out"
            output.mkdir()
            labels = np.zeros((64, 64), np.uint16)
            labels[15:50, 25:40] = 1
            seen = []

            with patch("batch_worm_roi.predict_raw",
                       side_effect=self._fake_predict(seen, labels)):
                stacked = process_image(
                    stack, output, model="fluorescence weights", device=None,
                    standard_count=1, fluorescence_plane=1, brightfield_plane=2,
                    brightfield_roi=self._brightfield_roi())
                alone = process_image(
                    single, output, model="fluorescence weights", device=None,
                    standard_count=1, fluorescence_plane=1, brightfield_plane=2,
                    brightfield_roi=self._brightfield_roi())

            self.assertEqual(seen[0][0], "brightfield weights")
            self.assertEqual(seen[1][0], "fluorescence weights",
                             "the single-plane file was segmented by the brightfield model")
            self.assertEqual(int(seen[1][1].min()), 700)

            self.assertEqual(stacked[-3:], [1, 2, 1])
            self.assertEqual(alone[-3:], [1, 1, 1],
                             "a single-plane file reported a plane it does not have")

    def test_a_plane_the_file_does_not_have_is_refused_and_named(self):
        """Out-of-range planes stop the run instead of measuring something else.

        The batch guard catches this before any file is touched, so the whole
        folder is refused rather than one image failing quietly; process_image
        carries the same check for callers that use it on a single file.
        """
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            stack, _, _ = self._two_planes(folder)
            output = folder / "out"

            with self.assertRaises(ValueError) as caught:
                run_gui_batch(str(folder), str(output), "worm.pt", "tip.pt",
                              brightfield_roi=True, fluorescence_plane=1,
                              brightfield_plane=4)
            self.assertIn("two_planes.tif", str(caught.exception))
            self.assertIn("4", str(caught.exception))
            self.assertFalse(output.exists())

            output.mkdir()
            seen = []
            with patch("batch_worm_roi.predict_raw", side_effect=self._fake_predict(
                    seen, np.zeros((64, 64), np.uint16))):
                with self.assertRaises(ValueError) as caught:
                    process_image(stack, output, model="fluorescence weights",
                                  device=None, standard_count=1,
                                  fluorescence_plane=1, brightfield_plane=4,
                                  brightfield_roi=self._brightfield_roi())
            self.assertIn("two_planes.tif", str(caught.exception))
            self.assertEqual(seen, [], "a file was segmented after the plane check failed")

    def test_a_brightfield_model_is_required_and_never_falls_back(self):
        # The one outcome nothing downstream can detect: a brightfield acquisition
        # segmented by weights trained on fluorescence gives ROIs that look
        # entirely reasonable and are not. So a missing checkpoint stops the batch
        # before a single file is opened.
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            self._two_planes(folder)
            for name in ("worm.pt", "tip.pt"):
                (folder / name).write_bytes(b"stub checkpoint")
            output = folder / "out"

            def refuse_to_predict(*args, **kwargs):
                raise AssertionError("predict_raw ran without brightfield weights")

            with patch("batch_worm_roi.torch.load", return_value={"model_state": {}}), \
                    patch("batch_worm_roi.WormUNet"), \
                    patch("batch_worm_roi.predict_raw", side_effect=refuse_to_predict):
                with self.assertRaises(FileNotFoundError) as caught:
                    run_gui_batch(
                        str(folder), str(output), str(folder / "worm.pt"),
                        str(folder / "tip.pt"), disable_tip_refinement=True,
                        brightfield_roi=True, fluorescence_plane=1,
                        brightfield_plane=2,
                        brightfield_checkpoint_path=str(folder / "not_there.pt"),
                        brightfield_tip_checkpoint_path=str(folder / "not_there_tip.pt"))
            self.assertIn("Brightfield checkpoint not found", str(caught.exception))

    def test_run_gui_batch_wires_the_planes_into_the_summary(self):
        # The summary is what the monitor and the ImageJ bridge both read, so the
        # three new columns have to be filled from this batch's planes and not
        # from whatever the window happens to have configured. The stand-in builds
        # its own last three fields out of the keywords it is called with, which
        # makes the row a record of what reached the per-image call rather than a
        # copy of what the test expects.
        calls = []

        def fake_process_image(path, *args, **kwargs):
            calls.append(kwargs)
            roi = kwargs.get("brightfield_roi")
            return [os.path.basename(str(path)), 1, 1, "PASS"] + [""] * 25 + [
                int(roi is not None),
                kwargs.get("brightfield_plane", 1) if roi is not None else 1,
                kwargs.get("fluorescence_plane", 1) if roi is not None else 1]

        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            self._two_planes(folder)
            for name in ("worm.pt", "tip.pt", "bf.pt", "bf_tip.pt"):
                (folder / name).write_bytes(b"stub checkpoint")
            output = folder / "out"

            with patch("batch_worm_roi.torch.load", return_value={"model_state": {}}), \
                    patch("batch_worm_roi.WormUNet"), \
                    patch("batch_worm_roi.process_image", side_effect=fake_process_image):
                run_gui_batch(
                    str(folder), str(output), str(folder / "worm.pt"),
                    str(folder / "tip.pt"), disable_tip_refinement=True,
                    brightfield_roi=True, fluorescence_plane=2, brightfield_plane=1,
                    brightfield_checkpoint_path=str(folder / "bf.pt"),
                    brightfield_tip_checkpoint_path=str(folder / "bf_tip.pt"))

            with (output / OTHER_DIRNAME / "batch_summary.csv").open(
                    newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1, rows)
            self.assertEqual(rows[0]["brightfield_roi_enabled"], "1")
            self.assertEqual(rows[0]["qc_plane"], "1")
            self.assertEqual(rows[0]["measured_plane"], "2")
            # And the planes reached the image itself: a summary filled in
            # correctly while process_image was told something else would describe
            # a run that never happened.
            self.assertEqual(len(calls), 1, calls)
            self.assertEqual(calls[0]["fluorescence_plane"], 2)
            self.assertEqual(calls[0]["brightfield_plane"], 1)
            self.assertIsNotNone(calls[0]["brightfield_roi"])

    def test_the_monitor_caption_names_the_plane(self):
        class Stub:
            current_folder = Path("C:/images")
            qc_planes = {"two_planes.tif": (2, 1), "single.tif": (1, 1)}
            _planes_for_qc = worm_roi_gui.WormRoiGui._planes_for_qc

            def _source_path_for_qc(self, qc_path):
                return None

            def _tr(self, chinese, english):
                return chinese

        label = worm_roi_gui.WormRoiGui._source_label_for_qc(
            Stub(), Path("two_planes_QC.png"))
        self.assertIn("第 2 层", label)
        self.assertIn("明场", label)

        # A single-plane file of the same batch has no plane to name: its QC came
        # from plane 1 like every other single-plane run, so calling it a
        # brightfield QC would be inventing a fact.
        plain = worm_roi_gui.WormRoiGui._source_label_for_qc(
            Stub(), Path("single_QC.png"))
        self.assertNotIn("明场", plain)

    def test_brightfield_roi_is_refused_against_an_old_bridge_protocol(self):
        # New window, old jar: the plug-in does not know about plane numbers and
        # would measure slice 1 of every image. With Brightfield ROI on, every row
        # would then be silently wrong, so the run is refused before it starts.
        class Stub:
            _start_processing = worm_roi_gui.WormRoiGui._start_processing

            def __init__(self):
                self.process = None
                self.root = None
                self.brightfield_roi_var = _FakeVar(True)
                self.said = []

            def _tr(self, chinese, english):
                return chinese

            def _terminal_write(self, text, tag):
                self.said.append(text)

        stub = Stub()
        with patch.object(worm_roi_gui, "IMAGEJ_MEASUREMENT_MODE", True), \
                patch.object(worm_roi_gui, "IMAGEJ_BRIDGE_PROTOCOL", ""), \
                patch.object(worm_roi_gui.messagebox, "showerror") as showerror, \
                patch.object(worm_roi_gui.messagebox, "showinfo") as showinfo:
            stub._start_processing()

        showerror.assert_called_once()
        self.assertIn("同版本", showerror.call_args[0][0])
        showinfo.assert_not_called()
        self.assertEqual(stub.said, [], "the batch was started anyway")

    def test_enhanced_tiff_rgb_reads_the_requested_plane(self):
        # The annotation view is drawn on the brightfield plane, which is not
        # plane 1. The default has to stay plane 1 for every older caller.
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            path = self._stack(folder, "annotate.tif", [
                np.full((32, 32), 0, np.uint8),
                np.full((32, 32), 200, np.uint8),
            ])
            first = np.asarray(enhanced_tiff_rgb(path))
            second = np.asarray(enhanced_tiff_rgb(path, plane=2))
            self.assertGreater(int(second.max()), int(first.max()),
                               "plane 2 was not the one that was read")
            self.assertTrue(np.array_equal(first, np.asarray(enhanced_tiff_rgb(path, 1))))

            with self.assertRaises(ValueError) as caught:
                enhanced_tiff_rgb(path, plane=3)
            self.assertIn("annotate.tif", str(caught.exception))


class SuiteIntegrityTests(unittest.TestCase):
    """The suite must test this repository's code, not some other copy of it.

    A file named test_*.py that is not a test still gets imported by discover,
    and if it puts another workspace on sys.path and imports a module at module
    level, sys.modules keeps that copy for the whole run -- sys.path order stops
    mattering once the name is bound. That happened: tests/erosion_0101_probe.py
    (then named test_erosion_0101.py) bound inspect_roi_dataset to the 0.3.2
    workspace, so batch_worm_roi and every assertion about pixel handling were
    exercising 0.3.2 code while appearing to pass. Nothing failed; the tests just
    stopped being about this repository.
    """

    def test_modules_under_test_come_from_this_repository(self):
        import batch_worm_roi
        import inspect_roi_dataset
        import worm_roi_gui

        for module in (batch_worm_roi, inspect_roi_dataset, worm_roi_gui):
            with self.subTest(module.__name__):
                path = Path(module.__file__).resolve()
                self.assertTrue(
                    path.is_relative_to(ROOT),
                    "%s was imported from %s, outside %s -- another workspace is "
                    "on sys.path ahead of this one" % (module.__name__, path, ROOT))
                self.assertEqual(path.parent, (ROOT / "src").resolve())


class InputSnapshotTests(unittest.TestCase):
    """The batch must process the list it was given, not re-scan the folder.

    The GUI checks the folder, shows the result, and keeps that list as the
    completion manifest it reports to ImageJ. If the batch is left to glob the
    directory a second time, an image that arrives in between -- an acquisition
    or a sync still writing into the folder -- gets segmented and exported, but
    is missing from the manifest, so ImageJ never measures it. The ROI ZIP and
    the QC image are both there and batch_summary.csv calls it a success, which
    is exactly why nobody would notice the missing fluorescence.

    The engine is stubbed: globbing is decided before any model runs, so a real
    checkpoint would add nothing here.
    """

    def _stub_image_engine(self, on_process=None):
        """Patch enough of the pipeline to run a batch without weights."""
        def fake_process_image(path, *args, **kwargs):
            name = os.path.basename(str(path))
            if on_process is not None:
                on_process(name)
            return [name, 10, 10, "PASS"] + [""] * 28

        return patch("batch_worm_roi.process_image", side_effect=fake_process_image)

    def _run(self, folder, output, input_paths, on_process=None):
        for name in ("worm.pt", "tip.pt"):
            (folder / name).write_bytes(b"stub checkpoint")
        with patch("batch_worm_roi.torch.load", return_value={"model_state": {}}), \
                patch("batch_worm_roi.WormUNet"), \
                self._stub_image_engine(on_process):
            return run_gui_batch(
                str(folder), str(output), str(folder / "worm.pt"),
                str(folder / "tip.pt"), disable_tip_refinement=True,
                input_paths=input_paths)

    def test_an_image_added_mid_batch_is_not_processed_when_the_list_is_given(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            for name in ("a.tif", "b.tif"):
                Image.fromarray(np.zeros((8, 8), np.uint16)).save(folder / name)
            output = folder / "out"
            snapshot = sorted(str(path) for path in folder.glob("*.tif"))

            def arrive_late(name):
                # A file lands in the folder while the batch is running.
                if not (folder / "late.tif").exists():
                    Image.fromarray(np.zeros((8, 8), np.uint16)).save(folder / "late.tif")

            processed = []

            def on_process(name):
                processed.append(name)
                arrive_late(name)

            results = self._run(folder, output, snapshot, on_process=on_process)

            self.assertEqual(sorted(processed), ["a.tif", "b.tif"],
                             "the batch processed an image that was not in the list "
                             "it was given; the GUI has no name to report to ImageJ "
                             "for it, so it would never be measured")
            self.assertNotIn("late.tif", [row[0] for row in results])

    def test_the_scan_happens_once_before_the_loop(self):
        """Why the window is the pre-check, not each image.

        run_gui_batch globs the folder once and keeps that list, so a file that
        arrives while image 3 of 10 is being processed is not picked up by this
        batch. The gap that matters is therefore earlier: between the GUI's
        pre-check and this call, which is where the pre-flight checks run. This
        test pins the scan-once behaviour so that a future change to per-image
        rescanning does not silently reopen the race in a new place.
        """
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            for name in ("a.tif", "b.tif"):
                Image.fromarray(np.zeros((8, 8), np.uint16)).save(folder / name)
            output = folder / "out"

            def arrive_late(name):
                if name == "a.tif" and not (folder / "late.tif").exists():
                    Image.fromarray(np.zeros((8, 8), np.uint16)).save(folder / "late.tif")

            processed = []

            def on_process(name):
                processed.append(name)
                arrive_late(name)

            results = self._run(folder, output, None, on_process=on_process)

            self.assertEqual(sorted(processed), ["a.tif", "b.tif"])
            # The file really did land, so the assertion above is not vacuous.
            self.assertTrue((folder / "late.tif").exists())


if __name__ == "__main__":
    unittest.main()
