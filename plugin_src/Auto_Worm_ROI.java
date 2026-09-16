import ij.IJ;
import ij.ImagePlus;
import ij.Prefs;
import ij.WindowManager;
import ij.gui.Roi;
import ij.io.FileInfo;
import ij.io.OpenDialog;
import ij.io.RoiDecoder;
import ij.measure.Calibration;
import ij.measure.Measurements;
import ij.measure.ResultsTable;
import ij.plugin.PlugIn;
import ij.plugin.frame.RoiManager;
import ij.process.ImageStatistics;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Properties;
import java.util.Set;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/** ImageJ 1.x front-end for the CUDA Auto Worm ROI engine. */
public class Auto_Worm_ROI implements PlugIn {
    private static final String VERSION = "0.4.2";
    private static final String PREF_GUI = "autoworm.imagej.gui";
    private static final String ROI_PREFIX = "AutoWorm:";
    /**
     * The output folder is handed to us as one directory; the GUI program splits
     * its contents into these two. The ROI sets and QC overlays we read are in
     * OTHER_DIR, the tables we write are in MEASUREMENTS_DIR.
     *
     * Kept as two literals on purpose: the Python side has the same two names in
     * batch_worm_roi.py (MEASUREMENTS_DIRNAME / OTHER_DIRNAME) and nothing checks
     * that they agree, so renaming one means renaming the other.
     */
    private static final String MEASUREMENTS_DIR = "measurements";
    private static final String OTHER_DIR = "other";
    private static final int MEASURES = Measurements.AREA | Measurements.MEAN |
            Measurements.MIN_MAX | Measurements.INTEGRATED_DENSITY | Measurements.MEDIAN;

    @Override
    public void run(String arg) {
        try {
            execute();
        } catch (Exception error) {
            IJ.handleException(error);
            IJ.error("自动圈虫 " + VERSION, error.getMessage() == null ? error.toString() : error.getMessage());
        }
    }

    private void execute() throws Exception {
        ImagePlus active = WindowManager.getCurrentImage();
        File activeFile = savedTiff(active);
        if (!confirmUnsavedChanges(active)) {
            IJ.showStatus("Nothing was processed: the current image has unsaved changes.");
            return;
        }
        if (!confirmSameCalibration(active, activeFile)) {
            IJ.showStatus("Nothing was processed: the calibration on screen differs from the saved file.");
            return;
        }
        File gui = locateGui();
        Path bridge = Files.createTempDirectory("autoworm-imagej-");

        List<String> command = new ArrayList<>();
        command.add(gui.getAbsolutePath());
        if (activeFile != null) command.add(activeFile.getAbsolutePath());
        ProcessBuilder builder = new ProcessBuilder(command);
        builder.redirectErrorStream(true);
        builder.environment().put("AUTOWORM_IMAGEJ_MODE", "1");
        builder.environment().put("AUTOWORM_IMAGEJ_BRIDGE_DIR", bridge.toString());
        IJ.log("[Auto Worm] Original UI: " + gui);
        IJ.showStatus("The GUI has been opened. Please perform image processing in it.");
        final Process process = builder.start();
        Thread logPump = new Thread(new Runnable() {
            @Override
            public void run() {
                try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                        process.getInputStream(), Charset.defaultCharset()))) {
                    String line;
                    while ((line = reader.readLine()) != null) IJ.log("[Auto Worm GUI] " + line);
                } catch (IOException error) {
                    IJ.log("[Auto Worm GUI] 日志读取结束：" + error.getMessage());
                }
            }
        }, "AutoWorm-GUI-log");
        logPump.setDaemon(true);
        logPump.start();

        int completedBatches = pumpNotifications(bridge, process, new BatchHandler() {
            @Override
            public Handled handle(Path notification, Properties values) {
                String status = values.getProperty("status", "error");
                if ("complete".equals(status)) {
                    String message = decode(values.getProperty("message_b64", ""));
                    File inputFolder = new File(decode(values.getProperty("input_b64", "")));
                    File outputFolder = new File(decode(values.getProperty("output_b64", "")));
                    Set<String> finishedImages = successfulImages(values);
                    // An explicitly empty list means the batch finished and produced
                    // no ROI at all -- every image failed to segment. Measuring then
                    // can only fail ("输出目录中没有可测量的 ROI ZIP"), and that
                    // measurement error would be the only thing the user sees: the
                    // message below, which names each failed image and why, is shown
                    // only alongside a successful measurement. Reporting the real
                    // reason is the whole point of the message, so this branch stops
                    // before measuring. null is different: an older GUI sends no list
                    // at all, and there the ROI ZIPs are still the only evidence.
                    if (finishedImages != null && finishedImages.isEmpty()) {
                        showBatchMessage(message, "本批没有可测量的图像");
                        return Handled.UNMEASURED;
                    }
                    boolean measured = false;
                    try {
                        ImagePlus current = WindowManager.getCurrentImage();
                        File currentFile = savedTiff(current);
                        measureOutputFolder(inputFolder, outputFolder, currentFile,
                                finishedImages, true, true);
                        measured = true;
                        IJ.showStatus("Measurement complete. The GUI stays open; you can process another folder.");
                    } catch (Exception error) {
                        IJ.handleException(error);
                        IJ.error("自动圈虫 " + VERSION, "ImageJ 测量失败：\n" +
                                (error.getMessage() == null ? error.toString() : error.getMessage()));
                    }
                    // A batch succeeds even when individual images failed: they were
                    // skipped, so they are simply absent from the table above. This
                    // message is the only place that says so.
                    if (measured) showBatchMessage(message, "部分图像未处理");
                    return measured ? Handled.MEASURED : Handled.UNMEASURED;
                }
                if ("cancelled".equals(status)) {
                    IJ.showStatus("Processing was cancelled. The GUI stays open.");
                    return Handled.OTHER;
                }
                if ("closed".equals(status)) return Handled.CLOSED;
                if ("error".equals(status)) {
                    String message = decode(values.getProperty("message_b64", ""));
                    IJ.error("自动圈虫 " + VERSION, message.isEmpty() ?
                            "原界面处理失败，请查看 Auto Worm 日志。" : message);
                    return Handled.OTHER;
                }
                IJ.log("[Auto Worm] 未知桥接状态，忽略：" + status);
                return Handled.OTHER;
            }
        });
        int exitCode = process.waitFor();
        removeBridge(bridge);
        if (exitCode != 0) throw new IOException("原界面异常退出（代码 " + exitCode + "）。");
        if (completedBatches == 0)
            IJ.showStatus("The GUI was closed without any ImageJ measurement.");
        else
            // "batches completed: N" rather than "N batches" so that a single batch
            // does not need plural agreement in a status string with no formatting.
            IJ.showStatus("The GUI was closed. ImageJ measurement batches completed: " + completedBatches);
    }

    /** What handling one notification came to. */
    enum Handled {
        /** A completed batch, measured. */
        MEASURED,
        /** A completed batch whose measurement failed; it was reported and let go. */
        UNMEASURED,
        /** The GUI is closing. Nothing further is read out of the current pass. */
        CLOSED,
        /** A cancellation, a failure, or a status this version does not know. */
        OTHER
    }

    /**
     * Handles one notification, once, in file-name order.
     *
     * Not private, and neither is the method below: they are what
     * tests/AutoWormBridgeQueueTest drives the queue through, and no other class
     * in this package exists.
     */
    interface BatchHandler {
        Handled handle(Path notification, Properties values);
    }

    /**
     * Reads and handles notifications until the GUI has exited and none are left.
     * Returns the number of batches measured.
     *
     * A notification is deleted once it has been handled, not when it is read.
     * The GUI watches this directory to decide whether it may start the next
     * batch, and a read takes milliseconds where a measurement takes minutes:
     * deleting on read opens that gate while the ROIs of the batch being measured
     * are still being read, and a second batch writing to the same output folder
     * would then replace them mid-measurement.
     *
     * The queue is also emptied after the GUI exits. A batch that finishes while
     * ImageJ is still measuring the previous one waits in it -- that is what it is
     * for -- and closing the window is usually the last thing the user does, so
     * stopping there would leave that batch with its ROIs on disk and no
     * measurements at all.
     */
    static int pumpNotifications(Path bridge, Process process, BatchHandler handler)
            throws InterruptedException {
        int measured = 0;
        // A notification is handled once. Its file is normally gone by the time
        // the next pass starts, so these two are for the case where deleting it
        // failed: the file is still there to be read again, and reading it again
        // would measure the same batch a second time.
        Set<String> handled = new HashSet<>();
        Set<String> undeletable = new HashSet<>();
        while (true) {
            // Sampled before the scan, never after it.
            //
            // The GUI writes its last notification and exits as one step, so a
            // check taken after a scan can see a dead process and conclude the
            // queue is finished while a notification written during that same scan
            // is still sitting on disk. That notification is the last batch of the
            // session, and it would be dropped with its ROIs unmeasured.
            //
            // Sampling first makes the answer stale in the one direction that is
            // safe: a process found alive here has already written everything it
            // ever will by the time the scan below runs, and a process found dead
            // here is scanned once more anyway.
            boolean alive = process.isAlive();
            // True when this pass got rid of at least one notification. Only the
            // drain below needs it: it looks again while something is happening,
            // and a pass that changes nothing is what ends it -- which is also
            // what keeps a file that cannot be deleted from being retried forever.
            boolean progressed = false;
            // One file per notification, so a batch that finishes while ImageJ is
            // still measuring the previous one waits in the queue instead of
            // overwriting it. The names carry the order.
            for (Path notification : pendingNotifications(bridge)) {
                String name = notification.getFileName().toString();
                if (!handled.add(name)) {
                    // Read on an earlier pass and still here, so deleting it failed
                    // then. Try again, and say so once: while it is here the queue
                    // is never empty, and that keeps the GUI from starting the next
                    // batch. It is not read a second time.
                    if (!removeNotification(notification) && undeletable.add(name))
                        IJ.log("[Auto Worm] 桥接通知无法删除：" + name);
                    continue;
                }
                Properties values;
                try {
                    values = readNotification(notification);
                } catch (IOException error) {
                    IJ.log("[Auto Worm] 桥接通知无法读取，已丢弃：" + name +
                            "（" + error.getMessage() + "）");
                    // Dropped rather than left in place: it will never be readable,
                    // and leaving it would bring the loop back to it every 150 ms
                    // for as long as the GUI is open.
                    removeNotification(notification);
                    continue;
                }
                Handled outcome;
                try {
                    outcome = handler.handle(notification, values);
                } catch (Exception error) {
                    // One bad notification must not end the session: an exception
                    // through here would leave the GUI running with nobody reading
                    // what it sends. The batch is dropped rather than retried --
                    // a handler that failed after measuring would measure twice.
                    IJ.log("[Auto Worm] 桥接通知处理失败，已放弃该批：" + name +
                            "（" + error + "）");
                    outcome = Handled.OTHER;
                }
                if (outcome == Handled.MEASURED) measured++;
                // Handled, whatever the outcome: a batch that failed to measure was
                // reported to the user and is let go of. Keeping it would only
                // measure it again on the next pass.
                if (removeNotification(notification) || undeletable.add(name))
                    progressed = true;
                if (outcome == Handled.CLOSED) break;
            }
            if (!alive) {
                // The GUI had exited when this pass started, so nothing else is
                // coming: whatever is in the queue now is all there will ever be,
                // and it was just read. A pass that removed nothing means the queue
                // is empty, or holds nothing that will ever go away, and neither
                // changes by looking again.
                if (!progressed) break;
                continue;
            }
            Thread.sleep(150L);
        }
        return measured;
    }

    /**
     * True when the plug-in should carry on with the current image.
     *
     * The window may hold edits that were never written to the file the GUI
     * reads, and the two sides would then be looking at different pixels. This
     * is the only point where that can still be said out loud; the alternative
     * is a results table that quietly describes a different image than the one
     * on screen.
     */
    private static boolean confirmUnsavedChanges(ImagePlus active) {
        if (active == null || !active.changes) return true;
        return IJ.showMessageWithCancel("自动圈虫 " + VERSION,
                "当前图像有未保存的修改。\n\n" +
                "原界面打开的是磁盘上的文件，圈虫画的 ROI 和随后的测量都只针对保存过的版本，" +
                "屏幕上未保存的改动不会参与处理。\n\n" +
                "要把改动一起处理，请先在 ImageJ 中保存图像，再重新调用本功能。\n\n" +
                "点「确定」继续将忽略未保存的改动。");
    }

    /**
     * True when the calibration on screen and the one in the file agree, or when
     * the user says to go ahead with the file's.
     *
     * The check above watches ImagePlus.changes, which does not move when the
     * intensity calibration is changed: Analyze > Calibrate runs a filter that
     * declares NO_CHANGES, and ImagePlus.setCalibration() only stores the
     * Calibration and points it at the image. So a calibration set in the window
     * and never saved used to pass unnoticed, while the GUI and the measurement
     * both read the file -- the same picture, measured with numbers from a
     * differently calibrated image.
     *
     * Only the saved file can be compared against, so it is opened here: one
     * extra decode of the image the GUI is about to be handed, which the
     * measurement repeats later on anyway.
     */
    private static boolean confirmSameCalibration(ImagePlus active, File activeFile) {
        if (active == null || activeFile == null) return true;
        ImagePlus disk = IJ.openImage(activeFile.getAbsolutePath());
        if (disk == null) return true;   // unreadable is not this check's business
        try {
            String difference = calibrationDifference(active, disk);
            if (difference == null) return true;
            return IJ.showMessageWithCancel("自动圈虫 " + VERSION,
                    "当前窗口的校准与磁盘文件上的不同。\n\n" + difference + "\n\n" +
                    "原界面圈画和随后的测量都只针对磁盘上保存过的文件，用的是文件里的校准，" +
                    "窗口里临时设的校准不参与计算。\n\n" +
                    "要把窗口里的校准用于测量，请先把它保存进文件（File > Save），再重新" +
                    "调用本功能；Analyze > Set Scale 里的 Global 选项另当别论，它一旦勾选，" +
                    "所有图都按它算，与文件里的校准无关。\n\n" +
                    "点「确定」继续将使用磁盘文件上的校准。");
        } finally {
            disk.close();
        }
    }

    /**
     * How the calibration of the window differs from the one in the file on disk,
     * or null when both would give the same numbers.
     *
     * Only what a measurement can see is compared: the pixel size and its unit
     * (Area, IntDen) and the intensity function with its coefficients and value
     * unit (Mean, IntDen, Median, and everything derived from them). A global
     * calibration is not a difference -- ImagePlus.getCalibration() hands the
     * same global calibration to both sides, so both are comparing it against
     * itself, which is also why a global calibration overrides the file's.
     */
    static String calibrationDifference(ImagePlus window, ImagePlus disk) {
        if (window == null || disk == null) return null;
        Calibration onScreen = window.getCalibration();
        Calibration inFile = disk.getCalibration();
        List<String> parts = new ArrayList<>();
        // pixelDepth is deliberately not compared. It is the spacing between
        // slices, and every measurement made here is an Area, a Mean or a CTCF
        // count on one 2-D plane -- none of them can see it. Comparing it only
        // added a way for a stack that had been saved with a different slice
        // spacing to be reported as a calibration difference, with a dialog whose
        // text does not even mention depth and whose advice cannot be acted on.
        if (!samePixelSize(onScreen.pixelWidth, inFile.pixelWidth)
                || !samePixelSize(onScreen.pixelHeight, inFile.pixelHeight)
                || !unitOf(onScreen).equals(unitOf(inFile)))
            parts.add("空间校准：窗口 " + spaceText(onScreen) +
                    "；文件 " + spaceText(inFile));
        if (intensityDiffers(onScreen, inFile))
            parts.add("强度校准：窗口 " + intensityText(onScreen) +
                    "；文件 " + intensityText(inFile));
        return parts.isEmpty() ? null : String.join("\n", parts);
    }

    /**
     * Whether two pixel sizes are the same for measurement purposes.
     *
     * Deliberately not an exact comparison. A TIFF stores the spatial
     * calibration in tags of limited precision, so a scale the experimenter
     * typed, saved and reloaded comes back a few units in the last place from
     * the double the window still holds -- 0.123456789 saves and reloads as
     * 0.1234567901234568. Comparing exactly made every such file differ from its
     * own window and asked for confirmation of a difference that was not there,
     * on every call, with nothing the experimenter could do to stop it.
     *
     * The tolerance is relative, one part in a million: far coarser than that
     * quantisation, and far finer than any scale someone would set on purpose
     * (the nearest scale that differs by one part in a million is already past
     * the sixth significant figure of the stored value). A size of zero, unset,
     * compares equal only to another zero.
     */
    private static boolean samePixelSize(double a, double b) {
        if (Double.isNaN(a) || Double.isNaN(b)) return Double.isNaN(a) && Double.isNaN(b);
        return Math.abs(a - b) <= 1e-6 * Math.max(Math.abs(a), Math.abs(b));
    }

    /** What one image's measurements are calibrated by, for the log. */
    private static String describeCalibration(Calibration calibration) {
        if (calibration == null) return "ImageJ 默认（1 x 1 pixel，未做强度校准）";
        return "空间 " + spaceText(calibration) + "；强度 " + intensityText(calibration);
    }

    private static boolean intensityDiffers(Calibration a, Calibration b) {
        boolean aCalibrated = a.getFunction() != Calibration.NONE;
        boolean bCalibrated = b.getFunction() != Calibration.NONE;
        if (aCalibrated != bCalibrated) return true;
        if (!valueUnitOf(a).equals(valueUnitOf(b))) return true;
        if (!aCalibrated) return false;
        return a.getFunction() != b.getFunction()
                || !Arrays.equals(a.getCoefficients(), b.getCoefficients());
    }

    private static String unitOf(Calibration calibration) {
        String unit = calibration.getUnit();
        return unit == null ? "" : unit;
    }

    private static String valueUnitOf(Calibration calibration) {
        String unit = calibration.getValueUnit();
        return unit == null ? Calibration.DEFAULT_VALUE_UNIT : unit;
    }

    private static String spaceText(Calibration calibration) {
        String unit = unitOf(calibration);
        return trim(calibration.pixelWidth) + " x " + trim(calibration.pixelHeight) +
                (unit.isEmpty() ? " pixel" : " " + unit);
    }

    private static String intensityText(Calibration calibration) {
        return (calibration.getFunction() == Calibration.NONE
                ? "未校准" : functionName(calibration.getFunction())) +
                "（" + valueUnitOf(calibration) + "）";
    }

    /**
     * The intensity calibration as a column value: "None", or the function with
     * its coefficients where they are short enough to be worth spelling out.
     */
    static String intensityFunctionText(Calibration calibration) {
        if (calibration == null || calibration.getFunction() == Calibration.NONE)
            return "None";
        double[] coefficients = calibration.getCoefficients();
        if (calibration.getFunction() == Calibration.STRAIGHT_LINE
                && coefficients != null && coefficients.length >= 2)
            return "Straight Line: y = " + trim(coefficients[0]) + " + " +
                    trim(coefficients[1]) + "*x";
        return functionName(calibration.getFunction());
    }

    /** The names the Calibrator dialog shows, for the functions it offers. */
    private static String functionName(int function) {
        switch (function) {
            case Calibration.STRAIGHT_LINE:    return "Straight Line";
            case Calibration.POLY2:            return "2nd Degree Polynomial";
            case Calibration.POLY3:            return "3rd Degree Polynomial";
            case Calibration.POLY4:            return "4th Degree Polynomial";
            case Calibration.EXPONENTIAL:      return "Exponential";
            case Calibration.POWER:            return "Power";
            case Calibration.LOG:              return "Log";
            case Calibration.RODBARD:          return "Rodbard";
            case Calibration.GAMMA_VARIATE:    return "Gamma Variate";
            case Calibration.LOG2:             return "Log2";
            case Calibration.RODBARD2:         return "Rodbard (NIH Image)";
            case Calibration.EXP_WITH_OFFSET:  return "Exponential with Offset";
            case Calibration.GAUSSIAN:         return "Gaussian";
            case Calibration.EXP_RECOVERY:     return "Exponential Recovery";
            case Calibration.UNCALIBRATED_OD:  return "Uncalibrated OD";
            case Calibration.CUSTOM:           return "Custom";
            default:                           return "Function " + function;
        }
    }

    /** A number as it should read in a message: 1, not 1.0; otherwise full precision. */
    private static String trim(double value) {
        if (value == Math.rint(value) && !Double.isInfinite(value))
            return String.valueOf((long) value);
        return String.valueOf(value);
    }

    /**
     * The images the GUI reports as finished, or null when it sent no list.
     *
     * A ROI ZIP is written before an image has been fully processed, so an image
     * that fails later leaves one behind whenever the partial results cannot be
     * deleted (a viewer holding the file open, for instance). Measuring by the
     * presence of a ZIP alone would therefore measure a batch's failed images
     * too, and hand back numbers for an image the log calls FAILED.
     *
     * The names are compared as they are: both sides list the same folder.
     */
    /**
     * Shows the GUI's own account of a batch, in the log and in front of the user.
     *
     * The message names the images that failed and the reason each one did. Fiji has
     * no other way to learn it: those images are absent from the measurement table,
     * and the GUI window has already been told it can close. A batch that measured
     * nothing is exactly when this matters most, so the empty-message case is
     * skipped rather than shown as a dialog with nothing in it.
     */
    private static void showBatchMessage(String message, String title) {
        if (message.isEmpty()) return;
        IJ.log("[Auto Worm] " + message);
        IJ.error("自动圈虫 " + VERSION + "：" + title, message);
    }

    private static Set<String> successfulImages(Properties values) {
        String count = values.getProperty("ok_count");
        if (count == null) return null;
        Set<String> names = new HashSet<>();
        for (String name : decode(values.getProperty("ok_b64", "")).split("\n")) {
            if (!name.isEmpty()) names.add(name);
        }
        if (!count.equals(String.valueOf(names.size())))
            IJ.log("[Auto Worm] 桥接清单与计数不符（" + count + " 对 " + names.size() + "），按清单测量。");
        return names;
    }

    /** Bridge notifications waiting to be read, oldest first (the names carry the order). */
    private static List<Path> pendingNotifications(Path bridge) {
        List<Path> pending = new ArrayList<>();
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(bridge, "*.properties")) {
            for (Path path : stream) pending.add(path);
        } catch (IOException error) {
            IJ.log("[Auto Worm] 无法读取桥接目录：" + error.getMessage());
            return pending;
        }
        Collections.sort(pending);
        return pending;
    }

    /**
     * The values in one notification. The file itself is left alone.
     *
     * Deleting it here would say "read" where the GUI is asking "measured": it
     * watches this directory to know whether the plug-in has finished with the
     * previous batch, and a measurement takes minutes while a read takes
     * milliseconds. A batch started in that window writes ROIs over the ones
     * still being read and the two batches end up in one table.
     */
    private static Properties readNotification(Path notification) throws IOException {
        Properties values = new Properties();
        try (BufferedReader reader = Files.newBufferedReader(notification, StandardCharsets.US_ASCII)) {
            values.load(reader);
        }
        return values;
    }

    /** Deletes one notification, reporting whether it is now gone. */
    private static boolean removeNotification(Path notification) {
        try {
            Files.deleteIfExists(notification);
            return true;
        } catch (IOException error) {
            return false;
        }
    }

    /** Best-effort removal of the queue directory; the system temp area reclaims it anyway. */
    private static void removeBridge(Path bridge) {
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(bridge)) {
            for (Path path : stream) Files.deleteIfExists(path);
        } catch (IOException ignored) {
            // Leftovers mean the GUI died mid-write; they are only temp files.
        }
        try {
            Files.deleteIfExists(bridge);
        } catch (IOException ignored) {
        }
    }

    /**
     * Deletes the combined table left behind by an earlier batch, before this one
     * starts measuring.
     *
     * The table is only written once a batch has measured something, so without
     * this a batch that measures nothing -- every image failed to segment, the
     * window was closed before anything finished -- leaves the previous batch's
     * numbers sitting in the output folder under this batch's name. Nothing in the
     * file says which run produced it, so afterwards the two cannot be told apart.
     * Deleting up front is what makes "no table" mean "nothing was measured".
     *
     * The header is checked first. This is the experimenter's own output folder,
     * and a file that happens to carry our name is not proof that we wrote it. The
     * check is for the two column names whose presence identifies the table rather
     * than the whole header, so that a table written by an older build, with a
     * different column order, is still recognised as ours.
     */
    /**
     * True when the CSV header line carries exactly these column names.
     *
     * Whole fields are compared, not the header as a string. A substring test is
     * the obvious way to write this and it is wrong in the direction that destroys
     * data: the header "Image,RawCTCFValue,MeasurementBackendNotes,Comment" contains
     * both "CTCF" and "MeasurementBackend" as substrings, so a substring test would
     * delete a file we cannot prove is ours. Matching whole fields is what makes the
     * header check mean what it says -- a table whose provenance we cannot establish
     * is left where it is.
     */
    private static boolean hasColumns(String header, String... columns) {
        Set<String> fields = new HashSet<>();
        for (String field : header.split(",", -1)) fields.add(field.trim());
        for (String column : columns) {
            if (!fields.contains(column)) return false;
        }
        return true;
    }

    private static void removeStaleCombinedTable(File measurementsDir) {
        File table = new File(measurementsDir, "ImageJ_measurements_all.csv");
        if (!table.isFile()) return;
        try (BufferedReader reader = Files.newBufferedReader(table.toPath(), StandardCharsets.UTF_8)) {
            String header = reader.readLine();
            if (header == null || !hasColumns(header, "CTCF", "MeasurementBackend"))
                return;
        } catch (IOException unreadable) {
            return;
        }
        if (table.delete()) IJ.log("[Auto Worm] 已删除上一批留下的合并测量表：" + table.getName());
        else IJ.log("[Auto Worm] 上一批的合并测量表删不掉，本批结果会覆盖它：" + table.getName());
    }

    /**
     * Removes this image's table from an earlier run, before this one measures it.
     *
     * Called only for the images this pass is really going to measure, so that
     * afterwards a <stem>_measurements.csv in the output folder means "this run
     * measured this image". Without it, an image that fails here -- a truncated
     * ROI ZIP, a file that stopped being a readable TIFF -- keeps the table an
     * earlier run left under the same name. That table reads as this run's
     * result: the ROIs beside it are this run's, the log line saying the image
     * was skipped is in a window that gets closed, and the numbers are from data
     * that has since been replaced. Nothing in the file contradicts any of it.
     *
     * The header is checked the same way removeStaleCombinedTable checks its own,
     * and for the same reason: a file in the experimenter's output folder that
     * merely carries our name is left alone.
     */
    private static void removeStaleImageTable(File measurementsDir, String imageName) {
        File table = new File(measurementsDir, stem(imageName) + "_measurements.csv");
        if (!table.isFile()) return;
        try (BufferedReader reader = Files.newBufferedReader(table.toPath(), StandardCharsets.UTF_8)) {
            String header = reader.readLine();
            if (header == null || !hasColumns(header, "CTCF", "MeasurementBackend"))
                return;
        } catch (IOException unreadable) {
            return;
        }
        if (table.delete()) IJ.log("[Auto Worm] 已删除上一轮留下的单图测量表，本轮将重新测量：" +
                table.getName());
    }

    static void measureOutputFolder(File inputFolder, File outputFolder,
                                    File activeFile, Set<String> finishedImages,
                                    boolean addToManager, boolean showQc) throws Exception {
        if (!inputFolder.isDirectory()) throw new IOException("输入文件夹不存在：" + inputFolder);
        if (!outputFolder.isDirectory()) throw new IOException("输出文件夹不存在：" + outputFolder);
        // The output folder holds the two subfolders the GUI program fills in.
        // Neither is created here: a folder without them is a folder from an older
        // version (or one the experimenter picked by hand), and the honest thing is
        // to find nothing in it and say so, not to leave two empty directories in
        // somebody's results.
        File otherDir = new File(outputFolder, OTHER_DIR);
        File measurementsDir = new File(outputFolder, MEASUREMENTS_DIR);
        File[] inputs = listTiffs(inputFolder);
        removeStaleCombinedTable(measurementsDir);
        ResultsTable combined = new ResultsTable();
        int measured = 0;
        int skipped = 0;
        boolean saidWhichPixels = false;
        for (File input : inputs) {
            // The GUI's list of images it finished is what decides; without it
            // (an older GUI sends none) fall back to the ROI ZIP, which is only
            // a guess about whether the image got that far.
            if (finishedImages != null && !finishedImages.contains(input.getName())) continue;
            File roiZip = new File(otherDir, stem(input.getName()) + "_RoiSet.zip");
            if (!roiZip.isFile()) continue;
            // From here on this image is one this pass will either measure or
            // report as skipped, so its table from an earlier run must not
            // survive. Images the loop has already skipped keep theirs: they are
            // not part of this pass, and the folder may be being filled image by
            // image.
            removeStaleImageTable(measurementsDir, input.getName());
            // Always measure the file itself, never a window copy of it: the ROI
            // set was produced from that file by another process, and the window
            // may have been edited since ImageJ read it.
            ImagePlus image = IJ.openImage(input.getAbsolutePath());
            if (image == null) {
                skipped++;
                IJ.log("[Auto Worm] ImageJ 无法打开，跳过测量：" + input);
                continue;
            }
            try {
                if (measured == 0) {
                    // Said once per batch: what each row was measured with is in
                    // the table itself (the calibration columns), but a log read
                    // afterwards should be able to explain why the numbers differ
                    // from another session's on the same folder.
                    Calibration global = ImagePlus.getStaticGlobalCalibration();
                    IJ.log("[Auto Worm] 本次测量使用的校准（首张图 " + input.getName() + "）：" +
                            describeCalibration(image.getCalibration()) +
                            (global == null ? "" : "；全局标定生效，覆盖各图自己的校准"));
                }
                if (!saidWhichPixels && sameFile(input, activeFile)) {
                    saidWhichPixels = true;
                    ImagePlus current = WindowManager.getCurrentImage();
                    if (current != null && current.changes)
                        IJ.log("[Auto Worm] 当前图像有未保存的修改；本次测量的是磁盘上的文件：" +
                                input.getName());
                    String difference = current == null ? null
                            : calibrationDifference(current, image);
                    if (difference != null)
                        IJ.log("[Auto Worm] 当前窗口的校准与磁盘文件不同，测量用的是文件上的" +
                                "校准：" + difference);
                }
                Roi[] rois = loadRois(roiZip);
                MeasuredImage measuredImage = measureWithImageJ(image, input.getName(), rois);
                // The image's own table is written and made durable BEFORE its
                // rows join the combined one. If this write fails the image is
                // reported as skipped, and a combined table that still carried
                // its rows would contradict both the log and the missing file
                // beside it -- the numbers in it are real, but they are numbers
                // the log says were not measured.
                saveTable(measuredImage.table, new File(measurementsDir,
                        stem(input.getName()) + "_measurements.csv"));
                for (Row row : measuredImage.rows) appendMeasurement(combined, row);
                measured++;
                if (addToManager && sameFile(input, activeFile)) addToRoiManager(rois);
            } catch (Exception error) {
                // One image that cannot be measured must not cost the others their
                // numbers. A batch is normally a folder full of images, and the
                // usual reason one of them fails -- a truncated ROI ZIP, a file
                // that is no longer a readable TIFF -- says nothing about the rest.
                // What was measured is reported as measured, and the image is
                // named in the log so a short table is explainable.
                skipped++;
                IJ.log("[Auto Worm] 无法测量，已跳过：" + input.getName() + "（" + error + "）");
            } finally {
                image.close();
            }
        }
        if (measured == 0) throw new IOException(skipped == 0
                ? "输出目录中没有可测量的 ROI ZIP。\n找的是：" + otherDir
                  + "\n（本版本的 ROI 与 QC 图放在输出目录的 " + OTHER_DIR + " 子目录下）"
                : "没有一张图像测量成功，已跳过 " + skipped + " 张；详见 ImageJ 日志。");
        saveTable(combined, new File(measurementsDir, "ImageJ_measurements_all.csv"));
        combined.show("Auto Worm ImageJ Measurements");
        if (showQc && activeFile != null) {
            File qc = new File(otherDir, stem(activeFile.getName()) + "_QC.png");
            if (qc.isFile()) IJ.open(qc.getAbsolutePath());
        }
        IJ.showStatus("ImageJ measurement complete. Images measured: " + measured);
        IJ.log("自动圈虫 " + VERSION + "：处理完成：" + measured + " 张图像；测量后端：ImageJ " +
                IJ.getVersion() + "；测量表：" + measurementsDir + "；ROI 与 QC 图：" + otherDir +
                (skipped > 0 ? "；跳过 " + skipped + " 张（见上方日志）" : ""));
    }

    private static String decode(String encoded) {
        if (encoded == null || encoded.isEmpty()) return "";
        return new String(Base64.getDecoder().decode(encoded), StandardCharsets.UTF_8);
    }

    private static MeasuredImage measureWithImageJ(ImagePlus image, String imageName,
                                                   Roi[] rois) {
        Roi background = null;
        for (Roi roi : rois)
            if ("99_background".equalsIgnoreCase(roi.getName())) background = roi;
        if (background == null)
            throw new IllegalArgumentException(imageName + " 的 ROI 集缺少 99_background。");

        Measurement bg = measureOne(image, background);
        // The calibration every row of this image was measured with. Recorded per
        // row, because the same folder can hold images saved with different
        // calibrations -- and because a global calibration silently overrides all
        // of them, which is otherwise invisible in a table of numbers.
        Calibration calibration = image.getCalibration();
        MeasuredImage measured = new MeasuredImage();
        int index = 1;
        for (Roi roi : rois) {
            Measurement value = measureOne(image, roi);
            boolean isBackground = roi == background;
            Row row = new Row();
            row.index = index;
            row.image = imageName;
            row.label = roi.getName();
            row.value = value;
            row.correctedMean = isBackground ? 0.0 : value.mean - bg.mean;
            row.ctcf = isBackground ? 0.0 : value.intDen - value.area * bg.mean;
            row.rawCtcf = isBackground ? 0.0 : value.rawIntDen - value.pixelArea * bg.rawMean;
            row.background = isBackground;
            row.calibration = calibration;
            appendMeasurement(measured.table, row);
            measured.rows.add(row);
            index++;
        }
        return measured;
    }

    /**
     * Saves a table so that no reader ever sees a half-written file.
     *
     * The table is written beside the target and then moved onto it, which is
     * atomic within one filesystem. Doing this for the per-image tables as well
     * as the combined one matters because the image's table is what says the
     * image was measured: a truncated file would be read as a short table rather
     * than as a failure, and nothing downstream re-checks it.
     */
    private static void saveTable(ResultsTable table, File target) throws IOException {
        // The temporary keeps its extension LAST, the way the Python side names
        // its own temporary files. ResultsTable.saveAs() picks the delimiter
        // from the file name -- anything not ending in ".csv" is written
        // tab-separated -- so "<name>.csv.tmp" here would quietly turn every
        // measurement table into a tab-delimited file.
        String name = target.getName();
        int dot = name.lastIndexOf('.');
        File parent = target.getParentFile();
        // The measurements/ subfolder is created here rather than up front, so
        // that pointing at a folder with no ROI sets in it stays a read-only
        // affair -- we only make the folder once there is a table to put in it.
        if (parent != null && !parent.isDirectory() && !parent.mkdirs() && !parent.isDirectory())
            throw new IOException("无法创建测量表目录：" + parent);
        File temporary = new File(parent, dot < 0
                ? name + ".tmp"
                : name.substring(0, dot) + ".tmp" + name.substring(dot));
        try {
            table.saveAs(temporary.getAbsolutePath());
            try {
                Files.move(temporary.toPath(), target.toPath(),
                        StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            } catch (AtomicMoveNotSupportedException unsupported) {
                // A mounted share or an unusual driver may not be able to rename
                // atomically. Replacing in one step is still better than writing
                // the target in place, where a reader can catch it half-written.
                Files.move(temporary.toPath(), target.toPath(),
                        StandardCopyOption.REPLACE_EXISTING);
            }
        } catch (IOException error) {
            temporary.delete();
            throw error;
        }
    }

    /**
     * Measures one ROI on one image.
     *
     * The statistics come straight from that image. Asking Analyzer instead
     * would inherit Analyze > Set Measurements > "Redirect to", which swaps the
     * image the numbers are taken from while the row keeps the name of the image
     * it was asked about; and getRawStatistics() above is not redirected, so the
     * same row would carry calibrated numbers from one image next to raw numbers
     * from another. Area, Mean, Min, Max, Median and IntDen are what Analyzer
     * writes when no redirect is set, computed the same way (IntDen is the
     * calibrated area times the mean).
     */
    private static Measurement measureOne(ImagePlus image, Roi roi) {
        image.setRoi(roi);
        ImageStatistics stats = image.getStatistics(MEASURES);
        ImageStatistics raw = image.getRawStatistics();
        Measurement m = new Measurement();
        m.area = stats.area;
        m.pixelArea = raw.pixelCount;
        m.mean = stats.mean;
        m.rawMean = raw.mean;
        m.minimum = stats.min;
        m.maximum = stats.max;
        m.median = stats.median;
        m.intDen = stats.area * stats.mean;
        m.rawIntDen = raw.pixelCount * raw.mean;
        return m;
    }

    private static void appendMeasurement(ResultsTable table, Row row) {
        int index = row.index;
        String image = row.image;
        String label = row.label;
        Measurement m = row.value;
        double correctedMean = row.correctedMean;
        double ctcf = row.ctcf;
        double rawCtcf = row.rawCtcf;
        boolean background = row.background;
        Calibration calibration = row.calibration;
        table.incrementCounter();
        // The seven numbers an experimenter actually reads come first, in the
        // order ImageJ's own Results window shows them. Everything else follows.
        // This is the order the table is written in -- ResultsTable lays its
        // columns out in the order they are first added -- so the per-image and
        // the combined table stay in step automatically, and the Python
        // backend's PYTHON_MEASUREMENT_HEADER uses the same seven first.
        table.addValue("Area", m.area);
        table.addValue("Mean", m.mean);
        table.addValue("Min", m.minimum);
        table.addValue("Max", m.maximum);
        table.addValue("IntDen", m.intDen);
        table.addValue("Median", m.median);
        table.addValue("RawIntDen", m.rawIntDen);
        table.addValue("Index", index);
        table.addValue("Image", image);
        table.addValue("Label", label == null ? "" : label);
        table.addValue("PixelArea", m.pixelArea);
        table.addValue("BackgroundCorrectedMean", correctedMean);
        table.addValue("CTCF", ctcf);
        table.addValue("RawCTCF", rawCtcf);
        table.addValue("Type", background ? "background" : "worm");
        table.addValue("MeasurementBackend", "ImageJ " + IJ.getVersion());
        // What the row above was measured with. CTCF is the calibrated Area times
        // the calibrated Mean, so two machines can disagree on the same file --
        // a global calibration of 0.5 x 0.5 micron turns 8000 into 2000 -- and
        // these five columns are what makes that explainable afterwards.
        table.addValue("PixelWidth", calibration == null ? 1.0 : calibration.pixelWidth);
        table.addValue("PixelHeight", calibration == null ? 1.0 : calibration.pixelHeight);
        table.addValue("SpatialUnit", calibration == null ? "" : unitOf(calibration));
        table.addValue("ValueUnit", calibration == null
                ? Calibration.DEFAULT_VALUE_UNIT : valueUnitOf(calibration));
        table.addValue("IntensityFunction", intensityFunctionText(calibration));
    }

    private static Roi[] loadRois(File zip) throws IOException {
        List<Roi> result = new ArrayList<>();
        try (ZipInputStream input = new ZipInputStream(new FileInputStream(zip))) {
            ZipEntry entry;
            byte[] buffer = new byte[8192];
            while ((entry = input.getNextEntry()) != null) {
                if (entry.isDirectory() || !entry.getName().toLowerCase(Locale.ROOT).endsWith(".roi"))
                    continue;
                ByteArrayOutputStream bytes = new ByteArrayOutputStream();
                int count;
                while ((count = input.read(buffer)) >= 0) bytes.write(buffer, 0, count);
                Roi roi = new RoiDecoder(bytes.toByteArray(), entry.getName()).getRoi();
                if (roi.getName() == null || roi.getName().isEmpty()) roi.setName(stem(entry.getName()));
                result.add(roi);
            }
        }
        if (result.isEmpty()) throw new IOException("ImageJ 无法读取 ROI ZIP：" + zip);
        return result.toArray(new Roi[0]);
    }

    private static void addToRoiManager(Roi[] rois) {
        RoiManager manager = RoiManager.getRoiManager();
        removeOwnedRois(manager);
        for (Roi roi : rois) {
            Roi clone = (Roi) roi.clone();
            clone.setName(ROI_PREFIX + (roi.getName() == null ? "" : roi.getName()));
            manager.addRoi(clone);
        }
    }

    private static void removeOwnedRois(RoiManager manager) {
        for (int index = manager.getCount() - 1; index >= 0; index--) {
            Roi roi = manager.getRoi(index);
            if (roi != null && roi.getName() != null && roi.getName().startsWith(ROI_PREFIX)) {
                manager.delete(index);
            }
        }
    }

    private static File locateGui() {
        String remembered = Prefs.get(PREF_GUI, "");
        List<File> candidates = new ArrayList<>();
        if (!remembered.isEmpty()) candidates.add(new File(remembered));
        String plugins = IJ.getDirectory("plugins");
        if (plugins != null) {
            candidates.add(new File(plugins, "AutoWormImageJ/AutoWormGUI.exe"));
            candidates.add(new File(plugins, "AutoWormGUI.exe"));
        }
        for (File candidate : candidates) if (candidate.isFile()) return candidate;
        OpenDialog chooser = new OpenDialog("Locate AutoWormGUI.exe", null);
        String path = chooser.getPath();
        if (path == null) throw new IllegalArgumentException("AutoWormGUI.exe was not found.");
        File chosen = new File(path);
        Prefs.set(PREF_GUI, chosen.getAbsolutePath());
        return chosen;
    }

    private static File savedTiff(ImagePlus image) {
        if (image == null) return null;
        FileInfo info = image.getOriginalFileInfo();
        if (info == null || info.directory == null || info.fileName == null) return null;
        String lower = info.fileName.toLowerCase(Locale.ROOT);
        if (!lower.endsWith(".tif") && !lower.endsWith(".tiff")) return null;
        File file = new File(info.directory, info.fileName);
        return file.isFile() ? file : null;
    }

    private static File[] listTiffs(File folder) {
        File[] files = folder.listFiles(file -> {
            String name = file.getName().toLowerCase(Locale.ROOT);
            return file.isFile() && (name.endsWith(".tif") || name.endsWith(".tiff"));
        });
        if (files == null) return new File[0];
        Arrays.sort(files, Comparator.comparing(File::getName, String.CASE_INSENSITIVE_ORDER));
        return files;
    }

    private static String stem(String name) {
        int dot = name.lastIndexOf('.');
        return dot > 0 ? name.substring(0, dot) : name;
    }

    private static boolean sameFile(File first, File second) {
        if (first == null || second == null) return false;
        try { return first.getCanonicalFile().equals(second.getCanonicalFile()); }
        catch (IOException ignored) { return first.getAbsoluteFile().equals(second.getAbsoluteFile()); }
    }

    private static class Measurement {
        double area, pixelArea, mean, rawMean, minimum, maximum, intDen, median, rawIntDen;
    }

    /**
     * One finished row of a measurement table, held as data rather than written
     * once.
     *
     * A row has to reach two files, and the order matters: the image's own table
     * first, the combined table only after that write has succeeded. Keeping the
     * numbers here is what makes the second write a consequence of the first
     * instead of something that has already happened.
     */
    private static class Row {
        int index;
        String image, label;
        Measurement value;
        double correctedMean, ctcf, rawCtcf;
        boolean background;
        Calibration calibration;
    }

    /** One image's rows, and the table they were written into. */
    private static class MeasuredImage {
        final ResultsTable table = new ResultsTable();
        final List<Row> rows = new ArrayList<>();
    }
}
