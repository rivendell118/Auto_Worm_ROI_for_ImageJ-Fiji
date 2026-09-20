import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.gui.Roi;
import ij.io.RoiEncoder;
import ij.measure.Calibration;
import ij.process.ShortProcessor;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

/**
 * The plug-in's measurement pass, driven through the plug-in's own method.
 *
 * The calibration the numbers were produced with is checked here too: the columns
 * that record it per row, and the comparison against the window that the warning
 * before processing is built on (Analyze > Calibrate does not set ImagePlus.changes,
 * so that difference used to pass unnoticed).
 *
 * Two review findings live in that loop and nowhere else. One image that cannot
 * be measured used to end the pass where it stood, so everything after it in the
 * folder went unmeasured -- and since the combined table is written at the end,
 * the images already measured in that pass were lost with it. And a batch that
 * measures nothing used to leave the previous batch's combined table in place,
 * where it reads as this batch's result.
 *
 * The output folder is split in two: the ROI sets the GUI wrote are read from
 * other/ and the tables this pass writes land in measurements/, so every fixture
 * here puts its ROI set in the one and looks for its table in the other.
 *
 * Run against the built classes, with a display -- the pass ends by showing
 * ImageJ's results window, as it does for the user:
 *
 *   javac --release 8 -encoding UTF-8 -cp "lib\ij.jar" -d build\plugin_classes ^
 *       plugin_src\Auto_Worm_ROI.java plugin_src\Auto_Worm_Annotations.java
 *   javac --release 8 -encoding UTF-8 -cp "lib\ij.jar;build\plugin_classes" ^
 *       -d build\measurement_test_classes tests\AutoWormMeasurementTest.java
 *   java -cp "build\measurement_test_classes;lib\ij.jar;build\plugin_classes" ^
 *       AutoWormMeasurementTest
 *
 * The class directory, not dist\AutoWorm_ROI.jar, is what goes on the classpath:
 * the plug-in is in the unnamed package and javac will not read a type of the
 * unnamed package out of a jar, so against the jar every reference to
 * Auto_Worm_ROI is "cannot find symbol". At runtime either one works.
 */
public class AutoWormMeasurementTest {

    private static final String COMBINED = "ImageJ_measurements_all.csv";
    private static final String COMBINED_HEADER =
            "Area,Mean,Min,Max,IntDen,Median,RawIntDen,Index,Image,Label,PixelArea,"
                    + "BackgroundCorrectedMean,CTCF,RawCTCF,Type,MeasurementBackend,"
                    + "PixelWidth,PixelHeight,SpatialUnit,ValueUnit,IntensityFunction";
    /** An image's own table, as an earlier batch of this build left it. */
    private static final String STALE_IMAGE_HEADER =
            "Area,Mean,Min,Max,IntDen,Median,RawIntDen,Index,Image,Label,PixelArea,"
                    + "BackgroundCorrectedMean,CTCF,RawCTCF,Type,MeasurementBackend,"
                    + "PixelWidth,PixelHeight,SpatialUnit,ValueUnit,IntensityFunction";
    /** The two subfolders of an output folder; the GUI and the plug-in share them. */
    private static final String MEASUREMENTS = "measurements";
    private static final String OTHER = "other";

    /**
     * The subfolder the GUI puts the ROI sets in, made if it is not there yet.
     * The plug-in only reads from here, so writing the fixtures is on us.
     */
    private static Path other(Path output) throws Exception {
        Path dir = output.resolve(OTHER);
        Files.createDirectories(dir);
        return dir;
    }

    /**
     * The subfolder the plug-in writes its tables into. It makes this one itself
     * on the way to the first write, but the fixtures below put files there too.
     */
    private static Path measurements(Path output) throws Exception {
        Path dir = output.resolve(MEASUREMENTS);
        Files.createDirectories(dir);
        return dir;
    }

    public static void main(String[] args) throws Exception {
        clearGlobalCalibration();
        try {
            oneUnmeasurableImageDoesNotStopTheRest();
            anImageWhoseTableCannotBeWrittenIsNotInTheCombinedTable();
            aTableFromAnEarlierBatchIsRemovedBeforeMeasuring();
            aFileThatOnlyLooksLikeOurTableIsLeftAlone();
            aTableWhoseColumnsMerelyContainOursIsLeftAlone();
            anImagesTableIsStillRecognisedByItsRealColumns();
            anImagesTableFromAnEarlierBatchIsRemovedBeforeMeasuring();
            aStaleTableIsNotLeftByAnImageThisBatchCouldNotMeasure();
            aFileThatOnlyLooksLikeAnImagesTableIsLeftAlone();
            theNumbersThatAreReadFirstComeFirst();
            theMeasurementFolderIsMadeIfItIsNotThere();
            theCalibrationOfTheImageIsRecordedInEveryRow();
            aCalibrationSetInTheWindowIsNoticed();
            aQuantizedPixelSizeIsNotADifference();
            aRealTiffRoundTripIsNotADifference();
            anUnvisitedImageKeepsItsTable();
            theRequestedPlaneIsTheOneMeasured();
            aPlaneBeyondTheStackIsSkippedAndReported();
            aSinglePlaneImageIgnoresThePlaneNumber();
            System.out.println("AUTOWORM_MEASUREMENT_OK");
        } catch (Throwable failure) {
            // The pass under test ends by showing ImageJ's results window, and
            // AWT's event thread is not a daemon: without an exit the process
            // would hang after main returns, and a failed case would look like
            // a run that is still working.
            failure.printStackTrace();
            System.exit(1);
        }
        System.exit(0);
    }

    /**
     * A batch of two images where the first one to be measured cannot be: the
     * second must still be measured, and reported.
     *
     * The images are named so that the broken one sorts first, which is the case
     * that used to end the pass before anything had been measured at all.
     */
    private static void oneUnmeasurableImageDoesNotStopTheRest() throws Exception {
        Path input = Files.createTempDirectory("autoworm-measure-in-");
        Path output = Files.createTempDirectory("autoworm-measure-out-");
        writeTiff(input.resolve("aaa_bad.tif"));
        writeTiff(input.resolve("bbb_good.tif"));
        // A file with the name the GUI gives a ROI set, and nothing readable in
        // it: what a truncated write from a killed batch looks like.
        Files.write(other(output).resolve("aaa_bad_RoiSet.zip"),
                "this is not a zip file".getBytes(StandardCharsets.US_ASCII));
        writeRoiZip(other(output).resolve("bbb_good_RoiSet.zip"));

        Auto_Worm_ROI.measureOutputFolder(input.toFile(), output.toFile(), null, null, false, false);

        check(Files.exists(measurements(output).resolve("bbb_good_measurements.csv")),
                "the image after the unmeasurable one was not measured");
        check(!Files.exists(measurements(output).resolve("aaa_bad_measurements.csv")),
                "an unmeasurable image produced a measurement table");
        check(Files.exists(measurements(output).resolve(COMBINED)),
                "no combined table was written");
        String combined = read(measurements(output).resolve(COMBINED));
        check(combined.contains("bbb_good.tif"), "the measured image is missing from the table: "
                + combined);
        check(!combined.contains("aaa_bad.tif"), "the unmeasured image is in the table");
    }

    /**
     * An image whose own table cannot be written is not in the combined table.
     *
     * Both tables come out of the same pass and the combined one is written
     * last, so it is easy to fill it as each image is measured and end up with
     * a row for an image the log calls skipped. The numbers in that row are
     * real, which is exactly what makes it dangerous: nothing else in the
     * output contradicts it, and the missing per-image table beside it is easy
     * to read as "the file was not needed". The rows join the combined table
     * only once the image's own table is safely on disk.
     */
    private static void anImageWhoseTableCannotBeWrittenIsNotInTheCombinedTable() throws Exception {
        Path input = Files.createTempDirectory("autoworm-measure-in-");
        Path output = Files.createTempDirectory("autoworm-measure-out-");
        writeTiff(input.resolve("aaa_blocked.tif"));
        writeTiff(input.resolve("bbb_good.tif"));
        writeRoiZip(other(output).resolve("aaa_blocked_RoiSet.zip"));
        writeRoiZip(other(output).resolve("bbb_good_RoiSet.zip"));
        // A directory where the table goes: the measurement itself succeeds and
        // only the write fails, which is the case this is about. Something of
        // the experimenter's is inside it, so a replace that went ahead anyway
        // would be visible here rather than silently losing their file.
        Path blocked = measurements(output).resolve("aaa_blocked_measurements.csv");
        Files.createDirectory(blocked);
        Files.write(blocked.resolve("keep.txt"), "mine".getBytes(StandardCharsets.UTF_8));

        Auto_Worm_ROI.measureOutputFolder(input.toFile(), output.toFile(), null, null, false, false);

        check(Files.isDirectory(blocked) && Files.exists(blocked.resolve("keep.txt")),
                "the directory in the way was replaced");
        check(noTemporaries(output), "a half-written table was left behind in the table folder");
        String combined = read(measurements(output).resolve(COMBINED));
        check(combined.contains("bbb_good.tif"),
                "the image that could be measured is missing from the table: " + combined);
        check(!combined.contains("aaa_blocked.tif"),
                "an image whose table could not be written is in the combined table: " + combined);
    }

    /**
     * True when a finished pass left no half-written tables behind.
     *
     * The marker goes before the extension, so a table being written is called
     * "x_measurements.tmp.csv" and not "x_measurements.csv.tmp" -- the extension
     * has to stay last or ResultsTable writes the file tab-separated. The
     * temporary is written beside its finished table, so the folder to look in
     * is the one the tables go in.
     */
    private static boolean noTemporaries(Path output) throws Exception {
        try (java.util.stream.Stream<Path> entries = Files.list(measurements(output))) {
            return !entries.anyMatch(entry -> entry.getFileName().toString().contains(".tmp."));
        }
    }

    /**
     * A batch that measures nothing must not leave the previous batch's table
     * where this batch's result would be.
     *
     * There is nothing in the file that records which run wrote it, so once a
     * batch has finished it cannot be told apart from its predecessor. Deleting
     * it before measuring is what makes its absence mean "nothing was measured".
     */
    private static void aTableFromAnEarlierBatchIsRemovedBeforeMeasuring() throws Exception {
        Path input = Files.createTempDirectory("autoworm-measure-in-");
        Path output = Files.createTempDirectory("autoworm-measure-out-");
        writeTiff(input.resolve("aaa_good.tif"));      // no ROI set: nothing to measure
        Path stale = measurements(output).resolve(COMBINED);
        Files.write(stale, (COMBINED_HEADER +
                "\r\n10,1.0,1,1,10,1,10,1,aaa_good.tif,01_worm,10,0.0,0.0,0,worm,ImageJ,"
                + "1,1,pixel,Gray Value,None\r\n").getBytes(StandardCharsets.UTF_8));

        boolean reported = false;
        try {
            Auto_Worm_ROI.measureOutputFolder(input.toFile(), output.toFile(), null, null, false, false);
        } catch (Exception expected) {
            reported = true;
        }

        check(reported, "measuring nothing was not reported as a failure");
        check(!Files.exists(stale), "the previous batch's combined table is still there");
    }

    /**
     * A file in the output folder that merely has our name is not deleted.
     *
     * The output folder belongs to the experimenter. The name alone is not proof
     * that the table is one of ours, and deleting a stranger's file because of its
     * name would be worse than leaving a stale table behind.
     */
    /**
     * A table whose header merely contains our column names is not our table.
     *
     * The two names that identify a table of ours are checked as whole fields. An
     * earlier build tested the header as one string with String.contains, so
     * "RawCTCFValue" and "MeasurementBackendNotes" matched "CTCF" and
     * "MeasurementBackend" and the file was deleted before it could be measured
     * into. The output folder is the experimenter's, and the header check exists
     * precisely so that a file we cannot prove is ours is left where it is.
     */
    private static void aTableWhoseColumnsMerelyContainOursIsLeftAlone() throws Exception {
        Path input = Files.createTempDirectory("autoworm-measure-in-");
        Path output = Files.createTempDirectory("autoworm-measure-out-");
        writeTiff(input.resolve("aaa_good.tif"));
        String header = "Image,RawCTCFValue,MeasurementBackendNotes,Comment";
        String body = header + "\r\ncell1,0.5,notes,hello\r\n";
        Path foreign = measurements(output).resolve(COMBINED);
        Files.write(foreign, body.getBytes(StandardCharsets.UTF_8));

        try {
            Auto_Worm_ROI.measureOutputFolder(input.toFile(), output.toFile(), null, null, false, false);
        } catch (Exception expected) {
            // Measuring nothing is the point of the fixture; it is reported.
        }

        check(Files.exists(foreign),
                "a table whose columns merely contain our names was deleted");
        check(read(foreign).equals(body),
                "a table whose columns merely contain our names was overwritten: " + read(foreign));
    }

    /**
     * The same header test still recognises a real table, so the stricter match
     * did not turn the cleanup off.
     *
     * The image has a ROI ZIP, so this pass measures it and writes a table of its
     * own. The stale table is therefore replaced rather than simply gone: the
     * assertion is that the old contents did not survive into the new file, which
     * is what "deleted before measuring" is for. Writing a marker row and looking
     * for its absence tests that directly; asserting the file is gone would only
     * be true when the pass fails to measure, which is not this fixture.
     */
    private static void anImagesTableIsStillRecognisedByItsRealColumns() throws Exception {
        Path input = Files.createTempDirectory("autoworm-measure-in-");
        Path output = Files.createTempDirectory("autoworm-measure-out-");
        writeTiff(input.resolve("aaa_good.tif"));
        writeRoiZip(other(output).resolve("aaa_good_RoiSet.zip"));
        Path ours = measurements(output).resolve("aaa_good_measurements.csv");
        // The Label field carries the marker; nothing this pass writes can
        // contain it, so finding it afterwards means the stale file survived.
        String marker = "STALE_FROM_EARLIER_BATCH";
        Files.write(ours, (STALE_IMAGE_HEADER + "\r\n1,2,3,4,5,6,7,0,aaa_good.tif," + marker
                + ",10,2,3,0,1,1,1,pixel,none,Raw\r\n").getBytes(StandardCharsets.UTF_8));

        try {
            Auto_Worm_ROI.measureOutputFolder(input.toFile(), output.toFile(), null, null, false, false);
        } catch (Exception expected) {
            // Only the header check is under test here.
        }

        check(!read(ours).contains(marker),
                "the table from the earlier batch survived into this one: " + read(ours));
    }

    private static void aFileThatOnlyLooksLikeOurTableIsLeftAlone() throws Exception {
        Path input = Files.createTempDirectory("autoworm-measure-in-");
        Path output = Files.createTempDirectory("autoworm-measure-out-");
        writeTiff(input.resolve("aaa_good.tif"));
        Path foreign = measurements(output).resolve(COMBINED);
        Files.write(foreign, "image,count\r\nmine,3\r\n".getBytes(StandardCharsets.UTF_8));

        try {
            Auto_Worm_ROI.measureOutputFolder(input.toFile(), output.toFile(), null, null, false, false);
        } catch (Exception expected) {
            // Measuring nothing is the point of the fixture; it is reported.
        }

        check(Files.exists(foreign), "a file that was not ours was deleted");
        check(read(foreign).equals("image,count\r\nmine,3\r\n"),
                "a file that was not ours was overwritten: " + read(foreign));
    }

    /**
     * This image's table from an earlier batch does not outlive this batch.
     *
     * The combined table is cleared up front and the per-image ones were not, so
     * an image re-measured after a run that was stopped, or after a run whose
     * numbers are known to be wrong, kept nothing of the sort: its table was
     * simply replaced. That hid the case below, where this batch does not get far
     * enough to replace it.
     */
    private static void anImagesTableFromAnEarlierBatchIsRemovedBeforeMeasuring() throws Exception {
        Path input = Files.createTempDirectory("autoworm-measure-in-");
        Path output = Files.createTempDirectory("autoworm-measure-out-");
        writeTiff(input.resolve("aaa_good.tif"));
        writeRoiZip(other(output).resolve("aaa_good_RoiSet.zip"));
        Path stale = measurements(output).resolve("aaa_good_measurements.csv");
        Files.write(stale, (STALE_IMAGE_HEADER + "\r\n"
                + "10,1.0,1,1,10,1,10,1,OLD.tif,01_worm,10,0.0,999.0,0,worm,ImageJ,"
                + "1,1,pixel,Gray Value,None\r\n").getBytes(StandardCharsets.UTF_8));

        Auto_Worm_ROI.measureOutputFolder(input.toFile(), output.toFile(), null, null, false, false);

        String table = read(measurements(output).resolve("aaa_good_measurements.csv"));
        check(!table.contains("OLD.tif"),
                "the earlier batch's table for this image was carried over: " + table);
        check(table.contains("aaa_good.tif"),
                "this batch did not write its own table for the image: " + table);
    }

    /**
     * An image this batch could not measure is left with no table at all.
     *
     * This is the case the per-image cleanup exists for. The image is in the
     * batch, its ROI set is there but unreadable, so the pass reports it as
     * skipped -- and a table left from an earlier run sits in the output folder
     * beside this run's other output under that image's name. Its numbers are
     * from data that has since been re-segmented, and nothing in the file says
     * so. Afterwards "no table" has to mean "not measured this time".
     */
    private static void aStaleTableIsNotLeftByAnImageThisBatchCouldNotMeasure() throws Exception {
        Path input = Files.createTempDirectory("autoworm-measure-in-");
        Path output = Files.createTempDirectory("autoworm-measure-out-");
        writeTiff(input.resolve("aaa_bad.tif"));
        writeTiff(input.resolve("bbb_good.tif"));
        Files.write(other(output).resolve("aaa_bad_RoiSet.zip"),
                "this is not a zip file".getBytes(StandardCharsets.US_ASCII));
        writeRoiZip(other(output).resolve("bbb_good_RoiSet.zip"));
        Path stale = measurements(output).resolve("aaa_bad_measurements.csv");
        Files.write(stale, (STALE_IMAGE_HEADER + "\r\n"
                + "10,1.0,1,1,10,1,10,1,OLD.tif,01_worm,10,0.0,999.0,0,worm,ImageJ,"
                + "1,1,pixel,Gray Value,None\r\n").getBytes(StandardCharsets.UTF_8));

        Auto_Worm_ROI.measureOutputFolder(input.toFile(), output.toFile(), null, null, false, false);

        check(!Files.exists(stale),
                "an image the log calls skipped still has a table beside it");
        check(Files.exists(measurements(output).resolve("bbb_good_measurements.csv")),
                "the image that could be measured was not");
    }

    /**
     * A file in the output folder that only has the name of an image's table is
     * not deleted, even for an image this batch visits and fails on.
     */
    private static void aFileThatOnlyLooksLikeAnImagesTableIsLeftAlone() throws Exception {
        Path input = Files.createTempDirectory("autoworm-measure-in-");
        Path output = Files.createTempDirectory("autoworm-measure-out-");
        writeTiff(input.resolve("aaa_bad.tif"));
        Files.write(other(output).resolve("aaa_bad_RoiSet.zip"),
                "this is not a zip file".getBytes(StandardCharsets.US_ASCII));
        Path foreign = measurements(output).resolve("aaa_bad_measurements.csv");
        Files.write(foreign, "image,count\r\nmine,3\r\n".getBytes(StandardCharsets.UTF_8));

        try {
            Auto_Worm_ROI.measureOutputFolder(input.toFile(), output.toFile(), null, null, false, false);
        } catch (Exception expected) {
            // The fixture measures nothing, which the pass reports.
        }

        check(Files.exists(foreign), "a file that was not ours was deleted");
        check(read(foreign).equals("image,count\r\nmine,3\r\n"),
                "a file that was not ours was overwritten: " + read(foreign));
    }

    /**
     * An image the batch never reaches keeps its table.
     *
     * The pass is run on a folder it did not fill: the GUI processed some of it,
     * or is still processing it, and an older batch's tables live there. Those
     * images are not part of this pass and their tables are not this pass's to
     * remove -- the cleanup is for the images it measures or reports as failed.
     */
    private static void anUnvisitedImageKeepsItsTable() throws Exception {
        Path input = Files.createTempDirectory("autoworm-measure-in-");
        Path output = Files.createTempDirectory("autoworm-measure-out-");
        writeTiff(input.resolve("aaa_good.tif"));
        writeTiff(input.resolve("zzz_untouched.tif"));
        writeRoiZip(other(output).resolve("aaa_good_RoiSet.zip"));
        writeRoiZip(other(output).resolve("zzz_untouched_RoiSet.zip"));
        Path untouched = measurements(output).resolve("zzz_untouched_measurements.csv");
        Files.write(untouched, (STALE_IMAGE_HEADER + "\r\n"
                + "10,1.0,1,1,10,1,10,1,zzz_untouched.tif,01_worm,10,0.0,7.0,0,worm,ImageJ,"
                + "1,1,pixel,Gray Value,None\r\n").getBytes(StandardCharsets.UTF_8));

        // The GUI's list of what it finished, with the older image left out.
        Auto_Worm_ROI.measureOutputFolder(input.toFile(), output.toFile(), null,
                new java.util.HashSet<String>(java.util.Arrays.asList("aaa_good.tif")),
                false, false);

        check(Files.exists(untouched),
                "an image this batch never visited had its table deleted");
    }

    /**
     * A pixel size that a TIFF round trip moved in the last few digits is not a
     * difference.
     *
     * ImageJ stores the spatial calibration in TIFF tags of limited precision, so
     * a scale that was typed, saved and reloaded comes back a hair away from the
     * double still in the window. Compared exactly, every such image differed
     * from its own window: the experimenter got the confirmation dialog on every
     * single call, with nothing in it to act on and no way to stop it.
     *
     * Both directions are checked here, because a tolerance that quietly swallows
     * a real difference would be worse than the dialog: 0.5 against 0.6 is still
     * a difference, as is a changed unit. pixelDepth is not one at all -- no Area
     * or Mean on a single 2-D plane can see it.
     */
    private static void aQuantizedPixelSizeIsNotADifference() throws Exception {
        clearGlobalCalibration();
        ImagePlus window = new ImagePlus("", new ShortProcessor(1, 1));
        ImagePlus disk = new ImagePlus("", new ShortProcessor(1, 1));
        try {
            for (ImagePlus image : new ImagePlus[] {window, disk}) {
                image.getCalibration().pixelWidth = 0.123456789;
                image.getCalibration().pixelHeight = 0.123456789;
                image.getCalibration().setUnit("micron");
            }
            // The value the review measured coming back out of a saved TIFF.
            disk.getCalibration().pixelHeight = 0.1234567901234568;
            check(Auto_Worm_ROI.calibrationDifference(window, disk) == null,
                    "a pixel size quantized by the file format was reported as a difference: "
                            + Auto_Worm_ROI.calibrationDifference(window, disk));

            disk.getCalibration().pixelHeight = 0.5;
            String coarser = Auto_Worm_ROI.calibrationDifference(window, disk);
            check(coarser != null && coarser.contains("空间校准"),
                    "a genuinely different pixel size was not reported: " + coarser);
            // Just inside the tolerance is not a difference; just outside is. A
            // hundredth of a percent is a scale no one sets by accident and one
            // that would change every Area and every CTCF in the table.
            disk.getCalibration().pixelHeight = 0.123456789 * (1 + 5e-7);
            check(Auto_Worm_ROI.calibrationDifference(window, disk) == null,
                    "a difference inside the tolerance was reported");
            disk.getCalibration().pixelHeight = 0.123456789 * (1 + 1e-3);
            check(Auto_Worm_ROI.calibrationDifference(window, disk) != null,
                    "a difference of a thousandth was not reported");

            // Depth alone: the same width, height and unit, different slice
            // spacing, which no measurement here can observe.
            disk.getCalibration().pixelWidth = window.getCalibration().pixelWidth;
            disk.getCalibration().pixelHeight = window.getCalibration().pixelHeight;
            disk.getCalibration().pixelDepth = 5.0;
            check(Auto_Worm_ROI.calibrationDifference(window, disk) == null,
                    "a slice spacing that no measurement can see was reported as a difference");
        } finally {
            window.close();
            disk.close();
        }
    }

    /**
     * A TIFF saved with a long pixel size reloads as the same image, not as a
     * difference -- the round trip the review reproduced the dialog from.
     */
    private static void aRealTiffRoundTripIsNotADifference() throws Exception {
        clearGlobalCalibration();
        Path path = Files.createTempDirectory("autoworm-cal-roundtrip-").resolve("aaa_odd.tif");
        ImagePlus saved = new ImagePlus("test", new ShortProcessor(40, 40));
        saved.getCalibration().pixelWidth = 0.123456789;
        saved.getCalibration().pixelHeight = 0.123456789;
        saved.getCalibration().setUnit("micron");
        IJ.saveAsTiff(saved, path.toString());

        ImagePlus window = IJ.openImage(path.toString());
        ImagePlus disk = IJ.openImage(path.toString());
        try {
            check(Auto_Worm_ROI.calibrationDifference(window, disk) == null,
                    "a file differed from itself after a round trip through the format");
        } finally {
            window.close();
            disk.close();
            saved.close();
        }
    }

    /**
     * The numbers the table is read for come first, in both tables.
     *
     * ImageJ's own Index and Image, and the plug-in's Label, used to sit in front
     * of Area and the rest, so the seven numbers an experimenter copies out were
     * behind a file name and a row counter. Both tables carry the same order, so
     * a row read from one reads the same way in the other, and the calibration
     * columns stay grouped at the end where they were.
     */
    private static void theNumbersThatAreReadFirstComeFirst() throws Exception {
        Path input = Files.createTempDirectory("autoworm-measure-in-");
        Path output = Files.createTempDirectory("autoworm-measure-out-");
        writeTiff(input.resolve("aaa_good.tif"));
        writeRoiZip(other(output).resolve("aaa_good_RoiSet.zip"));

        Auto_Worm_ROI.measureOutputFolder(input.toFile(), output.toFile(), null, null, false, false);

        for (Path table : new Path[] {measurements(output).resolve("aaa_good_measurements.csv"),
                measurements(output).resolve(COMBINED)}) {
            String[] lines = read(table).split("\r?\n");
            check(lines[0].startsWith("Area,Mean,Min,Max,IntDen,Median,RawIntDen,"),
                    "the numbers that are read first do not come first in " + table.getFileName()
                            + ": " + lines[0]);
            // The header and every row have to agree on where the boundary is,
            // so a row that lost its place is caught rather than written under
            // the wrong names -- a mislabelled row is still a well-formed CSV.
            for (int row = 1; row < lines.length; row++) {
                String[] values = lines[row].split(",");
                check(values.length == lines[0].split(",").length,
                        "a row does not line up with the header in " + table.getFileName()
                                + ": " + lines[row]);
                check(values[8].equals("aaa_good.tif"),
                        "the image name is not where the header says it is in "
                                + table.getFileName() + ": " + lines[row]);
            }
        }
    }

    /**
     * The plug-in makes the folder it writes its tables into.
     *
     * The GUI makes both subfolders when a batch starts, but ImageJ can be
     * pointed at a folder the GUI has not just filled -- a batch from an earlier
     * build, or a folder assembled by hand. Failing the write there would end
     * the pass, so the folder is made on the way to the first table instead.
     */
    private static void theMeasurementFolderIsMadeIfItIsNotThere() throws Exception {
        Path input = Files.createTempDirectory("autoworm-measure-in-");
        Path output = Files.createTempDirectory("autoworm-measure-out-");
        writeTiff(input.resolve("aaa_good.tif"));
        writeRoiZip(other(output).resolve("aaa_good_RoiSet.zip"));
        Files.delete(measurements(output));

        Auto_Worm_ROI.measureOutputFolder(input.toFile(), output.toFile(), null, null, false, false);

        check(Files.exists(measurements(output).resolve("aaa_good_measurements.csv")),
                "no table was written into a table folder that had to be made");
    }

    /**
     * Every measured row says what the image was calibrated by.
     *
     * CTCF is the calibrated Area times the calibrated Mean, so the same folder
     * measured on two machines -- one of them with a global calibration set --
     * gives two sets of numbers. Without these columns nothing in the output
     * explains which calibration produced them.
     */
    private static void theCalibrationOfTheImageIsRecordedInEveryRow() throws Exception {
        Path input = Files.createTempDirectory("autoworm-cal-in-");
        Path output = Files.createTempDirectory("autoworm-cal-out-");
        writeCalibratedTiff(input.resolve("aaa_cal.tif"));
        writeRoiZip(other(output).resolve("aaa_cal_RoiSet.zip"));

        Auto_Worm_ROI.measureOutputFolder(input.toFile(), output.toFile(), null, null, false, false);

        String[] lines = read(measurements(output).resolve("aaa_cal_measurements.csv"))
                .split("\r?\n");
        check(lines[0].endsWith("PixelWidth,PixelHeight,SpatialUnit,ValueUnit,IntensityFunction"),
                "the calibration columns are not at the end of the table: " + lines[0]);
        check(lines.length >= 3, "the worm and the background were not both measured");
        for (int row = 1; row < lines.length; row++) {
            String[] values = lines[row].split(",");
            int last = values.length - 5;
            // The two sizes are number columns, so ResultsTable writes them the way
            // it writes every other number in the table -- 0.500, not 0.5.
            check(Double.parseDouble(values[last]) == 0.5,
                    "PixelWidth is not the file's: " + lines[row]);
            check(Double.parseDouble(values[last + 1]) == 0.5,
                    "PixelHeight is not the file's: " + lines[row]);
            check(values[last + 2].equals("micron"), "SpatialUnit is not the file's: " + lines[row]);
            check(values[last + 3].equals("Gray"), "ValueUnit is not the file's: " + lines[row]);
            check(values[last + 4].equals("Straight Line: y = 10 + 2*x"),
                    "IntensityFunction is not the file's: " + lines[row]);
        }
    }

    /**
     * A calibration set in the window is a difference from the file, and one that
     * carries from the file into the window is not.
     *
     * Analyze > Calibrate does not set ImagePlus.changes, so this difference used
     * to be invisible: the window would measure one thing and the plug-in another,
     * with the same image name on both. The comparison is what the warning before
     * processing and the note in the log are made of.
     */
    private static void aCalibrationSetInTheWindowIsNoticed() throws Exception {
        clearGlobalCalibration();
        Path tiff = Files.createTempDirectory("autoworm-cal-diff-").resolve("aaa_cal.tif");
        writeCalibratedTiff(tiff);

        ImagePlus window = IJ.openImage(tiff.toString());
        ImagePlus disk = IJ.openImage(tiff.toString());
        try {
            check(Auto_Worm_ROI.calibrationDifference(window, disk) == null,
                    "the same calibration was reported as a difference");

            window.getCalibration().pixelWidth = 1.0;
            String spatial = Auto_Worm_ROI.calibrationDifference(window, disk);
            check(spatial != null && spatial.contains("空间校准"),
                    "a spatial calibration set in the window was not noticed: " + spatial);
            window.getCalibration().pixelWidth = 0.5;

            window.getCalibration().setFunction(Calibration.STRAIGHT_LINE,
                    new double[] {0.0, 1.0}, "Gray");
            String intensity = Auto_Worm_ROI.calibrationDifference(window, disk);
            check(intensity != null && intensity.contains("强度校准"),
                    "an intensity calibration set in the window was not noticed: " + intensity);
            // And this is why a new signal was needed: the flag the check used to
            // go by stays down, which is what made the difference silent.
            check(!window.changes, "this fixture no longer reproduces the silent case");

            // The unit alone is enough to change the numbers Area is reported in.
            window.getCalibration().setFunction(Calibration.NONE, null, "Gray");
            window.getCalibration().setUnit("inch");
            check(Auto_Worm_ROI.calibrationDifference(window, disk) != null,
                    "a spatial unit set in the window was not noticed");
        } finally {
            window.close();
            disk.close();
        }
    }

    /**
     * No global calibration, so that a calibration written to a TIFF is that
     * image's own and the comparison under test sees what the file says.
     *
     * ImageJ keeps one global calibration for the whole JVM, and while it is set
     * every image's getCalibration() hands back a copy of it instead of the
     * calibration the image carries -- which would hide the very difference these
     * cases look for. It is set through an instance method in ImageJ 1.54p even
     * though what it writes is static, so an image is needed to call it.
     */
    /**
     * The numbers come from the plane the GUI named, not from slice 1.
     *
     * This is the whole of 0.5.0's Java side, and it is the one thing a wrong
     * implementation gets silently right-looking: measuring slice 1 of every
     * image produces a complete, plausible table. Plane 1 of the fixture holds
     * 100 and plane 2 holds 10, so the two readings cannot be confused, and both
     * directions are checked -- a pass that always measured plane 2 would satisfy
     * only one of them.
     */
    private static void theRequestedPlaneIsTheOneMeasured() throws Exception {
        Path input = Files.createTempDirectory("autoworm-plane-in-");
        Path output = Files.createTempDirectory("autoworm-plane-out-");
        writeStackTiff(input.resolve("two_planes.tif"), 2);
        writeRoiZip(other(output).resolve("two_planes_RoiSet.zip"));

        Auto_Worm_ROI.measureOutputFolder(input.toFile(), output.toFile(), null, null,
                false, false, 2);
        String second = read(measurements(output).resolve("two_planes_measurements.csv"));
        check(meanOf(second, "01_worm") == 10.0,
                "the second plane was not the one that was measured: " + second);

        // The same file again, this time asking for the plane that is first. The
        // table from the previous pass has to be gone before the new one is
        // judged, or this would read the old numbers.
        Auto_Worm_ROI.measureOutputFolder(input.toFile(), output.toFile(), null, null,
                false, false, 1);
        String first = read(measurements(output).resolve("two_planes_measurements.csv"));
        check(meanOf(first, "01_worm") == 100.0,
                "the first plane was not the one that was measured: " + first);
    }

    /**
     * A plane the file does not have is skipped, and the batch says so.
     *
     * Skipping rather than measuring something else is the point: a number from
     * the wrong plane looks exactly like a number from the right one, which is
     * the failure the whole feature exists to prevent. With nothing measured the
     * pass reports it the way it reports an empty folder, and leaves no table
     * behind for anyone to read as a result.
     */
    private static void aPlaneBeyondTheStackIsSkippedAndReported() throws Exception {
        Path input = Files.createTempDirectory("autoworm-shallow-in-");
        Path output = Files.createTempDirectory("autoworm-shallow-out-");
        writeStackTiff(input.resolve("two_planes.tif"), 2);
        writeRoiZip(other(output).resolve("two_planes_RoiSet.zip"));

        boolean reported = false;
        try {
            Auto_Worm_ROI.measureOutputFolder(input.toFile(), output.toFile(), null, null,
                    false, false, 5);
        } catch (IOException expected) {
            reported = true;
        }
        check(reported, "a batch where no image could be measured was reported as done");
        check(!Files.exists(measurements(output).resolve("two_planes_measurements.csv")),
                "a skipped image still produced a measurement table");
    }

    /**
     * A single-plane image is measured whatever plane number the batch carries.
     *
     * Batches are allowed to hold both kinds (the window's decision 2), and the
     * plane number is a property of the batch, so a single-plane file will often
     * be handed one it has no plane for. Refusing it would throw away a perfectly
     * measurable image.
     */
    private static void aSinglePlaneImageIgnoresThePlaneNumber() throws Exception {
        Path input = Files.createTempDirectory("autoworm-single-in-");
        Path output = Files.createTempDirectory("autoworm-single-out-");
        writeTiff(input.resolve("single.tif"));
        writeRoiZip(other(output).resolve("single_RoiSet.zip"));

        Auto_Worm_ROI.measureOutputFolder(input.toFile(), output.toFile(), null, null,
                false, false, 4);

        check(Files.exists(measurements(output).resolve("single_measurements.csv")),
                "a single-plane image was skipped because of the batch's plane number");
    }

    /**
     * The Mean of the row for {@code label}, read by column and parsed.
     *
     * Located by header position rather than searched for as text: the table
     * legitimately holds other numbers that contain "100" as a substring, so a
     * contains() check could pass on a table measured from the wrong plane. And
     * parsed rather than compared as a string because ImageJ writes a whole
     * number as "10", not "10.0", which is not a fact these tests are about.
     */
    private static double meanOf(String table, String label) {
        for (String line : table.split("\r?\n")) {
            String[] cells = line.split(",", -1);
            if (cells.length > 9 && cells[9].equals(label)) return Double.parseDouble(cells[1]);
        }
        throw new AssertionError("no row for " + label + " in: " + table);
    }

    private static void clearGlobalCalibration() {
        ImagePlus image = new ImagePlus("", new ShortProcessor(1, 1));
        image.setGlobalCalibration(null);
    }

    private static void writeTiff(Path path) throws Exception {
        ImagePlus image = new ImagePlus("test", new ShortProcessor(40, 40));
        IJ.saveAsTiff(image, path.toString());
        image.close();
    }

    /**
     * A multi-plane TIFF: plane 1 filled with 100, every later plane with 10.
     *
     * The two values are the point. Uniform planes mean the Mean of a ROI is the
     * plane's value exactly, and different values mean a pass that measured the
     * wrong plane produces a visibly different number instead of an identical
     * one -- a fixture where both planes held the same picture could not tell the
     * difference the tests below are looking for.
     */
    private static void writeStackTiff(Path path, int planes) throws Exception {
        ImageStack stack = new ImageStack(40, 40);
        for (int index = 0; index < planes; index++) {
            ShortProcessor plane = new ShortProcessor(40, 40);
            int value = index == 0 ? 100 : 10;
            for (int pixel = 0; pixel < plane.getPixelCount(); pixel++) plane.set(pixel, value);
            stack.addSlice("plane " + (index + 1), plane);
        }
        ImagePlus image = new ImagePlus("test", stack);
        IJ.saveAsTiff(image, path.toString());
        image.close();
    }

    /** A TIFF that carries a spatial and an intensity calibration of its own. */
    private static void writeCalibratedTiff(Path path) throws Exception {
        ImagePlus image = new ImagePlus("test", new ShortProcessor(40, 40));
        Calibration calibration = image.getCalibration();
        calibration.pixelWidth = 0.5;
        calibration.pixelHeight = 0.5;
        calibration.setUnit("micron");
        calibration.setFunction(Calibration.STRAIGHT_LINE, new double[] {10.0, 2.0}, "Gray");
        IJ.saveAsTiff(image, path.toString());
        image.close();
    }

    /** A ROI set in a ZIP, shaped and named the way the GUI writes them. */
    private static void writeRoiZip(Path path) throws Exception {
        Roi worm = new Roi(5, 5, 20, 20);
        worm.setName("01_worm");
        Roi background = new Roi(30, 5, 5, 20);
        background.setName("99_background");
        try (ZipOutputStream zip = new ZipOutputStream(new FileOutputStream(path.toFile()))) {
            for (Roi roi : new Roi[] {worm, background}) {
                zip.putNextEntry(new ZipEntry(roi.getName() + ".roi"));
                zip.write(RoiEncoder.saveAsByteArray(roi));
                zip.closeEntry();
            }
        }
    }

    private static String read(Path path) throws Exception {
        return new String(Files.readAllBytes(path), StandardCharsets.UTF_8);
    }

    private static void check(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }
}
