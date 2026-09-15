import ij.IJ;
import ij.ImagePlus;
import ij.gui.Roi;
import ij.measure.ResultsTable;

import java.io.File;
import java.lang.reflect.Field;
import java.lang.reflect.Method;

/** Headless smoke test for ROI decoding and ImageJ-native measurements. */
public class ImageJMeasurementSmoke {
    public static void main(String[] args) throws Exception {
        if (args.length != 2) throw new IllegalArgumentException("image.tif RoiSet.zip");
        ImagePlus image = IJ.openImage(args[0]);
        if (image == null) throw new AssertionError("ImageJ failed to open TIFF");
        Method load = Auto_Worm_ROI.class.getDeclaredMethod("loadRois", File.class);
        load.setAccessible(true);
        Roi[] rois = (Roi[]) load.invoke(null, new File(args[1]));
        Method measure = Auto_Worm_ROI.class.getDeclaredMethod(
                "measureWithImageJ", ImagePlus.class, String.class, Roi[].class);
        measure.setAccessible(true);
        Object measured = measure.invoke(null, image, new File(args[0]).getName(), rois);
        // The rows now travel beside the table so the caller can write the
        // image's own table first and only then fold them into the combined one.
        Field rowsField = measured.getClass().getDeclaredField("rows");
        rowsField.setAccessible(true);
        @SuppressWarnings("unchecked")
        java.util.List<Object> rows = (java.util.List<Object>) rowsField.get(measured);
        Field tableField = measured.getClass().getDeclaredField("table");
        tableField.setAccessible(true);
        ResultsTable result = (ResultsTable) tableField.get(measured);
        if (rows.size() != rois.length)
            throw new AssertionError("row data count does not match the ROI count");
        if (result.size() != rois.length || !result.columnExists("CTCF") ||
                !result.columnExists("MeasurementBackend"))
            throw new AssertionError("ImageJ measurement columns/row count are invalid");
        boolean foundBackground = false;
        for (int row = 0; row < result.size(); row++)
            if ("background".equals(result.getStringValue("Type", row))) foundBackground = true;
        if (!foundBackground) throw new AssertionError("background measurement missing");
        System.out.println("IMAGEJ_MEASUREMENT_OK rows=" + result.size());
    }
}
