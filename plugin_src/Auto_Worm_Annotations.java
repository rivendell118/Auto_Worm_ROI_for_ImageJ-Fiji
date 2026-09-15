import ij.IJ;
import ij.ImagePlus;
import ij.gui.Roi;
import ij.gui.Line;
import ij.io.FileInfo;
import ij.plugin.PlugIn;
import ij.plugin.frame.RoiManager;
import ij.process.FloatPolygon;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/** Saves ImageJ ROI Manager annotations into a per-image JSON sidecar. */
public class Auto_Worm_Annotations implements PlugIn {
    @Override public void run(String arg) {
        try {
            ImagePlus image = IJ.getImage();
            FileInfo info = image.getOriginalFileInfo();
            if (info == null || info.directory == null || info.fileName == null)
                throw new IllegalArgumentException("请先打开并保存 TIFF 图像。");
            RoiManager manager = RoiManager.getInstance2();
            if (manager == null || manager.getCount() == 0)
                throw new IllegalArgumentException("ROI Manager 中没有标注。\n" + instructions());

            List<Roi> heads = new ArrayList<>(), boundaries = new ArrayList<>(), exclusions = new ArrayList<>();
            for (Roi roi : manager.getRoisAsArray()) {
                String name = roi.getName() == null ? "" : roi.getName().toLowerCase(Locale.ROOT);
                if (name.startsWith("head_")) heads.add(roi);
                else if (name.startsWith("boundary_")) boundaries.add(roi);
                else if (name.startsWith("exclude_")) exclusions.add(roi);
            }
            if (heads.isEmpty() && boundaries.isEmpty() && exclusions.isEmpty())
                throw new IllegalArgumentException("没有找到 head_ / boundary_ / exclude_ 标注。\n" + instructions());

            String json = buildJson(image.getWidth(), image.getHeight(), heads, boundaries, exclusions);
            Path sidecar = new File(info.directory, info.fileName + ".autoworm.json").toPath();
            Files.write(sidecar, json.getBytes(StandardCharsets.UTF_8));
            IJ.showMessage("自动圈虫标注", "已保存：" + sidecar + "\n\n" +
                    "头向 " + heads.size() + "，分界线 " + boundaries.size() +
                    "，排除区 " + exclusions.size());
        } catch (Exception error) {
            IJ.error("自动圈虫标注", error.getMessage() == null ? error.toString() : error.getMessage());
        }
    }

    private static String instructions() {
        return "头向：直线/箭头从虫体内部画向头部，命名 head_1、head_2…\n" +
                "分界线：命名 boundary_1…\n排除区：闭合面积 ROI，命名 exclude_1…";
    }

    private static String buildJson(int width, int height, List<Roi> heads,
                                    List<Roi> boundaries, List<Roi> exclusions) {
        StringBuilder out = new StringBuilder(1024);
        out.append("{\n  \"schema_version\": 3,\n  \"source\": \"ImageJ ROI Manager\",");
        out.append("\n  \"width\": ").append(width).append(",\n  \"height\": ").append(height);
        out.append(",\n  \"arrows\": [");
        boolean first = true;
        for (Roi roi : heads) {
            if (!(roi instanceof Line)) continue;
            FloatPolygon points = ((Line) roi).getFloatPoints();
            if (points == null || points.npoints < 2) continue;
            if (!first) out.append(','); first = false;
            out.append("\n    {\"tail_x\": ").append(number(points.xpoints[0]))
                    .append(", \"tail_y\": ").append(number(points.ypoints[0]))
                    .append(", \"head_x\": ").append(number(points.xpoints[points.npoints - 1]))
                    .append(", \"head_y\": ").append(number(points.ypoints[points.npoints - 1])).append('}');
        }
        if (!first) out.append('\n');
        out.append("  ],\n  \"boundaries\": [");
        appendPolygons(out, boundaries, false);
        out.append("  ],\n  \"exclusions\": [");
        appendPolygons(out, exclusions, true);
        out.append("  ]\n}\n");
        return out.toString();
    }

    private static void appendPolygons(StringBuilder out, List<Roi> rois, boolean requireArea) {
        boolean first = true;
        for (Roi roi : rois) {
            if (requireArea && !roi.isArea()) continue;
            FloatPolygon polygon = roi.getFloatPolygon();
            if (polygon == null || polygon.npoints < (requireArea ? 3 : 2)) continue;
            if (!first) out.append(','); first = false;
            out.append("\n    {\"points\": [");
            for (int i = 0; i < polygon.npoints; i++) {
                if (i > 0) out.append(',');
                out.append('[').append(number(polygon.xpoints[i])).append(',')
                        .append(number(polygon.ypoints[i])).append(']');
            }
            out.append("]}");
        }
        if (!first) out.append('\n');
    }

    private static String number(float value) {
        return String.format(Locale.US, "%.3f", value);
    }
}
