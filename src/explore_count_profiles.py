from __future__ import print_function

import glob
import os
import sys

if sys.version_info[0] < 3:
    print("Python 3.9+ required")
    sys.exit(2)

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.signal import find_peaks


def analyze(dataset_root, output_root):
    mask_dir = os.path.join(dataset_root, "masks")
    image_dir = os.path.join(dataset_root, "images")
    results = []
    os.makedirs(output_root, exist_ok=True)

    for mask_path in sorted(glob.glob(os.path.join(mask_dir, "*_masks.png"))):
        stem = os.path.basename(mask_path).replace("_masks.png", "")
        labels = np.asarray(Image.open(mask_path), dtype=np.uint16)
        image = np.asarray(Image.open(os.path.join(image_dir, stem + ".png")), dtype=np.float32) / 255.0
        true_count = int(labels.max())
        ys, xs = np.nonzero(labels)
        xmin, xmax = int(xs.min()), int(xs.max())
        ymin, ymax = int(ys.min()), int(ys.max())
        scale = image.shape[1] / 1024.0
        y0 = int(round(ymin + 0.18 * (ymax - ymin)))
        y1 = int(round(ymin + 0.82 * (ymax - ymin)))

        vertically_coherent = ndimage.gaussian_filter(image, sigma=(18.0 * scale, 2.0 * scale))
        horizontal_edge = np.abs(ndimage.sobel(vertically_coherent, axis=1))
        edge_profile = horizontal_edge[y0:y1].mean(axis=0)
        edge_profile = ndimage.gaussian_filter1d(edge_profile, 1.5 * scale)
        crop = edge_profile[xmin:xmax + 1]

        # Nearby double edges are consolidated by a minimum-distance rule.
        prominence = 0.10 * (float(crop.max()) - float(np.median(crop)))
        peaks, properties = find_peaks(crop, distance=max(3, int(round(22 * scale))),
                                       prominence=max(prominence, 1e-8))
        estimated_from_edges = max(1, len(peaks) - 1)
        results.append((stem, true_count, estimated_from_edges, len(peaks)))

        width = 900
        height = 260
        canvas = Image.new("RGB", (width, height), (15, 15, 15))
        draw = ImageDraw.Draw(canvas)
        values = crop - crop.min()
        values = values / max(float(values.max()), 1e-8)
        plot_x = np.linspace(20, width - 20, len(values))
        plot_y = height - 30 - values * (height - 60)
        points = [(int(x), int(y)) for x, y in zip(plot_x, plot_y)]
        draw.line(points, fill=(80, 255, 100), width=2)
        for peak in peaks:
            x = int(plot_x[peak])
            draw.line((x, 20, x, height - 20), fill=(255, 100, 100), width=1)
        draw.text((20, 10), "%s | true=%d, edge peaks=%d, estimate=%d" %
                  (stem, true_count, len(peaks), estimated_from_edges), fill=(255, 255, 255))
        canvas.save(os.path.join(output_root, stem + "_edge_profile.png"))
    return results


def main(roots):
    all_results = []
    for index, root in enumerate(roots):
        output = os.path.join(root, "count_profiles")
        all_results.extend(analyze(root, output))
    for stem, true_count, estimate, peaks in all_results:
        print("%s: true=%d, peaks=%d, estimated=%d" % (stem, true_count, peaks, estimate))
    correct = sum(int(true_count == estimate) for _, true_count, estimate, _ in all_results)
    print("Exact count: %d/%d" % (correct, len(all_results)))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python explore_count_profiles.py <dataset-folder> [dataset-folder ...]")
        sys.exit(2)
    main(sys.argv[1:])
