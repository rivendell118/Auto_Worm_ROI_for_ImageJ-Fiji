"""Cache low-mode probabilities and inspect connected interiors on TIFF fixtures."""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from train_worm_unet import WormUNet
from inspect_roi_dataset import image_array
from prepare_segmentation_dataset import normalize_for_segmentation
from evaluate_worm_unet import semantic_to_instances, colorize_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    output = ROOT / 'validation_outputs' / ('gap_' + args.folder.name)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = torch.load(ROOT / '自动圈虫/models/0.2.0/worm.pt', map_location=device)
    metadata = {k: v for k, v in checkpoint.items() if k not in ('model_state', 'optimizer_state')}
    print({k: v for k, v in metadata.items() if not isinstance(v, (list, dict))}, flush=True)
    size = int(checkpoint.get('image_size', 512))
    model = WormUNet(base=int(checkpoint.get('base', 16))).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    for path in sorted(args.folder.glob('*.tif')):
        cache = output / (path.stem + '.npz')
        if cache.exists():
            data = np.load(cache)
            image, prob = data['image'], data['prob']
        else:
            raw = image_array(Image.open(path))
            normalized = normalize_for_segmentation(raw, mode=checkpoint.get('normalization_mode', 'legacy'))
            image = np.asarray(Image.fromarray(normalized).resize((size, size), Image.Resampling.BILINEAR))
            tensor = torch.from_numpy(image.astype(np.float32)[None, None] / 255).to(device)
            with torch.no_grad():
                prob = torch.softmax(model(tensor), dim=1)[0].float().cpu().numpy()
            np.savez_compressed(cache, image=image, prob=prob)
        instances, _ = semantic_to_instances(
            prob, interior_threshold=checkpoint.get('postprocess_interior_threshold', .35),
            erosion_iterations=checkpoint.get('postprocess_erosion', 3),
            min_area_fraction=checkpoint.get('postprocess_min_area', .005),
            min_height_fraction=checkpoint.get('postprocess_min_height', .10),
            max_instances=checkpoint.get('postprocess_max_instances', 12))
        Image.fromarray(instances).save(output / (path.stem + '_coarse.png'))
        panels = [np.repeat(image[:, :, None], 3, axis=2),
                  (np.repeat(image[:, :, None], 3, axis=2) * .6 + colorize_labels(instances) * .4).astype(np.uint8),
                  np.stack((prob[2]*255, prob[1]*255, prob[0]*255), axis=2).astype(np.uint8)]
        canvas = Image.fromarray(np.concatenate(panels, axis=1))
        canvas.thumbnail((1536, 650))
        canvas.save(output / (path.stem + '_inspect.jpg'))
        stats = []
        for label in range(1, int(instances.max())+1):
            mask = instances == label
            yy, xx = np.nonzero(mask)
            splits = []
            for threshold in (.45,.55,.65,.75,.85):
                seeds, n = ndimage.label(ndimage.binary_erosion(mask & (prob[1] >= threshold), iterations=2))
                pieces = []
                for part in range(1,n+1):
                    y,x=np.nonzero(seeds==part)
                    if len(y)>mask.sum()*.08 and y.max()-y.min()>(yy.max()-yy.min())*.25:
                        pieces.append((len(y),int(y.max()-y.min()),round(float(x.mean()),1)))
                if len(pieces)>1:
                    splits.append((threshold,pieces))
            stats.append((label,len(yy),int(yy.max()-yy.min()),round(float(xx.mean()),1),splits))
        print(path.name, int(instances.max()), stats, flush=True)


if __name__ == '__main__':
    main()
