from __future__ import print_function

import argparse
import csv
import os
import random
import time

import numpy as np
from PIL import Image
from scipy import ndimage
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from evaluate_worm_unet import pair_instances
from train_worm_unet import WormUNet, discover_records, split_records_from_manifest


def crop_with_padding(array, x0, y0, size, fill=0):
    result = np.full((size, size), fill, dtype=array.dtype)
    x1, y1 = x0 + size, y0 + size
    src_x0, src_y0 = max(x0, 0), max(y0, 0)
    src_x1, src_y1 = min(x1, array.shape[1]), min(y1, array.shape[0])
    if src_x1 > src_x0 and src_y1 > src_y0:
        result[src_y0 - y0:src_y1 - y0, src_x0 - x0:src_x1 - x0] = \
            array[src_y0:src_y1, src_x0:src_x1]
    return result


def endpoint_crop_origin(mask, side, patch_size, jitter_x=0, jitter_y=0):
    yy, xx = np.nonzero(mask)
    if not len(yy):
        return 0, 0
    height = max(int(yy.max()) - int(yy.min()) + 1, 1)
    band = max(int(round(0.08 * height)), 2)
    selected = yy <= yy.min() + band if side == "top" else yy >= yy.max() - band
    center_x = int(round(float(xx[selected].mean()))) + jitter_x
    if side == "top":
        y0 = int(round(float(yy.min()) - 0.30 * patch_size)) + jitter_y
    else:
        y0 = int(round(float(yy.max()) - 0.70 * patch_size)) + jitter_y
    return center_x - patch_size // 2, y0


class TipDataset(Dataset):
    def __init__(self, records, guide_dir, image_size=768, patch_size=192, augment=False):
        self.patch_size = patch_size
        self.augment = augment
        self.items = []
        self.cache = {}
        for image_path, mask_path, count, stem in records:
            image = np.asarray(Image.open(image_path).convert("L").resize(
                (image_size, image_size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
            target = np.asarray(Image.open(mask_path).resize(
                (image_size, image_size), Image.Resampling.NEAREST), dtype=np.uint16)
            guide_path = os.path.join(guide_dir, stem + "_predicted_instances.png")
            if os.path.isfile(guide_path):
                guide = np.asarray(Image.open(guide_path).resize(
                    (image_size, image_size), Image.Resampling.NEAREST), dtype=np.uint16)
                pairs, _ = pair_instances(guide, target)
                mapping = {ref_id: pred_id for ref_id, pred_id, dice in pairs if dice >= 0.15}
            else:
                guide = np.zeros_like(target)
                mapping = {}
            self.cache[stem] = (image, target, guide, mapping)
            for label in range(1, count + 1):
                self.items.append((stem, label, "top"))
                self.items.append((stem, label, "bottom"))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        stem, label, side = self.items[index]
        image, labels, guide_labels, mapping = self.cache[stem]
        target_full = labels == label
        pred_id = mapping.get(label)
        guide_full = guide_labels == pred_id if pred_id is not None else target_full.copy()
        anchor = guide_full if guide_full.sum() >= 20 else target_full
        jitter_x = random.randint(-10, 10) if self.augment else 0
        jitter_y = random.randint(-8, 8) if self.augment else 0
        x0, y0 = endpoint_crop_origin(anchor, side, self.patch_size, jitter_x, jitter_y)
        image_patch = crop_with_padding(image, x0, y0, self.patch_size, 0.0)
        target_patch = crop_with_padding(target_full, x0, y0, self.patch_size, False)
        guide_patch = crop_with_padding(guide_full, x0, y0, self.patch_size, False)

        if self.augment:
            shift_x, shift_y = random.randint(-5, 5), random.randint(-5, 5)
            guide_patch = ndimage.shift(guide_patch.astype(np.float32), (shift_y, shift_x),
                                        order=0, mode="constant", cval=0) > 0.5
            morphology = random.randint(-3, 3)
            if morphology > 0:
                guide_patch = ndimage.binary_dilation(guide_patch, iterations=morphology)
            elif morphology < 0:
                guide_patch = ndimage.binary_erosion(guide_patch, iterations=-morphology)
            if random.random() < 0.5:
                image_patch = image_patch[:, ::-1]
                target_patch = target_patch[:, ::-1]
                guide_patch = guide_patch[:, ::-1]
            gamma = random.uniform(0.75, 1.35)
            image_patch = np.power(np.clip(image_patch, 0, 1), gamma)
            image_patch = np.clip(image_patch * random.uniform(0.85, 1.15) +
                                  random.uniform(-0.04, 0.04), 0, 1)

        edge = target_patch ^ ndimage.binary_erosion(target_patch)
        edge_zone = ndimage.binary_dilation(edge, iterations=4)
        pixel_weight = np.ones(target_patch.shape, dtype=np.float32)
        pixel_weight[edge_zone] = 4.0
        inputs = np.stack((image_patch, guide_patch.astype(np.float32)), axis=0)
        return (torch.from_numpy(np.ascontiguousarray(inputs, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(target_patch, dtype=np.int64)),
                torch.from_numpy(np.ascontiguousarray(pixel_weight, dtype=np.float32)))


def refinement_loss(logits, target, pixel_weight):
    ce = F.cross_entropy(logits, target, weight=torch.tensor(
        [1.0, 2.0], device=logits.device), reduction="none")
    ce = (ce * pixel_weight).sum() / pixel_weight.sum().clamp_min(1.0)
    probability = torch.softmax(logits, dim=1)[:, 1]
    reference = target.float()
    intersection = (probability * reference).sum(dim=(1, 2))
    dice = (2 * intersection + 1) / (
        probability.sum(dim=(1, 2)) + reference.sum(dim=(1, 2)) + 1)
    return ce + (1 - dice.mean())


def validate(model, loader, device):
    model.eval()
    losses, dices = [], []
    with torch.no_grad():
        for inputs, targets, weights in loader:
            inputs, targets, weights = inputs.to(device), targets.to(device), weights.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(inputs)
                loss = refinement_loss(logits, targets, weights)
            prediction = torch.argmax(logits, dim=1) > 0
            reference = targets > 0
            intersection = (prediction & reference).sum(dim=(1, 2)).float()
            denominator = prediction.sum(dim=(1, 2)).float() + reference.sum(dim=(1, 2)).float()
            losses.append(float(loss.item()))
            dices.extend(((2 * intersection + 1) / (denominator + 1)).cpu().tolist())
    return float(np.mean(losses)), float(np.mean(dices))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("guide_dir")
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--train-all", action="store_true",
                        help="train the production model on every annotated image")
    parser.add_argument("--output-name", default="tip_refiner")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--patch-size", type=int, default=192)
    parser.add_argument("--probability-threshold", type=float, default=0.40)
    parser.add_argument("--replace-fraction", type=float, default=0.14)
    parser.add_argument("--initial-checkpoint", default=None)
    parser.add_argument("--validate-every", type=int, default=1)
    args = parser.parse_args()
    random.seed(20260907)
    np.random.seed(20260907)
    torch.manual_seed(20260907)
    records = discover_records(args.dataset)
    if args.train_all:
        training = validation = records
    elif args.split_manifest:
        training, validation = split_records_from_manifest(records, args.split_manifest)
    else:
        raise SystemExit("Provide --split-manifest for validation training, or --train-all for production training.")
    train_dataset = TipDataset(training, args.guide_dir, args.image_size, args.patch_size, True)
    val_dataset = TipDataset(validation, args.guide_dir, args.image_size, args.patch_size, False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WormUNet(base=12, in_channels=2, out_channels=2).to(device)
    if args.initial_checkpoint:
        initial = torch.load(args.initial_checkpoint, map_location=device)
        model.load_state_dict(initial["model_state"])
        print("Initialized from:", args.initial_checkpoint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output_dir = os.path.join(args.dataset, args.output_name)
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "best_model.pt")
    log_path = os.path.join(output_dir, "training_log.csv")
    best_dice = -1.0
    start = time.time()
    with open(log_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_patch_dice", "elapsed_seconds"])
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for inputs, targets, weights in train_loader:
                inputs, targets, weights = inputs.to(device), targets.to(device), weights.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    logits = model(inputs)
                    loss = refinement_loss(logits, targets, weights)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.item()))
            validate_now = (epoch == 1 or epoch == args.epochs or
                            epoch % max(args.validate_every, 1) == 0)
            if validate_now:
                val_loss, val_dice = validate(model, val_loader, device)
            else:
                val_loss = val_dice = float("nan")
            elapsed = time.time() - start
            writer.writerow([epoch, float(np.mean(losses)), val_loss, val_dice, elapsed])
            handle.flush()
            if validate_now and val_dice > best_dice:
                best_dice = val_dice
                torch.save({
                    "model_state": model.state_dict(), "base": 12,
                    "in_channels": 2, "out_channels": 2,
                    "image_size": args.image_size, "patch_size": args.patch_size,
                    "probability_threshold": args.probability_threshold,
                    "replace_fraction": args.replace_fraction,
                    "trained_on_all_records": bool(args.train_all),
                    "initialized_from": args.initial_checkpoint,
                    "validation_records": [record[3] for record in validation],
                    "best_val_patch_dice": best_dice,
                }, checkpoint_path)
            if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
                print("epoch %d/%d train_loss=%.4f val_loss=%.4f val_patch_dice=%.4f elapsed=%.1fs" %
                      (epoch, args.epochs, np.mean(losses), val_loss, val_dice, elapsed), flush=True)
    print("Best validation patch Dice %.4f" % best_dice)
    print("Checkpoint:", checkpoint_path)


if __name__ == "__main__":
    main()
