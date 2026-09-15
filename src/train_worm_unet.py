from __future__ import print_function

import argparse
import csv
import glob
import os
import random
import time

import numpy as np
from PIL import Image
from scipy import ndimage

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader


IMAGE_SIZE = 512


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class WormUNet(nn.Module):
    def __init__(self, base=16, in_channels=1, out_channels=3):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.enc4 = ConvBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base * 8, base * 16)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = ConvBlock(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.output = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat((self.up4(b), e4), dim=1))
        d3 = self.dec3(torch.cat((self.up3(d4), e3), dim=1))
        d2 = self.dec2(torch.cat((self.up2(d3), e2), dim=1))
        d1 = self.dec1(torch.cat((self.up1(d2), e1), dim=1))
        return self.output(d1)


def semantic_target(instance_labels, boundary_radius=4):
    target = np.zeros(instance_labels.shape, dtype=np.int64)
    for label in np.unique(instance_labels):
        if label == 0:
            continue
        mask = instance_labels == label
        eroded = ndimage.binary_erosion(mask, iterations=boundary_radius, border_value=0)
        target[eroded] = 1
        target[np.logical_and(mask, np.logical_not(eroded))] = 2
    return target


def endpoint_weight_map(instance_labels, boost=1.0, fraction=0.18, dilation=8):
    weights = np.ones(instance_labels.shape, dtype=np.float32)
    if boost <= 1.0:
        return weights
    y_grid = np.arange(instance_labels.shape[0])[:, None]
    tip_zone = np.zeros(instance_labels.shape, dtype=bool)
    for label in np.unique(instance_labels):
        if label == 0:
            continue
        mask = instance_labels == label
        ys = np.nonzero(mask)[0]
        if not len(ys):
            continue
        y_min, y_max = int(ys.min()), int(ys.max())
        extent = max(y_max - y_min + 1, 1)
        cutoff = max(int(round(fraction * extent)), 2)
        tips = mask & ((y_grid <= y_min + cutoff) | (y_grid >= y_max - cutoff))
        tip_zone |= ndimage.binary_dilation(tips, iterations=dilation)
    weights[tip_zone] = float(boost)
    return weights


def issue_from_stem(stem):
    if not stem.startswith("boundary3__"):
        return "normal"
    parts = stem.split("__", 2)
    return parts[1] if len(parts) > 1 else "normal"


def boundary_issue_weight_map(instance_labels, target, issue="normal",
                              contact_boost=1.0, continuity_boost=1.0,
                              contact_dilation=8):
    """Weight the two complementary failure modes without changing the model API.

    Merge cases emphasize the narrow corridor where dilated masks from distinct
    worms meet. This includes both the labelled borders and any true background
    gap. Split cases emphasize true worm interior, penalizing false transverse
    boundaries and holes that would break one animal into several instances.
    """
    weights = np.ones(instance_labels.shape, dtype=np.float32)
    if "merge" in issue and contact_boost > 1.0:
        proximity_count = np.zeros(instance_labels.shape, dtype=np.uint8)
        structure = ndimage.generate_binary_structure(2, 1)
        for label in np.unique(instance_labels):
            if label == 0:
                continue
            nearby = ndimage.binary_dilation(
                instance_labels == label, structure=structure,
                iterations=max(int(contact_dilation), 1))
            proximity_count += nearby.astype(np.uint8)
        contact_zone = proximity_count >= 2
        weights[contact_zone] *= float(contact_boost)
    if "split" in issue and continuity_boost > 1.0:
        weights[target == 1] *= float(continuity_boost)
    return weights


def load_pair(image_path, mask_path, image_size=IMAGE_SIZE, boundary_radius=4,
              tip_boost=1.0, tip_fraction=0.18, issue="normal",
              contact_boost=1.0, continuity_boost=1.0):
    image = Image.open(image_path).convert("L").resize((image_size, image_size), Image.Resampling.BILINEAR)
    labels = Image.open(mask_path).resize((image_size, image_size), Image.Resampling.NEAREST)
    image = np.asarray(image, dtype=np.float32) / 255.0
    labels = np.asarray(labels, dtype=np.uint16)
    target = semantic_target(labels, boundary_radius=boundary_radius)
    endpoint_weights = endpoint_weight_map(
        labels, boost=tip_boost, fraction=tip_fraction,
        dilation=max(2 * boundary_radius, 2))
    endpoint_weights *= boundary_issue_weight_map(
        labels, target, issue=issue, contact_boost=contact_boost,
        continuity_boost=continuity_boost,
        contact_dilation=max(2 * boundary_radius, 2))
    return image, labels, target, endpoint_weights


class WormDataset(Dataset):
    def __init__(self, records, augment=False, image_size=IMAGE_SIZE,
                 boundary_radius=4, tip_boost=1.0, tip_fraction=0.18,
                 contact_boost=1.0, continuity_boost=1.0):
        self.records = records
        self.augment = augment
        self.image_size = image_size
        self.boundary_radius = boundary_radius
        self.tip_boost = tip_boost
        self.tip_fraction = tip_fraction
        self.contact_boost = contact_boost
        self.continuity_boost = continuity_boost

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image, labels, target, endpoint_weights = load_pair(
            self.records[index][0], self.records[index][1], self.image_size,
            self.boundary_radius, self.tip_boost, self.tip_fraction,
            issue_from_stem(self.records[index][3]), self.contact_boost,
            self.continuity_boost)
        if self.augment:
            if random.random() < 0.5:
                image = image[:, ::-1]
                target = target[:, ::-1]
                endpoint_weights = endpoint_weights[:, ::-1]
            if random.random() < 0.5:
                image = image[::-1, :]
                target = target[::-1, :]
                endpoint_weights = endpoint_weights[::-1, :]
            gamma = random.uniform(0.65, 1.5)
            image = np.power(np.clip(image, 0, 1), gamma)
            image = np.clip(image * random.uniform(0.75, 1.25) + random.uniform(-0.08, 0.08), 0, 1)
            image = np.clip(image + np.random.normal(0, random.uniform(0, 0.025), image.shape), 0, 1)
            shift_y = random.randint(-12, 12)
            shift_x = random.randint(-12, 12)
            image = np.roll(image, (shift_y, shift_x), axis=(0, 1))
            target = np.roll(target, (shift_y, shift_x), axis=(0, 1))
            endpoint_weights = np.roll(endpoint_weights, (shift_y, shift_x), axis=(0, 1))
            if shift_y > 0:
                image[:shift_y] = 0; target[:shift_y] = 0; endpoint_weights[:shift_y] = 1
            elif shift_y < 0:
                image[shift_y:] = 0; target[shift_y:] = 0; endpoint_weights[shift_y:] = 1
            if shift_x > 0:
                image[:, :shift_x] = 0; target[:, :shift_x] = 0; endpoint_weights[:, :shift_x] = 1
            elif shift_x < 0:
                image[:, shift_x:] = 0; target[:, shift_x:] = 0; endpoint_weights[:, shift_x:] = 1
        image = np.ascontiguousarray(image[None], dtype=np.float32)
        target = np.ascontiguousarray(target, dtype=np.int64)
        endpoint_weights = np.ascontiguousarray(endpoint_weights, dtype=np.float32)
        return torch.from_numpy(image), torch.from_numpy(target), torch.from_numpy(endpoint_weights)


def discover_records(root):
    image_dir = os.path.join(root, "images")
    mask_dir = os.path.join(root, "masks")
    records = []
    for image_path in sorted(glob.glob(os.path.join(image_dir, "*.png"))):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        mask_path = os.path.join(mask_dir, stem + "_masks.png")
        labels = np.asarray(Image.open(mask_path), dtype=np.uint16)
        records.append((image_path, mask_path, int(labels.max()), stem))
    return records


def split_records(records):
    # One deterministic validation image per observed count; selection from the
    # middle of each sorted group avoids choosing only one resolution family.
    grouped = {}
    for record in records:
        grouped.setdefault(record[2], []).append(record)
    train_records, val_records = [], []
    for count, group in sorted(grouped.items()):
        group = sorted(group, key=lambda item: item[3])
        val_index = len(group) // 2
        for index, record in enumerate(group):
            (val_records if index == val_index else train_records).append(record)
    return train_records, val_records


def split_records_from_manifest(records, manifest_path):
    assignments = {}
    with open(manifest_path, "r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            assignments[row["Stem"]] = row["Split"].strip().lower()
    train_records = [record for record in records if assignments.get(record[3]) == "training"]
    val_records = [record for record in records if assignments.get(record[3]) == "validation"]
    missing = [record[3] for record in records if record[3] not in assignments]
    if missing:
        raise ValueError("Records missing from split manifest: " + ", ".join(missing))
    if not train_records or not val_records:
        raise ValueError("Split manifest must contain both training and validation records")
    return train_records, val_records


def dice_for_mask(prediction, reference):
    intersection = (prediction & reference).sum(dim=(1, 2)).float()
    denominator = prediction.sum(dim=(1, 2)).float() + reference.sum(dim=(1, 2)).float()
    return ((2 * intersection + 1) / (denominator + 1)).mean()


def weighted_cross_entropy(logits, targets, endpoint_weights, class_weights):
    loss_map = F.cross_entropy(logits, targets, weight=class_weights, reduction="none")
    return (loss_map * endpoint_weights).sum() / endpoint_weights.sum().clamp_min(1.0)


def run_validation(model, loader, class_weights, device):
    model.eval()
    losses, foreground_dices, interior_dices, boundary_dices, tip_dices = [], [], [], [], []
    with torch.no_grad():
        for images, targets, endpoint_weights in loader:
            images = images.to(device)
            targets = targets.to(device)
            endpoint_weights = endpoint_weights.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(images)
                loss = weighted_cross_entropy(logits, targets, endpoint_weights, class_weights)
            losses.append(float(loss.item()))
            prediction = torch.argmax(logits, dim=1)
            foreground_dices.append(float(dice_for_mask(prediction > 0, targets > 0).item()))
            interior_dices.append(float(dice_for_mask(prediction == 1, targets == 1).item()))
            boundary_dices.append(float(dice_for_mask(prediction == 2, targets == 2).item()))
            tip_region = endpoint_weights > 1.0
            if bool(tip_region.any()):
                tip_dices.append(float(dice_for_mask(
                    (prediction > 0) & tip_region, (targets > 0) & tip_region).item()))
            else:
                tip_dices.append(foreground_dices[-1])
    return (float(np.mean(losses)), float(np.mean(foreground_dices)),
            float(np.mean(interior_dices)), float(np.mean(boundary_dices)),
            float(np.mean(tip_dices)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--tip-boost", type=float, default=1.0)
    parser.add_argument("--tip-fraction", type=float, default=0.18)
    parser.add_argument("--hard-repeat", type=int, default=1,
                        help="repeat marked supplement training records per epoch")
    parser.add_argument("--contact-boost", type=float, default=1.0,
                        help="pixel-loss multiplier near two worms in ! cases")
    parser.add_argument("--continuity-boost", type=float, default=1.0,
                        help="interior pixel-loss multiplier in = cases")
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--postprocess-interior-threshold", type=float, default=None)
    parser.add_argument("--postprocess-erosion", type=int, default=0)
    parser.add_argument("--postprocess-min-area", type=float, default=0.005)
    parser.add_argument("--postprocess-min-height", type=float, default=0.10)
    parser.add_argument("--postprocess-max-instances", type=int, default=12)
    parser.add_argument("--normalization-mode", choices=("legacy", "background_aware"),
                        default="legacy",
                        help="records how TIFFs must be normalized during inference")
    parser.add_argument("--initial-checkpoint", default=None,
                        help="optional compatible checkpoint used to initialize fine-tuning")
    parser.add_argument("--train-all", action="store_true",
                        help="fit the production model on all annotated images")
    args = parser.parse_args()

    random.seed(20260905)
    np.random.seed(20260905)
    torch.manual_seed(20260905)
    records = discover_records(args.dataset)
    if args.train_all:
        train_records, val_records = records, records
    elif args.split_manifest:
        train_records, val_records = split_records_from_manifest(records, args.split_manifest)
    else:
        train_records, val_records = split_records(records)
    if args.hard_repeat < 1:
        raise ValueError("--hard-repeat must be at least 1")
    original_train_count = len(train_records)
    marked_records = [record for record in train_records
                      if issue_from_stem(record[3]) != "normal"]
    if args.hard_repeat > 1:
        train_records = train_records + marked_records * (args.hard_repeat - 1)
    print("Training:", [(r[3], r[2]) for r in train_records])
    print("Validation:", [(r[3], r[2]) for r in val_records])
    print("Training records: %d unique, %d marked, %d effective" %
          (original_train_count, len(marked_records), len(train_records)))

    if args.image_size % 16:
        raise ValueError("--image-size must be divisible by 16")
    boundary_radius = max(2, int(round(4.0 * args.image_size / IMAGE_SIZE)))
    train_loader = DataLoader(WormDataset(
        train_records, augment=True, image_size=args.image_size,
        boundary_radius=boundary_radius, tip_boost=args.tip_boost,
        tip_fraction=args.tip_fraction, contact_boost=args.contact_boost,
        continuity_boost=args.continuity_boost), batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(WormDataset(
        val_records, augment=False, image_size=args.image_size,
        boundary_radius=boundary_radius, tip_boost=args.tip_boost,
        tip_fraction=args.tip_fraction, contact_boost=1.0,
        continuity_boost=1.0), batch_size=1,
                            shuffle=False, num_workers=0, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WormUNet().to(device)
    if args.initial_checkpoint:
        initial = torch.load(args.initial_checkpoint, map_location=device)
        model.load_state_dict(initial["model_state"])
        print("Initialized from:", args.initial_checkpoint)
    class_weights = torch.tensor([1.0, 1.5, 5.0], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    default_output_name = "worm_unet_final" if args.train_all else "worm_unet"
    output_dir = os.path.join(args.dataset, args.output_name or default_output_name)
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "training_log.csv")
    checkpoint_path = os.path.join(output_dir, "best_model.pt")
    best_score = -1.0
    start = time.time()

    with open(log_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_foreground_dice",
                         "val_interior_dice", "val_boundary_dice", "val_tip_dice",
                         "selection_score", "elapsed_seconds"])
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_losses = []
            for images, targets, endpoint_weights in train_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                endpoint_weights = endpoint_weights.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    logits = model(images)
                    loss = weighted_cross_entropy(logits, targets, endpoint_weights, class_weights)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_losses.append(float(loss.item()))
            validate_now = (epoch == 1 or epoch == args.epochs or
                            epoch % max(args.validate_every, 1) == 0)
            if validate_now:
                val_loss, val_dice, val_interior_dice, val_boundary_dice, val_tip_dice = run_validation(
                    model, val_loader, class_weights, device)
                if args.tip_boost > 1.0:
                    selection_score = (0.25 * val_interior_dice + 0.35 * val_boundary_dice +
                                       0.40 * val_tip_dice)
                else:
                    selection_score = 0.35 * val_interior_dice + 0.65 * val_boundary_dice
            else:
                val_loss = val_dice = val_interior_dice = val_boundary_dice = val_tip_dice = float("nan")
                selection_score = float("nan")
            train_loss = float(np.mean(train_losses))
            elapsed = time.time() - start
            writer.writerow([epoch, train_loss, val_loss, val_dice, val_interior_dice,
                             val_boundary_dice, val_tip_dice, selection_score, elapsed])
            handle.flush()
            if validate_now and selection_score > best_score:
                best_score = selection_score
                torch.save({
                    "model_state": model.state_dict(),
                    "base": 16,
                    "image_size": args.image_size,
                    "validation_records": [r[3] for r in val_records],
                    "val_foreground_dice_at_best": val_dice,
                    "val_interior_dice_at_best": val_interior_dice,
                    "val_boundary_dice_at_best": val_boundary_dice,
                    "best_selection_score": best_score,
                    "boundary_radius": boundary_radius,
                    "tip_boost": args.tip_boost,
                    "tip_fraction": args.tip_fraction,
                    "hard_repeat": args.hard_repeat,
                    "contact_boost": args.contact_boost,
                    "continuity_boost": args.continuity_boost,
                    "val_tip_dice_at_best": val_tip_dice,
                    "trained_on_all_records": bool(args.train_all),
                    "postprocess_interior_threshold": args.postprocess_interior_threshold,
                    "postprocess_erosion": args.postprocess_erosion,
                    "postprocess_min_area": args.postprocess_min_area,
                    "postprocess_min_height": args.postprocess_min_height,
                    "postprocess_max_instances": args.postprocess_max_instances,
                    "normalization_mode": args.normalization_mode,
                    "initialized_from": args.initial_checkpoint,
                }, checkpoint_path)
            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                print(("epoch %d/%d train_loss=%.4f val_loss=%.4f val_fg=%.4f "
                       "val_interior=%.4f val_boundary=%.4f val_tip=%.4f score=%.4f elapsed=%.1fs") %
                      (epoch, args.epochs, train_loss, val_loss, val_dice,
                       val_interior_dice, val_boundary_dice, val_tip_dice,
                       selection_score, elapsed), flush=True)
    print("Best validation selection score %.4f" % best_score)
    print("Checkpoint:", checkpoint_path)


if __name__ == "__main__":
    main()
