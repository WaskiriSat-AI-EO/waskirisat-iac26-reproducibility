"""Simple MobileNetV2 transfer learning baseline for cloud/clear classification.

Loads from balanced_manifest.json (50/50 cloud:clear).
Does a scene-level train/val split to avoid spatial leakage.
Trains only the classifier head; backbone frozen throughout.

Usage:
    python train_cloud.py
    python train_cloud.py --dataset_dir data/cloud/04_dataset --epochs 15
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS     = 10
LR         = 1e-3
VAL_FRAC   = 0.25
SEED       = 42


class CloudDataset(Dataset):
    def __init__(self, records: list, image_dir: Path, transform):
        self.records   = records
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r   = self.records[idx]
        img = Image.open(self.image_dir / r["image"]).convert("RGB")
        return self.transform(img), torch.tensor(r["label"], dtype=torch.float32)


def scene_split(records: list, val_frac: float, seed: int):
    """Split records at scene level to avoid spatial leakage."""
    rng = random.Random(seed)
    by_scene = defaultdict(list)
    for r in records:
        by_scene[r["scene_id"]].append(r)

    scenes = sorted(by_scene.keys())
    rng.shuffle(scenes)
    n_val = max(1, round(len(scenes) * val_frac))
    val_scenes   = set(scenes[:n_val])
    train_scenes = set(scenes[n_val:])

    train = [r for r in records if r["scene_id"] in train_scenes]
    val   = [r for r in records if r["scene_id"] in val_scenes]
    return train, val, train_scenes, val_scenes


def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train() if training else model.eval()
    total_loss = correct = n = 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            labels = labels.unsqueeze(1).to(device)
            if training:
                optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            preds = (out.sigmoid() > 0.5).long()
            correct += (preds == labels.long()).sum().item()
            n += imgs.size(0)
    return total_loss / n, correct / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="data/cloud/04_dataset")
    parser.add_argument("--epochs",      type=int, default=EPOCHS)
    parser.add_argument("--batch_size",  type=int, default=BATCH_SIZE)
    parser.add_argument("--lr",          type=float, default=LR)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    dataset_dir = Path(args.dataset_dir)
    image_dir   = dataset_dir / "images"

    manifest = json.load(open(dataset_dir / "balanced_manifest.json"))
    records  = manifest["records"]
    print(f"Balanced records: {len(records)}  "
          f"(cloud={sum(r['label']==1 for r in records)}  "
          f"clear={sum(r['label']==0 for r in records)})")

    train_records, val_records, train_scenes, val_scenes = scene_split(
        records, VAL_FRAC, SEED
    )
    print(f"Scene split — train scenes: {len(train_scenes)}  val scenes: {len(val_scenes)}")
    print(f"  train: {len(train_records)} images  "
          f"(cloud={sum(r['label']==1 for r in train_records)}  "
          f"clear={sum(r['label']==0 for r in train_records)})")
    print(f"  val:   {len(val_records)} images  "
          f"(cloud={sum(r['label']==1 for r in val_records)}  "
          f"clear={sum(r['label']==0 for r in val_records)})")

    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_loader = DataLoader(
        CloudDataset(train_records, image_dir, train_tf),
        batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        CloudDataset(val_records, image_dir, val_tf),
        batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}\n")

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for p in model.features.parameters():
        p.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, 1),
    )
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}\n")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}")
    print("-" * 52)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, training=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, optimizer, device, training=False)
        scheduler.step()

        marker = " ←" if vl_acc > best_val_acc else ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), dataset_dir / "mobilenetv2_cloud_best.pt")

        print(f"{epoch:>5}  {tr_loss:>10.4f}  {tr_acc:>9.3f}  {vl_loss:>8.4f}  {vl_acc:>7.3f}{marker}")

    print(f"\nBest val acc: {best_val_acc:.3f}")
    print(f"Model saved → {dataset_dir}/mobilenetv2_cloud_best.pt")


if __name__ == "__main__":
    main()
