"""Manifest-driven PyTorch dataset and transforms (training augmentation + evaluation)."""
import csv
import io
import os
import random

import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms

from training import config


def read_manifest(path=config.MANIFEST, split=None):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [r for r in rows if split is None or r["split"] == split]


class RandomJpeg:
    """Re-encode as JPEG with a random quality (cheap in-car cameras, video compression)."""

    def __init__(self, quality=(35, 90), p=0.5):
        self.quality, self.p = quality, p

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(*self.quality))
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomBlur:
    def __init__(self, radius=(0.3, 1.5), p=0.3):
        self.radius, self.p = radius, p

    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(random.uniform(*self.radius))) if random.random() < self.p else img


def train_transform(size=224):
    """Realistic variation only: light, colour, blur, compression, small pose/scale changes."""
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
        transforms.RandomRotation(8),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.35, contrast=0.3, saturation=0.3, hue=0.04),
        transforms.RandomGrayscale(p=0.1),          # weakens the colour-cast shortcut found in the audit
        RandomBlur(),
        RandomJpeg(),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


def eval_transform(size=224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


class FaceCropDataset(Dataset):
    def __init__(self, rows, transform, root=config.ROOT):
        self.rows, self.transform, self.root = rows, transform, root

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(os.path.join(self.root, r["image_path"])).convert("RGB")
        return self.transform(img), torch.tensor(config.CLASSES.index(r["label"]))
