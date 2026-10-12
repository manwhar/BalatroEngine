"""
Stage 2 Shop Item Crop Dataset Loader
=====================================
Extracts individual card/item crops from shop screenshots (joker bar, consumable area, shop, pack)
and provides multi-head target dictionaries for classification.
Splits data by frame ID to prevent data leakage between train and val splits.
"""

import json
import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

try:
    from .config import (
        KEY_TO_IDX, EDITION_TO_IDX,
        CROP_WIDTH, CROP_HEIGHT
    )
except ImportError:
    from config import (
        KEY_TO_IDX, EDITION_TO_IDX,
        CROP_WIDTH, CROP_HEIGHT
    )

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(is_train: bool = True):
    if is_train:
        return T.Compose([
            T.Resize((CROP_HEIGHT, CROP_WIDTH), interpolation=T.InterpolationMode.BILINEAR),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return T.Compose([
            T.Resize((CROP_HEIGHT, CROP_WIDTH), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


class ItemCropDataset(Dataset):
    def __init__(self, data_root, split="train", val_ratio=0.20, seed=42, transform=None):
        super().__init__()
        self.data_root = Path(data_root)
        self.images_dir = self.data_root / "images"
        self.labels_dir = self.data_root / "labels_json"
        self.is_train = (split == "train")
        self.transform = transform or get_transforms(is_train=self.is_train)

        # 1. Discover all matching frame image & json pairs
        json_files = sorted(list(self.labels_dir.glob("*.json")))
        valid_frames = []
        for jf in json_files:
            img_path = self.images_dir / f"{jf.stem}.png"
            if not img_path.exists():
                img_path = self.images_dir / f"{jf.stem}.jpg"
            if img_path.exists():
                valid_frames.append((img_path, jf))

        # 2. Split by frame to prevent data leakage between train and val
        rng = random.Random(seed)
        shuffled = list(valid_frames)
        rng.shuffle(shuffled)
        val_count = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0

        if split == "train":
            selected_frames = shuffled[val_count:] if len(shuffled) > 1 else shuffled
        else:
            selected_frames = shuffled[:val_count] if len(shuffled) > 1 else shuffled

        # 3. Flatten into individual item crop samples
        self.samples = []
        for img_path, jf in selected_frames:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    items = json.load(f)
                for item in items:
                    self.samples.append({
                        "img_path": img_path,
                        "box_pixel": item.get("box_pixel", [0, 0, 50, 50]),
                        "key": item.get("key", "unknown"),
                        "edition": item.get("edition", "None"),
                        "eternal": item.get("eternal", False),
                        "perishable": item.get("perishable", False),
                        "rental": item.get("rental", False),
                        "debuffed": item.get("debuffed", False),
                    })
            except Exception:
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample["img_path"]
        box = sample["box_pixel"]

        # Crop the item from the source ROI image
        with Image.open(img_path) as full_img:
            img_rgb = full_img.convert("RGB")
            iw, ih = img_rgb.size

            # box_pixel is [x, y, w, h] relative to the ROI image
            x, y, w, h = box
            x1 = max(0, int(round(x)))
            y1 = max(0, int(round(y)))
            x2 = min(iw, int(round(x + w)))
            y2 = min(ih, int(round(y + h)))

            if x2 <= x1 or y2 <= y1:
                crop = img_rgb
            else:
                crop = img_rgb.crop((x1, y1, x2, y2))

        crop_tensor = self.transform(crop)

        targets = {
            "key": torch.tensor(KEY_TO_IDX.get(sample["key"], KEY_TO_IDX.get("unknown", 0)), dtype=torch.long),
            "edition": torch.tensor(EDITION_TO_IDX.get(sample["edition"], 0), dtype=torch.long),
            "eternal": torch.tensor(1.0 if sample["eternal"] else 0.0, dtype=torch.float32),
            "perishable": torch.tensor(1.0 if sample["perishable"] else 0.0, dtype=torch.float32),
            "rental": torch.tensor(1.0 if sample["rental"] else 0.0, dtype=torch.float32),
            "debuffed": torch.tensor(1.0 if sample["debuffed"] else 0.0, dtype=torch.float32),
        }

        return crop_tensor, targets
