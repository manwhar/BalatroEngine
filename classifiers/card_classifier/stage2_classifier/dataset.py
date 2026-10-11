"""
Stage 2 Card Crop Dataset Loader
================================
Extracts individual card crops from full hand frames and provides
attribute target dictionaries for multi-task classification.
Splits data by frame ID so cards from the same screenshot remain in the same split.
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
        RANK_TO_IDX, SUIT_TO_IDX, EDITION_TO_IDX, ENHANCEMENT_TO_IDX, SEAL_TO_IDX,
        CROP_WIDTH, CROP_HEIGHT
    )
except ImportError:
    from config import (
        RANK_TO_IDX, SUIT_TO_IDX, EDITION_TO_IDX, ENHANCEMENT_TO_IDX, SEAL_TO_IDX,
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


class CardCropDataset(Dataset):
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
        val_count = max(1, int(len(shuffled) * val_ratio))

        if split == "train":
            selected_frames = shuffled[val_count:]
        else:
            selected_frames = shuffled[:val_count]

        # 3. Flatten into individual card crop samples
        self.samples = []
        for img_path, jf in selected_frames:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    cards = json.load(f)
                for card in cards:
                    self.samples.append({
                        "img_path": img_path,
                        "box_pixel": card["box_pixel"],
                        "rank": card.get("rank", "None"),
                        "suit": card.get("suit", "None"),
                        "edition": card.get("edition", "None"),
                        "enhancement": card.get("enhancement", "None"),
                        "seal": card.get("seal", "None"),
                        "selected": card.get("selected", False)
                    })
            except Exception as e:
                continue

        print(f"[{split.upper()}] Loaded {len(self.samples)} card crops across {len(selected_frames)} frames.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # Load image and crop card region
        try:
            with Image.open(item["img_path"]) as full_img:
                full_img = full_img.convert("RGB")
                px, py, pw, ph = item["box_pixel"]
                # Safeguard against out-of-bound coords
                iw, ih = full_img.size
                x1 = max(0, min(iw - 1, px))
                y1 = max(0, min(ih - 1, py))
                x2 = max(x1 + 1, min(iw, x1 + pw))
                y2 = max(y1 + 1, min(ih, y1 + ph))
                crop = full_img.crop((x1, y1, x2, y2))
        except Exception:
            crop = Image.new("RGB", (CROP_WIDTH, CROP_HEIGHT), (128, 128, 128))

        tensor_img = self.transform(crop)

        targets = {
            "rank": torch.tensor(RANK_TO_IDX.get(item["rank"], 0), dtype=torch.long),
            "suit": torch.tensor(SUIT_TO_IDX.get(item["suit"], 4), dtype=torch.long),
            "edition": torch.tensor(EDITION_TO_IDX.get(item["edition"], 0), dtype=torch.long),
            "enhancement": torch.tensor(ENHANCEMENT_TO_IDX.get(item["enhancement"], 0), dtype=torch.long),
            "seal": torch.tensor(SEAL_TO_IDX.get(item["seal"], 0), dtype=torch.long),
        }

        return tensor_img, targets
