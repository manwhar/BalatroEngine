"""
Stage 2 Card Crop Dataset Loader
================================
Extracts individual card crops from full hand frames and provides
attribute target dictionaries for multi-task classification.
Splits data by frame ID so cards from the same screenshot remain in the same split.
"""

import os
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
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

import sys

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# pyrefly: ignore [missing-import]
from classifiers.augmentations import BalatroStreamAugmentation

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(is_train: bool = True):
    if is_train:
        return T.Compose([
            T.Resize((CROP_HEIGHT, CROP_WIDTH), interpolation=T.InterpolationMode.BILINEAR),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            T.ToTensor(),
            BalatroStreamAugmentation(compression_prob=0.5, cursor_prob=0.3),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return T.Compose([
            T.Resize((CROP_HEIGHT, CROP_WIDTH), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


class CardCropDataset(Dataset):
    def __init__(self, data_root, split="train", val_ratio=0.20, seed=42, transform=None, cache_in_memory=True, max_cache_gb=12.0):
        super().__init__()
        self.data_root = Path(data_root)
        self.images_dir = self.data_root / "images"
        self.labels_dir = self.data_root / "labels_json"
        self.is_train = (split == "train")
        self.raw_tensor_mode = (transform is None)
        self.transform = transform

        # 1. Discover all matching frame image & json pairs
        json_files = sorted(list(self.labels_dir.glob("*.json")))
        valid_frames = []
        for jf in json_files:
            img_path = self.images_dir / f"{jf.stem}.png"
            if not img_path.exists():
                img_path = self.images_dir / f"{jf.stem}.jpg"
            if img_path.exists():
                valid_frames.append((img_path, jf))

        # Also discover shop & pack playing card crops (from self.data_root / "shop_crops")
        shop_crops_dir = self.data_root / "shop_crops"
        if shop_crops_dir.exists():
            shop_images = shop_crops_dir / "images"
            shop_labels = shop_crops_dir / "labels_json"
            if not shop_labels.exists():
                shop_labels = shop_crops_dir / "labels"
            if shop_labels.exists() and shop_images.exists():
                for jf in sorted(list(shop_labels.glob("*.json"))):
                    img_path = shop_images / f"{jf.stem}.png"
                    if not img_path.exists():
                        img_path = shop_images / f"{jf.stem}.jpg"
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
                        "selected": card.get("selected", False),
                        "debuffed": card.get("debuffed", False),
                        "face_down": card.get("face_down", False)
                    })
            except Exception as e:
                continue

        print(f"[{split.upper()}] Loaded {len(self.samples)} card crops across {len(selected_frames)} frames.")

        # Precompute contiguous target tensors
        n = len(self.samples)
        none_rank_idx = RANK_TO_IDX.get("None", 0)
        self._targets = {
            "rank": torch.tensor([RANK_TO_IDX.get(s["rank"], none_rank_idx) for s in self.samples], dtype=torch.long),
            "suit": torch.tensor([SUIT_TO_IDX.get(s["suit"], 4) for s in self.samples], dtype=torch.long),
            "edition": torch.tensor([EDITION_TO_IDX.get(s["edition"], 0) for s in self.samples], dtype=torch.long),
            "enhancement": torch.tensor([ENHANCEMENT_TO_IDX.get(s["enhancement"], 0) for s in self.samples], dtype=torch.long),
            "seal": torch.tensor([SEAL_TO_IDX.get(s["seal"], 0) for s in self.samples], dtype=torch.long),
            "debuffed": torch.tensor([1.0 if s["debuffed"] else 0.0 for s in self.samples], dtype=torch.float32),
            "face_down": torch.tensor([1.0 if s["face_down"] else 0.0 for s in self.samples], dtype=torch.float32),
        }

        self._cached_images = None
        self.uses_ram_cache = False
        est_bytes = n * 3 * CROP_HEIGHT * CROP_WIDTH
        if cache_in_memory and self.raw_tensor_mode and n > 0 and est_bytes <= int(max_cache_gb * (1024 ** 3)):
            self._preload_crops_to_ram()

    def _preload_crops_to_ram(self):
        n = len(self.samples)
        buf = torch.empty((n, 3, CROP_HEIGHT, CROP_WIDTH), dtype=torch.uint8)
        groups = defaultdict(list)
        for idx, s in enumerate(self.samples):
            groups[s["img_path"]].append((idx, s["box_pixel"]))

        def _process_frame(item):
            img_path, entries = item
            try:
                with Image.open(img_path) as full_img:
                    img_rgb = full_img.convert("RGB")
                    iw, ih = img_rgb.size
                    for idx, box in entries:
                        px, py, pw, ph = box
                        if px == 0 and py == 0 and abs(pw - iw) <= 2 and abs(ph - ih) <= 2:
                            crop = img_rgb
                        else:
                            x1 = max(0, min(iw - 1, px))
                            y1 = max(0, min(ih - 1, py))
                            x2 = max(x1 + 1, min(iw, x1 + pw))
                            y2 = max(y1 + 1, min(ih, y1 + ph))
                            crop = img_rgb.crop((x1, y1, x2, y2))
                        resized = crop.resize((CROP_WIDTH, CROP_HEIGHT), Image.Resampling.BILINEAR)
                        arr = np.asarray(resized, dtype=np.uint8)
                        buf[idx].copy_(torch.from_numpy(arr).permute(2, 0, 1))
            except Exception:
                for idx, _ in entries:
                    buf[idx].fill_(128)

        workers = min(16, max(4, os.cpu_count() or 8))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_process_frame, groups.items()))

        self._cached_images = buf
        self.uses_ram_cache = True

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.uses_ram_cache:
            tensor_img = self._cached_images[idx]
        else:
            item = self.samples[idx]
            try:
                with Image.open(item["img_path"]) as full_img:
                    full_img = full_img.convert("RGB")
                    px, py, pw, ph = item["box_pixel"]
                    iw, ih = full_img.size
                    if px == 0 and py == 0 and abs(pw - iw) <= 2 and abs(ph - ih) <= 2:
                        crop = full_img.copy()
                    else:
                        x1 = max(0, min(iw - 1, px))
                        y1 = max(0, min(ih - 1, py))
                        x2 = max(x1 + 1, min(iw, x1 + pw))
                        y2 = max(y1 + 1, min(ih, y1 + ph))
                        crop = full_img.crop((x1, y1, x2, y2))
            except Exception:
                crop = Image.new("RGB", (CROP_WIDTH, CROP_HEIGHT), (128, 128, 128))

            if self.raw_tensor_mode:
                resized = crop.resize((CROP_WIDTH, CROP_HEIGHT), Image.Resampling.BILINEAR)
                tensor_img = torch.from_numpy(np.asarray(resized, dtype=np.uint8)).permute(2, 0, 1).contiguous()
            else:
                tensor_img = self.transform(crop)

        targets = {k: v[idx] for k, v in self._targets.items()}
        return tensor_img, targets
