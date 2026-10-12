r"""
Sync Shop Dataset from Balatro Mod to Shop Classifier
=====================================================
Copies newly captured images and labels from the Balatro Mod:
- `dataset_shop/images/*.png` -> `dataset/images/`
- `dataset_shop/labels/*.txt` -> `dataset/labels_detect/`
- `dataset_shop/labels/*.json` -> `dataset/labels_json/`

Checks both `%APPDATA%\Balatro\Mods\gamestate_dataset_cv\dataset_shop`
and `balatro_mod/dataset_shop`.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
TARGET_DATASET_DIR = CURRENT_DIR / "dataset"


def find_source_dataset_dir() -> Path:
    candidates = [
        Path(os.path.expandvars(r"%APPDATA%\Balatro\Mods\gamestate_dataset_cv\dataset_shop")),
        PROJECT_ROOT / "balatro_mod" / "dataset_shop",
    ]
    for c in candidates:
        if c.exists() and (c / "images").exists():
            img_count = len(list((c / "images").glob("*.png"))) + len(list((c / "images").glob("*.jpg")))
            if img_count > 0:
                return c

    # Fallback to the first candidate that exists
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def sync_dataset(clean: bool = False):
    src_dir = find_source_dataset_dir()
    src_images = src_dir / "images"
    src_labels = src_dir / "labels"

    print(f"Source Directory: {src_dir}")
    if not src_images.exists() or not src_labels.exists():
        print(f"[Error] Source directories missing in: {src_dir}")
        sys.exit(1)

    target_images = TARGET_DATASET_DIR / "images"
    target_labels_detect = TARGET_DATASET_DIR / "labels_detect"
    target_labels_json = TARGET_DATASET_DIR / "labels_json"

    for d in [target_images, target_labels_detect, target_labels_json]:
        if clean and d.exists():
            print(f"Cleaning {d.name}...")
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    img_files = list(src_images.glob("*.png")) + list(src_images.glob("*.jpg"))
    print(f"Found {len(img_files):,} images in source directory: {src_images}")

    synced_images = 0
    synced_txt = 0
    synced_json = 0

    for img_path in img_files:
        stem = img_path.stem
        txt_path = src_labels / f"{stem}.txt"
        json_path = src_labels / f"{stem}.json"

        # Copy image
        shutil.copy2(img_path, target_images / img_path.name)
        synced_images += 1

        # Copy Stage 1 YOLO detection label
        if txt_path.exists():
            shutil.copy2(txt_path, target_labels_detect / f"{stem}.txt")
            synced_txt += 1

        # Copy Stage 2 attribute JSON label
        if json_path.exists():
            shutil.copy2(json_path, target_labels_json / f"{stem}.json")
            synced_json += 1

    print("\n" + "=" * 60)
    print("SHOP DATASET SYNC COMPLETE!")
    print(f"  Images synced:      {synced_images:,} -> {target_images}")
    print(f"  Stage 1 txt labels: {synced_txt:,} -> {target_labels_detect}")
    print(f"  Stage 2 json labels:{synced_json:,} -> {target_labels_json}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync newly generated dataset from Balatro mod")
    parser.add_argument("--clean", action="store_true", help="Wipe target dataset folders before copying")
    args = parser.parse_args()
    sync_dataset(clean=args.clean)
