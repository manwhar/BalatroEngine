"""
Sync Dataset from Balatro Mod to Card Classifier
================================================
Copies newly generated images and labels from `balatro_mod/dataset/`
into `classifiers/card_classifier/dataset/`:
- `balatro_mod/dataset/images/*.png` -> `dataset/images/`
- `balatro_mod/dataset/labels/*.txt` -> `dataset/labels_card/`
- `balatro_mod/dataset/labels/*.json` -> `dataset/labels_json/`

Usage:
  python sync_dataset.py
  python sync_dataset.py --clean  # Clear existing images/labels before syncing
"""

import sys
import shutil
import argparse
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
MOD_DATASET_DIR = PROJECT_ROOT / "balatro_mod" / "dataset"
TARGET_DATASET_DIR = CURRENT_DIR / "dataset"

def sync_dataset(clean: bool = False):
    mod_images = MOD_DATASET_DIR / "images"
    mod_labels = MOD_DATASET_DIR / "labels"

    if not mod_images.exists() or not mod_labels.exists():
        print(f"[Error] Mod dataset directory not found at: {MOD_DATASET_DIR}")
        sys.exit(1)

    target_images = TARGET_DATASET_DIR / "images"
    target_labels_card = TARGET_DATASET_DIR / "labels_card"
    target_labels_json = TARGET_DATASET_DIR / "labels_json"

    for d in [target_images, target_labels_card, target_labels_json]:
        if clean and d.exists():
            print(f"Cleaning {d.name}...")
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    img_files = list(mod_images.glob("*.png")) + list(mod_images.glob("*.jpg"))
    print(f"Found {len(img_files)} images in mod output directory: {mod_images}")

    synced_images = 0
    synced_txt = 0
    synced_json = 0

    for img_path in img_files:
        stem = img_path.stem
        txt_path = mod_labels / f"{stem}.txt"
        json_path = mod_labels / f"{stem}.json"

        # Copy image
        shutil.copy2(img_path, target_images / img_path.name)
        synced_images += 1

        # Copy Stage 1 YOLO detection label
        if txt_path.exists():
            shutil.copy2(txt_path, target_labels_card / f"{stem}.txt")
            synced_txt += 1

        # Copy Stage 2 attribute JSON label
        if json_path.exists():
            shutil.copy2(json_path, target_labels_json / f"{stem}.json")
            synced_json += 1

    print("\n" + "=" * 60)
    print("DATASET SYNC COMPLETE!")
    print(f"  Images synced:      {synced_images} -> {target_images}")
    print(f"  Stage 1 txt labels: {synced_txt} -> {target_labels_card}")
    print(f"  Stage 2 json labels:{synced_json} -> {target_labels_json}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync newly generated dataset from Balatro mod")
    parser.add_argument("--clean", action="store_true", help="Wipe target dataset folders before copying")
    args = parser.parse_args()
    sync_dataset(clean=args.clean)
