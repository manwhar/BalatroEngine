"""
Sync Dataset from Balatro Mod to Card Classifier
================================================
Copies newly generated images and labels from `balatro_mod/dataset/`
into `classifiers/card_classifier/dataset/`:
- `balatro_mod/dataset/images/*.png` -> `dataset/images/`
- `balatro_mod/dataset/labels/*.txt` -> `dataset/labels_card/`
- `balatro_mod/dataset/labels/*.json` -> `dataset/labels_json/`
"""

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
TARGET_DATASET_DIR = CURRENT_DIR / "dataset"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# pyrefly: ignore [missing-import]
from classifiers.common.dataset_sync import sync_mod_dataset


def sync_dataset(clean: bool = False):
    return sync_mod_dataset(
        source_subfolder="dataset",
        target_dataset_dir=TARGET_DATASET_DIR,
        labels_subdir_name="labels_card",
        clean=clean
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync Card Dataset from Balatro Mod")
    parser.add_argument("--clean", action="store_true", help="Clear existing images/labels before syncing")
    args = parser.parse_args()
    sync_dataset(clean=args.clean)
