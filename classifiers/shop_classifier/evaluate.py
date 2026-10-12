"""
Evaluate Shop Recognition Pipeline
==================================
Runs end-to-end evaluation on validation dataset images using `pipeline.py`,
computing localization precision/recall and attribute classification accuracy.
"""

import json
import argparse
from pathlib import Path
from PIL import Image

CURRENT_DIR = Path(__file__).resolve().parent
DATASET_DIR = CURRENT_DIR / "dataset"
WEIGHTS_DIR = CURRENT_DIR / "weights"

from pipeline import ShopRecognitionPipeline


def evaluate(max_samples=200):
    images_dir = DATASET_DIR / "images"
    labels_dir = DATASET_DIR / "labels_json"

    json_files = sorted(list(labels_dir.glob("*.json")))
    if not json_files:
        print("[Error] No validation JSON labels found in dataset/labels_json/")
        return

    pipeline = ShopRecognitionPipeline()

    tested = 0
    total_gt = 0
    total_pred = 0
    correct_keys = 0
    correct_editions = 0

    print(f"\nEvaluating up to {max_samples} validation frames...")

    for jf in json_files[:max_samples]:
        img_path = images_dir / f"{jf.stem}.png"
        if not img_path.exists():
            img_path = images_dir / f"{jf.stem}.jpg"
        if not img_path.exists():
            continue

        with open(jf, "r", encoding="utf-8") as f:
            gt_items = json.load(f)

        results = pipeline.predict(img_path)

        tested += 1
        total_gt += len(gt_items)
        total_pred += len(results)

        # Simple greedy matching by order (both are sorted left-to-right)
        for i in range(min(len(gt_items), len(results))):
            gt = gt_items[i]
            pred = results[i]
            if pred.key == gt.get("key"):
                correct_keys += 1
            if pred.edition == gt.get("edition", "None"):
                correct_editions += 1

    print("\n" + "=" * 60)
    print("SHOP RECOGNITION EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Frames Evaluated    : {tested:,}")
    print(f"Ground Truth Items  : {total_gt:,}")
    print(f"Detected Items      : {total_pred:,}")
    if total_gt > 0:
        print(f"Key Accuracy        : {(correct_keys / total_gt) * 100:.2f}%")
        print(f"Edition Accuracy    : {(correct_editions / total_gt) * 100:.2f}%")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Shop Pipeline")
    parser.add_argument("--samples", type=int, default=200, help="Max frames to evaluate")
    args = parser.parse_args()
    evaluate(max_samples=args.samples)
