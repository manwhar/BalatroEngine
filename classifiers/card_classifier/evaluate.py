"""
Evaluate Two-Stage Card Recognition Pipeline
============================================
Runs Stage 1 detector + Stage 2 classifier on a sample image or validation set.
Prints reconstructed cards and compares directly against ground truth JSON.
"""

import sys
import json
import argparse
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
DATASET_DIR = CURRENT_DIR / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
JSON_LABELS_DIR = DATASET_DIR / "labels_json"

from pipeline import CardRecognitionPipeline


def run_evaluation(image_path=None, detector_weights=None, classifier_weights=None):
    pipeline = CardRecognitionPipeline(
        detector_path=detector_weights,
        classifier_path=classifier_weights
    )

    if image_path is None:
        val_imgs = sorted(list(IMAGES_DIR.glob("*.png")))
        if not val_imgs:
            print("No images found in dataset/images/!")
            return
        image_path = val_imgs[0]

    image_path = Path(image_path)
    print(f"\nEvaluating on image: {image_path.name}")

    # Run Two-Stage Pipeline
    results = pipeline.predict(image_path)

    # Check for ground-truth JSON
    gt_file = JSON_LABELS_DIR / f"{image_path.stem}.json"
    gt_cards = []
    if gt_file.exists():
        with open(gt_file, "r", encoding="utf-8") as f:
            gt_cards = json.load(f)

    print("\n" + "=" * 70)
    print(f"RECONSTRUCTED HAND ({len(results)} cards detected left-to-right)")
    print("=" * 70)

    for i, c in enumerate(results):
        gt = gt_cards[i] if i < len(gt_cards) else None
        
        # Check match with ground truth
        match_str = ""
        if gt:
            rank_match = (c.rank == gt.get("rank"))
            suit_match = (c.suit == gt.get("suit"))
            ed_match = (c.edition == gt.get("edition"))
            enh_match = (c.enhancement == gt.get("enhancement"))
            seal_match = (c.seal == gt.get("seal"))
            all_match = rank_match and suit_match and ed_match and enh_match and seal_match
            match_str = " [ALL CORRECT]" if all_match else " [MISMATCH]"

        print(f"Card {c.index:2d}: {c.display_name}{match_str}")
        print(f"   Rank       : {c.rank:<8} (conf: {c.confidences['rank']:.2f})")
        print(f"   Suit       : {c.suit:<8} (conf: {c.confidences['suit']:.2f})")
        print(f"   Edition    : {c.edition:<8} (conf: {c.confidences['edition']:.2f})")
        print(f"   Enhancement: {c.enhancement:<8} (conf: {c.confidences['enhancement']:.2f})")
        print(f"   Seal       : {c.seal:<8} (conf: {c.confidences['seal']:.2f})")
        print(f"   Selected   : {c.selected}")
        if gt and not all_match:
            print(f"   --> Ground Truth: {gt.get('rank')} of {gt.get('suit')} | Ed: {gt.get('edition')} | Enh: {gt.get('enhancement')} | Seal: {gt.get('seal')}")
        print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Two-Stage Card Recognition Pipeline")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--detector", type=str, default=None, help="Path to detector weights")
    parser.add_argument("--classifier", type=str, default=None, help="Path to classifier weights")
    args = parser.parse_args()

    run_evaluation(args.image, args.detector, args.classifier)
