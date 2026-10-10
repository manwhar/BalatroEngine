"""
Validation and Inference Test Script
====================================
Runs the trained YOLO weights on a sample image or validation set,
groups bounding boxes by x-coordinate, and displays reconstructed cards.
"""

import sys
from pathlib import Path
from collections import defaultdict
import torch
# pyrefly: ignore [missing-import]
from ultralytics import YOLO

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
BEST_WEIGHTS = CURRENT_DIR / "runs" / "sanity_check" / "weights" / "best.pt"

# Categories based on the naming convention in classes.txt
def categorize_class(class_name: str) -> str:
    if class_name.startswith("Edition_"):
        return "edition"
    elif class_name.startswith("Enhancement_"):
        return "enhancement"
    elif class_name.startswith("Seal_"):
        return "seal"
    elif class_name == "Card_Selected":
        return "selected"
    else:
        # e.g. Ace_Spades, 5_Hearts, Stone_Card
        return "identity"

def parse_detections(results, x_threshold=25.0):
    boxes = results[0].boxes
    names = results[0].names

    raw_items = []
    for box in boxes:
        cls_id = int(box.cls[0])
        class_name = names[cls_id]
        conf = float(box.conf[0])
        # xywh format: x_center, y_center, width, height
        x_center = float(box.xywh[0][0])
        y_center = float(box.xywh[0][1])

        raw_items.append({
            "class": class_name,
            "category": categorize_class(class_name),
            "conf": conf,
            "x": x_center,
            "y": y_center
        })

    # Sort left to right
    raw_items.sort(key=lambda item: item["x"])

    # Group into cards by horizontal clustering
    cards = []
    current_card_items = []
    current_cluster_x = None

    for item in raw_items:
        if current_cluster_x is None:
            current_cluster_x = item["x"]
            current_card_items.append(item)
        elif abs(item["x"] - current_cluster_x) <= x_threshold:
            # Update running average or keep base
            current_card_items.append(item)
        else:
            # New card cluster
            cards.append(_summarize_card_items(current_card_items))
            current_card_items = [item]
            current_cluster_x = item["x"]

    if current_card_items:
        cards.append(_summarize_card_items(current_card_items))

    return cards

def _summarize_card_items(items):
    summary = {
        "x": sum(it["x"] for it in items) / len(items),
        "identity": None,
        "edition": None,
        "enhancement": None,
        "seal": None,
        "selected": False,
        "raw_items": items
    }
    for it in items:
        cat = it["category"]
        if cat == "identity":
            if summary["identity"] is None or it["conf"] > summary["raw_items"][0].get("conf", 0):
                summary["identity"] = it["class"]
        elif cat == "edition":
            summary["edition"] = it["class"].replace("Edition_", "")
        elif cat == "enhancement":
            summary["enhancement"] = it["class"].replace("Enhancement_", "")
        elif cat == "seal":
            summary["seal"] = it["class"].replace("Seal_", "")
        elif cat == "selected":
            summary["selected"] = True

    return summary

def get_latest_weights() -> Path | None:
    runs_dir = CURRENT_DIR / "runs"
    if not runs_dir.exists():
        return None
    candidates = list(runs_dir.glob("*/weights/best.pt"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def run_eval(weights_path=None, test_img_path=None):
    if weights_path is None:
        weights_path = get_latest_weights()

    if weights_path is None:
        print("No trained weights found in runs/! Train a model first with train.py.")
        return

    weights_file = Path(weights_path)
    # If a run folder name was passed, resolve to weights/best.pt
    if weights_file.is_dir():
        weights_file = weights_file / "weights" / "best.pt"
    elif not weights_file.is_absolute() and not weights_file.exists():
        candidate_run = CURRENT_DIR / "runs" / weights_path / "weights" / "best.pt"
        candidate_weight = CURRENT_DIR / "weights" / weights_path
        if candidate_run.exists():
            weights_file = candidate_run
        elif candidate_weight.exists():
            weights_file = candidate_weight

    if not weights_file.exists():
        print(f"Weights file not found at: {weights_file}")
        return

    model = YOLO(str(weights_file))
    if test_img_path is None:
        sample_images = list((CURRENT_DIR / "data_split" / "val" / "images").glob("*.png"))
        if not sample_images:
            sample_images = list((CURRENT_DIR / "dataset" / "images").glob("*.png"))
        if not sample_images:
            sample_images = list((PROJECT_ROOT / "dataset" / "card_classifier" / "images").glob("*.png"))
        if not sample_images:
            print("No test images found!")
            return
        test_img_path = sample_images[0]

    print(f"Testing weights: {weights_file}")
    print(f"Testing on image: {test_img_path}")
    results = model(str(test_img_path))
    
    parsed_cards = parse_detections(results)
    print(f"\n--- Parsed Hand ({len(parsed_cards)} cards detected left-to-right) ---")
    for idx, card in enumerate(parsed_cards, 1):
        print(f"Card {idx:2d} (x={card['x']:.1f}):")
        print(f"   Identity:    {card['identity']}")
        print(f"   Edition:     {card['edition']}")
        print(f"   Enhancement: {card['enhancement']}")
        print(f"   Seal:        {card['seal']}")
        print(f"   Selected:    {card['selected']}")


if __name__ == "__main__":
    weights = sys.argv[1] if len(sys.argv) > 1 else None
    test_img = sys.argv[2] if len(sys.argv) > 2 else None
    run_eval(weights, test_img)

