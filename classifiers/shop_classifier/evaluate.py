"""
Evaluate Shop Recognition Pipeline
==================================
Runs end-to-end evaluation on validation dataset images using `pipeline.py`,
computing spatial IoU-based localization precision/recall and attribute classification accuracy.
"""

import json
import argparse
from pathlib import Path
from PIL import Image

CURRENT_DIR = Path(__file__).resolve().parent
DATASET_DIR = CURRENT_DIR / "dataset"
WEIGHTS_DIR = CURRENT_DIR / "weights"

# pyrefly: ignore [missing-import]
from pipeline import ShopRecognitionPipeline


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """
    Computes Intersection-over-Union (IoU) between two bounding boxes.
    Boxes are in [x1, y1, x2, y2] format.
    """
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    box_a_area = max(0.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    box_b_area = max(0.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))

    union_area = box_a_area + box_b_area - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def match_predictions_to_gt(gt_items: list[dict], predictions, iou_threshold: float = 0.4):
    """
    Greedy bipartite matching based on spatial IoU between predicted boxes and ground-truth boxes.
    Handles any layout geometry (1-row, 2-row, vouchers, booster packs, etc.).
    """
    gt_boxes = []
    for gt in gt_items:
        bp = gt.get("box_pixel", [0, 0, 0, 0])
        # bp is [x, y, w, h] relative to ROI -> convert to [x1, y1, x2, y2]
        gt_boxes.append([bp[0], bp[1], bp[0] + bp[2], bp[1] + bp[3]])

    pred_boxes = [p.box for p in predictions]

    # Compute candidate pairs with IoU >= iou_threshold
    pairs = []
    for g_idx, g_box in enumerate(gt_boxes):
        for p_idx, p_box in enumerate(pred_boxes):
            iou = compute_iou(g_box, p_box)
            if iou >= iou_threshold:
                pairs.append((iou, g_idx, p_idx))

    # Match greedily by descending IoU
    pairs.sort(key=lambda x: x[0], reverse=True)

    matched_gt = set()
    matched_pred = set()
    matches = []

    for iou, g_idx, p_idx in pairs:
        if g_idx not in matched_gt and p_idx not in matched_pred:
            matched_gt.add(g_idx)
            matched_pred.add(p_idx)
            matches.append((g_idx, p_idx, iou))

    unmatched_gt = set(range(len(gt_items))) - matched_gt
    unmatched_pred = set(range(len(predictions))) - matched_pred

    return matches, unmatched_gt, unmatched_pred


def evaluate(max_samples: int = 200, iou_threshold: float = 0.4):
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
    total_matched = 0
    correct_keys = 0
    correct_editions = 0
    correct_card_types = 0
    sum_iou = 0.0

    print(f"\nEvaluating up to {max_samples} validation frames (IoU threshold: {iou_threshold:.2f})...")

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

        matches, _, _ = match_predictions_to_gt(gt_items, results, iou_threshold=iou_threshold)
        total_matched += len(matches)

        for g_idx, p_idx, iou in matches:
            gt = gt_items[g_idx]
            pred = results[p_idx]
            sum_iou += iou

            if pred.key == gt.get("key"):
                correct_keys += 1
            if pred.edition == gt.get("edition", "None"):
                correct_editions += 1
            if pred.card_type.lower() == str(gt.get("card_type", "")).lower():
                correct_card_types += 1

    precision = (total_matched / total_pred * 100.0) if total_pred > 0 else 0.0
    recall = (total_matched / total_gt * 100.0) if total_gt > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_iou = (sum_iou / total_matched) if total_matched > 0 else 0.0

    print("\n" + "=" * 65)
    print("SHOP RECOGNITION SPATIAL EVALUATION SUMMARY")
    print("=" * 65)
    print(f"Frames Evaluated           : {tested:,}")
    print(f"Ground Truth Items         : {total_gt:,}")
    print(f"Detected Items             : {total_pred:,}")
    print(f"True Positives (IoU >= {iou_threshold:.2f}) : {total_matched:,}")
    print(f"Mean Matched IoU           : {mean_iou:.3f}")
    print(f"Detection Precision        : {precision:.2f}%")
    print(f"Detection Recall           : {recall:.2f}%")
    print(f"Detection F1 Score         : {f1:.2f}%")
    print("-" * 65)
    if total_matched > 0:
        print(f"Card Type Accuracy (Matched): {(correct_card_types / total_matched) * 100.0:.2f}%")
        print(f"Key Accuracy (Matched)     : {(correct_keys / total_matched) * 100.0:.2f}%")
        print(f"Edition Accuracy (Matched) : {(correct_editions / total_matched) * 100.0:.2f}%")
    if total_gt > 0:
        print(f"Overall Key Accuracy (All) : {(correct_keys / total_gt) * 100.0:.2f}%")
        print(f"Overall Edition Acc (All)  : {(correct_editions / total_gt) * 100.0:.2f}%")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Shop Pipeline")
    parser.add_argument("--samples", type=int, default=200, help="Max frames to evaluate")
    parser.add_argument("--iou", type=float, default=0.4, help="IoU threshold for box matching (default: 0.4)")
    args = parser.parse_args()
    evaluate(max_samples=args.samples, iou_threshold=args.iou)
