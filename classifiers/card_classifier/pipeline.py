"""
Unified Two-Stage Card Recognition Pipeline
===========================================
End-to-end inference orchestrator:
1. Stage 1 (YOLO Detector): Localizes all cards in the hand screenshot.
2. Stage 2 (Multi-Head Classifier): Batched attribute classification on card crops.
Returns sorted left-to-right list of CardResult objects.
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from ultralytics import YOLO

CURRENT_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = CURRENT_DIR / "weights"

from stage2_classifier.config import (
    IDX_TO_RANK, IDX_TO_SUIT, IDX_TO_EDITION, IDX_TO_ENHANCEMENT, IDX_TO_SEAL,
    CROP_WIDTH, CROP_HEIGHT
)
from stage2_classifier.dataset import IMAGENET_MEAN, IMAGENET_STD
from stage2_classifier.model import MultiHeadCardClassifier


@dataclass
class CardResult:
    index: int
    rank: str
    suit: str
    edition: str
    enhancement: str
    seal: str
    selected: bool
    box: list[int]  # [x1, y1, x2, y2]
    confidences: dict[str, float]

    @property
    def display_name(self) -> str:
        parts = []
        if self.edition != "None":
            parts.append(self.edition)
        if self.enhancement != "None":
            parts.append(self.enhancement)
        if self.rank == "Stone":
            parts.append("Stone Card")
        else:
            parts.append(f"{self.rank} of {self.suit}")
        if self.seal != "None":
            parts.append(f"({self.seal} Seal)")
        if self.selected:
            parts.append("[Selected]")
        return " ".join(parts)


class CardRecognitionPipeline:
    def __init__(self, detector_path=None, classifier_path=None, device=None):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 1. Resolve Stage 1 Detector Weights
        if detector_path is None:
            detector_path = WEIGHTS_DIR / "detector_best.pt"
            if not detector_path.exists():
                candidates = list(CURRENT_DIR.glob("stage1_detector/runs/*/weights/best.pt"))
                if candidates:
                    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    detector_path = candidates[0]

        # 2. Resolve Stage 2 Classifier Weights
        if classifier_path is None:
            classifier_path = WEIGHTS_DIR / "classifier_best.pt"
            if not classifier_path.exists():
                candidates = list(CURRENT_DIR.glob("stage2_classifier/runs/*/weights/best.pt"))
                if candidates:
                    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    classifier_path = candidates[0]

        print(f"[Card Pipeline] Loading Detector   : {detector_path}")
        print(f"[Card Pipeline] Loading Classifier : {classifier_path}")

        self.detector = YOLO(str(detector_path))
        
        self.classifier = MultiHeadCardClassifier(backbone_name="mobilenet_v3_small", pretrained=False).to(self.device)
        ckpt = torch.load(str(classifier_path), map_location=self.device)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        self.classifier.load_state_dict(state_dict)
        self.classifier.eval()

        self.crop_transform = T.Compose([
            T.Resize((CROP_HEIGHT, CROP_WIDTH), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def predict(self, image_input, conf=0.25) -> list[CardResult]:
        """
        Runs two-stage detection and classification.
        Args:
            image_input: PIL.Image, numpy ndarray, or file path.
            conf: Confidence threshold for Stage 1 card detector.
        Returns:
            list[CardResult] sorted left to right by x position.
        """
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input).convert("RGB")
        else:
            pil_img = image_input.convert("RGB")

        # 1. Stage 1: Card Detection
        det_results = self.detector(pil_img, conf=conf, verbose=False)
        boxes_data = det_results[0].boxes

        if len(boxes_data) == 0:
            return []

        # Sort detected card bounding boxes left-to-right
        xyxy_boxes = boxes_data.xyxy.cpu().numpy()
        order = np.argsort(xyxy_boxes[:, 0])
        sorted_boxes = xyxy_boxes[order]

        # 2. Extract and batch card crops
        crop_tensors = []
        iw, ih = pil_img.size
        valid_boxes = []

        for b in sorted_boxes:
            x1 = max(0, min(iw - 1, int(b[0])))
            y1 = max(0, min(ih - 1, int(b[1])))
            x2 = max(x1 + 1, min(iw, int(b[2])))
            y2 = max(y1 + 1, min(ih, int(b[3])))

            crop = pil_img.crop((x1, y1, x2, y2))
            crop_tensor = self.crop_transform(crop)
            crop_tensors.append(crop_tensor)
            valid_boxes.append([x1, y1, x2, y2])

        batched_crops = torch.stack(crop_tensors, dim=0).to(self.device)

        # 3. Stage 2: Batched Attribute Classification
        with torch.no_grad():
            preds = self.classifier.predict_attributes(batched_crops)

        # 4. Assemble CardResult objects
        results = []
        for i in range(len(valid_boxes)):
            rank = IDX_TO_RANK.get(preds["rank_idx"][i].item(), "Unknown")
            suit = IDX_TO_SUIT.get(preds["suit_idx"][i].item(), "Unknown")
            edition = IDX_TO_EDITION.get(preds["edition_idx"][i].item(), "None")
            enhancement = IDX_TO_ENHANCEMENT.get(preds["enhancement_idx"][i].item(), "None")
            seal = IDX_TO_SEAL.get(preds["seal_idx"][i].item(), "None")
            selected = bool(preds["selected"][i].item())

            confs = {
                "rank": float(preds["rank_conf"][i].item()),
                "suit": float(preds["suit_conf"][i].item()),
                "edition": float(preds["edition_conf"][i].item()),
                "enhancement": float(preds["enhancement_conf"][i].item()),
                "seal": float(preds["seal_conf"][i].item()),
                "selected": float(preds["selected_conf"][i].item()),
            }

            results.append(CardResult(
                index=i + 1,
                rank=rank,
                suit=suit,
                edition=edition,
                enhancement=enhancement,
                seal=seal,
                selected=selected,
                box=valid_boxes[i],
                confidences=confs
            ))

        return results
