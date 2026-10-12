"""
Unified Two-Stage Shop Item Recognition Pipeline
================================================
End-to-end inference orchestrator for Balatro shop screens:
1. Stage 1 (YOLO Detector): Localizes all cards/items in any shop ROI crop (joker bar, consumables, shop, pack).
2. Stage 2 (Multi-Head Classifier): Batched attribute classification on item crops.
Returns sorted left-to-right list of ShopItemResult objects.
"""

from dataclasses import dataclass
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as T
from ultralytics import YOLO

CURRENT_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = CURRENT_DIR / "weights"

from stage2_classifier.config import (
    IDX_TO_KEY, IDX_TO_EDITION,
    CROP_WIDTH, CROP_HEIGHT
)
from stage2_classifier.dataset import IMAGENET_MEAN, IMAGENET_STD
from stage2_classifier.model import MultiHeadItemClassifier


@dataclass
class ShopItemResult:
    index: int
    card_type: str  # 'Joker', 'Tarot', 'Spectral', 'Planet'
    key: str        # 'j_blueprint', 'c_fool', etc.
    edition: str    # 'None', 'Foil', 'Holo', 'Polychrome', 'Negative'
    eternal: bool
    perishable: bool
    rental: bool
    debuffed: bool
    box: list[int]  # [x1, y1, x2, y2]
    confidences: dict[str, float]

    @property
    def display_name(self) -> str:
        parts = []
        if self.debuffed:
            parts.append("[DEBUFFED]")
        if self.edition != "None":
            parts.append(self.edition)
        if self.eternal:
            parts.append("[Eternal]")
        if self.perishable:
            parts.append("[Perishable]")
        if self.rental:
            parts.append("[Rental]")

        # Clean key to readable title (e.g. j_half_joker -> Half Joker, v_hone -> Hone, p_arcana_mega_1 -> Arcana Mega 1)
        clean_key = self.key
        if clean_key.startswith("j_") or clean_key.startswith("c_") or clean_key.startswith("v_") or clean_key.startswith("p_"):
            clean_key = clean_key[2:]
        readable_title = clean_key.replace("_", " ").title()
        parts.append(readable_title)

        return " ".join(parts)


class ShopRecognitionPipeline:
    def __init__(self, detector_path=None, classifier_path=None, backbone="mobilenet_v3_small", device=None):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        detector_path = Path(detector_path) if detector_path else WEIGHTS_DIR / "detector_best.pt"
        classifier_path = Path(classifier_path) if classifier_path else WEIGHTS_DIR / "classifier_best.pt"

        print(f"[Pipeline] Loading Stage 1 detector: {detector_path}")
        self.detector = YOLO(str(detector_path))

        print(f"[Pipeline] Loading Stage 2 classifier: {classifier_path} ({backbone})")
        self.classifier = MultiHeadItemClassifier(backbone_name=backbone, pretrained=False).to(self.device)

        if classifier_path.exists():
            checkpoint = torch.load(classifier_path, map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            self.classifier.load_state_dict(state_dict)
            self.classifier.eval()
            print("[Pipeline] Weights loaded successfully.")
        else:
            print(f"[Pipeline Warning] Classifier weights not found at: {classifier_path}")

        self.transform = T.Compose([
            T.Resize((CROP_HEIGHT, CROP_WIDTH), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

        self.class_names = {
            0: "Joker",
            1: "Tarot",
            2: "Spectral",
            3: "Planet",
            4: "Voucher",
            5: "Booster"
        }

    def predict(self, image_input, conf_threshold=0.35, iou_threshold=0.45) -> list[ShopItemResult]:
        """
        Runs full two-stage inference on an ROI crop or screenshot.
        Args:
            image_input: PIL Image or path to image
        Returns:
            list[ShopItemResult] sorted left to right
        """
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise TypeError("Expected PIL Image or file path.")

        # 1. Stage 1 YOLO Detection
        det_results = self.detector.predict(
            source=pil_img,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device.type if self.device.type == "cpu" else 0,
            verbose=False
        )

        boxes_with_info = []
        if det_results and len(det_results) > 0:
            boxes_obj = det_results[0].boxes
            if boxes_obj is not None and len(boxes_obj) > 0:
                xyxy = boxes_obj.xyxy.cpu().numpy()
                confs = boxes_obj.conf.cpu().numpy()
                classes = boxes_obj.cls.cpu().numpy()

                for i in range(len(xyxy)):
                    box = [int(round(coord)) for coord in xyxy[i]]
                    cls_id = int(classes[i])
                    card_type = self.class_names.get(cls_id, "Joker")
                    det_conf = float(confs[i])
                    boxes_with_info.append((box, card_type, det_conf))

        if not boxes_with_info:
            return []

        # Sort items strictly left-to-right by horizontal coordinate x1
        boxes_with_info.sort(key=lambda item: item[0][0])

        # 2. Stage 2 Multi-Head Classification
        crop_tensors = []
        iw, ih = pil_img.size
        for box, _, _ in boxes_with_info:
            x1, y1, x2, y2 = box
            x1 = max(0, min(iw - 1, x1))
            y1 = max(0, min(ih - 1, y1))
            x2 = max(x1 + 1, min(iw, x2))
            y2 = max(y1 + 1, min(ih, y2))

            crop = pil_img.crop((x1, y1, x2, y2))
            crop_tensors.append(self.transform(crop))

        batch_tensor = torch.stack(crop_tensors).to(self.device)
        attr_predictions = self.classifier.predict_attributes(batch_tensor)

        # 3. Assemble Output Results
        results = []
        for idx, ((box, card_type, det_conf), attrs) in enumerate(zip(boxes_with_info, attr_predictions)):
            results.append(ShopItemResult(
                index=idx,
                card_type=card_type,
                key=attrs["key"],
                edition=attrs["edition"],
                eternal=attrs["eternal"],
                perishable=attrs["perishable"],
                rental=attrs["rental"],
                debuffed=attrs["debuffed"],
                box=box,
                confidences={
                    "detection": det_conf,
                    "key": attrs["key_conf"],
                    "edition": attrs["edition_conf"],
                    "eternal": attrs["eternal_conf"],
                    "perishable": attrs["perishable_conf"],
                    "rental": attrs["rental_conf"],
                    "debuffed": attrs["debuffed_conf"],
                }
            ))

        return results
