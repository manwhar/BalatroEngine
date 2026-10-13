"""
Balatro Frame Recognition Interface for Post-Processing Pipeline
================================================================
Exposes lightweight, high-performance recognition functions intended for direct
import and execution in `post_processing.py` (or live bot loops).

Uses the production models stored in `classifiers/prod_weights/`:
  - card_detector.pt + card_classifier.pt
  - shop_detector.pt + shop_classifier.pt

Supports:
  1. Single-frame immediate inference (`recognize_hand_cards`, `recognize_shop`)
  2. High-throughput grouped batching (`batch_recognize_hands`, `batch_recognize_shops`)
     Group-batches frames by identical ROI shelf dimensions so the GPU processes
     hundreds of crops in uniform tensor batches at full saturation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence, Union
import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROD_WEIGHTS_DIR = PROJECT_ROOT / "classifiers" / "prod_weights"

# Explicit relative imports from pipelines
# pyrefly: ignore [missing-import]
from classifiers.card_classifier.pipeline import CardRecognitionPipeline, CardResult
# pyrefly: ignore [missing-import]
from classifiers.shop_classifier.pipeline import ShopRecognitionPipeline, ShopItemResult
# pyrefly: ignore [missing-import]
from classifiers.common.video_utils import scale_roi_pixels

BASE_WIDTH = 1920.0
BASE_HEIGHT = 1080.0

# 1920x1080 Balatro UI ROI coordinates
ROIS = {
    "card_hand": {"x": 508.319934, "y": 567.839093, "w": 1071.056521, "h": 322.734447},
    "joker_bar": {"x": 320.0, "y": 10.0, "w": 1280.0, "h": 260.0},
    "consumeables": {"x": 1250.0, "y": 10.0, "w": 420.0, "h": 260.0},
    "shop": {"x": 360.0, "y": 140.0, "w": 1200.0, "h": 380.0},
    "voucher": {"x": 880.0, "y": 355.0, "w": 340.0, "h": 300.0},
    "booster": {"x": 1540.0, "y": 355.0, "w": 360.0, "h": 300.0},
    "pack": {"x": 300.0, "y": 260.0, "w": 1320.0, "h": 500.0},
}


# =====================================================================
# Output Data Structures
# =====================================================================

@dataclass(slots=True)
class HandRecognitionOutput:
    """Output results for player hand detection."""
    frame_index: int
    timestamp_seconds: float
    cards: list[CardResult] = field(default_factory=list)
    roi: tuple[int, int, int, int] = (0, 0, 0, 0)  # (x, y, w, h)


@dataclass(slots=True)
class ShopRecognitionOutput:
    """Complete shop frame evaluation results across all shelves."""
    frame_index: int
    timestamp_seconds: float
    mode: str = "shop"  # 'shop' or 'pack'
    joker_bar: list[ShopItemResult] = field(default_factory=list)
    consumeables: list[ShopItemResult] = field(default_factory=list)
    shop_items: list[ShopItemResult] = field(default_factory=list)
    vouchers: list[ShopItemResult] = field(default_factory=list)
    boosters: list[ShopItemResult] = field(default_factory=list)
    pack_items: list[ShopItemResult] = field(default_factory=list)

    @property
    def all_items(self) -> list[ShopItemResult]:
        """Flattened list of all detected shop items across all regions."""
        return (
            self.joker_bar +
            self.consumeables +
            self.shop_items +
            self.vouchers +
            self.boosters +
            self.pack_items
        )


# =====================================================================
# Model Cache & Lazy Initialization
# =====================================================================

_CARD_PIPELINE: CardRecognitionPipeline | None = None
_SHOP_PIPELINE: ShopRecognitionPipeline | None = None


def get_card_pipeline(device: str | None = None) -> CardRecognitionPipeline:
    """Returns singleton CardRecognitionPipeline loaded from classifiers/prod_weights/."""
    global _CARD_PIPELINE
    if _CARD_PIPELINE is None:
        det_path = PROD_WEIGHTS_DIR / "card_detector.pt"
        cls_path = PROD_WEIGHTS_DIR / "card_classifier.pt"
        if not det_path.exists():
            raise FileNotFoundError(f"Missing production card detector weights at {det_path}")
        if not cls_path.exists():
            raise FileNotFoundError(f"Missing production card classifier weights at {cls_path}")

        _CARD_PIPELINE = CardRecognitionPipeline(
            detector_path=det_path,
            classifier_path=cls_path,
            device=device
        )
    return _CARD_PIPELINE


def get_shop_pipeline(device: str | None = None) -> ShopRecognitionPipeline:
    """Returns singleton ShopRecognitionPipeline loaded from classifiers/prod_weights/."""
    global _SHOP_PIPELINE
    if _SHOP_PIPELINE is None:
        det_path = PROD_WEIGHTS_DIR / "shop_detector.pt"
        cls_path = PROD_WEIGHTS_DIR / "shop_classifier.pt"
        if not det_path.exists():
            raise FileNotFoundError(f"Missing production shop detector weights at {det_path}")
        if not cls_path.exists():
            raise FileNotFoundError(f"Missing production shop classifier weights at {cls_path}")

        _SHOP_PIPELINE = ShopRecognitionPipeline(
            detector_path=det_path,
            classifier_path=cls_path,
            device=device
        )
    return _SHOP_PIPELINE


def _extract_frame_image_and_meta(video_frame: Any) -> tuple[np.ndarray, int, float]:
    """Helper to unwrap VideoFrame, tuple, or raw np.ndarray."""
    if hasattr(video_frame, "image") and hasattr(video_frame, "index"):
        return video_frame.image, getattr(video_frame, "index", 0), getattr(video_frame, "timestamp_seconds", 0.0)
    elif isinstance(video_frame, tuple) and len(video_frame) >= 2:
        return video_frame[0], video_frame[1], (video_frame[2] if len(video_frame) > 2 else 0.0)
    elif isinstance(video_frame, np.ndarray):
        return video_frame, 0, 0.0
    else:
        raise TypeError(f"Unsupported frame type: {type(video_frame)}. Expected VideoFrame or np.ndarray.")


def _crop_roi_to_pil(full_bgr: np.ndarray, roi_key: str) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Crops a region from full frame BGR array to PIL Image in RGB format."""
    fh, fw = full_bgr.shape[:2]
    rx, ry, rw, rh = scale_roi_pixels(ROIS[roi_key], fw, fh, BASE_WIDTH, BASE_HEIGHT)
    crop_bgr = full_bgr[ry:ry + rh, rx:rx + rw]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop_rgb), (rx, ry, rw, rh)


# =====================================================================
# Single Frame Recognition Functions
# =====================================================================

def recognize_hand_cards(
    video_frame: Any,
    conf_threshold: float = 0.50,
    device: str | None = None
) -> HandRecognitionOutput:
    """Evaluates the player hand on a single video frame."""
    img_bgr, f_idx, ts = _extract_frame_image_and_meta(video_frame)
    pipeline = get_card_pipeline(device=device)

    crop_pil, roi_coords = _crop_roi_to_pil(img_bgr, "card_hand")
    cards = pipeline.predict(crop_pil, conf=conf_threshold)

    return HandRecognitionOutput(
        frame_index=f_idx,
        timestamp_seconds=ts,
        cards=cards,
        roi=roi_coords
    )


def recognize_shop(
    video_frame: Any,
    mode: str = "shop",
    conf_threshold: float = 0.35,
    device: str | None = None
) -> ShopRecognitionOutput:
    """Evaluates all relevant shelves on a single shop or booster pack video frame."""
    img_bgr, f_idx, ts = _extract_frame_image_and_meta(video_frame)
    pipeline = get_shop_pipeline(device=device)

    output = ShopRecognitionOutput(
        frame_index=f_idx,
        timestamp_seconds=ts,
        mode=mode
    )

    joker_pil, _ = _crop_roi_to_pil(img_bgr, "joker_bar")
    output.joker_bar = pipeline.predict(joker_pil, conf_threshold=conf_threshold)

    cons_pil, _ = _crop_roi_to_pil(img_bgr, "consumeables")
    output.consumeables = pipeline.predict(cons_pil, conf_threshold=conf_threshold)

    if mode == "pack":
        pack_pil, _ = _crop_roi_to_pil(img_bgr, "pack")
        output.pack_items = pipeline.predict(pack_pil, conf_threshold=conf_threshold)
    else:
        shop_pil, _ = _crop_roi_to_pil(img_bgr, "shop")
        output.shop_items = pipeline.predict(shop_pil, conf_threshold=conf_threshold)

        voucher_pil, _ = _crop_roi_to_pil(img_bgr, "voucher")
        output.vouchers = pipeline.predict(voucher_pil, conf_threshold=conf_threshold)

        booster_pil, _ = _crop_roi_to_pil(img_bgr, "booster")
        output.boosters = pipeline.predict(booster_pil, conf_threshold=conf_threshold)

    return output


# =====================================================================
# High-Throughput Grouped Batch Inference Functions
# =====================================================================

def batch_recognize_hands(
    frames: Sequence[Any],
    conf_threshold: float = 0.50,
    batch_size: int = 32,
    device: str | None = None
) -> list[HandRecognitionOutput]:
    """
    Batches player hand card detection across a collected list of frames.
    All hand crops share identical dimensions (1071x323 px), so they are batched
    directly into YOLO Stage 1 and Stage 2 CNN in parallel.
    """
    if not frames:
        return []

    pipeline = get_card_pipeline(device=device)
    unwrapped = [_extract_frame_image_and_meta(f) for f in frames]

    outputs: list[HandRecognitionOutput] = []

    for chunk_start in range(0, len(unwrapped), batch_size):
        chunk = unwrapped[chunk_start:chunk_start + batch_size]
        pil_crops = []
        metas = []

        for img_bgr, f_idx, ts in chunk:
            crop_pil, roi_coords = _crop_roi_to_pil(img_bgr, "card_hand")
            pil_crops.append(crop_pil)
            metas.append((f_idx, ts, roi_coords))

        # Batch Stage 1 YOLO detection
        det_results = pipeline.detector.predict(
            source=pil_crops,
            conf=conf_threshold,
            device=pipeline.device.type if pipeline.device.type == "cpu" else 0,
            verbose=False
        )

        # Collect card crops for global Stage 2 CNN batching
        chunk_cards_per_frame: list[list[CardResult]] = []

        all_crops_tensors = []
        crop_metadata = []  # (frame_idx_in_chunk, box, ih)

        for f_i, det in enumerate(det_results):
            boxes_obj = det.boxes
            if boxes_obj is None or len(boxes_obj) == 0:
                continue

            xyxy = boxes_obj.xyxy.cpu().numpy()
            order = np.argsort(xyxy[:, 0])
            sorted_boxes = xyxy[order]

            cur_pil = pil_crops[f_i]
            iw, ih = cur_pil.size

            for b in sorted_boxes:
                x1 = max(0, min(iw - 1, int(b[0])))
                y1 = max(0, min(ih - 1, int(b[1])))
                x2 = max(x1 + 1, min(iw, int(b[2])))
                y2 = max(y1 + 1, min(ih, int(b[3])))

                crop = cur_pil.crop((x1, y1, x2, y2))
                all_crops_tensors.append(pipeline.crop_transform(crop))
                crop_metadata.append((f_i, [x1, y1, x2, y2], ih))

        # Stage 2 Batched CNN forward pass across all cards in this chunk
        all_card_preds = []
        if all_crops_tensors:
            batched_cards = torch.stack(all_crops_tensors).to(pipeline.device)
            with torch.no_grad():
                all_card_preds = pipeline.classifier.predict_attributes(batched_cards)

        # Assemble per-frame outputs
        frame_cards_map: dict[int, list[CardResult]] = {i: [] for i in range(len(chunk))}
        for (f_i, box, ih), attrs in zip(crop_metadata, all_card_preds):
            card_res = CardResult(
                index=len(frame_cards_map[f_i]),
                rank=attrs["rank"],
                suit=attrs["suit"],
                edition=attrs["edition"],
                enhancement=attrs["enhancement"],
                seal=attrs["seal"],
                selected=False,  # Updated below
                debuffed=attrs["debuffed"],
                face_down=attrs["face_down"],
                box=box,
                confidences={
                    "rank": attrs["rank_conf"],
                    "suit": attrs["suit_conf"],
                    "edition": attrs["edition_conf"],
                    "enhancement": attrs["enhancement_conf"],
                    "seal": attrs["seal_conf"],
                    "debuffed": attrs["debuffed_conf"],
                    "face_down": attrs["face_down_conf"],
                }
            )
            frame_cards_map[f_i].append(card_res)

        # Calculate card raise/selection per frame
        for f_i, cards_list in frame_cards_map.items():
            if cards_list:
                _, _, roi_coords = metas[f_i]
                ih = roi_coords[3]
                boxes_only = [c.box for c in cards_list]
                from classifiers.card_classifier.pipeline import determine_card_selection
                selected_flags = determine_card_selection(boxes_only, ih)
                for c_obj, is_sel in zip(cards_list, selected_flags):
                    c_obj.selected = is_sel

            f_idx, ts, roi_coords = metas[f_i]
            outputs.append(HandRecognitionOutput(
                frame_index=f_idx,
                timestamp_seconds=ts,
                cards=cards_list,
                roi=roi_coords
            ))

    return outputs


def batch_recognize_shops(
    frames: Sequence[Any],
    mode: str = "shop",
    conf_threshold: float = 0.35,
    batch_size: int = 32,
    device: str | None = None
) -> list[ShopRecognitionOutput]:
    """
    Group-batches shop detection across collected frames.
    Groups crops by shelf region (joker_bar, shop, voucher, booster, etc.) so that
    each batch consists of uniformly-sized images sent to the GPU together.
    """
    if not frames:
        return []

    pipeline = get_shop_pipeline(device=device)
    unwrapped = [_extract_frame_image_and_meta(f) for f in frames]

    outputs = [
        ShopRecognitionOutput(frame_index=f_idx, timestamp_seconds=ts, mode=mode)
        for _, f_idx, ts in unwrapped
    ]

    shelves = ["joker_bar", "consumeables"]
    if mode == "pack":
        shelves.append("pack")
    else:
        shelves.extend(["shop", "voucher", "booster"])

    for shelf_key in shelves:
        # 1. Extract uniformly shaped crops for this specific shelf across all frames
        shelf_crops = [_crop_roi_to_pil(img_bgr, shelf_key)[0] for img_bgr, _, _ in unwrapped]

        # 2. Batch process in chunks of `batch_size`
        for chunk_start in range(0, len(shelf_crops), batch_size):
            chunk_crops = shelf_crops[chunk_start:chunk_start + batch_size]
            chunk_indices = list(range(chunk_start, chunk_start + len(chunk_crops)))

            det_results = pipeline.detector.predict(
                source=chunk_crops,
                conf=conf_threshold,
                device=pipeline.device.type if pipeline.device.type == "cpu" else 0,
                verbose=False
            )

            # Collect Stage 2 crops across the chunk
            all_stage2_tensors = []
            item_metadata = []  # (frame_idx_in_all, box, card_type, det_conf)

            for local_idx, det in enumerate(det_results):
                global_idx = chunk_indices[local_idx]
                cur_pil = chunk_crops[local_idx]
                iw, ih = cur_pil.size

                boxes_obj = det.boxes
                if boxes_obj is None or len(boxes_obj) == 0:
                    continue

                xyxy = boxes_obj.xyxy.cpu().numpy()
                confs = boxes_obj.conf.cpu().numpy()
                classes = boxes_obj.cls.cpu().numpy()

                for b, c_id, conf in zip(xyxy, classes, confs):
                    x1 = max(0, min(iw - 1, int(round(b[0]))))
                    y1 = max(0, min(ih - 1, int(round(b[1]))))
                    x2 = max(x1 + 1, min(iw, int(round(b[2]))))
                    y2 = max(y1 + 1, min(ih, int(round(b[3]))))

                    crop = cur_pil.crop((x1, y1, x2, y2))
                    all_stage2_tensors.append(pipeline.transform(crop))
                    item_metadata.append((
                        global_idx,
                        [x1, y1, x2, y2],
                        pipeline.class_names.get(int(c_id), "Joker"),
                        float(conf)
                    ))

            # Batch Stage 2 MobileNet inference
            attr_preds = []
            if all_stage2_tensors:
                batched_tensor = torch.stack(all_stage2_tensors).to(pipeline.device)
                with torch.no_grad():
                    attr_preds = pipeline.classifier.predict_attributes(batched_tensor)

            # Distribute items back into their respective frame output
            for (g_idx, box, card_type, det_conf), attrs in zip(item_metadata, attr_preds):
                item_res = ShopItemResult(
                    index=0,  # Ordered below
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
                )

                out_obj = outputs[g_idx]
                if shelf_key == "joker_bar":
                    out_obj.joker_bar.append(item_res)
                elif shelf_key == "consumeables":
                    out_obj.consumeables.append(item_res)
                elif shelf_key == "shop":
                    out_obj.shop_items.append(item_res)
                elif shelf_key == "voucher":
                    out_obj.vouchers.append(item_res)
                elif shelf_key == "booster":
                    out_obj.boosters.append(item_res)
                elif shelf_key == "pack":
                    out_obj.pack_items.append(item_res)

        # Sort items in each shelf left-to-right
        for out_obj in outputs:
            target_list = getattr(out_obj, "pack_items" if shelf_key == "pack" else ("shop_items" if shelf_key == "shop" else ("vouchers" if shelf_key == "voucher" else ("boosters" if shelf_key == "booster" else shelf_key))))
            target_list.sort(key=lambda item: item.box[0])
            for i, it in enumerate(target_list):
                it.index = i

    return outputs


def process_frames_iterator(
    frames_iterator: Iterable[Any],
    target: str = "hand",
    mode: str = "shop",
    conf_threshold: float | None = None
) -> Iterator[Union[HandRecognitionOutput, ShopRecognitionOutput]]:
    """Yields frame-by-frame recognition outputs from an iterator."""
    if target == "hand":
        thresh = conf_threshold if conf_threshold is not None else 0.50
        for f in frames_iterator:
            yield recognize_hand_cards(f, conf_threshold=thresh)
    elif target == "shop":
        thresh = conf_threshold if conf_threshold is not None else 0.35
        for f in frames_iterator:
            yield recognize_shop(f, mode=mode, conf_threshold=thresh)
    else:
        raise ValueError(f"Unknown target '{target}'. Must be 'hand' or 'shop'.")
