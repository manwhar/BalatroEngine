"""
YOLO Training Script for Balatro Card Feature Detection
======================================================
Trains YOLO on the prepared dataset with:
1. Video compression noise transformation (~50% of frames).
2. Synthetic cursor occlusion augmentation (~30% of frames) using
   the realistic Windows cursor (assets/classifiers/cursor.png).
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
# pyrefly: ignore [missing-import]
from ultralytics import YOLO

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
DEFAULT_DATA_YAML = CURRENT_DIR / "data.yaml"
CURSOR_PATH = PROJECT_ROOT / "assets" / "classifiers" / "cursor.png"


def load_cursor_asset(cursor_path=CURSOR_PATH):
    """Loads cursor RGBA image and converts RGB to float32 with alpha in [0, 1]."""
    if not Path(cursor_path).exists():
        return None
    cursor_img = cv2.imread(str(cursor_path), cv2.IMREAD_UNCHANGED)
    if cursor_img is None or cursor_img.shape[-1] < 4:
        return None
    bgr = cursor_img[..., :3]
    alpha = cursor_img[..., 3:4].astype(np.float32) / 255.0
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgba = np.concatenate([rgb, alpha], axis=-1)
    return rgba


def apply_video_compression_noise(img_tensor: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """
    Simulates realistic video compression artifacts on a batch of images.
    Applies to roughly `p` fraction of images in the batch.
    - Low-pass / bilinear down-sampling (loss of fine detail / bitrate drop)
    - Macroblock quantization (quantizes discrete color levels)
    - Slight high-frequency noise / dithering along reconstructed edges
    """
    if not isinstance(img_tensor, torch.Tensor) or img_tensor.ndim != 4:
        return img_tensor

    batch_size = img_tensor.shape[0]
    device = img_tensor.device

    mask = torch.rand(batch_size, device=device) < p
    if not mask.any():
        return img_tensor

    out = img_tensor.clone()
    selected = out[mask]
    orig_h, orig_w = selected.shape[-2:]

    scale = torch.empty(1, device=device).uniform_(0.35, 0.75).item()
    down_h = max(8, int(orig_h * scale))
    down_w = max(8, int(orig_w * scale))

    down = F.interpolate(selected, size=(down_h, down_w), mode="bilinear", align_corners=False)
    down = torch.round(down * 32.0) / 32.0
    up = F.interpolate(down, size=(orig_h, orig_w), mode="bilinear", align_corners=False)

    max_val = 255.0 if selected.max() > 1.5 else 1.0
    noise = (torch.rand_like(selected) - 0.5) * 0.04 * max_val

    selected_augmented = torch.clamp(up + noise, 0.0, max_val)
    out[mask] = selected_augmented
    return out


def overlay_cursor_noise(img_tensor: torch.Tensor, cursor_rgba: np.ndarray, p: float = 0.3) -> torch.Tensor:
    """
    Randomly overlays a Windows cursor onto ~`p` fraction of images in the batch
    to build robustness against mouse occlusions over cards and UI elements.
    """
    if cursor_rgba is None or not isinstance(img_tensor, torch.Tensor) or img_tensor.ndim != 4:
        return img_tensor

    batch_size, _, H, W = img_tensor.shape
    device = img_tensor.device

    mask = torch.rand(batch_size, device=device) < p
    if not mask.any():
        return img_tensor

    out = img_tensor.clone()
    c_h, c_w = cursor_rgba.shape[:2]

    for idx in range(batch_size):
        if not mask[idx]:
            continue

        # Scale cursor to realistic screen proportion (around 24px - 48px height)
        scale = torch.empty(1).uniform_(0.07, 0.14).item()
        target_h = max(16, min(H, int(c_h * scale)))
        target_w = max(12, min(W, int(c_w * scale)))

        max_y = max(1, H - target_h)
        max_x = max(1, W - target_w)
        y = torch.randint(0, max_y, (1,)).item()
        x = torch.randint(0, max_x, (1,)).item()

        resized = cv2.resize(cursor_rgba, (target_w, target_h), interpolation=cv2.INTER_AREA)
        c_rgb = torch.tensor(resized[..., :3], dtype=torch.float32, device=device).permute(2, 0, 1)
        c_alpha = torch.tensor(resized[..., 3:], dtype=torch.float32, device=device).permute(2, 0, 1)

        # Scale cursor RGB range if input image is normalized [0, 1]
        if out[idx].max() <= 1.5:
            c_rgb = c_rgb / 255.0

        bg_patch = out[idx, :, y:y+target_h, x:x+target_w]
        out[idx, :, y:y+target_h, x:x+target_w] = bg_patch * (1.0 - c_alpha) + c_rgb * c_alpha

    return out


def attach_batch_augmentations(model: YOLO, compression_prob: float = 0.5, cursor_prob: float = 0.3):
    """
    Hooks into the Ultralytics training pipeline via 'on_train_start' callback
    by wrapping trainer.preprocess_batch to apply video compression noise
    and synthetic cursor occlusion on training batches on-device.
    """
    cursor_rgba = load_cursor_asset()
    if cursor_rgba is not None:
        print(f"Loaded cursor asset from {CURSOR_PATH}")
    else:
        print(f"Notice: Cursor asset not found at {CURSOR_PATH}. Cursor augmentation skipped.")

    def on_train_start(trainer):
        orig_preprocess = trainer.preprocess_batch
        batch_counter = 0

        def custom_preprocess(batch):
            nonlocal batch_counter
            batch = orig_preprocess(batch)

            if batch_counter == 0 and "img" in batch:
                device_str = str(batch["img"].device)
                print(f"[Augmentations] Active: custom augmentations applying to batches on device '{device_str}'.")

            batch_counter += 1

            if "img" in batch:
                # 1. Apply compression noise
                if compression_prob > 0.0:
                    batch["img"] = apply_video_compression_noise(batch["img"], p=compression_prob)
                # 2. Apply cursor occlusion
                if cursor_prob > 0.0 and cursor_rgba is not None:
                    batch["img"] = overlay_cursor_noise(batch["img"], cursor_rgba, p=cursor_prob)

            return batch

        trainer.preprocess_batch = custom_preprocess
        print("[Augmentations] Successfully attached to trainer preprocessing pipeline.")

    model.add_callback("on_train_start", on_train_start)
    print(f"Configured batch augmentations: compression_p={compression_prob}, cursor_p={cursor_prob}")


def organize_run_figures(run_dir: Path):
    """Moves all generated figures (*.png, *.jpg) into a dedicated figures/ subfolder."""
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for f in run_dir.glob(ext):
            dest = fig_dir / f.name
            shutil.move(str(f), str(dest))
            count += 1
    if count > 0:
        print(f"[Run Output] Organized {count} figures into: {fig_dir}")


def resolve_model_weights(model_name: str) -> str:
    """Resolves model weights path, checking classifiers/card_classifier/weights first."""
    p = Path(model_name)
    if p.exists():
        return str(p)
    # Check in weights/ folder inside card_classifier
    cand_weights = CURRENT_DIR / "weights" / model_name
    if cand_weights.exists():
        return str(cand_weights)
    cand_curr = CURRENT_DIR / model_name
    if cand_curr.exists():
        return str(cand_curr)
    # Default to placing downloaded base weights inside classifiers/card_classifier/weights/
    weights_dir = CURRENT_DIR / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    return str(weights_dir / model_name)


def run_train(
    data_yaml=DEFAULT_DATA_YAML,
    model_name="yolov8n.pt",
    epochs=100,
    imgsz=640,
    batch=16,
    project=None,
    name="balatro_cards",
    exist_ok=False,
    compression_noise_prob=0.5,
    cursor_prob=0.3
):
    if not Path(data_yaml).exists():
        print(f"data.yaml not found at {data_yaml}. Running prepare_dataset.py first...")
        from prepare_dataset import prepare_dataset
        prepare_dataset()

    print("\n" + "=" * 60)
    print("Balatro Card Classifier Training Setup")
    print("=" * 60)

    # CUDA & Device verification
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[CUDA OK] GPU Device : {gpu_name}")
        print(f"[CUDA OK] Total VRAM : {vram_gb:.2f} GB")
        print(f"[CUDA OK] CUDA Ver   : {torch.version.cuda}")
        device = 0
    else:
        print("[WARNING] CUDA is NOT available! Training will run on CPU.")
        device = "cpu"

    if project is None:
        project = (CURRENT_DIR / "runs").as_posix()

    print(f"\nConfiguration:")
    print(f"  - Dataset config : {data_yaml}")
    print(f"  - Base model     : {model_name}")
    print(f"  - Target epochs  : {epochs}")
    print(f"  - Batch size     : {batch}")
    print(f"  - Image size     : {imgsz}x{imgsz}")
    print(f"  - Base run name  : {name} (auto-increments if exists: {not exist_ok})")
    print(f"  - Output root    : {project}")
    print(f"  - Device         : {device}")
    print("=" * 60 + "\n")

    resolved_model = resolve_model_weights(model_name)
    print(f"Loading base model: {resolved_model}")
    model = YOLO(resolved_model)

    # Attach smart label-stacking and high-resolution figure plotting patch
    try:
        from regenerate_figures import apply_smart_plotting_patch
        apply_smart_plotting_patch(font_size=11, line_width=1, max_size=2880)
    except Exception as e:
        print(f"Notice: Could not load plotting patch ({e}), continuing with default plotting.")

    if compression_noise_prob > 0.0 or cursor_prob > 0.0:
        attach_batch_augmentations(model, compression_prob=compression_noise_prob, cursor_prob=cursor_prob)

    print(f"\nStarting training loop...")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        workers=2,
        val=True,
        save=True,
        exist_ok=exist_ok,
        verbose=True
    )

    actual_run_dir = Path(results.save_dir) if hasattr(results, "save_dir") else (Path(project) / name)
    organize_run_figures(actual_run_dir)

    print("\nTraining complete!")
    best_pt = actual_run_dir / "weights" / "best.pt"
    if best_pt.exists():
        print(f"Best model weights saved to : {best_pt}")
    print(f"Output figures saved to     : {actual_run_dir / 'figures'}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO on Balatro Card Hand Dataset")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model weights")
    parser.add_argument("--name", type=str, default="balatro_cards", help="Run name (auto-increments if exists)")
    parser.add_argument("--exist-ok", action="store_true", help="Overwrite existing run directory if it shares the same name")
    parser.add_argument("--compression-prob", type=float, default=0.5, help="Probability of applying compression noise per frame")
    parser.add_argument("--cursor-prob", type=float, default=0.3, help="Probability of applying cursor overlay per frame")
    args = parser.parse_args()

    run_train(
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        exist_ok=args.exist_ok,
        compression_noise_prob=args.compression_prob,
        cursor_prob=args.cursor_prob
    )
