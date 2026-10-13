"""
Shared Augmentations for Balatro Detection & Classification
============================================================
Provides:
1. Video compression artifact simulation (downsampling + macroblock quantization + noise)
2. Synthetic Windows cursor occlusion (assets/classifiers/cursor.png)
3. Ultralytics YOLO batch augmentation hook (on_train_batch_start)
4. PyTorch / PIL transforms for Stage 2 crop classifiers
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
CURSOR_PATH = PROJECT_ROOT / "assets" / "classifiers" / "cursor.png"

_CACHED_CURSOR_RGBA = None


def load_cursor_asset(cursor_path=CURSOR_PATH):
    """Loads cursor RGBA image and converts RGB to float32 with alpha in [0, 1]."""
    global _CACHED_CURSOR_RGBA
    if _CACHED_CURSOR_RGBA is not None:
        return _CACHED_CURSOR_RGBA

    p = Path(cursor_path)
    if not p.exists():
        return None

    cursor_img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if cursor_img is None or cursor_img.shape[-1] < 4:
        return None

    bgr = cursor_img[..., :3]
    alpha = cursor_img[..., 3:4].astype(np.float32) / 255.0
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgba = np.concatenate([rgb, alpha], axis=-1)
    _CACHED_CURSOR_RGBA = rgba
    return rgba


def apply_video_compression_noise(img_tensor: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """
    Simulates realistic video compression artifacts on a batch (4D) or single (3D) tensor.
    - Low-pass / bilinear down-sampling (loss of high frequency bitrate)
    - Macroblock quantization (quantizes discrete color levels)
    - High-frequency dithering / sensor noise along reconstructed edges
    """
    if not isinstance(img_tensor, torch.Tensor):
        return img_tensor

    is_single = False
    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)
        is_single = True
    elif img_tensor.ndim != 4:
        return img_tensor

    batch_size = img_tensor.shape[0]
    device = img_tensor.device

    mask = torch.rand(batch_size, device=device) < p
    if not mask.any():
        return img_tensor.squeeze(0) if is_single else img_tensor

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

    return out.squeeze(0) if is_single else out


def overlay_cursor_noise(img_tensor: torch.Tensor, cursor_rgba: np.ndarray = None, p: float = 0.3) -> torch.Tensor:
    """
    Randomly overlays a Windows cursor onto ~`p` fraction of images in the batch or single tensor
    to build robustness against mouse occlusions over cards and UI elements.
    """
    if cursor_rgba is None:
        cursor_rgba = load_cursor_asset()
    if cursor_rgba is None or not isinstance(img_tensor, torch.Tensor):
        return img_tensor

    is_single = False
    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)
        is_single = True
    elif img_tensor.ndim != 4:
        return img_tensor

    batch_size, _, H, W = img_tensor.shape
    device = img_tensor.device

    mask = torch.rand(batch_size, device=device) < p
    if not mask.any():
        return img_tensor.squeeze(0) if is_single else img_tensor

    out = img_tensor.clone()
    c_h, c_w = cursor_rgba.shape[:2]

    for idx in range(batch_size):
        if not mask[idx]:
            continue

        # Scale cursor to realistic screen proportion (around 20px - 44px)
        scale = torch.empty(1).uniform_(0.08, 0.22).item()
        target_h = max(14, min(H, int(c_h * scale)))
        target_w = max(10, min(W, int(c_w * scale)))

        max_y = max(1, H - target_h)
        max_x = max(1, W - target_w)
        y = torch.randint(0, max_y, (1,)).item()
        x = torch.randint(0, max_x, (1,)).item()

        resized = cv2.resize(cursor_rgba, (target_w, target_h), interpolation=cv2.INTER_AREA)
        c_rgb = torch.tensor(resized[..., :3], dtype=torch.float32, device=device).permute(2, 0, 1)
        c_alpha = torch.tensor(resized[..., 3:], dtype=torch.float32, device=device).permute(2, 0, 1)

        if out[idx].max() <= 1.5:
            c_rgb = c_rgb / 255.0

        bg_patch = out[idx, :, y:y+target_h, x:x+target_w]
        out[idx, :, y:y+target_h, x:x+target_w] = bg_patch * (1.0 - c_alpha) + c_rgb * c_alpha

    return out.squeeze(0) if is_single else out


def attach_batch_augmentations(model, compression_prob: float = 0.5, cursor_prob: float = 0.3):
    """
    Hooks into the Ultralytics training loop via 'on_train_batch_start' callback.
    """
    cursor_rgba = load_cursor_asset()
    if cursor_rgba is not None:
        print(f"[Augmentations] Loaded cursor asset from {CURSOR_PATH}")
    else:
        print(f"[Augmentations] Notice: Cursor asset not found at {CURSOR_PATH}. Cursor augmentation skipped.")

    def on_train_batch_start(trainer):
        if hasattr(trainer, "batch") and isinstance(trainer.batch, dict) and "img" in trainer.batch:
            if compression_prob > 0.0:
                trainer.batch["img"] = apply_video_compression_noise(trainer.batch["img"], p=compression_prob)
            if cursor_prob > 0.0 and cursor_rgba is not None:
                trainer.batch["img"] = overlay_cursor_noise(trainer.batch["img"], cursor_rgba, p=cursor_prob)

    model.add_callback("on_train_batch_start", on_train_batch_start)
    print(f"[Augmentations] Attached YOLO batch hooks: compression_p={compression_prob}, cursor_p={cursor_prob}")


class BalatroStreamAugmentation:
    """
    PyTorch transform applied after ToTensor() (values in [0, 1])
    to inject realistic video compression artifacts and mouse cursor occlusions.
    """
    def __init__(self, compression_prob: float = 0.5, cursor_prob: float = 0.3):
        self.compression_prob = compression_prob
        self.cursor_prob = cursor_prob
        self.cursor_rgba = load_cursor_asset()

    def __call__(self, img_tensor: torch.Tensor) -> torch.Tensor:
        if self.compression_prob > 0.0:
            img_tensor = apply_video_compression_noise(img_tensor, p=self.compression_prob)
        if self.cursor_prob > 0.0 and self.cursor_rgba is not None:
            img_tensor = overlay_cursor_noise(img_tensor, self.cursor_rgba, p=self.cursor_prob)
        return img_tensor
