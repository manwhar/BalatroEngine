"""
Regenerate Training and Validation Visualizations
==================================================
Patches Ultralytics plotting to:
1. Vertically stack labels that would otherwise overlap on the same card.
2. Color prediction labels and bounding boxes in _pred.jpg:
   - GREEN (34, 177, 76) if the model got it right (correct class & IoU >= 0.50).
   - RED (220, 20, 20) if the model got it wrong (misclassified or false positive).
   Accurate 1-to-1 per-detection color indexing with zero queue shift or desynchronization.
3. Reduce font size and bounding-box outline thickness for compactness.
4. Increase mosaic resolution to 2880px so card details and text remain crisp.
5. Expand confusion matrix figure dimensions (24x20 inches) for readable class names.
6. Re-runs validation on a chosen run directory (defaults to latest run in runs/).
7. Moves all regenerated figures into that run's figures/ subfolder.
8. Re-generates results.png from results.csv.
"""

import sys
import shutil
import math
import argparse
import warnings
from pathlib import Path
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from ultralytics.utils import ops
from ultralytics.utils.metrics import box_iou
import ultralytics.utils.plotting as plotting_module
import ultralytics.utils.metrics as metrics_module
import ultralytics.models.yolo.detect.val as val_module
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.plotting import Annotator, colors, plot_results
from ultralytics import YOLO

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
DATA_YAML = CURRENT_DIR / "data.yaml"
RUNS_ROOT = CURRENT_DIR / "runs"

COLOR_CORRECT = (34, 177, 76)  # RGB Green
COLOR_WRONG   = (220, 20, 20)  # RGB Red


def find_target_run(target=None) -> Path:
    """Finds the target run directory, defaulting to the latest run with best.pt."""
    if target:
        p = Path(target)
        if not p.is_absolute():
            p = RUNS_ROOT / target
        if p.exists() and (p / "weights" / "best.pt").exists():
            return p
        raise FileNotFoundError(f"Run directory not found or missing weights: {p}")

    if not RUNS_ROOT.exists():
        raise FileNotFoundError(f"No runs directory found at {RUNS_ROOT}")

    candidates = [
        d for d in RUNS_ROOT.iterdir()
        if d.is_dir() and (d / "weights" / "best.pt").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No valid run directories with weights/best.pt found in {RUNS_ROOT}")

    # Return newest by modification time
    candidates.sort(key=lambda d: (d / "weights" / "best.pt").stat().st_mtime, reverse=True)
    return candidates[0]


def organize_run_figures(run_dir: Path):
    """Moves all figure files in run_dir into run_dir/figures/."""
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


def apply_smart_plotting_patch(font_size=11, line_width=1, max_size=2880):
    """Patches Annotator, DetectionValidator, and ConfusionMatrix for clean, color-coded visualization."""
    orig_annotator_init = Annotator.__init__

    def patched_annotator_init(self, im, *args, **kwargs):
        kwargs["line_width"] = line_width
        kwargs["font_size"] = font_size
        orig_annotator_init(self, im, *args, **kwargs)
        self.lw = line_width
        self._drawn_labels = []

    def patched_box_label(self, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255)):
        txt_color = self.get_txt_color(color, txt_color)
        if isinstance(box, (torch.Tensor, np.ndarray)):
            box = box.tolist()

        multi_points = isinstance(box[0], list)
        p1 = [int(b) for b in box[0]] if multi_points else (int(box[0]), int(box[1]))

        if self.pil:
            if multi_points:
                self.draw.polygon([tuple(b) for b in box], width=self.lw, outline=color)
            else:
                self.draw.rectangle(box, width=self.lw, outline=color)

            if label:
                w, h = self.font.getsize(label)
                x1 = p1[0]
                y1 = p1[1] - h if p1[1] >= h else p1[1]
                if x1 > self.im.size[0] - w:
                    x1 = max(0, self.im.size[0] - w)

                if not hasattr(self, "_drawn_labels"):
                    self._drawn_labels = []

                # Smart collision avoidance: shift downward if overlapping any existing text tag
                shifted = True
                while shifted:
                    shifted = False
                    for (ox1, oy1, ox2, oy2) in self._drawn_labels:
                        if not (x1 + w < ox1 or x1 > ox2 or y1 + h < oy1 or y1 > oy2):
                            y1 = oy2 + 1
                            shifted = True
                            break
                    if y1 + h > self.im.size[1]:
                        break

                self._drawn_labels.append((x1, y1, x1 + w, y1 + h))
                self.draw.rectangle((x1, y1, x1 + w + 1, y1 + h + 1), fill=color)
                self.draw.text((x1, y1), label, fill=txt_color, font=self.font)
        else:
            import cv2
            if multi_points:
                cv2.polylines(self.im, [np.asarray(box, dtype=int)], True, color, self.lw)
            else:
                cv2.rectangle(self.im, p1, (int(box[2]), int(box[3])), color, thickness=self.lw, lineType=cv2.LINE_AA)
            if label:
                w, h = cv2.getTextSize(label, 0, fontScale=self.sf, thickness=self.tf)[0]
                h += 3
                outside = p1[1] >= h
                if p1[0] > self.im.shape[1] - w:
                    p1 = self.im.shape[1] - w, p1[1]
                p2 = p1[0] + w, p1[1] - h if outside else p1[1] + h
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)
                cv2.putText(self.im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h - 1), 0, self.sf, txt_color, thickness=self.tf, lineType=cv2.LINE_AA)

    Annotator.__init__ = patched_annotator_init
    Annotator.box_label = patched_box_label

    def patched_plot_images(
        labels: dict,
        images=None,
        paths=None,
        fname="images.jpg",
        names=None,
        on_plot=None,
        max_size: int = max_size,
        max_subplots: int = 16,
        save: bool = True,
        conf_thres: float = 0.25,
        show_labels: bool = True,
        show_conf: bool = True,
        **kwargs
    ):
        """Custom plot_images that applies custom_colors directly per detection without queue."""
        images = np.zeros((0, 3, 640, 640), dtype=np.float32) if images is None else images
        for k in ("cls", "bboxes", "conf", "masks", "keypoints", "batch_idx", "images", "semantic_mask", "depth", "custom_colors"):
            if k not in labels:
                continue
            if k == "cls" and labels[k].ndim == 2:
                labels[k] = labels[k].squeeze(1)
            if isinstance(labels[k], torch.Tensor):
                labels[k] = labels[k].cpu().numpy()

        cls = labels.get("cls", np.zeros(0, dtype=np.int64))
        batch_idx = labels.get("batch_idx", np.zeros(cls.shape, dtype=np.int64))
        bboxes = labels.get("bboxes", np.zeros(0, dtype=np.float32))
        confs = labels.get("conf", None)
        masks = labels.get("masks", np.zeros(0, dtype=np.uint8))
        kpts = labels.get("keypoints", np.zeros(0, dtype=np.float32))
        semantic_masks = labels.get("semantic_mask", np.zeros(0, dtype=np.int64))
        depth_maps = labels.get("depth", np.zeros(0, dtype=np.float32))
        images = labels.get("img", images)
        custom_colors = labels.get("custom_colors", None)

        if len(images) and isinstance(images, torch.Tensor):
            images = images.cpu().float().numpy()

        c = images.shape[1]
        if c == 2:
            zero = np.zeros_like(images[:, :1])
            images = np.concatenate((images, zero), axis=1)
        elif c > 3:
            images = images[:, :3]

        bs, _, h, w = images.shape
        bs = min(bs, max_subplots)
        ns = np.ceil(bs**0.5)
        if np.max(images[0]) <= 1:
            images *= 255

        mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)
        for i in range(bs):
            x, y = int(w * (i // ns)), int(h * (i % ns))
            mosaic[y : y + h, x : x + w, :] = images[i].transpose(1, 2, 0)

        scale = max_size / ns / max(h, w)
        if scale < 1:
            h = math.ceil(scale * h)
            w = math.ceil(scale * w)
            mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))

        fs = int((h + w) * ns * 0.01)
        fs = max(fs, 18)
        annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True, example=str(names))
        for i in range(bs):
            x, y = int(w * (i // ns)), int(h * (i % ns))
            annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)
            if paths:
                annotator.text([x + 5, y + 5], text=Path(paths[i]).name[:40], txt_color=(220, 220, 220))
            if len(cls) > 0:
                idx = batch_idx == i
                classes = cls[idx].astype("int")
                is_gt_labels = confs is None
                conf = confs[idx] if confs is not None else None
                sub_colors = custom_colors[idx] if custom_colors is not None else None

                if len(bboxes):
                    boxes = bboxes[idx]
                    if len(boxes):
                        if boxes[:, :4].max() <= 1.1:
                            boxes[..., [0, 2]] *= w
                            boxes[..., [1, 3]] *= h
                        elif scale < 1:
                            boxes[..., :4] *= scale
                    boxes[..., 0] += x
                    boxes[..., 1] += y
                    is_obb = boxes.shape[-1] == 5
                    boxes = ops.xywhr2xyxyxyxy(boxes) if is_obb else ops.xywh2xyxy(boxes)
                    for j, box in enumerate(boxes.astype(np.int64).tolist()):
                        c = classes[j]
                        if sub_colors is not None and j < len(sub_colors):
                            color = tuple(int(val) for val in sub_colors[j])
                        else:
                            color = colors(c)
                        c = names.get(c, c) if names else c
                        if is_gt_labels or conf[j] > conf_thres:
                            conf_text = f"{conf[j]:.1f}" if conf is not None else ""
                            label = f"{c}" if show_labels else ""
                            label += f" {conf_text}".strip() if show_conf else ""
                            annotator.box_label(box, label, color=color)

        if not save:
            return np.asarray(annotator.im)
        annotator.im.save(fname)
        if on_plot:
            on_plot(fname)

    plotting_module.plot_images = patched_plot_images
    val_module.plot_images = patched_plot_images

    # Patch DetectionValidator.plot_predictions to evaluate predictions against GT and color-code
    def custom_plot_predictions(self, batch, preds, ni, max_det=None):
        if not preds:
            return

        device = self.device
        max_det = max_det or self.args.max_det

        b_idx_t = torch.as_tensor(batch["batch_idx"], device=device)
        cls_t = torch.as_tensor(batch["cls"], device=device).reshape(-1)
        bboxes_t = torch.as_tensor(batch["bboxes"], device=device)
        imgsz = batch["img"].shape[2:]

        for i, pred in enumerate(preds):
            pred["batch_idx"] = torch.ones_like(pred["conf"]) * i
            p_cls = pred["cls"][:max_det]
            p_boxes = pred["bboxes"][:max_det]

            idx = (b_idx_t == i)
            gt_cls = cls_t[idx]
            gt_boxes = bboxes_t[idx]

            n_p = len(p_cls)
            if n_p == 0:
                pred["custom_colors"] = torch.zeros((0, 3), dtype=torch.uint8, device=device)
                continue

            if len(gt_cls) > 0:
                gt_boxes_xyxy = ops.xywh2xyxy(gt_boxes) * torch.tensor(imgsz, device=device)[[1, 0, 1, 0]]
                iou = box_iou(gt_boxes_xyxy, p_boxes)
                tp = self.match_predictions(p_cls, gt_cls, iou)
                is_correct = tp[:, 0].cpu().numpy().astype(bool)
            else:
                is_correct = np.zeros(n_p, dtype=bool)

            colors_arr = np.where(is_correct[:, None], COLOR_CORRECT, COLOR_WRONG).astype(np.uint8)
            pred["custom_colors"] = torch.tensor(colors_arr, device=device)

        keys = ["cls", "bboxes", "conf", "batch_idx", "custom_colors"]
        batched_preds = {k: torch.cat([x[k][:max_det] for x in preds], dim=0) for k in keys}
        batched_preds["bboxes"] = ops.xyxy2xywh(batched_preds["bboxes"])

        patched_plot_images(
            images=batch["img"],
            labels=batched_preds,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    DetectionValidator.plot_predictions = custom_plot_predictions

    # Patch ConfusionMatrix.plot for high-res readable 68-class visualization
    def patched_cm_plot(self, normalize=True, save_dir="", on_plot=None):
        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1e-9) if normalize else 1)
        array[array < 0.005] = np.nan

        fig, ax = plt.subplots(1, 1, figsize=(24, 20), dpi=150)
        names = list(self.names.values())
        nc = self.nc + 1
        ticklabels = [*names, "background"]
        xy_ticks = np.arange(len(ticklabels))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            im = ax.imshow(array, cmap="Blues", vmin=0.0, interpolation="none")
            cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)

        title = "Confusion Matrix" + " Normalized" * normalize
        ax.set_xlabel("True", fontsize=14, labelpad=10)
        ax.set_ylabel("Predicted", fontsize=14, labelpad=10)
        ax.set_title(title, fontsize=16, pad=20)
        ax.set_xticks(xy_ticks)
        ax.set_yticks(xy_ticks)
        ax.set_xticklabels(ticklabels, fontsize=7, rotation=90, ha="center")
        ax.set_yticklabels(ticklabels, fontsize=7)

        for s in ("left", "right", "bottom", "top", "outline"):
            if s != "outline":
                ax.spines[s].set_visible(False)
            cbar.ax.spines[s].set_visible(False)

        fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.18)
        plot_fname = Path(save_dir) / f"{title.lower().replace(' ', '_')}.png"
        fig.savefig(plot_fname, dpi=150)
        plt.close(fig)
        if on_plot:
            on_plot(plot_fname, {"type": "confusion_matrix", "matrix": self.matrix.tolist()})

    metrics_module.ConfusionMatrix.plot = patched_cm_plot
    print("[Patch] Deterministic green/red prediction coloring, smart label stacking, and high-res plotting active.")


def regenerate(target_run=None):
    run_dir = find_target_run(target_run)
    weights_path = run_dir / "weights" / "best.pt"
    csv_path = run_dir / "results.csv"

    print("=" * 60)
    print("Regenerating Visualizations with Green/Red Prediction Coloring")
    print(f"Target Run : {run_dir.name}")
    print(f"Weights    : {weights_path}")
    print(f"Data       : {DATA_YAML}")
    print("=" * 60)

    apply_smart_plotting_patch(font_size=11, line_width=1, max_size=2880)

    print("\n1. Running validation to generate color-coded prediction figures...")
    model = YOLO(str(weights_path))
    val_results = model.val(
        data=str(DATA_YAML),
        project=run_dir.parent.as_posix(),
        name=run_dir.name,
        exist_ok=True,
        plots=True,
        batch=32,
        workers=0,
        device=0 if torch.cuda.is_available() else "cpu"
    )

    if csv_path.exists():
        print("\n2. Re-generating results.png metric plots...")
        plot_results(file=csv_path)

    organize_run_figures(run_dir)

    print("\n" + "=" * 60)
    print("Regeneration complete!")
    print(f"Figures updated in : {run_dir / 'figures'}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate high-resolution training/val plots for a run")
    parser.add_argument("--run", type=str, default=None, help="Run directory name or path (defaults to latest)")
    args = parser.parse_args()
    regenerate(args.run)
