"""
Common Utilities and Video Extraction Helpers for Balatro Classifiers
"""

import cv2
import numpy as np
from pathlib import Path


def parse_timestamp(ts_str: str) -> float:
    """Parses timestamps like '04:00', '1:23:45', or '240.5' into float seconds."""
    ts_str = str(ts_str).strip()
    if ":" in ts_str:
        parts = [float(p) for p in ts_str.split(":")]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        elif len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        else:
            raise ValueError(f"Invalid timestamp format: {ts_str}")
    return float(ts_str)


def format_seconds(seconds: float) -> str:
    """Formats float seconds into HH:MM:SS.mmm string."""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"
    return f"{int(m):02d}:{s:06.3f}"


def extract_video_frame(video_path: Path, timestamp_sec: float) -> tuple[np.ndarray, float, int]:
    """Extracts a single BGR frame from a video at the closest timestamp."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frame = int(round(timestamp_sec * fps))
    target_frame = max(0, min(total_frames - 1, target_frame))

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    actual_pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
    actual_sec = actual_pos_msec / 1000.0 if actual_pos_msec > 0 else (target_frame / fps)
    cap.release()

    if not ret or frame is None:
        raise RuntimeError(f"Failed to read frame at timestamp {timestamp_sec:.2f}s from {video_path}")

    return frame, actual_sec, target_frame


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

    area_a = max(0.0, (box_a[2] - box_a[0])) * max(0.0, (box_a[3] - box_a[1]))
    area_b = max(0.0, (box_b[2] - box_b[0])) * max(0.0, (box_b[3] - box_b[1]))

    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def scale_roi_pixels(roi: dict, frame_w: int, frame_h: int, base_w: float = 1920.0, base_h: float = 1080.0) -> tuple[int, int, int, int]:
    """Calculates pixel ROI coordinates scaled to the frame's actual resolution."""
    scale_x = frame_w / base_w
    scale_y = frame_h / base_h

    rx = int(round(roi["x"] * scale_x))
    ry = int(round(roi["y"] * scale_y))
    rw = int(round(roi["w"] * scale_x))
    rh = int(round(roi["h"] * scale_y))

    rx = max(0, min(frame_w - 1, rx))
    ry = max(0, min(frame_h - 1, ry))
    rw = min(frame_w - rx, rw)
    rh = min(frame_h - ry, rh)
    return rx, ry, rw, rh
