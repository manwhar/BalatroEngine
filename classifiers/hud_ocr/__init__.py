"""Lightweight GPU-batched HUD OCR package for Balatro telemetry reconstruction."""

from .config import (
    BLANK_IDX,
    CHAR_TO_IDX,
    DEFAULT_WEIGHT_CANDIDATES,
    HUD_ROIS,
    HUD_ROI_FIELDS,
    IDX_TO_CHAR,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    VOCAB,
)
from .model import HudOcrCRNN, ctc_greedy_decode
from .pipeline import HudOcrPipeline, HudOcrResult, parse_hud_value, resolve_weights_path

__all__ = [
    "BLANK_IDX",
    "CHAR_TO_IDX",
    "DEFAULT_WEIGHT_CANDIDATES",
    "HUD_ROIS",
    "HUD_ROI_FIELDS",
    "HudOcrCRNN",
    "HudOcrPipeline",
    "HudOcrResult",
    "IDX_TO_CHAR",
    "INPUT_HEIGHT",
    "INPUT_WIDTH",
    "VOCAB",
    "ctc_greedy_decode",
    "parse_hud_value",
    "resolve_weights_path",
]
