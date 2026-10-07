import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cv2
from cv2.typing import MatLike
from geometry import ImageSize, NormalizedROI, PixelROI

@dataclass
class Template:
    name: str
    filenames: list[str]
    norm_rois: list[NormalizedROI]
    px_rois: list[PixelROI]
    images: list[MatLike]
    images_gray: list[MatLike]
    masks: list[MatLike]
    shadow_dy: int
    shadow_dw: int
    shadow_dh: int
    adjusted: bool


def template_loader(
    template_json_path: str, template_img_dir: str, resolution: ImageSize
) -> list[Template]:
    """Load all templates from json."""
    with open(template_json_path, "r") as f:
        TEMPLATES: dict[str, dict[str, Any]] = json.load(f)

    templates = []

    for name, data in TEMPLATES.items():
        filenames, rois = data["filenames"], data["rois"]
        filenames: list[str]
        rois: list[list[float]]

        norm_rois = []
        px_rois = []
        for roi in rois:
            normalized_roi = NormalizedROI(*roi)
            norm_rois.append(normalized_roi)
            px_rois.append(
                PixelROI(
                    round(normalized_roi.x_min * resolution.width),
                    round(normalized_roi.y_min * resolution.height),
                    round(normalized_roi.x_max * resolution.width),
                    round(normalized_roi.y_max * resolution.height),
                )
            )

        shadow_dy = data.get("shadow_dy", 0)
        shadow_dw = data.get("shadow_dw", 0)
        shadow_dh = data.get("shadow_dh", 0)
        adjusted = data.get("adjusted", False)

        images = [
            img
            for f in filenames
            if (img := cv2.imread(f"{template_img_dir}/{f}", cv2.IMREAD_UNCHANGED))
            is not None
        ]

        images_gray = []
        masks = []
        for img in images:
            images_gray.append(cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY))
            masks.append(img[:, :, 3] if img.shape[2] == 4 else None)

        templates.append(Template(name, filenames, norm_rois, px_rois, images, images_gray, masks, shadow_dy, shadow_dw, shadow_dh, adjusted))

    return templates
