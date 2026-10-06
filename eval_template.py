"""
Evaluates a given template for a given frame.
"""

import logging
import json
from itertools import product

import cv2

from config import Config
from debug_plotter import plot
from geometry import PixelROI, crop_image_to_roi
from matcher import find_element
from models import EvaluationResult
from shadow_evaluator import analyze_shadow_presence
from template import Template

logger = logging.getLogger(__name__)


def format_timestamp(timestamp_seconds: float, frame_index: int) -> str:
    sec = int(timestamp_seconds)
    hh = sec // 3600
    mm = (sec % 3600) // 60
    ss = sec % 60
    ff = frame_index % 60
    return f"{timestamp_seconds:.2f}s | {hh:02}:{mm:02}:{ss:02}:{ff:02}"


def eval(
    template: Template, img: cv2.mat_wrapper.Mat, img_gray: cv2.mat_wrapper.Mat, config: Config, video_timestamp_str: str = ""
) -> EvaluationResult:
    if img is None or img_gray is None:
        raise ValueError("Got None for image")

    from geometry import ImageSize
    for (image, image_gray, mask), px_roi in product(zip(template.images, template.images_gray, template.masks), template.px_rois):
        h_inp, w_inp = image.shape[:2]

        img_h, img_w = img.shape[:2]
        expanded_x1, expanded_y1, expanded_x2, expanded_y2 = px_roi.expanded_bounds(
            config.expand_roi_factor
        )

        y1, y2 = max(0, expanded_y1), min(img_h, expanded_y2)
        x1, x2 = max(0, expanded_x1), min(img_w, expanded_x2)

        # ensure roi is large enough for template
        if (y2 - y1) < h_inp:
            diff = h_inp - (y2 - y1)
            y1 = max(0, y1 - diff // 2)
            y2 = y1 + h_inp
            if y2 > img_h:
                y2 = img_h
                y1 = max(0, y2 - h_inp)
        if (x2 - x1) < w_inp:
            diff = w_inp - (x2 - x1)
            x1 = max(0, x1 - diff // 2)
            x2 = x1 + w_inp
            if x2 > img_w:
                x2 = img_w
                x1 = max(0, x2 - w_inp)

        if (y2 - y1) < h_inp or (x2 - x1) < w_inp:
            logger.warning("ROI is too small and cannot be expanded!")
            continue

        roi_img_gray = img_gray[y1:y2, x1:x2]
        roi_img = img[y1:y2, x1:x2]

        match = find_element(
            roi_img_gray,
            image_gray,
            mask,
            roi_img,
            image[:, :, :3],
            ImageSize(width=w_inp, height=h_inp),
            config.matching_method,
            px_roi,
            config.elem_conf_thresh,
        )
        top_left = (match.top_left[0] + x1, match.top_left[1] + y1)

        if match.found:
            bottom_left = (top_left[0], top_left[1] + h_inp)
            bottom_right = (top_left[0] + w_inp, top_left[1] + h_inp)

            dy = template.shadow_dy
            dw = template.shadow_dw
            dh = template.shadow_dh
            height = config.shadow_region_height
            shadow_region = PixelROI(
                x_min=bottom_left[0],
                y_min=bottom_left[1] + dy,
                x_max=bottom_right[0] + dw,
                y_max=bottom_right[1] + height + dy + dh,
            )

            shadow_img_gray = crop_image_to_roi(img_gray, shadow_region)
            has_shadow = analyze_shadow_presence(
                shadow_img_gray, threshold=config.shadow_threshold
            )
            clicked = not has_shadow

            logger.info(
                f"FOUND {template.name} at {top_left} with confidence of {match.confidence:.4f}; Clicked={clicked}"
            )
        else:
            logger.debug(
                f"Did not find {template.name} in ROI; highest confidence of {match.confidence:.4f}"
            )
            clicked = False

        should_plot = config.debug_plot or (config.setup_mode and not template.adjusted)
        if should_plot and match.found:
            plot_kwargs = {
                "img": img,
                "template": image,
                "roi": (x1, y1, x2, y2),
                "found_target": match.found,
                "match": match.score_map,
                "top_left": top_left,
                "max_val": match.confidence,
                "clicked": clicked,
                "label": template.name,
                "shadow_threshold": config.shadow_threshold,
                "template_name": template.name,
                "template_json_path": config.template_json_path,
                "current_shadow_dy": template.shadow_dy,
                "current_shadow_dw": template.shadow_dw,
                "current_shadow_dh": template.shadow_dh,
                "shadow_height": config.shadow_region_height,
                "video_timestamp_str": video_timestamp_str,
            }
            if match.found:
                plot_kwargs["shadow_region"] = shadow_region

            plot(**plot_kwargs)
            
            try:
                with open(config.template_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if template.name in data:
                    template.adjusted = data[template.name].get("adjusted", template.adjusted)
                    template.shadow_dy = data[template.name].get("shadow_dy", template.shadow_dy)
                    template.shadow_dw = data[template.name].get("shadow_dw", template.shadow_dw)
                    template.shadow_dh = data[template.name].get("shadow_dh", template.shadow_dh)
            except Exception as e:
                logger.error(f"Failed to reload template data for {template.name}: {e}")

        if match.found:
            return EvaluationResult(
                template_name=template.name,
                found=True,
                match=match,
                clicked=clicked,
                template_filename=None,  # not implemented rn, maybe will later
            )

    return EvaluationResult(
        template_name=template.name,
        found=False,
        match=None,
        clicked=None,
        template_filename=None,
    )
