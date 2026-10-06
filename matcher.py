from dataclasses import dataclass

import cv2
from cv2.typing import MatLike

from geometry import ImageSize, PixelROI


@dataclass(frozen=True, slots=True)
class TemplateMatch:
    found: bool
    top_left: tuple[int, int]
    confidence: float
    score_map: MatLike | None
    roi: PixelROI
    template_size: ImageSize


def find_element(
    img_gray,
    template_gray,
    mask,
    img_bgr,
    template_bgr,
    template_size,
    method,
    roi: PixelROI,
    confidence_threshold: float,
) -> TemplateMatch:
    if mask is not None:
        result: MatLike = cv2.matchTemplate(img_gray, template_gray, method, mask=mask)
    else:
        result: MatLike = cv2.matchTemplate(img_gray, template_gray, method)

    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    
    # Grayscale fast reject (allow 0.03 margin of error for color differences)
    if max_val < confidence_threshold - 0.03:
        return TemplateMatch(False, tuple(max_loc), max_val, result, roi, template_size)
        
    # Second pass BGR for true confidence
    if mask is not None:
        result_bgr = cv2.matchTemplate(img_bgr, template_bgr, method, mask=mask)
    else:
        result_bgr = cv2.matchTemplate(img_bgr, template_bgr, method)
        
    _, max_val_bgr, _, max_loc_bgr = cv2.minMaxLoc(result_bgr)
    found_target = max_val_bgr > confidence_threshold
    
    return TemplateMatch(found_target, tuple(max_loc_bgr), max_val_bgr, result_bgr, roi, template_size)
