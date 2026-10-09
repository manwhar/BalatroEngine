import argparse
import logging
from dataclasses import dataclass

import cv2

from geometry import ImageSize


@dataclass
class Config:
    template_json_path: str = "assets/templates/drspectred/templates.json"
    template_img_dir: str = "assets/templates/drspectred"
    input_img_path: str = "assets/sample_inputs/scraped_frames/test_frame_1.jpg"
    input_video_path: str = "assets/sample_inputs/videos/bu_full.mp4"
    output_dir: str = "output"
    resolution_width: int = 1920
    resolution_height: int = 1080
    matching_method: int = cv2.TM_CCORR_NORMED

    elem_conf_thresh: float = 0.946  # found element? confidence threshold
    expand_roi_factor: float = 1.1
    shadow_region_height: int = 10  # pixels
    shadow_threshold: float = 52.0
    skip_frames: int = 2  # skip n frames then use 1

    start_timestamp: float = 0
    end_timestamp: float = 1

    debug_plot: bool = False
    setup_mode: bool = False
    run_benchmark: bool = True

    @property
    def resolution(self) -> ImageSize:
        return ImageSize(width=self.resolution_width, height=self.resolution_height)


def get_config() -> Config:
    default_cfg = Config()
    parser = argparse.ArgumentParser(description="Botlatro Video Processing Config")
    parser.add_argument(
        "--template-json", type=str, default=default_cfg.template_json_path
    )
    parser.add_argument(
        "--template-dir", type=str, default=default_cfg.template_img_dir
    )
    parser.add_argument("--input-img", type=str, default=default_cfg.input_img_path)
    parser.add_argument("--input-video", type=str, default=default_cfg.input_video_path)
    parser.add_argument(
        "--start-timestamp", type=float, default=default_cfg.start_timestamp
    )
    parser.add_argument(
        "--end-timestamp", type=float, default=default_cfg.end_timestamp
    )
    parser.add_argument("--skip-frames", type=int, default=default_cfg.skip_frames)
    parser.add_argument(
        "--debug-plot",
        action=argparse.BooleanOptionalAction,
        default=default_cfg.debug_plot,
    )
    parser.add_argument(
        "--setup-mode",
        action=argparse.BooleanOptionalAction,
        default=default_cfg.setup_mode,
    )
    parser.add_argument(
        "--run-benchmark",
        action=argparse.BooleanOptionalAction,
        default=default_cfg.run_benchmark,
    )

    args, unknown = parser.parse_known_args()

    return Config(
        template_json_path=args.template_json,
        template_img_dir=args.template_dir,
        input_img_path=args.input_img,
        input_video_path=args.input_video,
        start_timestamp=args.start_timestamp,
        end_timestamp=args.end_timestamp,
        skip_frames=args.skip_frames,
        debug_plot=args.debug_plot,
        setup_mode=args.setup_mode,
        run_benchmark=args.run_benchmark,
    )


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="[%(levelname)s] %(message)s",
    )
