import concurrent.futures
import itertools
import json
import logging
import math
import os
from pathlib import Path
import queue
import threading
import time
from typing import Any, Sequence

import cv2
from tqdm import tqdm

from config import Config, configure_logging, get_config
from tools.debug_plotter import plot
from eval_template import eval, format_timestamp
from frame import iter_video_frames
from screen_state import ScreenStateTracker
from template import template_loader


def build_output_json_path(
    output_dir: str, input_video_path: str, timestamp: int | None = None
) -> str:
    """Builds the output JSON path from any Windows or POSIX video file path."""
    ts = int(time.time()) if timestamp is None else timestamp
    base_name = Path(input_video_path).stem
    return str(Path(output_dir) / f"{base_name}_{ts}.json")


def get_expected_frame_count(config: Config) -> int | None:
    cap = cv2.VideoCapture(config.input_video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps <= 0:
        return None

    start_frame = (
        math.ceil(config.start_timestamp * fps)
        if config.start_timestamp is not None
        else 0
    )
    end_frame = (
        math.ceil(config.end_timestamp * fps)
        if config.end_timestamp is not None
        else total_frames
    )
    end_frame = min(end_frame, total_frames)

    frame_span = max(0, end_frame - start_frame)
    step = config.skip_frames + 1
    return math.ceil(frame_span / step)


def _process_frame(
    video_frame: Any,
    active_templates: Sequence[Any],
    template_key: dict[str, int],
    config: Config,
    logger: logging.Logger,
) -> tuple[int, list, dict[str, float], dict | None, str | None]:
    frame_idx = video_frame.index
    need_ts = config.debug_plot or config.setup_mode
    ts_str = (
        format_timestamp(video_frame.timestamp_seconds, frame_idx)
        if need_ts
        else ""
    )

    if logger.isEnabledFor(logging.INFO):
        logger.info(f"Processing frame {frame_idx}...")

    t0 = time.perf_counter()
    img_gray = cv2.cvtColor(video_frame.image, cv2.COLOR_BGR2GRAY)
    img_gray_half = (
        cv2.resize(
            img_gray,
            (img_gray.shape[1] // 2, img_gray.shape[0] // 2),
            interpolation=cv2.INTER_NEAREST,
        )
        if config.coarse_to_fine and not need_ts
        else None
    )
    t1 = time.perf_counter()

    matches = []
    template_clicked = None
    clicked_name = None
    for template in active_templates:
        result = eval(
            template,
            video_frame.image,
            img_gray,
            config,
            video_timestamp_str=ts_str,
            img_gray_half=img_gray_half,
        )
        if hasattr(result, "plot_kwargs") and result.found and result.plot_kwargs:
            matches.append((template, result.plot_kwargs))
        if result.found and result.clicked:
            clicked_name = result.template_name
            template_clicked = {
                "action": template_key[result.template_name],
                "box": list(result.box) if result.box is not None else None,
            }
            break  # only one template can be clicked in a given frame
    t2 = time.perf_counter()

    metrics = {
        "grayscale_time": t1 - t0,
        "eval_time": t2 - t1,
    }
    return frame_idx, matches, metrics, template_clicked, clicked_name


def _chunked_iterable(iterable, size: int):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk


def _prefetch_generator(iterable, prefetch: int = 64):
    q = queue.Queue(maxsize=prefetch)
    sentinel = object()
    producer_exc: list[BaseException | None] = [None]

    def producer():
        try:
            for item in iterable:
                q.put(item)
        except BaseException as exc:
            producer_exc[0] = exc
        finally:
            q.put(sentinel)

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    while True:
        item = q.get()
        if item is sentinel:
            if producer_exc[0] is not None:
                raise producer_exc[0]
            break
        yield item


def detect_video_actions(
    config: Config,
    *,
    write_output: bool = True,
    output_json_path: str | None = None,
    templates: Sequence[Any] | None = None,
    extra_metadata: dict[str, Any] | None = None,
    progress_desc: str = "Processing video",
) -> tuple[dict[str, Any], str | None]:
    """Runs first-pass template action detection on `config.input_video_path`.

    Returns `(output_payload, written_json_path)` where `output_payload` always
    contains `"key"` and `"frames"` (plus `"meta"` when `extra_metadata` is supplied).
    """
    logger = logging.getLogger(__name__)
    loaded_templates = (
        list(templates)
        if templates is not None
        else template_loader(
            config.template_json_path,
            config.template_img_dir,
            config.resolution,
        )
    )
    template_key = {template.name: i for i, template in enumerate(loaded_templates)}
    # Starting mid-run is out of scope: screen gating begins in PRE_GAME.
    screen_tracker = ScreenStateTracker()

    raw_frames = iter_video_frames(
        config.input_video_path,
        config.resolution,
        config.skip_frames,
        start_timestamp_seconds=config.start_timestamp,
        end_timestamp_seconds=config.end_timestamp,
        use_hwaccel=config.use_hwaccel,
    )

    video_extract_times = []

    def timed_frames(it):
        while True:
            t0 = time.perf_counter()
            try:
                frame = next(it)
            except StopIteration:
                break
            t1 = time.perf_counter()
            if config.run_benchmark:
                video_extract_times.append(t1 - t0)
            yield frame

    prefetched_frames = _prefetch_generator(raw_frames, prefetch=128)
    frames = timed_frames(prefetched_frames)

    start = time.perf_counter()
    results = []
    total_grayscale_time = 0.0
    total_eval_time = 0.0

    num_workers = 1
    action_frames: dict[int, dict[str, Any]] = {}

    if config.debug_plot or config.setup_mode:
        # Sequential processing to ensure safe UI/Plotting on the main thread
        for video_frame in frames:
            active_templates = screen_tracker.filter_templates(loaded_templates)
            frame_idx, matches, metrics, template_clicked, clicked_name = (
                _process_frame(
                    video_frame, active_templates, template_key, config, logger
                )
            )
            if template_clicked is not None and clicked_name is not None:
                action_frames[frame_idx] = template_clicked
                screen_tracker.on_template_clicked(clicked_name)
            total_grayscale_time += metrics["grayscale_time"]
            total_eval_time += metrics["eval_time"]
            for template, plot_kwargs in matches:
                should_plot = config.debug_plot or (
                    config.setup_mode and not template.adjusted
                )
                if should_plot:
                    plot(**plot_kwargs)
                    try:
                        with open(
                            config.template_json_path, "r", encoding="utf-8"
                        ) as f:
                            data = json.load(f)
                        if template.name in data:
                            template.adjusted = data[template.name].get(
                                "adjusted", template.adjusted
                            )
                            template.shadow_dy = data[template.name].get(
                                "shadow_dy", template.shadow_dy
                            )
                            template.shadow_dw = data[template.name].get(
                                "shadow_dw", template.shadow_dw
                            )
                            template.shadow_dh = data[template.name].get(
                                "shadow_dh", template.shadow_dh
                            )
                    except Exception as e:
                        logger.error(
                            f"Failed to reload template data for {template.name}: {e}"
                        )
            results.append(frame_idx)
    else:
        # Chunked multithreaded processing for CPU maximum speed (avoids OOM on large videos)
        total_frames = get_expected_frame_count(config)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=config.max_workers
        ) as executor:
            num_workers = executor._max_workers
            with tqdm(
                total=total_frames, desc=progress_desc, unit="frame"
            ) as pbar:
                for chunk in _chunked_iterable(frames, 64):
                    pending = list(chunk)
                    while pending:
                        active_templates = screen_tracker.filter_templates(
                            loaded_templates
                        )
                        chunk_outputs = list(
                            executor.map(
                                lambda vf, tmpls=active_templates: _process_frame(
                                    vf, tmpls, template_key, config, logger
                                ),
                                pending,
                            )
                        )
                        transitioned_at = None
                        for idx_in_chunk, (
                            frame_idx,
                            matches,
                            metrics,
                            template_clicked,
                            clicked_name,
                        ) in enumerate(chunk_outputs):
                            results.append(frame_idx)
                            total_grayscale_time += metrics["grayscale_time"]
                            total_eval_time += metrics["eval_time"]
                            pbar.update(1)
                            if template_clicked is not None and clicked_name is not None:
                                action_frames[frame_idx] = template_clicked
                                prev_state = (
                                    screen_tracker.state,
                                    screen_tracker.allow_play_game,
                                    screen_tracker.pre_pack_location,
                                )
                                screen_tracker.on_template_clicked(clicked_name)
                                cur_state = (
                                    screen_tracker.state,
                                    screen_tracker.allow_play_game,
                                    screen_tracker.pre_pack_location,
                                )
                                if cur_state != prev_state:
                                    transitioned_at = idx_in_chunk
                                    break
                        if transitioned_at is not None:
                            # Re-run only the suffix after a state transition under the updated whitelist
                            pending = pending[transitioned_at + 1 :]
                        else:
                            pending = []

    output_payload: dict[str, Any] = {
        "key": template_key,
        "frames": action_frames,
    }
    if extra_metadata is not None:
        output_payload["meta"] = extra_metadata

    num_frames = len(results)
    out_path: str | None = None
    if num_frames > 0:
        elapsed = time.perf_counter() - start
        avg_wall_time = elapsed / num_frames
        print(f"Finished in {elapsed:.2f} seconds; {avg_wall_time:.4f} per frame.")
        if config.run_benchmark:
            wall_extract = sum(video_extract_times) / num_frames
            wall_gray = (total_grayscale_time / num_frames) / num_workers
            wall_eval = (total_eval_time / num_frames) / num_workers
            wall_other = avg_wall_time - (wall_extract + wall_gray + wall_eval)

            pct_extract = (wall_extract / avg_wall_time) * 100
            pct_gray = (wall_gray / avg_wall_time) * 100
            pct_eval = (wall_eval / avg_wall_time) * 100
            pct_other = (wall_other / avg_wall_time) * 100

            print("\n--- Benchmark Breakdown (Effective Wall Time per frame) ---")
            print(f"Video Extraction:   {wall_extract:.5f}s ({pct_extract:4.1f}%)")
            print(f"Grayscale Convert:  {wall_gray:.5f}s ({pct_gray:4.1f}%)")
            print(f"Template Matching:  {wall_eval:.5f}s ({pct_eval:4.1f}%)")
            print(f"Other (Overhead):   {wall_other:.5f}s ({pct_other:4.1f}%)")
            print("-----------------------------------------------------------")

        if write_output:
            out_path = output_json_path or build_output_json_path(
                config.output_dir, config.input_video_path
            )
            os.makedirs(Path(out_path).parent, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(output_payload, f, indent=2)

    return output_payload, out_path


if __name__ == "__main__":
    configure_logging()
    detect_video_actions(get_config())
