import itertools
import json
import logging
import time
import concurrent.futures
import cv2

from config import get_config, configure_logging
from debug_plotter import plot
from eval_template import eval, format_timestamp
from frame import iter_video_frames
from template import template_loader

if __name__ == "__main__":
    configure_logging()

    logger = logging.getLogger(__name__)

    config = get_config()
    templates = template_loader(
        config.template_json_path,
        config.template_img_dir,
        config.resolution,
    )

    def process_frame(arg):
        idx, video_frame = arg
        ts_str = format_timestamp(video_frame.timestamp_seconds, video_frame.index)
        if not config.run_benchmark:
            print(f"Processing frame {idx}...", flush=True)
        
        t0 = time.perf_counter()
        img_gray = cv2.cvtColor(video_frame.image, cv2.COLOR_BGR2GRAY)
        t1 = time.perf_counter()
        
        matches = []
        for template in templates:
            result = eval(template, video_frame.image, img_gray, config, video_timestamp_str=ts_str)
            if hasattr(result, "plot_kwargs") and result.found and result.plot_kwargs:
                matches.append((template, result.plot_kwargs))
            if result.found and result.clicked:
                break
        t2 = time.perf_counter()
        
        metrics = {
            "grayscale_time": t1 - t0,
            "eval_time": t2 - t1,
        }
        return idx, matches, metrics

    def chunked_iterable(iterable, size):
        it = iter(iterable)
        while True:
            chunk = tuple(itertools.islice(it, size))
            if not chunk:
                break
            yield chunk

    def prefetch_generator(iterable, prefetch=64):
        import queue
        import threading
        q = queue.Queue(maxsize=prefetch)
        sentinel = object()
        
        def producer():
            try:
                for item in iterable:
                    q.put(item)
            finally:
                q.put(sentinel)
                
        thread = threading.Thread(target=producer, daemon=True)
        thread.start()
        
        while True:
            item = q.get()
            if item is sentinel:
                break
            yield item

    raw_frames = iter_video_frames(
        config.input_video_path,
        config.resolution,
        config.skip_frames,
        start_timestamp_seconds=config.start_timestamp,
        end_timestamp_seconds=config.end_timestamp,
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

    prefetched_frames = prefetch_generator(raw_frames, prefetch=128)
    frames = timed_frames(prefetched_frames)

    start = time.perf_counter()
    results = []
    total_grayscale_time = 0.0
    total_eval_time = 0.0

    num_workers = 1

    if config.debug_plot or config.setup_mode:
        # Sequential processing to ensure safe UI/Plotting on the main thread
        for arg in enumerate(frames):
            idx, matches, metrics = process_frame(arg)
            total_grayscale_time += metrics["grayscale_time"]
            total_eval_time += metrics["eval_time"]
            for template, plot_kwargs in matches:
                should_plot = config.debug_plot or (config.setup_mode and not template.adjusted)
                if should_plot:
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
            results.append(idx)
    else:
        # Chunked multithreaded processing for CPU maximum speed (avoids OOM on large videos)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            num_workers = executor._max_workers
            for chunk in chunked_iterable(enumerate(frames), 64):
                chunk_results = list(executor.map(process_frame, chunk))
                results.extend([r[0] for r in chunk_results])
                for r in chunk_results:
                    total_grayscale_time += r[2]["grayscale_time"]
                    total_eval_time += r[2]["eval_time"]

    num_frames = len(results)
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
