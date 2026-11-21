"""Overlapped concurrent downloader + Stage 1/2 processor for Balatro VODs."""

from __future__ import annotations

import concurrent.futures
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time
from typing import Any, Callable, Sequence

from post_processing import parse_deck, parse_stake
from vod_pipeline.metadata import (
    ExtractedRunMetadata,
    compact_raw_metadata,
    extract_run_metadata,
    fetch_metadata_batch,
)

# VideoReader requires 1920x1080 and audio is unused; prefer H.264/MP4 for fastest decode.
DEFAULT_VIDEO_FORMAT: str = (
    "bestvideo[width=1920][height=1080][vcodec^=avc1]/"
    "bestvideo[width=1920][height=1080][ext=mp4]/"
    "bestvideo[width=1920][height=1080]/"
    "best[width=1920][height=1080]/"
    "bestvideo[height=1080]/"
    "best[height=1080]"
)


@dataclass
class PipelineConfig:
    """Configuration for autonomous VOD batch processing."""

    work_dir: Path = Path("batch_runs")
    max_backlog: int = 5
    download_workers: int = 3
    metadata_workers: int = 6
    concurrent_fragments: int = 4
    video_format: str = DEFAULT_VIDEO_FORMAT
    keep_videos: bool = False
    resume: bool = True
    dry_run: bool = False
    allow_deck_template_fallback: bool = True
    allow_ocr_seed_fallback: bool = False
    skip_frames: int = 1
    batch_size: int = 32
    conf_hand: float = 0.50
    conf_shop: float = 0.15
    conf_ocr: float = 0.90
    device: str | None = None
    expressive: bool = False
    metadata_timeout_sec: int = 60
    download_timeout_sec: int = 3600


def _cleanup_partial_downloads(downloads_dir: Path, video_id: str) -> None:
    """Removes partial/temporary yt-dlp files for `video_id` after error or abort."""
    if not downloads_dir.exists():
        return
    for candidate in downloads_dir.glob(f"{video_id}*"):
        if candidate.is_file() and (
            candidate.name.endswith(".part")
            or candidate.name.endswith(".ytdl")
            or ".temp." in candidate.name
        ):
            candidate.unlink(missing_ok=True)


def _find_downloaded_file(downloads_dir: Path, video_id: str) -> Path | None:
    """Finds a completed media file for `video_id` inside `downloads_dir`."""
    candidates = [
        p
        for p in sorted(downloads_dir.glob(f"{video_id}.*"))
        if p.is_file()
        and not p.name.endswith((".part", ".ytdl", ".json"))
        and ".temp." not in p.name
        and p.stat().st_size > 0
    ]
    if not candidates:
        return None
    for p in candidates:
        if p.suffix.lower() == ".mp4":
            return p
    return candidates[0]


def download_video(
    url: str,
    video_id: str,
    downloads_dir: Path,
    *,
    video_format: str = DEFAULT_VIDEO_FORMAT,
    concurrent_fragments: int = 4,
    timeout_sec: int = 3600,
    stop_event: threading.Event | None = None,
) -> Path:
    """Downloads a single 1080p video (no audio needed) into `downloads_dir` via `yt-dlp`."""
    downloads_dir.mkdir(parents=True, exist_ok=True)
    existing = _find_downloaded_file(downloads_dir, video_id)
    if existing is not None:
        return existing

    if stop_event is not None and stop_event.is_set():
        raise InterruptedError("Download aborted before start")

    out_tmpl = str(downloads_dir / f"{video_id}.%(ext)s")
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--no-warnings",
        "--newline",
        "--concurrent-fragments",
        str(max(1, int(concurrent_fragments))),
        "-f",
        video_format,
        "--merge-output-format",
        "mp4",
        "-o",
        out_tmpl,
        "--print",
        "after_move:filepath",
        url,
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    deadline = time.monotonic() + max(10, int(timeout_sec))
    try:
        while proc.poll() is None:
            if stop_event is not None and stop_event.is_set():
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise InterruptedError(f"Download cancelled for {video_id}")
            if time.monotonic() > deadline:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise TimeoutError(f"yt-dlp download timed out after {timeout_sec}s for {video_id}")
            time.sleep(0.25)

        stdout_text, stderr_text = proc.communicate()
        if proc.returncode != 0:
            err_msg = (stderr_text or stdout_text or f"yt-dlp exit {proc.returncode}").strip()
            raise RuntimeError(f"yt-dlp failed ({proc.returncode}): {err_msg[:500]}")

        for line in reversed([ln.strip() for ln in stdout_text.splitlines() if ln.strip()]):
            p = Path(line)
            if p.exists() and p.is_file():
                return p
        found = _find_downloaded_file(downloads_dir, video_id)
        if found is None:
            raise FileNotFoundError(
                f"yt-dlp finished for {video_id} cleanly, but no output file was found in {downloads_dir}"
            )
        return found
    except BaseException:
        _cleanup_partial_downloads(downloads_dir, video_id)
        raise


def _release_runtime_memory() -> None:
    """Reclaims Python objects and cached CUDA memory between sequential VOD runs."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def process_downloaded_video(
    video_path: Path,
    meta: ExtractedRunMetadata,
    cfg: PipelineConfig,
) -> dict[str, Any]:
    """Runs `main.run_end_to_end` on a downloaded video with extracted `deck`, `stake`, `seed`."""
    import main as vod_main

    actions_dir = cfg.work_dir / "actions"
    trajectories_dir = cfg.work_dir / "trajectories"
    actions_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    actions_out = actions_dir / f"{meta.video_id}_actions.json"
    trajectory_out = trajectories_dir / f"{meta.video_id}_trajectory.json"

    # Validate enums before spawning Stage 1/2
    parse_deck(meta.deck_cli_value)
    parse_stake(meta.stake_cli_value)

    argv: list[str] = [
        "--input-video",
        str(video_path),
        "--full-video",
        "--output-dir",
        str(actions_dir),
        "--actions-out",
        str(actions_out),
        "--trajectory-out",
        str(trajectory_out),
        "--deck",
        meta.deck_cli_value,
        "--stake",
        meta.stake_cli_value,
        "--seed",
        meta.seed,
        "--skip-frames",
        str(cfg.skip_frames),
        "--batch-size",
        str(cfg.batch_size),
        "--conf-hand",
        str(cfg.conf_hand),
        "--conf-shop",
        str(cfg.conf_shop),
        "--conf-ocr",
        str(cfg.conf_ocr),
    ]
    if cfg.device:
        argv.extend(["--device", cfg.device])
    if cfg.expressive:
        argv.append("--expressive")

    args = vod_main.build_cli_parser().parse_args(argv)
    t0 = time.perf_counter()
    res = vod_main.run_end_to_end(args)
    elapsed = time.perf_counter() - t0
    traj = res.get("trajectory")
    st = res.get("state")
    return {
        "actions_json": str(res.get("actions_json") or actions_out),
        "trajectory_json": str(res.get("trajectory_json") or trajectory_out),
        "num_steps": len(traj.deltas) if traj is not None else None,
        "final_ante": int(st.ante) if st is not None else None,
        "final_round": int(st.round) if st is not None else None,
        "elapsed_sec": round(elapsed, 2),
    }


_DECK_RECOVERABLE_SKIP_REASONS = frozenset(
    {"missing_deck", "ambiguous_multiple_decks", "ambiguous_deck_conflict"}
)


def _try_deck_template_fallback(video_path: Path, cfg: PipelineConfig) -> Any | None:
    """Samples a few frames of `video_path` to identify the deck back when metadata lacks it."""
    try:
        from vod_pipeline.deck_matcher import detect_deck_from_video

        return detect_deck_from_video(video_path)
    except Exception:
        return None


def _try_ocr_seed_fallback(video_path: Path, cfg: PipelineConfig) -> tuple[str, float] | None:
    """Attempts end-of-run Seed OCR recovery when `--allow-ocr-seed-fallback` is enabled."""
    try:
        from classifiers.seed_ocr.pipeline import (
            extract_seed_from_video,
            resolve_seed_weights_path,
        )

        weights_path = resolve_seed_weights_path()
        if weights_path is None:
            return None
        res = extract_seed_from_video(video_path, device=cfg.device)
        if res is not None and res.seed:
            return res.seed, float(res.confidence)
    except Exception:
        return None
    return None


def _append_manifest_row(manifest_path: Path, row: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _default_fetch_all(urls: Sequence[str], cfg: PipelineConfig) -> list[tuple[dict[str, Any], ExtractedRunMetadata]]:
    meta_dir = cfg.work_dir / "metadata"
    return fetch_metadata_batch(
        urls,
        max_workers=cfg.metadata_workers,
        save_dir=meta_dir,
        timeout=cfg.metadata_timeout_sec,
    )


def _print_batch_summary(report: dict[str, Any]) -> None:
    counts = report["counts"]
    print("\n================ VOD Batch Pipeline Summary ================")
    print(
        f"Total URLs: {counts['total']} | Ready: {counts['eligible']} | "
        f"Processed: {counts['processed']} | Skipped: {counts['skipped']} | "
        f"Failed: {counts['failed']} | Dry-run: {report['config']['dry_run']}"
    )
    if report["processed"]:
        print("\n[Processed]")
        for row in report["processed"]:
            print(
                f"  - {row['video_id']} | {row['deck']} ({row.get('deck_source', '')}) | "
                f"{row['stake']} | Seed={row['seed']} ({row.get('seed_source', '')}) | "
                f"steps={row.get('num_steps')} -> {row.get('trajectory_json')}"
            )
    if report["skipped"]:
        print("\n[Skipped]")
        for row in report["skipped"]:
            print(
                f"  - {row['video_id']} | reason={row['skip_reason']} | "
                f"date={row.get('upload_date') or '?'} | title={row.get('title')!r}"
            )
    if report["failed"]:
        print("\n[Failed]")
        for row in report["failed"]:
            print(
                f"  - {row['video_id']} ({row['stage']}): {row['error']}"
            )
    print(f"\nReport saved: {report['report_path']}")
    print("============================================================\n")


def run_batch_pipeline(
    urls: Sequence[str],
    cfg: PipelineConfig,
    *,
    fetch_metadata_fn: Callable[[Sequence[str], PipelineConfig], list[tuple[dict[str, Any], ExtractedRunMetadata]]] | None = None,
    download_fn: Callable[[str, str, Path, PipelineConfig, threading.Event], Path] | None = None,
    process_fn: Callable[[Path, ExtractedRunMetadata, PipelineConfig], dict[str, Any]] | None = None,
    deck_matcher_fn: Callable[[Path, PipelineConfig], Any] | None = None,
) -> dict[str, Any]:
    """Executes metadata gating + overlapped bounded-backlog download & VOD processing."""
    t_start = time.perf_counter()
    work_dir = Path(cfg.work_dir)
    meta_dir = work_dir / "metadata"
    downloads_dir = work_dir / "downloads"
    trajectories_dir = work_dir / "trajectories"
    reports_dir = work_dir / "reports"
    meta_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    if not cfg.dry_run:
        downloads_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "actions").mkdir(parents=True, exist_ok=True)
        trajectories_dir.mkdir(parents=True, exist_ok=True)

    fetcher = fetch_metadata_fn or _default_fetch_all
    downloader = download_fn or (
        lambda u, vid, ddir, c, ev: download_video(
            u,
            vid,
            ddir,
            video_format=c.video_format,
            concurrent_fragments=c.concurrent_fragments,
            timeout_sec=c.download_timeout_sec,
            stop_event=ev,
        )
    )
    processor = process_fn or process_downloaded_video
    deck_matcher = deck_matcher_fn or _try_deck_template_fallback

    fetched = fetcher(urls, cfg)
    # Persist metadata if caller injected a custom fetcher
    if fetch_metadata_fn is not None and fetched:
        idx_rows = []
        for raw, meta in fetched:
            p = meta_dir / f"{meta.video_id}.json"
            payload = dict(raw) if "title" in raw else compact_raw_metadata(raw)
            payload["extracted"] = meta.to_dict()
            with p.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            idx_rows.append(meta.to_dict())
        with (meta_dir / "index.json").open("w", encoding="utf-8") as f:
            json.dump(idx_rows, f, indent=2, ensure_ascii=False)

    seed_ocr_weights_available = False
    if cfg.allow_ocr_seed_fallback:
        try:
            from classifiers.seed_ocr.pipeline import resolve_seed_weights_path

            seed_ocr_weights_available = resolve_seed_weights_path() is not None
        except Exception:
            seed_ocr_weights_available = False

    eligible: list[ExtractedRunMetadata] = []
    skipped: list[dict[str, Any]] = []
    for _raw, meta in fetched:
        traj_path = trajectories_dir / f"{meta.video_id}_trajectory.json"
        if cfg.resume and traj_path.exists() and traj_path.stat().st_size > 0:
            row = meta.to_dict()
            row["skip_reason"] = "already_processed"
            row["trajectory_json"] = str(traj_path)
            skipped.append(row)
            continue

        if meta.ready:
            eligible.append(meta)
            continue

        # Recover missing/ambiguous deck via sparse video frame template matching when enabled
        if (
            cfg.allow_deck_template_fallback
            and meta.skip_reason in _DECK_RECOVERABLE_SKIP_REASONS
            and meta.seed
        ):
            meta.deck = None
            meta.deck_source = "pending_deck_template_match"
            eligible.append(meta)
            continue

        # Optional OCR fallback ONLY if deck is valid, missing_seed is the sole gap, and weights exist
        if (
            cfg.allow_ocr_seed_fallback
            and seed_ocr_weights_available
            and meta.skip_reason == "missing_seed"
            and meta.deck is not None
        ):
            meta.seed_source = "pending_ocr_end_of_run"
            eligible.append(meta)
            continue

        if (
            cfg.allow_deck_template_fallback
            and cfg.allow_ocr_seed_fallback
            and seed_ocr_weights_available
            and (meta.skip_reason == "missing_deck_and_seed" or meta.skip_reason in _DECK_RECOVERABLE_SKIP_REASONS)
            and not meta.seed
        ):
            meta.deck = None
            meta.deck_source = "pending_deck_template_match"
            meta.seed_source = "pending_ocr_end_of_run"
            eligible.append(meta)
            continue

        skipped.append(meta.to_dict())

    processed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    manifest_path = reports_dir / "manifest.jsonl"

    if not cfg.dry_run and eligible:
        backlog = max(1, int(cfg.max_backlog))
        dl_workers = max(1, min(int(cfg.download_workers), backlog))
        slot_sem = threading.Semaphore(backlog)
        ready_queue: queue.Queue[tuple[ExtractedRunMetadata, Path, float] | None] = queue.Queue()
        stop_event = threading.Event()
        lock = threading.Lock()

        prev_sigint = signal.getsignal(signal.SIGINT)
        interrupted = [False]

        def _handle_sigint(signum, frame) -> None:
            interrupted[0] = True
            stop_event.set()
            raise KeyboardInterrupt

        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, _handle_sigint)

        def _download_one(item: ExtractedRunMetadata) -> None:
            t_dl0 = time.perf_counter()
            handed_off = False
            try:
                vpath = downloader(item.url, item.video_id, downloads_dir, cfg, stop_event)
                ready_queue.put((item, vpath, time.perf_counter() - t_dl0))
                handed_off = True
            except Exception as exc:
                _cleanup_partial_downloads(downloads_dir, item.video_id)
                fail_row = {
                    **item.to_dict(),
                    "stage": "download",
                    "error": str(exc),
                }
                with lock:
                    failed.append(fail_row)
                    _append_manifest_row(manifest_path, {"status": "failed", **fail_row})
            finally:
                if not handed_off:
                    slot_sem.release()

        def _dispatcher() -> None:
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=dl_workers) as pool:
                    futures: list[concurrent.futures.Future[Any]] = []
                    for item in eligible:
                        acquired = False
                        while not stop_event.is_set():
                            if slot_sem.acquire(timeout=0.25):
                                acquired = True
                                break
                        if not acquired:
                            break
                        if stop_event.is_set():
                            slot_sem.release()
                            break
                        futures.append(pool.submit(_download_one, item))
                    for fut in futures:
                        try:
                            fut.result()
                        except Exception:
                            pass
            finally:
                ready_queue.put(None)

        disp_thread = threading.Thread(target=_dispatcher, name="vod-download-dispatcher", daemon=True)
        disp_thread.start()

        try:
            while True:
                try:
                    pkt = ready_queue.get(timeout=0.5)
                except queue.Empty:
                    if stop_event.is_set() and not disp_thread.is_alive():
                        break
                    continue
                if pkt is None:
                    break

                item, vpath, dl_sec = pkt
                try:
                    if item.deck is None and cfg.allow_deck_template_fallback:
                        deck_match = deck_matcher(vpath, cfg)
                        if deck_match is None:
                            skip_row = {
                                **item.to_dict(),
                                "skip_reason": "deck_template_not_found",
                            }
                            skipped.append(skip_row)
                            _append_manifest_row(
                                manifest_path, {"status": "skipped", **skip_row}
                            )
                            continue
                        if hasattr(deck_match, "deck"):
                            item.deck = deck_match.deck
                            item.deck_source = (
                                f"deck_template_match(conf={deck_match.confidence:.2f},"
                                f"{deck_match.matched_frames}/{deck_match.sampled_frames})"
                            )
                        else:
                            item.deck = deck_match
                            item.deck_source = "deck_template_match"
                        if item.seed:
                            item.skip_reason = None
                            item.ready = True

                    if not item.seed and cfg.allow_ocr_seed_fallback:
                        ocr_res = _try_ocr_seed_fallback(vpath, cfg)
                        if ocr_res is None:
                            skip_row = {**item.to_dict(), "skip_reason": "ocr_seed_not_found"}
                            skipped.append(skip_row)
                            _append_manifest_row(manifest_path, {"status": "skipped", **skip_row})
                            continue
                        item.seed, conf = ocr_res
                        item.seed_source = f"seed_ocr(conf={conf:.2f})"
                        item.skip_reason = None
                        item.ready = True

                    stats = processor(vpath, item, cfg)
                    ok_row = {
                        **item.to_dict(),
                        "download_sec": round(dl_sec, 2),
                        **stats,
                    }
                    processed.append(ok_row)
                    _append_manifest_row(manifest_path, {"status": "processed", **ok_row})
                except Exception as exc:
                    fail_row = {
                        **item.to_dict(),
                        "stage": "processing",
                        "download_sec": round(dl_sec, 2),
                        "error": str(exc),
                    }
                    with lock:
                        failed.append(fail_row)
                        _append_manifest_row(manifest_path, {"status": "failed", **fail_row})
                finally:
                    _release_runtime_memory()
                    if not cfg.keep_videos and vpath.exists():
                        vpath.unlink(missing_ok=True)
                    slot_sem.release()
        except KeyboardInterrupt:
            stop_event.set()
            # Drain queued downloads so their files are cleaned up before exit if needed
            while not ready_queue.empty():
                leftover = ready_queue.get_nowait()
                if leftover is not None:
                    _, lvpath, _ = leftover
                    if not cfg.keep_videos and lvpath.exists():
                        lvpath.unlink(missing_ok=True)
                    slot_sem.release()
        finally:
            stop_event.set()
            disp_thread.join(timeout=10.0)
            if threading.current_thread() is threading.main_thread():
                signal.signal(signal.SIGINT, prev_sigint)

    ts_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = reports_dir / f"batch_report_{ts_tag}.json"
    cfg_dict = asdict(cfg)
    cfg_dict["work_dir"] = str(cfg.work_dir)
    report: dict[str, Any] = {
        "generated_at_utc": ts_tag,
        "elapsed_sec": round(time.perf_counter() - t_start, 2),
        "config": cfg_dict,
        "counts": {
            "total": len(fetched),
            "eligible": len(eligible),
            "processed": len(processed),
            "skipped": len(skipped),
            "failed": len(failed),
        },
        "eligible_preview": [m.to_dict() for m in eligible] if cfg.dry_run else [],
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "report_path": str(report_path),
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    _print_batch_summary(report)
    return report
