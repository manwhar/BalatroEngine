"""
Wait for Balatro Dataset Generation and Auto-Launch Retraining
==============================================================
Monitors dataset generation until target image count is reached,
then automatically launches the end-to-end retraining pipeline:
    python classifiers/{shop_classifier|card_classifier}/train.py --sync --stage all

Supported Models:
    --model shop   (default: monitors dataset_shop/images, trains shop_classifier)
    --model card   (monitors dataset/images, trains card_classifier)
    --model auto   (detects active generation directory automatically)

Usage:
    python auto_retrain.py
    python auto_retrain.py --model shop --target 10000 --interval 5
    python auto_retrain.py --model card --target 10000
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent

APPDATA_MOD_DIR = Path(os.path.expandvars(r"%APPDATA%\Balatro\Mods\gamestate_dataset_cv"))
REPO_MOD_DIR = ROOT_DIR / "balatro_mod"


def format_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def resolve_images_directory(model_type: str = "shop") -> tuple[Path, str, Path]:
    """
    Finds the active dataset images directory, model name, and train script.
    Checks both AppData roaming directory and repo directory.
    """
    if model_type == "auto":
        # Check which directory has newly growing files
        shop_appdata = APPDATA_MOD_DIR / "dataset_shop" / "images"
        shop_repo = REPO_MOD_DIR / "dataset_shop" / "images"
        card_appdata = APPDATA_MOD_DIR / "dataset" / "images"
        card_repo = REPO_MOD_DIR / "dataset" / "images"

        shop_count = (len(list(shop_appdata.glob("*.png"))) if shop_appdata.exists() else 0) + \
                     (len(list(shop_repo.glob("*.png"))) if shop_repo.exists() else 0)
        card_count = (len(list(card_appdata.glob("*.png"))) if card_appdata.exists() else 0) + \
                     (len(list(card_repo.glob("*.png"))) if card_repo.exists() else 0)

        model_type = "shop" if shop_count >= card_count else "card"

    if model_type == "shop":
        folder_name = "dataset_shop"
        train_script = ROOT_DIR / "classifiers" / "shop_classifier" / "train.py"
    else:
        folder_name = "dataset"
        train_script = ROOT_DIR / "classifiers" / "card_classifier" / "train.py"

    appdata_candidate = APPDATA_MOD_DIR / folder_name / "images"
    repo_candidate = REPO_MOD_DIR / folder_name / "images"

    # Prefer AppData if files exist there, otherwise check repo
    if appdata_candidate.exists() and len(list(appdata_candidate.glob("*.png")) + list(appdata_candidate.glob("*.jpg"))) > 0:
        active_dir = appdata_candidate
    elif repo_candidate.exists() and len(list(repo_candidate.glob("*.png")) + list(repo_candidate.glob("*.jpg"))) > 0:
        active_dir = repo_candidate
    else:
        # Default to AppData (where Love2D writes by default)
        active_dir = appdata_candidate

    return active_dir, model_type, train_script


import re

FRAME_REGEX = re.compile(r"frame_(\d+)", re.IGNORECASE)


def get_dataset_counts(directory: Path) -> tuple[int, int]:
    """
    Returns (unique_frames_count, total_images_count).
    Extracts frame numbers using regex 'frame_(\\d+)' so that multiple crop files
    per frame (e.g. joker, consumables, shop/pack) map to 1 unique frame.
    """
    if not directory.exists():
        return 0, 0
    frame_ids = set()
    total_images = 0
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg"):
            total_images += 1
            m = FRAME_REGEX.search(p.name)
            if m:
                frame_ids.add(int(m.group(1)))
            else:
                frame_ids.add(p.name)
    return len(frame_ids), total_images


def wait_for_images(images_dir: Path, model_type: str, target_count: int = 10000, check_interval: int = 5):
    print("\n" + "=" * 70)
    print("BALATRO DATASET MONITOR & AUTO-RETRAIN ORCHESTRATOR")
    print("=" * 70)
    print(f"Target Model         : {model_type.upper()} CLASSIFIER")
    print(f"Monitoring Directory : {images_dir}")
    print(f"Target Frame Count   : {target_count:,} frames")
    print(f"Polling Interval     : Every {check_interval} seconds")
    print(f"Start Time           : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")

    images_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    last_frames = -1
    consecutive_done_checks = 0

    while True:
        # Count unique frames and total images in monitored directory
        frames, total_imgs = get_dataset_counts(images_dir)

        # If AppData is empty but repo has files (or vice versa), check both
        alt_dir = REPO_MOD_DIR / images_dir.parent.name / "images" if "AppData" in str(images_dir) else APPDATA_MOD_DIR / images_dir.parent.name / "images"
        alt_frames, alt_imgs = get_dataset_counts(alt_dir)

        total_frames = max(frames, alt_frames)
        total_crops = max(total_imgs, alt_imgs)

        elapsed = time.time() - start_time

        if total_frames != last_frames:
            pct = (total_frames / target_count) * 100.0 if target_count > 0 else 0.0
            rate = total_frames / elapsed if elapsed > 0 else 0.0
            remaining_secs = (target_count - total_frames) / rate if rate > 0 and total_frames < target_count else 0
            eta_str = format_elapsed(remaining_secs) if total_frames < target_count else "Complete"

            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"Frames: {total_frames:5,d} / {target_count:,d} ({pct:5.1f}%) | "
                f"Crops: {total_crops:5,d} | "
                f"Speed: {rate:4.1f} frames/s | "
                f"Elapsed: {format_elapsed(elapsed)} | "
                f"ETA: {eta_str}"
            )
            last_frames = total_frames

        if total_frames >= target_count:
            consecutive_done_checks += 1
            if consecutive_done_checks >= 2:
                print(f"\n[TARGET REACHED] Generated {total_frames:,} frames ({total_crops:,} crop images) in {format_elapsed(elapsed)}!")
                break
        else:
            consecutive_done_checks = 0

        time.sleep(check_interval)


def launch_training(train_script: Path, model_type: str, epochs_det: int = 15, epochs_cls: int = 25):
    print("\n" + "=" * 70)
    print(f"LAUNCHING {model_type.upper()} CLASSIFIER RETRAINING PIPELINE")
    print("=" * 70)
    cmd = [
        sys.executable,
        str(train_script),
        "--sync",
        "--stage", "all",
        "--epochs-detector", str(epochs_det),
        "--epochs-classifier", str(epochs_cls),
        "--batch-classifier", "256",
        "--workers", "4"
    ]
    print(f"Executing: {' '.join(cmd)}\n" + "=" * 70 + "\n")

    # Run and stream output directly to terminal
    proc = subprocess.run(cmd, cwd=str(train_script.parent))
    if proc.returncode == 0:
        print("\n" + "=" * 70)
        print(f"ALL {model_type.upper()} TRAINING STAGES COMPLETED SUCCESSFULLY!")
        print(f"Weights saved in: {train_script.parent / 'weights'}")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70 + "\n")
    else:
        print(f"\n[ERROR] Training process exited with return code: {proc.returncode}")
        sys.exit(proc.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wait for dataset generation and trigger retraining")
    parser.add_argument("--model", type=str, default="shop", choices=["shop", "card", "auto"], help="Model to train (default: shop)")
    parser.add_argument("--target", type=int, default=10000, help="Target image count (default: 10000)")
    parser.add_argument("--interval", type=int, default=5, help="Poll interval in seconds (default: 5)")
    parser.add_argument("--epochs-detector", type=int, default=15, help="Stage 1 detector epochs (default: 15)")
    parser.add_argument("--epochs-classifier", type=int, default=15, help="Stage 2 classifier epochs (default: 15)")
    args = parser.parse_args()

    images_dir, model_type, train_script = resolve_images_directory(args.model)
    wait_for_images(images_dir, model_type, target_count=args.target, check_interval=args.interval)
    launch_training(train_script, model_type, epochs_det=args.epochs_detector, epochs_cls=args.epochs_classifier)
