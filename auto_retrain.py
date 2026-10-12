"""
Wait for Balatro Dataset Generation and Auto-Launch Retraining
==============================================================
Monitors `balatro_mod/dataset/images` until 5,000 frames are generated,
then automatically executes the end-to-end retraining pipeline:
    python classifiers/card_classifier/train.py --sync --stage all --epochs-detector 25 --epochs-classifier 25

Usage:
    python auto_retrain.py
    python auto_retrain.py --target 10000 --interval 5
"""

import sys
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent
MOD_IMAGES_DIR = ROOT_DIR / "balatro_mod" / "dataset" / "images"
TRAIN_SCRIPT = ROOT_DIR / "classifiers" / "card_classifier" / "train.py"


def format_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def wait_for_images(target_count: int = 10000, check_interval: int = 5):
    print("\n" + "=" * 70)
    print("BALATRO DATASET MONITOR & AUTO-RETRAIN ORCHESTRATOR")
    print("=" * 70)
    print(f"Monitoring Directory : {MOD_IMAGES_DIR}")
    print(f"Target Frame Count   : {target_count:,} images")
    print(f"Polling Interval     : Every {check_interval} seconds")
    print(f"Start Time           : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")

    MOD_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    last_count = -1
    consecutive_done_checks = 0

    while True:
        # Count .png and .jpg files
        images = list(MOD_IMAGES_DIR.glob("*.png")) + list(MOD_IMAGES_DIR.glob("*.jpg"))
        count = len(images)
        elapsed = time.time() - start_time

        if count != last_count:
            pct = (count / target_count) * 100.0 if target_count > 0 else 0.0
            rate = count / elapsed if elapsed > 0 else 0.0
            remaining_secs = (target_count - count) / rate if rate > 0 and count < target_count else 0
            eta_str = format_elapsed(remaining_secs) if count < target_count else "Complete"

            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"Progress: {count:4d} / {target_count} ({pct:5.1f}%) | "
                f"Speed: {rate:4.1f} img/s | "
                f"Elapsed: {format_elapsed(elapsed)} | "
                f"ETA: {eta_str}"
            )
            last_count = count

        if count >= target_count:
            # Let it stabilize for one interval to ensure the final frame has finished writing
            consecutive_done_checks += 1
            if consecutive_done_checks >= 2:
                print(f"\n[TARGET REACHED] Successfully detected {count:,} images in {format_elapsed(elapsed)}!")
                break
        else:
            consecutive_done_checks = 0

        time.sleep(check_interval)


def launch_training():
    print("\n" + "=" * 70)
    print("LAUNCHING FULL TWO-STAGE RETRAINING PIPELINE")
    print("=" * 70)
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--sync",
        "--stage", "all",
        "--epochs-detector", "12",
        "--epochs-classifier", "25",
        "--batch-classifier", "128",
        "--workers", "4"
    ]
    print(f"Executing: {' '.join(cmd)}\n" + "=" * 70 + "\n")

    # Run and stream output directly to terminal
    proc = subprocess.run(cmd, cwd=str(TRAIN_SCRIPT.parent))
    if proc.returncode == 0:
        print("\n" + "=" * 70)
        print("ALL TRAINING STAGES COMPLETED SUCCESSFULLY!")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70 + "\n")
    else:
        print(f"\n[ERROR] Training process exited with return code: {proc.returncode}")
        sys.exit(proc.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wait for dataset generation and trigger retraining")
    parser.add_argument("--target", type=int, default=10000, help="Target image count (default: 10000)")
    parser.add_argument("--interval", type=int, default=5, help="Poll interval in seconds (default: 5)")
    args = parser.parse_args()

    wait_for_images(target_count=args.target, check_interval=args.interval)
    launch_training()
