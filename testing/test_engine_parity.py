"""
Integration test script: Validates Python GameState engine parity against Lua telemetry log,
simulating a video pipeline with perfect CV/OCR observations and zero ground-truth leakage
across closed-loop state transitions.

Usage:
    python testing/test_engine_parity.py [--telemetry path/to/run_telemetry.jsonl] [--verbose]
"""

import argparse
import os
from pathlib import Path
import sys

# Ensure 'game' and repo root are in pythonpath
REPO_ROOT = Path(__file__).resolve().parent.parent
GAME_DIR = REPO_ROOT / "game"
if str(GAME_DIR) not in sys.path:
    sys.path.insert(0, str(GAME_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from testing.test_engine import (
        run_synthetic_test,
        test_trajectory_parity as run_parity_check,
    )
except ImportError:
    from test_engine import (
        run_synthetic_test,
        test_trajectory_parity as run_parity_check,
    )


def find_telemetry_file(explicit_path: Path | None = None) -> Path | None:
    if explicit_path and explicit_path.exists() and explicit_path.stat().st_size > 0:
        return explicit_path

    candidates = [
        REPO_ROOT / "run_telemetry.jsonl",
        REPO_ROOT / "output" / "run_telemetry.jsonl",
    ]
    appdata = os.environ.get("APPDATA")
    if appdata:
        candidates.append(Path(appdata) / "Balatro" / "run_telemetry.jsonl")
        candidates.append(Path(appdata) / "Balatro" / "Mods" / "run_telemetry.jsonl")
    candidates.append(Path.home() / "AppData" / "Roaming" / "Balatro" / "run_telemetry.jsonl")

    non_empty = [p for p in candidates if p.exists() and p.stat().st_size > 0]
    if non_empty:
        return max(non_empty, key=lambda p: p.stat().st_mtime)

    if explicit_path and explicit_path.exists():
        return explicit_path
    for p in candidates:
        if p.exists():
            return p

    return None


def main(telemetry_path: Path | None = None, verbose: bool = False) -> int:
    target_path = find_telemetry_file(telemetry_path)
    if target_path is None or not target_path.exists():
        print("No telemetry file found at default locations or %APPDATA%\\Balatro\\run_telemetry.jsonl.")
        print("Running synthetic integration test instead...")
        run_synthetic_test()
        return 0

    if target_path.stat().st_size == 0:
        print(f"Found telemetry file at: {target_path}")
        print("Notice: The telemetry log is currently 0 bytes (no gameplay actions recorded yet).")
        print("\nHow to record telemetry:")
        print("  1. Launch Balatro (with Steamodded + GameState Dataset CV mod installed).")
        print("  2. Start a run (play hands, discard, or interact with shop/blinds).")
        print("  3. The mod automatically records every action to run_telemetry.jsonl.")
        print("  4. Rerun this script to verify parity against your live run!\n")
        print("Running synthetic integration test in the meantime...")
        run_synthetic_test()
        return 0

    errors = run_parity_check(target_path, verbose=verbose)
    print(f"\nParity Test Completed on {target_path}. Discrepancies: {errors}")
    if errors == 0:
        print("PERFECT PARITY ACHIEVED!")
    return errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    sys.exit(main(args.telemetry, verbose=args.verbose))
