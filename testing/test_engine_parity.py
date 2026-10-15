"""
Integration test script: Validates Python GameState engine parity against Lua telemetry log.

Usage:
    python testing/test_engine_parity.py [--telemetry path/to/run_telemetry.jsonl]
"""

import argparse
import json
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
    import game.balatro_models as bm
    import game.gamestate as gs
except ImportError:
    import balatro_models as bm
    import gamestate as gs

try:
    from testing.test_engine import TestEngine
except ImportError:
    from test_engine import TestEngine


def parse_lua_action(action_name: str, meta: dict, frame_idx: int, timestamp: float) -> bm.Action:
    action_map = {
        "Play Hand": bm.ActionType.PLAY_HAND,
        "Discard": bm.ActionType.DISCARD,
        "Reroll Shop": bm.ActionType.REROLL_SHOP,
        "Cash Out": bm.ActionType.CASH_OUT,
        "Buy Card": bm.ActionType.BUY_CARD,
        "Buy and Use": bm.ActionType.BUY_AND_USE,
        "Sell Card": bm.ActionType.SELL_CARD,
        "Select Blind": bm.ActionType.SELECT_BLIND,
        "Skip Blind": bm.ActionType.SKIP_BLIND,
    }
    act_type = action_map.get(action_name, bm.ActionType.PLAY_HAND)
    return bm.Action(
        type=act_type,
        frame_idx=frame_idx,
        timestamp=timestamp,
        metadata=meta,
    )


def test_trajectory_parity(telemetry_path: Path):
    if not telemetry_path.exists():
        print(f"No telemetry file found at {telemetry_path}. Generating dummy test...")
        return run_synthetic_test()

    print(f"Loading Lua telemetry from {telemetry_path}...")
    with open(telemetry_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        print("Telemetry file is empty.")
        return

    first_entry = json.loads(lines[0])
    first_state_data = first_entry["state"]

    # Initialize Python GameState from first Lua snapshot
    sim_state = gs.GameState(
        step_idx=0,
        ante=first_state_data.get("ante", 1),
        round=first_state_data.get("round", 1),
        stats=bm.RunStats(
            money=first_state_data.get("money", 4),
            reroll_cost=first_state_data.get("reroll_cost", 5),
            hands_left=first_state_data.get("hands_left", 4),
            discards_left=first_state_data.get("discards_left", 3),
        ),
    )

    trajectory = gs.RunTrajectory(keyframe_interval=10)
    errors = 0

    for i, line in enumerate(lines):
        entry = json.loads(line)
        act_name = entry.get("action", "Play Hand")
        meta = entry.get("action_metadata", {})
        lua_state = entry.get("state", {})
        step = entry.get("step", i)
        ts = entry.get("timestamp", 0.0)

        action = parse_lua_action(act_name, meta, frame_idx=step, timestamp=ts)
        sim_state, delta = gs.step_simulation(sim_state, action)
        trajectory.record_step(sim_state, action)

        # Verification checks
        if "money" in lua_state and sim_state.stats.money != lua_state["money"]:
            print(f"[Mismatch Step {step} '{act_name}'] Money: Python={sim_state.stats.money} vs Lua={lua_state['money']}")
            errors += 1

        if "reroll_cost" in lua_state and act_name == "Reroll Shop":
            if sim_state.stats.reroll_cost != lua_state["reroll_cost"]:
                print(f"[Mismatch Step {step} '{act_name}'] Reroll Cost: Python={sim_state.stats.reroll_cost} vs Lua={lua_state['reroll_cost']}")
                errors += 1

        if "deck_cards_count" in lua_state and sim_state.deck_cards_count != lua_state["deck_cards_count"]:
            print(f"[Mismatch Step {step} '{act_name}'] Deck Cards Count: Python={sim_state.deck_cards_count} vs Lua={lua_state['deck_cards_count']}")
            errors += 1

        if "discard_pile" in lua_state and len(sim_state.discard_pile) != len(lua_state["discard_pile"]):
            print(f"[Mismatch Step {step} '{act_name}'] Discard Pile Count: Python={len(sim_state.discard_pile)} vs Lua={len(lua_state['discard_pile'])}")
            errors += 1

    print(f"\nParity Test Completed. Total Steps: {len(lines)}, Discrepancies: {errors}")
    if errors == 0:
        print("PERFECT PARITY ACHIEVED!")


def run_synthetic_test():
    print("Running synthetic integration test...")
    sim_state = gs.GameState(
        stats=bm.RunStats(money=15, reroll_cost=5),
        inventory=bm.Inventory(
            jokers=[bm.Joker(name=bm.JokerName.GIFT_CARD), bm.Joker(name=bm.JokerName.JOKER)]
        ),
    )
    traj = gs.RunTrajectory(keyframe_interval=3)

    # 1. Reroll
    sim_state, _ = gs.step_simulation(sim_state, bm.Action(type=bm.ActionType.REROLL_SHOP))
    traj.record_step(sim_state)
    assert sim_state.stats.money == 10
    assert sim_state.stats.reroll_cost == 6

    # 2. Cash out (triggers Gift Card)
    sim_state, _ = gs.step_simulation(sim_state, bm.Action(type=bm.ActionType.CASH_OUT))
    traj.record_step(sim_state)
    assert sim_state.inventory.jokers[0].extra_value == 1
    assert sim_state.inventory.jokers[1].extra_value == 1
    assert sim_state.stats.reroll_cost == 5  # Resets in shop

    # 3. Test trajectory seek
    recovered_s0 = traj.get_state(0)
    assert recovered_s0.stats.money == 10
    print("Synthetic integration test passed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry", type=Path, default=Path("run_telemetry.jsonl"))
    args = parser.parse_args()
    test_trajectory_parity(args.telemetry)
