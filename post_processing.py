"""
Methods used to continue modifying and evaluating data after the
initial first-pass producing the json mapping frames to button
presses.
"""

import json
from itertools import count
from pprint import pp as pprint


from config import configure_logging, get_config

from game.balatro_models import ActionType, Location


def remove_duplicates(d: dict[str, int]) -> dict:
    new_dict: dict[str, int] = {}
    prev_action = None

    # reversed because we want the LAST frame, i.e., frame where
    # user lifted up their mouse button to trigger the button press
    for frame, action in reversed(d.items()):
        if prev_action != action:
            new_dict[frame] = action
        prev_action = action

    return dict(reversed(new_dict.items()))


def process_actions(d: dict[str, dict[str, int]]):
    """Takes a dict of action frames after duplicates have been cleaned."""
    A = ActionType

    key = d["key"]
    frames = d["frames"]
    inv_key = {v: k for k, v in key.items()}

    # Frame offsets relative to the logged button press frame.
    # A negative offset samples frames slightly before the click (so items haven't started dissolving/flying).
    # A positive offset samples frames after animations have finished settling.
    FRAME_OFFSETS = {
        # Sort operations sample pre-sort selection and post-sort settled cards
        "post_sort": 15,
        "pre_sort": -15,
        # Action-specific offsets
        A.PLAY_HAND: 0,
        A.DISCARD: 0,
        A.BUY_CARD: -2,
        A.BUY_AND_USE: -2,
        A.OPEN_PACK: -2,
        A.USE_IN_PACK: -2,
        A.SKIP_PACK: -2,
        A.REDEEM_VOUCHER: -2,
        A.REROLL_SHOP: -2,
        A.NEXT_ROUND: -2,
        A.SELECT_BLIND: -2,
        A.SKIP_BLIND: -2,
        A.REROLL_BOSS: -2,
        A.SELL_CARD: -2,
        A.USE_CONSUMABLE: -2,
        A.END_ROUND: -2,
        A.CASH_OUT: -2,
        A.PLAY_GAME: -2,
    }

    # Actions that strictly belong to / define a specific location
    LOCATION_DEFINING_MAP = {
        # Select screen actions
        A.PLAY_GAME: Location.SELECT,
        A.SELECT_BLIND: Location.SELECT,
        A.SKIP_BLIND: Location.SELECT,
        A.REROLL_BOSS: Location.SELECT,
        # Shop actions
        A.REROLL_SHOP: Location.SHOP,
        A.NEXT_ROUND: Location.SHOP,
        A.BUY_CARD: Location.SHOP,
        A.BUY_AND_USE: Location.SHOP,
        A.OPEN_PACK: Location.SHOP,
        A.REDEEM_VOUCHER: Location.SHOP,
        # Pack actions
        A.SKIP_PACK: Location.PACK,
        A.USE_IN_PACK: Location.PACK,
        # Blind play actions
        A.PLAY_HAND: Location.BLIND,
        A.DISCARD: Location.BLIND,
        A.SORT_RANK: Location.BLIND,
        A.SORT_SUIT: Location.BLIND,
        A.END_ROUND: Location.BLIND,
        # Cash out screen
        A.CASH_OUT: Location.CASH_OUT,
    }

    action_list = [ActionType(inv_key[a]) for a in frames.values()]

    def peek_next_location(start_idx: int) -> Location | None:
        """Scans forward to find the location implied by the next decisive action."""
        for act in action_list[start_idx:]:
            if act in LOCATION_DEFINING_MAP:
                return LOCATION_DEFINING_MAP[act]
        return None

    # Initial location: find the first location-defining action
    location = peek_next_location(0) or Location.SELECT

    def get_next_location(action: ActionType, current_idx: int) -> Location | None:
        """
        Returns location for subsequent actions;
        Returns None if given action does not change location
        """
        match action:
            case A.NEXT_ROUND:
                return Location.SELECT
            case A.SELECT_BLIND:
                return Location.BLIND
            case A.END_ROUND:
                return Location.CASH_OUT
            case A.CASH_OUT:
                return Location.SHOP
            case A.OPEN_PACK:
                return Location.PACK
            case A.SKIP_BLIND:
                # If skipped for a pack, next decisive action is in-pack; otherwise stays on select
                next_loc = peek_next_location(current_idx + 1)
                return Location.PACK if next_loc == Location.PACK else Location.SELECT
            case A.SKIP_PACK:
                # Return to wherever the pack was opened from (SHOP or SELECT)
                return peek_next_location(current_idx + 1) or Location.SELECT
            case A.USE_IN_PACK:
                # If choices remain in a mega pack, next decisive action is still PACK; otherwise returns to SHOP/SELECT
                next_loc = peek_next_location(current_idx + 1)
                return Location.PACK if next_loc == Location.PACK else next_loc
            case _:
                return None

    # list of frames on which to query various CV models
    # mapping target_frame_id -> ActionType

    # queries card classifier on cards in hand
    playing_card_cv_queue: dict[int, ActionType] = {}

    # queries shop classifier on joker bar (only when full shop / pack is not evaluated)
    joker_cv_queue: dict[int, ActionType] = {}

    # queries shop classifier on full shop shelves (evaluates shop, vouchers, boosters, jokers, consumables)
    shop_cv_queue: dict[int, ActionType] = {}

    # queries shop classifier on pack choices (evaluates pack items, jokers, consumables)
    pack_cv_queue: dict[int, ActionType] = {}

    for idx, (frame_str, action_val) in enumerate(frames.items()):
        str_action = inv_key[action_val]
        action = ActionType(str_action)
        frame_id = int(frame_str)

        # 1. Self-healing location anchor:
        # If this action definitively belongs to a specific screen, snap to it immediately.
        # This auto-corrects any missed transitional button clicks (e.g. missed "Next Round").
        if action in LOCATION_DEFINING_MAP:
            location = LOCATION_DEFINING_MAP[action]

        # Calculate action-specific frame with offset (clamped to 0)
        action_offset = FRAME_OFFSETS.get(action, -2)
        target_frame = max(0, frame_id + action_offset)

        # 2. Queue appropriate spatial CV models based on action and ambient location
        match action:
            # --- Hand Playing Card Actions ---
            case A.PLAY_HAND | A.DISCARD:
                playing_card_cv_queue[target_frame] = action

            case A.SORT_RANK | A.SORT_SUIT:
                playing_card_cv_queue[max(0, frame_id + FRAME_OFFSETS["pre_sort"])] = (
                    action
                )
                playing_card_cv_queue[frame_id + FRAME_OFFSETS["post_sort"]] = action

            # --- Shop Shelf Actions ---
            case (
                A.BUY_CARD
                | A.BUY_AND_USE
                | A.OPEN_PACK
                | A.REDEEM_VOUCHER
                | A.REROLL_SHOP
            ):
                shop_cv_queue[target_frame] = action

            case A.NEXT_ROUND:
                # Capture final shop offerings before leaving the shop
                shop_cv_queue[target_frame] = action

            # --- Booster Pack Actions ---
            case A.USE_IN_PACK:
                pack_cv_queue[target_frame] = action
                # In Arcana or Spectral packs, hand cards are visible underneath and can be targeted
                if location == Location.PACK:
                    playing_card_cv_queue[target_frame] = action

            case A.SKIP_PACK:
                pack_cv_queue[target_frame] = action

            # --- Location-Neutral Actions (SELL_CARD, USE_CONSUMABLE) ---
            case A.SELL_CARD | A.USE_CONSUMABLE:
                if location == Location.BLIND:
                    playing_card_cv_queue[target_frame] = action
                elif location == Location.SHOP:
                    shop_cv_queue[target_frame] = action
                elif location == Location.PACK:
                    pack_cv_queue[target_frame] = action
                    playing_card_cv_queue[target_frame] = action

            # --- Screens with no hand/shop items (SELECT, CASH_OUT, END_ROUND, PLAY_GAME) ---
            case (
                A.SELECT_BLIND
                | A.SKIP_BLIND
                | A.REROLL_BOSS
                | A.CASH_OUT
                | A.END_ROUND
                | A.PLAY_GAME
            ):
                pass

        # 3. Queue joker bar evaluation
        # Note: shop_cv_queue and pack_cv_queue already evaluate the joker bar,
        # so only queue to joker_cv_queue when those full-screen classifiers are not running on this frame.
        if action not in {A.SORT_RANK, A.SORT_SUIT}:
            if target_frame not in shop_cv_queue and target_frame not in pack_cv_queue:
                joker_cv_queue[target_frame] = action

        # 4. Advance location state for subsequent iterations
        new_loc = get_next_location(action, idx)
        if new_loc is not None:
            location = new_loc


if __name__ == "__main__":
    configure_logging()

    config = get_config()

    filename = "bu_full_1791436070.json"
    path = rf"{config.output_dir}\{filename}"

    with open(path) as f:
        data = json.load(f)
        f.close()

    cleaned = remove_duplicates(data["frames"])

    pprint(cleaned)
