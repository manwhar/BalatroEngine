"""
Post-processing and video-to-GameState trajectory orchestration pipeline.

Converts raw template-matching click logs (`main.py` output JSON) plus gameplay
video frames into batched CV observations and a deterministic `(GameState, RunTrajectory)`
sequence.

Scope notes:
- Starting mid-run is out of scope; replay reconstruction assumes a fresh run
  starting at Ante 1, Small Blind (`Location.SELECT`).
- Exact chip/mult round score calculation is intentionally omitted; `current_chips`
  and HUD counters (money, hands, discards, round, ante, reroll cost) are left
  for OCR integration hooks.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pp as pprint
from typing import Any, Sequence

from config import configure_logging, get_config
from game.balatro_models import (
    Action,
    ActionType,
    BlindName,
    BoosterPack,
    BoosterSize,
    BoosterType,
    Card,
    CardBounds,
    Deck,
    HandCard,
    Joker,
    Location,
    Planet,
    PlayingCard,
    Rank,
    Spectral,
    Stake,
    Tarot,
    Voucher,
)
from game.engine.cv_stubs import (
    _convert_card_result_to_hand_card,
    _convert_items_of_type,
)
from game.engine.rules import on_action
from game.gamestate import GameState, RunTrajectory, _cards_match


ACTION_REMAP: dict[str, ActionType] = {
    "Select Card": ActionType.USE_IN_PACK,
}

NON_GAMEPLAY_ACTIONS: set[str] = {
    ActionType.OPEN_OPTIONS.value,
    ActionType.RUN_INFO.value,
    ActionType.PLAY_GAME.value,
}

# Frame offsets relative to the logged button press frame.
# Negative offsets sample just before the click so dissolving/flying animations have not started.
# Positive offsets sample after animations (like sort) have settled.
FRAME_OFFSETS: dict[str | ActionType, int] = {
    "post_sort": 15,
    "pre_sort": -15,
    ActionType.PLAY_HAND: 0,
    ActionType.DISCARD: 0,
    ActionType.BUY_CARD: 0,
    # HUMAN TODO: manually verify remaining offsets below (above verified empirically)
    ActionType.BUY_AND_USE: -2,
    ActionType.OPEN_PACK: -2,
    ActionType.USE_IN_PACK: -2,
    ActionType.SKIP_PACK: -2,
    ActionType.REDEEM_VOUCHER: -2,
    ActionType.REROLL_SHOP: -2,
    ActionType.NEXT_ROUND: -2,
    ActionType.SELECT_BLIND: -2,
    ActionType.SKIP_BLIND: -2,
    ActionType.REROLL_BOSS: -2,
    ActionType.SELL_CARD: -2,
    ActionType.USE_CONSUMABLE: -2,
    ActionType.END_ROUND: -2,
    ActionType.CASH_OUT: -2,
}

# Actions that strictly belong to / define a specific location.
# NOTE: Location.CASH_OUT is only used internally by post_processing.py to separate
# the post-blind reward screen from active blind play when scheduling CV queues.
# When translating into GameState calls, use Location.BLIND with round_won=True.
LOCATION_DEFINING_MAP: dict[ActionType, Location] = {
    ActionType.SELECT_BLIND: Location.SELECT,
    ActionType.SKIP_BLIND: Location.SELECT,
    ActionType.REROLL_BOSS: Location.SELECT,
    ActionType.REROLL_SHOP: Location.SHOP,
    ActionType.NEXT_ROUND: Location.SHOP,
    ActionType.BUY_CARD: Location.SHOP,
    ActionType.BUY_AND_USE: Location.SHOP,
    ActionType.OPEN_PACK: Location.SHOP,
    ActionType.REDEEM_VOUCHER: Location.SHOP,
    ActionType.SKIP_PACK: Location.PACK,
    ActionType.USE_IN_PACK: Location.PACK,
    ActionType.PLAY_HAND: Location.BLIND,
    ActionType.DISCARD: Location.BLIND,
    ActionType.SORT_RANK: Location.BLIND,
    ActionType.SORT_SUIT: Location.BLIND,
    ActionType.END_ROUND: Location.BLIND,
    ActionType.CASH_OUT: Location.CASH_OUT,
}


@dataclass(slots=True)
class ScheduledAction:
    """Represents one filtered gameplay click and the video frames scheduled for CV evaluation."""

    idx: int
    frame_id: int
    action_type: ActionType
    target_frame: int
    ambient_location: Location
    button_box: list[int] | None = None
    pre_sort_frame: int | None = None
    post_sort_frame: int | None = None
    pre_pack_location_hint: Location | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _extract_action_id(entry: int | dict) -> int:
    return entry["action"] if isinstance(entry, dict) else entry


def _extract_box(entry: int | dict) -> list[int] | None:
    return entry.get("box") if isinstance(entry, dict) else None


def _boxes_match(
    box_a: list[int] | None, box_b: list[int] | None, tol_px: int = 20
) -> bool:
    """Returns True if both boxes are within `tol_px` on every coordinate (or if either is missing)."""
    if box_a is None or box_b is None or len(box_a) != 4 or len(box_b) != 4:
        return True
    return max(abs(a - b) for a, b in zip(box_a, box_b)) <= tol_px


def remove_duplicates(
    d: dict[str, int | dict],
    *,
    max_frame_gap: int = 15,
) -> dict[str, int | dict]:
    """Collapses contiguous held button presses to the final click-release frame.

    Two detections of the same action and box are only merged if they occur within
    `max_frame_gap` frames of each other, so repeated distinct clicks (e.g. multiple
    Reroll Shop or consecutive Buy Card clicks at the same slot) are preserved.
    """
    new_dict: dict[str, int | dict] = {}
    prev_frame: int | None = None
    prev_action: int | None = None
    prev_box: list[int] | None = None

    # Reverse chronological order keeps the LAST frame (mouse release triggering the action)
    ordered_items = sorted(d.items(), key=lambda kv: int(kv[0]), reverse=True)
    for frame, entry in ordered_items:
        cur_frame = int(frame)
        action_id = _extract_action_id(entry)
        box = _extract_box(entry)
        is_same_hold = (
            prev_action == action_id
            and _boxes_match(prev_box, box)
            and prev_frame is not None
            and (prev_frame - cur_frame) <= max_frame_gap
        )
        if not is_same_hold:
            new_dict[frame] = entry
        prev_frame = cur_frame
        prev_action = action_id
        prev_box = box

    return dict(reversed(new_dict.items()))


def _resolve_action(raw_label: str) -> ActionType | None:
    if raw_label in NON_GAMEPLAY_ACTIONS:
        return None
    if raw_label in ACTION_REMAP:
        return ACTION_REMAP[raw_label]
    return ActionType(raw_label)


def plan_action_pipeline(d: dict[str, Any]) -> dict[str, Any]:
    """Parses cleaned action frames into location-aware CV queues and scheduled steps."""
    A = ActionType
    key = d["key"]
    frames = d["frames"]
    inv_key = {v: k for k, v in key.items()}

    parsed_frames: list[tuple[int, int | dict, ActionType]] = []
    for frame_str, frame_entry in frames.items():
        str_action = inv_key[_extract_action_id(frame_entry)]
        action = _resolve_action(str_action)
        if action is None:
            continue
        parsed_frames.append((int(frame_str), frame_entry, action))

    action_list = [act for _, _, act in parsed_frames]

    def peek_next_location(start_idx: int) -> Location | None:
        for act in action_list[start_idx:]:
            if act in LOCATION_DEFINING_MAP:
                return LOCATION_DEFINING_MAP[act]
        return None

    location = peek_next_location(0) or Location.SELECT

    def get_next_location(action: ActionType, current_idx: int) -> Location | None:
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
                next_loc = peek_next_location(current_idx + 1)
                return Location.PACK if next_loc == Location.PACK else Location.SELECT
            case A.SKIP_PACK:
                return peek_next_location(current_idx + 1) or Location.SELECT
            case A.USE_IN_PACK:
                next_loc = peek_next_location(current_idx + 1)
                return Location.PACK if next_loc == Location.PACK else next_loc
            case _:
                return None

    playing_card_cv_queue: dict[int, ActionType] = {}
    joker_cv_queue: dict[int, ActionType] = {}
    shop_cv_queue: dict[int, ActionType] = {}
    pack_cv_queue: dict[int, ActionType] = {}
    scheduled_actions: list[ScheduledAction] = []
    pre_pack_loc: Location | None = None

    for idx, (frame_id, frame_entry, action) in enumerate(parsed_frames):
        button_box = _extract_box(frame_entry)

        if action in LOCATION_DEFINING_MAP:
            location = LOCATION_DEFINING_MAP[action]

        action_offset = FRAME_OFFSETS.get(action, -2)
        target_frame = max(0, frame_id + action_offset)
        pre_sort_frame: int | None = None
        post_sort_frame: int | None = None

        match action:
            case A.PLAY_HAND | A.DISCARD:
                playing_card_cv_queue[target_frame] = action

            case A.SORT_RANK | A.SORT_SUIT:
                pre_sort_frame = max(0, frame_id + FRAME_OFFSETS["pre_sort"])
                post_sort_frame = max(0, frame_id + FRAME_OFFSETS["post_sort"])
                playing_card_cv_queue[pre_sort_frame] = action
                playing_card_cv_queue[post_sort_frame] = action

            case (
                A.BUY_CARD
                | A.BUY_AND_USE
                | A.OPEN_PACK
                | A.REDEEM_VOUCHER
                | A.REROLL_SHOP
                | A.NEXT_ROUND
            ):
                shop_cv_queue[target_frame] = action

            case A.USE_IN_PACK | A.SKIP_PACK:
                pack_cv_queue[target_frame] = action
                if action == A.USE_IN_PACK and location == Location.PACK:
                    playing_card_cv_queue[target_frame] = action

            case A.SELL_CARD | A.USE_CONSUMABLE:
                if location == Location.BLIND:
                    playing_card_cv_queue[target_frame] = action
                elif location == Location.SHOP:
                    shop_cv_queue[target_frame] = action
                elif location == Location.PACK:
                    pack_cv_queue[target_frame] = action
                    playing_card_cv_queue[target_frame] = action

            case (
                A.SELECT_BLIND
                | A.SKIP_BLIND
                | A.REROLL_BOSS
                | A.CASH_OUT
                | A.END_ROUND
            ):
                pass

        if action not in {A.SORT_RANK, A.SORT_SUIT}:
            if target_frame not in shop_cv_queue and target_frame not in pack_cv_queue:
                joker_cv_queue[target_frame] = action

        extra_meta = (
            dict(frame_entry.get("metadata", {}))
            if isinstance(frame_entry, dict) and isinstance(frame_entry.get("metadata"), dict)
            else {}
        )
        scheduled_actions.append(
            ScheduledAction(
                idx=idx,
                frame_id=frame_id,
                action_type=action,
                target_frame=target_frame,
                ambient_location=location,
                button_box=button_box,
                pre_sort_frame=pre_sort_frame,
                post_sort_frame=post_sort_frame,
                pre_pack_location_hint=pre_pack_loc if location == Location.PACK else None,
                metadata=extra_meta,
            )
        )

        new_loc = get_next_location(action, idx)
        if new_loc is not None:
            if new_loc == Location.PACK and location != Location.PACK:
                pre_pack_loc = Location.SHOP if action == A.OPEN_PACK else Location.SELECT
            elif new_loc != Location.PACK:
                pre_pack_loc = None
            location = new_loc

    return {
        "scheduled_actions": scheduled_actions,
        "playing_card_cv_queue": playing_card_cv_queue,
        "joker_cv_queue": joker_cv_queue,
        "shop_cv_queue": shop_cv_queue,
        "pack_cv_queue": pack_cv_queue,
    }


def _pack_shows_hand_cards(pack_objects: Sequence[Any]) -> bool | None:
    """Infers from converted pack items whether the open pack shows hand cards underneath.

    Returns True for Arcana/Spectral packs (Tarot or Spectral items present),
    False for Celestial/Buffoon/Standard packs, or None if pack items are empty.
    """
    if not pack_objects:
        return None
    if any(isinstance(item, (Tarot, Spectral)) for item in pack_objects):
        return True
    if any(isinstance(item, (Planet, Joker, PlayingCard)) for item in pack_objects):
        return False
    return None


def _infer_booster_pack_from_items(pack_objects: Sequence[Any]) -> BoosterPack:
    """Synthesizes a fallback BoosterPack descriptor when a pack opens without a seed."""
    count = len(pack_objects)
    if any(isinstance(item, Spectral) for item in pack_objects):
        b_type = BoosterType.SPECTRAL
        size = BoosterSize.MEGA if count >= 4 else BoosterSize.REGULAR
    elif any(isinstance(item, Planet) for item in pack_objects):
        b_type = BoosterType.CELESTIAL
        size = BoosterSize.MEGA if count >= 5 else BoosterSize.REGULAR
    elif any(isinstance(item, Joker) for item in pack_objects):
        b_type = BoosterType.BUFFOON
        size = BoosterSize.MEGA if count >= 4 else BoosterSize.REGULAR
    elif any(isinstance(item, PlayingCard) for item in pack_objects):
        b_type = BoosterType.STANDARD
        size = BoosterSize.MEGA if count >= 5 else BoosterSize.REGULAR
    else:
        b_type = BoosterType.ARCANA
        size = BoosterSize.MEGA if count >= 5 else BoosterSize.REGULAR

    return BoosterPack(
        booster_type=b_type,
        size=size,
        cost=0,
        choose=2 if size == BoosterSize.MEGA else 1,
        extra=max(count, 2),
    )


def match_button_to_item_index(
    button_box: Sequence[int] | None,
    items: Sequence[Any],
    *,
    attach: str = "bottom_center",
) -> int | None:
    """Matches a clicked button `[x1, y1, x2, y2]` box to the closest detected item box.

    In Balatro (`card.lua:Card:highlight` and `UI_definitions.lua:use_and_sell_buttons`):
    - Shop / voucher / booster / pack buttons align `'bmi'` (centered horizontally under card).
    - Joker / consumable `'Sell'` and `'Use'` buttons align `'cr'` (anchored on card's right edge).
    """
    if not items:
        return None
    if button_box is None or len(button_box) != 4:
        return 0 if len(items) == 1 else None

    btn_cx = (button_box[0] + button_box[2]) / 2.0
    btn_cy = (button_box[1] + button_box[3]) / 2.0

    best_idx: int | None = None
    best_dist = float("inf")
    for idx, item in enumerate(items):
        box = getattr(item, "box", None)
        if not box or len(box) != 4 or box == [0, 0, 0, 0]:
            continue
        if attach == "center_right":
            item_w = max(1.0, float(box[2] - box[0]))
            anchor_x = float(box[2]) + 0.1 * item_w
            anchor_y = (box[1] + box[3]) / 2.0
        else:
            anchor_x = (box[0] + box[2]) / 2.0
            anchor_y = (box[1] + box[3]) / 2.0

        dist = abs(btn_cx - anchor_x) + 0.25 * abs(btn_cy - anchor_y)
        if dist < best_dist:
            best_dist = dist
            best_idx = idx

    if best_idx is None and len(items) == 1:
        return 0
    return best_idx


def _run_batched_cv(
    scheduled_actions: Sequence[ScheduledAction],
    queues: dict[str, dict[int, ActionType]],
    frames_by_idx: dict[int, Any],
    *,
    fps: float,
    conf_threshold_hand: float,
    conf_threshold_shop: float,
    batch_size: int,
    device: str | None,
) -> tuple[dict[int, Any], dict[int, Any]]:
    """Runs shop/pack/joker-bar CV first, then selectively runs hand CV."""
    from classifiers.inference import batch_recognize_hands, batch_recognize_shops

    def _pack_frame_inputs(frame_ids: Sequence[int]) -> list[tuple[Any, int, float]]:
        return [
            (frames_by_idx[fid], fid, fid / fps)
            for fid in frame_ids
            if fid in frames_by_idx
        ]

    shop_res_by_frame: dict[int, Any] = {}

    shop_ids = sorted(queues["shop_cv_queue"].keys())
    if shop_ids:
        shop_outs = batch_recognize_shops(
            _pack_frame_inputs(shop_ids),
            mode="shop",
            conf_threshold=conf_threshold_shop,
            batch_size=batch_size,
            device=device,
        )
        for out in shop_outs:
            shop_res_by_frame[out.frame_index] = out

    pack_ids = sorted(queues["pack_cv_queue"].keys())
    if pack_ids:
        pack_outs = batch_recognize_shops(
            _pack_frame_inputs(pack_ids),
            mode="pack",
            conf_threshold=conf_threshold_shop,
            batch_size=batch_size,
            device=device,
        )
        for out in pack_outs:
            shop_res_by_frame.setdefault(out.frame_index, out)

    joker_ids = sorted(queues["joker_cv_queue"].keys())
    if joker_ids:
        bar_outs = batch_recognize_shops(
            _pack_frame_inputs(joker_ids),
            mode="bars",
            conf_threshold=conf_threshold_shop,
            batch_size=batch_size,
            device=device,
        )
        for out in bar_outs:
            shop_res_by_frame.setdefault(out.frame_index, out)

    # Dual-CV pack optimization: inspect recognized pack items first so Celestial/Buffoon/Standard
    # packs skip unnecessary hand card detection.
    pack_shows_hand_by_frame: dict[int, bool | None] = {}
    last_pack_hint: bool | None = None
    for sa in scheduled_actions:
        if sa.ambient_location != Location.PACK:
            last_pack_hint = None
            continue
        pack_out = shop_res_by_frame.get(sa.target_frame)
        converted_pack = (
            _convert_items_of_type(
                getattr(pack_out, "pack_items", None),
                (Joker, Tarot, Planet, Spectral, PlayingCard),
            )
            if pack_out is not None
            else []
        )
        hint = _pack_shows_hand_cards(converted_pack)
        if hint is not None:
            last_pack_hint = hint
        pack_shows_hand_by_frame[sa.target_frame] = (
            hint if hint is not None else last_pack_hint
        )

    filtered_hand_ids: list[int] = []
    for fid in sorted(queues["playing_card_cv_queue"].keys()):
        if fid in queues["pack_cv_queue"]:
            if pack_shows_hand_by_frame.get(fid) is False:
                continue
        filtered_hand_ids.append(fid)

    hand_res_by_frame: dict[int, Any] = {}
    if filtered_hand_ids:
        hand_outs = batch_recognize_hands(
            _pack_frame_inputs(filtered_hand_ids),
            conf_threshold=conf_threshold_hand,
            batch_size=batch_size,
            device=device,
        )
        for out in hand_outs:
            hand_res_by_frame[out.frame_index] = out

    return shop_res_by_frame, hand_res_by_frame


def _sync_bars_into_state(
    state: GameState, shop_out: Any, action: Action
) -> tuple[list[Any], list[Any]]:
    """Syncs Joker and Consumable bars from CV into `state` & populates `action.joker_order`."""
    raw_jokers = list(getattr(shop_out, "joker_bar", None) or [])
    obs_jokers = _convert_items_of_type(raw_jokers, Joker)

    if state.inventory and state.inventory.jokers:
        matched_ids = state.match_cv_joker_order(obs_jokers)
        if matched_ids:
            action.joker_order = matched_ids
        elif (
            not obs_jokers
            and state.current_blind == BlindName.AMBER_ACORN
            and any(j.facing_back for j in state.inventory.jokers)
        ):
            # Amber Acorn flips owned jokers face-down, hiding their faces from CV
            action.joker_order = [j.id for j in state.inventory.jokers if j.id != 0]
        else:
            state.reconcile_cv_jokers(obs_jokers, trigger_added_hooks=False)
            action.joker_order = [j.id for j in state.inventory.jokers]
    elif obs_jokers:
        state.reconcile_cv_jokers(obs_jokers, trigger_added_hooks=False)
        action.joker_order = [j.id for j in state.inventory.jokers]

    raw_cons = list(getattr(shop_out, "consumeables", None) or [])
    obs_cons = _convert_items_of_type(raw_cons, (Tarot, Planet, Spectral))
    if state.inventory and (obs_cons or state.inventory.consumables):
        if len(obs_cons) == len(state.inventory.consumables) and _consumable_multiset_matches(
            state.inventory.consumables, obs_cons
        ):
            state.inventory.consumables = _reorder_existing_consumables(
                state.inventory.consumables, obs_cons
            )
        elif obs_cons:
            state.inventory.consumables = _reconcile_existing_consumables(
                state.inventory.consumables, obs_cons
            )

    return raw_jokers, raw_cons


def _consumable_sig(card: Any) -> tuple[Any, Any]:
    return (getattr(card, "name", None), getattr(card, "edition", None))


def _consumable_multiset_matches(a: Sequence[Any], b: Sequence[Any]) -> bool:
    if len(a) != len(b):
        return False
    used: set[int] = set()
    for item_b in b:
        sig_b = _consumable_sig(item_b)
        found = False
        for i, item_a in enumerate(a):
            if i not in used and _consumable_sig(item_a) == sig_b:
                used.add(i)
                found = True
                break
        if not found:
            return False
    return True


def _reconcile_existing_consumables(
    existing: Sequence[Any], observed: Sequence[Any]
) -> list[Any]:
    used: set[int] = set()
    reconciled: list[Any] = []
    for obs in observed:
        sig_obs = _consumable_sig(obs)
        match_obj = None
        for i, prev in enumerate(existing):
            if i not in used and _consumable_sig(prev) == sig_obs:
                used.add(i)
                match_obj = prev
                break
        reconciled.append(match_obj if match_obj is not None else copy.deepcopy(obs))
    return reconciled


def _reorder_existing_consumables(
    existing: Sequence[Any], observed: Sequence[Any]
) -> list[Any]:
    return _reconcile_existing_consumables(existing, observed)


def _sync_hand_into_state(
    state: GameState,
    obs_hand: Sequence[HandCard],
    *,
    action: Action | None = None,
    selected_slot_uid: int | None = None,
    apply_direct_order: bool = False,
) -> int | None:
    """Reconciles CV-observed hand cards into `state.hand.cards` and populates Action fields.

    Preserves persistent `PlayingCard.id` and `HandCard.slot_uid` across draws and drags.
    Returns the `slot_uid` of the first raised/selected card if any.
    """
    state.ensure_slot_uids()
    existing = list(state.hand.cards)
    used_existing: set[int] = set()
    matched_slots: list[HandCard | None] = [None] * len(obs_hand)

    # 1. If tracking a pre-sort selected face-down slot across a sort, anchor that slot first
    if selected_slot_uid is not None:
        for i, obs in enumerate(obs_hand):
            if obs.selected and obs.facing_back:
                for j, ex in enumerate(existing):
                    if j not in used_existing and ex.slot_uid == selected_slot_uid:
                        matched_slots[i] = ex
                        used_existing.add(j)
                        break
                break

    # 2. Match visible face-up cards against previously held face-up cards
    for i, obs in enumerate(obs_hand):
        if matched_slots[i] is not None or obs.facing_back:
            continue
        for j, ex in enumerate(existing):
            if j in used_existing or ex.facing_back:
                continue
            if _cards_match(ex.card, obs.card):
                matched_slots[i] = ex
                used_existing.add(j)
                break

    # 3. Match face-up observations against previously face-down slots (e.g., after resolve/reveal)
    for i, obs in enumerate(obs_hand):
        if matched_slots[i] is not None or obs.facing_back:
            continue
        for j, ex in enumerate(existing):
            if j in used_existing:
                continue
            cand = ex._real_card or ex.card
            if cand.rank != Rank.NONE and _cards_match(cand, obs.card):
                ex.resolve_face_down(copy.deepcopy(obs.card), lift_facing_back=True)
                matched_slots[i] = ex
                used_existing.add(j)
                break
            if ex.bounds and any(_cards_match(c, obs.card) for c in ex.bounds.candidate_cards):
                match_cand = next(c for c in ex.bounds.candidate_cards if _cards_match(c, obs.card))
                ex.resolve_face_down(match_cand, lift_facing_back=True)
                matched_slots[i] = ex
                used_existing.add(j)
                break

    # 4. Match remaining face-down observations against remaining unused face-down slots
    for i, obs in enumerate(obs_hand):
        if matched_slots[i] is not None or not obs.facing_back:
            continue
        for j, ex in enumerate(existing):
            if j not in used_existing and ex.facing_back:
                matched_slots[i] = ex
                used_existing.add(j)
                break

    # 5. Any remaining unmatched observations are newly drawn cards from remaining_deck
    claimed_deck_ids = {
        hc.card.id
        for hc in matched_slots
        if hc is not None and hc.card and hc.card.id != 0
    }
    newly_drawn: list[HandCard] = []
    next_deck_id = max((c.id for c in state.full_deck), default=0) + 1

    for i, obs in enumerate(obs_hand):
        hc = matched_slots[i]
        if hc is not None:
            hc.selected = bool(obs.selected)
            if not obs.facing_back:
                hc.facing_back = False
                hc.card.debuffed = bool(obs.card.debuffed)
            continue

        if obs.facing_back:
            new_hc = HandCard(
                card=copy.deepcopy(obs.card),
                selected=bool(obs.selected),
                facing_back=True,
                slot_uid=state._get_next_slot_uid(),
                bounds=CardBounds(is_face_down=True),
            )
            new_hc.bounds.set_candidates(
                [
                    c
                    for c in state.get_face_down_candidate_pool(new_hc)
                    if c.id not in claimed_deck_ids
                ]
            )
        else:
            rem_match = None
            for deck_card in state.remaining_deck:
                if deck_card.id not in claimed_deck_ids and _cards_match(deck_card, obs.card):
                    rem_match = deck_card
                    break
            if rem_match is None:
                rem_match = copy.deepcopy(obs.card)
                rem_match.id = next_deck_id
                next_deck_id += 1
                state.full_deck.append(rem_match)
            else:
                rem_match.debuffed = bool(obs.card.debuffed)
            claimed_deck_ids.add(rem_match.id)
            new_hc = HandCard(
                card=rem_match,
                selected=bool(obs.selected),
                facing_back=False,
                slot_uid=state._get_next_slot_uid(),
            )
        matched_slots[i] = new_hc
        newly_drawn.append(new_hc)

    ordered_hand = [hc for hc in matched_slots if hc is not None]
    retained_in_prev_order = [
        ex for j, ex in enumerate(existing) if j in used_existing
    ]
    pre_reorder = retained_in_prev_order + newly_drawn

    if apply_direct_order:
        state.hand.cards = ordered_hand
    else:
        state.hand.cards = pre_reorder
        if action is not None and ordered_hand:
            obj_pos = {id(hc): idx for idx, hc in enumerate(pre_reorder)}
            action.hand_order = [obj_pos[id(hc)] for hc in ordered_hand]

    if action is not None:
        action.selected_cards = [
            copy.deepcopy(hc.card) for hc in ordered_hand if hc.selected
        ]

    state.reconcile_face_down_adjacencies()

    first_selected = next((hc for hc in ordered_hand if hc.selected), None)
    return first_selected.slot_uid if first_selected is not None else None


def build_run_trajectory(
    d: dict[str, Any],
    video_path: str | Path | None = None,
    *,
    preloaded_frames: dict[int, Any] | None = None,
    seed: str = "",
    deck: Deck = Deck.RED_DECK,
    stake: Stake = Stake.WHITE,
    fps: float | None = None,
    conf_threshold_hand: float = 0.50,
    conf_threshold_shop: float = 0.35,
    batch_size: int = 32,
    device: str | None = None,
    deduplicate: bool = True,
) -> tuple[GameState, RunTrajectory]:
    """Reconstructs a complete `(GameState, RunTrajectory)` from action JSON + video frames.

    Note: Starting mid-run is out of scope; always initializes from Ante 1 / Round 1.
    """
    from classifiers.common.video_utils import extract_video_frames

    cleaned_input = {
        "key": d["key"],
        "frames": remove_duplicates(d["frames"]) if deduplicate else d["frames"],
    }
    plan = plan_action_pipeline(cleaned_input)
    scheduled_actions: list[ScheduledAction] = plan["scheduled_actions"]

    resolved_fps = fps
    if resolved_fps is None and video_path is not None:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        cap_fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0.0
        cap.release()
        resolved_fps = float(cap_fps) if cap_fps and cap_fps > 0 else 60.0
    if not resolved_fps or resolved_fps <= 0:
        resolved_fps = 60.0

    needed_frames = sorted(
        set(plan["playing_card_cv_queue"])
        | set(plan["joker_cv_queue"])
        | set(plan["shop_cv_queue"])
        | set(plan["pack_cv_queue"])
    )
    if preloaded_frames is not None:
        frames_by_idx = {idx: preloaded_frames[idx] for idx in needed_frames if idx in preloaded_frames}
    elif video_path is not None and needed_frames:
        frames_by_idx = extract_video_frames(video_path, needed_frames)
    else:
        frames_by_idx = {}

    shop_res_by_frame, hand_res_by_frame = _run_batched_cv(
        scheduled_actions,
        plan,
        frames_by_idx,
        fps=resolved_fps,
        conf_threshold_hand=conf_threshold_hand,
        conf_threshold_shop=conf_threshold_shop,
        batch_size=batch_size,
        device=device,
    )

    state = GameState(
        seed=seed,
        deck=deck,
        stake=stake,
        ante=1,
        round=1,
        current_blind=BlindName.SMALL,
        location=Location.SELECT,
        apply_deck_stake_on_init=True,
    )
    trajectory = RunTrajectory()
    trajectory.record_step(state, action=None)

    A = ActionType
    for sa in scheduled_actions:
        action = Action(
            type=sa.action_type,
            frame_idx=sa.frame_id,
            timestamp=sa.frame_id / resolved_fps,
            metadata=dict(sa.metadata),
        )
        # Preserve CASH_OUT round payout calculation unless caller explicitly provided OCR override
        if sa.action_type == A.CASH_OUT and "include_round_payout" not in action.metadata:
            action.metadata["include_round_payout"] = True

        # Ensure blind transition leaves no stale hand cards from the prior round
        if sa.action_type in (A.SELECT_BLIND, A.SKIP_BLIND, A.CASH_OUT):
            state.hand.cards.clear()

        shop_out = shop_res_by_frame.get(sa.target_frame)
        raw_jokers: list[Any] = []
        raw_cons: list[Any] = []
        if shop_out is not None:
            raw_jokers, raw_cons = _sync_bars_into_state(state, shop_out, action)

        # Sync main shop shelves before matching/stepping shop actions
        if sa.ambient_location == Location.SHOP and shop_out is not None:
            raw_shop_items = list(getattr(shop_out, "shop_items", None) or [])
            raw_vouchers = list(getattr(shop_out, "vouchers", None) or [])
            raw_boosters = list(getattr(shop_out, "boosters", None) or [])

            state.shop_cards = _convert_items_of_type(raw_shop_items)
            state.shop_vouchers = _convert_items_of_type(raw_vouchers, Voucher)
            state.shop_packs = _convert_items_of_type(raw_boosters, BoosterPack)
            if getattr(state, "coupon_shop_active", False):
                for item in state.shop_cards + state.shop_packs:
                    item.couponed = True

            if sa.action_type in (A.BUY_CARD, A.BUY_AND_USE):
                action.target_idx = match_button_to_item_index(
                    sa.button_box, raw_shop_items, attach="bottom_center"
                )
            elif sa.action_type == A.OPEN_PACK:
                action.target_idx = match_button_to_item_index(
                    sa.button_box, raw_boosters, attach="bottom_center"
                )
            elif sa.action_type == A.REDEEM_VOUCHER:
                action.target_idx = match_button_to_item_index(
                    sa.button_box, raw_vouchers, attach="bottom_center"
                )

        # Sync booster pack contents before matching/stepping pack actions
        if sa.ambient_location == Location.PACK and shop_out is not None:
            raw_pack_items = list(getattr(shop_out, "pack_items", None) or [])
            pack_objs = _convert_items_of_type(
                raw_pack_items, (Joker, Tarot, Planet, Spectral, PlayingCard)
            )
            state.pack_cards = pack_objs
            if state.active_pack is None:
                inferred = _infer_booster_pack_from_items(pack_objs)
                state.active_pack = inferred
                state.pack_choices_remaining = inferred.choose
                state.pre_pack_location = sa.pre_pack_location_hint or Location.SHOP
            state.location = Location.PACK

            if sa.action_type == A.USE_IN_PACK:
                action.target_idx = match_button_to_item_index(
                    sa.button_box, raw_pack_items, attach="bottom_center"
                )

        # Spatial matching for top-bar actions (USE_CONSUMABLE, SELL_CARD)
        if sa.action_type == A.USE_CONSUMABLE:
            action.target_idx = match_button_to_item_index(
                sa.button_box, raw_cons, attach="center_right"
            )
        elif sa.action_type == A.SELL_CARD:
            combined_bar = raw_jokers + raw_cons
            action.target_idx = match_button_to_item_index(
                sa.button_box, combined_bar, attach="center_right"
            )

        # Hand synchronization & sort handling
        pre_selected_uid: int | None = None
        if sa.action_type in (A.SORT_RANK, A.SORT_SUIT):
            pre_hand = (
                hand_res_by_frame.get(sa.pre_sort_frame)
                if sa.pre_sort_frame is not None
                else None
            )
            if pre_hand is not None:
                obs_pre = [
                    _convert_card_result_to_hand_card(c) for c in pre_hand.cards
                ]
                pre_selected_uid = _sync_hand_into_state(state, obs_pre, action=action)
                if pre_selected_uid is not None:
                    action.metadata["selected_slot_uid"] = pre_selected_uid
        else:
            hand_out = hand_res_by_frame.get(sa.target_frame)
            if hand_out is not None:
                obs_cards = [
                    _convert_card_result_to_hand_card(c) for c in hand_out.cards
                ]
                _sync_hand_into_state(state, obs_cards, action=action)

        # Advance simulation
        # OCR hook point: populate current_chips and HUD counters (money, hands, discards)
        # around PLAY_HAND / END_ROUND / CASH_OUT when OCR is wired in.
        state.frame_idx = action.frame_idx
        state.timestamp = action.timestamp
        prev_loc = state.location
        on_action(state, action)

        # Post-sort reconciliation aligns actual settled positions + tightens face-down CardBounds
        if (
            sa.action_type in (A.SORT_RANK, A.SORT_SUIT)
            and sa.post_sort_frame is not None
            and sa.post_sort_frame in hand_res_by_frame
        ):
            post_hand = hand_res_by_frame[sa.post_sort_frame]
            obs_post = [
                _convert_card_result_to_hand_card(c) for c in post_hand.cards
            ]
            _sync_hand_into_state(
                state,
                obs_post,
                action=None,
                selected_slot_uid=pre_selected_uid,
                apply_direct_order=True,
            )
            state.apply_sort_observation(
                sort_type="rank" if sa.action_type == A.SORT_RANK else "suit",
                selected_slot_uid=pre_selected_uid,
            )

        # Clear temporary hands when closing a booster pack or ending a round
        if sa.action_type in (A.END_ROUND, A.CASH_OUT, A.OPEN_PACK):
            state.hand.cards.clear()
        elif prev_loc == Location.PACK and state.location != Location.PACK:
            state.hand.cards.clear()

        trajectory.record_step(state, action)

    trajectory.impute_retroactive_boss_blinds()
    return state, trajectory


def process_actions(
    d: dict[str, Any],
    video_path: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Schedules CV queues for action frames and optionally reconstructs full GameState trajectory.

    When `video_path` and `preloaded_frames` are omitted, returns the 4 CV queue dicts.
    When either is provided, runs batched CV and returns the queues alongside `state` and `trajectory`.
    """
    plan = plan_action_pipeline(d)
    queues = {
        "playing_card_cv_queue": plan["playing_card_cv_queue"],
        "joker_cv_queue": plan["joker_cv_queue"],
        "shop_cv_queue": plan["shop_cv_queue"],
        "pack_cv_queue": plan["pack_cv_queue"],
    }
    if video_path is not None or kwargs.get("preloaded_frames") is not None:
        state, trajectory = build_run_trajectory(d, video_path=video_path, **kwargs)
        return {
            **queues,
            "state": state,
            "trajectory": trajectory,
        }
    return queues


if __name__ == "__main__":
    configure_logging()

    config = get_config()

    filename = "bu_full_1791436070.json"
    path = rf"{config.output_dir}\{filename}"

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    cleaned = remove_duplicates(data["frames"])

    pprint(cleaned)
