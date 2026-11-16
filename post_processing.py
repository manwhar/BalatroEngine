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

import argparse
from collections import Counter
import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from tqdm import tqdm

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
    Enhancement,
    HandCard,
    Joker,
    JokerName,
    Location,
    Planet,
    PlayingCard,
    Rank,
    Spectral,
    Stake,
    Suit,
    Tarot,
    TarotName,
    Voucher,
)
from game.engine.cv_stubs import (
    _convert_card_result_to_hand_card,
    _convert_items_of_type,
)
from game.engine.jokers import on_joker_added
from game.engine.rules import on_action, trigger_end_of_round_effects
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
    "post_sort": 50,
    "pre_sort": -15,
    ActionType.PLAY_HAND: 0,
    ActionType.DISCARD: 0,
    ActionType.BUY_CARD: 0,
    ActionType.REROLL_SHOP: -10,
    # HUMAN TODO: manually verify remaining offsets below (above verified empirically)
    ActionType.BUY_AND_USE: -2,
    ActionType.OPEN_PACK: -2,
    ActionType.USE_IN_PACK: -2,
    ActionType.SKIP_PACK: -2,
    ActionType.REDEEM_VOUCHER: -2,
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


def filter_premature_next_rounds(
    frames: dict[str, int | dict],
    key: dict[str, int],
    *,
    min_shop_frames: int = 60,
    diag: dict[str, Any] | None = None,
) -> dict[str, int | dict]:
    """Filters false-positive `Next Round` clicks fired while Shop UI slides up after `Cash Out`."""
    inv_key = {v: k for k, v in key.items()}
    ordered = sorted(frames.items(), key=lambda kv: int(kv[0]))
    filtered: dict[str, int | dict] = {}
    last_cash_out_frame: int | None = None

    for frame_str, entry in ordered:
        frame_idx = int(frame_str)
        raw_label = inv_key.get(_extract_action_id(entry))
        action = _resolve_action(raw_label) if raw_label is not None else None

        if action == ActionType.CASH_OUT:
            last_cash_out_frame = frame_idx
        elif action == ActionType.NEXT_ROUND:
            if (
                last_cash_out_frame is not None
                and 0 <= (frame_idx - last_cash_out_frame) < min_shop_frames
            ):
                if diag is not None:
                    diag["filtered_premature_next_rounds"].append(frame_idx)
                continue
            last_cash_out_frame = None
        elif action in (
            ActionType.SELECT_BLIND,
            ActionType.SKIP_BLIND,
            ActionType.PLAY_HAND,
        ):
            last_cash_out_frame = None

        filtered[frame_str] = entry

    return filtered


def _resolve_action(raw_label: str) -> ActionType | None:
    if raw_label in NON_GAMEPLAY_ACTIONS:
        return None
    if raw_label in ACTION_REMAP:
        return ACTION_REMAP[raw_label]
    return ActionType(raw_label)


def find_missing_end_round_segments(d: dict[str, Any]) -> list[tuple[int, int]]:
    """Finds `[PLAY_HAND, CASH_OUT]` frame windows lacking an intervening `END_ROUND`."""
    key = d.get("key") or {}
    frames = d.get("frames") or {}
    inv_key = {v: k for k, v in key.items()}

    ordered = sorted(frames.items(), key=lambda kv: int(kv[0]))
    segments: list[tuple[int, int]] = []
    pending_play_hand: int | None = None

    for frame_str, frame_entry in ordered:
        frame_idx = int(frame_str)
        raw_label = inv_key.get(_extract_action_id(frame_entry))
        if raw_label is None:
            continue
        action = _resolve_action(raw_label)
        if action is None:
            continue

        if action == ActionType.PLAY_HAND:
            pending_play_hand = frame_idx
        elif action in (
            ActionType.END_ROUND,
            ActionType.SELECT_BLIND,
            ActionType.SKIP_BLIND,
            ActionType.NEXT_ROUND,
        ):
            pending_play_hand = None
        elif action == ActionType.CASH_OUT:
            if pending_play_hand is not None and frame_idx > pending_play_hand + 1:
                segments.append((pending_play_hand, frame_idx))
            pending_play_hand = None

    return segments


def inject_end_round_actions(
    d: dict[str, Any],
    *,
    video_path: str | Path | None = None,
    preloaded_frames: dict[int, Any] | None = None,
    fps: float = 60.0,
    batch_size: int = 8,
    device: str | None = None,
    expressive: bool = False,
) -> dict[str, Any]:
    """Predicts and injects missing `ActionType.END_ROUND` events using the temporal model."""
    segments = find_missing_end_round_segments(d)
    if not segments or (video_path is None and preloaded_frames is None):
        return d

    from classifiers.inference import batch_recognize_end_rounds

    results = batch_recognize_end_rounds(
        segments,
        video_path=video_path,
        preloaded_frames=preloaded_frames,
        fps=fps,
        batch_size=batch_size,
        device=device,
        show_progress=expressive,
    )
    if not results:
        return d

    key = dict(d["key"])
    frames = dict(d["frames"])
    end_round_label = ActionType.END_ROUND.value
    end_round_id = key.setdefault(end_round_label, max(key.values(), default=-1) + 1)

    for (play_f, cash_f), res in zip(segments, results):
        pred_f = max(play_f + 1, min(cash_f - 1, int(res.end_round_frame)))
        if str(pred_f) in frames:
            chosen: int | None = None
            max_offset = max(pred_f - (play_f + 1), (cash_f - 1) - pred_f)
            for delta in range(1, max_offset + 1):
                for cand in (pred_f + delta, pred_f - delta):
                    if play_f < cand < cash_f and str(cand) not in frames:
                        chosen = cand
                        break
                if chosen is not None:
                    break
            if chosen is None:
                continue
            pred_f = chosen

        frames[str(pred_f)] = {
            "action": end_round_id,
            "box": None,
            "metadata": {
                "source": "end_round_classifier",
                "confidence": float(res.confidence),
                "end_round_timestamp": float(res.end_round_timestamp),
                "play_hand_frame": play_f,
                "cash_out_frame": cash_f,
            },
        }

    ordered_frames = dict(sorted(frames.items(), key=lambda kv: int(kv[0])))
    return {"key": key, "frames": ordered_frames}


FACE_DOWN_BOSS_BLINDS: frozenset[BlindName] = frozenset(
    {
        BlindName.THE_HOUSE,
        BlindName.THE_WHEEL,
        BlindName.THE_FISH,
        BlindName.THE_MARK,
    }
)


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
    hud_ocr_queue: dict[int, ActionType] = {}
    scheduled_actions: list[ScheduledAction] = []
    pre_pack_loc: Location | None = None
    blind_idx_in_ante = 0  # 0=Small, 1=Big, 2=Boss

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
                if blind_idx_in_ante == 2:
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
                A.SELECT_BLIND | A.SKIP_BLIND | A.REROLL_BOSS | A.CASH_OUT | A.END_ROUND
            ):
                pass

        if action not in {A.SORT_RANK, A.SORT_SUIT}:
            hud_ocr_queue[target_frame] = action
            if target_frame not in shop_cv_queue and target_frame not in pack_cv_queue:
                joker_cv_queue[target_frame] = action

        extra_meta = (
            dict(frame_entry.get("metadata", {}))
            if isinstance(frame_entry, dict)
            and isinstance(frame_entry.get("metadata"), dict)
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
                pre_pack_location_hint=pre_pack_loc
                if location == Location.PACK
                else None,
                metadata=extra_meta,
            )
        )

        new_loc = get_next_location(action, idx)
        if new_loc is not None:
            if new_loc == Location.PACK and location != Location.PACK:
                pre_pack_loc = (
                    Location.SHOP if action == A.OPEN_PACK else Location.SELECT
                )
            elif new_loc != Location.PACK:
                pre_pack_loc = None
            location = new_loc

        if action in (A.END_ROUND, A.SKIP_BLIND):
            blind_idx_in_ante = (blind_idx_in_ante + 1) % 3

    return {
        "scheduled_actions": scheduled_actions,
        "playing_card_cv_queue": playing_card_cv_queue,
        "joker_cv_queue": joker_cv_queue,
        "shop_cv_queue": shop_cv_queue,
        "pack_cv_queue": pack_cv_queue,
        "hud_ocr_queue": hud_ocr_queue,
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
    median_y1: float | None = None
    if attach == "center_right" and len(items) > 1:
        y1s = sorted(
            float(b[1])
            for it in items
            if (b := getattr(it, "box", None)) and len(b) == 4 and b != [0, 0, 0, 0]
        )
        if y1s:
            median_y1 = y1s[len(y1s) // 2]

    for idx, item in enumerate(items):
        box = getattr(item, "box", None)
        if not box or len(box) != 4 or box == [0, 0, 0, 0]:
            continue
        if attach == "center_right":
            item_w = max(1.0, float(box[2] - box[0]))
            anchor_x = float(box[2]) + 0.55 * item_w
            anchor_y = (box[1] + box[3]) / 2.0
        else:
            anchor_x = (box[0] + box[2]) / 2.0
            anchor_y = (box[1] + box[3]) / 2.0

        dist = abs(btn_cx - anchor_x) + 0.25 * abs(btn_cy - anchor_y)
        if attach == "center_right" and len(items) > 1:
            if btn_cx < float(box[0]):
                dist += 1000.0
            if (
                median_y1 is not None
                and float(box[1]) <= median_y1 - 10.0
                and btn_cx >= float(box[0])
            ):
                dist -= 40.0
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
    conf_threshold_ocr: float = 0.90,
    batch_size: int,
    device: str | None,
    expressive: bool = False,
) -> tuple[dict[int, Any], dict[int, Any], dict[int, Any]]:
    """Runs shop/pack/joker-bar CV first, selectively runs hand CV, and batches HUD OCR."""
    from classifiers.inference import (
        batch_recognize_hands,
        batch_recognize_hud,
        batch_recognize_shops,
    )

    def _pack_frame_inputs(frame_ids: Sequence[int]) -> list[tuple[Any, int, float]]:
        return [
            (frames_by_idx[fid], fid, fid / fps)
            for fid in frame_ids
            if fid in frames_by_idx
        ]

    shop_ids = sorted(queues["shop_cv_queue"].keys())
    pack_ids = sorted(queues["pack_cv_queue"].keys())
    joker_ids = sorted(queues["joker_cv_queue"].keys())
    hud_queue = queues.get("hud_ocr_queue", {})
    hud_ids = sorted(hud_queue.keys())
    hand_candidates = sorted(queues["playing_card_cv_queue"].keys())

    pbar = (
        tqdm(
            total=len(shop_ids)
            + len(pack_ids)
            + len(joker_ids)
            + len(hand_candidates)
            + len(hud_ids),
            desc="Batched CV & OCR",
            unit="frame",
        )
        if expressive
        else None
    )

    shop_res_by_frame: dict[int, Any] = {}

    if shop_ids:
        if pbar is not None:
            pbar.set_postfix_str("shop")
        shop_outs = batch_recognize_shops(
            _pack_frame_inputs(shop_ids),
            mode="shop",
            conf_threshold=conf_threshold_shop,
            batch_size=batch_size,
            device=device,
        )
        for out in shop_outs:
            shop_res_by_frame[out.frame_index] = out
        if pbar is not None:
            pbar.update(len(shop_ids))

    if pack_ids:
        if pbar is not None:
            pbar.set_postfix_str("pack")
        pack_outs = batch_recognize_shops(
            _pack_frame_inputs(pack_ids),
            mode="pack",
            conf_threshold=conf_threshold_shop,
            batch_size=batch_size,
            device=device,
        )
        for out in pack_outs:
            shop_res_by_frame.setdefault(out.frame_index, out)
        if pbar is not None:
            pbar.update(len(pack_ids))

    if joker_ids:
        if pbar is not None:
            pbar.set_postfix_str("bars")
        bar_outs = batch_recognize_shops(
            _pack_frame_inputs(joker_ids),
            mode="bars",
            conf_threshold=conf_threshold_shop,
            batch_size=batch_size,
            device=device,
        )
        for out in bar_outs:
            shop_res_by_frame.setdefault(out.frame_index, out)
        if pbar is not None:
            pbar.update(len(joker_ids))

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
    skipped_hand = 0
    for fid in hand_candidates:
        if fid in queues["pack_cv_queue"]:
            if pack_shows_hand_by_frame.get(fid) is False:
                skipped_hand += 1
                continue
        filtered_hand_ids.append(fid)
    if pbar is not None and skipped_hand:
        pbar.update(skipped_hand)

    hand_res_by_frame: dict[int, Any] = {}
    if filtered_hand_ids:
        blind_hand_ids = [
            fid for fid in filtered_hand_ids if fid not in queues["pack_cv_queue"]
        ]
        pack_hand_ids = [
            fid for fid in filtered_hand_ids if fid in queues["pack_cv_queue"]
        ]
        if blind_hand_ids:
            if pbar is not None:
                pbar.set_postfix_str("hand")
            hand_outs = batch_recognize_hands(
                _pack_frame_inputs(blind_hand_ids),
                conf_threshold=conf_threshold_hand,
                batch_size=batch_size,
                device=device,
            )
            for out in hand_outs:
                hand_res_by_frame[out.frame_index] = out
            if pbar is not None:
                pbar.update(len(blind_hand_ids))
        if pack_hand_ids:
            if pbar is not None:
                pbar.set_postfix_str("pack_hand")
            pack_hand_outs = batch_recognize_hands(
                _pack_frame_inputs(pack_hand_ids),
                conf_threshold=min(conf_threshold_hand, 0.30),
                batch_size=batch_size,
                device=device,
                roi_key="pack_hand",
            )
            for out in pack_hand_outs:
                hand_res_by_frame[out.frame_index] = out
            if pbar is not None:
                pbar.update(len(pack_hand_ids))

    hud_res_by_frame: dict[int, Any] = {}
    if hud_ids:
        if pbar is not None:
            pbar.set_postfix_str("hud_ocr")
        base_hud_fields = (
            "hud_money",
            "hud_hands",
            "hud_discards",
            "hud_ante",
            "hud_round",
        )
        fields_by_frame: dict[int, list[str]] = {}
        for sa in scheduled_actions:
            if sa.target_frame not in hud_queue:
                continue
            f_list = fields_by_frame.setdefault(sa.target_frame, list(base_hud_fields))
            if (
                sa.ambient_location == Location.BLIND
                or sa.action_type == ActionType.END_ROUND
            ):
                for extra_field in ("hud_round_score", "hud_blind_target"):
                    if extra_field not in f_list:
                        f_list.append(extra_field)
            if sa.ambient_location == Location.SHOP and "hud_reroll_cost" not in f_list:
                f_list.append("hud_reroll_cost")

        hud_outs = batch_recognize_hud(
            _pack_frame_inputs(hud_ids),
            fields_by_frame=fields_by_frame,
            conf_threshold=conf_threshold_ocr,
            batch_size=max(batch_size, 128),
            device=device,
        )
        for out in hud_outs:
            hud_res_by_frame[out.frame_index] = out
        if pbar is not None:
            pbar.update(len(hud_ids))

    if pbar is not None:
        pbar.close()

    return shop_res_by_frame, hand_res_by_frame, hud_res_by_frame


def _has_end_round_after(
    scheduled_actions: Sequence[ScheduledAction], start_idx: int
) -> bool:
    """Returns True if an explicit END_ROUND action follows before CASH_OUT/SELECT_BLIND."""
    for nxt in scheduled_actions[start_idx + 1 :]:
        if nxt.action_type == ActionType.END_ROUND:
            return True
        if nxt.action_type in (
            ActionType.CASH_OUT,
            ActionType.SELECT_BLIND,
            ActionType.SKIP_BLIND,
            ActionType.NEXT_ROUND,
        ):
            return False
    return False


def _init_diagnostics() -> dict[str, Any]:
    """Creates a mutable diagnostics dictionary for `--expressive` reporting."""
    return {
        "timings": {},
        "end_round_segments_checked": 0,
        "end_rounds_injected": 0,
        "filtered_premature_next_rounds": [],
        "missing_frames": [],
        "hud_frames_checked": 0,
        "ocr_score_updates": 0,
        "ocr_none_by_field": {},
        "ocr_overrides_total": 0,
        "ocr_overrides_by_field": {},
        "ocr_override_samples": [],
        "unexplained_joker_changes": [],
        "unexplained_consumable_changes": [],
        "unmatched_button_targets": [],
        "location_mismatches": [],
        "cashout_without_round_won": [],
        "empty_hand_cv_actions": [],
        "missing_shop_cv_actions": [],
    }


def _record_ocr_override(
    diag: dict[str, Any] | None,
    *,
    frame_id: int,
    action_type: ActionType,
    field_name: str,
    prev_val: Any,
    new_val: Any,
) -> None:
    """Records an OCR override when observed HUD state differs from simulated state."""
    if diag is None or prev_val == new_val:
        return
    diag["ocr_overrides_total"] += 1
    by_field = diag["ocr_overrides_by_field"]
    by_field[field_name] = by_field.get(field_name, 0) + 1
    samples = diag["ocr_override_samples"]
    if len(samples) < 12:
        samples.append((frame_id, action_type.value, field_name, prev_val, new_val))


def _print_expressive_report(diag: dict[str, Any]) -> None:
    """Prints end-of-run diagnostics when `--expressive` is enabled."""
    print("\n=== Expressive Post-Processing Diagnostics ===")
    timings = diag.get("timings", {})
    if timings:
        parts = [f"{k}={v:.2f}s" for k, v in timings.items()]
        print(f"Timings: {', '.join(parts)}")

    segs = diag.get("end_round_segments_checked", 0)
    inj = diag.get("end_rounds_injected", 0)
    if segs or inj:
        print(
            f"END_ROUND transitions: {inj} injected across {segs} [PLAY_HAND, CASH_OUT] segment(s)"
        )

    premature_nr = diag.get("filtered_premature_next_rounds", [])
    if premature_nr:
        print(
            f"Filtered premature Next Round detections (<60f after Cash Out): "
            f"{len(premature_nr)} (frames {premature_nr[:10]})"
        )

    missing_frames = diag.get("missing_frames", [])
    if missing_frames:
        print(
            f"[WARN] Missing CV frames: {len(missing_frames)} requested frame(s) could not be read "
            f"(e.g. {missing_frames[:5]})"
        )

    hud_checked = diag.get("hud_frames_checked", 0)
    score_updates = diag.get("ocr_score_updates", 0)
    overrides = diag.get("ocr_overrides_total", 0)
    by_field = diag.get("ocr_overrides_by_field", {})
    print(
        f"OCR drift overrides: {overrides} field override(s) across {hud_checked} HUD-checked action(s) "
        f"(+{score_updates} blind score update(s))"
    )
    if by_field:
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(by_field.items()))
        print(f"  Override breakdown: {breakdown}")
        for fid, act, fld, old_v, new_v in diag.get("ocr_override_samples", []):
            print(f"  - frame {fid} ({act}) {fld}: sim={old_v} -> ocr={new_v}")

    none_counts = diag.get("ocr_none_by_field", {})
    if none_counts:
        dropped = ", ".join(f"{k}={v}" for k, v in sorted(none_counts.items()))
        print(f"OCR skipped/low-conf (None) counts: {dropped}")

    unexp_jokers = diag.get("unexplained_joker_changes", [])
    if unexp_jokers:
        print(
            f"[WARN] Unexplained Joker inventory changes: {len(unexp_jokers)} time(s)"
        )
        for fid, act, lost, gained in unexp_jokers[:10]:
            print(f"  - frame {fid} ({act}): lost={lost}, gained={gained}")

    unexp_cons = diag.get("unexplained_consumable_changes", [])
    if unexp_cons:
        print(
            f"[WARN] Unexplained Consumable inventory changes: {len(unexp_cons)} time(s)"
        )
        for fid, act, lost, gained in unexp_cons[:10]:
            print(f"  - frame {fid} ({act}): lost={lost}, gained={gained}")

    unmatched = diag.get("unmatched_button_targets", [])
    if unmatched:
        print(
            f"[WARN] Unmatched spatial button targets: {len(unmatched)} action(s) had target_idx=None"
        )
        for fid, act, loc in unmatched[:8]:
            print(f"  - frame {fid} ({act} @ {loc})")

    loc_mismatches = diag.get("location_mismatches", [])
    if loc_mismatches:
        print(
            f"[WARN] Location drift: {len(loc_mismatches)} action(s) had ambient_location != state.location"
        )
        for fid, act, exp_loc, got_loc in loc_mismatches[:8]:
            print(f"  - frame {fid} ({act}): expected={exp_loc}, state={got_loc}")

    cashout_warn = diag.get("cashout_without_round_won", [])
    if cashout_warn:
        print(
            f"[WARN] CASH_OUT before round_won=True: {len(cashout_warn)} time(s) "
            f"(frames {cashout_warn[:8]})"
        )

    empty_hand = diag.get("empty_hand_cv_actions", [])
    if empty_hand:
        print(
            f"[WARN] PLAY_HAND/DISCARD with 0 detected hand cards: {len(empty_hand)} time(s) (frames {empty_hand[:8]})"
        )

    missing_shop = diag.get("missing_shop_cv_actions", [])
    if missing_shop:
        print(
            f"[WARN] SHOP/PACK actions missing CV output: {len(missing_shop)} time(s) (frames {missing_shop[:8]})"
        )
    print("==============================================")


def _sync_hud_into_state(
    state: GameState,
    hud_out: Any,
    sa: ScheduledAction,
    action: Action,
    *,
    has_end_round_after: bool = False,
    diag: dict[str, Any] | None = None,
) -> int | float | None:
    """Applies validated HUD OCR readings into `state`, falling back cleanly on `None`.

    Returns a deferred `current_chips` value when `PLAY_HAND` observes a score already at or
    above `target_chips` while an `END_ROUND` action is scheduled afterwards.
    """
    values = getattr(hud_out, "values", None)
    if not isinstance(values, dict) or not values:
        return None

    if diag is not None:
        diag["hud_frames_checked"] += 1
        none_by_field = diag["ocr_none_by_field"]
        for k, v in values.items():
            if v is None:
                none_by_field[k] = none_by_field.get(k, 0) + 1

    observed = {k: v for k, v in values.items() if v is not None}
    if observed:
        action.metadata["hud_ocr"] = observed

    money_val = values.get("hud_money")
    if money_val is not None:
        if sa.action_type == ActionType.CASH_OUT and not state.round_won:
            if diag is not None:
                diag["cashout_without_round_won"].append(sa.frame_id)
            trigger_end_of_round_effects(state, action)
        new_money = int(money_val)
        if (
            sa.ambient_location == Location.BLIND
            and sa.action_type != ActionType.SELECT_BLIND
            and state.hands_played_this_round > 0
            and state.last_hand_played is not None
            and state.inventory is not None
            and new_money - state.money == 4
        ):
            for j in state.inventory.jokers:
                if (
                    j.name == JokerName.TO_DO_LIST
                    and not j.debuffed
                    and j.state.get("target_hand") != state.last_hand_played
                ):
                    j.state["target_hand"] = state.last_hand_played
                    state.to_do_poker_hand = state.last_hand_played
                    state.money += 4
                    break
        _record_ocr_override(
            diag,
            frame_id=sa.frame_id,
            action_type=sa.action_type,
            field_name="money",
            prev_val=state.money,
            new_val=new_money,
        )
        state.money = new_money

    hands_val = values.get("hud_hands")
    if (
        hands_val is not None
        and sa.ambient_location == Location.BLIND
        and sa.action_type != ActionType.SELECT_BLIND
        and int(hands_val) >= 0
    ):
        new_hands = int(hands_val)
        _record_ocr_override(
            diag,
            frame_id=sa.frame_id,
            action_type=sa.action_type,
            field_name="hands_left",
            prev_val=state.hands_left,
            new_val=new_hands,
        )
        state.hands_left = new_hands

    discards_val = values.get("hud_discards")
    if (
        discards_val is not None
        and sa.ambient_location == Location.BLIND
        and sa.action_type != ActionType.SELECT_BLIND
        and int(discards_val) >= 0
    ):
        new_discards = int(discards_val)
        _record_ocr_override(
            diag,
            frame_id=sa.frame_id,
            action_type=sa.action_type,
            field_name="discards_left",
            prev_val=state.discards_left,
            new_val=new_discards,
        )
        state.discards_left = new_discards

    blind_target_val = values.get("hud_blind_target")
    if (
        blind_target_val is not None
        and sa.ambient_location == Location.BLIND
        and blind_target_val > 0
    ):
        _record_ocr_override(
            diag,
            frame_id=sa.frame_id,
            action_type=sa.action_type,
            field_name="target_chips",
            prev_val=state.target_chips,
            new_val=blind_target_val,
        )
        state.target_chips = blind_target_val

    deferred_round_score: int | float | None = None
    round_score_val = values.get("hud_round_score")
    if (
        round_score_val is not None
        and sa.ambient_location == Location.BLIND
        and round_score_val >= 0
    ):
        if (
            sa.action_type == ActionType.PLAY_HAND
            and has_end_round_after
            and state.target_chips
            and round_score_val >= state.target_chips
        ):
            deferred_round_score = round_score_val
        else:
            if diag is not None and state.current_chips != round_score_val:
                diag["ocr_score_updates"] += 1
            state.current_chips = round_score_val

    ante_val = values.get("hud_ante")
    if (
        ante_val is not None
        and sa.ambient_location == Location.BLIND
        and not state.round_won
        and not getattr(state, "pending_blind_advance", False)
        and 1 <= int(ante_val) <= 39
    ):
        new_ante = int(ante_val)
        _record_ocr_override(
            diag,
            frame_id=sa.frame_id,
            action_type=sa.action_type,
            field_name="ante",
            prev_val=state.ante,
            new_val=new_ante,
        )
        state.ante = new_ante

    round_val = values.get("hud_round")
    if (
        round_val is not None
        and sa.ambient_location == Location.BLIND
        and sa.action_type != ActionType.END_ROUND
        and not getattr(state, "round_advanced", False)
        and int(round_val) >= 1
    ):
        new_round = int(round_val)
        _record_ocr_override(
            diag,
            frame_id=sa.frame_id,
            action_type=sa.action_type,
            field_name="round",
            prev_val=state.round,
            new_val=new_round,
        )
        state.round = new_round

    reroll_val = values.get("hud_reroll_cost")
    if (
        reroll_val is not None
        and sa.ambient_location == Location.SHOP
        and int(reroll_val) >= 0
    ):
        reroll_cost = int(reroll_val)
        if sa.action_type == ActionType.REROLL_SHOP and (
            (state.free_rerolls == 0 and reroll_cost == state.reroll_cost + 1)
            or (state.free_rerolls > 0 and reroll_cost == state.reroll_cost)
        ):
            # At frame_id - 2 on Reroll Shop, the depressed button already shows
            # the post-reroll cost that on_action(REROLL_SHOP) is about to set.
            pass
        elif reroll_cost == 0 and state.free_rerolls > 0:
            pass
        elif reroll_cost > 0 and state.free_rerolls > 0:
            _record_ocr_override(
                diag,
                frame_id=sa.frame_id,
                action_type=sa.action_type,
                field_name="reroll_cost",
                prev_val=state.reroll_cost,
                new_val=reroll_cost,
            )
            state.free_rerolls = 0
            state.reroll_cost = reroll_cost
        else:
            _record_ocr_override(
                diag,
                frame_id=sa.frame_id,
                action_type=sa.action_type,
                field_name="reroll_cost",
                prev_val=state.reroll_cost,
                new_val=reroll_cost,
            )
            state.reroll_cost = reroll_cost

    return deferred_round_score


def _display_card_name(card: Any) -> str:
    """Returns a concise human-readable name for any card, joker, voucher, or pack."""
    if card is None:
        return "None"
    rank = getattr(card, "rank", None)
    suit = getattr(card, "suit", None)
    if rank is not None and suit is not None:
        r_str = getattr(rank, "value", str(rank))
        s_str = getattr(suit, "value", str(suit))
        return f"{r_str} of {s_str}"
    name = getattr(card, "name", None)
    if name is not None:
        return str(getattr(name, "value", name))
    return str(card)


def _counter_diff_names(
    before: Sequence[Any], after: Sequence[Any]
) -> tuple[list[str], list[str]]:
    """Returns `(lost_names, gained_names)` multiset differences in original order."""
    after_counts = Counter(_display_card_name(x) for x in after)
    lost: list[str] = []
    for item in before:
        label = _display_card_name(item)
        if after_counts[label] > 0:
            after_counts[label] -= 1
        else:
            lost.append(label)

    before_counts = Counter(_display_card_name(x) for x in before)
    gained: list[str] = []
    for item in after:
        label = _display_card_name(item)
        if before_counts[label] > 0:
            before_counts[label] -= 1
        else:
            gained.append(label)
    return lost, gained


def _sync_bars_into_state(
    state: GameState,
    shop_out: Any,
    action: Action,
    diag: dict[str, Any] | None = None,
    *,
    next_obs_cons_counts: Counter[tuple[Any, Any]] | None = None,
    suppress_consumable_loss_diag: bool = False,
    sync_jokers: bool = True,
    sync_consumables: bool = True,
) -> tuple[list[Any], list[Any]]:
    """Syncs Joker and Consumable bars from CV into `state` & records non-identity `joker_order`."""
    raw_jokers = list(getattr(shop_out, "joker_bar", None) or [])
    if sync_jokers:
        obs_jokers = _convert_items_of_type(raw_jokers, Joker)
        prev_jokers = list(state.inventory.jokers) if state.inventory else []
        prev_joker_ids = [j.id for j in prev_jokers]

        if state.inventory and state.inventory.jokers:
            if (
                state.current_blind == BlindName.AMBER_ACORN
                and not state.boss_disabled
                and any(j.facing_back for j in state.inventory.jokers)
            ):
                # Amber Acorn flips owned jokers face-down, hiding their faces from CV;
                # Stage 2 forces red card backs into bogus classes so preserve tracked jokers.
                pass
            else:
                matched_ids = state.match_cv_joker_order(obs_jokers)
                if matched_ids:
                    if matched_ids != prev_joker_ids:
                        action.joker_order = matched_ids
                else:
                    state.reconcile_cv_jokers(obs_jokers, trigger_added_hooks=False)
        elif obs_jokers:
            state.reconcile_cv_jokers(obs_jokers, trigger_added_hooks=False)

        if diag is not None and state.inventory is not None:
            lost_jokers, gained_jokers = _counter_diff_names(
                prev_jokers, state.inventory.jokers
            )
            if lost_jokers or gained_jokers:
                diag["unexplained_joker_changes"].append(
                    (action.frame_idx, action.type.value, lost_jokers, gained_jokers)
                )

    raw_cons = list(getattr(shop_out, "consumeables", None) or [])
    if sync_consumables:
        prev_cons = list(state.inventory.consumables) if state.inventory else []
        obs_cons = _convert_items_of_type(raw_cons, (Tarot, Planet, Spectral))
        if state.inventory and (obs_cons or state.inventory.consumables):
            effective_obs_cons = list(obs_cons)
            if next_obs_cons_counts is not None:
                obs_counts = Counter(_consumable_sig(c) for c in effective_obs_cons)
                prev_counts = Counter(
                    _consumable_sig(c) for c in state.inventory.consumables
                )
                for prev_c in state.inventory.consumables:
                    sig = _consumable_sig(prev_c)
                    expected_keep = min(prev_counts[sig], next_obs_cons_counts[sig])
                    if obs_counts[sig] < expected_keep:
                        effective_obs_cons.append(copy.deepcopy(prev_c))
                        obs_counts[sig] += 1

            if len(effective_obs_cons) == len(
                state.inventory.consumables
            ) and _consumable_multiset_matches(
                state.inventory.consumables, effective_obs_cons
            ):
                state.inventory.consumables = _reorder_existing_consumables(
                    state.inventory.consumables, effective_obs_cons
                )
            elif effective_obs_cons:
                state.inventory.consumables = _reconcile_existing_consumables(
                    state.inventory.consumables, effective_obs_cons
                )
            elif next_obs_cons_counts is None or not next_obs_cons_counts:
                state.inventory.consumables = []

        if diag is not None and state.inventory is not None:
            lost_cons, gained_cons = _counter_diff_names(
                prev_cons, state.inventory.consumables
            )
            if suppress_consumable_loss_diag:
                lost_cons = []
            if lost_cons or gained_cons:
                diag["unexplained_consumable_changes"].append(
                    (action.frame_idx, action.type.value, lost_cons, gained_cons)
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


_HAND_SELECTION_ACTIONS: frozenset[ActionType] = frozenset(
    {
        ActionType.PLAY_HAND,
        ActionType.DISCARD,
        ActionType.USE_CONSUMABLE,
        ActionType.USE_IN_PACK,
    }
)


_RANDOM_DESTROY_SPECTRALS = frozenset({"Immolate", "Familiar", "Grim", "Incantation"})


def _collect_pack_surviving_hand_ids(
    state: GameState,
    scheduled_actions: Sequence[ScheduledAction],
    sa_idx: int,
    hand_res_by_frame: dict[int, Any],
) -> list[int]:
    """Identifies card IDs in current hand that still appear in the immediate next pack observation."""
    if not state.hand.cards or sa_idx + 1 >= len(scheduled_actions):
        return []
    next_sa = scheduled_actions[sa_idx + 1]
    cur_sa = scheduled_actions[sa_idx]
    still_in_same_pack = (
        state.location == Location.PACK
        and next_sa.ambient_location == Location.PACK
        and (
            cur_sa.action_type == ActionType.USE_CONSUMABLE
            or state.pack_choices_remaining > 1
        )
    )
    if not still_in_same_pack:
        return []
    next_hand = hand_res_by_frame.get(next_sa.target_frame)
    if next_hand is None or not getattr(next_hand, "cards", None):
        return []
    obs_cards = [_convert_card_result_to_hand_card(c) for c in next_hand.cards]
    matched_ids: list[int] = []
    used_indices: set[int] = set()
    for obs in obs_cards:
        if obs.facing_back:
            continue
        hit_idx = next(
            (
                j
                for j, hc in enumerate(state.hand.cards)
                if j not in used_indices and _cards_match(hc.card, obs.card)
            ),
            None,
        )
        if hit_idx is None:
            hit_idx = next(
                (
                    j
                    for j, hc in enumerate(state.hand.cards)
                    if j not in used_indices
                    and hc.card.rank == obs.card.rank
                    and hc.card.suit == obs.card.suit
                ),
                None,
            )
        if hit_idx is not None:
            used_indices.add(hit_idx)
            cid = state.hand.cards[hit_idx].card.id
            if cid:
                matched_ids.append(cid)
    return matched_ids


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
    live_deck_ids = {dc.id for dc in state.full_deck if dc.id != 0}

    def _is_live_candidate(c: PlayingCard) -> bool:
        return not getattr(c, "destroyed", False) and (
            c.id == 0 or c.id in live_deck_ids
        )

    for i, obs in enumerate(obs_hand):
        if matched_slots[i] is not None or obs.facing_back:
            continue
        for j, ex in enumerate(existing):
            if j in used_existing:
                continue
            cand = ex._real_card or ex.card
            if (
                cand.rank != Rank.NONE
                and _is_live_candidate(cand)
                and _cards_match(cand, obs.card)
            ):
                ex.resolve_face_down(copy.deepcopy(obs.card), lift_facing_back=True)
                matched_slots[i] = ex
                used_existing.add(j)
                break
            if ex.bounds and any(
                _is_live_candidate(c) and _cards_match(c, obs.card)
                for c in ex.bounds.candidate_cards
            ):
                match_cand = next(
                    c
                    for c in ex.bounds.candidate_cards
                    if _is_live_candidate(c) and _cards_match(c, obs.card)
                )
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
                if deck_card.id not in claimed_deck_ids and _cards_match(
                    deck_card, obs.card
                ):
                    rem_match = deck_card
                    break
            if (
                rem_match is None
                and not obs.facing_back
                and obs.card.enhancement != Enhancement.NONE
            ):
                for deck_card in state.remaining_deck:
                    if (
                        deck_card.id not in claimed_deck_ids
                        and deck_card.rank == obs.card.rank
                        and deck_card.suit == obs.card.suit
                        and deck_card.seal == obs.card.seal
                        and deck_card.edition == obs.card.edition
                        and deck_card.enhancement != Enhancement.NONE
                    ):
                        rem_match = deck_card
                        break
            if rem_match is None:
                is_corrupt_crop = (
                    not obs.facing_back
                    and obs.card.enhancement != Enhancement.STONE
                    and (obs.card.rank == Rank.NONE or obs.card.suit == Suit.NONE)
                )
                if is_corrupt_crop:
                    continue
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
    retained_in_prev_order = [ex for j, ex in enumerate(existing) if j in used_existing]
    pre_reorder = retained_in_prev_order + newly_drawn

    if apply_direct_order:
        state.hand.cards = ordered_hand
    else:
        state.hand.cards = pre_reorder
        if action is not None and ordered_hand:
            obj_pos = {id(hc): idx for idx, hc in enumerate(pre_reorder)}
            hand_order = [obj_pos[id(hc)] for hc in ordered_hand]
            if hand_order != list(range(len(ordered_hand))):
                action.hand_order = hand_order

    if action is not None and action.type in _HAND_SELECTION_ACTIONS:
        action.selected_cards = [
            copy.deepcopy(hc.card) for hc in ordered_hand if hc.selected
        ]

    state.reconcile_face_down_adjacencies()

    first_selected = next((hc for hc in ordered_hand if hc.selected), None)
    return first_selected.slot_uid if first_selected is not None else None


def _resolve_video_fps(video_path: str | Path | None, fps: float | None) -> float:
    resolved_fps = fps
    if resolved_fps is None and video_path is not None:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        cap_fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0.0
        cap.release()
        resolved_fps = float(cap_fps) if cap_fps and cap_fps > 0 else 60.0
    if not resolved_fps or resolved_fps <= 0:
        resolved_fps = 60.0
    return resolved_fps


def _resolve_target_card_name(state: GameState, action: Action) -> str | None:
    """Resolves the display name of `action.target_idx` right before `on_action` runs."""
    idx = action.target_idx
    if idx is None or idx < 0:
        return None
    A = ActionType
    if action.type in (A.BUY_CARD, A.BUY_AND_USE):
        pool: Sequence[Any] = state.shop_cards
    elif action.type == A.OPEN_PACK:
        pool = state.shop_packs
    elif action.type == A.REDEEM_VOUCHER:
        pool = state.shop_vouchers
    elif action.type == A.USE_IN_PACK:
        pool = state.pack_cards
    elif action.type == A.USE_CONSUMABLE:
        pool = state.inventory.consumables if state.inventory else []
    elif action.type == A.SELL_CARD:
        jokers = list(state.inventory.jokers) if state.inventory else []
        if action.joker_order and jokers:
            by_id = {j.id: j for j in jokers if j.id != 0}
            if len(action.joker_order) == len(jokers) and all(
                jid in by_id for jid in action.joker_order
            ):
                jokers = [by_id[jid] for jid in action.joker_order]
            elif len(action.joker_order) == len(jokers) and all(
                isinstance(x, int) and 0 <= x < len(jokers) for x in action.joker_order
            ):
                jokers = [jokers[i] for i in action.joker_order]
        consumables = list(state.inventory.consumables) if state.inventory else []
        pool = jokers + consumables
    else:
        return None

    if idx < len(pool):
        return _display_card_name(pool[idx])
    return None


def _serialize_shop_shelves(state: GameState) -> dict[str, list[dict[str, Any]]]:
    """Serializes `state`'s visible shop shelves for pre-action metadata snapshots."""
    from game.trajectory_codec import (
        _serialize_booster_pack,
        _serialize_card,
        _serialize_voucher,
    )

    return {
        "cards": [_serialize_card(c) for c in state.shop_cards],
        "vouchers": [_serialize_voucher(v) for v in state.shop_vouchers],
        "packs": [_serialize_booster_pack(bp) for bp in state.shop_packs],
    }


def _build_run_trajectory_with_plan(
    d: dict[str, Any],
    video_path: str | Path | None = None,
    *,
    preloaded_frames: dict[int, Any] | None = None,
    seed: str = "",
    deck: Deck = Deck.RED_DECK,
    stake: Stake = Stake.WHITE,
    fps: float | None = None,
    conf_threshold_hand: float = 0.50,
    conf_threshold_shop: float = 0.15,
    conf_threshold_ocr: float = 0.90,
    batch_size: int = 32,
    device: str | None = None,
    deduplicate: bool = True,
    expressive: bool = False,
) -> tuple[GameState, RunTrajectory, dict[str, Any]]:
    from classifiers.common.video_utils import extract_video_frames

    diag = _init_diagnostics() if expressive else None
    resolved_fps = _resolve_video_fps(video_path, fps)
    dedup_frames = remove_duplicates(d["frames"]) if deduplicate else d["frames"]
    cleaned_input = {
        "key": d["key"],
        "frames": filter_premature_next_rounds(dedup_frames, d["key"], diag=diag),
    }
    if diag is not None:
        diag["end_round_segments_checked"] = len(
            find_missing_end_round_segments(cleaned_input)
        )
        pre_frames_count = len(cleaned_input["frames"])

    t0 = time.perf_counter()
    cleaned_input = inject_end_round_actions(
        cleaned_input,
        video_path=video_path,
        preloaded_frames=preloaded_frames,
        fps=resolved_fps,
        batch_size=max(1, batch_size // 4),
        device=device,
        expressive=expressive,
    )
    if diag is not None:
        diag["timings"]["end_round"] = time.perf_counter() - t0
        diag["end_rounds_injected"] = max(
            0, len(cleaned_input["frames"]) - pre_frames_count
        )

    plan = plan_action_pipeline(cleaned_input)
    scheduled_actions: list[ScheduledAction] = plan["scheduled_actions"]

    needed_frames = sorted(
        set(plan["playing_card_cv_queue"])
        | set(plan["joker_cv_queue"])
        | set(plan["shop_cv_queue"])
        | set(plan["pack_cv_queue"])
        | set(plan["hud_ocr_queue"])
    )
    t1 = time.perf_counter()
    if preloaded_frames is not None:
        frames_by_idx = {
            idx: preloaded_frames[idx]
            for idx in needed_frames
            if idx in preloaded_frames
        }
    elif video_path is not None and needed_frames:
        if expressive:
            with tqdm(
                total=len(needed_frames), desc="Extracting CV frames", unit="frame"
            ) as ex_bar:
                frames_by_idx = extract_video_frames(
                    video_path,
                    needed_frames,
                    progress_callback=ex_bar.update,
                )
        else:
            frames_by_idx = extract_video_frames(video_path, needed_frames)
    else:
        frames_by_idx = {}
    if diag is not None:
        diag["timings"]["extract_frames"] = time.perf_counter() - t1
        diag["missing_frames"] = [
            fid for fid in needed_frames if fid not in frames_by_idx
        ]

    t2 = time.perf_counter()
    shop_res_by_frame, hand_res_by_frame, hud_res_by_frame = _run_batched_cv(
        scheduled_actions,
        plan,
        frames_by_idx,
        fps=resolved_fps,
        conf_threshold_hand=conf_threshold_hand,
        conf_threshold_shop=conf_threshold_shop,
        conf_threshold_ocr=conf_threshold_ocr,
        batch_size=batch_size,
        device=device,
        expressive=expressive,
    )
    if diag is not None:
        diag["timings"]["batched_cv_ocr"] = time.perf_counter() - t2

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

    t3 = time.perf_counter()
    A = ActionType
    targeted_actions = {
        A.BUY_CARD,
        A.BUY_AND_USE,
        A.OPEN_PACK,
        A.REDEEM_VOUCHER,
        A.USE_IN_PACK,
        A.USE_CONSUMABLE,
        A.SELL_CARD,
    }
    obs_cons_counts_by_sa_idx: dict[int, Counter[tuple[Any, Any]]] = {}
    for i, s_act in enumerate(scheduled_actions):
        s_out = shop_res_by_frame.get(s_act.target_frame)
        if s_out is not None:
            raw_c = list(getattr(s_out, "consumeables", None) or [])
            obs_c = _convert_items_of_type(raw_c, (Tarot, Planet, Spectral))
            obs_cons_counts_by_sa_idx[i] = Counter(_consumable_sig(c) for c in obs_c)

    def _next_obs_cons_counts(cur_idx: int) -> Counter[tuple[Any, Any]] | None:
        for j in range(cur_idx + 1, len(scheduled_actions)):
            if j in obs_cons_counts_by_sa_idx:
                return obs_cons_counts_by_sa_idx[j]
        return None

    action_iter = (
        tqdm(
            enumerate(scheduled_actions),
            total=len(scheduled_actions),
            desc="Simulating trajectory",
            unit="action",
        )
        if expressive
        else enumerate(scheduled_actions)
    )
    for sa_idx, sa in action_iter:
        is_face_down_boss = (
            state.current_blind in FACE_DOWN_BOSS_BLINDS and not state.boss_disabled
        )
        if sa.action_type in (A.SORT_RANK, A.SORT_SUIT) and not is_face_down_boss:
            continue

        action = Action(
            type=sa.action_type,
            frame_idx=sa.frame_id,
            timestamp=sa.frame_id / resolved_fps,
            metadata=dict(sa.metadata),
        )
        # Preserve CASH_OUT round payout calculation unless caller explicitly provided OCR override
        if (
            sa.action_type == A.CASH_OUT
            and "include_round_payout" not in action.metadata
        ):
            action.metadata["include_round_payout"] = True

        # Ensure blind transition leaves no stale hand cards from the prior round
        if sa.action_type in (A.SELECT_BLIND, A.SKIP_BLIND, A.CASH_OUT):
            state.hand.cards.clear()

        shop_out = shop_res_by_frame.get(sa.target_frame)
        if state.location == Location.PACK and sa.ambient_location in (
            Location.SHOP,
            Location.SELECT,
        ):
            if (
                state.active_pack is not None
                and state.active_pack.booster_type == BoosterType.BUFFOON
                and shop_out is not None
                and state.inventory is not None
            ):
                obs_j_list = _convert_items_of_type(
                    list(getattr(shop_out, "joker_bar", None) or []), Joker
                )
                prev_counts = Counter(
                    (j.name, j.edition, j.eternal, j.perishable, j.rental)
                    for j in state.inventory.jokers
                )
                gained_from_pack: list[Joker] = []
                for oj in obs_j_list:
                    sig = (oj.name, oj.edition, oj.eternal, oj.perishable, oj.rental)
                    if prev_counts[sig] > 0:
                        prev_counts[sig] -= 1
                    else:
                        gained_from_pack.append(oj)
                if 1 <= len(gained_from_pack) <= max(1, state.active_pack.choose):
                    for gj in gained_from_pack:
                        new_j = copy.deepcopy(gj)
                        state.inventory.jokers.append(new_j)
                        on_joker_added(state, new_j)
                    state.ensure_joker_ids()
            state.active_pack = None
            state.pack_cards.clear()
            state.pack_choices_remaining = 0
            if getattr(state, "pending_packs", None):
                state.pending_packs.clear()
            state.location = sa.ambient_location
            state.pre_pack_location = None

        raw_jokers: list[Any] = []
        raw_cons: list[Any] = []
        next_cons_counts = _next_obs_cons_counts(sa_idx)
        if shop_out is not None:
            # Retroactively infer missed USE_CONSUMABLE actions when a held consumable
            # disappears in both the current and next observed CV frame.
            if state.inventory and state.inventory.consumables:
                cur_cons_counts = obs_cons_counts_by_sa_idx.get(sa_idx, Counter())
                prev_frame_id = (
                    scheduled_actions[sa_idx - 1].frame_id if sa_idx > 0 else 0
                )
                inferred_frame = max(prev_frame_id + 1, sa.frame_id - 1)
                hud_for_infer = hud_res_by_frame.get(sa.target_frame)
                hud_money_val = (
                    getattr(hud_for_infer, "hud_money", None)
                    if hud_for_infer is not None
                    else None
                )
                idx_cursor = 0
                while idx_cursor < len(state.inventory.consumables):
                    held_c = state.inventory.consumables[idx_cursor]
                    sig = _consumable_sig(held_c)
                    held_sig_total = sum(
                        1
                        for c in state.inventory.consumables
                        if _consumable_sig(c) == sig
                    )
                    next_count = (
                        next_cons_counts[sig] if next_cons_counts is not None else 0
                    )
                    if (
                        cur_cons_counts[sig] < held_sig_total
                        and next_count < held_sig_total
                    ):
                        sell_val = max(1, int(getattr(held_c, "sell_value", 1)))
                        c_name = getattr(held_c, "name", None)
                        is_sell = (
                            hud_money_val is not None
                            and hud_money_val >= state.money + sell_val
                            and c_name not in (TarotName.HERMIT, TarotName.TEMPERANCE)
                        )
                        if is_sell:
                            joker_offset = len(state.inventory.jokers)
                            inferred_action = Action(
                                type=A.SELL_CARD,
                                frame_idx=inferred_frame,
                                timestamp=inferred_frame / resolved_fps,
                                target_idx=joker_offset + idx_cursor,
                                metadata={
                                    "inferred": True,
                                    "target_card": str(c_name or ""),
                                },
                            )
                        else:
                            inferred_action = Action(
                                type=A.USE_CONSUMABLE,
                                frame_idx=inferred_frame,
                                timestamp=inferred_frame / resolved_fps,
                                target_idx=idx_cursor,
                                metadata={
                                    "inferred": True,
                                    "target_card": str(c_name or ""),
                                },
                            )
                        state.frame_idx = inferred_action.frame_idx
                        state.timestamp = inferred_action.timestamp
                        on_action(state, inferred_action)
                        trajectory.record_step(state, inferred_action)
                    else:
                        idx_cursor += 1

            raw_jokers, raw_cons = _sync_bars_into_state(
                state,
                shop_out,
                action,
                diag=diag,
                next_obs_cons_counts=next_cons_counts,
                sync_jokers=True,
                sync_consumables=(sa.action_type != A.END_ROUND),
            )

        # Sync main shop shelves before matching/stepping shop actions
        if sa.ambient_location == Location.SHOP and shop_out is not None:
            pre_cv_shelves = _serialize_shop_shelves(state)
            raw_shop_items = list(getattr(shop_out, "shop_items", None) or [])
            raw_vouchers = list(getattr(shop_out, "vouchers", None) or [])
            raw_boosters = list(getattr(shop_out, "boosters", None) or [])

            state.shop_cards = _convert_items_of_type(raw_shop_items)
            state.shop_vouchers = _convert_items_of_type(raw_vouchers, Voucher)
            state.shop_packs = _convert_items_of_type(raw_boosters, BoosterPack)
            if getattr(state, "coupon_shop_active", False):
                for item in state.shop_cards + state.shop_packs:
                    item.couponed = True
            post_cv_shelves = _serialize_shop_shelves(state)
            if any(post_cv_shelves.values()) and (
                post_cv_shelves != pre_cv_shelves
                or sa.action_type in (A.NEXT_ROUND, A.SELECT_BLIND)
            ):
                action.metadata["observed_shop"] = post_cv_shelves

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
            if pack_objs:
                from game.trajectory_codec import _serialize_card

                action.metadata["observed_pack"] = [
                    _serialize_card(p) for p in pack_objs
                ]
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
                obs_pre = [_convert_card_result_to_hand_card(c) for c in pre_hand.cards]
                pre_selected_uid = _sync_hand_into_state(state, obs_pre, action=action)
                if pre_selected_uid is not None:
                    action.metadata["selected_slot_uid"] = pre_selected_uid
        else:
            # If a round transition was missed, return prior round's discard/played cards
            # to draw_pile before syncing the new round's hand so full_deck doesn't bloat.
            hud_pre = hud_res_by_frame.get(sa.target_frame)
            hud_r_val = (
                getattr(hud_pre, "hud_round", None) if hud_pre is not None else None
            )
            if (
                hud_r_val is not None
                and hud_r_val > state.round
                and (state.discard_pile or state.played_cards_this_round)
            ):
                state.hand.cards.clear()
                state.discard_pile.clear()
                state.played_cards_this_round.clear()
                state.clear_boss_round_modifiers()
            hand_out = hand_res_by_frame.get(sa.target_frame)
            if hand_out is not None:
                obs_cards = [
                    _convert_card_result_to_hand_card(c) for c in hand_out.cards
                ]
                _sync_hand_into_state(state, obs_cards, action=action)
            if diag is not None and sa.action_type in (A.PLAY_HAND, A.DISCARD):
                if hand_out is None or not getattr(hand_out, "cards", None):
                    diag["empty_hand_cv_actions"].append(sa.frame_id)

        if diag is not None:
            if (
                sa.ambient_location in (Location.SHOP, Location.PACK)
                and shop_out is None
            ):
                diag["missing_shop_cv_actions"].append(sa.frame_id)
            if sa.action_type in targeted_actions and action.target_idx is None:
                diag["unmatched_button_targets"].append(
                    (sa.frame_id, sa.action_type.value, sa.ambient_location.value)
                )
            expected_loc = (
                Location.BLIND
                if sa.ambient_location == Location.CASH_OUT
                else sa.ambient_location
            )
            if expected_loc != state.location:
                diag["location_mismatches"].append(
                    (
                        sa.frame_id,
                        sa.action_type.value,
                        sa.ambient_location.value,
                        state.location.value,
                    )
                )

        if sa.action_type in targeted_actions and "target_card" not in action.metadata:
            target_card_name = _resolve_target_card_name(state, action)
            if target_card_name is not None:
                action.metadata["target_card"] = target_card_name

        # Sync HUD OCR counters before deterministic simulation (falling back on None predictions)
        hud_out = hud_res_by_frame.get(sa.target_frame)
        deferred_round_score: int | float | None = None
        if hud_out is not None:
            deferred_round_score = _sync_hud_into_state(
                state,
                hud_out,
                sa,
                action,
                has_end_round_after=_has_end_round_after(scheduled_actions, sa_idx),
                diag=diag,
            )

        if (
            sa.action_type in (A.USE_CONSUMABLE, A.USE_IN_PACK)
            and action.metadata.get("target_card") in _RANDOM_DESTROY_SPECTRALS
        ):
            survivors = _collect_pack_surviving_hand_ids(
                state, scheduled_actions, sa_idx, hand_res_by_frame
            )
            if survivors:
                action.metadata["_surviving_hand_ids"] = survivors

        # Advance simulation
        state.frame_idx = action.frame_idx
        state.timestamp = action.timestamp
        prev_loc = state.location
        on_action(state, action)
        action.metadata.pop("_surviving_hand_ids", None)
        if sa.action_type == A.END_ROUND and shop_out is not None:
            _, raw_cons = _sync_bars_into_state(
                state,
                shop_out,
                action,
                diag=diag,
                next_obs_cons_counts=next_cons_counts,
                sync_jokers=False,
                sync_consumables=True,
            )
        if deferred_round_score is not None:
            if diag is not None and state.current_chips != deferred_round_score:
                diag["ocr_score_updates"] += 1
            state.current_chips = deferred_round_score

        # Post-sort reconciliation aligns actual settled positions + tightens face-down CardBounds
        if (
            is_face_down_boss
            and sa.action_type in (A.SORT_RANK, A.SORT_SUIT)
            and sa.post_sort_frame is not None
            and sa.post_sort_frame in hand_res_by_frame
        ):
            post_hand = hand_res_by_frame[sa.post_sort_frame]
            obs_post = [_convert_card_result_to_hand_card(c) for c in post_hand.cards]
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
    if diag is not None:
        diag["timings"]["simulation"] = time.perf_counter() - t3
        plan["diagnostics"] = diag
        _print_expressive_report(diag)
    return state, trajectory, plan


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
    conf_threshold_shop: float = 0.15,
    conf_threshold_ocr: float = 0.90,
    batch_size: int = 32,
    device: str | None = None,
    deduplicate: bool = True,
    expressive: bool = False,
) -> tuple[GameState, RunTrajectory]:
    """Reconstructs a complete `(GameState, RunTrajectory)` from action JSON + video frames.

    Note: Starting mid-run is out of scope; always initializes from Ante 1 / Round 1.
    """
    state, trajectory, _ = _build_run_trajectory_with_plan(
        d,
        video_path=video_path,
        preloaded_frames=preloaded_frames,
        seed=seed,
        deck=deck,
        stake=stake,
        fps=fps,
        conf_threshold_hand=conf_threshold_hand,
        conf_threshold_shop=conf_threshold_shop,
        conf_threshold_ocr=conf_threshold_ocr,
        batch_size=batch_size,
        device=device,
        deduplicate=deduplicate,
        expressive=expressive,
    )
    return state, trajectory


def process_actions(
    d: dict[str, Any],
    video_path: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Schedules CV queues for action frames and optionally reconstructs full GameState trajectory.

    When `video_path` and `preloaded_frames` are omitted, returns the 5 CV queue dicts.
    When either is provided, automatically detects missing `END_ROUND` transitions, runs batched
    CV + HUD OCR, and returns the scheduled queues alongside `state` and `trajectory`.
    """
    if video_path is not None or kwargs.get("preloaded_frames") is not None:
        state, trajectory, plan = _build_run_trajectory_with_plan(
            d, video_path=video_path, **kwargs
        )
        return {
            "playing_card_cv_queue": plan["playing_card_cv_queue"],
            "joker_cv_queue": plan["joker_cv_queue"],
            "shop_cv_queue": plan["shop_cv_queue"],
            "pack_cv_queue": plan["pack_cv_queue"],
            "hud_ocr_queue": plan["hud_ocr_queue"],
            "state": state,
            "trajectory": trajectory,
        }
    plan = plan_action_pipeline(d)
    return {
        "playing_card_cv_queue": plan["playing_card_cv_queue"],
        "joker_cv_queue": plan["joker_cv_queue"],
        "shop_cv_queue": plan["shop_cv_queue"],
        "pack_cv_queue": plan["pack_cv_queue"],
        "hud_ocr_queue": plan["hud_ocr_queue"],
    }


def _normalize_enum_token(value: str) -> str:
    return "".join(ch for ch in value.strip().lower() if ch.isalnum())


def parse_deck(value: str | Deck) -> Deck:
    """Parses a CLI deck string (e.g. 'red', 'Red Deck', 'PLASMA_DECK') into `Deck`."""
    if isinstance(value, Deck):
        return value
    norm = _normalize_enum_token(value)
    for member in Deck:
        if norm in (
            _normalize_enum_token(member.name),
            _normalize_enum_token(member.value),
        ):
            return member
        short_name = _normalize_enum_token(member.name.removesuffix("_DECK"))
        if norm == short_name:
            return member
    valid = ", ".join(
        m.name.removesuffix("_DECK").lower() for m in Deck if m != Deck.CHALLENGE_DECK
    )
    raise ValueError(f"Unknown deck {value!r}. Valid choices: {valid}")


def parse_stake(value: str | Stake) -> Stake:
    """Parses a CLI stake string (e.g. 'white', 'White Stake', 'GOLD') into `Stake`."""
    if isinstance(value, Stake):
        return value
    norm = _normalize_enum_token(value)
    for member in Stake:
        if norm in (
            _normalize_enum_token(member.name),
            _normalize_enum_token(member.value),
        ):
            return member
    valid = ", ".join(m.name.lower() for m in Stake)
    raise ValueError(f"Unknown stake {value!r}. Valid choices: {valid}")


def serialize_trajectory_summary(
    state: GameState, trajectory: RunTrajectory
) -> dict[str, Any]:
    """Serializes reconstructed `(GameState, RunTrajectory)` into sparse delta JSON."""
    from game.trajectory_codec import serialize_trajectory

    return serialize_trajectory(state, trajectory)


def _build_post_processing_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Reconstruct Balatro GameState trajectory from action JSON + video."
    )
    p.add_argument(
        "--actions-json",
        type=str,
        required=True,
        help="Path to main.py action detection JSON",
    )
    p.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to gameplay video (.mp4/.mkv/.avi)",
    )
    p.add_argument("--seed", type=str, default="", help="Optional run seed")
    p.add_argument(
        "--deck",
        type=str,
        default="red",
        help="Deck name (e.g. red, blue, plasma, abandoned)",
    )
    p.add_argument(
        "--stake", type=str, default="white", help="Stake name (e.g. white, red, gold)"
    )
    p.add_argument("--fps", type=float, default=None, help="Override source video FPS")
    p.add_argument(
        "--conf-hand",
        type=float,
        default=0.50,
        help="Hand card detector confidence threshold",
    )
    p.add_argument(
        "--conf-shop",
        type=float,
        default=0.15,
        help="Shop detector confidence threshold",
    )
    p.add_argument(
        "--conf-ocr", type=float, default=0.90, help="HUD OCR confidence threshold"
    )
    p.add_argument("--batch-size", type=int, default=32, help="CV inference batch size")
    p.add_argument(
        "--device", type=str, default=None, help="Torch device (e.g. cuda, cpu)"
    )
    p.add_argument(
        "--expressive",
        action="store_true",
        help="Show progress bars and end-of-run diagnostics (OCR drift overrides, mismatches, timings)",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output JSON path for trajectory summary",
    )
    return p


def default_trajectory_output_path(actions_path: str | Path) -> Path:
    """Resolves default trajectory JSON path under a `trajectories/` folder."""
    ap = Path(actions_path)
    traj_dir = (
        ap.parent.parent / "trajectories"
        if ap.parent.name == "vod_actions"
        else ap.parent / "trajectories"
    )
    return traj_dir / f"{ap.stem}_trajectory.json"


if __name__ == "__main__":
    configure_logging()
    args = _build_post_processing_arg_parser().parse_args()
    with open(args.actions_json, encoding="utf-8") as f:
        data = json.load(f)

    state, trajectory = build_run_trajectory(
        data,
        video_path=args.video,
        seed=args.seed,
        deck=parse_deck(args.deck),
        stake=parse_stake(args.stake),
        fps=args.fps,
        conf_threshold_hand=args.conf_hand,
        conf_threshold_shop=args.conf_shop,
        conf_threshold_ocr=args.conf_ocr,
        batch_size=args.batch_size,
        device=args.device,
        expressive=args.expressive,
    )
    summary = serialize_trajectory_summary(state, trajectory)
    out_path = (
        Path(args.out)
        if args.out
        else default_trajectory_output_path(args.actions_json)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(
        f"Reconstructed {summary['num_steps']} steps | "
        f"Ante {state.ante} Round {state.round} ({state.current_blind.value}) | "
        f"Money=${state.money} | Saved: {out_path}"
    )
