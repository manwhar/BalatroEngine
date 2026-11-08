"""
Unit tests and telemetry parity validation for Balatro GameState simulation engine.

Usage:
    python -m unittest testing/test_engine.py
    python testing/test_engine.py
"""

import json
import os
from pathlib import Path
import sys
import unittest

# Ensure repo root and 'game' directory are in pythonpath
REPO_ROOT = Path(__file__).resolve().parent.parent
GAME_DIR = REPO_ROOT / "game"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(GAME_DIR) not in sys.path:
    sys.path.insert(0, str(GAME_DIR))

try:
    import game.balatro_models as bm
    import game.gamestate as gs
    from game.engine.jokers import on_joker_added, sync_dynamic_joker_states
except ImportError:
    import balatro_models as bm
    import gamestate as gs
    from engine.jokers import on_joker_added, sync_dynamic_joker_states


# =========================================================================
# LUA TELEMETRY <-> PYTHON MODEL DESERIALIZERS
# =========================================================================

RANK_MAP: dict[str, bm.Rank] = {
    "2": bm.Rank.TWO,
    "3": bm.Rank.THREE,
    "4": bm.Rank.FOUR,
    "5": bm.Rank.FIVE,
    "6": bm.Rank.SIX,
    "7": bm.Rank.SEVEN,
    "8": bm.Rank.EIGHT,
    "9": bm.Rank.NINE,
    "10": bm.Rank.TEN,
    "Jack": bm.Rank.JACK,
    "Queen": bm.Rank.QUEEN,
    "King": bm.Rank.KING,
    "Ace": bm.Rank.ACE,
    "none": bm.Rank.NONE,
}

SUIT_MAP: dict[str, bm.Suit] = {
    "Spades": bm.Suit.SPADES,
    "Hearts": bm.Suit.HEARTS,
    "Clubs": bm.Suit.CLUBS,
    "Diamonds": bm.Suit.DIAMONDS,
    "none": bm.Suit.NONE,
}

ENHANCEMENT_MAP: dict[str, bm.Enhancement] = {
    "none": bm.Enhancement.NONE,
    "bonus": bm.Enhancement.BONUS,
    "mult": bm.Enhancement.MULT,
    "wild": bm.Enhancement.WILD,
    "glass": bm.Enhancement.GLASS,
    "steel": bm.Enhancement.STEEL,
    "stone": bm.Enhancement.STONE,
    "gold": bm.Enhancement.GOLD,
    "lucky": bm.Enhancement.LUCKY,
}

EDITION_MAP: dict[str, bm.Edition] = {
    "none": bm.Edition.NONE,
    "foil": bm.Edition.FOIL,
    "holographic": bm.Edition.HOLOGRAPHIC,
    "polychrome": bm.Edition.POLYCHROME,
    "negative": bm.Edition.NEGATIVE,
}

SEAL_MAP: dict[str, bm.Seal] = {
    "none": bm.Seal.NONE,
    "red": bm.Seal.RED,
    "blue": bm.Seal.BLUE,
    "gold": bm.Seal.GOLD,
    "purple": bm.Seal.PURPLE,
}

JOKER_NAME_BY_STR: dict[str, bm.JokerName] = {j.value: j for j in bm.JokerName}
JOKER_NAME_BY_STR.update(
    {
        "Caino": bm.JokerName.CANIO,
        "Seance": bm.JokerName.SEANCE,
        "Riff-raff": bm.JokerName.RIFF_RAFF,
    }
)

BLIND_BY_STR: dict[str, bm.BlindName] = {b.value: b for b in bm.BlindName}
BLIND_BY_STR.update(
    {
        "bl_small": bm.BlindName.SMALL,
        "bl_big": bm.BlindName.BIG,
        "bl_hook": bm.BlindName.THE_HOOK,
        "bl_ox": bm.BlindName.THE_OX,
        "bl_house": bm.BlindName.THE_HOUSE,
        "bl_wall": bm.BlindName.THE_WALL,
        "bl_wheel": bm.BlindName.THE_WHEEL,
        "bl_arm": bm.BlindName.THE_ARM,
        "bl_club": bm.BlindName.THE_CLUB,
        "bl_fish": bm.BlindName.THE_FISH,
        "bl_psychic": bm.BlindName.THE_PSYCHIC,
        "bl_goad": bm.BlindName.THE_GOAD,
        "bl_water": bm.BlindName.THE_WATER,
        "bl_window": bm.BlindName.THE_WINDOW,
        "bl_manacle": bm.BlindName.THE_MANACLE,
        "bl_eye": bm.BlindName.THE_EYE,
        "bl_mouth": bm.BlindName.THE_MOUTH,
        "bl_plant": bm.BlindName.THE_PLANT,
        "bl_serpent": bm.BlindName.THE_SERPENT,
        "bl_pillar": bm.BlindName.THE_PILLAR,
        "bl_needle": bm.BlindName.THE_NEEDLE,
        "bl_head": bm.BlindName.THE_HEAD,
        "bl_tooth": bm.BlindName.THE_TOOTH,
        "bl_flint": bm.BlindName.THE_FLINT,
        "bl_mark": bm.BlindName.THE_MARK,
        "bl_final_acorn": bm.BlindName.AMBER_ACORN,
        "bl_final_leaf": bm.BlindName.VERDANT_LEAF,
        "bl_final_vessel": bm.BlindName.VIOLET_VESSEL,
        "bl_final_heart": bm.BlindName.CRIMSON_HEART,
        "bl_final_bell": bm.BlindName.CERULEAN_BELL,
    }
)

POKER_HAND_BY_STR: dict[str, bm.PokerHand] = {h.value: h for h in bm.PokerHand}
TAROT_NAME_BY_STR: dict[str, bm.TarotName] = {t.value: t for t in bm.TarotName}
PLANET_NAME_BY_STR: dict[str, bm.PlanetName] = {p.value: p for p in bm.PlanetName}
SPECTRAL_NAME_BY_STR: dict[str, bm.SpectralName] = {s.value: s for s in bm.SpectralName}
VOUCHER_NAME_BY_STR: dict[str, bm.VoucherName] = {v.value: v for v in bm.VoucherName}


def parse_playing_card(d: dict) -> bm.PlayingCard:
    enh = ENHANCEMENT_MAP.get(d.get("enhancement", "none"), bm.Enhancement.NONE)
    rank = (
        bm.Rank.NONE
        if enh == bm.Enhancement.STONE
        else RANK_MAP.get(d.get("value", "none"), bm.Rank.ACE)
    )
    suit = (
        bm.Suit.NONE
        if enh == bm.Enhancement.STONE
        else SUIT_MAP.get(d.get("suit", "none"), bm.Suit.SPADES)
    )
    return bm.PlayingCard(
        id=d.get("id", 0),
        rank=rank,
        suit=suit,
        enhancement=enh,
        edition=EDITION_MAP.get(d.get("edition", "none"), bm.Edition.NONE),
        seal=SEAL_MAP.get(d.get("seal", "none"), bm.Seal.NONE),
        bonus_chips=int(d.get("bonus_chips", 0)),
        debuffed=bool(d.get("debuffed", False)),
        extra_value=int(d.get("extra_value", 0)),
    )


def parse_hand_card(d: dict) -> bm.HandCard:
    return bm.HandCard(
        card=parse_playing_card(d),
        selected=bool(d.get("selected", False)),
        facing_back=bool(d.get("facing_back", False)),
        forced_selection=bool(d.get("forced_selection", False)),
    )


def parse_joker(d: dict) -> bm.Joker | None:
    name_str = d.get("name", "")
    j_name = JOKER_NAME_BY_STR.get(name_str)
    if j_name is None:
        return None
    j = bm.Joker(
        name=j_name,
        id=int(d.get("id", 0)),
        edition=EDITION_MAP.get(d.get("edition", "none"), bm.Edition.NONE),
        extra_value=int(d.get("extra_value", 0)),
        debuffed=bool(d.get("debuffed", False)),
        eternal=bool(d.get("eternal", False)),
        perishable=bool(d.get("perishable", False)),
        perish_tally=int(d.get("perish_tally", 5)),
        rental=bool(d.get("rental", False)),
        facing_back=bool(d.get("facing_back", False)),
    )
    if d.get("chips"):
        j.state["chips"] = d["chips"]
    if d.get("mult"):
        j.state["mult"] = d["mult"]
    if d.get("x_mult") and d["x_mult"] != 1.0:
        j.state["x_mult"] = d["x_mult"]
    if d.get("h_size"):
        j.state["h_size"] = d["h_size"]
    if d.get("yorick_discards") is not None:
        j.state["discards"] = d["yorick_discards"]
    if d.get("invis_rounds"):
        j.state["rounds"] = d["invis_rounds"]
    if d.get("to_do_poker_hand") in POKER_HAND_BY_STR:
        j.state["target_hand"] = POKER_HAND_BY_STR[d["to_do_poker_hand"]]
    if d.get("loyalty_remaining") is not None:
        j.state["remaining"] = int(d["loyalty_remaining"])
    if d.get("payout") is not None:
        j.state["payout"] = int(d["payout"])
    extra = d.get("extra")
    if isinstance(extra, dict):
        if "dollars" in extra and j_name == bm.JokerName.ROCKET:
            try:
                j.state["payout"] = int(extra["dollars"])
            except (ValueError, TypeError):
                pass
        if "remaining" in extra and j_name == bm.JokerName.LOYALTY_CARD and "remaining" not in j.state:
            val = extra["remaining"]
            if isinstance(val, (int, float)):
                j.state["remaining"] = int(val)
            elif isinstance(val, str):
                import re

                m = re.search(r"\d+", val)
                if m:
                    j.state["remaining"] = int(m.group(0))
    return j


def parse_any_card(d: dict):
    if not isinstance(d, dict):
        return None
    s = d.get("set", "")
    ed = EDITION_MAP.get(d.get("edition", "none"), bm.Edition.NONE)
    ev = int(d.get("extra_value", 0))
    if s == "Joker":
        return parse_joker(d)
    elif s == "Tarot":
        t_name = TAROT_NAME_BY_STR.get(d.get("name"), bm.TarotName.FOOL)
        return bm.Tarot(name=t_name, edition=ed, extra_value=ev)
    elif s == "Planet":
        p_name = PLANET_NAME_BY_STR.get(d.get("name"), bm.PlanetName.MERCURY)
        return bm.Planet(name=p_name, edition=ed, extra_value=ev)
    elif s == "Spectral":
        sp_name = SPECTRAL_NAME_BY_STR.get(d.get("name"), bm.SpectralName.FAMILIAR)
        return bm.Spectral(name=sp_name, edition=ed, extra_value=ev)
    elif s == "Voucher":
        v_name = VOUCHER_NAME_BY_STR.get(d.get("name"), bm.VoucherName.OVERSTOCK)
        return bm.Voucher(name=v_name)
    elif s in ("Default", "Enhanced") or d.get("value") in RANK_MAP:
        return parse_playing_card(d)
    return None


def parse_lua_action(
    action_name: str, meta: dict | list, frame_idx: int, timestamp: float
) -> bm.Action:
    """Maps a Lua telemetry action to a Python Action model."""
    meta_dict = dict(meta) if isinstance(meta, dict) else {}
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
        "Use Consumable": bm.ActionType.USE_CONSUMABLE,
        "Open Pack": bm.ActionType.OPEN_PACK,
        "Use In Pack": bm.ActionType.USE_IN_PACK,
        "Skip Pack": bm.ActionType.SKIP_PACK,
        "Redeem Voucher": bm.ActionType.REDEEM_VOUCHER,
        "Next Round": bm.ActionType.NEXT_ROUND,
        "Reroll Boss": bm.ActionType.REROLL_BOSS,
        "Sort by Rank": bm.ActionType.SORT_RANK,
        "Sort by Suit": bm.ActionType.SORT_SUIT,
        "Reorder Jokers": bm.ActionType.REORDER_JOKERS,
        "Reorder Hand": bm.ActionType.REORDER_HAND,
        "Setup Test Suite": bm.ActionType.SELECT_CARD,
    }
    act_type = action_map.get(action_name, bm.ActionType.PLAY_HAND)
    if act_type == bm.ActionType.CASH_OUT:
        meta_dict.setdefault("include_round_payout", True)
    if act_type in (bm.ActionType.SORT_RANK, bm.ActionType.SORT_SUIT):
        sel_ids = meta_dict.get("selected_card_ids")
        if sel_ids and isinstance(sel_ids, list) and len(sel_ids) > 0:
            meta_dict.setdefault("selected_card_id", sel_ids[0])
    target_idx = meta_dict.get("target_idx")
    joker_order = meta_dict.get("joker_order")
    return bm.Action(
        type=act_type,
        frame_idx=frame_idx,
        timestamp=timestamp,
        target_idx=target_idx if isinstance(target_idx, int) else None,
        joker_order=joker_order if isinstance(joker_order, list) else [],
        metadata=meta_dict,
    )


def sync_cv_observations_before_action(
    sim_state: gs.GameState, lua_state: dict, act_name: str = ""
) -> None:
    """Simulates a Perfect-CV/OCR camera observing ONLY screen-visible inputs at action time.

    Does NOT touch closed-loop engine state (money, hands_left, discards_left,
    reroll_cost, discard_pile, deck_cards_count) so the test harness can verify
    that the Python engine predicts them without ground truth.
    """
    # 0. Sync deterministic seed if discovered dynamically
    if not sim_state.seed and lua_state.get("seed"):
        sim_state.set_seed(lua_state["seed"])

    # 1. Location, boss disabled flag, inflation, and pack choices
    loc_str = lua_state.get("location")
    if loc_str in ("blind", "shop", "select", "pack"):
        sim_state.location = bm.Location(loc_str)
    if "boss_disabled" in lua_state:
        sim_state.boss_disabled = bool(lua_state["boss_disabled"])
    if "inflation" in lua_state:
        sim_state.inflation = int(lua_state["inflation"])
    if "pack_choices_remaining" in lua_state:
        sim_state.pack_choices_remaining = int(lua_state["pack_choices_remaining"])
    if "reroll_cost" in lua_state:
        sim_state.reroll_cost = int(lua_state["reroll_cost"])
    if "active_tags" in lua_state:
        tags = []
        for t in lua_state["active_tags"]:
            tag_name = t.get("name") if isinstance(t, dict) else str(t)
            try:
                tags.append(bm.SkipTag(tag_name))
            except ValueError:
                pass
        sim_state.active_tags = tags

    # 2. Dealt hand cards & player card selection (visible on screen)
    if "hand" in lua_state:
        sim_state.hand.cards = [parse_hand_card(c) for c in lua_state["hand"]]

    # 2. Current blind & boss blind (visible on HUD)
    blind_str = lua_state.get("blind", "")
    if blind_str in BLIND_BY_STR:
        sim_state.current_blind = BLIND_BY_STR[blind_str]
    boss_str = lua_state.get("boss_blind", "")
    if boss_str in BLIND_BY_STR:
        sim_state.boss_blind = BLIND_BY_STR[boss_str]
    if "target_chips" in lua_state:
        sim_state.target_chips = int(lua_state["target_chips"])
    if "current_chips" in lua_state:
        sim_state.current_chips = int(lua_state["current_chips"])

    # 3. Rotating target suits/ranks (visible on joker tooltips)
    if lua_state.get("castle_suit") in SUIT_MAP:
        sim_state.castle_suit = SUIT_MAP[lua_state["castle_suit"]]
        for j in sim_state.inventory.jokers:
            if j.name == bm.JokerName.CASTLE:
                j.state["target_suit"] = sim_state.castle_suit
    if lua_state.get("ancient_suit") in SUIT_MAP:
        sim_state.ancient_suit = SUIT_MAP[lua_state["ancient_suit"]]
        for j in sim_state.inventory.jokers:
            if j.name == bm.JokerName.ANCIENT_JOKER:
                j.state["target_suit"] = sim_state.ancient_suit
    if lua_state.get("mail_card_rank") in RANK_MAP:
        sim_state.mail_card_rank = RANK_MAP[lua_state["mail_card_rank"]]
        for j in sim_state.inventory.jokers:
            if j.name == bm.JokerName.MAIL_IN_REBATE:
                j.state["target_rank"] = sim_state.mail_card_rank
    idol_data = lua_state.get("idol_card")
    if isinstance(idol_data, dict):
        r = RANK_MAP.get(idol_data.get("rank", "Ace"), bm.Rank.ACE)
        s = SUIT_MAP.get(idol_data.get("suit", "Spades"), bm.Suit.SPADES)
        sim_state.idol_card = (r, s)

    # 4. If Joker lineup or order changed externally (e.g. player dragged Jokers, mod generator, or RNG spawn),
    # preserve existing Python Joker instances by ID so closed-loop accumulator/extra_value predictions are kept.
    lua_jokers_raw = lua_state.get("jokers", [])
    if "jokers" in sim_state.pending_cv_checks or [
        getattr(j, "id", 0) for j in sim_state.inventory.jokers
    ] != [int(d.get("id", 0)) for d in lua_jokers_raw]:
        existing_by_id = {
            j.id: j for j in sim_state.inventory.jokers if getattr(j, "id", 0) != 0
        }
        updated_jokers: list[bm.Joker] = []
        for d in lua_jokers_raw:
            j_id = int(d.get("id", 0))
            if j_id != 0 and j_id in existing_by_id:
                updated_jokers.append(existing_by_id.pop(j_id))
            else:
                pj = parse_joker(d)
                if pj is not None:
                    updated_jokers.append(pj)
                    if act_name == "Setup Test Suite":
                        on_joker_added(sim_state, pj)
        sim_state.inventory.jokers = updated_jokers
        sim_state.pending_cv_checks.discard("jokers")

    # Sync per-Joker rotating To Do List target poker hand (visible on Joker tooltip)
    todo_targets = [
        POKER_HAND_BY_STR[d["to_do_poker_hand"]]
        for d in lua_jokers_raw
        if d.get("name") == "To Do List"
        and d.get("to_do_poker_hand") in POKER_HAND_BY_STR
    ]
    todo_idx = 0
    for j in sim_state.inventory.jokers:
        if j.name == bm.JokerName.TO_DO_LIST and todo_idx < len(todo_targets):
            j.state["target_hand"] = todo_targets[todo_idx]
            sim_state.to_do_poker_hand = todo_targets[todo_idx]
            todo_idx += 1

    # 5. Screen-visible Shop, Consumables, and Booster pack contents
    if "shop" in lua_state:
        sim_state.shop_cards = [
            c for d in lua_state["shop"] if (c := parse_any_card(d)) is not None
        ]
    if "shop_vouchers" in lua_state:
        sim_state.shop_vouchers = [
            c
            for d in lua_state["shop_vouchers"]
            if isinstance(c := parse_any_card(d), bm.Voucher)
        ]
    if "consumables" in lua_state:
        sim_state.inventory.consumables = [
            c for d in lua_state["consumables"] if (c := parse_any_card(d)) is not None
        ]
    if "pack_cards" in lua_state:
        sim_state.pack_cards = [
            c for d in lua_state["pack_cards"] if (c := parse_any_card(d)) is not None
        ]

    # 6. If action is "Setup Test Suite", refresh mutated deck from telemetry
    if act_name == "Setup Test Suite":
        if "deck" in lua_state and lua_state["deck"]:
            sim_state.full_deck = [parse_playing_card(c) for c in lua_state["deck"]]
            sim_state.deck_cards_count = len(sim_state.full_deck)
            sim_state.starting_deck_size = len(sim_state.full_deck)


def test_trajectory_parity(telemetry_path: Path, verbose: bool = False) -> int:
    """Validates game engine parity against a Lua telemetry jsonl log.

    Models a Perfect-CV/OCR Video Pipeline:
    - Each telemetry entry i records the pre-action state S_i at the moment action A_i is clicked.
    - At step i, the harness feeds ONLY CV-visible inputs (dealt hand & selection, blind name,
      shop offerings) into sim_state, executes step_simulation(sim_state, A_i), and compares
      the predicted post-action state against S_{i+1} (entry i+1's pre-action state).

    Returns the count of discrepancies encountered.
    """
    if not telemetry_path.exists():
        print(f"No telemetry file found at {telemetry_path}.")
        return 0

    print(f"Loading Lua telemetry from {telemetry_path}...")
    with open(telemetry_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        print("Telemetry file is empty.")
        return 0

    entries = [json.loads(line) for line in lines]
    first_state = entries[0].get("state", {})

    sim_state = gs.GameState(
        step_idx=0,
        seed=first_state.get("seed", ""),
        ante=first_state.get("ante", 1),
        round=first_state.get("round", 1),
        money=first_state.get("money", 4),
        reroll_cost=first_state.get("reroll_cost", 5),
        hands_left=first_state.get("hands_left", 4),
        discards_left=first_state.get("discards_left", 3),
        hands_played_this_round=first_state.get("hands_played_this_round", 0),
        discards_used_this_round=first_state.get("discards_used_this_round", 0),
    )
    if "deck" in first_state and first_state["deck"]:
        sim_state.full_deck = [parse_playing_card(c) for c in first_state["deck"]]
        sim_state.deck_cards_count = len(sim_state.full_deck)
        sim_state.starting_deck_size = len(sim_state.full_deck)
    if "jokers" in first_state:
        sim_state.inventory.jokers = [
            pj for d in first_state["jokers"] if (pj := parse_joker(d)) is not None
        ]

    trajectory = gs.RunTrajectory(keyframe_interval=10)
    errors = 0

    for i, entry in enumerate(entries):
        act_name = entry.get("action", "Play Hand")
        meta = entry.get("action_metadata", {})
        lua_state = entry.get("state", {})
        step = entry.get("step", i + 1)
        ts = entry.get("timestamp", 0.0)

        # 1. Sync CV-visible screen inputs before executing action A_i
        sync_cv_observations_before_action(sim_state, lua_state, act_name=act_name)

        # 2. Record (S_i, A_i) pair and step the Python simulation engine
        action = parse_lua_action(act_name, meta, frame_idx=step, timestamp=ts)
        if i + 1 < len(entries):
            next_entry_peek = entries[i + 1]
            next_act_peek = next_entry_peek.get("action", "")
            next_lua_peek = next_entry_peek.get("state", {})
            # Detect The Hook mid-play discard logged before evaluate_play
            if (
                act_name == "Play Hand"
                and sim_state.current_blind == bm.BlindName.THE_HOOK
                and not sim_state.boss_disabled
                and next_act_peek == "Discard"
                and next_lua_peek.get("hands_left") == sim_state.hands_left - 1
            ):
                action.metadata["defer_play_for_hook"] = True
            elif (
                act_name == "Discard"
                and getattr(sim_state, "_hook_pending_cards", None) is not None
            ):
                action.metadata["hook"] = True

            # If Riff-Raff spawns Jokers during Select Blind, pass observed spawns from post-blind state
            if act_name == "Select Blind" and "jokers" in next_lua_peek:
                curr_ids = {
                    int(d.get("id", 0))
                    for d in lua_state.get("jokers", [])
                    if d.get("id")
                }
                new_jokers = [
                    parse_joker(d)
                    for d in next_lua_peek["jokers"]
                    if int(d.get("id", 0)) not in curr_ids
                ]
                new_jokers = [j for j in new_jokers if j is not None]
                if new_jokers:
                    action.metadata["riff_raff_spawned"] = new_jokers

            # Detect when Play Hand (or The Hook's mid-play Discard) beats the blind and triggers end_round()
            if (
                act_name == "Play Hand"
                and not action.metadata.get("defer_play_for_hook")
            ) or (act_name == "Discard" and action.metadata.get("hook")):
                if (
                    lua_state.get("blind", "") != ""
                    and next_lua_peek.get("blind", "") == ""
                ):
                    action.metadata["won_round"] = True

        trajectory.record_step(sim_state.copy(), action)
        sim_state, _ = gs.step_simulation(sim_state, action)

        # 3. If there is a next step S_{i+1}, compare our predicted post-action state against S_{i+1}!
        if i + 1 < len(entries):
            next_entry = entries[i + 1]
            next_act = next_entry.get("action", "")
            next_lua = next_entry.get("state", {})

            # Check Money (unless an RNG money event or Tarot/Spectral modified money)
            if "money" in next_lua and "money" not in sim_state.pending_cv_checks:
                # IfCash Out had a specific round_dollars from tags/modifiers, or deterministic payout
                if act_name in (
                    "Play Hand",
                    "Discard",
                    "Reroll Shop",
                    "Select Blind",
                    "Cash Out",
                ):
                    if sim_state.money != next_lua["money"]:
                        print(
                            f"[Mismatch Step {step} '{act_name}' -> Step {step + 1}] "
                            f"Money: Python={sim_state.money} vs Lua={next_lua['money']}"
                        )
                        errors += 1
                    elif verbose:
                        print(f"[OK Step {step} '{act_name}'] Money={sim_state.money}")
            sim_state.Money = next_lua.get("money", sim_state.money)
            sim_state.money = next_lua.get("money", sim_state.money)
            sim_state.pending_cv_checks.discard("money")

            # Check Hands Left & Discards Left
            if act_name in ("Play Hand", "Discard", "Select Blind"):
                if (
                    "hands_left" in next_lua
                    and sim_state.hands_left != next_lua["hands_left"]
                ):
                    print(
                        f"[Mismatch Step {step} '{act_name}' -> Step {step + 1}] "
                        f"Hands Left: Python={sim_state.hands_left} vs Lua={next_lua['hands_left']}"
                    )
                    errors += 1
                    sim_state.hands_left = next_lua["hands_left"]
                if (
                    "discards_left" in next_lua
                    and sim_state.discards_left != next_lua["discards_left"]
                ):
                    print(
                        f"[Mismatch Step {step} '{act_name}' -> Step {step + 1}] "
                        f"Discards Left: Python={sim_state.discards_left} vs Lua={next_lua['discards_left']}"
                    )
                    errors += 1
                    sim_state.discards_left = next_lua["discards_left"]

            # Check Reroll Cost
            if act_name in ("Reroll Shop", "Cash Out") and "reroll_cost" in next_lua:
                if sim_state.reroll_cost != next_lua["reroll_cost"]:
                    print(
                        f"[Mismatch Step {step} '{act_name}' -> Step {step + 1}] "
                        f"Reroll Cost: Python={sim_state.reroll_cost} vs Lua={next_lua['reroll_cost']}"
                    )
                    errors += 1
                    sim_state.reroll_cost = next_lua["reroll_cost"]

            # Check Deck Cards Count
            if "deck_cards_count" in next_lua and act_name in (
                "Play Hand",
                "Discard",
                "Select Blind",
                "Cash Out",
            ):
                if sim_state.deck_cards_count != next_lua["deck_cards_count"]:
                    print(
                        f"[Mismatch Step {step} '{act_name}' -> Step {step + 1}] "
                        f"Deck Cards Count: Python={sim_state.deck_cards_count} vs Lua={next_lua['deck_cards_count']}"
                    )
                    errors += 1
                    sim_state.deck_cards_count = next_lua["deck_cards_count"]

            # Check Discard Pile Count (Note: when blind is beaten or before Cash Out, Balatro's end_round already shuffled discard pile back into deck)
            target_chips = next_lua.get("target_chips", float("inf"))
            current_chips = next_lua.get("current_chips", 0)
            if current_chips >= target_chips:
                sim_state.discard_pile.clear()

            if (
                "discard_pile" in next_lua
                and next_act != "Cash Out"
                and act_name in ("Discard", "Play Hand", "Select Blind")
            ):
                if len(sim_state.discard_pile) != len(next_lua["discard_pile"]):
                    print(
                        f"[Mismatch Step {step} '{act_name}' -> Step {step + 1}] "
                        f"Discard Pile Count: Python={len(sim_state.discard_pile)} vs Lua={len(next_lua['discard_pile'])}"
                    )
                    errors += 1

            # Check Joker Accumulator States (mult, x_mult, chips, extra_value) when Joker lineup is unchanged
            next_jokers_raw = next_lua.get("jokers", [])
            if len(next_jokers_raw) == len(sim_state.inventory.jokers):
                for j_idx, (py_j, lua_j) in enumerate(
                    zip(sim_state.inventory.jokers, next_jokers_raw)
                ):
                    if py_j.name.value != lua_j.get("name"):
                        continue
                    if (
                        py_j.id != 0
                        and lua_j.get("id", 0) != 0
                        and py_j.id != lua_j.get("id")
                    ):
                        continue
                    # Compare extra_value (Egg, Gift Card)
                    if (
                        "extra_value" in lua_j
                        and py_j.extra_value != lua_j["extra_value"]
                    ):
                        print(
                            f"[Mismatch Step {step} '{act_name}' -> Step {step + 1}] "
                            f"Joker[{j_idx}] {py_j.name.value} extra_value: Python={py_j.extra_value} vs Lua={lua_j['extra_value']}"
                        )
                        errors += 1
                        py_j.extra_value = lua_j["extra_value"]
                    # Compare mult accumulator (Green Joker, Ride the Bus, Flash Card, Red Card, Spare Trousers, Popcorn, Ceremonial Dagger)
                    if py_j.name in (
                        bm.JokerName.GREEN_JOKER,
                        bm.JokerName.RIDE_THE_BUS,
                        bm.JokerName.FLASH_CARD,
                        bm.JokerName.RED_CARD,
                        bm.JokerName.SPARE_TROUSERS,
                        bm.JokerName.POPCORN,
                        bm.JokerName.CEREMONIAL_DAGGER,
                    ):
                        py_m = py_j.state.get("mult", 0)
                        lua_m = lua_j.get("mult", 0)
                        if py_m != lua_m:
                            print(
                                f"[Mismatch Step {step} '{act_name}' -> Step {step + 1}] "
                                f"Joker[{j_idx}] {py_j.name.value} mult: Python={py_m} vs Lua={lua_m}"
                            )
                            errors += 1
                            py_j.state["mult"] = lua_m
                    # Compare x_mult accumulator (Campfire, Constellation, Hologram, Vampire, Obelisk, Yorick, Ramen, Hit the Road, Canio, Lucky Cat)
                    if py_j.name in (
                        bm.JokerName.CAMPFIRE,
                        bm.JokerName.CONSTELLATION,
                        bm.JokerName.HOLOGRAM,
                        bm.JokerName.VAMPIRE,
                        bm.JokerName.OBELISK,
                        bm.JokerName.YORICK,
                        bm.JokerName.RAMEN,
                        bm.JokerName.HIT_THE_ROAD,
                        bm.JokerName.CANIO,
                        bm.JokerName.LUCKY_CAT,
                    ) or (
                        py_j.name == bm.JokerName.GLASS_JOKER and bool(sim_state.seed)
                    ):
                        py_xm = round(float(py_j.state.get("x_mult", 1.0)), 2)
                        lua_xm = round(float(lua_j.get("x_mult", 1.0)), 2)
                        if abs(py_xm - lua_xm) > 1e-2:
                            print(
                                f"[Mismatch Step {step} '{act_name}' -> Step {step + 1}] "
                                f"Joker[{j_idx}] {py_j.name.value} x_mult: Python={py_xm} vs Lua={lua_xm}"
                            )
                            errors += 1
                            py_j.state["x_mult"] = lua_xm
                    # Compare Loyalty Card remaining hands
                    if py_j.name == bm.JokerName.LOYALTY_CARD and "remaining" in py_j.state:
                        lua_rem = lua_j.get("loyalty_remaining")
                        if lua_rem is not None and py_j.state["remaining"] != int(lua_rem):
                            print(
                                f"[Mismatch Step {step} '{act_name}' -> Step {step + 1}] "
                                f"Joker[{j_idx}] Loyalty Card remaining: Python={py_j.state['remaining']} vs Lua={lua_rem}"
                            )
                            errors += 1
                    # Compare chips accumulator when extra table is present in telemetry (Runner, Square Joker, Wee Joker, Castle, Ice Cream)
                    if "extra" in lua_j and py_j.name in (
                        bm.JokerName.RUNNER,
                        bm.JokerName.SQUARE_JOKER,
                        bm.JokerName.WEE_JOKER,
                        bm.JokerName.CASTLE,
                        bm.JokerName.ICE_CREAM,
                    ):
                        py_c = py_j.state.get("chips", 0)
                        lua_c = lua_j.get("chips", 0)
                        if py_c != lua_c:
                            print(
                                f"[Mismatch Step {step} '{act_name}' -> Step {step + 1}] "
                                f"Joker[{j_idx}] {py_j.name.value} chips: Python={py_c} vs Lua={lua_c}"
                            )
                            errors += 1

            # Check PRNG stream synchronization when RNG is active and telemetry includes prng_streams
            if "prng_streams" in next_lua and getattr(sim_state, "rng", None) is not None:
                lua_prng = next_lua["prng_streams"]
                if isinstance(lua_prng, dict):
                    for ch_key, ch in sim_state.rng.channels.items():
                        if ch_key in lua_prng and ch.call_count > 0:
                            py_val = round(float(ch.state), 7)
                            lua_val = round(float(lua_prng[ch_key]), 7)
                            if abs(py_val - lua_val) > 1e-5 and verbose:
                                print(
                                    f"[Mismatch Step {step} '{act_name}' -> Step {step + 1}] "
                                    f"PRNG channel '{ch_key}': Python={py_val} vs Lua={lua_val}"
                                )

    return errors


# =========================================================================
# UNIT TESTS
# =========================================================================


class TestEngine(unittest.TestCase):
    """Unit tests for the Balatro simulation engine, Jokers, and state transitions."""

    def test_synthetic_simulation(self):
        """Validates sequential shop reroll, Cash Out triggers, and trajectory keyframes."""
        sim_state = gs.GameState(
            money=15,
            reroll_cost=5,
            inventory=bm.Inventory(
                jokers=[
                    bm.Joker(name=bm.JokerName.GIFT_CARD),
                    bm.Joker(name=bm.JokerName.JOKER),
                ]
            ),
        )
        traj = gs.RunTrajectory(keyframe_interval=3)

        # 1. Reroll Shop
        sim_state, _ = gs.step_simulation(
            sim_state, bm.Action(type=bm.ActionType.REROLL_SHOP)
        )
        traj.record_step(sim_state)
        self.assertEqual(sim_state.money, 10)
        self.assertEqual(sim_state.reroll_cost, 6)

        # 2. Cash out (triggers Gift Card on both jokers)
        sim_state, _ = gs.step_simulation(
            sim_state, bm.Action(type=bm.ActionType.CASH_OUT)
        )
        traj.record_step(sim_state)
        self.assertEqual(sim_state.inventory.jokers[0].extra_value, 1)
        self.assertEqual(sim_state.inventory.jokers[1].extra_value, 1)
        self.assertEqual(sim_state.reroll_cost, 5)  # Resets in shop

        # 3. Test trajectory keyframe seek
        recovered_s0 = traj.get_state(0)
        self.assertIsNotNone(recovered_s0)
        self.assertEqual(recovered_s0.money, 10)

    def test_wee_joker_only_scores_on_active_twos(self):
        """Wee Joker must only gain chips when a 2 is part of the scoring hand."""
        wee = bm.Joker(name=bm.JokerName.WEE_JOKER)
        sim_state = gs.GameState(
            inventory=bm.Inventory(jokers=[wee]),
            hand=bm.Hand(
                cards=[
                    bm.HandCard(
                        card=bm.PlayingCard(rank=bm.Rank.ACE, suit=bm.Suit.SPADES),
                        selected=True,
                    ),
                    bm.HandCard(
                        card=bm.PlayingCard(rank=bm.Rank.TWO, suit=bm.Suit.HEARTS),
                        selected=True,
                    ),
                ]
            ),
        )
        # Playing High Card Ace with an un-scored 2: Wee Joker should NOT gain chips
        sim_state, _ = gs.step_simulation(
            sim_state, bm.Action(type=bm.ActionType.PLAY_HAND)
        )
        self.assertEqual(wee.state.get("chips", 0), 0)

        # Now playing Pair of 2s with King kicker: Wee Joker gains 16 chips (2 * 8)
        sim_state.hand = bm.Hand(
            cards=[
                bm.HandCard(
                    card=bm.PlayingCard(rank=bm.Rank.TWO, suit=bm.Suit.SPADES),
                    selected=True,
                ),
                bm.HandCard(
                    card=bm.PlayingCard(rank=bm.Rank.TWO, suit=bm.Suit.CLUBS),
                    selected=True,
                ),
                bm.HandCard(
                    card=bm.PlayingCard(rank=bm.Rank.KING, suit=bm.Suit.DIAMONDS),
                    selected=True,
                ),
            ]
        )
        sim_state, _ = gs.step_simulation(
            sim_state, bm.Action(type=bm.ActionType.PLAY_HAND)
        )
        self.assertEqual(wee.state.get("chips", 0), 16)

    def test_ride_the_bus_ignores_unscored_face_cards(self):
        """Ride the Bus must NOT reset when face cards are played as un-scored kickers."""
        bus = bm.Joker(name=bm.JokerName.RIDE_THE_BUS)
        sim_state = gs.GameState(
            inventory=bm.Inventory(jokers=[bus]),
            hand=bm.Hand(
                cards=[
                    bm.HandCard(
                        card=bm.PlayingCard(rank=bm.Rank.EIGHT, suit=bm.Suit.SPADES),
                        selected=True,
                    ),
                    bm.HandCard(
                        card=bm.PlayingCard(rank=bm.Rank.EIGHT, suit=bm.Suit.CLUBS),
                        selected=True,
                    ),
                    bm.HandCard(
                        card=bm.PlayingCard(rank=bm.Rank.KING, suit=bm.Suit.DIAMONDS),
                        selected=True,
                    ),
                ]
            ),
        )
        # Hand is Pair of 8s. King is un-scored kicker. Ride the Bus should GAIN +1 mult, not reset!
        sim_state, _ = gs.step_simulation(
            sim_state, bm.Action(type=bm.ActionType.PLAY_HAND)
        )
        self.assertEqual(bus.state.get("mult", 0), 1)

        # Now playing High Card King: King is the scoring card, so Ride the Bus must reset to 0!
        sim_state.hand = bm.Hand(
            cards=[
                bm.HandCard(
                    card=bm.PlayingCard(rank=bm.Rank.KING, suit=bm.Suit.SPADES),
                    selected=True,
                ),
                bm.HandCard(
                    card=bm.PlayingCard(rank=bm.Rank.EIGHT, suit=bm.Suit.HEARTS),
                    selected=True,
                ),
            ]
        )
        sim_state, _ = gs.step_simulation(
            sim_state, bm.Action(type=bm.ActionType.PLAY_HAND)
        )
        self.assertEqual(bus.state.get("mult", 0), 0)

    def test_hiker_and_vampire_only_affect_scoring_cards(self):
        """Hiker and Vampire must exclusively modify scoring cards, leaving kickers intact."""
        hiker = bm.Joker(name=bm.JokerName.HIKER)
        vampire = bm.Joker(name=bm.JokerName.VAMPIRE)
        c8a = bm.PlayingCard(
            rank=bm.Rank.EIGHT, suit=bm.Suit.SPADES, enhancement=bm.Enhancement.GOLD
        )
        c8b = bm.PlayingCard(
            rank=bm.Rank.EIGHT, suit=bm.Suit.CLUBS, enhancement=bm.Enhancement.GOLD
        )
        c5 = bm.PlayingCard(
            rank=bm.Rank.FIVE, suit=bm.Suit.HEARTS, enhancement=bm.Enhancement.GOLD
        )

        sim_state = gs.GameState(
            inventory=bm.Inventory(jokers=[hiker, vampire]),
            hand=bm.Hand(
                cards=[
                    bm.HandCard(card=c8a, selected=True),
                    bm.HandCard(card=c8b, selected=True),
                    bm.HandCard(card=c5, selected=True),
                ]
            ),
        )
        sim_state, _ = gs.step_simulation(
            sim_state, bm.Action(type=bm.ActionType.PLAY_HAND)
        )

        # The two 8s scored: gained +5 chips each, lost their Gold enhancement, Vampire gained 0.2x
        self.assertEqual(c8a.bonus_chips, 5)
        self.assertEqual(c8b.bonus_chips, 5)
        self.assertEqual(c8a.enhancement, bm.Enhancement.NONE)
        self.assertEqual(c8b.enhancement, bm.Enhancement.NONE)
        self.assertAlmostEqual(vampire.state.get("x_mult", 1.0), 1.2)

        # The 5 was an un-scored kicker: must retain Gold and gain NO bonus chips
        self.assertEqual(c5.bonus_chips, 0)
        self.assertEqual(c5.enhancement, bm.Enhancement.GOLD)

    def test_ramen_discard_decay(self):
        """Ramen decreases x_mult by 0.01 per card discarded."""
        ramen = bm.Joker(name=bm.JokerName.RAMEN)
        sim_state = gs.GameState(
            inventory=bm.Inventory(jokers=[ramen]),
            hand=bm.Hand(
                cards=[
                    bm.HandCard(
                        card=bm.PlayingCard(rank=bm.Rank.TWO, suit=bm.Suit.SPADES),
                        selected=True,
                    ),
                    bm.HandCard(
                        card=bm.PlayingCard(rank=bm.Rank.THREE, suit=bm.Suit.CLUBS),
                        selected=True,
                    ),
                    bm.HandCard(
                        card=bm.PlayingCard(rank=bm.Rank.FOUR, suit=bm.Suit.HEARTS),
                        selected=True,
                    ),
                ]
            ),
        )
        sim_state, _ = gs.step_simulation(
            sim_state, bm.Action(type=bm.ActionType.DISCARD)
        )
        self.assertAlmostEqual(ramen.state.get("x_mult", 2.0), 1.97)

    def test_dna_card_duplication_to_deck_and_hand(self):
        """DNA clones the single played card on first hand: adds to full_deck and directly to hand."""
        dna = bm.Joker(name=bm.JokerName.DNA)
        hologram = bm.Joker(name=bm.JokerName.HOLOGRAM)
        target_card = bm.PlayingCard(rank=bm.Rank.ACE, suit=bm.Suit.SPADES)
        other_card = bm.PlayingCard(rank=bm.Rank.TEN, suit=bm.Suit.HEARTS)

        sim_state = gs.GameState(
            inventory=bm.Inventory(jokers=[dna, hologram]),
            hand=bm.Hand(
                cards=[
                    bm.HandCard(card=target_card, selected=True),
                    bm.HandCard(card=other_card, selected=False),
                ]
            ),
        )
        self.assertEqual(len(sim_state.full_deck), 52)

        # 1. Play first hand with 1 card (Ace of Spades)
        sim_state, _ = gs.step_simulation(
            sim_state, bm.Action(type=bm.ActionType.PLAY_HAND)
        )

        # Deck should now have 53 cards, and Hologram scaled by +0.25x from DNA's card addition!
        self.assertEqual(len(sim_state.full_deck), 53)
        self.assertEqual(sim_state.deck_cards_count, 53)
        self.assertAlmostEqual(hologram.state.get("x_mult", 1.0), 1.25)

        # Hand should contain: the other card + the newly copied Ace of Spades (unselected)
        self.assertEqual(len(sim_state.hand.cards), 2)
        ace_hc = [c for c in sim_state.hand.cards if c.card.rank == bm.Rank.ACE][0]
        self.assertFalse(ace_hc.selected)

        # 2. Playing a second hand with 1 card should NOT trigger DNA again
        ace_hc.selected = True
        sim_state, _ = gs.step_simulation(
            sim_state, bm.Action(type=bm.ActionType.PLAY_HAND)
        )
        self.assertEqual(len(sim_state.full_deck), 53)

    def test_dna_blueprint_synergy(self):
        """Blueprint copying DNA produces two cloned cards in full_deck and hand."""
        dna = bm.Joker(name=bm.JokerName.DNA)
        bp = bm.Joker(name=bm.JokerName.BLUEPRINT)
        card = bm.PlayingCard(rank=bm.Rank.KING, suit=bm.Suit.HEARTS)

        sim_state = gs.GameState(
            inventory=bm.Inventory(jokers=[bp, dna]),
            hand=bm.Hand(cards=[bm.HandCard(card=card, selected=True)]),
        )
        sim_state, _ = gs.step_simulation(
            sim_state, bm.Action(type=bm.ActionType.PLAY_HAND)
        )

        self.assertEqual(len(sim_state.full_deck), 54)
        self.assertEqual(sim_state.deck_cards_count, 54)
        self.assertEqual(len(sim_state.hand.cards), 2)

    def test_trading_card_destruction_and_discard_pile(self):
        """Trading Card destroys single card on first discard: removes from full_deck and triggers Canio."""
        tc = bm.Joker(name=bm.JokerName.TRADING_CARD)
        canio = bm.Joker(name=bm.JokerName.CANIO)
        face_card = bm.PlayingCard(rank=bm.Rank.QUEEN, suit=bm.Suit.DIAMONDS)
        other_card = bm.PlayingCard(rank=bm.Rank.TWO, suit=bm.Suit.CLUBS)

        sim_state = gs.GameState(
            inventory=bm.Inventory(jokers=[tc, canio]),
            hand=bm.Hand(
                cards=[
                    bm.HandCard(card=face_card, selected=True),
                    bm.HandCard(card=other_card, selected=False),
                ]
            ),
        )
        initial_money = sim_state.money

        sim_state, _ = gs.step_simulation(
            sim_state, bm.Action(type=bm.ActionType.DISCARD)
        )

        self.assertEqual(sim_state.money, initial_money + 3)
        self.assertEqual(sim_state.deck_cards_count, 51)
        self.assertNotIn(face_card, sim_state.full_deck)
        self.assertNotIn(face_card, sim_state.discard_pile)
        self.assertEqual(len(sim_state.discard_pile), 0)
        self.assertAlmostEqual(canio.state.get("x_mult", 1.0), 2.0)

    def test_discard_pile_and_played_cards_tracking_and_reset(self):
        """Verifies accumulation in discard_pile and played_cards_this_round, and reset on SELECT_BLIND."""
        c1 = bm.PlayingCard(rank=bm.Rank.FOUR, suit=bm.Suit.SPADES)
        c2 = bm.PlayingCard(rank=bm.Rank.FIVE, suit=bm.Suit.HEARTS)
        c3 = bm.PlayingCard(rank=bm.Rank.SEVEN, suit=bm.Suit.CLUBS)

        sim_state = gs.GameState(
            hand=bm.Hand(
                cards=[
                    bm.HandCard(card=c1, selected=True),
                    bm.HandCard(card=c2, selected=False),
                    bm.HandCard(card=c3, selected=False),
                ]
            ),
        )

        sim_state, _ = gs.step_simulation(
            sim_state, bm.Action(type=bm.ActionType.DISCARD)
        )
        self.assertIn(c1, sim_state.discard_pile)
        self.assertEqual(len(sim_state.discard_pile), 1)

        sim_state.hand.cards[0].selected = True  # c2
        sim_state, _ = gs.step_simulation(
            sim_state, bm.Action(type=bm.ActionType.PLAY_HAND)
        )
        self.assertIn(c2, sim_state.played_cards_this_round)
        self.assertEqual(len(sim_state.played_cards_this_round), 1)

        sim_state, _ = gs.step_simulation(
            sim_state, bm.Action(type=bm.ActionType.SELECT_BLIND)
        )
        self.assertEqual(len(sim_state.discard_pile), 0)
        self.assertEqual(len(sim_state.played_cards_this_round), 0)

    def test_sixth_sense_destruction(self):
        """Sixth Sense destroys a single played 6 on first hand from hand and deck."""
        sixth = bm.Joker(name=bm.JokerName.SIXTH_SENSE)
        c6 = bm.PlayingCard(rank=bm.Rank.SIX, suit=bm.Suit.SPADES)

        sim_state = gs.GameState(
            inventory=bm.Inventory(jokers=[sixth]),
            hand=bm.Hand(cards=[bm.HandCard(card=c6, selected=True)]),
        )
        sim_state, _ = gs.step_simulation(
            sim_state, bm.Action(type=bm.ActionType.PLAY_HAND)
        )

        self.assertEqual(sim_state.deck_cards_count, 51)
        self.assertNotIn(c6, sim_state.full_deck)
        self.assertEqual(len(sim_state.hand.cards), 0)
        self.assertNotIn(c6, sim_state.played_cards_this_round)

    def test_marble_joker_adds_stone_card(self):
        """Marble Joker appends a Stone card to full_deck when selecting a blind."""
        marble = bm.Joker(name=bm.JokerName.MARBLE_JOKER)
        sim_state = gs.GameState(inventory=bm.Inventory(jokers=[marble]))
        initial_count = len(sim_state.full_deck)

        sim_state, _ = gs.step_simulation(
            sim_state, bm.Action(type=bm.ActionType.SELECT_BLIND)
        )
        self.assertEqual(len(sim_state.full_deck), initial_count + 1)
        self.assertEqual(sim_state.full_deck[-1].enhancement, bm.Enhancement.STONE)

    def test_cloud_9_evaluates_full_deck(self):
        """Cloud 9 counts all 9s in the full_deck at end of round."""
        c9 = bm.Joker(name=bm.JokerName.CLOUD_9)
        sim_state = gs.GameState(inventory=bm.Inventory(jokers=[c9]))
        initial_money = sim_state.money

        sim_state, _ = gs.step_simulation(
            sim_state, bm.Action(type=bm.ActionType.CASH_OUT)
        )
        self.assertEqual(sim_state.money, initial_money + 4)

    def test_joker_fixes_campfire_yorick_obelisk_todolist_turtlebean_retriggers(self):
        """Validates source-accurate fixes for Campfire, Yorick, Obelisk, To Do List, Turtle Bean, Hiker+Blueprint, and Retriggers."""
        # 1. Campfire adds +0.25x per card sold and resets on Boss Blind
        campfire = bm.Joker(name=bm.JokerName.CAMPFIRE)
        fodder = bm.Joker(name=bm.JokerName.JOKER)
        state = gs.GameState(
            current_blind=bm.BlindName.THE_GOAD,
            inventory=bm.Inventory(jokers=[campfire, fodder]),
        )
        state, _ = gs.step_simulation(
            state, bm.Action(type=bm.ActionType.SELL_CARD, target_idx=1)
        )
        self.assertAlmostEqual(campfire.state["x_mult"], 1.25)
        state, _ = gs.step_simulation(state, bm.Action(type=bm.ActionType.CASH_OUT))
        self.assertAlmostEqual(campfire.state["x_mult"], 1.0)

        # 2. Yorick carries over excess discards across the 23 threshold
        yorick = bm.Joker(name=bm.JokerName.YORICK)
        yorick.state["discards"] = 2
        state_y = gs.GameState(
            inventory=bm.Inventory(jokers=[yorick]),
            hand=bm.Hand(
                cards=[
                    bm.HandCard(
                        card=bm.PlayingCard(rank=bm.Rank.TWO, suit=bm.Suit.SPADES),
                        selected=True,
                    ),
                    bm.HandCard(
                        card=bm.PlayingCard(rank=bm.Rank.THREE, suit=bm.Suit.SPADES),
                        selected=True,
                    ),
                    bm.HandCard(
                        card=bm.PlayingCard(rank=bm.Rank.FOUR, suit=bm.Suit.SPADES),
                        selected=True,
                    ),
                    bm.HandCard(
                        card=bm.PlayingCard(rank=bm.Rank.FIVE, suit=bm.Suit.SPADES),
                        selected=True,
                    ),
                    bm.HandCard(
                        card=bm.PlayingCard(rank=bm.Rank.SIX, suit=bm.Suit.SPADES),
                        selected=True,
                    ),
                ]
            ),
        )
        state_y, _ = gs.step_simulation(state_y, bm.Action(type=bm.ActionType.DISCARD))
        self.assertAlmostEqual(yorick.state["x_mult"], 2.0)
        self.assertEqual(yorick.state["discards"], 20)  # 2 - 5 + 23 = 20

        # 3. Obelisk does NOT reset when tying the most played hand; only resets when strictly most played
        obelisk = bm.Joker(name=bm.JokerName.OBELISK)
        obelisk.state["x_mult"] = 1.4
        state_ob = gs.GameState(
            inventory=bm.Inventory(jokers=[obelisk]),
            hand_play_counts={bm.PokerHand.PAIR: 3, bm.PokerHand.HIGH_CARD: 2},
            hand=bm.Hand(
                cards=[
                    bm.HandCard(
                        card=bm.PlayingCard(rank=bm.Rank.ACE, suit=bm.Suit.SPADES),
                        selected=True,
                    ),
                ]
            ),
        )
        # Playing High Card brings HIGH_CARD to 3 (tied with PAIR=3) -> must NOT reset, scales to 1.6!
        state_ob, _ = gs.step_simulation(
            state_ob, bm.Action(type=bm.ActionType.PLAY_HAND)
        )
        self.assertAlmostEqual(obelisk.state["x_mult"], 1.6)
        # Playing High Card again brings HIGH_CARD to 4 (strictly > PAIR=3) -> resets to 1.0!
        state_ob.hand.cards = [
            bm.HandCard(
                card=bm.PlayingCard(rank=bm.Rank.KING, suit=bm.Suit.SPADES),
                selected=True,
            )
        ]
        state_ob, _ = gs.step_simulation(
            state_ob, bm.Action(type=bm.ActionType.PLAY_HAND)
        )
        self.assertAlmostEqual(obelisk.state["x_mult"], 1.0)

        # 4. Hiker + Blueprint + Hanging Chad retrigger synergy
        bp = bm.Joker(name=bm.JokerName.BLUEPRINT)
        hiker = bm.Joker(name=bm.JokerName.HIKER)
        chad = bm.Joker(name=bm.JokerName.HANGING_CHAD)
        c_ace = bm.PlayingCard(rank=bm.Rank.ACE, suit=bm.Suit.SPADES)
        state_hk = gs.GameState(
            inventory=bm.Inventory(jokers=[bp, hiker, chad]),
            hand=bm.Hand(cards=[bm.HandCard(card=c_ace, selected=True)]),
        )
        # Hanging Chad gives 1 + 2 = 3 triggers; Blueprint+Hiker gives 2 * 5 = +10 chips per trigger -> +30 bonus_chips!
        state_hk, _ = gs.step_simulation(
            state_hk, bm.Action(type=bm.ActionType.PLAY_HAND)
        )
        self.assertEqual(c_ace.bonus_chips, 30)

        # 5. Turtle Bean modifies hand.max_size on add and decrements hand.max_size at end of round
        bean = bm.Joker(name=bm.JokerName.TURTLE_BEAN)
        state_tb = gs.GameState()
        state_tb.inventory.jokers.append(bean)
        on_joker_added(state_tb, bean)
        self.assertEqual(state_tb.hand.max_size, 13)  # 8 + 5
        state_tb, _ = gs.step_simulation(
            state_tb, bm.Action(type=bm.ActionType.CASH_OUT)
        )
        self.assertEqual(bean.state["h_size"], 4)
        self.assertEqual(state_tb.hand.max_size, 12)

    def test_gamestate_new_features_and_deltas(self):
        """Validates pack opening, skip tags, forced selection, duplicate removal, and delta diffs."""
        pack = bm.BoosterPack(
            booster_type=bm.BoosterType.ARCANA,
            size=bm.BoosterSize.MEGA,
            cost=8,
            choose=2,
            extra=5,
        )
        ace1 = bm.PlayingCard(rank=bm.Rank.ACE, suit=bm.Suit.SPADES, id=101)
        ace2 = bm.PlayingCard(rank=bm.Rank.ACE, suit=bm.Suit.SPADES, id=102)

        state = gs.GameState(
            location=bm.Location.SELECT,
            money=20,
            small_blind_tag=bm.SkipTag.DOUBLE_TAG,
            shop_packs=[pack],
            hand=bm.Hand(
                cards=[
                    bm.HandCard(card=ace1, selected=True, forced_selection=True),
                    bm.HandCard(card=ace2, selected=False),
                ]
            ),
        )
        state.full_deck.extend([ace1, ace2])
        state.deck_cards_count = len(state.full_deck)

        state.hand.select(0)
        self.assertTrue(state.hand.cards[0].selected)
        state.hand.clear_selection()
        self.assertTrue(state.hand.cards[0].selected)

        state.remove_card_from_deck(ace1)
        self.assertEqual(len(state.hand.cards), 1)
        self.assertIs(state.hand.cards[0].card, ace2)

        traj = gs.RunTrajectory(keyframe_interval=5)
        traj.record_step(state)
        state, _ = gs.step_simulation(state, bm.Action(type=bm.ActionType.SKIP_BLIND))
        traj.record_step(state)
        self.assertEqual(state.blinds_skipped, 1)
        self.assertIn(bm.SkipTag.DOUBLE_TAG, state.active_tags)
        self.assertEqual(state.current_blind, bm.BlindName.BIG)

        state, _ = gs.step_simulation(
            state, bm.Action(type=bm.ActionType.OPEN_PACK, target_idx=0)
        )
        state.pack_cards = [
            bm.Tarot(name=bm.TarotName.HERMIT),
            bm.Spectral(name=bm.SpectralName.ECTOPLASM),
        ]
        state.hand_levels[bm.PokerHand.PAIR] = 3
        state.inventory.joker_slots = 6
        traj.record_step(state)
        self.assertEqual(state.location, bm.Location.PACK)
        self.assertEqual(state.pack_choices_remaining, 2)

        state, _ = gs.step_simulation(
            state, bm.Action(type=bm.ActionType.USE_IN_PACK, target_idx=0)
        )
        traj.record_step(state)
        self.assertEqual(state.last_tarot_planet, bm.TarotName.HERMIT)
        self.assertEqual(state.tarots_used, 1)
        self.assertEqual(state.pack_choices_remaining, 1)
        self.assertEqual(state.location, bm.Location.PACK)

        state, _ = gs.step_simulation(
            state, bm.Action(type=bm.ActionType.USE_IN_PACK, target_idx=0)
        )
        traj.record_step(state)
        self.assertEqual(state.hand.max_size, 7)
        self.assertEqual(state.ectoplasm_minus, 2)
        self.assertEqual(state.location, bm.Location.SHOP)

        replayed = traj.get_state(4)
        self.assertEqual(replayed.hand_levels[bm.PokerHand.PAIR], 3)
        self.assertEqual(replayed.inventory.joker_slots, 6)
        self.assertEqual(replayed.hand.max_size, 7)
        self.assertEqual(replayed.ectoplasm_minus, 2)
        self.assertEqual(replayed.last_tarot_planet, bm.TarotName.HERMIT)
        self.assertEqual(replayed.tarots_used, 1)

    def test_action_parsing(self):
        """Validates that Lua action names map correctly to ActionType enums."""
        action_cases = [
            ("Play Hand", bm.ActionType.PLAY_HAND),
            ("Discard", bm.ActionType.DISCARD),
            ("Reroll Shop", bm.ActionType.REROLL_SHOP),
            ("Cash Out", bm.ActionType.CASH_OUT),
            ("Buy Card", bm.ActionType.BUY_CARD),
            ("Buy and Use", bm.ActionType.BUY_AND_USE),
            ("Sell Card", bm.ActionType.SELL_CARD),
            ("Select Blind", bm.ActionType.SELECT_BLIND),
            ("Skip Blind", bm.ActionType.SKIP_BLIND),
            ("Use Consumable", bm.ActionType.USE_CONSUMABLE),
            ("Open Pack", bm.ActionType.OPEN_PACK),
            ("Use In Pack", bm.ActionType.USE_IN_PACK),
            ("Skip Pack", bm.ActionType.SKIP_PACK),
            ("Redeem Voucher", bm.ActionType.REDEEM_VOUCHER),
            ("Next Round", bm.ActionType.NEXT_ROUND),
            ("Reroll Boss", bm.ActionType.REROLL_BOSS),
        ]
        for name, expected_type in action_cases:
            action = parse_lua_action(name, {}, frame_idx=1, timestamp=0.5)
            self.assertEqual(action.type, expected_type)
            self.assertEqual(action.frame_idx, 1)
            self.assertEqual(action.timestamp, 0.5)

    def test_telemetry_file_parity_if_present(self):
        """Runs full parity validation if run_telemetry.jsonl is present and non-empty."""
        telemetry_paths = [
            REPO_ROOT / "run_telemetry.jsonl",
            REPO_ROOT / "output" / "run_telemetry.jsonl",
        ]
        for p in telemetry_paths:
            if p.exists() and p.stat().st_size > 0:
                errors = test_trajectory_parity(p)
                self.assertEqual(
                    errors, 0, f"Encountered {errors} telemetry parity errors in {p}"
                )


class TestFaceDownCardBounds(unittest.TestCase):
    """Unit tests for face-down card representation, partial observability, bounds deduction, and persistence."""

    def test_face_down_card_masks_rank_and_suit(self):
        """Card initialized face-down must mask rank and suit to NONE to prevent ML ground-truth leakage."""
        c = bm.PlayingCard(rank=bm.Rank.ACE, suit=bm.Suit.SPADES, id=42)
        hc = bm.HandCard(card=c, facing_back=True)

        self.assertEqual(hc.card.rank, bm.Rank.NONE)
        self.assertEqual(hc.card.suit, bm.Suit.NONE)
        self.assertEqual(hc.card.id, 0)
        self.assertIsNotNone(hc._real_card)
        self.assertEqual(hc._real_card.rank, bm.Rank.ACE)
        self.assertEqual(hc._real_card.suit, bm.Suit.SPADES)
        self.assertEqual(hc._real_card.id, 42)
        self.assertTrue(hc.bounds.is_face_down)
        self.assertFalse(hc.bounds.is_resolved)

    def test_marginal_probability_vectors_and_shapes(self):
        """CardBounds and HandCard must expose fixed-dimensional probability distributions."""
        c = bm.PlayingCard(rank=bm.Rank.ACE, suit=bm.Suit.SPADES)
        hc = bm.HandCard(card=c, facing_back=True)

        rv = hc.get_rank_vector()
        sv = hc.get_suit_vector()
        slv = hc.get_seal_vector()

        self.assertEqual(len(rv), 13)
        self.assertEqual(len(sv), 4)
        self.assertEqual(len(slv), 5)
        self.assertAlmostEqual(sum(rv), 1.0, places=5)
        self.assertAlmostEqual(sum(sv), 1.0, places=5)
        self.assertAlmostEqual(sum(slv), 1.0, places=5)

        # For a face-up card, verify one-hot vectors
        hc_up = bm.HandCard(card=c, facing_back=False)
        self.assertEqual(hc_up.get_rank_vector()[-1], 1.0)  # Ace is index 12 (last)
        self.assertEqual(hc_up.get_suit_vector()[0], 1.0)  # Spades is index 0 (first)
        self.assertEqual(hc_up.get_seal_vector()[0], 1.0)  # Seal.NONE is index 0

    def test_rank_sort_observation_bounds(self):
        """'Sort by Rank' (descending) constrains possible ranks between flanking face-up cards."""
        state = gs.GameState()
        ace = bm.PlayingCard(rank=bm.Rank.ACE, suit=bm.Suit.SPADES, id=1)
        ten = bm.PlayingCard(rank=bm.Rank.TEN, suit=bm.Suit.CLUBS, id=2)
        eight = bm.PlayingCard(rank=bm.Rank.EIGHT, suit=bm.Suit.HEARTS, id=3)

        hc_ace = bm.HandCard(card=ace, facing_back=False)
        hc_ten = bm.HandCard(card=ten, facing_back=True)
        hc_eight = bm.HandCard(card=eight, facing_back=False)

        state.hand = bm.Hand(cards=[hc_ace, hc_ten, hc_eight])
        state.apply_sort_observation(sort_type="rank")

        # Ranks 2..7 have nominal < 8.03 and must be ruled out
        for r in [
            bm.Rank.TWO,
            bm.Rank.THREE,
            bm.Rank.FOUR,
            bm.Rank.FIVE,
            bm.Rank.SIX,
            bm.Rank.SEVEN,
        ]:
            self.assertNotIn(r, hc_ten.bounds.possible_ranks)
        self.assertIn(bm.Rank.TEN, hc_ten.bounds.possible_ranks)
        self.assertIn(bm.Rank.NINE, hc_ten.bounds.possible_ranks)

    def test_suit_sort_observation_bounds(self):
        """'Sort by Suit' restricts candidate cards to the flanking suit block."""
        state = gs.GameState()
        kh = bm.PlayingCard(rank=bm.Rank.KING, suit=bm.Suit.HEARTS, id=101)
        five_h = bm.PlayingCard(rank=bm.Rank.FIVE, suit=bm.Suit.HEARTS, id=102)
        two_h = bm.PlayingCard(rank=bm.Rank.TWO, suit=bm.Suit.HEARTS, id=103)

        hc_kh = bm.HandCard(card=kh, facing_back=False)
        hc_five = bm.HandCard(card=five_h, facing_back=True)
        hc_two = bm.HandCard(card=two_h, facing_back=False)

        state.hand = bm.Hand(cards=[hc_kh, hc_five, hc_two])
        state.apply_sort_observation(sort_type="suit")

        self.assertEqual(hc_five.bounds.possible_suits, {bm.Suit.HEARTS})

    def test_rank_and_suit_sort_resolves_card(self):
        """Intersection of Rank sort and Suit sort observations resolves face-down card to exactly 1 candidate."""
        state = gs.GameState()
        c_ad = bm.PlayingCard(rank=bm.Rank.ACE, suit=bm.Suit.DIAMONDS, id=1)
        c_10d = bm.PlayingCard(rank=bm.Rank.TEN, suit=bm.Suit.DIAMONDS, id=2)
        c_8d = bm.PlayingCard(rank=bm.Rank.EIGHT, suit=bm.Suit.DIAMONDS, id=3)
        c_2s = bm.PlayingCard(rank=bm.Rank.TWO, suit=bm.Suit.SPADES, id=4)
        state.full_deck = [c_ad, c_10d, c_8d, c_2s]

        hc_ad = bm.HandCard(card=c_ad, facing_back=False)
        hc_10d = bm.HandCard(card=c_10d, facing_back=True)
        hc_8d = bm.HandCard(card=c_8d, facing_back=False)

        state.hand = bm.Hand(cards=[hc_ad, hc_10d, hc_8d])

        state.apply_sort_observation(sort_type="rank")
        state.apply_sort_observation(sort_type="suit")

        self.assertTrue(hc_10d.bounds.is_resolved)
        self.assertTrue(hc_10d.facing_back)
        self.assertEqual(hc_10d.card.rank, bm.Rank.TEN)
        self.assertEqual(hc_10d.card.suit, bm.Suit.DIAMONDS)
        self.assertEqual(hc_10d.card.id, 2)

    def test_persistent_slot_uid_and_bounds_across_play(self):
        """Held face-down cards retain slot_uid and bounds across PLAY_HAND actions while new cards append right."""
        state = gs.GameState()
        c1 = bm.PlayingCard(rank=bm.Rank.TWO, suit=bm.Suit.CLUBS, id=1)
        c2 = bm.PlayingCard(
            rank=bm.Rank.KING, suit=bm.Suit.HEARTS, seal=bm.Seal.BLUE, id=2
        )
        c3 = bm.PlayingCard(rank=bm.Rank.THREE, suit=bm.Suit.CLUBS, id=3)
        c4 = bm.PlayingCard(rank=bm.Rank.FOUR, suit=bm.Suit.CLUBS, id=4)
        c5 = bm.PlayingCard(rank=bm.Rank.FIVE, suit=bm.Suit.CLUBS, id=5)

        hc1 = bm.HandCard(card=c1, selected=True)
        hc2 = bm.HandCard(card=c2, facing_back=True, slot_uid=99)
        hc2.bounds.possible_seals = {bm.Seal.BLUE}
        hc3 = bm.HandCard(card=c3, selected=True)
        hc4 = bm.HandCard(card=c4, selected=True)
        hc5 = bm.HandCard(card=c5, selected=True)

        state.hand = bm.Hand(cards=[hc1, hc2, hc3, hc4, hc5])
        state, _ = gs.step_simulation(state, bm.Action(type=bm.ActionType.PLAY_HAND))

        self.assertEqual(len(state.hand.cards), 1)
        survivor = state.hand.cards[0]
        self.assertEqual(survivor.slot_uid, 99)
        self.assertEqual(survivor.bounds.possible_seals, {bm.Seal.BLUE})
        self.assertTrue(survivor.facing_back)

        # Replenish hand
        drawn = state.draw_cards()
        self.assertEqual(len(state.hand.cards), state.hand.max_size)
        self.assertEqual(state.hand.cards[0].slot_uid, 99)
        for i in range(1, len(state.hand.cards)):
            self.assertGreater(state.hand.cards[i].slot_uid, 0)
            self.assertNotEqual(state.hand.cards[i].slot_uid, 99)

    def test_purple_seal_discard_prunes_candidates(self):
        """Discarding a Purple Seal face-down card prunes it from remaining candidates and triggers tarot spawn."""
        state = gs.GameState()
        state.inventory = bm.Inventory()
        state.seed = "TEST"
        state.set_seed("TEST")

        c_purple = bm.PlayingCard(
            rank=bm.Rank.ACE, suit=bm.Suit.SPADES, seal=bm.Seal.PURPLE, id=10
        )
        c_normal = bm.PlayingCard(
            rank=bm.Rank.ACE, suit=bm.Suit.HEARTS, seal=bm.Seal.NONE, id=20
        )
        state.full_deck = [c_purple, c_normal]

        hc_p = bm.HandCard(card=c_purple, facing_back=True, selected=True)
        hc_n = bm.HandCard(card=c_normal, facing_back=True, selected=False)
        state.hand = bm.Hand(cards=[hc_p, hc_n])
        state.ensure_slot_uids()

        state, _ = gs.step_simulation(state, bm.Action(type=bm.ActionType.DISCARD))

        self.assertEqual(len(state.inventory.consumables), 1)
        survivor = state.hand.cards[0]
        self.assertNotIn(10, survivor.bounds.candidate_card_ids)
        self.assertEqual(survivor.bounds.candidate_card_ids, {20})
        self.assertTrue(survivor.bounds.is_resolved)
        self.assertEqual(survivor.card.rank, bm.Rank.ACE)
        self.assertEqual(survivor.card.suit, bm.Suit.HEARTS)
        self.assertEqual(survivor.card.id, 20)

    def test_tarot_actions_narrow_face_down_cards(self):
        """Tarot / Spectral actions (Death, Magician, Sun, Trance) narrow and resolve face-down cards."""
        state = gs.GameState()
        state.inventory = bm.Inventory()

        # 1. Death on [face_down, face_up_King_Hearts] resolves left card
        c_left = bm.PlayingCard(rank=bm.Rank.TWO, suit=bm.Suit.CLUBS, id=1)
        c_right = bm.PlayingCard(rank=bm.Rank.KING, suit=bm.Suit.HEARTS, id=2)
        hc_left = bm.HandCard(card=c_left, facing_back=True, selected=True)
        hc_right = bm.HandCard(card=c_right, facing_back=False, selected=True)
        state.hand = bm.Hand(cards=[hc_left, hc_right])
        state.inventory.consumables = [bm.Tarot(name=bm.TarotName.DEATH)]

        state, _ = gs.step_simulation(state, bm.Action(type=bm.ActionType.USE_CONSUMABLE, target_idx=0))
        self.assertTrue(hc_left.bounds.is_resolved)
        self.assertTrue(hc_left.facing_back)
        self.assertEqual(hc_left.card.rank, bm.Rank.KING)
        self.assertEqual(hc_left.card.suit, bm.Suit.HEARTS)

        # 2. Magician applies Lucky enhancement
        c3 = bm.PlayingCard(rank=bm.Rank.FIVE, suit=bm.Suit.SPADES, id=3)
        hc3 = bm.HandCard(card=c3, facing_back=True, selected=True)
        state.hand = bm.Hand(cards=[hc3])
        state.inventory.consumables = [bm.Tarot(name=bm.TarotName.MAGICIAN)]

        state, _ = gs.step_simulation(state, bm.Action(type=bm.ActionType.USE_CONSUMABLE, target_idx=0))
        self.assertEqual(hc3.bounds.possible_enhancements, {bm.Enhancement.LUCKY})
        self.assertEqual(hc3.card.enhancement, bm.Enhancement.LUCKY)

        # 3. The Sun converts suit to Hearts
        state.inventory.consumables = [bm.Tarot(name=bm.TarotName.SUN)]
        hc3.selected = True
        state, _ = gs.step_simulation(state, bm.Action(type=bm.ActionType.USE_CONSUMABLE, target_idx=0))
        self.assertEqual(hc3.bounds.possible_suits, {bm.Suit.HEARTS})
        self.assertEqual(hc3.card.suit, bm.Suit.HEARTS)

        # 4. Trance adds Blue seal
        state.inventory.consumables = [bm.Spectral(name=bm.SpectralName.TRANCE)]
        hc3.selected = True
        state, _ = gs.step_simulation(state, bm.Action(type=bm.ActionType.USE_CONSUMABLE, target_idx=0))
        self.assertEqual(hc3.bounds.possible_seals, {bm.Seal.BLUE})
        self.assertEqual(hc3.card.seal, bm.Seal.BLUE)

    def test_tarot_flip_reveals_face_down_card(self):
        """Tarot usage temporarily flips card face-up: card stays facing_back=True, but its full identity is observed."""
        state = gs.GameState()
        state.inventory = bm.Inventory()
        c = bm.PlayingCard(rank=bm.Rank.SEVEN, suit=bm.Suit.SPADES, id=77)
        hc = bm.HandCard(card=c, facing_back=True, selected=True)
        self.assertEqual(hc.card.rank, bm.Rank.NONE)
        self.assertEqual(hc.card.suit, bm.Suit.NONE)
        self.assertFalse(hc.bounds.is_resolved)

        state.hand = bm.Hand(cards=[hc])
        state.inventory.consumables = [bm.Tarot(name=bm.TarotName.SUN)]

        state, _ = gs.step_simulation(state, bm.Action(type=bm.ActionType.USE_CONSUMABLE, target_idx=0))

        # Flips back face-down, but identity is now fully observed and resolved!
        self.assertTrue(hc.facing_back)
        self.assertTrue(hc.bounds.is_resolved)
        self.assertEqual(hc.card.rank, bm.Rank.SEVEN)
        self.assertEqual(hc.card.suit, bm.Suit.HEARTS)
        self.assertEqual(hc.bounds.possible_ranks, {bm.Rank.SEVEN})
        self.assertEqual(hc.bounds.possible_suits, {bm.Suit.HEARTS})

    def test_draw_face_up_retains_face_down_identity(self):
        """Drawing face-up cards into hand preserves known face-down card identity."""
        state = gs.GameState()
        state.inventory = bm.Inventory()
        c = bm.PlayingCard(rank=bm.Rank.EIGHT, suit=bm.Suit.CLUBS, id=88)
        hc = bm.HandCard(card=c, facing_back=True, selected=True)
        state.hand = bm.Hand(cards=[hc], max_size=4)
        state.inventory.consumables = [bm.Tarot(name=bm.TarotName.MAGICIAN)]
        state, _ = gs.step_simulation(state, bm.Action(type=bm.ActionType.USE_CONSUMABLE, target_idx=0))
        self.assertTrue(hc.bounds.is_resolved)

        # Draw remaining cards face-up
        c_up1 = bm.PlayingCard(rank=bm.Rank.TWO, suit=bm.Suit.HEARTS, id=2)
        c_up2 = bm.PlayingCard(rank=bm.Rank.THREE, suit=bm.Suit.HEARTS, id=3)
        state.full_deck = [c, c_up1, c_up2]
        state.draw_cards(count=2, face_down=False)

        # Known face-down card still retained!
        self.assertTrue(hc.bounds.is_resolved)
        self.assertEqual(hc.card.rank, bm.Rank.EIGHT)
        self.assertEqual(hc.card.enhancement, bm.Enhancement.LUCKY)

    def test_draw_non_adjacent_face_down_retains_identity(self):
        """Drawing face-down cards visually separated by a face-up card does not cause confusion."""
        state = gs.GameState()
        c_revealed = bm.PlayingCard(rank=bm.Rank.KING, suit=bm.Suit.HEARTS, id=101)
        hc_revealed = bm.HandCard(card=c_revealed, facing_back=True)
        hc_revealed.reveal_temporarily()

        c_divider = bm.PlayingCard(rank=bm.Rank.ACE, suit=bm.Suit.SPADES, id=102)
        hc_divider = bm.HandCard(card=c_divider, facing_back=False)

        # Hand: [FaceDown (King of Hearts), FaceUp (Ace of Spades)]
        state.hand = bm.Hand(cards=[hc_revealed, hc_divider], max_size=3)

        # Draw a face-down card to the right: Hand will be [FaceDown, FaceUp, FaceDown]
        c_new_fd = bm.PlayingCard(rank=bm.Rank.FIVE, suit=bm.Suit.DIAMONDS, id=103)
        state.full_deck = [c_revealed, c_divider, c_new_fd]
        state.draw_cards(count=1, face_down=True)

        self.assertEqual(len(state.hand.cards), 3)
        # hc_revealed at index 0 is NOT adjacent to index 2 (separated by index 1 face-up card)!
        self.assertTrue(state.hand.cards[0].bounds.is_resolved)
        self.assertEqual(state.hand.cards[0].card.rank, bm.Rank.KING)
        self.assertEqual(state.hand.cards[0].card.suit, bm.Suit.HEARTS)

    def test_draw_adjacent_face_down_forgets_identity(self):
        """Drawing a face-down card immediately adjacent to a known face-down card blurs candidates."""
        state = gs.GameState()
        c_revealed = bm.PlayingCard(rank=bm.Rank.KING, suit=bm.Suit.HEARTS, id=101)
        hc_revealed = bm.HandCard(card=c_revealed, facing_back=True)
        hc_revealed.reveal_temporarily()

        # Hand: [FaceDown (King of Hearts)] at the far right of current hand
        state.hand = bm.Hand(cards=[hc_revealed], max_size=2)

        # Draw a new face-down card: it will be placed immediately adjacent at index 1!
        c_new_fd = bm.PlayingCard(rank=bm.Rank.FIVE, suit=bm.Suit.DIAMONDS, id=102)
        state.full_deck = [c_revealed, c_new_fd]
        state.draw_cards(count=1, face_down=True)

        self.assertEqual(len(state.hand.cards), 2)
        c0 = state.hand.cards[0]
        c1 = state.hand.cards[1]

        # Both are now adjacent face-down cards with identical card backs
        # The known card "forgets" single-card certainty and masks back to NONE
        self.assertFalse(c0.bounds.is_resolved)
        self.assertFalse(c1.bounds.is_resolved)
        self.assertEqual(c0.card.rank, bm.Rank.NONE)
        self.assertEqual(c0.card.suit, bm.Suit.NONE)
        # Both share the pooled candidates {101, 102}
        self.assertEqual(c0.bounds.candidate_card_ids, {101, 102})
        self.assertEqual(c1.bounds.candidate_card_ids, {101, 102})
        self.assertEqual(c0.bounds.possible_ranks, {bm.Rank.KING, bm.Rank.FIVE})
        self.assertEqual(c0.bounds.possible_suits, {bm.Suit.HEARTS, bm.Suit.DIAMONDS})

    def test_adjacent_face_down_distinguished_by_selection_and_seal(self):
        """Adjacent face-down cards do not conflate if visually distinguished by selection or seal sticker."""
        state = gs.GameState()
        c1 = bm.PlayingCard(rank=bm.Rank.KING, suit=bm.Suit.HEARTS, id=1)
        c2 = bm.PlayingCard(rank=bm.Rank.QUEEN, suit=bm.Suit.CLUBS, id=2)
        hc1 = bm.HandCard(card=c1, facing_back=True, selected=True)  # Raised!
        hc2 = bm.HandCard(card=c2, facing_back=True, selected=False)  # Resting!
        hc1.reveal_temporarily()
        hc2.reveal_temporarily()

        state.hand = bm.Hand(cards=[hc1, hc2])
        state.reconcile_face_down_adjacencies()

        # Selection state distinguishes them: no blurring!
        self.assertTrue(hc1.bounds.is_resolved)
        self.assertTrue(hc2.bounds.is_resolved)
        self.assertEqual(hc1.card.rank, bm.Rank.KING)
        self.assertEqual(hc2.card.rank, bm.Rank.QUEEN)

        # Now test seal distinction: both unselected, but one has a visible Blue Seal
        hc1.selected = False
        hc2.selected = False
        hc1.card.seal = bm.Seal.BLUE
        if hc1._real_card:
            hc1._real_card.seal = bm.Seal.BLUE
        hc2.card.seal = bm.Seal.NONE
        if hc2._real_card:
            hc2._real_card.seal = bm.Seal.NONE

        state.reconcile_face_down_adjacencies()
        self.assertTrue(hc1.bounds.is_resolved)
        self.assertTrue(hc2.bounds.is_resolved)
        self.assertEqual(hc1.card.rank, bm.Rank.KING)
        self.assertEqual(hc2.card.rank, bm.Rank.QUEEN)


class TestTelemetryParityAndLoggingAudit(unittest.TestCase):
    """Unit tests validating newly hooked telemetry actions, state deserialization, and audit fixes."""

    def test_action_parsing_new_types(self):
        """Validates mapping of Sort by Rank, Sort by Suit, Reorder Jokers, Reorder Hand, and Setup Test Suite."""
        act_rank = parse_lua_action("Sort by Rank", {"sort_type": "rank", "selected_card_ids": [42]}, 1, 0.0)
        self.assertEqual(act_rank.type, bm.ActionType.SORT_RANK)
        self.assertEqual(act_rank.metadata.get("selected_card_id"), 42)

        act_suit = parse_lua_action("Sort by Suit", {"sort_type": "suit", "selected_card_ids": [10]}, 2, 0.0)
        self.assertEqual(act_suit.type, bm.ActionType.SORT_SUIT)
        self.assertEqual(act_suit.metadata.get("selected_card_id"), 10)

        act_reorder_j = parse_lua_action("Reorder Jokers", {"joker_order": [102, 101]}, 3, 0.0)
        self.assertEqual(act_reorder_j.type, bm.ActionType.REORDER_JOKERS)

        act_reorder_h = parse_lua_action("Reorder Hand", {"hand_order": [5, 4]}, 4, 0.0)
        self.assertEqual(act_reorder_h.type, bm.ActionType.REORDER_HAND)

        act_setup = parse_lua_action("Setup Test Suite", {}, 5, 0.0)
        self.assertEqual(act_setup.type, bm.ActionType.SELECT_CARD)

    def test_parse_joker_loyalty_and_payout(self):
        """Validates that Loyalty Card loyalty_remaining and Rocket payout are properly extracted."""
        j_loyalty_raw = {
            "name": "Loyalty Card",
            "id": 99,
            "loyalty_remaining": 3,
            "extra": {"Xmult": 4, "every": 5},
        }
        j_loyalty = parse_joker(j_loyalty_raw)
        self.assertIsNotNone(j_loyalty)
        self.assertEqual(j_loyalty.state.get("remaining"), 3)

        j_rocket_raw = {
            "name": "Rocket",
            "id": 100,
            "payout": 6,
            "extra": {"dollars": 6, "increase": 2},
        }
        j_rocket = parse_joker(j_rocket_raw)
        self.assertIsNotNone(j_rocket)
        self.assertEqual(j_rocket.state.get("payout"), 6)

    def test_reorder_actions_execution(self):
        """REORDER_JOKERS and REORDER_HAND reorder elements by card IDs, and implicit joker_order works on any action."""
        j1 = bm.Joker(name=bm.JokerName.JOKER, id=1)
        j2 = bm.Joker(name=bm.JokerName.BLUEPRINT, id=2)
        state = gs.GameState(inventory=bm.Inventory(jokers=[j1, j2]))

        # Implicit reorder via PLAY_HAND action with joker_order
        c1 = bm.PlayingCard(rank=bm.Rank.TWO, suit=bm.Suit.HEARTS, id=11)
        state.hand = bm.Hand(cards=[bm.HandCard(card=c1, selected=True)])
        state, _ = gs.step_simulation(
            state,
            bm.Action(
                type=bm.ActionType.PLAY_HAND,
                joker_order=[2, 1],
            ),
        )
        self.assertEqual([j.id for j in state.inventory.jokers], [2, 1])

        # Explicit REORDER_JOKERS action
        state, _ = gs.step_simulation(
            state,
            bm.Action(
                type=bm.ActionType.REORDER_JOKERS,
                metadata={"joker_order": [1, 2]},
            ),
        )
        self.assertEqual([j.id for j in state.inventory.jokers], [1, 2])

        # Reorder hand cards by ID
        c2 = bm.PlayingCard(rank=bm.Rank.THREE, suit=bm.Suit.SPADES, id=12)
        state.hand = bm.Hand(cards=[bm.HandCard(card=c1), bm.HandCard(card=c2)])

        state, _ = gs.step_simulation(
            state,
            bm.Action(
                type=bm.ActionType.REORDER_HAND,
                metadata={"hand_order": [12, 11]},
            ),
        )
        self.assertEqual([hc.card.id for hc in state.hand.cards], [12, 11])

    def test_sync_observations_and_setup_test_suite(self):
        """Validates that sync_cv_observations_before_action updates state variables and refreshes deck on Setup Test Suite."""
        state = gs.GameState()
        lua_state = {
            "location": "shop",
            "boss_disabled": True,
            "inflation": 3,
            "pack_choices_remaining": 2,
            "deck": [
                {"value": "Ace", "suit": "Spades", "enhancement": "lucky", "id": 1},
                {"value": "King", "suit": "Hearts", "enhancement": "glass", "id": 2},
            ],
        }
        sync_cv_observations_before_action(state, lua_state, act_name="Setup Test Suite")

        self.assertEqual(state.location, bm.Location.SHOP)
        self.assertTrue(state.boss_disabled)
        self.assertEqual(state.inflation, 3)
        self.assertEqual(state.pack_choices_remaining, 2)
        self.assertEqual(len(state.full_deck), 2)
        self.assertEqual(state.full_deck[0].enhancement, bm.Enhancement.LUCKY)
        self.assertEqual(state.full_deck[1].enhancement, bm.Enhancement.GLASS)

    def test_archived_telemetry_regression_suite(self):
        """Validates zero discrepancies across all archived telemetry runs in testing/telemetry_data/."""
        archive_dir = REPO_ROOT / "testing" / "telemetry_data"
        if not archive_dir.exists():
            self.skipTest("No telemetry archive directory found.")
        jsonl_files = sorted(archive_dir.glob("*.jsonl"))
        if not jsonl_files:
            self.skipTest("No archived telemetry runs found in testing/telemetry_data/.")
        for f in jsonl_files:
            if f.stat().st_size == 0:
                continue
            with self.subTest(run=f.name):
                errors = test_trajectory_parity(f, verbose=False)
                self.assertEqual(
                    errors,
                    0,
                    f"Archived telemetry run {f.name} failed parity check with {errors} discrepancies!",
                )


def run_synthetic_test() -> None:
    """Convenience function running the synthetic test directly."""
    test = TestEngine()
    test.test_synthetic_simulation()
    print("Synthetic integration test passed successfully!")


if __name__ == "__main__":
    unittest.main()
