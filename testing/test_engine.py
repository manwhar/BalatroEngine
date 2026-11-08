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
JOKER_NAME_BY_STR.update({
    "Caino": bm.JokerName.CANIO,
    "Seance": bm.JokerName.SEANCE,
    "Riff-raff": bm.JokerName.RIFF_RAFF,
})

BLIND_BY_STR: dict[str, bm.BlindName] = {b.value: b for b in bm.BlindName}
BLIND_BY_STR.update({
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
})

POKER_HAND_BY_STR: dict[str, bm.PokerHand] = {h.value: h for h in bm.PokerHand}
TAROT_NAME_BY_STR: dict[str, bm.TarotName] = {t.value: t for t in bm.TarotName}
PLANET_NAME_BY_STR: dict[str, bm.PlanetName] = {p.value: p for p in bm.PlanetName}
SPECTRAL_NAME_BY_STR: dict[str, bm.SpectralName] = {s.value: s for s in bm.SpectralName}
VOUCHER_NAME_BY_STR: dict[str, bm.VoucherName] = {v.value: v for v in bm.VoucherName}


def parse_playing_card(d: dict) -> bm.PlayingCard:
    enh = ENHANCEMENT_MAP.get(d.get("enhancement", "none"), bm.Enhancement.NONE)
    rank = bm.Rank.NONE if enh == bm.Enhancement.STONE else RANK_MAP.get(d.get("value", "none"), bm.Rank.ACE)
    suit = bm.Suit.NONE if enh == bm.Enhancement.STONE else SUIT_MAP.get(d.get("suit", "none"), bm.Suit.SPADES)
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
    extra = d.get("extra")
    if isinstance(extra, dict):
        if "dollars" in extra and j_name == bm.JokerName.ROCKET:
            try:
                j.state["payout"] = int(extra["dollars"])
            except (ValueError, TypeError):
                pass
        if "remaining" in extra and j_name == bm.JokerName.LOYALTY_CARD:
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


def parse_lua_action(action_name: str, meta: dict | list, frame_idx: int, timestamp: float) -> bm.Action:
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
    }
    act_type = action_map.get(action_name, bm.ActionType.PLAY_HAND)
    if act_type == bm.ActionType.CASH_OUT:
        meta_dict.setdefault("include_round_payout", True)
    target_idx = meta_dict.get("target_idx")
    return bm.Action(
        type=act_type,
        frame_idx=frame_idx,
        timestamp=timestamp,
        target_idx=target_idx if isinstance(target_idx, int) else None,
        metadata=meta_dict,
    )


def sync_cv_observations_before_action(sim_state: gs.GameState, lua_state: dict) -> None:
    """Simulates a Perfect-CV/OCR camera observing ONLY screen-visible inputs at action time.

    Does NOT touch closed-loop engine state (money, hands_left, discards_left,
    reroll_cost, discard_pile, deck_cards_count) so the test harness can verify
    that the Python engine predicts them without ground truth.
    """
    # 1. Dealt hand cards & player card selection (visible on screen)
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
    if (
        "jokers" in sim_state.pending_cv_checks
        or [getattr(j, "id", 0) for j in sim_state.inventory.jokers]
        != [int(d.get("id", 0)) for d in lua_jokers_raw]
    ):
        existing_by_id = {j.id: j for j in sim_state.inventory.jokers if getattr(j, "id", 0) != 0}
        updated_jokers: list[bm.Joker] = []
        for d in lua_jokers_raw:
            j_id = int(d.get("id", 0))
            if j_id != 0 and j_id in existing_by_id:
                updated_jokers.append(existing_by_id.pop(j_id))
            else:
                pj = parse_joker(d)
                if pj is not None:
                    updated_jokers.append(pj)
                    on_joker_added(sim_state, pj)
        sim_state.inventory.jokers = updated_jokers
        sim_state.pending_cv_checks.discard("jokers")

    # Sync per-Joker rotating To Do List target poker hand (visible on Joker tooltip)
    todo_targets = [
        POKER_HAND_BY_STR[d["to_do_poker_hand"]]
        for d in lua_jokers_raw
        if d.get("name") == "To Do List" and d.get("to_do_poker_hand") in POKER_HAND_BY_STR
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
            c for d in lua_state["shop"]
            if (c := parse_any_card(d)) is not None
        ]
    if "shop_vouchers" in lua_state:
        sim_state.shop_vouchers = [
            c for d in lua_state["shop_vouchers"]
            if isinstance(c := parse_any_card(d), bm.Voucher)
        ]
    if "consumables" in lua_state:
        sim_state.inventory.consumables = [
            c for d in lua_state["consumables"]
            if (c := parse_any_card(d)) is not None
        ]
    if "pack_cards" in lua_state:
        sim_state.pack_cards = [
            c for d in lua_state["pack_cards"]
            if (c := parse_any_card(d)) is not None
        ]


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
        sync_cv_observations_before_action(sim_state, lua_state)

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
            elif act_name == "Discard" and getattr(sim_state, "_hook_pending_cards", None) is not None:
                action.metadata["hook"] = True

            # Detect when Play Hand (or The Hook's mid-play Discard) beats the blind and triggers end_round()
            if (
                (act_name == "Play Hand" and not action.metadata.get("defer_play_for_hook"))
                or (act_name == "Discard" and action.metadata.get("hook"))
            ):
                if lua_state.get("blind", "") != "" and next_lua_peek.get("blind", "") == "":
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
                if act_name in ("Play Hand", "Discard", "Reroll Shop", "Select Blind", "Cash Out"):
                    if sim_state.money != next_lua["money"]:
                        print(
                            f"[Mismatch Step {step} '{act_name}' -> Step {step+1}] "
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
                if "hands_left" in next_lua and sim_state.hands_left != next_lua["hands_left"]:
                    print(
                        f"[Mismatch Step {step} '{act_name}' -> Step {step+1}] "
                        f"Hands Left: Python={sim_state.hands_left} vs Lua={next_lua['hands_left']}"
                    )
                    errors += 1
                if "discards_left" in next_lua and sim_state.discards_left != next_lua["discards_left"]:
                    print(
                        f"[Mismatch Step {step} '{act_name}' -> Step {step+1}] "
                        f"Discards Left: Python={sim_state.discards_left} vs Lua={next_lua['discards_left']}"
                    )
                    errors += 1

            # Check Reroll Cost
            if act_name in ("Reroll Shop", "Cash Out") and "reroll_cost" in next_lua:
                if sim_state.reroll_cost != next_lua["reroll_cost"]:
                    print(
                        f"[Mismatch Step {step} '{act_name}' -> Step {step+1}] "
                        f"Reroll Cost: Python={sim_state.reroll_cost} vs Lua={next_lua['reroll_cost']}"
                    )
                    errors += 1

            # Check Deck Cards Count
            if "deck_cards_count" in next_lua and act_name in ("Play Hand", "Discard", "Select Blind", "Cash Out"):
                if sim_state.deck_cards_count != next_lua["deck_cards_count"]:
                    print(
                        f"[Mismatch Step {step} '{act_name}' -> Step {step+1}] "
                        f"Deck Cards Count: Python={sim_state.deck_cards_count} vs Lua={next_lua['deck_cards_count']}"
                    )
                    errors += 1

            # Check Discard Pile Count (Note: when blind is beaten or before Cash Out, Balatro's end_round already shuffled discard pile back into deck)
            target_chips = next_lua.get("target_chips", float("inf"))
            current_chips = next_lua.get("current_chips", 0)
            if current_chips >= target_chips:
                sim_state.discard_pile.clear()

            if "discard_pile" in next_lua and next_act != "Cash Out" and act_name in ("Discard", "Play Hand", "Select Blind"):
                if len(sim_state.discard_pile) != len(next_lua["discard_pile"]):
                    print(
                        f"[Mismatch Step {step} '{act_name}' -> Step {step+1}] "
                        f"Discard Pile Count: Python={len(sim_state.discard_pile)} vs Lua={len(next_lua['discard_pile'])}"
                    )
                    errors += 1

            # Check Joker Accumulator States (mult, x_mult, chips, extra_value) when Joker lineup is unchanged
            next_jokers_raw = next_lua.get("jokers", [])
            if len(next_jokers_raw) == len(sim_state.inventory.jokers):
                for j_idx, (py_j, lua_j) in enumerate(zip(sim_state.inventory.jokers, next_jokers_raw)):
                    if py_j.name.value != lua_j.get("name"):
                        continue
                    if py_j.id != 0 and lua_j.get("id", 0) != 0 and py_j.id != lua_j.get("id"):
                        continue
                    # Compare extra_value (Egg, Gift Card)
                    if "extra_value" in lua_j and py_j.extra_value != lua_j["extra_value"]:
                        print(
                            f"[Mismatch Step {step} '{act_name}' -> Step {step+1}] "
                            f"Joker[{j_idx}] {py_j.name.value} extra_value: Python={py_j.extra_value} vs Lua={lua_j['extra_value']}"
                        )
                        errors += 1
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
                                f"[Mismatch Step {step} '{act_name}' -> Step {step+1}] "
                                f"Joker[{j_idx}] {py_j.name.value} mult: Python={py_m} vs Lua={lua_m}"
                            )
                            errors += 1
                    # Compare x_mult accumulator (Campfire, Constellation, Hologram, Vampire, Obelisk, Yorick, Ramen, Hit the Road, Canio)
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
                    ) or (py_j.name == bm.JokerName.GLASS_JOKER and bool(sim_state.seed)):
                        py_xm = round(float(py_j.state.get("x_mult", 1.0)), 2)
                        lua_xm = round(float(lua_j.get("x_mult", 1.0)), 2)
                        if abs(py_xm - lua_xm) > 1e-2:
                            print(
                                f"[Mismatch Step {step} '{act_name}' -> Step {step+1}] "
                                f"Joker[{j_idx}] {py_j.name.value} x_mult: Python={py_xm} vs Lua={lua_xm}"
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
                                f"[Mismatch Step {step} '{act_name}' -> Step {step+1}] "
                                f"Joker[{j_idx}] {py_j.name.value} chips: Python={py_c} vs Lua={lua_c}"
                            )
                            errors += 1

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
                jokers=[bm.Joker(name=bm.JokerName.GIFT_CARD), bm.Joker(name=bm.JokerName.JOKER)]
            ),
        )
        traj = gs.RunTrajectory(keyframe_interval=3)

        # 1. Reroll Shop
        sim_state, _ = gs.step_simulation(sim_state, bm.Action(type=bm.ActionType.REROLL_SHOP))
        traj.record_step(sim_state)
        self.assertEqual(sim_state.money, 10)
        self.assertEqual(sim_state.reroll_cost, 6)

        # 2. Cash out (triggers Gift Card on both jokers)
        sim_state, _ = gs.step_simulation(sim_state, bm.Action(type=bm.ActionType.CASH_OUT))
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
            hand=bm.Hand(cards=[
                bm.HandCard(card=bm.PlayingCard(rank=bm.Rank.ACE, suit=bm.Suit.SPADES), selected=True),
                bm.HandCard(card=bm.PlayingCard(rank=bm.Rank.TWO, suit=bm.Suit.HEARTS), selected=True),
            ]),
        )
        # Playing High Card Ace with an un-scored 2: Wee Joker should NOT gain chips
        sim_state, _ = gs.step_simulation(sim_state, bm.Action(type=bm.ActionType.PLAY_HAND))
        self.assertEqual(wee.state.get("chips", 0), 0)

        # Now playing Pair of 2s with King kicker: Wee Joker gains 16 chips (2 * 8)
        sim_state.hand = bm.Hand(cards=[
            bm.HandCard(card=bm.PlayingCard(rank=bm.Rank.TWO, suit=bm.Suit.SPADES), selected=True),
            bm.HandCard(card=bm.PlayingCard(rank=bm.Rank.TWO, suit=bm.Suit.CLUBS), selected=True),
            bm.HandCard(card=bm.PlayingCard(rank=bm.Rank.KING, suit=bm.Suit.DIAMONDS), selected=True),
        ])
        sim_state, _ = gs.step_simulation(sim_state, bm.Action(type=bm.ActionType.PLAY_HAND))
        self.assertEqual(wee.state.get("chips", 0), 16)

    def test_ride_the_bus_ignores_unscored_face_cards(self):
        """Ride the Bus must NOT reset when face cards are played as un-scored kickers."""
        bus = bm.Joker(name=bm.JokerName.RIDE_THE_BUS)
        sim_state = gs.GameState(
            inventory=bm.Inventory(jokers=[bus]),
            hand=bm.Hand(cards=[
                bm.HandCard(card=bm.PlayingCard(rank=bm.Rank.EIGHT, suit=bm.Suit.SPADES), selected=True),
                bm.HandCard(card=bm.PlayingCard(rank=bm.Rank.EIGHT, suit=bm.Suit.CLUBS), selected=True),
                bm.HandCard(card=bm.PlayingCard(rank=bm.Rank.KING, suit=bm.Suit.DIAMONDS), selected=True),
            ]),
        )
        # Hand is Pair of 8s. King is un-scored kicker. Ride the Bus should GAIN +1 mult, not reset!
        sim_state, _ = gs.step_simulation(sim_state, bm.Action(type=bm.ActionType.PLAY_HAND))
        self.assertEqual(bus.state.get("mult", 0), 1)

        # Now playing High Card King: King is the scoring card, so Ride the Bus must reset to 0!
        sim_state.hand = bm.Hand(cards=[
            bm.HandCard(card=bm.PlayingCard(rank=bm.Rank.KING, suit=bm.Suit.SPADES), selected=True),
            bm.HandCard(card=bm.PlayingCard(rank=bm.Rank.EIGHT, suit=bm.Suit.HEARTS), selected=True),
        ])
        sim_state, _ = gs.step_simulation(sim_state, bm.Action(type=bm.ActionType.PLAY_HAND))
        self.assertEqual(bus.state.get("mult", 0), 0)

    def test_hiker_and_vampire_only_affect_scoring_cards(self):
        """Hiker and Vampire must exclusively modify scoring cards, leaving kickers intact."""
        hiker = bm.Joker(name=bm.JokerName.HIKER)
        vampire = bm.Joker(name=bm.JokerName.VAMPIRE)
        c8a = bm.PlayingCard(rank=bm.Rank.EIGHT, suit=bm.Suit.SPADES, enhancement=bm.Enhancement.GOLD)
        c8b = bm.PlayingCard(rank=bm.Rank.EIGHT, suit=bm.Suit.CLUBS, enhancement=bm.Enhancement.GOLD)
        c5 = bm.PlayingCard(rank=bm.Rank.FIVE, suit=bm.Suit.HEARTS, enhancement=bm.Enhancement.GOLD)

        sim_state = gs.GameState(
            inventory=bm.Inventory(jokers=[hiker, vampire]),
            hand=bm.Hand(cards=[
                bm.HandCard(card=c8a, selected=True),
                bm.HandCard(card=c8b, selected=True),
                bm.HandCard(card=c5, selected=True),
            ]),
        )
        sim_state, _ = gs.step_simulation(sim_state, bm.Action(type=bm.ActionType.PLAY_HAND))

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
            hand=bm.Hand(cards=[
                bm.HandCard(card=bm.PlayingCard(rank=bm.Rank.TWO, suit=bm.Suit.SPADES), selected=True),
                bm.HandCard(card=bm.PlayingCard(rank=bm.Rank.THREE, suit=bm.Suit.CLUBS), selected=True),
                bm.HandCard(card=bm.PlayingCard(rank=bm.Rank.FOUR, suit=bm.Suit.HEARTS), selected=True),
            ]),
        )
        sim_state, _ = gs.step_simulation(sim_state, bm.Action(type=bm.ActionType.DISCARD))
        self.assertAlmostEqual(ramen.state.get("x_mult", 2.0), 1.97)

    def test_dna_card_duplication_to_deck_and_hand(self):
        """DNA clones the single played card on first hand: adds to full_deck and directly to hand."""
        dna = bm.Joker(name=bm.JokerName.DNA)
        hologram = bm.Joker(name=bm.JokerName.HOLOGRAM)
        target_card = bm.PlayingCard(rank=bm.Rank.ACE, suit=bm.Suit.SPADES)
        other_card = bm.PlayingCard(rank=bm.Rank.TEN, suit=bm.Suit.HEARTS)

        sim_state = gs.GameState(
            inventory=bm.Inventory(jokers=[dna, hologram]),
            hand=bm.Hand(cards=[
                bm.HandCard(card=target_card, selected=True),
                bm.HandCard(card=other_card, selected=False),
            ]),
        )
        self.assertEqual(len(sim_state.full_deck), 52)

        # 1. Play first hand with 1 card (Ace of Spades)
        sim_state, _ = gs.step_simulation(sim_state, bm.Action(type=bm.ActionType.PLAY_HAND))

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
        sim_state, _ = gs.step_simulation(sim_state, bm.Action(type=bm.ActionType.PLAY_HAND))
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
        sim_state, _ = gs.step_simulation(sim_state, bm.Action(type=bm.ActionType.PLAY_HAND))

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
            hand=bm.Hand(cards=[
                bm.HandCard(card=face_card, selected=True),
                bm.HandCard(card=other_card, selected=False),
            ]),
        )
        initial_money = sim_state.money

        sim_state, _ = gs.step_simulation(sim_state, bm.Action(type=bm.ActionType.DISCARD))

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
            hand=bm.Hand(cards=[
                bm.HandCard(card=c1, selected=True),
                bm.HandCard(card=c2, selected=False),
                bm.HandCard(card=c3, selected=False),
            ]),
        )

        sim_state, _ = gs.step_simulation(sim_state, bm.Action(type=bm.ActionType.DISCARD))
        self.assertIn(c1, sim_state.discard_pile)
        self.assertEqual(len(sim_state.discard_pile), 1)

        sim_state.hand.cards[0].selected = True  # c2
        sim_state, _ = gs.step_simulation(sim_state, bm.Action(type=bm.ActionType.PLAY_HAND))
        self.assertIn(c2, sim_state.played_cards_this_round)
        self.assertEqual(len(sim_state.played_cards_this_round), 1)

        sim_state, _ = gs.step_simulation(sim_state, bm.Action(type=bm.ActionType.SELECT_BLIND))
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
        sim_state, _ = gs.step_simulation(sim_state, bm.Action(type=bm.ActionType.PLAY_HAND))

        self.assertEqual(sim_state.deck_cards_count, 51)
        self.assertNotIn(c6, sim_state.full_deck)
        self.assertEqual(len(sim_state.hand.cards), 0)
        self.assertNotIn(c6, sim_state.played_cards_this_round)

    def test_marble_joker_adds_stone_card(self):
        """Marble Joker appends a Stone card to full_deck when selecting a blind."""
        marble = bm.Joker(name=bm.JokerName.MARBLE_JOKER)
        sim_state = gs.GameState(inventory=bm.Inventory(jokers=[marble]))
        initial_count = len(sim_state.full_deck)

        sim_state, _ = gs.step_simulation(sim_state, bm.Action(type=bm.ActionType.SELECT_BLIND))
        self.assertEqual(len(sim_state.full_deck), initial_count + 1)
        self.assertEqual(sim_state.full_deck[-1].enhancement, bm.Enhancement.STONE)

    def test_cloud_9_evaluates_full_deck(self):
        """Cloud 9 counts all 9s in the full_deck at end of round."""
        c9 = bm.Joker(name=bm.JokerName.CLOUD_9)
        sim_state = gs.GameState(inventory=bm.Inventory(jokers=[c9]))
        initial_money = sim_state.money

        sim_state, _ = gs.step_simulation(sim_state, bm.Action(type=bm.ActionType.CASH_OUT))
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
        state, _ = gs.step_simulation(state, bm.Action(type=bm.ActionType.SELL_CARD, target_idx=1))
        self.assertAlmostEqual(campfire.state["x_mult"], 1.25)
        state, _ = gs.step_simulation(state, bm.Action(type=bm.ActionType.CASH_OUT))
        self.assertAlmostEqual(campfire.state["x_mult"], 1.0)

        # 2. Yorick carries over excess discards across the 23 threshold
        yorick = bm.Joker(name=bm.JokerName.YORICK)
        yorick.state["discards"] = 2
        state_y = gs.GameState(
            inventory=bm.Inventory(jokers=[yorick]),
            hand=bm.Hand(cards=[
                bm.HandCard(card=bm.PlayingCard(rank=bm.Rank.TWO, suit=bm.Suit.SPADES), selected=True),
                bm.HandCard(card=bm.PlayingCard(rank=bm.Rank.THREE, suit=bm.Suit.SPADES), selected=True),
                bm.HandCard(card=bm.PlayingCard(rank=bm.Rank.FOUR, suit=bm.Suit.SPADES), selected=True),
                bm.HandCard(card=bm.PlayingCard(rank=bm.Rank.FIVE, suit=bm.Suit.SPADES), selected=True),
                bm.HandCard(card=bm.PlayingCard(rank=bm.Rank.SIX, suit=bm.Suit.SPADES), selected=True),
            ]),
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
            hand=bm.Hand(cards=[
                bm.HandCard(card=bm.PlayingCard(rank=bm.Rank.ACE, suit=bm.Suit.SPADES), selected=True),
            ]),
        )
        # Playing High Card brings HIGH_CARD to 3 (tied with PAIR=3) -> must NOT reset, scales to 1.6!
        state_ob, _ = gs.step_simulation(state_ob, bm.Action(type=bm.ActionType.PLAY_HAND))
        self.assertAlmostEqual(obelisk.state["x_mult"], 1.6)
        # Playing High Card again brings HIGH_CARD to 4 (strictly > PAIR=3) -> resets to 1.0!
        state_ob.hand.cards = [bm.HandCard(card=bm.PlayingCard(rank=bm.Rank.KING, suit=bm.Suit.SPADES), selected=True)]
        state_ob, _ = gs.step_simulation(state_ob, bm.Action(type=bm.ActionType.PLAY_HAND))
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
        state_hk, _ = gs.step_simulation(state_hk, bm.Action(type=bm.ActionType.PLAY_HAND))
        self.assertEqual(c_ace.bonus_chips, 30)

        # 5. Turtle Bean modifies hand.max_size on add and decrements hand.max_size at end of round
        bean = bm.Joker(name=bm.JokerName.TURTLE_BEAN)
        state_tb = gs.GameState()
        state_tb.inventory.jokers.append(bean)
        on_joker_added(state_tb, bean)
        self.assertEqual(state_tb.hand.max_size, 13)  # 8 + 5
        state_tb, _ = gs.step_simulation(state_tb, bm.Action(type=bm.ActionType.CASH_OUT))
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
            hand=bm.Hand(cards=[
                bm.HandCard(card=ace1, selected=True, forced_selection=True),
                bm.HandCard(card=ace2, selected=False),
            ]),
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

        state, _ = gs.step_simulation(state, bm.Action(type=bm.ActionType.OPEN_PACK, target_idx=0))
        state.pack_cards = [
            bm.Tarot(name=bm.TarotName.HERMIT),
            bm.Spectral(name=bm.SpectralName.ECTOPLASM),
        ]
        state.hand_levels[bm.PokerHand.PAIR] = 3
        state.inventory.joker_slots = 6
        traj.record_step(state)
        self.assertEqual(state.location, bm.Location.PACK)
        self.assertEqual(state.pack_choices_remaining, 2)

        state, _ = gs.step_simulation(state, bm.Action(type=bm.ActionType.USE_IN_PACK, target_idx=0))
        traj.record_step(state)
        self.assertEqual(state.last_tarot_planet, bm.TarotName.HERMIT)
        self.assertEqual(state.tarots_used, 1)
        self.assertEqual(state.pack_choices_remaining, 1)
        self.assertEqual(state.location, bm.Location.PACK)

        state, _ = gs.step_simulation(state, bm.Action(type=bm.ActionType.USE_IN_PACK, target_idx=0))
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
        appdata = os.environ.get("APPDATA")
        if appdata:
            telemetry_paths.append(Path(appdata) / "Balatro" / "run_telemetry.jsonl")
        for p in telemetry_paths:
            if p.exists() and p.stat().st_size > 0:
                errors = test_trajectory_parity(p)
                self.assertEqual(errors, 0, f"Encountered {errors} telemetry parity errors in {p}")


def run_synthetic_test() -> None:
    """Convenience function running the synthetic test directly."""
    test = TestEngine()
    test.test_synthetic_simulation()
    print("Synthetic integration test passed successfully!")


if __name__ == "__main__":
    unittest.main()
