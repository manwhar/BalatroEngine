"""Tests for `tools/visualize_trajectory.py` sprite manifest, diffs, zones, and HTTP endpoints."""

from __future__ import annotations

import http.client
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from game.balatro_models import (
    Action,
    ActionType,
    BlindName,
    BoosterPack,
    BoosterSize,
    BoosterType,
    Deck,
    Edition,
    Enhancement,
    Hand,
    HandCard,
    Inventory,
    Joker,
    JokerName,
    Location,
    Planet,
    PlanetName,
    PlayingCard,
    PokerHand,
    Rank,
    Seal,
    SkipTag,
    SpectralName,
    Stake,
    Suit,
    Tarot,
    TarotName,
    Voucher,
    VoucherName,
)
from game.gamestate import GameState, RunTrajectory
from game.trajectory_codec import serialize_trajectory
from tools.visualize_trajectory import (
    TEXTURES_2X_DIR,
    TrajectoryVisualizerServer,
    build_visualization_payload,
    find_matching_video,
    load_balatro_sprite_manifest,
)


def _sample_card(cid: int, rank: Rank, suit: Suit, **kwargs) -> PlayingCard:
    return PlayingCard(id=cid, rank=rank, suit=suit, **kwargs)


class TestSpriteManifest(unittest.TestCase):
    """Verifies Balatro `game.lua` sprite extraction covers all engine enums."""

    def test_manifest_covers_all_game_enums(self) -> None:
        manifest = load_balatro_sprite_manifest()

        missing: dict[str, list[str]] = {}
        enum_checks: list[tuple[str, list[str]]] = [
            ("jokers", [e.value for e in JokerName]),
            ("tarots", [e.value for e in TarotName]),
            ("planets", [e.value for e in PlanetName]),
            ("spectrals", [e.value for e in SpectralName]),
            ("vouchers", [e.value for e in VoucherName]),
            ("backs", [e.value for e in Deck]),
            ("tags", [e.value for e in SkipTag]),
            ("blinds", [e.value for e in BlindName]),
            ("stakes", [e.value for e in Stake]),
            ("enhancements", [e.value for e in Enhancement]),
            ("seals", [s.value for s in Seal if s != Seal.NONE]),
            ("boosters", [f"{sz.value} {bt.value}" for bt in BoosterType for sz in BoosterSize]),
        ]
        for bucket, names in enum_checks:
            miss = [n for n in names if n not in manifest[bucket]]
            if miss:
                missing[bucket] = miss

        self.assertEqual(missing, {})

        grid = manifest["playing_card_grid"]
        for rank in Rank:
            if rank != Rank.NONE:
                self.assertIn(rank.value, grid["ranks"])
        for suit in Suit:
            if suit != Suit.NONE:
                self.assertIn(suit.value, grid["suits"])

        # Soul overlay layers on Hologram & Legendaries
        for soul_joker in (
            JokerName.HOLOGRAM,
            JokerName.CANIO,
            JokerName.TRIBOULET,
            JokerName.YORICK,
            JokerName.CHICOT,
            JokerName.PERKEO,
        ):
            self.assertIn("soul", manifest["jokers"][soul_joker.value])

        self.assertIn("debuff", manifest["overlays"])
        for st_key in ("eternal", "perishable", "rental"):
            self.assertIn(st_key, manifest["stickers"])


class TestVisualizationPayload(unittest.TestCase):
    """Verifies step reconstruction, pre-action targets, deck zones, and diffs."""

    def _build_synthetic_trajectory_json(self) -> dict:
        c1 = _sample_card(1, Rank.ACE, Suit.SPADES)
        c2 = _sample_card(2, Rank.KING, Suit.SPADES)
        c3 = _sample_card(3, Rank.QUEEN, Suit.HEARTS)
        c4 = _sample_card(4, Rank.JACK, Suit.CLUBS)
        c5 = _sample_card(5, Rank.TEN, Suit.DIAMONDS)
        deck = [c1, c2, c3, c4, c5]

        # Step 0: Blind with 2 cards in hand, 1 in discard, 2 in draw pile
        state = GameState(
            step_idx=0,
            frame_idx=100,
            timestamp=1.5,
            location=Location.BLIND,
            deck=Deck.ANAGLYPH_DECK,
            stake=Stake.GOLD,
            seed="TESTSEED",
            ante=1,
            round=1,
            money=10,
            hands_left=4,
            discards_left=3,
            reroll_cost=5,
            target_chips=300,
            current_chips=0,
            full_deck=[PlayingCard(**vars(c)) for c in deck],
            hand=Hand(cards=[HandCard(card=_sample_card(1, Rank.ACE, Suit.SPADES), selected=True, slot_uid=101),
                             HandCard(card=_sample_card(2, Rank.KING, Suit.SPADES), slot_uid=102)]),
            discard_pile=[_sample_card(3, Rank.QUEEN, Suit.HEARTS)],
            played_cards_this_round=[],
            played_cards_this_ante=[],
            inventory=Inventory(
                jokers=[Joker(name=JokerName.JOKER, id=11)],
                consumables=[Tarot(name=TarotName.EMPRESS)],
            ),
            shop_cards=[Joker(name=JokerName.GREEDY_JOKER, id=12, edition=Edition.FOIL)],
            shop_vouchers=[Voucher(name=VoucherName.OVERSTOCK)],
            shop_packs=[BoosterPack(booster_type=BoosterType.ARCANA, size=BoosterSize.REGULAR, cost=4, choose=1, extra=3)],
            apply_deck_stake_on_init=False,
        )
        traj = RunTrajectory()
        traj.record_step(state)

        # Step 1: PLAY_HAND with Ace of Spades + PokerHand level & play count increment
        state = state.copy()
        state.frame_idx = 200
        state.timestamp = 3.2
        state.hands_left = 3
        state.current_chips = 150
        state.hand_levels[PokerHand.HIGH_CARD] = 2
        state.hand_play_counts[PokerHand.HIGH_CARD] = 1
        state.hands_played_this_round_by_type.add(PokerHand.HIGH_CARD)
        state.played_cards_this_round.append(_sample_card(1, Rank.ACE, Suit.SPADES))
        state.played_cards_this_ante.append(_sample_card(1, Rank.ACE, Suit.SPADES))
        state.hand.cards = [HandCard(card=_sample_card(2, Rank.KING, Suit.SPADES), slot_uid=102)]
        traj.record_step(
            state,
            Action(
                type=ActionType.PLAY_HAND,
                frame_idx=200,
                timestamp=3.2,
                selected_cards=[_sample_card(1, Rank.ACE, Suit.SPADES)],
            ),
        )

        # Step 2: BUY_CARD at shop index 0 (Foil Greedy Joker) & deck card modification + addition
        state = state.copy()
        state.frame_idx = 300
        state.timestamp = 5.0
        state.location = Location.SHOP
        state.money = 3
        bought = state.shop_cards.pop(0)
        state.inventory.jokers.append(bought)
        state.full_deck[1].enhancement = Enhancement.GLASS
        state.full_deck.append(_sample_card(6, Rank.TWO, Suit.HEARTS, seal=Seal.RED))
        traj.record_step(
            state,
            Action(type=ActionType.BUY_CARD, frame_idx=300, timestamp=5.0, target_idx=0),
        )

        # Step 3: SELL_CARD at index 2 (consumable slot 0: The Empress, after 2 jokers)
        state = state.copy()
        state.frame_idx = 400
        state.timestamp = 6.7
        state.money = 4
        state.inventory.consumables.pop(0)
        traj.record_step(
            state,
            Action(type=ActionType.SELL_CARD, frame_idx=400, timestamp=6.7, target_idx=2),
        )

        return serialize_trajectory(state, traj)

    def test_zones_targets_and_step_diffs(self) -> None:
        raw_json = self._build_synthetic_trajectory_json()
        payload = build_visualization_payload(raw_json)

        self.assertEqual(payload["num_steps"], 4)
        self.assertEqual(payload["run_meta"]["deck"], "Anaglyph Deck")
        self.assertEqual(payload["run_meta"]["stake"], "Gold Stake")

        # Step 0 zones: #1, #2 in_hand; #3 discarded; #4, #5 in_draw_pile
        s0 = payload["steps"][0]["state"]
        zones0 = {c["id"]: c["zone"] for c in s0["full_deck"]}
        self.assertEqual(
            zones0,
            {
                1: "in_hand",
                2: "in_hand",
                3: "discarded",
                4: "in_draw_pile",
                5: "in_draw_pile",
            },
        )
        self.assertFalse(payload["steps"][0]["diff"]["has_changes"])

        # Step 1 (PLAY_HAND): zone moves Ace♠ -> played_round; PokerHand High Card diffed
        step1 = payload["steps"][1]
        self.assertEqual(step1["action"]["target_item"]["area"], "hand")
        self.assertEqual(step1["action"]["target_item"]["label"], "Ace♠")
        zones1 = {c["id"]: c["zone"] for c in step1["state"]["full_deck"]}
        self.assertEqual(zones1[1], "played_round")
        self.assertEqual(
            step1["diff"]["full_deck"]["zone_changed"]["1"],
            {"prev": "in_hand", "curr": "played_round"},
        )
        self.assertEqual(
            step1["diff"]["poker_hands"]["High Card"]["level"],
            {"prev": 1, "curr": 2},
        )
        self.assertEqual(
            step1["diff"]["poker_hands"]["High Card"]["play_count"],
            {"prev": 0, "curr": 1},
        )

        # Step 2 (BUY_CARD): target resolved to pre-action shop_cards[0] (Greedy Joker),
        # joker added_ids=[12], full_deck card #6 added, card #2 modified -> glass
        step2 = payload["steps"][2]
        self.assertEqual(step2["action"]["target_item"]["area"], "shop_cards")
        self.assertEqual(step2["action"]["target_item"]["label"], "Greedy Joker")
        self.assertEqual(step2["diff"]["jokers"]["added_ids"], [12])
        self.assertEqual(step2["diff"]["full_deck"]["added_ids"], [6])
        self.assertEqual(
            step2["diff"]["full_deck"]["modified"]["2"]["enhancement"],
            {"prev": "none", "curr": "glass"},
        )

        # Step 3 (SELL_CARD idx=2 -> consumable #0 The Empress after 2 jokers)
        step3 = payload["steps"][3]
        self.assertEqual(step3["action"]["target_item"]["area"], "consumables")
        self.assertEqual(step3["action"]["target_item"]["label"], "The Empress")
        self.assertEqual(len(step3["diff"]["consumables"]["removed"]), 1)
        self.assertEqual(step3["diff"]["consumables"]["removed"][0]["name"], "The Empress")

    def test_anomaly_fixes_sell_reorder_shop_stickers_observed_shop_and_hook(self) -> None:
        from game.engine.rules import on_action
        from game.trajectory_codec import _serialize_card, _serialize_voucher

        deck = [
            _sample_card(1, Rank.ACE, Suit.SPADES),
            _sample_card(2, Rank.KING, Suit.SPADES),
            _sample_card(3, Rank.QUEEN, Suit.HEARTS),
            _sample_card(4, Rank.EIGHT, Suit.SPADES),
        ]
        state = GameState(
            seed="FIXTEST",
            deck=Deck.RED_DECK,
            stake=Stake.GOLD,
            location=Location.SHOP,
            money=20,
            current_blind=BlindName.THE_HOOK,
            full_deck=[PlayingCard(**vars(c)) for c in deck],
            inventory=Inventory(
                jokers=[
                    Joker(name=JokerName.RAMEN, id=1),
                    Joker(name=JokerName.JOKER, id=2, rental=True),
                ],
            ),
            apply_deck_stake_on_init=False,
        )
        traj = RunTrajectory()
        traj.record_step(state)

        # 1) BUY_CARD with observed_shop (Supernova + Perishable Jolly Joker + Overstock)
        supernova = Joker(name=JokerName.SUPERNOVA)
        jolly = Joker(name=JokerName.JOLLY_JOKER, perishable=True, perish_tally=5)
        overstock = Voucher(name=VoucherName.OVERSTOCK)
        state.shop_cards = [supernova, jolly]
        state.shop_vouchers = [overstock]
        buy_act = Action(
            type=ActionType.BUY_CARD,
            frame_idx=100,
            timestamp=1.5,
            target_idx=0,
            metadata={
                "target_card": "Supernova",
                "observed_shop": {
                    "cards": [_serialize_card(supernova), _serialize_card(jolly)],
                    "vouchers": [_serialize_voucher(overstock)],
                    "packs": [],
                },
            },
        )
        on_action(state, buy_act)
        traj.record_step(state, buy_act)

        # 2) NEXT_ROUND clears shop shelves cleanly (_leave_shop fix)
        leave_act = Action(type=ActionType.NEXT_ROUND, frame_idx=150, timestamp=2.5)
        on_action(state, leave_act)
        traj.record_step(state, leave_act)
        self.assertEqual(state.shop_cards, [])
        self.assertEqual(state.shop_vouchers, [])
        self.assertEqual(state.shop_packs, [])

        # 3) Enter Blind + SELL_CARD with joker_order reordering Joker(#2) ahead of Ramen(#1)
        state.location = Location.BLIND
        state.current_blind = BlindName.THE_HOOK
        state.hand = Hand(
            cards=[
                HandCard(card=_sample_card(1, Rank.ACE, Suit.SPADES), selected=True, slot_uid=101),
                HandCard(card=_sample_card(2, Rank.KING, Suit.SPADES), selected=False, slot_uid=102),
                HandCard(card=_sample_card(3, Rank.QUEEN, Suit.HEARTS), selected=False, slot_uid=103),
                HandCard(card=_sample_card(4, Rank.EIGHT, Suit.SPADES), selected=False, slot_uid=104),
            ]
        )
        sell_act = Action(
            type=ActionType.SELL_CARD,
            frame_idx=200,
            timestamp=3.3,
            target_idx=0,
            joker_order=[2, 1, 3],
            metadata={"target_card": "Joker"},
        )
        on_action(state, sell_act)
        traj.record_step(state, sell_act)

        # 4) PLAY_HAND against The Hook separates played card vs Hook forced discards
        play_act = Action(
            type=ActionType.PLAY_HAND,
            frame_idx=250,
            timestamp=4.1,
            selected_cards=[_sample_card(1, Rank.ACE, Suit.SPADES)],
        )
        on_action(state, play_act)
        traj.record_step(state, play_act)

        payload = build_visualization_payload(serialize_trajectory(state, traj))
        s1 = payload["steps"][1]
        self.assertEqual(s1["action"]["target_item"]["label"], "Supernova")
        self.assertEqual(len(s1["action"]["observed_shop"]["cards"]), 2)
        # Shop Jolly Joker keeps perishable metadata cleanly
        self.assertTrue(s1["state"]["shop_cards"][0]["perishable"])
        self.assertEqual(s1["state"]["shop_cards"][0]["perish_tally"], 5)

        s2 = payload["steps"][2]
        self.assertEqual(s2["state"]["shop_cards"], [])

        s3 = payload["steps"][3]
        self.assertEqual(s3["action"]["target_item"]["label"], "Joker")

        s4 = payload["steps"][4]
        self.assertEqual([c["rank"] for c in s4["diff"]["hand"]["played_cards"]], ["Ace"])
        self.assertEqual(len(s4["diff"]["hand"]["discarded_cards"]), 2)


class TestVisualizerServer(unittest.TestCase):
    """Tests HTTP API endpoints, static textures, security checks, and frame serving."""

    def test_http_endpoints_and_security(self) -> None:
        real_traj = (
            REPO_ROOT
            / "output"
            / "trajectories"
            / "bu_anaglyph_deck_11-13-26_1794699311_trajectory.json"
        )
        sample_video = (
            REPO_ROOT / "assets" / "sample_inputs" / "videos" / "bu_anaglyph_deck_11-13-26.mp4"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_traj = Path(tmpdir) / "mini_trajectory.json"
            if real_traj.is_file():
                traj_to_serve = real_traj
            else:
                state = GameState(apply_deck_stake_on_init=False)
                traj = RunTrajectory()
                traj.record_step(state)
                tmp_traj.write_text(json.dumps(serialize_trajectory(state, traj)), encoding="utf-8")
                traj_to_serve = tmp_traj

            matched = find_matching_video(traj_to_serve)
            if real_traj.is_file() and sample_video.is_file():
                self.assertEqual(matched, sample_video.resolve())

            server = TrajectoryVisualizerServer(
                trajectory_path=traj_to_serve,
                video_path=matched,
                host="127.0.0.1",
                port=0,
            )
            thread = threading.Thread(target=server.httpd.serve_forever, daemon=True)
            thread.start()
            try:
                conn = http.client.HTTPConnection(server.host, server.port, timeout=10)
                try:
                    # 1. HTML root
                    conn.request("GET", "/")
                    r_html = conn.getresponse()
                    html_body = r_html.read()
                    self.assertEqual(r_html.status, 200)
                    self.assertIn(b"BALATRO TRAJECTORY", html_body)
                    self.assertIn(b"item.eternal", html_body)
                    self.assertIn(b"item.perishable", html_body)
                    self.assertIn(b"HOOK DISCARD", html_body)

                    # 2. Manifest
                    conn.request("GET", "/api/manifest")
                    r_man = conn.getresponse()
                    man_data = json.loads(r_man.read().decode("utf-8"))
                    self.assertEqual(r_man.status, 200)
                    self.assertIn("jokers", man_data)

                    # 3. Trajectories list & active trajectory
                    conn.request("GET", "/api/trajectories")
                    r_list = conn.getresponse()
                    r_list.read()
                    self.assertEqual(r_list.status, 200)

                    conn.request("GET", "/api/trajectory")
                    r_traj = conn.getresponse()
                    traj_data = json.loads(r_traj.read().decode("utf-8"))
                    self.assertEqual(r_traj.status, 200)
                    self.assertGreaterEqual(traj_data["num_steps"], 1)

                    # 4. Texture asset + traversal guard
                    if (TEXTURES_2X_DIR / "Jokers.png").is_file():
                        conn.request("GET", "/textures/Jokers.png")
                        r_tex = conn.getresponse()
                        tex_bytes = r_tex.read()
                        self.assertEqual(r_tex.status, 200)
                        self.assertTrue(tex_bytes.startswith(b"\x89PNG"))

                    conn.request("GET", "/textures/../game.lua")
                    r_sec = conn.getresponse()
                    r_sec.read()
                    self.assertEqual(r_sec.status, 403)

                    # 5. Live MP4 frame endpoint (when video present)
                    if matched and matched.is_file():
                        conn.request("GET", "/api/frame/0")
                        r_frm = conn.getresponse()
                        jpg_bytes = r_frm.read()
                        self.assertEqual(r_frm.status, 200)
                        self.assertTrue(jpg_bytes.startswith(b"\xff\xd8"))
                finally:
                    conn.close()
            finally:
                server.close()
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
