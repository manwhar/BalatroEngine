"""Unit and mocked end-to-end integration tests for `post_processing.py` and `video_utils.extract_video_frames`."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from classifiers.common.video_utils import extract_video_frames
from classifiers.inference import HandRecognitionOutput, ShopRecognitionOutput
import game.balatro_models as bm
import game.gamestate as gs
import post_processing as pp


def _make_card_res(
    rank: str,
    suit: str,
    *,
    selected: bool = False,
    face_down: bool = False,
    edition: str = "None",
    enhancement: str = "None",
    seal: str = "None",
    debuffed: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        rank=rank,
        suit=suit,
        edition=edition,
        enhancement=enhancement,
        seal=seal,
        selected=selected,
        debuffed=debuffed,
        face_down=face_down,
        box=[0, 0, 100, 150],
    )


def _make_shop_item(
    key: str,
    card_type: str,
    box: list[int],
    *,
    edition: str = "None",
    eternal: bool = False,
    perishable: bool = False,
    rental: bool = False,
    debuffed: bool = False,
    card_details: Any = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        key=key,
        card_type=card_type,
        edition=edition,
        eternal=eternal,
        perishable=perishable,
        rental=rental,
        debuffed=debuffed,
        box=box,
        card_details=card_details,
    )


class TestVideoBatchExtraction(unittest.TestCase):
    """Tests single-pass batched video frame extraction."""

    def test_extract_video_frames_real_temp_video(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test_clip.avi"
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (64, 64))
            self.assertTrue(writer.isOpened())
            try:
                for i in range(60):
                    # Encode distinct uniform grayscale/color value per frame (step 4 avoids JPEG rounding ambiguity)
                    val = (i * 4) % 256
                    frame = np.full((64, 64, 3), val, dtype=np.uint8)
                    writer.write(frame)
            finally:
                writer.release()

            # Mixed unsorted indices spanning small sequential gaps (<=32) and large jumps (>32)
            req = [45, 2, 10, 2, -5, 100]
            extracted = extract_video_frames(video_path, req)

            self.assertEqual(set(extracted.keys()), {-5, 2, 10, 45, 100})
            self.assertAlmostEqual(float(extracted[2].mean()), 8.0, delta=3.0)
            self.assertAlmostEqual(float(extracted[10].mean()), 40.0, delta=3.0)
            self.assertAlmostEqual(float(extracted[45].mean()), 180.0, delta=3.0)
            # Negative clamped to 0, out-of-range clamped to 59
            self.assertAlmostEqual(float(extracted[-5].mean()), 0.0, delta=3.0)
            self.assertAlmostEqual(float(extracted[100].mean()), (59 * 4) % 256, delta=4.0)

    def test_extract_video_frames_empty_request(self):
        self.assertEqual(extract_video_frames("nonexistent.mp4", []), {})


class TestSpatialButtonMatching(unittest.TestCase):
    """Tests spatial button-box matching across shelf cards and tight-packed top bars."""

    def test_shop_shelf_bottom_center_matching(self):
        items = [
            _make_shop_item("j_joker", "Joker", [830, 380, 970, 560]),      # cx=900
            _make_shop_item("c_fool", "Tarot", [1100, 380, 1240, 560]),     # cx=1170
        ]
        btn_buy_card0 = [855, 575, 945, 625]  # cx=900
        btn_buy_card1 = [1125, 575, 1215, 625]  # cx=1170

        self.assertEqual(
            pp.match_button_to_item_index(btn_buy_card0, items, attach="bottom_center"),
            0,
        )
        self.assertEqual(
            pp.match_button_to_item_index(btn_buy_card1, items, attach="bottom_center"),
            1,
        )

    def test_tight_joker_and_consumable_center_right_matching(self):
        # Overlapping 140px Jokers spaced 110px apart + Consumable on right
        jokers = [
            _make_shop_item("j_blueprint", "Joker", [520, 40, 660, 220]),    # x2=660 -> anchor=674
            _make_shop_item("j_brainstorm", "Joker", [630, 40, 770, 220]),   # x2=770 -> anchor=784
        ]
        consumables = [
            _make_shop_item("c_hermit", "Tarot", [1390, 40, 1520, 220]),     # x2=1520 -> anchor=1533
        ]
        combined = jokers + consumables

        # Balatro attaches SELL button with align='cr', offset=-0.4 (~28px inside right edge, ~60px past it)
        sell_joker0_btn = [632, 95, 720, 155]   # cx=676 -> closer to Joker 0 anchor (674) than Joker 1 (784)
        use_cons0_btn = [1485, 140, 1575, 200]  # cx=1530

        self.assertEqual(
            pp.match_button_to_item_index(sell_joker0_btn, combined, attach="center_right"),
            0,
        )
        self.assertEqual(
            pp.match_button_to_item_index(use_cons0_btn, consumables, attach="center_right"),
            0,
        )


class TestPostProcessingOrchestration(unittest.TestCase):
    """End-to-end mocked tests for dual-CV optimization, spatial matching, sort tracking, and RunTrajectory."""

    def test_dual_cv_optimization_skips_celestial_standard_buffoon_packs(self):
        d = {
            "key": {
                "Skip Blind": 0,
                "Select Card": 1,
                "Open Pack": 2,
            },
            "frames": {
                "100": {"action": 0, "box": [100, 100, 180, 140]},  # Skip -> enters skip pack -> target 98
                "150": {"action": 1, "box": [830, 820, 910, 860]},  # Celestial pack item -> target 148 (SKIP hand CV)
                "200": {"action": 2, "box": [1120, 900, 1220, 950]},  # Open Pack in shop -> target 198
                "260": {"action": 1, "box": [830, 820, 910, 860]},  # Arcana pack item -> target 258 (RUN hand CV)
            },
        }
        dummy_frames = {f: np.zeros((1080, 1920, 3), dtype=np.uint8) for f in (98, 148, 198, 258)}

        observed_hand_frame_ids: list[int] = []

        def fake_batch_shops(frames, mode="shop", **kwargs):
            outs = []
            for _, f_idx, ts in frames:
                out = ShopRecognitionOutput(frame_index=f_idx, timestamp_seconds=ts, mode=mode)
                if mode == "pack" and f_idx == 148:
                    out.pack_items = [_make_shop_item("c_mercury", "Planet", [800, 600, 940, 800])]
                elif mode == "pack" and f_idx == 258:
                    out.pack_items = [_make_shop_item("c_hermit", "Tarot", [800, 600, 940, 800])]
                elif mode == "shop" and f_idx == 198:
                    out.boosters = [_make_shop_item("p_arcana_normal_1", "Booster", [1100, 730, 1240, 900])]
                outs.append(out)
            return outs

        def fake_batch_hands(frames, **kwargs):
            observed_hand_frame_ids.extend(f_idx for _, f_idx, _ in frames)
            return [
                HandRecognitionOutput(frame_index=f_idx, timestamp_seconds=ts, cards=[])
                for _, f_idx, ts in frames
            ]

        with (
            patch("classifiers.inference.batch_recognize_shops", side_effect=fake_batch_shops),
            patch("classifiers.inference.batch_recognize_hands", side_effect=fake_batch_hands),
        ):
            pp.build_run_trajectory(d, preloaded_frames=dummy_frames, fps=60.0)

        self.assertNotIn(148, observed_hand_frame_ids)
        self.assertIn(258, observed_hand_frame_ids)

    def test_end_to_end_trajectory_sort_hand_shop_and_sell(self):
        d = {
            "key": {
                "Play Game": 0,
                "Open Options Menu": 1,
                "Select Blind": 2,
                "Sort by Rank": 3,
                "Play Hand": 4,
                "Discard": 5,
                "End Round": 6,
                "Cash Out": 7,
                "Buy Card": 8,
                "Redeem Voucher": 9,
                "Open Pack": 10,
                "Select Card": 11,
                "Use Consumable": 12,
                "Sell Card": 13,
                "Next Round": 14,
            },
            "frames": {
                "5": 0,                                                     # filtered
                "10": 1,                                                    # filtered
                "100": {"action": 2, "box": [300, 200, 420, 250]},          # Select Blind -> 98
                "200": {"action": 3, "box": [850, 900, 930, 940]},          # Sort Rank -> pre=185, post=215
                "300": {"action": 5, "box": [1000, 900, 1100, 950]},        # Discard -> 300
                "400": {"action": 4, "box": [780, 900, 880, 950]},          # Play Hand -> 400
                "500": {"action": 6, "box": [500, 500, 600, 550]},          # End Round -> 498
                "600": {"action": 7, "box": [400, 300, 600, 370]},          # Cash Out -> 598
                "700": {"action": 8, "box": [860, 580, 950, 630]},          # Buy Card idx 0 (Blueprint) -> 700
                "800": {"action": 8, "box": [860, 580, 950, 630]},          # Buy Card idx 0 (Hermit) -> 800
                "900": {"action": 9, "box": [730, 910, 830, 960]},          # Redeem Voucher -> 898
                "1000": {"action": 10, "box": [1130, 910, 1230, 960]},      # Open Pack -> 998
                "1100": {"action": 11, "box": [1000, 810, 1080, 850]},      # Use In Pack (Empress on idx 1) -> 1098
                "1200": {"action": 12, "box": [1490, 120, 1570, 180]},      # Use Consumable (Hermit in SHOP) -> 1198
                "1300": {"action": 13, "box": [750, 80, 830, 140]},         # Sell Card (Blueprint) -> 1298
                "1400": {"action": 14, "box": [550, 420, 680, 480]},        # Next Round -> 1398
            },
        }

        needed_indices = {98, 185, 215, 300, 400, 498, 598, 700, 800, 898, 998, 1098, 1198, 1298, 1398}
        dummy_frames = {f: np.zeros((1080, 1920, 3), dtype=np.uint8) for f in needed_indices}

        bp_top_box = [530, 30, 670, 210]   # x2=670 -> anchor~684
        # Sell button centered at x=790 would match a second joker at [640..780], wait for 1 joker test [530..670] it uniquely matches index 0
        hermit_top_box = [1390, 35, 1520, 215]  # x2=1520 -> anchor~1533

        def fake_batch_shops(frames, mode="shop", **kwargs):
            outs = []
            for _, f_idx, ts in frames:
                out = ShopRecognitionOutput(frame_index=f_idx, timestamp_seconds=ts, mode=mode)
                if f_idx == 700:
                    out.shop_items = [
                        _make_shop_item("j_blueprint", "Joker", [840, 370, 970, 550]),
                        _make_shop_item("j_joker", "Joker", [1080, 370, 1210, 550]),
                    ]
                elif f_idx == 800:
                    out.joker_bar = [_make_shop_item("j_blueprint", "Joker", bp_top_box)]
                    out.shop_items = [
                        _make_shop_item("c_hermit", "Tarot", [840, 370, 970, 550]),
                    ]
                elif f_idx == 898:
                    out.joker_bar = [_make_shop_item("j_blueprint", "Joker", bp_top_box)]
                    out.consumeables = [_make_shop_item("c_hermit", "Tarot", hermit_top_box)]
                    out.vouchers = [_make_shop_item("v_overstock", "Voucher", [710, 720, 850, 900])]
                elif f_idx == 998:
                    out.joker_bar = [_make_shop_item("j_blueprint", "Joker", bp_top_box)]
                    out.consumeables = [_make_shop_item("c_hermit", "Tarot", hermit_top_box)]
                    out.boosters = [_make_shop_item("p_arcana_normal_1", "Booster", [1110, 730, 1250, 900])]
                elif f_idx == 1098:
                    out.joker_bar = [_make_shop_item("j_blueprint", "Joker", bp_top_box)]
                    out.consumeables = [_make_shop_item("c_hermit", "Tarot", hermit_top_box)]
                    out.pack_items = [
                        _make_shop_item("c_fool", "Tarot", [720, 600, 860, 790]),
                        _make_shop_item("c_empress", "Tarot", [970, 600, 1110, 790]),
                    ]
                elif f_idx in (1198, 1298):
                    out.joker_bar = [_make_shop_item("j_blueprint", "Joker", bp_top_box)]
                    if f_idx == 1198:
                        out.consumeables = [_make_shop_item("c_hermit", "Tarot", hermit_top_box)]
                elif f_idx == 1398:
                    out.joker_bar = []
                outs.append(out)
            return outs

        def fake_batch_hands(frames, **kwargs):
            outs = []
            for _, f_idx, ts in frames:
                if f_idx == 185:
                    # Pre-sort: unsorted with a selected face-down card at index 0
                    cards = [
                        _make_card_res("none", "none", selected=True, face_down=True),
                        _make_card_res("5", "Clubs"),
                        _make_card_res("King", "Hearts"),
                    ]
                elif f_idx == 215:
                    # Post-sort: King first, selected face-down card settled after 5? Or between King and 5 -> narrows bounds <= King, >= 5!
                    cards = [
                        _make_card_res("King", "Hearts"),
                        _make_card_res("none", "none", selected=True, face_down=True),
                        _make_card_res("5", "Clubs"),
                    ]
                elif f_idx == 300:
                    # Dragged 5 Clubs to front + selected it for Discard
                    cards = [
                        _make_card_res("5", "Clubs", selected=True),
                        _make_card_res("King", "Hearts"),
                        _make_card_res("none", "none", face_down=True),
                    ]
                elif f_idx == 400:
                    # Drew Ace of Spades after Discard; play King + Ace
                    cards = [
                        _make_card_res("Ace", "Spades", selected=True),
                        _make_card_res("King", "Hearts", selected=True),
                        _make_card_res("none", "none", face_down=True),
                    ]
                elif f_idx == 1098:
                    # Arcana pack hand: Empress targets 2 Clubs + 3 Hearts
                    cards = [
                        _make_card_res("2", "Clubs", selected=True),
                        _make_card_res("3", "Hearts", selected=True),
                    ]
                else:
                    cards = []
                outs.append(HandRecognitionOutput(frame_index=f_idx, timestamp_seconds=ts, cards=cards))
            return outs

        with (
            patch("classifiers.inference.batch_recognize_shops", side_effect=fake_batch_shops),
            patch("classifiers.inference.batch_recognize_hands", side_effect=fake_batch_hands),
        ):
            res = pp.process_actions(
                d,
                preloaded_frames=dummy_frames,
                seed="GLUESEED1",
                fps=60.0,
            )

        state: gs.GameState = res["state"]
        traj: gs.RunTrajectory = res["trajectory"]

        # Step 0 initial + 14 gameplay actions (Play Game and Open Options filtered out)
        self.assertEqual(len(traj.deltas), 15)

        # Verify step 2 (Sort by Rank): pre_sort selected_slot_uid recorded & face-down bounds narrowed between 5 and King
        sort_delta = traj.deltas[2]
        self.assertEqual(sort_delta.action.type, bm.ActionType.SORT_RANK)
        pre_uid = sort_delta.action.metadata.get("selected_slot_uid")
        self.assertIsNotNone(pre_uid)
        post_sort_state = traj.get_state(2)
        self.assertEqual(post_sort_state.hand.cards[1].slot_uid, pre_uid)
        self.assertTrue(post_sort_state.hand.cards[1].selected)
        fd_bounds = post_sort_state.hand.cards[1].bounds
        self.assertIsNotNone(fd_bounds)
        self.assertTrue(all(c.rank != bm.Rank.ACE for c in fd_bounds.candidate_cards))
        self.assertTrue(all(bm.Rank.FOUR not in {c.rank} for c in fd_bounds.candidate_cards))

        # Verify step 3 (Discard) recorded hand_order permutation [2, 0, 1] & selected 5 of Clubs
        discard_delta = traj.deltas[3]
        self.assertEqual(discard_delta.action.type, bm.ActionType.DISCARD)
        self.assertEqual(discard_delta.action.hand_order, [2, 0, 1])
        self.assertEqual(len(discard_delta.action.selected_cards), 1)
        self.assertEqual(discard_delta.action.selected_cards[0].rank, bm.Rank.FIVE)

        # Verify step 5 (End Round -> GameState BLIND + round_won=True) & step 6 (Cash Out -> SHOP + payout)
        end_round_state = traj.get_state(5)
        self.assertEqual(end_round_state.location, bm.Location.BLIND)
        self.assertTrue(end_round_state.round_won)

        cash_out_state = traj.get_state(6)
        self.assertEqual(cash_out_state.location, bm.Location.SHOP)
        self.assertFalse(cash_out_state.round_won)
        self.assertGreater(cash_out_state.money, 4)

        # Verify shop & pack steps matched targets spatially
        self.assertEqual(traj.deltas[7].action.target_idx, 0)   # Blueprint
        self.assertEqual(traj.deltas[7].action.joker_order, [])
        self.assertEqual(traj.deltas[8].action.target_idx, 0)   # Hermit
        self.assertEqual(traj.deltas[8].action.joker_order, [1])
        self.assertEqual(traj.deltas[9].action.target_idx, 0)   # Overstock
        self.assertIn(bm.VoucherName.OVERSTOCK, traj.get_state(9).vouchers_redeemed)
        self.assertEqual(traj.deltas[10].action.target_idx, 0)  # Open Arcana Pack
        self.assertEqual(traj.deltas[11].action.target_idx, 1)  # Select Card (Empress)
        # Empress enhanced 2 of Clubs and 3 of Hearts to Mult Cards in full_deck
        pack_step_state = traj.get_state(11)
        mult_cards = [c for c in pack_step_state.full_deck if c.enhancement == bm.Enhancement.MULT]
        self.assertEqual(len(mult_cards), 2)
        self.assertEqual(pack_step_state.location, bm.Location.SHOP)  # Exited back to shop

        # Verify Sell Card sold Blueprint (index 0) and Next Round moved to Ante 1 Big Blind SELECT
        self.assertEqual(traj.deltas[13].action.type, bm.ActionType.SELL_CARD)
        self.assertEqual(traj.deltas[13].action.target_idx, 0)
        self.assertEqual(len(state.inventory.jokers), 0)
        self.assertEqual(state.location, bm.Location.SELECT)
        self.assertEqual(state.current_blind, bm.BlindName.BIG)
        self.assertEqual(state.last_tarot_planet, bm.TarotName.HERMIT)

    def test_remove_duplicates_preserves_distinct_spaced_clicks_and_merges_holds(self):
        raw = {
            "100": {"action": 1, "box": [500, 500, 600, 550]},
            "103": {"action": 1, "box": [502, 500, 602, 550]},  # same hold -> keeps 103
            "160": {"action": 1, "box": [500, 500, 600, 550]},  # >15 frames later -> second click!
            "163": {"action": 1, "box": [500, 500, 600, 550]},  # same hold -> keeps 163
        }
        cleaned = pp.remove_duplicates(raw)
        self.assertEqual(list(cleaned.keys()), ["103", "163"])


if __name__ == "__main__":
    unittest.main()
