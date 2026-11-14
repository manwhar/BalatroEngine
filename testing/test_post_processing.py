"""Unit and mocked end-to-end integration tests for `post_processing.py` and `video_utils.extract_video_frames`."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import cv2
import numpy as np

import game.balatro_models as bm
import game.gamestate as gs
import post_processing as pp
from classifiers.common.video_utils import extract_video_frames
from classifiers.inference import HandRecognitionOutput, ShopRecognitionOutput


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
            self.assertAlmostEqual(
                float(extracted[100].mean()), (59 * 4) % 256, delta=4.0
            )

    def test_extract_video_frames_empty_request(self):
        self.assertEqual(extract_video_frames("nonexistent.mp4", []), {})


class TestSpatialButtonMatching(unittest.TestCase):
    """Tests spatial button-box matching across shelf cards and tight-packed top bars."""

    def test_shop_shelf_bottom_center_matching(self):
        items = [
            _make_shop_item("j_joker", "Joker", [830, 380, 970, 560]),  # cx=900
            _make_shop_item("c_fool", "Tarot", [1100, 380, 1240, 560]),  # cx=1170
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
            _make_shop_item(
                "j_blueprint", "Joker", [520, 40, 660, 220]
            ),  # x2=660 -> anchor=674
            _make_shop_item(
                "j_brainstorm", "Joker", [630, 40, 770, 220]
            ),  # x2=770 -> anchor=784
        ]
        consumables = [
            _make_shop_item(
                "c_hermit", "Tarot", [1390, 40, 1520, 220]
            ),  # x2=1520 -> anchor=1533
        ]
        combined = jokers + consumables

        # Balatro attaches SELL button with align='cr', offset=-0.4 (~28px inside right edge, ~60px past it)
        sell_joker0_btn = [
            632,
            95,
            720,
            155,
        ]  # cx=676 -> closer to Joker 0 anchor (674) than Joker 1 (784)
        use_cons0_btn = [1485, 140, 1575, 200]  # cx=1530

        self.assertEqual(
            pp.match_button_to_item_index(
                sell_joker0_btn, combined, attach="center_right"
            ),
            0,
        )
        self.assertEqual(
            pp.match_button_to_item_index(
                use_cons0_btn, consumables, attach="center_right"
            ),
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
                "100": {
                    "action": 0,
                    "box": [100, 100, 180, 140],
                },  # Skip -> enters skip pack -> target 98
                "150": {
                    "action": 1,
                    "box": [830, 820, 910, 860],
                },  # Celestial pack item -> target 148 (SKIP hand CV)
                "200": {
                    "action": 2,
                    "box": [1120, 900, 1220, 950],
                },  # Open Pack in shop -> target 198
                "260": {
                    "action": 1,
                    "box": [830, 820, 910, 860],
                },  # Arcana pack item -> target 258 (RUN hand CV)
            },
        }
        dummy_frames = {
            f: np.zeros((1080, 1920, 3), dtype=np.uint8) for f in (98, 148, 198, 258)
        }

        observed_hand_frame_ids: list[int] = []

        def fake_batch_shops(frames, mode="shop", **kwargs):
            outs = []
            for _, f_idx, ts in frames:
                out = ShopRecognitionOutput(
                    frame_index=f_idx, timestamp_seconds=ts, mode=mode
                )
                if mode == "pack" and f_idx == 148:
                    out.pack_items = [
                        _make_shop_item("c_mercury", "Planet", [800, 600, 940, 800])
                    ]
                elif mode == "pack" and f_idx == 258:
                    out.pack_items = [
                        _make_shop_item("c_hermit", "Tarot", [800, 600, 940, 800])
                    ]
                elif mode == "shop" and f_idx == 198:
                    out.boosters = [
                        _make_shop_item(
                            "p_arcana_normal_1", "Booster", [1100, 730, 1240, 900]
                        )
                    ]
                outs.append(out)
            return outs

        def fake_batch_hands(frames, **kwargs):
            observed_hand_frame_ids.extend(f_idx for _, f_idx, _ in frames)
            return [
                HandRecognitionOutput(frame_index=f_idx, timestamp_seconds=ts, cards=[])
                for _, f_idx, ts in frames
            ]

        with (
            patch(
                "classifiers.inference.batch_recognize_shops",
                side_effect=fake_batch_shops,
            ),
            patch(
                "classifiers.inference.batch_recognize_hands",
                side_effect=fake_batch_hands,
            ),
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
                "5": 0,  # filtered
                "10": 1,  # filtered
                "100": {"action": 2, "box": [300, 200, 420, 250]},  # Select Blind -> 98
                "200": {
                    "action": 3,
                    "box": [850, 900, 930, 940],
                },  # Sort Rank -> pre=185, post=215
                "300": {"action": 5, "box": [1000, 900, 1100, 950]},  # Discard -> 300
                "400": {"action": 4, "box": [780, 900, 880, 950]},  # Play Hand -> 400
                "500": {"action": 6, "box": [500, 500, 600, 550]},  # End Round -> 498
                "600": {"action": 7, "box": [400, 300, 600, 370]},  # Cash Out -> 598
                "700": {
                    "action": 8,
                    "box": [860, 580, 950, 630],
                },  # Buy Card idx 0 (Blueprint) -> 700
                "800": {
                    "action": 8,
                    "box": [860, 580, 950, 630],
                },  # Buy Card idx 0 (Hermit) -> 800
                "900": {
                    "action": 9,
                    "box": [730, 910, 830, 960],
                },  # Redeem Voucher -> 898
                "1000": {
                    "action": 10,
                    "box": [1130, 910, 1230, 960],
                },  # Open Pack -> 998
                "1100": {
                    "action": 11,
                    "box": [1000, 810, 1080, 850],
                },  # Use In Pack (Empress on idx 1) -> 1098
                "1200": {
                    "action": 12,
                    "box": [1490, 120, 1570, 180],
                },  # Use Consumable (Hermit in SHOP) -> 1198
                "1300": {
                    "action": 13,
                    "box": [750, 80, 830, 140],
                },  # Sell Card (Blueprint) -> 1298
                "1400": {
                    "action": 14,
                    "box": [550, 420, 680, 480],
                },  # Next Round -> 1398
            },
        }

        needed_indices = {
            98,
            185,
            215,
            300,
            400,
            498,
            598,
            700,
            800,
            898,
            998,
            1098,
            1198,
            1298,
            1398,
        }
        dummy_frames = {
            f: np.zeros((1080, 1920, 3), dtype=np.uint8) for f in needed_indices
        }

        bp_top_box = [530, 30, 670, 210]  # x2=670 -> anchor~684
        # Sell button centered at x=790 would match a second joker at [640..780], wait for 1 joker test [530..670] it uniquely matches index 0
        hermit_top_box = [1390, 35, 1520, 215]  # x2=1520 -> anchor~1533

        def fake_batch_shops(frames, mode="shop", **kwargs):
            outs = []
            for _, f_idx, ts in frames:
                out = ShopRecognitionOutput(
                    frame_index=f_idx, timestamp_seconds=ts, mode=mode
                )
                if f_idx == 700:
                    out.shop_items = [
                        _make_shop_item("j_blueprint", "Joker", [840, 370, 970, 550]),
                        _make_shop_item("j_joker", "Joker", [1080, 370, 1210, 550]),
                    ]
                elif f_idx == 800:
                    out.joker_bar = [
                        _make_shop_item("j_blueprint", "Joker", bp_top_box)
                    ]
                    out.shop_items = [
                        _make_shop_item("c_hermit", "Tarot", [840, 370, 970, 550]),
                    ]
                elif f_idx == 898:
                    out.joker_bar = [
                        _make_shop_item("j_blueprint", "Joker", bp_top_box)
                    ]
                    out.consumeables = [
                        _make_shop_item("c_hermit", "Tarot", hermit_top_box)
                    ]
                    out.vouchers = [
                        _make_shop_item("v_overstock", "Voucher", [710, 720, 850, 900])
                    ]
                elif f_idx == 998:
                    out.joker_bar = [
                        _make_shop_item("j_blueprint", "Joker", bp_top_box)
                    ]
                    out.consumeables = [
                        _make_shop_item("c_hermit", "Tarot", hermit_top_box)
                    ]
                    out.boosters = [
                        _make_shop_item(
                            "p_arcana_normal_1", "Booster", [1110, 730, 1250, 900]
                        )
                    ]
                elif f_idx == 1098:
                    out.joker_bar = [
                        _make_shop_item("j_blueprint", "Joker", bp_top_box)
                    ]
                    out.consumeables = [
                        _make_shop_item("c_hermit", "Tarot", hermit_top_box)
                    ]
                    out.pack_items = [
                        _make_shop_item("c_fool", "Tarot", [720, 600, 860, 790]),
                        _make_shop_item("c_empress", "Tarot", [970, 600, 1110, 790]),
                    ]
                elif f_idx in (1198, 1298):
                    out.joker_bar = [
                        _make_shop_item("j_blueprint", "Joker", bp_top_box)
                    ]
                    if f_idx == 1198:
                        out.consumeables = [
                            _make_shop_item("c_hermit", "Tarot", hermit_top_box)
                        ]
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
                outs.append(
                    HandRecognitionOutput(
                        frame_index=f_idx, timestamp_seconds=ts, cards=cards
                    )
                )
            return outs

        with (
            patch(
                "classifiers.inference.batch_recognize_shops",
                side_effect=fake_batch_shops,
            ),
            patch(
                "classifiers.inference.batch_recognize_hands",
                side_effect=fake_batch_hands,
            ),
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
        self.assertTrue(
            all(bm.Rank.FOUR not in {c.rank} for c in fd_bounds.candidate_cards)
        )

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
        self.assertEqual(traj.deltas[7].action.target_idx, 0)  # Blueprint
        self.assertEqual(traj.deltas[7].action.joker_order, [])
        self.assertEqual(traj.deltas[8].action.target_idx, 0)  # Hermit
        self.assertEqual(traj.deltas[8].action.joker_order, [1])
        self.assertEqual(traj.deltas[9].action.target_idx, 0)  # Overstock
        self.assertIn(bm.VoucherName.OVERSTOCK, traj.get_state(9).vouchers_redeemed)
        self.assertEqual(traj.deltas[10].action.target_idx, 0)  # Open Arcana Pack
        self.assertEqual(traj.deltas[11].action.target_idx, 1)  # Select Card (Empress)
        # Empress enhanced 2 of Clubs and 3 of Hearts to Mult Cards in full_deck
        pack_step_state = traj.get_state(11)
        mult_cards = [
            c for c in pack_step_state.full_deck if c.enhancement == bm.Enhancement.MULT
        ]
        self.assertEqual(len(mult_cards), 2)
        self.assertEqual(
            pack_step_state.location, bm.Location.SHOP
        )  # Exited back to shop

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
            "160": {
                "action": 1,
                "box": [500, 500, 600, 550],
            },  # >15 frames later -> second click!
            "163": {"action": 1, "box": [500, 500, 600, 550]},  # same hold -> keeps 163
        }
        cleaned = pp.remove_duplicates(raw)
        self.assertEqual(list(cleaned.keys()), ["103", "163"])

    def test_automatic_end_round_injection_and_joker_order_at_end_round(self):
        from classifiers.end_round_classifier.pipeline import EndRoundResult

        d = {
            "key": {
                "Select Blind": 0,
                "Play Hand": 1,
                "Cash Out": 2,
            },
            "frames": {
                "100": {"action": 0, "box": [300, 200, 420, 250]},  # target 98
                "400": {"action": 1, "box": [780, 900, 880, 950]},  # target 400
                "600": {"action": 2, "box": [400, 300, 600, 370]},  # target 598
            },
        }
        # Sparse preloaded frames (no video): fake_end_rounds is patched to supply 500
        dummy_frames = {
            f: np.zeros((1080, 1920, 3), dtype=np.uint8) for f in (98, 400, 498, 598)
        }

        def fake_end_rounds(segments, **kwargs):
            self.assertEqual(list(segments), [(400, 600)])
            return [
                EndRoundResult(
                    end_round_frame=500,
                    end_round_timestamp=500 / 60.0,
                    confidence=0.97,
                    model_step_index=25,
                    model_step_float=25.0,
                    viterbi_transition_step=25,
                    boundary_peak_step=25,
                    src_fps=60.0,
                    num_sampled_frames=50,
                )
            ]

        def fake_batch_shops(frames, mode="shop", **kwargs):
            outs = []
            for _, f_idx, ts in frames:
                out = ShopRecognitionOutput(
                    frame_index=f_idx, timestamp_seconds=ts, mode=mode
                )
                if f_idx in (98, 400):
                    # Mime on left, Blueprint on right (Blueprint copies nothing)
                    out.joker_bar = [
                        _make_shop_item("j_mime", "Joker", [520, 40, 660, 220]),
                        _make_shop_item("j_blueprint", "Joker", [680, 40, 820, 220]),
                    ]
                elif f_idx in (498, 598):
                    # Reordered before END_ROUND so Blueprint copies Mime (+$9 held Gold Card)
                    out.joker_bar = [
                        _make_shop_item("j_blueprint", "Joker", [520, 40, 660, 220]),
                        _make_shop_item("j_mime", "Joker", [680, 40, 820, 220]),
                    ]
                outs.append(out)
            return outs

        def fake_batch_hands(frames, **kwargs):
            outs = []
            for _, f_idx, ts in frames:
                cards = []
                if f_idx == 400:
                    cards = [
                        _make_card_res("Ace", "Spades", selected=True),
                        _make_card_res(
                            "2", "Clubs", selected=False, enhancement="Gold"
                        ),
                    ]
                outs.append(
                    HandRecognitionOutput(
                        frame_index=f_idx, timestamp_seconds=ts, cards=cards
                    )
                )
            return outs

        with (
            patch(
                "classifiers.inference.batch_recognize_end_rounds",
                side_effect=fake_end_rounds,
            ),
            patch(
                "classifiers.inference.batch_recognize_shops",
                side_effect=fake_batch_shops,
            ),
            patch(
                "classifiers.inference.batch_recognize_hands",
                side_effect=fake_batch_hands,
            ),
        ):
            res = pp.process_actions(
                d,
                preloaded_frames=dummy_frames,
                seed="ENDROUND1",
                fps=60.0,
            )

        self.assertIn(498, res["joker_cv_queue"])
        self.assertEqual(res["joker_cv_queue"][498], bm.ActionType.END_ROUND)

        traj: gs.RunTrajectory = res["trajectory"]
        self.assertEqual(
            len(traj.deltas), 5
        )  # init, SelectBlind, PlayHand, EndRound, CashOut

        er_delta = traj.deltas[3]
        self.assertEqual(er_delta.action.type, bm.ActionType.END_ROUND)
        self.assertEqual(er_delta.action.frame_idx, 500)
        self.assertEqual(er_delta.action.metadata.get("source"), "end_round_classifier")

        er_state = traj.get_state(3)
        self.assertEqual(er_state.location, bm.Location.BLIND)
        self.assertTrue(er_state.round_won)
        self.assertEqual(
            [j.name for j in er_state.inventory.jokers],
            [bm.JokerName.BLUEPRINT, bm.JokerName.MIME],
        )
        # $4 start + $9 from 3x triggered Gold card (card + Mime + Blueprint)
        self.assertEqual(er_state.money, 13)

        cash_out_state = traj.get_state(4)
        self.assertEqual(cash_out_state.location, bm.Location.SHOP)
        self.assertFalse(cash_out_state.round_won)
        # No double-trigger of end_round Gold card; +$3 Small Blind + $2 interest + $3 hands_left
        self.assertEqual(cash_out_state.money, 21)

    def test_end_round_prod_weights_and_video_segment_batch(self):
        import classifiers.inference as inf

        prod_weights = (
            Path(__file__).resolve().parent.parent
            / "classifiers"
            / "prod_weights"
            / "end_round_classifier.pt"
        )
        self.assertTrue(
            prod_weights.exists(), f"Missing production weights at {prod_weights}"
        )

        inf._END_ROUND_PIPELINE = None
        pipe = inf.get_end_round_pipeline()
        self.assertEqual(pipe.weights_path.resolve(), prod_weights.resolve())

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "end_round_seg.avi"
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (320, 180))
            self.assertTrue(writer.isOpened())
            try:
                for i in range(60):
                    # Draw simple varying frames across the segment
                    frame = np.full((180, 320, 3), 25 if i < 25 else 80, dtype=np.uint8)
                    writer.write(frame)
            finally:
                writer.release()

            results = inf.batch_recognize_end_rounds([(10, 40)], video_path=video_path)

        self.assertEqual(len(results), 1)
        self.assertGreaterEqual(results[0].end_round_frame, 10)
        self.assertLess(results[0].end_round_frame, 40)
        self.assertGreaterEqual(results[0].confidence, 0.0)
        self.assertLessEqual(results[0].confidence, 1.0)

    def test_hud_ocr_prod_weights_and_batch_recognize_hud(self):
        import random
        import classifiers.inference as inf
        from classifiers.common.video_utils import scale_roi_pixels
        from classifiers.hud_ocr.config import HUD_ROIS, HUD_ROI_FIELDS
        from classifiers.hud_ocr.synth import BalatroHudSynthesizer

        prod_weights = (
            Path(__file__).resolve().parent.parent
            / "classifiers"
            / "prod_weights"
            / "hud_ocr.pt"
        )
        self.assertTrue(
            prod_weights.exists(), f"Missing production weights at {prod_weights}"
        )

        inf._HUD_OCR_PIPELINE = None
        pipe = inf.get_hud_ocr_pipeline(device="cpu")
        self.assertEqual(pipe.weights_path.resolve(), prod_weights.resolve())
        self.assertAlmostEqual(pipe.conf_threshold, 0.90)

        synth = BalatroHudSynthesizer()
        rng = random.Random(77)
        expected = {
            "hud_round_score": "1,250",
            "hud_money": "37",
            "hud_hands": "3",
            "hud_discards": "2",
            "hud_ante": "4",
            "hud_round": "10",
            "hud_reroll_cost": "7",
            "hud_blind_target": "6,400",
        }
        expected_parsed = {
            "hud_round_score": 1250,
            "hud_money": 37,
            "hud_hands": 3,
            "hud_discards": 2,
            "hud_ante": 4,
            "hud_round": 10,
            "hud_reroll_cost": 7,
            "hud_blind_target": 6400,
        }

        frame_bgr = np.zeros((1080, 1920, 3), dtype=np.uint8)
        for fname in HUD_ROI_FIELDS:
            crop_rgb, _, _ = synth.render_sample(
                rng,
                field_name=fname,
                text_override=expected[fname],
                augment=False,
            )
            rx, ry, rw, rh = scale_roi_pixels(HUD_ROIS[fname], 1920, 1080)
            bg_rgb = crop_rgb[0, 0]
            frame_bgr[ry : ry + rh, rx : rx + rw] = bg_rgb[::-1]
            # Trim uniform outer margins before uniform scaling so alignment fits narrow ROIs
            diff = np.abs(crop_rgb.astype(np.int16) - bg_rgb.astype(np.int16)).sum(
                axis=2
            )
            ys, xs = np.where(diff > 8)
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            fg_crop = crop_rgb[:, max(0, x0 - 6) : min(crop_rgb.shape[1], x1 + 6)]
            scale = min(rw / max(1, fg_crop.shape[1]), rh / max(1, fg_crop.shape[0]))
            nw = max(1, int(round(fg_crop.shape[1] * scale)))
            nh = max(1, int(round(fg_crop.shape[0] * scale)))
            scaled = cv2.resize(cv2.cvtColor(fg_crop, cv2.COLOR_RGB2BGR), (nw, nh))
            ox = rx + (rw - nw) // 2
            oy = ry + (rh - nh) // 2
            frame_bgr[oy : oy + nh, ox : ox + nw] = scaled

        outs = inf.batch_recognize_hud(
            [(frame_bgr, 42, 0.7)], conf_threshold=0.90, device="cpu"
        )
        self.assertEqual(len(outs), 1)
        out = outs[0]
        self.assertEqual(out.frame_index, 42)
        for fname, exp_val in expected_parsed.items():
            self.assertEqual(
                out.values[fname], exp_val, f"Mismatch on {fname}: {out.results[fname]}"
            )

    def test_hud_ocr_post_processing_sync_and_none_fallback(self):
        from classifiers.hud_ocr.pipeline import HudOcrResult
        from classifiers.inference import HudRecognitionOutput

        d = {
            "key": {
                "Select Blind": 0,
                "Play Hand": 1,
                "End Round": 2,
                "Cash Out": 3,
                "Reroll Shop": 4,
            },
            "frames": {
                "100": {
                    "action": 0,
                    "box": [300, 200, 420, 250],
                },  # Select Blind -> target 98
                "200": {
                    "action": 1,
                    "box": [780, 900, 880, 950],
                },  # Play Hand -> target 200
                "300": {
                    "action": 2,
                    "box": [500, 500, 600, 550],
                },  # End Round -> target 298
                "400": {
                    "action": 3,
                    "box": [400, 300, 600, 370],
                },  # Cash Out -> target 398
                "500": {
                    "action": 4,
                    "box": [650, 575, 740, 640],
                },  # Reroll Shop -> target 498
            },
        }
        needed = (98, 200, 298, 398, 498)
        dummy_frames = {f: np.zeros((1080, 1920, 3), dtype=np.uint8) for f in needed}

        observed_hud_fields: dict[int, list[str]] = {}

        def fake_batch_shops(frames, mode="shop", **kwargs):
            return [
                ShopRecognitionOutput(
                    frame_index=f_idx, timestamp_seconds=ts, mode=mode
                )
                for _, f_idx, ts in frames
            ]

        def fake_batch_hands(frames, **kwargs):
            outs = []
            for _, f_idx, ts in frames:
                cards = (
                    [_make_card_res("Ace", "Spades", selected=True)]
                    if f_idx == 200
                    else []
                )
                outs.append(
                    HandRecognitionOutput(
                        frame_index=f_idx, timestamp_seconds=ts, cards=cards
                    )
                )
            return outs

        def fake_batch_hud(frames, fields_by_frame=None, conf_threshold=0.90, **kwargs):
            outs = []
            for _, f_idx, ts in frames:
                flds = list((fields_by_frame or {}).get(f_idx, ()))
                observed_hud_fields[f_idx] = flds
                vals: dict[str, int | float | None] = {f: None for f in flds}
                if f_idx == 200:
                    # Play Hand observes money=12 (e.g. Temperance/Hermit bonus), score=450 (>=300 target -> deferred until END_ROUND!)
                    vals.update(
                        {
                            "hud_money": 12,
                            "hud_hands": 3,
                            "hud_discards": 2,
                            "hud_ante": 2,
                            "hud_round": 4,
                            "hud_blind_target": 800,
                            "hud_round_score": 900,
                        }
                    )
                elif f_idx == 298:
                    # END_ROUND observes final chips=950 and money=15
                    vals.update(
                        {
                            "hud_money": 15,
                            "hud_round_score": 950,
                            "hud_blind_target": 800,
                        }
                    )
                elif f_idx == 398:
                    # CASH_OUT observes post-end_round / pre-cash-out money=18 (before +payout)
                    vals.update({"hud_money": 18})
                elif f_idx == 498:
                    # REROLL_SHOP: hud_reroll_cost is None (e.g. <90% or failed green check) -> engine falls back to state.reroll_cost (5)!
                    vals.update({"hud_money": 26, "hud_reroll_cost": None})
                outs.append(
                    HudRecognitionOutput(
                        frame_index=f_idx, timestamp_seconds=ts, values=vals
                    )
                )
            return outs

        with (
            patch(
                "classifiers.inference.batch_recognize_shops",
                side_effect=fake_batch_shops,
            ),
            patch(
                "classifiers.inference.batch_recognize_hands",
                side_effect=fake_batch_hands,
            ),
            patch(
                "classifiers.inference.batch_recognize_hud", side_effect=fake_batch_hud
            ),
        ):
            res = pp.process_actions(d, preloaded_frames=dummy_frames, fps=60.0)

        self.assertIn("hud_ocr_queue", res)
        self.assertEqual(set(res["hud_ocr_queue"].keys()), set(needed))
        # Location-aware HUD field subsets
        self.assertIn("hud_round_score", observed_hud_fields[200])
        self.assertIn("hud_blind_target", observed_hud_fields[200])
        self.assertNotIn("hud_reroll_cost", observed_hud_fields[200])
        self.assertIn("hud_reroll_cost", observed_hud_fields[498])
        self.assertNotIn("hud_round_score", observed_hud_fields[498])

        traj: gs.RunTrajectory = res["trajectory"]
        # Step 2 (PLAY_HAND): round_won stays False because END_ROUND follows (score deferred to after PLAY_HAND step)
        play_state = traj.get_state(2)
        self.assertFalse(play_state.round_won)
        self.assertEqual(play_state.current_chips, 900)
        self.assertEqual(play_state.target_chips, 800)
        self.assertEqual(play_state.money, 12)
        self.assertEqual(
            play_state.hands_left, 2
        )  # 3 observed before PLAY_HAND -> decremented to 2
        self.assertEqual(play_state.discards_left, 2)
        self.assertEqual(traj.deltas[2].action.metadata["hud_ocr"]["hud_money"], 12)

        # Step 3 (END_ROUND): round_won becomes True, round advances 4 -> 5
        er_state = traj.get_state(3)
        self.assertTrue(er_state.round_won)
        self.assertEqual(er_state.current_chips, 950)
        self.assertEqual(er_state.round, 5)

        # Step 4 (CASH_OUT): synced money=18 then added base payout (3 + 2 hands + 3 interest = +8 -> 26), reroll_cost=5
        co_state = traj.get_state(4)
        self.assertEqual(co_state.location, bm.Location.SHOP)
        self.assertEqual(co_state.reroll_cost, 5)
        self.assertEqual(co_state.money, 26)

        # Step 5 (REROLL_SHOP): hud_reroll_cost was None -> fell back to engine reroll_cost=5 (26 - 5 = 21, reroll_cost -> 6)
        reroll_state = traj.get_state(5)
        self.assertEqual(reroll_state.money, 21)
        self.assertEqual(reroll_state.reroll_cost, 6)

    def test_sync_hud_into_state_cash_out_and_free_reroll_edges(self):
        from classifiers.inference import HudRecognitionOutput

        state = gs.GameState(apply_deck_stake_on_init=True)
        state.location = bm.Location.BLIND
        state.round_won = False
        state.money = 10
        state.hand.cards = [
            bm.HandCard(
                card=bm.PlayingCard(
                    rank=bm.Rank.TWO,
                    suit=bm.Suit.CLUBS,
                    enhancement=bm.Enhancement.GOLD,
                )
            )
        ]
        sa_cash = pp.ScheduledAction(
            idx=0,
            frame_id=300,
            action_type=bm.ActionType.CASH_OUT,
            target_frame=298,
            ambient_location=bm.Location.CASH_OUT,
        )
        act_cash = bm.Action(type=bm.ActionType.CASH_OUT, frame_idx=300)
        hud_cash = HudRecognitionOutput(
            frame_index=298, timestamp_seconds=4.9, values={"hud_money": 20}
        )
        pp._sync_hud_into_state(state, hud_cash, sa_cash, act_cash)
        self.assertTrue(state.round_won)
        self.assertEqual(state.money, 20)

        # Free reroll preserves underlying reroll_cost when HUD reads $0, clears free_rerolls when > 0
        state.location = bm.Location.SHOP
        state.reroll_cost = 5
        state.free_rerolls = 1
        sa_shop = pp.ScheduledAction(
            idx=1,
            frame_id=400,
            action_type=bm.ActionType.REROLL_SHOP,
            target_frame=398,
            ambient_location=bm.Location.SHOP,
        )
        act_shop = bm.Action(type=bm.ActionType.REROLL_SHOP, frame_idx=400)
        pp._sync_hud_into_state(
            state,
            HudRecognitionOutput(
                frame_index=398, timestamp_seconds=6.6, values={"hud_reroll_cost": 0}
            ),
            sa_shop,
            act_shop,
        )
        self.assertEqual(state.free_rerolls, 1)
        self.assertEqual(state.reroll_cost, 5)

        pp._sync_hud_into_state(
            state,
            HudRecognitionOutput(
                frame_index=398, timestamp_seconds=6.6, values={"hud_reroll_cost": 6}
            ),
            sa_shop,
            act_shop,
        )
        self.assertEqual(state.free_rerolls, 0)
        self.assertEqual(state.reroll_cost, 6)


if __name__ == "__main__":
    unittest.main()
