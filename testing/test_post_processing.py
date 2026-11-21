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
    confidences: dict[str, float] | None = None,
    top_probs: dict[str, dict[str, float]] | None = None,
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
        confidences=confidences or {},
        top_probs=top_probs or {},
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
                "Skip Blind": 15,
            },
            "frames": {
                "5": 0,  # filtered
                "10": 1,  # filtered
                "50": {"action": 15, "box": [200, 200, 300, 250]},  # Skip Small -> 48
                "70": {"action": 15, "box": [200, 200, 300, 250]},  # Skip Big -> 68
                "100": {"action": 2, "box": [300, 200, 420, 250]},  # Select Boss Blind -> 98
                "200": {
                    "action": 3,
                    "box": [850, 900, 930, 940],
                },  # Sort Rank -> pre=185, post=250
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
            48,
            68,
            98,
            185,
            250,
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
                elif f_idx == 250:
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
            patch.object(
                pp,
                "FACE_DOWN_BOSS_BLINDS",
                pp.FACE_DOWN_BOSS_BLINDS
                | {
                    bm.BlindName.THE_HOOK,
                    bm.BlindName.THE_WALL,
                    bm.BlindName.THE_CLUB,
                    bm.BlindName.THE_GOAD,
                    bm.BlindName.THE_OX,
                    bm.BlindName.THE_PSYCHIC,
                    bm.BlindName.THE_PILLAR,
                    bm.BlindName.THE_WINDOW,
                    bm.BlindName.THE_MANACLE,
                    bm.BlindName.THE_HEAD,
                },
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

        # Step 0 initial + 16 gameplay actions (Play Game and Open Options filtered out)
        self.assertEqual(len(traj.deltas), 17)

        # Verify step 4 (Sort by Rank on Boss Blind): pre_sort selected_slot_uid recorded & face-down bounds narrowed between 5 and King
        sort_delta = traj.deltas[4]
        self.assertEqual(sort_delta.action.type, bm.ActionType.SORT_RANK)
        pre_uid = sort_delta.action.metadata.get("selected_slot_uid")
        self.assertIsNotNone(pre_uid)
        post_sort_state = traj.get_state(4)
        self.assertEqual(post_sort_state.hand.cards[1].slot_uid, pre_uid)
        self.assertTrue(post_sort_state.hand.cards[1].selected)
        fd_bounds = post_sort_state.hand.cards[1].bounds
        self.assertIsNotNone(fd_bounds)
        self.assertTrue(all(c.rank != bm.Rank.ACE for c in fd_bounds.candidate_cards))
        self.assertTrue(
            all(bm.Rank.FOUR not in {c.rank} for c in fd_bounds.candidate_cards)
        )

        # Verify step 5 (Discard) recorded hand_order permutation [2, 0, 1] & selected 5 of Clubs
        discard_delta = traj.deltas[5]
        self.assertEqual(discard_delta.action.type, bm.ActionType.DISCARD)
        self.assertEqual(discard_delta.action.hand_order, [2, 0, 1])
        self.assertEqual(len(discard_delta.action.selected_cards), 1)
        self.assertEqual(discard_delta.action.selected_cards[0].rank, bm.Rank.FIVE)

        # Verify step 7 (End Round -> GameState BLIND + round_won=True) & step 8 (Cash Out -> SHOP + payout)
        end_round_state = traj.get_state(7)
        self.assertEqual(end_round_state.location, bm.Location.BLIND)
        self.assertTrue(end_round_state.round_won)

        cash_out_state = traj.get_state(8)
        self.assertEqual(cash_out_state.location, bm.Location.SHOP)
        self.assertFalse(cash_out_state.round_won)
        self.assertGreater(cash_out_state.money, 4)

        # Verify shop & pack steps matched targets spatially
        self.assertEqual(traj.deltas[9].action.target_idx, 0)  # Blueprint
        self.assertEqual(traj.deltas[9].action.joker_order, [])
        self.assertEqual(traj.deltas[10].action.target_idx, 0)  # Hermit
        self.assertEqual(traj.deltas[10].action.joker_order, [])
        self.assertEqual(traj.deltas[11].action.target_idx, 0)  # Overstock
        self.assertIn(bm.VoucherName.OVERSTOCK, traj.get_state(11).vouchers_redeemed)
        self.assertEqual(traj.deltas[12].action.target_idx, 0)  # Open Arcana Pack
        self.assertEqual(traj.deltas[13].action.target_idx, 1)  # Select Card (Empress)
        # Empress enhanced 2 of Clubs and 3 of Hearts to Mult Cards in full_deck
        pack_step_state = traj.get_state(13)
        mult_cards = [
            c for c in pack_step_state.full_deck if c.enhancement == bm.Enhancement.MULT
        ]
        self.assertEqual(len(mult_cards), 2)
        self.assertEqual(
            pack_step_state.location, bm.Location.SHOP
        )  # Exited back to shop

        # Verify Sell Card sold Blueprint (index 0) and Next Round moved to Ante 2 Small Blind SELECT
        self.assertEqual(traj.deltas[15].action.type, bm.ActionType.SELL_CARD)
        self.assertEqual(traj.deltas[15].action.target_idx, 0)
        self.assertEqual(len(state.inventory.jokers), 0)
        self.assertEqual(state.location, bm.Location.SELECT)
        self.assertEqual(state.current_blind, bm.BlindName.SMALL)
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
                },  # Reroll Shop -> target 490 (offset -10)
            },
        }
        needed = (98, 200, 298, 398, 490)
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
                elif f_idx == 490:
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
        self.assertIn("hud_reroll_cost", observed_hud_fields[490])
        self.assertNotIn("hud_round_score", observed_hud_fields[490])

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

    def test_cli_deck_stake_parsing_and_main_end_to_end_wiring(self):
        import json
        import main as main_module

        self.assertEqual(pp.parse_deck("red"), bm.Deck.RED_DECK)
        self.assertEqual(pp.parse_deck("Plasma Deck"), bm.Deck.PLASMA_DECK)
        self.assertEqual(pp.parse_deck("ABANDONED_DECK"), bm.Deck.ABANDONED_DECK)
        with self.assertRaises(ValueError):
            pp.parse_deck("not_a_deck")

        self.assertEqual(pp.parse_stake("white"), bm.Stake.WHITE)
        self.assertEqual(pp.parse_stake("Gold Stake"), bm.Stake.GOLD)
        self.assertEqual(pp.parse_stake("PURPLE"), bm.Stake.PURPLE)
        with self.assertRaises(ValueError):
            pp.parse_stake("not_a_stake")

        parser = main_module.build_cli_parser()
        args = parser.parse_args(
            [
                "--input-video",
                "sample.mp4",
                "--full-video",
                "--deck",
                "plasma",
                "--stake",
                "gold",
                "--seed",
                "CLISEED1",
                "--conf-ocr",
                "0.90",
            ]
        )
        cfg = main_module.config_from_cli_args(args)
        self.assertEqual(cfg.input_video_path, "sample.mp4")
        self.assertIsNone(cfg.end_timestamp)

        with tempfile.TemporaryDirectory() as tmpdir:
            actions_out = Path(tmpdir) / "actions.json"
            traj_out = Path(tmpdir) / "traj.json"
            args.actions_out = str(actions_out)
            args.trajectory_out = str(traj_out)

            fake_payload = {"key": {"Select Blind": 0}, "frames": {}}
            fake_state = gs.GameState(
                seed="CLISEED1", deck=bm.Deck.PLASMA_DECK, stake=bm.Stake.GOLD
            )
            fake_traj = gs.RunTrajectory()
            fake_traj.record_step(fake_state, action=None)

            with (
                patch(
                    "main.detect_video_actions",
                    return_value=(fake_payload, str(actions_out)),
                ) as mock_det,
                patch(
                    "main.build_run_trajectory",
                    return_value=(fake_state, fake_traj),
                ) as mock_pp,
            ):
                res = main_module.run_end_to_end(args)

            mock_det.assert_called_once()
            mock_pp.assert_called_once()
            pp_kwargs = mock_pp.call_args.kwargs
            self.assertEqual(pp_kwargs["deck"], bm.Deck.PLASMA_DECK)
            self.assertEqual(pp_kwargs["stake"], bm.Stake.GOLD)
            self.assertEqual(pp_kwargs["seed"], "CLISEED1")
            self.assertEqual(pp_kwargs["conf_threshold_ocr"], 0.90)
            self.assertTrue(traj_out.exists())
            saved = json.loads(traj_out.read_text(encoding="utf-8"))
            self.assertEqual(saved["final_state"]["deck"], "Plasma Deck")
            self.assertEqual(saved["final_state"]["stake"], "Gold Stake")
            self.assertEqual(res["trajectory_json"], str(traj_out))


class TestDiagnosticAndCleanupRegressions(unittest.TestCase):
    """Tests premature Next Round filtering, target_card/noise cleanup, and expressive diagnostics."""

    def test_filter_premature_next_rounds_respects_window_and_intervening_actions(self):
        key = {
            "Cash Out": 0,
            "Next Round": 1,
            "Select Blind": 2,
        }
        frames = {
            "3600": {"action": 0, "box": [400, 300, 600, 370]},  # Cash Out
            "3648": {
                "action": 1,
                "box": [100, 100, 200, 150],
            },  # False Next Round (<60f) -> drop
            "4000": {"action": 0, "box": [400, 300, 600, 370]},  # Cash Out
            "4120": {
                "action": 1,
                "box": [100, 100, 200, 150],
            },  # Real Next Round (>=60f) -> keep
            "4200": {"action": 0, "box": [400, 300, 600, 370]},  # Cash Out
            "4220": {
                "action": 2,
                "box": [300, 200, 420, 250],
            },  # Select Blind resets Cash Out timer
            "4240": {
                "action": 1,
                "box": [100, 100, 200, 150],
            },  # Not dropped since Select Blind occurred
        }
        diag = pp._init_diagnostics()
        filtered = pp.filter_premature_next_rounds(frames, key, diag=diag)
        self.assertNotIn("3648", filtered)
        self.assertIn("4120", filtered)
        self.assertIn("4240", filtered)
        self.assertEqual(diag["filtered_premature_next_rounds"], [3648])

    def test_screen_state_tracker_shop_next_round_cooldown(self):
        from screen_state import CoarseScreenState, ScreenStateTracker

        tracker = ScreenStateTracker(state=CoarseScreenState.CASH_OUT)
        tracker.on_template_clicked("Cash Out", frame_idx=1000)
        self.assertEqual(tracker.state, CoarseScreenState.SHOP)
        self.assertEqual(tracker.last_cash_out_frame, 1000)

        self.assertNotIn("Next Round", tracker.allowed_template_names(frame_idx=1048))
        self.assertIn("Next Round", tracker.allowed_template_names(frame_idx=1060))
        self.assertIn("Next Round", tracker.allowed_template_names(frame_idx=None))

        # Premature Next Round should not be included during cooldown
        filtered_templates = tracker.filter_templates(
            [SimpleNamespace(name="Next Round"), SimpleNamespace(name="Reroll Shop")],
            frame_idx=1048,
        )
        self.assertEqual([t.name for t in filtered_templates], ["Reroll Shop"])

        # Legitimate Next Round after cooldown transitions to SELECT and clears last_cash_out_frame
        tracker.on_template_clicked("Next Round", frame_idx=1120)
        self.assertEqual(tracker.state, CoarseScreenState.SELECT)
        self.assertIsNone(tracker.last_cash_out_frame)

    def test_rules_select_blind_from_shop_advances_cleanly(self):
        from game.engine.rules import on_action

        state = gs.GameState()
        state.location = bm.Location.SHOP
        state.current_blind = bm.BlindName.SMALL
        state.pending_blind_advance = True
        state.coupon_shop_active = True

        on_action(state, bm.Action(type=bm.ActionType.SELECT_BLIND, frame_idx=100))
        self.assertEqual(state.location, bm.Location.BLIND)
        self.assertFalse(state.pending_blind_advance)
        self.assertEqual(state.current_blind, bm.BlindName.BIG)
        self.assertFalse(state.coupon_shop_active)

    def test_noise_cleanup_target_card_unexplained_jokers_and_cash_out_location(self):
        from classifiers.inference import HudRecognitionOutput

        d = {
            "key": {
                "Select Blind": 0,
                "Play Hand": 1,
                "End Round": 2,
                "Cash Out": 3,
                "Next Round": 4,
                "Buy Card": 5,
                "Sell Card": 6,
            },
            "frames": {
                "100": {
                    "action": 0,
                    "box": [300, 200, 420, 250],
                },  # Select Blind -> 98 (Blind R1)
                "200": {"action": 1, "box": [780, 900, 880, 950]},  # Play Hand -> 200
                "300": {"action": 2, "box": [500, 500, 600, 550]},  # End Round -> 298
                "400": {"action": 3, "box": [400, 300, 600, 370]},  # Cash Out -> 398
                "445": {
                    "action": 4,
                    "box": [100, 100, 200, 150],
                },  # Premature Next Round (<60f) -> filtered!
                "500": {
                    "action": 5,
                    "box": [855, 575, 945, 625],
                },  # Buy Card[0] (Ramen) -> 500 (offset 0)
                "600": {
                    "action": 4,
                    "box": [100, 100, 200, 150],
                },  # Legitimate Next Round -> 598
                "700": {
                    "action": 0,
                    "box": [300, 200, 420, 250],
                },  # Select Blind -> 698 (Ramen drops!)
                "800": {
                    "action": 6,
                    "box": [665, 95, 735, 165],
                },  # Sell Card -> 798 (Ramen re-appears, sold!)
            },
        }
        needed = (98, 200, 298, 398, 500, 598, 698, 798)
        dummy_frames = {f: np.zeros((1080, 1920, 3), dtype=np.uint8) for f in needed}

        def fake_batch_shops(frames, mode="shop", **kwargs):
            outs = []
            for _, f_idx, ts in frames:
                jokers: list[Any] = []
                shop_items: list[Any] = []
                if f_idx == 500:
                    shop_items = [
                        _make_shop_item("j_ramen", "Joker", [830, 380, 970, 560])
                    ]
                elif f_idx == 598:
                    jokers = [_make_shop_item("j_ramen", "Joker", [520, 40, 660, 220])]
                elif f_idx == 698:
                    # Ramen vanished without a Sell Card action!
                    jokers = []
                elif f_idx == 798:
                    # Ramen reappears before Sell Card resolves at idx 0
                    jokers = [_make_shop_item("j_ramen", "Joker", [520, 40, 660, 220])]
                outs.append(
                    ShopRecognitionOutput(
                        frame_index=f_idx,
                        timestamp_seconds=ts,
                        mode=mode,
                        joker_bar=jokers,
                        shop_items=shop_items,
                    )
                )
            return outs

        def fake_batch_hands(frames, **kwargs):
            outs = []
            for _, f_idx, ts in frames:
                cards = []
                if f_idx == 200:
                    cards = [
                        _make_card_res("Ace", "Spades", selected=True),
                        _make_card_res("King", "Spades", selected=False),
                    ]
                elif f_idx == 798:
                    # Raised hand card visible during Sell Card should NOT leak into Sell Card selected_cards
                    cards = [_make_card_res("6", "Hearts", selected=True)]
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
                vals: dict[str, int | float | None] = {f: None for f in flds}
                if f_idx == 200:
                    vals.update({"hud_round_score": 450, "hud_blind_target": 300})
                elif f_idx == 298:
                    vals.update({"hud_round_score": 450, "hud_blind_target": 300})
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
            state, traj, plan = pp._build_run_trajectory_with_plan(
                d, preloaded_frames=dummy_frames, fps=60.0, expressive=True
            )

        diag = plan["diagnostics"]
        self.assertEqual(diag["filtered_premature_next_rounds"], [445])
        # CASH_OUT ambient vs BLIND round_won state should NOT trigger false location drift warning
        self.assertEqual(diag["location_mismatches"], [])
        # Score updates on PLAY_HAND/END_ROUND counted separately, not as OCR drift overrides
        self.assertGreaterEqual(diag["ocr_score_updates"], 1)
        self.assertNotIn("current_chips", diag["ocr_overrides_by_field"])
        # Frame 700 lost Ramen without a Sell Card, frame 800 re-appeared
        self.assertEqual(
            diag["unexplained_joker_changes"],
            [
                (700, "Select Blind", ["Ramen"], []),
                (800, "Sell Card", [], ["Ramen"]),
            ],
        )

        # Play Hand had sequential hand order -> identity hand_order omitted
        play_act = traj.deltas[2].action
        self.assertEqual(play_act.type, bm.ActionType.PLAY_HAND)
        self.assertEqual(play_act.hand_order, [])
        self.assertEqual(len(play_act.selected_cards), 1)

        # Buy Card recorded target_card="Ramen"
        buy_act = traj.deltas[5].action
        self.assertEqual(buy_act.type, bm.ActionType.BUY_CARD)
        self.assertEqual(buy_act.metadata.get("target_card"), "Ramen")

        # Sell Card recorded target_card="Ramen", omitted identity joker_order/hand_order, and omitted selected_cards
        sell_act = traj.deltas[8].action
        self.assertEqual(sell_act.type, bm.ActionType.SELL_CARD)
        self.assertEqual(sell_act.target_idx, 0)
        self.assertEqual(sell_act.metadata.get("target_card"), "Ramen")
        self.assertEqual(sell_act.joker_order, [])
        self.assertEqual(sell_act.hand_order, [])
        self.assertEqual(sell_act.selected_cards, [])
        self.assertEqual(state.inventory.jokers, [])
        # Buy Card recorded pre-buy shop observation and Next Round cleared stale shelves
        self.assertIn("observed_shop", buy_act.metadata)
        self.assertEqual(buy_act.metadata["observed_shop"]["cards"][0]["name"], "Ramen")
        next_round_state = traj.get_state(6)
        self.assertEqual(next_round_state.location, bm.Location.SELECT)
        self.assertEqual(next_round_state.shop_cards, [])
        self.assertEqual(next_round_state.shop_vouchers, [])
        self.assertEqual(next_round_state.shop_packs, [])

    def test_impute_retroactive_boss_blinds_only_records_boss_on_changes(self):
        traj = gs.RunTrajectory()
        st = gs.GameState()
        traj.record_step(st, action=None)
        st.round = 1
        traj.record_step(
            st, action=bm.Action(type=bm.ActionType.SELECT_BLIND, frame_idx=10)
        )
        st.round = 2
        st.boss_blind = bm.BlindName.THE_HOOK
        traj.record_step(
            st, action=bm.Action(type=bm.ActionType.SELECT_BLIND, frame_idx=20)
        )
        st.ante = 2
        st.round = 3
        st.boss_blind = bm.BlindName.THE_WALL
        traj.record_step(
            st, action=bm.Action(type=bm.ActionType.SELECT_BLIND, frame_idx=30)
        )
        st.round = 4
        traj.record_step(
            st, action=bm.Action(type=bm.ActionType.SELECT_BLIND, frame_idx=40)
        )

        traj.impute_retroactive_boss_blinds()
        self.assertEqual(traj.deltas[0].boss_blind, bm.BlindName.THE_HOOK)
        self.assertIsNone(traj.deltas[1].boss_blind)
        self.assertEqual(traj.deltas[3].boss_blind, bm.BlindName.THE_WALL)
        self.assertIsNone(traj.deltas[4].boss_blind)

    def test_extract_video_frames_parallel_threshold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test_parallel.avi"
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (64, 64))
            self.assertTrue(writer.isOpened())
            try:
                for i in range(60):
                    val = (i * 4) % 256
                    writer.write(np.full((64, 64, 3), val, dtype=np.uint8))
            finally:
                writer.release()

            req = list(range(0, 56))
            progress_ticks: list[int] = []
            extracted = extract_video_frames(
                video_path, req, progress_callback=progress_ticks.append
            )
            self.assertEqual(set(extracted.keys()), set(req))
            self.assertEqual(sum(progress_ticks), len(req))
            self.assertAlmostEqual(float(extracted[45].mean()), 180.0, delta=3.0)

    def test_sort_skips_hand_cv_outside_face_down_boss_blinds(self):
        d = {
            "key": {"Select Blind": 0, "Sort by Rank": 1, "Sort by Suit": 2},
            "frames": {
                "100": {"action": 0, "box": [300, 200, 420, 250]},
                "200": {"action": 1, "box": [850, 900, 930, 940]},
                "300": {"action": 2, "box": [940, 900, 1020, 940]},
            },
        }
        plan = pp.plan_action_pipeline(d)
        self.assertEqual(plan["playing_card_cv_queue"], {})

    def test_consumable_center_right_matches_lifted_slot_over_next_card(self):
        # Exact geometry from f=85824: lifted Devil (slot 0) + unlifted Strength (slot 1)
        cons = [
            _make_shop_item("c_devil", "Tarot", [1370, 36, 1446, 264]),
            _make_shop_item("c_strength", "Tarot", [1576, 59, 1668, 291]),
        ]
        use_btn = [1527, 161, 1600, 227]
        self.assertEqual(
            pp.match_button_to_item_index(use_btn, cons, attach="center_right"),
            0,
        )

    def test_retroactive_consumable_inference_and_transient_miss_preservation(self):
        d = {
            "key": {
                "Select Blind": 0,
                "Play Hand": 1,
                "Use Consumable": 2,
                "End Round": 3,
                "Cash Out": 4,
                "Buy Card": 5,
            },
            "frames": {
                "100": {"action": 0, "box": [300, 200, 420, 250]},
                "200": {"action": 1, "box": [780, 900, 880, 950]},
                "300": {"action": 2, "box": [1637, 168, 1710, 234]},
                "400": {"action": 1, "box": [780, 900, 880, 950]},
            },
        }
        dummy_frames = {f: np.zeros((1080, 1920, 3), dtype=np.uint8) for f in (98, 200, 298, 400)}

        def fake_shops(frames, **kwargs):
            outs = []
            for _, f_idx, ts in frames:
                out = ShopRecognitionOutput(frame_index=f_idx, timestamp_seconds=ts)
                if f_idx == 98:
                    out.consumeables = [
                        _make_shop_item("c_death", "Tarot", [1370, 60, 1447, 287]),
                        _make_shop_item("c_fool", "Tarot", [1576, 63, 1667, 289]),
                    ]
                elif f_idx == 200:
                    # Transient 1-frame CV miss of c_death (still present at 298? No, here test both: keep at 200 if at 298, or infer if gone at 298 & 400)
                    out.consumeables = [
                        _make_shop_item("c_fool", "Tarot", [1576, 63, 1667, 289]),
                    ]
                elif f_idx == 298:
                    # c_death is gone in both 200 and 298 -> Death was used before 200; Fool is at slot 0
                    out.consumeables = [
                        _make_shop_item("c_fool", "Tarot", [1468, 40, 1553, 266]),
                    ]
                elif f_idx == 400:
                    out.consumeables = [
                        _make_shop_item("c_death", "Tarot", [1468, 60, 1553, 286]),
                    ]
                outs.append(out)
            return outs

        def fake_hands(frames, **kwargs):
            return [
                HandRecognitionOutput(frame_index=f_idx, timestamp_seconds=ts, cards=[])
                for _, f_idx, ts in frames
            ]

        with (
            patch("classifiers.inference.batch_recognize_shops", side_effect=fake_shops),
            patch("classifiers.inference.batch_recognize_hands", side_effect=fake_hands),
        ):
            res = pp.process_actions(d, preloaded_frames=dummy_frames, fps=60.0)

        traj: gs.RunTrajectory = res["trajectory"]
        inferred_steps = [
            dl.action for dl in traj.deltas if dl.action and dl.action.metadata.get("inferred")
        ]
        self.assertEqual(len(inferred_steps), 1)
        self.assertEqual(inferred_steps[0].type, bm.ActionType.USE_CONSUMABLE)
        self.assertEqual(inferred_steps[0].metadata.get("target_card"), "Death")
        # Because Death was inferred before Fool, Fool spawned Death back!
        final_cons = [c.name for c in res["state"].inventory.consumables]
        self.assertEqual(final_cons, [bm.TarotName.DEATH])

    def test_pack_hand_uses_pack_hand_roi_and_preserves_blind_card_hand(self):
        blank_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        dummy_frames = {i: blank_frame for i in range(0, 600)}
        d = {
            "key": {
                "Select Blind": 0,
                "Play Hand": 1,
                "Cash Out": 2,
                "Open Pack": 3,
                "Use In Pack": 4,
            },
            "frames": {
                "100": {"action": 0, "box": [10, 10, 50, 50]},
                "150": {"action": 1, "box": [10, 10, 50, 50]},
                "200": {"action": 2, "box": [10, 10, 50, 50]},
                "250": {"action": 3, "box": [10, 10, 50, 50]},
                "300": {"action": 4, "box": [960, 620, 1040, 670]},
            },
        }

        hand_calls: list[tuple[list[int], str | None]] = []

        def fake_shops(frames, mode="shop", **kwargs):
            outs = []
            for _, f_idx, ts in frames:
                out = ShopRecognitionOutput(frame_index=f_idx, timestamp_seconds=ts)
                if f_idx == 248:
                    out.boosters = [_make_shop_item("p_arcana_normal_1", "Booster", [900, 600, 1100, 820])]
                elif f_idx == 298:
                    out.pack_items = [_make_shop_item("c_chariot", "Tarot", [920, 420, 1020, 620])]
                outs.append(out)
            return outs

        def fake_hands(frames, **kwargs):
            hand_calls.append(([f_idx for _, f_idx, _ in frames], kwargs.get("roi_key")))
            return [
                HandRecognitionOutput(frame_index=f_idx, timestamp_seconds=ts, cards=[])
                for _, f_idx, ts in frames
            ]

        with (
            patch("classifiers.inference.batch_recognize_shops", side_effect=fake_shops),
            patch("classifiers.inference.batch_recognize_hands", side_effect=fake_hands),
        ):
            pp.build_run_trajectory(d, preloaded_frames=dummy_frames, fps=60.0)

        self.assertIn(([150], None), hand_calls)
        self.assertIn(([298], "pack_hand"), hand_calls)

    def test_stacked_sell_above_use_collision_resolves_sell_click(self):
        import logging

        import main as app_main
        from config import Config
        from eval_template import EvaluationResult
        from frame import VideoFrame
        from template import Template

        # Build synthetic image with unshadowed Sell (gry=46 in y=150..153) above bright Use top edge (y=154)
        img_bgr = np.full((300, 300, 3), 46, dtype=np.uint8)
        img_bgr[154:220, 80:160] = 220  # Bright Use Consumable button directly below Sell Card

        sell_tpl = Template(
            name="Sell Card",
            filenames=[],
            norm_rois=[],
            px_rois=[],
            images=[],
            images_gray=[],
            masks=[],
            shadow_dy=0,
            shadow_dw=0,
            shadow_dh=0,
            adjusted=False,
        )
        use_tpl = Template(
            name="Use Consumable",
            filenames=[],
            norm_rois=[],
            px_rois=[],
            images=[],
            images_gray=[],
            masks=[],
            shadow_dy=3,
            shadow_dw=0,
            shadow_dh=0,
            adjusted=False,
        )

        def fake_eval(tpl, _frame, _gray, _cfg, **_kw):
            if tpl.name == "Sell Card":
                # Simulated full 10px check sees Use button border -> False
                return EvaluationResult(
                    template_name="Sell Card",
                    found=True,
                    match=None,
                    clicked=False,
                    box=(80, 68, 155, 150),
                    plot_kwargs={},
                )
            return EvaluationResult(
                template_name="Use Consumable",
                found=True,
                match=None,
                clicked=False,
                box=(82, 154, 155, 220),
                plot_kwargs={},
            )

        cfg = Config()
        logger = logging.getLogger("test_sell_use_collision")
        vf = VideoFrame(image=img_bgr, index=100, timestamp_seconds=1.0)
        with patch.object(app_main, "eval", side_effect=fake_eval):
            _, _, _, clicked, clicked_name = app_main._process_frame(
                vf,
                [sell_tpl, use_tpl],
                {"Sell Card": 0, "Use Consumable": 1},
                cfg,
                logger,
            )
        self.assertEqual(clicked_name, "Sell Card")
        self.assertEqual(clicked, {"action": 0, "box": [80, 68, 155, 150]})

    def test_remaining_deck_outside_blind_and_played_discard_dedup(self):
        c1 = bm.PlayingCard(rank=bm.Rank.KING, suit=bm.Suit.CLUBS, id=1)
        c2 = bm.PlayingCard(rank=bm.Rank.KING, suit=bm.Suit.CLUBS, id=2)
        c3 = bm.PlayingCard(rank=bm.Rank.ACE, suit=bm.Suit.SPADES, id=3)
        state = gs.GameState(
            location=bm.Location.BLIND,
            round_won=False,
            full_deck=[c1, c2, c3],
            played_cards_this_round=[c1],
            discard_pile=[c1],
        )
        # c1 appears in both played_cards_this_round and discard_pile -> only 1 subtracted, c2 remains.
        self.assertEqual([c.id for c in state.remaining_deck], [2, 3])
        # Once round_won=True or inside PACK/SHOP, discard/played cards return to draw pool.
        state.round_won = True
        self.assertEqual([c.id for c in state.remaining_deck], [1, 2, 3])
        state.round_won = False
        state.location = bm.Location.PACK
        self.assertEqual([c.id for c in state.remaining_deck], [1, 2, 3])
        self.assertEqual([c.id for c in state.get_face_down_candidate_pool()], [1, 2, 3])

    def test_amber_acorn_preserves_face_down_jokers_and_dense_joker_bar_dedup(self):
        from classifiers.inference import _filter_kwargs_for_shelf
        from classifiers.shop_classifier.pipeline import ShopItemResult, filter_shop_items

        j1 = bm.Joker(name=bm.JokerName.VAMPIRE, id=1, facing_back=True)
        j2 = bm.Joker(name=bm.JokerName.MIDAS_MASK, id=2, facing_back=True)
        state = gs.GameState(
            location=bm.Location.BLIND,
            current_blind=bm.BlindName.AMBER_ACORN,
            inventory=bm.Inventory(jokers=[j1, j2]),
        )
        bogus_out = ShopRecognitionOutput(
            frame_index=10,
            timestamp_seconds=1.0,
            joker_bar=[
                _make_shop_item("j_raised_fist", "Joker", [0, 40, 80, 260]),
                _make_shop_item("j_blueprint", "Joker", [65, 40, 145, 260]),
            ],
        )
        act = bm.Action(type=bm.ActionType.PLAY_HAND, frame_idx=10, timestamp=1.0)
        pp._sync_bars_into_state(state, bogus_out, act)
        self.assertEqual([j.name for j in state.inventory.jokers], [bm.JokerName.VAMPIRE, bm.JokerName.MIDAS_MASK])

        # Dense 65px-spaced jokers stay separate while split-half (<38px) & sliver (<50w) drop.
        items = [
            ShopItemResult(0, "Joker", "j_vampire", "None", False, False, False, False, [0, 40, 80, 260], {"detection": 0.95, "key": 0.99}),
            ShopItemResult(1, "Joker", "j_trio", "Negative", False, False, False, False, [65, 40, 145, 260], {"detection": 0.94, "key": 0.98}),
            ShopItemResult(2, "Joker", "j_trio", "Negative", False, False, False, False, [75, 40, 130, 260], {"detection": 0.55, "key": 0.85}),
            ShopItemResult(3, "Joker", "j_red_card", "None", False, False, False, False, [815, 1, 857, 250], {"detection": 0.50, "key": 0.55}),
        ]
        filtered = filter_shop_items(items, **_filter_kwargs_for_shelf("joker_bar", 0.30))
        self.assertEqual([it.key for it in filtered], ["j_vampire", "j_trio"])

    def test_destroyed_candidate_not_resurrected_and_immolate_witness(self):
        from game.engine.rules import on_action

        surv = bm.PlayingCard(rank=bm.Rank.TWO, suit=bm.Suit.HEARTS, id=27)
        victim = bm.PlayingCard(rank=bm.Rank.SEVEN, suit=bm.Suit.CLUBS, id=6)
        other = bm.PlayingCard(rank=bm.Rank.KING, suit=bm.Suit.DIAMONDS, id=17)
        fd_slot = bm.HandCard(
            card=bm.PlayingCard(rank=bm.Rank.NONE, suit=bm.Suit.NONE, id=0),
            facing_back=True,
            slot_uid=1,
            bounds=bm.CardBounds(is_face_down=True),
        )
        fd_slot.bounds.set_candidates([victim])
        state = gs.GameState(
            location=bm.Location.PACK,
            full_deck=[surv, victim, other],
            hand=bm.Hand(cards=[bm.HandCard(card=surv, slot_uid=2), fd_slot, bm.HandCard(card=other, slot_uid=3)]),
        )
        state.remove_card_from_deck(victim)
        self.assertTrue(victim.destroyed)
        self.assertEqual(fd_slot.bounds.candidate_cards, [])
        # Observing destroyed card's rank/suit adds new instance rather than reusing destroyed id=6
        pp._sync_hand_into_state(
            state,
            [
                bm.HandCard(card=bm.PlayingCard(rank=bm.Rank.TWO, suit=bm.Suit.HEARTS)),
                bm.HandCard(card=bm.PlayingCard(rank=bm.Rank.SEVEN, suit=bm.Suit.CLUBS)),
                bm.HandCard(card=bm.PlayingCard(rank=bm.Rank.KING, suit=bm.Suit.DIAMONDS)),
            ],
        )
        hand_ids = [hc.card.id for hc in state.hand.cards]
        self.assertNotIn(6, hand_ids)
        self.assertEqual(sorted(hand_ids), [17, 27, 28])

        # With _surviving_hand_ids witnessing `surv` (2H) still present, Immolate spares 2H.
        deck = [surv] + [
            bm.PlayingCard(rank=r, suit=bm.Suit.SPADES, id=100 + i)
            for i, r in enumerate([bm.Rank.THREE, bm.Rank.FOUR, bm.Rank.FIVE, bm.Rank.SIX, bm.Rank.SEVEN])
        ]
        imm_state = gs.GameState(
            seed="XARBTZU7",
            location=bm.Location.PACK,
            pack_choices_remaining=2,
            active_pack=bm.BoosterPack(booster_type=bm.BoosterType.SPECTRAL, size=bm.BoosterSize.MEGA, choose=2, extra=4, cost=8),
            full_deck=list(deck),
            hand=bm.Hand(cards=[bm.HandCard(card=c, slot_uid=i + 1) for i, c in enumerate(deck)]),
            inventory=bm.Inventory(),
        )
        imm_state.pack_cards = [bm.Spectral(name=bm.SpectralName.IMMOLATE)]
        imm_act = bm.Action(
            type=bm.ActionType.USE_IN_PACK,
            target_idx=0,
            metadata={"target_card": "Immolate", "_surviving_hand_ids": [27]},
        )
        on_action(imm_state, imm_act)
        self.assertIn(27, [c.id for c in imm_state.full_deck])
        self.assertEqual(len(imm_state.full_deck), 1)

    def test_deck_constrained_matching_fixes_gold_q_and_preserves_new_pack_cards(self):
        # 1. Duplicate conflict: deck has 1 Jack♦ Gold and 1 Queen♦ Gold;
        # CV observed [Jack♦ Gold (0.93/0.07), Jack♦ Gold (1.00)].
        qd_gold = bm.PlayingCard(
            rank=bm.Rank.QUEEN,
            suit=bm.Suit.DIAMONDS,
            enhancement=bm.Enhancement.GOLD,
            id=1,
        )
        jd_gold = bm.PlayingCard(
            rank=bm.Rank.JACK,
            suit=bm.Suit.DIAMONDS,
            enhancement=bm.Enhancement.GOLD,
            id=2,
        )
        qh_gold = bm.PlayingCard(
            rank=bm.Rank.QUEEN,
            suit=bm.Suit.HEARTS,
            enhancement=bm.Enhancement.GOLD,
            id=3,
        )
        base_2c = bm.PlayingCard(rank=bm.Rank.TWO, suit=bm.Suit.CLUBS, id=4)
        state = gs.GameState(
            location=bm.Location.BLIND,
            full_deck=[qd_gold, jd_gold, qh_gold, base_2c],
        )

        raw_dup = [
            _make_card_res(
                "Jack",
                "Diamonds",
                enhancement="Gold",
                confidences={"rank": 0.929, "suit": 0.999, "enhancement": 0.999, "edition": 1.0, "seal": 1.0},
                top_probs={
                    "rank": {"Jack": 0.929, "Queen": 0.071},
                    "suit": {"Diamonds": 0.999},
                    "enhancement": {"Gold": 0.999},
                    "edition": {"None": 1.0},
                    "seal": {"None": 1.0},
                },
            ),
            _make_card_res(
                "Jack",
                "Diamonds",
                enhancement="Gold",
                confidences={"rank": 1.0, "suit": 1.0, "enhancement": 1.0, "edition": 1.0, "seal": 1.0},
                top_probs={
                    "rank": {"Jack": 1.0},
                    "suit": {"Diamonds": 1.0},
                    "enhancement": {"Gold": 1.0},
                    "edition": {"None": 1.0},
                    "seal": {"None": 1.0},
                },
            ),
            # Ambiguous J♥ Gold (0.764 vs Q=0.226) with no J♥ Gold in deck -> snaps to Q♥ Gold
            _make_card_res(
                "Jack",
                "Hearts",
                enhancement="Gold",
                confidences={"rank": 0.764, "suit": 1.0, "enhancement": 1.0, "edition": 1.0, "seal": 1.0},
                top_probs={
                    "rank": {"Jack": 0.764, "Queen": 0.226},
                    "suit": {"Hearts": 1.0},
                    "enhancement": {"Gold": 1.0},
                    "edition": {"None": 1.0},
                    "seal": {"None": 1.0},
                },
            ),
            # High-confidence new Polychrome Ace♠ (e.g., missed Standard Pack take) -> must NOT remap!
            _make_card_res(
                "Ace",
                "Spades",
                edition="Polychrome",
                confidences={"rank": 0.999, "suit": 0.999, "enhancement": 1.0, "edition": 0.999, "seal": 1.0},
                top_probs={
                    "rank": {"Ace": 0.999},
                    "suit": {"Spades": 0.999},
                    "enhancement": {"None": 1.0},
                    "edition": {"Polychrome": 0.999},
                    "seal": {"None": 1.0},
                },
            ),
        ]
        obs_hcs = [pp._convert_card_result_to_hand_card(c) for c in raw_dup]
        diag = pp._init_diagnostics()
        pp._resolve_hand_with_deck_constraints(state, raw_dup, obs_hcs, diag=diag)

        self.assertEqual(obs_hcs[0].card.rank, bm.Rank.QUEEN)
        self.assertEqual(obs_hcs[0].card.suit, bm.Suit.DIAMONDS)
        self.assertEqual(obs_hcs[1].card.rank, bm.Rank.JACK)
        self.assertEqual(obs_hcs[1].card.suit, bm.Suit.DIAMONDS)
        self.assertEqual(obs_hcs[2].card.rank, bm.Rank.QUEEN)
        self.assertEqual(obs_hcs[2].card.suit, bm.Suit.HEARTS)
        # New pack card preserved untouched!
        self.assertEqual(obs_hcs[3].card.rank, bm.Rank.ACE)
        self.assertEqual(obs_hcs[3].card.suit, bm.Suit.SPADES)
        self.assertEqual(obs_hcs[3].card.edition, bm.Edition.POLYCHROME)
        self.assertEqual(diag["deck_constrained_remaps"], 2)

        # Syncing adds the genuinely new Polychrome A♠ cleanly into full_deck
        pp._sync_hand_into_state(state, obs_hcs)
        self.assertEqual(len(state.full_deck), 5)
        self.assertTrue(
            any(
                c.rank == bm.Rank.ACE and c.suit == bm.Suit.SPADES and c.edition == bm.Edition.POLYCHROME
                for c in state.full_deck
            )
        )

    def test_face_down_played_cards_play_area_resolution_and_bounds_pruning(self):
        # Simulate The Fish: Small + Big blind skipped? Or play directly on Small Blind where face-down cards are dealt
        d = {
            "key": {"Select Blind": 0, "Play Hand": 1},
            "frames": {
                "100": {"action": 0, "box": [100, 100, 180, 140]},  # Select Blind -> 98
                "200": {"action": 1, "box": [780, 900, 880, 950]},  # Play Hand -> target 200; probe at 240, 265, 290
            },
        }
        dummy_frames = {f: np.zeros((1080, 1920, 3), dtype=np.uint8) for f in (98, 200, 240, 265, 290)}

        def fake_shops(frames, mode="shop", **kwargs):
            return [
                ShopRecognitionOutput(frame_index=f_idx, timestamp_seconds=ts, mode=mode)
                for _, f_idx, ts in frames
            ]

        def fake_hands(frames, **kwargs):
            roi_key = kwargs.get("roi_key")
            outs = []
            for _, f_idx, ts in frames:
                cards = []
                if roi_key == "play_area" and f_idx in (240, 265, 290):
                    # Played 4 cards revealed in play_area: Pair of Aces (Ace♥, Ace♣, 8♠, 2♠)
                    cards = [
                        _make_card_res("Ace", "Hearts", confidences={"rank": 0.99, "suit": 0.99}),
                        _make_card_res("Ace", "Clubs", confidences={"rank": 0.99, "suit": 0.99}),
                        _make_card_res("8", "Spades", confidences={"rank": 0.99, "suit": 0.99}),
                        _make_card_res("2", "Spades", confidences={"rank": 0.99, "suit": 0.99}),
                    ]
                elif roi_key is None and f_idx == 200:
                    # Hand has 4 selected (2 face-up Aces + 2 face-down) plus 1 unselected face-down card
                    cards = [
                        _make_card_res("Ace", "Hearts", selected=True),
                        _make_card_res("Ace", "Clubs", selected=True),
                        _make_card_res("None", "None", selected=True, face_down=True),
                        _make_card_res("None", "None", selected=True, face_down=True),
                        _make_card_res("None", "None", selected=False, face_down=True),
                    ]
                outs.append(HandRecognitionOutput(frame_index=f_idx, timestamp_seconds=ts, cards=cards))
            return outs

        with (
            patch("classifiers.inference.batch_recognize_shops", side_effect=fake_shops),
            patch("classifiers.inference.batch_recognize_hands", side_effect=fake_hands),
        ):
            state, traj, plan = pp._build_run_trajectory_with_plan(
                d, preloaded_frames=dummy_frames, fps=60.0, expressive=True
            )

        self.assertEqual(plan["diagnostics"]["face_down_plays_resolved"], 1)
        play_act = traj.deltas[-1].action
        self.assertEqual(play_act.type, bm.ActionType.PLAY_HAND)
        self.assertEqual(
            [(c.rank, c.suit) for c in play_act.selected_cards],
            [
                (bm.Rank.ACE, bm.Suit.HEARTS),
                (bm.Rank.ACE, bm.Suit.CLUBS),
                (bm.Rank.EIGHT, bm.Suit.SPADES),
                (bm.Rank.TWO, bm.Suit.SPADES),
            ],
        )
        self.assertTrue(all(c.id > 0 for c in play_act.selected_cards))
        self.assertEqual(state.hand_play_counts.get(bm.PokerHand.PAIR, 0), 1)
        self.assertEqual(state.hand_play_counts.get(bm.PokerHand.HIGH_CARD, 0), 0)

        played_ids = {c.id for c in state.played_cards_this_round}
        rem_ids = {c.id for c in state.remaining_deck}
        self.assertTrue({c.id for c in play_act.selected_cards}.issubset(played_ids))
        self.assertTrue(played_ids.isdisjoint(rem_ids))
        # Held unselected face-down card must have pruned all 4 played cards from its bounds
        held_fd = [hc for hc in state.hand.cards if hc.facing_back]
        self.assertEqual(len(held_fd), 1)
        cand_ids = {c.id for c in held_fd[0].bounds.candidate_cards}
        self.assertTrue(played_ids.isdisjoint(cand_ids))

    def test_preclicked_cash_out_and_skip_pack_synthesis(self):
        # 1. Preclicked Cash Out + Preclicked Skip Pack (Red Card +3 Mult)
        d = {
            "key": {
                "Select Blind": 0,
                "Play Hand": 1,
                "End Round": 2,
                "Buy Card": 3,
                "Open Pack": 4,
                "Next Round": 5,
            },
            "frames": {
                "100": {"action": 0, "box": [100, 100, 180, 140]},  # Select Blind -> 98
                "200": {"action": 1, "box": [780, 900, 880, 950]},  # Play Hand -> 200
                "300": {"action": 2, "box": [500, 500, 600, 550]},  # End Round -> 298
                # CASH_OUT omitted (preclicked)! Directly Buy Card (Red Card) in shop:
                "400": {"action": 3, "box": [855, 575, 945, 625]},  # Buy Card[0] -> 400
                "500": {"action": 4, "box": [1120, 900, 1220, 950]},  # Open Pack -> 498
                # SKIP_PACK omitted (preclicked)! Directly Next Round -> 598
                "600": {"action": 5, "box": [100, 100, 200, 150]},
            },
        }
        dummy_frames = {f: np.zeros((1080, 1920, 3), dtype=np.uint8) for f in (98, 200, 298, 400, 498, 598)}

        def fake_shops(frames, mode="shop", **kwargs):
            outs = []
            for _, f_idx, ts in frames:
                out = ShopRecognitionOutput(frame_index=f_idx, timestamp_seconds=ts, mode=mode)
                if f_idx == 400:
                    out.shop_items = [_make_shop_item("j_red_card", "Joker", [830, 380, 970, 560])]
                elif f_idx == 498:
                    out.joker_bar = [_make_shop_item("j_red_card", "Joker", [520, 40, 660, 220])]
                    out.boosters = [_make_shop_item("p_celestial_normal_1", "Booster", [1100, 730, 1240, 900])]
                elif f_idx == 598:
                    out.joker_bar = [_make_shop_item("j_red_card", "Joker", [520, 40, 660, 220])]
                outs.append(out)
            return outs

        def fake_hands(frames, **kwargs):
            return [
                HandRecognitionOutput(
                    frame_index=f_idx,
                    timestamp_seconds=ts,
                    cards=[_make_card_res("Ace", "Spades", selected=True)] if f_idx == 200 else [],
                )
                for _, f_idx, ts in frames
            ]

        with (
            patch("classifiers.inference.batch_recognize_shops", side_effect=fake_shops),
            patch("classifiers.inference.batch_recognize_hands", side_effect=fake_hands),
        ):
            state, traj, plan = pp._build_run_trajectory_with_plan(
                d, preloaded_frames=dummy_frames, fps=60.0, expressive=True
            )

        self.assertEqual(plan["diagnostics"]["inferred_cash_outs"], 1)
        self.assertEqual(plan["diagnostics"]["inferred_skip_packs"], 1)
        act_types = [delta.action.type for delta in traj.deltas[1:]]
        self.assertIn(bm.ActionType.CASH_OUT, act_types)
        self.assertIn(bm.ActionType.SKIP_PACK, act_types)
        # Red Card should have gained +3 mult from the inferred preclicked Skip Pack!
        red_cards = [j for j in state.inventory.jokers if j.name == bm.JokerName.RED_CARD]
        self.assertEqual(len(red_cards), 1)
        self.assertEqual(red_cards[0].state.get("mult"), 3)

        # 2. Buffoon pack where a new Joker appears on joker_bar must recover pick, NOT infer Skip Pack
        d_buf = {
            "key": {"Cash Out": 0, "Open Pack": 1, "Next Round": 2},
            "frames": {
                "100": {"action": 0, "box": [400, 300, 600, 370]},  # Cash Out -> 98
                "200": {"action": 1, "box": [1120, 900, 1220, 950]},  # Open Pack -> 198
                "300": {"action": 2, "box": [100, 100, 200, 150]},  # Next Round -> 298
            },
        }
        buf_frames = {f: np.zeros((1080, 1920, 3), dtype=np.uint8) for f in (98, 198, 298)}

        def fake_buf_shops(frames, mode="shop", **kwargs):
            outs = []
            for _, f_idx, ts in frames:
                out = ShopRecognitionOutput(frame_index=f_idx, timestamp_seconds=ts, mode=mode)
                if f_idx == 198:
                    out.boosters = [_make_shop_item("p_buffoon_normal_1", "Booster", [1100, 730, 1240, 900])]
                elif f_idx == 298:
                    out.joker_bar = [_make_shop_item("j_joker", "Joker", [520, 40, 660, 220])]
                outs.append(out)
            return outs

        with (
            patch("classifiers.inference.batch_recognize_shops", side_effect=fake_buf_shops),
            patch("classifiers.inference.batch_recognize_hands", side_effect=fake_hands),
        ):
            buf_state, buf_traj, buf_plan = pp._build_run_trajectory_with_plan(
                d_buf, preloaded_frames=buf_frames, fps=60.0, expressive=True
            )

        self.assertEqual(buf_plan["diagnostics"]["inferred_skip_packs"], 0)
        self.assertEqual([j.name for j in buf_state.inventory.jokers], [bm.JokerName.JOKER])
        self.assertNotIn(bm.ActionType.SKIP_PACK, [delta.action.type for delta in buf_traj.deltas[1:]])

    def test_to_do_list_on_screen_shop_pack_tracking_and_high_card_spawn(self):
        from game.engine.rng_sim.api import BalatroRun

        # Seed LX6LFV2Q: roll 1 = High Card; roll 2 = next hand on stream
        expected_run = BalatroRun(seed="LX6LFV2Q")
        exp_roll1 = expected_run.get_to_do_list_hand(is_initial_spawn=True)
        exp_roll2 = expected_run.get_to_do_list_hand(is_initial_spawn=True)
        self.assertEqual(exp_roll1, bm.PokerHand.HIGH_CARD)

        d = {
            "key": {
                "Cash Out": 0,
                "Reroll Shop": 1,
                "Open Pack": 2,
                "Use In Pack": 3,
                "Next Round": 4,
            },
            "frames": {
                "100": {"action": 0, "box": [400, 300, 600, 370]},  # Cash Out -> 98 (enter Shop)
                "200": {"action": 1, "box": [650, 575, 740, 640]},  # Reroll Shop (sees To Do List #1 on pre-reroll shelf!) -> 190
                "300": {"action": 2, "box": [1120, 900, 1220, 950]},  # Open Pack -> 298
                "400": {"action": 3, "box": [855, 575, 945, 625]},  # Use In Pack (picks To Do List #2!) -> 398
            },
        }
        dummy_frames = {f: np.zeros((1080, 1920, 3), dtype=np.uint8) for f in (98, 190, 298, 398)}

        def fake_shops(frames, mode="shop", **kwargs):
            outs = []
            for _, f_idx, ts in frames:
                out = ShopRecognitionOutput(frame_index=f_idx, timestamp_seconds=ts, mode=mode)
                if f_idx == 190:
                    # Pre-reroll shop shelf has To Do List (roll 1 -> High Card), rerolled away without buying
                    out.shop_items = [_make_shop_item("j_todo_list", "Joker", [830, 380, 970, 560])]
                elif f_idx == 298:
                    out.boosters = [_make_shop_item("p_buffoon_normal_1", "Booster", [1100, 730, 1240, 900])]
                elif f_idx == 398:
                    # Buffoon pack contains To Do List -> must advance to roll 2 (exp_roll2), NOT reuse roll 1
                    out.pack_items = [_make_shop_item("j_todo_list", "Joker", [830, 380, 970, 560])]
                outs.append(out)
            return outs

        def fake_hands(frames, **kwargs):
            return [HandRecognitionOutput(frame_index=f_idx, timestamp_seconds=ts, cards=[]) for _, f_idx, ts in frames]

        with (
            patch("classifiers.inference.batch_recognize_shops", side_effect=fake_shops),
            patch("classifiers.inference.batch_recognize_hands", side_effect=fake_hands),
        ):
            state, traj, _ = pp._build_run_trajectory_with_plan(
                d, preloaded_frames=dummy_frames, seed="LX6LFV2Q", fps=60.0
            )

        # Reroll action observed shop To Do List with roll 1 (High Card)
        reroll_obs = traj.deltas[2].action.metadata["observed_shop"]["cards"][0]
        self.assertEqual(reroll_obs["state"]["target_hand"], bm.PokerHand.HIGH_CARD.value)
        # Picking To Do List in Buffoon Pack got roll 2 without double-rolling!
        self.assertEqual(len(state.inventory.jokers), 1)
        td = state.inventory.jokers[0]
        self.assertEqual(td.name, bm.JokerName.TO_DO_LIST)
        self.assertEqual(td.state.get("target_hand"), exp_roll2)
        self.assertEqual(state.to_do_poker_hand, exp_roll2)
        self.assertEqual(state.rng.get_channel_call_count("to_do"), 2)

    def test_hand_size_over_15_flags_corrupted_steps(self):
        # 1. Direct helper guard: max_size <= 15 OR deck remaining + hand <= 15 stays uncorrupted
        small_state = gs.GameState(location=bm.Location.BLIND, apply_deck_stake_on_init=True)
        small_state.hand.max_size = 8
        act_ok = bm.Action(type=bm.ActionType.PLAY_HAND, frame_idx=10)
        self.assertFalse(pp._check_over_15_hand_size(small_state, act_ok, [], has_hand_cv=True))
        self.assertNotIn("corrupted", act_ok.metadata)

        # Even if max_size > 15, if only 12 cards remain in the entire deck, <= 15 can appear on screen
        tiny_deck_state = gs.GameState(
            location=bm.Location.BLIND,
            full_deck=[bm.PlayingCard(rank=bm.Rank.TWO, suit=bm.Suit.HEARTS, id=i + 1) for i in range(12)],
            apply_deck_stake_on_init=False,
        )
        tiny_deck_state.hand.max_size = 18
        act_tiny = bm.Action(type=bm.ActionType.PLAY_HAND, frame_idx=20)
        self.assertFalse(pp._check_over_15_hand_size(tiny_deck_state, act_tiny, [], has_hand_cv=True))
        self.assertNotIn("corrupted", act_tiny.metadata)

        # 2. End-to-end orchestration: Play Hand with expected hand > 15 (e.g. stacked Juggle Tags)
        # flags step metadata with corrupted=True and corruption_reasons=["hand_size_over_15"]
        d = {
            "key": {"Select Blind": 0, "Play Hand": 1},
            "frames": {
                "100": {"action": 0, "box": [100, 100, 180, 140]},  # Select Blind
                "200": {"action": 1, "box": [780, 900, 880, 950]},  # Play Hand with max_size=17
            },
        }
        dummy_frames = {f: np.zeros((1080, 1920, 3), dtype=np.uint8) for f in (98, 200)}

        def fake_shops(frames, mode="shop", **kwargs):
            return [ShopRecognitionOutput(frame_index=f_idx, timestamp_seconds=ts, mode=mode) for _, f_idx, ts in frames]

        def fake_hands(frames, **kwargs):
            # YOLO only detects 10 distinct cards out of 17+ in hand
            sample_ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "Ace"]
            return [
                HandRecognitionOutput(
                    frame_index=f_idx,
                    timestamp_seconds=ts,
                    cards=[_make_card_res(sample_ranks[i], "Spades", selected=(i == 0)) for i in range(10)]
                    if f_idx == 200
                    else [],
                )
                for _, f_idx, ts in frames
            ]

        orig_game_state = pp.GameState

        class JuggledInitState(orig_game_state):
            def __post_init__(self):
                super().__post_init__()
                self.hand.max_size = 17

        with (
            patch.object(pp, "GameState", JuggledInitState),
            patch("classifiers.inference.batch_recognize_shops", side_effect=fake_shops),
            patch("classifiers.inference.batch_recognize_hands", side_effect=fake_hands),
        ):
            _, traj, plan = pp._build_run_trajectory_with_plan(
                d, preloaded_frames=dummy_frames, fps=60.0, expressive=True
            )

        diag = plan["diagnostics"]
        self.assertEqual(diag["hand_size_over_15_actions"], [(200, "Play Hand", 17, 10)])
        self.assertEqual(diag["corrupted_actions"], [(200, "Play Hand", "hand_size_over_15")])
        play_meta = traj.deltas[-1].action.metadata
        self.assertTrue(play_meta.get("corrupted"))
        self.assertEqual(play_meta.get("corruption_reasons"), ["hand_size_over_15"])
        self.assertEqual(play_meta.get("expected_hand_max_size"), 17)
        self.assertEqual(play_meta.get("detected_hand_cards"), 10)

    def test_cascade_quarantine_scanner(self):
        from classifiers.inference import HudRecognitionOutput

        # 1. Isolated single unexpected deck addition (+1 missed Standard Pack card)
        # must NOT quarantine downstream steps.
        d_ok = {
            "key": {"Select Blind": 0, "Play Hand": 1, "End Round": 2, "Cash Out": 3, "Next Round": 4},
            "frames": {
                "100": {"action": 0, "box": [100, 100, 180, 140]},  # Select Blind -> 98
                "200": {"action": 1, "box": [780, 900, 880, 950]},  # Play Hand (+1 Holographic A♠) -> 200
                "300": {"action": 2, "box": [500, 500, 600, 550]},  # End Round -> 298
                "400": {"action": 3, "box": [400, 300, 600, 370]},  # Cash Out -> 398
                "500": {"action": 4, "box": [100, 100, 200, 150]},  # Next Round -> 498
            },
        }
        frames_ok = {f: np.zeros((1080, 1920, 3), dtype=np.uint8) for f in (98, 200, 298, 398, 498)}

        def fake_shops_empty(frames, mode="shop", **kwargs):
            return [ShopRecognitionOutput(frame_index=f_idx, timestamp_seconds=ts, mode=mode) for _, f_idx, ts in frames]

        def fake_hands_single_new(frames, **kwargs):
            return [
                HandRecognitionOutput(
                    frame_index=f_idx,
                    timestamp_seconds=ts,
                    cards=[_make_card_res("Ace", "Spades", edition="Holo", selected=True)] if f_idx == 200 else [],
                )
                for _, f_idx, ts in frames
            ]

        with (
            patch("classifiers.inference.batch_recognize_shops", side_effect=fake_shops_empty),
            patch("classifiers.inference.batch_recognize_hands", side_effect=fake_hands_single_new),
        ):
            _, traj_ok, plan_ok = pp._build_run_trajectory_with_plan(
                d_ok, preloaded_frames=frames_ok, fps=60.0, expressive=True
            )

        self.assertIsNone(plan_ok["diagnostics"]["quarantine_watermark"])
        self.assertFalse(any((d.action.metadata or {}).get("corrupted") for d in traj_ok.deltas[1:]))

        # 2. Deck discovery burst (+2 then +2 unexpected adds in Blind) followed by clean Shop/Next Round:
        # watermark must quarantine Blind burst AND all downstream Shop steps!
        d_burst = {
            "key": {"Select Blind": 0, "Play Hand": 1, "End Round": 2, "Cash Out": 3, "Next Round": 4},
            "frames": {
                "100": {"action": 0, "box": [100, 100, 180, 140]},  # Step 1: Select Blind (clean)
                "200": {"action": 1, "box": [780, 900, 880, 950]},  # Step 2: Play Hand (+2 Holo A♠)
                "250": {"action": 1, "box": [780, 900, 880, 950]},  # Step 3: Play Hand (+2 Holo K♠)
                "300": {"action": 2, "box": [500, 500, 600, 550]},  # Step 4: End Round
                "400": {"action": 3, "box": [400, 300, 600, 370]},  # Step 5: Cash Out
                "500": {"action": 4, "box": [100, 100, 200, 150]},  # Step 6: Next Round (clean screen, poisoned deck!)
            },
        }
        frames_burst = {f: np.zeros((1080, 1920, 3), dtype=np.uint8) for f in (98, 200, 250, 298, 398, 498)}

        def fake_hands_burst(frames, **kwargs):
            outs = []
            for _, f_idx, ts in frames:
                if f_idx == 200:
                    cards = [
                        _make_card_res("Ace", "Spades", edition="Holo", selected=True),
                        _make_card_res("Ace", "Hearts", edition="Holo", selected=False),
                    ]
                elif f_idx == 250:
                    cards = [
                        _make_card_res("King", "Spades", edition="Holo", selected=True),
                        _make_card_res("King", "Hearts", edition="Holo", selected=False),
                    ]
                else:
                    cards = []
                outs.append(HandRecognitionOutput(frame_index=f_idx, timestamp_seconds=ts, cards=cards))
            return outs

        def fake_hud_ok(frames, fields_by_frame=None, **kwargs):
            outs = []
            for _, f_idx, ts in frames:
                flds = list((fields_by_frame or {}).get(f_idx, ()))
                vals = {f: None for f in flds}
                if f_idx in (200, 250):
                    vals["hud_hands"] = 4 if f_idx == 200 else 3
                outs.append(HudRecognitionOutput(frame_index=f_idx, timestamp_seconds=ts, values=vals))
            return outs

        with (
            patch("classifiers.inference.batch_recognize_shops", side_effect=fake_shops_empty),
            patch("classifiers.inference.batch_recognize_hands", side_effect=fake_hands_burst),
            patch("classifiers.inference.batch_recognize_hud", side_effect=fake_hud_ok),
        ):
            _, traj_burst, plan_burst = pp._build_run_trajectory_with_plan(
                d_burst, preloaded_frames=frames_burst, fps=60.0, expressive=True
            )

        self.assertEqual(plan_burst["diagnostics"]["quarantine_watermark"], 2)
        self.assertFalse((traj_burst.deltas[1].action.metadata or {}).get("corrupted", False))
        for s in range(2, len(traj_burst.deltas)):
            meta = traj_burst.deltas[s].action.metadata
            self.assertTrue(meta.get("corrupted"), f"Expected step {s} quarantined")
            self.assertIn("downstream_cascade_quarantine", meta.get("corruption_reasons", []))

    def test_ante_9_endless_safeguard_halts_and_requires_agreement(self):
        from classifiers.inference import HudRecognitionOutput

        key = {
            "Select Blind": 0,
            "End Round": 1,
            "Cash Out": 2,
            "Next Round": 3,
            "Buy Card": 4,
            "Play Hand": 5,
        }

        def _build_blind_cycle(
            frames_dict: dict[str, dict[str, Any]],
            ante_by_frame: dict[int, int],
            *,
            t_start: int,
            ante: int,
            include_next_round_after_boss: bool = True,
        ) -> int:
            t = t_start
            for b_idx in range(3):
                frames_dict[str(t)] = {"action": key["Select Blind"], "box": [100, 100, 180, 140]}
                ante_by_frame[t - 2] = ante
                t += 100
                frames_dict[str(t)] = {"action": key["End Round"], "box": [100, 100, 180, 140]}
                ante_by_frame[t - 2] = ante
                t += 100
                if b_idx == 2 and not include_next_round_after_boss:
                    break
                frames_dict[str(t)] = {"action": key["Cash Out"], "box": [400, 300, 600, 370]}
                ante_by_frame[t - 2] = ante + 1 if b_idx == 2 else ante
                t += 100
                frames_dict[str(t)] = {"action": key["Next Round"], "box": [100, 100, 200, 150]}
                ante_by_frame[t - 2] = ante + 1 if b_idx == 2 else ante
                t += 100
            return t

        def _make_run_through_ante_8_boss() -> tuple[dict[str, Any], dict[int, int], int]:
            fd: dict[str, dict[str, Any]] = {}
            ante_map: dict[int, int] = {}
            t = 100
            for a in range(1, 8):
                t = _build_blind_cycle(fd, ante_map, t_start=t, ante=a)
            t = _build_blind_cycle(
                fd, ante_map, t_start=t, ante=8, include_next_round_after_boss=False
            )
            return fd, ante_map, t

        def _run_with_hud(fd: dict[str, dict[str, Any]], hud_overrides: dict[int, int | None]):
            all_frames = {0: np.zeros((10, 10, 3), dtype=np.uint8)}
            for k_str in fd:
                fid = int(k_str)
                all_frames[fid] = np.zeros((10, 10, 3), dtype=np.uint8)
                all_frames[max(0, fid - 2)] = np.zeros((10, 10, 3), dtype=np.uint8)
                all_frames[max(0, fid - 10)] = np.zeros((10, 10, 3), dtype=np.uint8)

            def fake_shops(frames, **kwargs):
                return [
                    ShopRecognitionOutput(frame_index=f_idx, timestamp_seconds=ts)
                    for _, f_idx, ts in frames
                ]

            def fake_hands(frames, **kwargs):
                return [
                    HandRecognitionOutput(frame_index=f_idx, timestamp_seconds=ts, cards=[])
                    for _, f_idx, ts in frames
                ]

            def fake_hud(frames, fields_by_frame=None, **kwargs):
                outs = []
                for _, f_idx, ts in frames:
                    flds = list((fields_by_frame or {}).get(f_idx, ()))
                    vals: dict[str, Any] = {f: None for f in flds}
                    if "hud_ante" in vals and f_idx in hud_overrides:
                        vals["hud_ante"] = hud_overrides[f_idx]
                    outs.append(
                        HudRecognitionOutput(
                            frame_index=f_idx, timestamp_seconds=ts, values=vals
                        )
                    )
                return outs

            with (
                patch("classifiers.inference.batch_recognize_shops", side_effect=fake_shops),
                patch("classifiers.inference.batch_recognize_hands", side_effect=fake_hands),
                patch("classifiers.inference.batch_recognize_hud", side_effect=fake_hud),
            ):
                return pp._build_run_trajectory_with_plan(
                    {"key": key, "frames": fd},
                    preloaded_frames=all_frames,
                    fps=60.0,
                    expressive=True,
                )

        # 1. Endless Cash Out immediately agrees (OCR=9, sim _effective_ante=9):
        # drops Endless Cash Out, Buy Card, Next Round, and Ante 9 Play Hand.
        fd_1, hud_1, t_end = _make_run_through_ante_8_boss()
        endless_cashout_f = t_end
        fd_1[str(endless_cashout_f)] = {"action": key["Cash Out"], "box": [400, 300, 600, 370]}
        hud_1[endless_cashout_f - 2] = 9
        fd_1[str(t_end + 100)] = {"action": key["Buy Card"], "box": [300, 300, 400, 400]}
        hud_1[t_end + 98] = 9
        fd_1[str(t_end + 200)] = {"action": key["Next Round"], "box": [100, 100, 200, 150]}
        hud_1[t_end + 198] = 9
        fd_1[str(t_end + 300)] = {"action": key["Select Blind"], "box": [100, 100, 180, 140]}
        hud_1[t_end + 298] = 9
        fd_1[str(t_end + 400)] = {"action": key["Play Hand"], "box": [780, 900, 880, 950]}
        hud_1[t_end + 398] = 9

        state_1, traj_1, plan_1 = _run_with_hud(fd_1, hud_1)
        stop_1 = plan_1["diagnostics"]["stopped_at_ante_9"]
        self.assertIsNotNone(stop_1)
        self.assertEqual(stop_1["trigger_frame_id"], endless_cashout_f)
        self.assertEqual(state_1.ante, 8)
        self.assertEqual(pp._effective_ante(state_1), 8)
        self.assertEqual(traj_1.deltas[-1].action.type, bm.ActionType.END_ROUND)
        self.assertEqual(len(traj_1.deltas), stop_1["kept_steps"])
        self.assertEqual(stop_1["skipped_actions"], 5)

        # 2. Delayed OCR confirmation (None across Cash Out / Next Round, then 9 at Select Blind):
        # halts on Select Blind and rolls back tentative post-Ante-8 steps cleanly to End Round.
        fd_2, hud_2, t_end2 = _make_run_through_ante_8_boss()
        fd_2[str(t_end2)] = {"action": key["Cash Out"], "box": [400, 300, 600, 370]}
        hud_2[t_end2 - 2] = None
        fd_2[str(t_end2 + 100)] = {"action": key["Next Round"], "box": [100, 100, 200, 150]}
        hud_2[t_end2 + 98] = None
        ante9_select_f = t_end2 + 200
        fd_2[str(ante9_select_f)] = {"action": key["Select Blind"], "box": [100, 100, 180, 140]}
        hud_2[ante9_select_f - 2] = 9
        fd_2[str(t_end2 + 300)] = {"action": key["Play Hand"], "box": [780, 900, 880, 950]}
        hud_2[t_end2 + 298] = 9

        state_2, traj_2, plan_2 = _run_with_hud(fd_2, hud_2)
        stop_2 = plan_2["diagnostics"]["stopped_at_ante_9"]
        self.assertIsNotNone(stop_2)
        self.assertEqual(stop_2["trigger_frame_id"], ante9_select_f)
        self.assertEqual(stop_2["rolled_back_steps"], 2)
        self.assertEqual(state_2.ante, 8)
        self.assertEqual(pp._effective_ante(state_2), 8)
        self.assertEqual(traj_2.deltas[-1].action.type, bm.ActionType.END_ROUND)
        self.assertEqual(len(traj_2.deltas), len(traj_1.deltas))

        # 3a. Disagreement: noisy OCR spike (hud_ante=9 in Ante 1 Blind) must NOT halt or poison state.ante.
        fd_3a = {
            "100": {"action": key["Select Blind"], "box": [100, 100, 180, 140]},
            "200": {"action": key["Play Hand"], "box": [780, 900, 880, 950]},
            "300": {"action": key["End Round"], "box": [100, 100, 180, 140]},
        }
        state_3a, traj_3a, plan_3a = _run_with_hud(fd_3a, {98: 1, 200: 9, 298: 1})
        self.assertIsNone(plan_3a["diagnostics"]["stopped_at_ante_9"])
        self.assertEqual(state_3a.ante, 1)
        self.assertEqual(len(traj_3a.deltas), 4)

        # 3b. Disagreement: sim reaching 9 while OCR reports 8 must NOT halt.
        fd_3b, hud_3b, t_end3b = _make_run_through_ante_8_boss()
        fd_3b[str(t_end3b)] = {"action": key["Cash Out"], "box": [400, 300, 600, 370]}
        hud_3b[t_end3b - 2] = 8
        fd_3b[str(t_end3b + 100)] = {"action": key["Next Round"], "box": [100, 100, 200, 150]}
        hud_3b[t_end3b + 98] = 8
        fd_3b[str(t_end3b + 200)] = {"action": key["Select Blind"], "box": [100, 100, 180, 140]}
        hud_3b[t_end3b + 198] = 8
        fd_3b[str(t_end3b + 300)] = {"action": key["Play Hand"], "box": [780, 900, 880, 950]}
        hud_3b[t_end3b + 300] = 8
        state_3b, traj_3b, plan_3b = _run_with_hud(fd_3b, hud_3b)
        self.assertIsNone(plan_3b["diagnostics"]["stopped_at_ante_9"])
        self.assertEqual(state_3b.ante, 8)
        self.assertEqual(traj_3b.deltas[-1].action.type, bm.ActionType.PLAY_HAND)

    def test_cancelled_play_hand_with_ocr_score_jitter_is_dropped(self):
        from classifiers.inference import HudRecognitionOutput

        d = {
            "key": {"Select Blind": 0, "Play Hand": 1},
            "frames": {
                "100": {"action": 0, "box": [100, 100, 180, 140]},  # Select Blind -> 98
                "200": {"action": 1, "box": [705, 907, 912, 1032]},  # Cancelled Play Hand drag-off
                "270": {"action": 1, "box": [705, 907, 912, 1032]},  # Real Play Hand 1.17s later
            },
        }
        dummy_frames = {f: np.zeros((1080, 1920, 3), dtype=np.uint8) for f in (98, 200, 270)}

        def fake_shops(frames, mode="shop", **kwargs):
            return [ShopRecognitionOutput(frame_index=f_idx, timestamp_seconds=ts, mode=mode) for _, f_idx, ts in frames]

        hand_cards = [
            _make_card_res("5", "Spades", selected=True),
            _make_card_res("5", "Hearts", selected=True),
            _make_card_res("5", "Diamonds", selected=True),
            _make_card_res("3", "Spades", selected=True),
            _make_card_res("3", "Hearts", selected=True),
            _make_card_res("Ace", "Clubs", selected=False),
            _make_card_res("King", "Clubs", selected=False),
            _make_card_res("Queen", "Clubs", selected=False),
        ]

        def fake_hands(frames, **kwargs):
            return [
                HandRecognitionOutput(
                    frame_index=f_idx,
                    timestamp_seconds=ts,
                    cards=list(hand_cards) if f_idx in (200, 270) else [],
                )
                for _, f_idx, ts in frames
            ]

        def fake_hud(frames, fields_by_frame=None, **kwargs):
            outs = []
            for _, f_idx, ts in frames:
                flds = list((fields_by_frame or {}).get(f_idx, ()))
                vals = {f: None for f in flds}
                if f_idx == 200:
                    vals.update({"hud_hands": 3, "hud_round": 1, "hud_round_score": 294})
                elif f_idx == 270:
                    vals.update({"hud_hands": 3, "hud_round": 1, "hud_round_score": 296})
                outs.append(HudRecognitionOutput(frame_index=f_idx, timestamp_seconds=ts, values=vals))
            return outs

        with (
            patch("classifiers.inference.batch_recognize_shops", side_effect=fake_shops),
            patch("classifiers.inference.batch_recognize_hands", side_effect=fake_hands),
            patch("classifiers.inference.batch_recognize_hud", side_effect=fake_hud),
        ):
            state, traj, plan = pp._build_run_trajectory_with_plan(
                d, preloaded_frames=dummy_frames, fps=60.0, expressive=True
            )

        play_steps = [dl for dl in traj.deltas[1:] if dl.action and dl.action.type == bm.ActionType.PLAY_HAND]
        self.assertEqual(len(play_steps), 1)
        self.assertEqual(play_steps[0].action.frame_idx, 270)
        self.assertEqual(len(state.full_deck), 52)
        self.assertEqual(plan["diagnostics"]["unexpected_deck_additions"], [])
        self.assertFalse(any((dl.action.metadata or {}).get("corrupted") for dl in traj.deltas[1:]))

    def test_cancelled_sell_card_immediately_before_use_consumable_is_dropped(self):
        from classifiers.inference import HudRecognitionOutput

        d = {
            "key": {
                "Select Blind": 0,
                "Play Hand": 1,
                "End Round": 2,
                "Cash Out": 3,
                "Buy Card": 4,
                "Sell Card": 5,
                "Use Consumable": 6,
            },
            "frames": {
                "100": {"action": 0, "box": [100, 100, 180, 140]},  # Select Blind -> 98
                "200": {"action": 1, "box": [780, 900, 880, 950]},  # Play Hand -> 200
                "300": {"action": 2, "box": [500, 500, 600, 550]},  # End Round -> 298
                "400": {"action": 3, "box": [400, 300, 600, 370]},  # Cash Out -> 398
                "500": {"action": 4, "box": [420, 380, 520, 440]},  # Buy Pluto in Shop -> 498
                "550": {"action": 5, "box": [1760, 68, 1835, 150]},  # Cancelled SELL_CARD hover/click -> 548
                "560": {"action": 6, "box": [1747, 172, 1820, 238]},  # Actual USE_CONSUMABLE right below -> 558
            },
        }
        all_f = {98, 200, 298, 398, 498, 548, 558}
        dummy_frames = {f: np.zeros((1080, 1920, 3), dtype=np.uint8) for f in all_f}

        def fake_shops(frames, mode="shop", **kwargs):
            outs = []
            for _, f_idx, ts in frames:
                out = ShopRecognitionOutput(frame_index=f_idx, timestamp_seconds=ts, mode=mode)
                if f_idx == 498:
                    out.shop_items = [_make_shop_item("c_pluto", "Planet", [400, 220, 540, 370])]
                elif f_idx in (548, 558):
                    out.consumeables = [_make_shop_item("c_pluto", "Planet", [1610, 40, 1745, 220])]
                outs.append(out)
            return outs

        def fake_hands(frames, **kwargs):
            return [
                HandRecognitionOutput(
                    frame_index=f_idx,
                    timestamp_seconds=ts,
                    cards=[_make_card_res("Ace", "Spades", selected=True)] if f_idx == 200 else [],
                )
                for _, f_idx, ts in frames
            ]

        def fake_hud(frames, fields_by_frame=None, **kwargs):
            outs = []
            for _, f_idx, ts in frames:
                flds = list((fields_by_frame or {}).get(f_idx, ()))
                vals = {f: None for f in flds}
                if f_idx in (548, 558) and "hud_money" in vals:
                    vals["hud_money"] = 14
                outs.append(HudRecognitionOutput(frame_index=f_idx, timestamp_seconds=ts, values=vals))
            return outs

        with (
            patch("classifiers.inference.batch_recognize_shops", side_effect=fake_shops),
            patch("classifiers.inference.batch_recognize_hands", side_effect=fake_hands),
            patch("classifiers.inference.batch_recognize_hud", side_effect=fake_hud),
        ):
            state, traj, _ = pp._build_run_trajectory_with_plan(
                d, preloaded_frames=dummy_frames, deck=bm.Deck.YELLOW_DECK, fps=60.0, expressive=True
            )

        act_types = [dl.action.type for dl in traj.deltas[1:] if dl.action]
        self.assertNotIn(bm.ActionType.SELL_CARD, act_types)
        self.assertIn(bm.ActionType.USE_CONSUMABLE, act_types)
        self.assertEqual(state.hand_levels[bm.PokerHand.HIGH_CARD], 2)


if __name__ == "__main__":
    unittest.main()
