"""Tests for `tools/compare_trajectories.py` non-cascading trajectory comparison."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from game.balatro_models import (
    Action,
    ActionType,
    Deck,
    Enhancement,
    Hand,
    HandCard,
    Inventory,
    Joker,
    JokerName,
    Location,
    PlayingCard,
    PokerHand,
    Rank,
    Seal,
    Stake,
    Suit,
)
from game.gamestate import GameState, RunTrajectory
from tools.compare_trajectories import compare_trajectories


def _card(cid: int, rank: Rank, suit: Suit, **kwargs) -> PlayingCard:
    return PlayingCard(id=cid, rank=rank, suit=suit, **kwargs)


def _base_state(*, money: int = 10, seed: str = "CMPSEED") -> GameState:
    cards = [
        _card(1, Rank.ACE, Suit.SPADES),
        _card(2, Rank.KING, Suit.CLUBS),
        _card(3, Rank.QUEEN, Suit.HEARTS),
        _card(4, Rank.TEN, Suit.DIAMONDS),
    ]
    return GameState(
        step_idx=0,
        frame_idx=0,
        timestamp=0.0,
        seed=seed,
        deck=Deck.RED_DECK,
        stake=Stake.WHITE,
        ante=1,
        round=1,
        location=Location.SELECT,
        money=money,
        hands_left=4,
        discards_left=3,
        reroll_cost=5,
        full_deck=[PlayingCard(**vars(c)) for c in cards],
        apply_deck_stake_on_init=False,
    )


class TestTrajectoryComparator(unittest.TestCase):
    """Validates sequence alignment and non-cascading delta comparison."""

    def test_identical_trajectories_zero_errors(self) -> None:
        st = _base_state()
        traj = RunTrajectory()
        traj.record_step(st)

        st = st.copy()
        st.location = Location.BLIND
        st.frame_idx = 60
        st.target_chips = 300
        traj.record_step(st, Action(type=ActionType.SELECT_BLIND, frame_idx=60, timestamp=1.0))

        st = st.copy()
        st.frame_idx = 120
        st.hands_left = 3
        st.current_chips = 350
        st.hand_play_counts[PokerHand.PAIR] = 1
        traj.record_step(
            st,
            Action(
                type=ActionType.PLAY_HAND,
                frame_idx=120,
                timestamp=2.0,
                selected_cards=[_card(1, Rank.ACE, Suit.SPADES)],
            ),
        )

        report = compare_trajectories(traj, copy.deepcopy(traj))
        self.assertEqual(report["summary"]["error_step_count"], 0)
        self.assertEqual(report["summary"]["total_discrepancies"], 0)
        self.assertEqual(report["summary"]["error_step_indices"], [])

    def test_spurious_deck_card_does_not_cascade(self) -> None:
        def _build_run(*, inject_extra_card: bool) -> RunTrajectory:
            st = _base_state()
            traj = RunTrajectory()
            traj.record_step(st)  # 0

            st = st.copy()
            st.location = Location.BLIND
            st.frame_idx = 10
            st.target_chips = 300
            traj.record_step(st, Action(type=ActionType.SELECT_BLIND, frame_idx=10))  # 1

            st = st.copy()
            st.frame_idx = 20
            st.hands_left = 3
            st.current_chips = 120
            if inject_extra_card:
                st.full_deck.append(_card(99, Rank.QUEEN, Suit.CLUBS, seal=Seal.GOLD))
            traj.record_step(
                st,
                Action(
                    type=ActionType.PLAY_HAND,
                    frame_idx=20,
                    selected_cards=[_card(1, Rank.ACE, Suit.SPADES)],
                ),
            )  # 2

            st = st.copy()
            st.frame_idx = 30
            st.hands_left = 2
            st.current_chips = 350
            traj.record_step(
                st,
                Action(
                    type=ActionType.PLAY_HAND,
                    frame_idx=30,
                    selected_cards=[_card(2, Rank.KING, Suit.CLUBS)],
                ),
            )  # 3

            st = st.copy()
            st.frame_idx = 40
            st.round = 2
            traj.record_step(st, Action(type=ActionType.END_ROUND, frame_idx=40))  # 4

            st = st.copy()
            st.frame_idx = 50
            st.location = Location.SHOP
            st.money = 15
            traj.record_step(st, Action(type=ActionType.CASH_OUT, frame_idx=50))  # 5
            return traj

        gt = _build_run(inject_extra_card=False)
        actual = _build_run(inject_extra_card=True)

        report = compare_trajectories(actual, gt)
        self.assertEqual(report["summary"]["error_step_indices"], [2])
        step2_discs = report["steps_by_actual_idx"]["2"]
        self.assertEqual(len(step2_discs), 1)
        self.assertEqual(step2_discs[0]["category"], "deck")
        for later in ("3", "4", "5"):
            self.assertNotIn(later, report["steps_by_actual_idx"])

    def test_money_offset_and_cash_out_interest_do_not_cascade(self) -> None:
        def _build_run(*, miss_buy_cost: bool) -> RunTrajectory:
            st = _base_state(money=15)
            st.location = Location.SHOP
            traj = RunTrajectory()
            traj.record_step(st)  # 0

            # Step 1: Buy Card (-$5 expected; actual stays $15 if missed)
            st = st.copy()
            st.frame_idx = 10
            if not miss_buy_cost:
                st.money = 10
            traj.record_step(
                st,
                Action(
                    type=ActionType.BUY_CARD,
                    frame_idx=10,
                    metadata={"target_card": "The Empress"},
                ),
            )  # 1

            # Step 2: Leave shop
            st = st.copy()
            st.frame_idx = 20
            st.location = Location.SELECT
            traj.record_step(st, Action(type=ActionType.NEXT_ROUND, frame_idx=20))  # 2

            # Step 3: Enter blind
            st = st.copy()
            st.frame_idx = 30
            st.location = Location.BLIND
            traj.record_step(st, Action(type=ActionType.SELECT_BLIND, frame_idx=30))  # 3

            # Step 4: Cash Out -> payout $3 + interest (15//5=3 vs 10//5=2)
            st = st.copy()
            st.frame_idx = 40
            st.location = Location.SHOP
            interest = st.money // 5
            st.money += 3 + interest
            traj.record_step(st, Action(type=ActionType.CASH_OUT, frame_idx=40))  # 4
            return traj

        gt = _build_run(miss_buy_cost=False)
        actual = _build_run(miss_buy_cost=True)

        report = compare_trajectories(actual, gt)
        self.assertEqual(report["summary"]["error_step_indices"], [1])
        self.assertEqual(report["steps_by_actual_idx"]["1"][0]["category"], "economy")
        self.assertGreaterEqual(report["summary"]["suppressed_cascades"], 1)

        report_with_casc = compare_trajectories(actual, gt, include_cascaded=True)
        self.assertEqual(report_with_casc["summary"]["error_step_indices"], [1, 4])
        self.assertEqual(report_with_casc["steps_by_actual_idx"]["4"][0]["kind"], "interest_cascade")

    def test_missed_enhancement_does_not_cascade_into_later_hands(self) -> None:
        def _build_run(*, actual_mode: bool) -> RunTrajectory:
            st = _base_state()
            st.location = Location.BLIND
            st.hand = Hand(
                cards=[
                    HandCard(card=_card(1, Rank.ACE, Suit.SPADES), slot_uid=101),
                    HandCard(card=_card(2, Rank.KING, Suit.CLUBS), slot_uid=102),
                    HandCard(card=_card(3, Rank.QUEEN, Suit.HEARTS), slot_uid=103),
                ]
            )
            traj = RunTrajectory()
            traj.record_step(st)  # 0

            # Step 1: Use Consumable (Empress) -> GT enhances Ace of Spades to Mult; actual misses enhancement
            st = st.copy()
            st.frame_idx = 10
            if not actual_mode:
                st.full_deck[0].enhancement = Enhancement.MULT
                st.hand.cards[0].card.enhancement = Enhancement.MULT
            traj.record_step(
                st,
                Action(
                    type=ActionType.USE_CONSUMABLE,
                    frame_idx=10,
                    selected_cards=[_card(1, Rank.ACE, Suit.SPADES)],
                    metadata={"target_card": "The Empress"},
                ),
            )  # 1

            # Step 2: Play King of Clubs; drifted Ace of Spades remains visible in hand -> MUST suppress drift
            st = st.copy()
            st.frame_idx = 20
            st.hands_left = 3
            a_card = _card(
                1,
                Rank.ACE,
                Suit.SPADES,
                enhancement=Enhancement.NONE if actual_mode else Enhancement.MULT,
            )
            st.hand = Hand(
                cards=[
                    HandCard(card=a_card, slot_uid=101),
                    HandCard(card=_card(3, Rank.QUEEN, Suit.HEARTS), slot_uid=103),
                ]
            )
            traj.record_step(
                st,
                Action(
                    type=ActionType.PLAY_HAND,
                    frame_idx=20,
                    selected_cards=[_card(2, Rank.KING, Suit.CLUBS)],
                ),
            )  # 2

            # Step 3: Discard Queen of Hearts, but in hand actual misreads 10 of Diamonds as Jack of Diamonds
            st = st.copy()
            st.frame_idx = 30
            st.discards_left = 2
            st.hand = Hand(
                cards=[
                    HandCard(card=PlayingCard(**vars(a_card)), slot_uid=101),
                    HandCard(
                        card=_card(4, Rank.JACK if actual_mode else Rank.TEN, Suit.DIAMONDS),
                        slot_uid=104,
                    ),
                ]
            )
            traj.record_step(
                st,
                Action(
                    type=ActionType.DISCARD,
                    frame_idx=30,
                    selected_cards=[_card(3, Rank.QUEEN, Suit.HEARTS)],
                ),
            )  # 3
            return traj

        gt = _build_run(actual_mode=False)
        actual = _build_run(actual_mode=True)

        report = compare_trajectories(actual, gt)
        # Step 1 flagged at source; Step 2 suppressed (0 cascade); Step 3 flagged ONLY for J♦ vs 10♦ misread
        self.assertEqual(report["summary"]["error_step_indices"], [1, 3])
        self.assertIn("deck", [d["category"] for d in report["steps_by_actual_idx"]["1"]])
        self.assertEqual([d["category"] for d in report["steps_by_actual_idx"]["3"]], ["hand"])
        self.assertGreaterEqual(report["summary"]["suppressed_cascades"], 1)

    def test_missing_and_extra_actions_align_without_shifting_later_steps(self) -> None:
        def _step_play(st: GameState, frame: int, cid: int, rank: Rank, suit: Suit) -> tuple[GameState, Action]:
            nxt = st.copy()
            nxt.frame_idx = frame
            nxt.hands_left = (nxt.hands_left or 4) - 1
            return nxt, Action(
                type=ActionType.PLAY_HAND,
                frame_idx=frame,
                selected_cards=[_card(cid, rank, suit)],
            )

        # Build 8-step GT run
        gt = RunTrajectory()
        st = _base_state()
        gt.record_step(st)  # 0

        st = st.copy(); st.location = Location.BLIND; st.frame_idx = 10
        gt.record_step(st, Action(type=ActionType.SELECT_BLIND, frame_idx=10))  # 1

        st = st.copy(); st.frame_idx = 20; st.discards_left = 2
        gt.record_step(st, Action(type=ActionType.DISCARD, frame_idx=20, selected_cards=[_card(4, Rank.TEN, Suit.DIAMONDS)]))  # 2 (will be missed)

        st, act3 = _step_play(st, 30, 1, Rank.ACE, Suit.SPADES)
        gt.record_step(st, act3)  # 3

        st, act4 = _step_play(st, 40, 2, Rank.KING, Suit.CLUBS)
        gt.record_step(st, act4)  # 4

        st = st.copy(); st.frame_idx = 50; st.round = 2
        gt.record_step(st, Action(type=ActionType.END_ROUND, frame_idx=50))  # 5

        st = st.copy(); st.frame_idx = 60; st.location = Location.SHOP; st.money = 16
        gt.record_step(st, Action(type=ActionType.CASH_OUT, frame_idx=60))  # 6

        st = st.copy(); st.frame_idx = 70; st.money = 12; st.inventory.jokers.append(Joker(name=JokerName.JOKER, id=11))
        gt.record_step(st, Action(type=ActionType.BUY_CARD, frame_idx=70, metadata={"target_card": "Joker"}))  # 7

        # Actual misses GT step 2 (Discard) and inserts an extra Reroll Shop between Cash Out and Buy Card
        actual = RunTrajectory()
        a = _base_state()
        actual.record_step(a)  # 0

        a = a.copy(); a.location = Location.BLIND; a.frame_idx = 10
        actual.record_step(a, Action(type=ActionType.SELECT_BLIND, frame_idx=10))  # 1

        # (GT step 2 Discard missed here!)
        a, a_act2 = _step_play(a, 30, 1, Rank.ACE, Suit.SPADES)
        actual.record_step(a, a_act2)  # 2 -> aligns with GT 3

        a, a_act3 = _step_play(a, 40, 2, Rank.KING, Suit.CLUBS)
        actual.record_step(a, a_act3)  # 3 -> aligns with GT 4

        a = a.copy(); a.frame_idx = 50; a.round = 2
        actual.record_step(a, Action(type=ActionType.END_ROUND, frame_idx=50))  # 4 -> aligns with GT 5

        a = a.copy(); a.frame_idx = 60; a.location = Location.SHOP; a.money = 16
        actual.record_step(a, Action(type=ActionType.CASH_OUT, frame_idx=60))  # 5 -> aligns with GT 6

        a = a.copy(); a.frame_idx = 65; a.money = 11
        actual.record_step(a, Action(type=ActionType.REROLL_SHOP, frame_idx=65))  # 6 -> EXTRA action!

        a = a.copy(); a.frame_idx = 70; a.money = 7; a.inventory.jokers.append(Joker(name=JokerName.JOKER, id=21))
        actual.record_step(a, Action(type=ActionType.BUY_CARD, frame_idx=70, metadata={"target_card": "Joker"}))  # 7 -> aligns with GT 7

        report = compare_trajectories(actual, gt)
        self.assertEqual(report["summary"]["missing_actions"], 1)
        self.assertEqual(report["summary"]["extra_actions"], 1)
        # Missing Discard attaches to step 1 (anchored before actual step 2); extra Reroll is step 6
        self.assertEqual(report["summary"]["error_step_indices"], [1, 6])
        self.assertEqual(report["steps_by_actual_idx"]["1"][0]["kind"], "missing_action")
        self.assertEqual(report["steps_by_actual_idx"]["6"][0]["kind"], "extra_action")
        # Aligned steps 2, 3, 4, 5, 7 produce ZERO errors!
        for ok_step in ("2", "3", "4", "5", "7"):
            self.assertNotIn(ok_step, report["steps_by_actual_idx"])

    def test_inventory_and_poker_hand_diffs_do_not_cascade(self) -> None:
        def _build_run(*, miss_updates: bool) -> RunTrajectory:
            st = _base_state()
            st.location = Location.SHOP
            traj = RunTrajectory()
            traj.record_step(st)  # 0

            st = st.copy()
            st.frame_idx = 10
            traj.record_step(st, Action(type=ActionType.REROLL_SHOP, frame_idx=10))  # 1

            # Step 2: Buy Joker (actual misses adding to inventory)
            st = st.copy()
            st.frame_idx = 20
            if not miss_updates:
                st.inventory.jokers.append(Joker(name=JokerName.JOKER, id=1))
            traj.record_step(
                st,
                Action(type=ActionType.BUY_CARD, frame_idx=20, metadata={"target_card": "Joker"}),
            )  # 2

            # Step 3: Use Planet Jupiter (actual misses level-up)
            st = st.copy()
            st.frame_idx = 30
            if not miss_updates:
                st.hand_levels[PokerHand.FLUSH] = 2
            traj.record_step(
                st,
                Action(type=ActionType.USE_CONSUMABLE, frame_idx=30, metadata={"target_card": "Jupiter"}),
            )  # 3

            # Step 4 & 5: Normal transitions with no more diffs
            st = st.copy()
            st.frame_idx = 40
            st.location = Location.BLIND
            traj.record_step(st, Action(type=ActionType.SELECT_BLIND, frame_idx=40))  # 4

            st = st.copy()
            st.frame_idx = 50
            st.hands_left = 3
            st.hand_play_counts[PokerHand.HIGH_CARD] = 1
            traj.record_step(
                st,
                Action(
                    type=ActionType.PLAY_HAND,
                    frame_idx=50,
                    selected_cards=[_card(1, Rank.ACE, Suit.SPADES)],
                ),
            )  # 5
            return traj

        gt = _build_run(miss_updates=False)
        actual = _build_run(miss_updates=True)

        report = compare_trajectories(actual, gt)
        self.assertEqual(report["summary"]["error_step_indices"], [2, 3])
        self.assertEqual(report["steps_by_actual_idx"]["2"][0]["category"], "inventory")
        self.assertEqual(report["steps_by_actual_idx"]["3"][0]["category"], "poker_hands")

    def test_real_telemetry_jsonl_reconstruction_and_single_injection(self) -> None:
        gt_json_path = REPO_ROOT / "output" / "trajectories" / "run_A8CQP141_14steps_gt_trajectory.json"
        tel_path = REPO_ROOT / "assets" / "local" / "run_telemetry.jsonl"
        if not gt_json_path.is_file() or not tel_path.is_file():
            self.skipTest("Sample A8CQP141 telemetry/trajectory fixture not present.")

        report = compare_trajectories(gt_json_path, tel_path)
        self.assertEqual(report["summary"]["error_step_count"], 0)
        self.assertEqual(report["summary"]["total_discrepancies"], 0)

        # Mutate 1 step's full_deck (+1 Ace♠ Red Seal) and persist across all later steps: only 1 step flagged!
        raw_data = json.loads(gt_json_path.read_text(encoding="utf-8"))
        mutated_traj = RunTrajectory()
        from game.trajectory_codec import rebuild_trajectory

        orig_traj = rebuild_trajectory(raw_data)
        for idx in range(len(orig_traj.deltas)):
            st = orig_traj.get_state(idx).copy()
            if idx >= 3 and st.full_deck is not None:
                st.full_deck.append(_card(999, Rank.ACE, Suit.SPADES, seal=Seal.RED))
            act = orig_traj.deltas[idx].action if idx > 0 else None
            mutated_traj.record_step(st, act)

        mut_report = compare_trajectories(mutated_traj, tel_path)
        self.assertEqual(mut_report["summary"]["error_step_indices"], [3])
        self.assertEqual(mut_report["steps_by_actual_idx"]["3"][0]["category"], "deck")


if __name__ == "__main__":
    unittest.main()
