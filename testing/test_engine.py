"""
Unit tests and telemetry parity validation for Balatro GameState simulation engine.

Usage:
    python -m unittest testing/test_engine.py
    python testing/test_engine.py
"""

import json
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
except ImportError:
    import balatro_models as bm
    import gamestate as gs


def parse_lua_action(action_name: str, meta: dict, frame_idx: int, timestamp: float) -> bm.Action:
    """Maps a Lua telemetry action to a Python Action model."""
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
    }
    act_type = action_map.get(action_name, bm.ActionType.PLAY_HAND)
    return bm.Action(
        type=act_type,
        frame_idx=frame_idx,
        timestamp=timestamp,
        metadata=meta,
    )


def test_trajectory_parity(telemetry_path: Path) -> int:
    """Validates game engine parity against a Lua telemetry jsonl log.
    
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

    first_entry = json.loads(lines[0])
    first_state_data = first_entry.get("state", {})

    sim_state = gs.GameState(
        step_idx=0,
        ante=first_state_data.get("ante", 1),
        round=first_state_data.get("round", 1),
        stats=bm.RunStats(
            money=first_state_data.get("money", 4),
            reroll_cost=first_state_data.get("reroll_cost", 5),
            hands_left=first_state_data.get("hands_left", 4),
            discards_left=first_state_data.get("discards_left", 3),
        ),
    )

    trajectory = gs.RunTrajectory(keyframe_interval=10)
    errors = 0

    for i, line in enumerate(lines):
        entry = json.loads(line)
        act_name = entry.get("action", "Play Hand")
        meta = entry.get("action_metadata", {})
        lua_state = entry.get("state", {})
        step = entry.get("step", i)
        ts = entry.get("timestamp", 0.0)

        action = parse_lua_action(act_name, meta, frame_idx=step, timestamp=ts)
        sim_state, _ = gs.step_simulation(sim_state, action)
        trajectory.record_step(sim_state, action)

        if "money" in lua_state and sim_state.stats.money != lua_state["money"]:
            print(f"[Mismatch Step {step} '{act_name}'] Money: Python={sim_state.stats.money} vs Lua={lua_state['money']}")
            errors += 1

        if "reroll_cost" in lua_state and act_name == "Reroll Shop":
            if sim_state.stats.reroll_cost != lua_state["reroll_cost"]:
                print(f"[Mismatch Step {step} '{act_name}'] Reroll Cost: Python={sim_state.stats.reroll_cost} vs Lua={lua_state['reroll_cost']}")
                errors += 1

        if "deck_cards_count" in lua_state and sim_state.deck_cards_count != lua_state["deck_cards_count"]:
            print(f"[Mismatch Step {step} '{act_name}'] Deck Cards Count: Python={sim_state.deck_cards_count} vs Lua={lua_state['deck_cards_count']}")
            errors += 1

        if "discard_pile" in lua_state and len(sim_state.discard_pile) != len(lua_state["discard_pile"]):
            print(f"[Mismatch Step {step} '{act_name}'] Discard Pile Count: Python={len(sim_state.discard_pile)} vs Lua={len(lua_state['discard_pile'])}")
            errors += 1

    return errors


class TestEngine(unittest.TestCase):
    """Unit tests for the Balatro simulation engine and state transitions."""

    def test_synthetic_simulation(self):
        """Validates sequential shop reroll, Cash Out triggers, and trajectory keyframes."""
        sim_state = gs.GameState(
            stats=bm.RunStats(money=15, reroll_cost=5),
            inventory=bm.Inventory(
                jokers=[bm.Joker(name=bm.JokerName.GIFT_CARD), bm.Joker(name=bm.JokerName.JOKER)]
            ),
        )
        traj = gs.RunTrajectory(keyframe_interval=3)

        # 1. Reroll Shop
        sim_state, _ = gs.step_simulation(sim_state, bm.Action(type=bm.ActionType.REROLL_SHOP))
        traj.record_step(sim_state)
        self.assertEqual(sim_state.stats.money, 10)
        self.assertEqual(sim_state.stats.reroll_cost, 6)

        # 2. Cash out (triggers Gift Card on both jokers)
        sim_state, _ = gs.step_simulation(sim_state, bm.Action(type=bm.ActionType.CASH_OUT))
        traj.record_step(sim_state)
        self.assertEqual(sim_state.inventory.jokers[0].extra_value, 1)
        self.assertEqual(sim_state.inventory.jokers[1].extra_value, 1)
        self.assertEqual(sim_state.stats.reroll_cost, 5)  # Resets in shop

        # 3. Test trajectory keyframe seek
        recovered_s0 = traj.get_state(0)
        self.assertIsNotNone(recovered_s0)
        self.assertEqual(recovered_s0.stats.money, 10)

    def test_wee_joker_only_scores_on_active_twos(self):
        """Wee Joker must only gain chips when a 2 is part of the scoring hand."""
        wee = bm.Joker(name=bm.JokerName.WEE_JOKER)
        sim_state = gs.GameState(
            inventory=bm.Inventory(jokers=[wee]),
            hand=bm.Hand(cards=[
                bm.HandCard(card=bm.PlayingCard(rank=bm.Rank.ACE, suit=bm.Suit.SPADES), selected=True),
                bm.HandCard(card=bm.PlayingCard(rank=bm.Rank.TWO, suit=bm.Suit.HEARTS), selected=True),
            ])
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
            ])
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
            ])
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
            ])
        )
        sim_state, _ = gs.step_simulation(sim_state, bm.Action(type=bm.ActionType.DISCARD))
        self.assertAlmostEqual(ramen.state.get("x_mult", 2.0), 1.97)

    def test_dna_card_duplication_to_deck_and_hand(self):
        """DNA clones the single played card on first hand: adds to full_deck and directly to hand."""
        dna = bm.Joker(name=bm.JokerName.DNA)
        target_card = bm.PlayingCard(rank=bm.Rank.ACE, suit=bm.Suit.SPADES, enhancement=bm.Enhancement.FOIL if hasattr(bm.Enhancement, "FOIL") else bm.Enhancement.NONE)
        other_card = bm.PlayingCard(rank=bm.Rank.TEN, suit=bm.Suit.HEARTS)

        sim_state = gs.GameState(
            inventory=bm.Inventory(jokers=[dna]),
            hand=bm.Hand(cards=[
                bm.HandCard(card=target_card, selected=True),
                bm.HandCard(card=other_card, selected=False),
            ])
        )
        initial_deck_count = len(sim_state.full_deck)
        self.assertEqual(initial_deck_count, 52)

        # 1. Play first hand with 1 card (Ace of Spades)
        sim_state, _ = gs.step_simulation(sim_state, bm.Action(type=bm.ActionType.PLAY_HAND))

        # Deck should now have 53 cards, with the new Ace of Spades appended
        self.assertEqual(len(sim_state.full_deck), 53)
        self.assertEqual(sim_state.deck_cards_count, 53)
        self.assertEqual(sim_state.full_deck[-1].rank, bm.Rank.ACE)
        self.assertEqual(sim_state.full_deck[-1].suit, bm.Suit.SPADES)

        # Hand should contain: the other card + the newly copied Ace of Spades (unselected)
        self.assertEqual(len(sim_state.hand.cards), 2)
        hand_ranks = [c.card.rank for c in sim_state.hand.cards]
        self.assertIn(bm.Rank.TEN, hand_ranks)
        self.assertIn(bm.Rank.ACE, hand_ranks)

        # Verify the copied card is unselected
        ace_hc = [c for c in sim_state.hand.cards if c.card.rank == bm.Rank.ACE][0]
        self.assertFalse(ace_hc.selected)

        # Played card should be tracked in played_cards_this_round
        self.assertIn(target_card, sim_state.played_cards_this_round)

        # 2. Playing a second hand with 1 card should NOT trigger DNA again
        ace_hc.selected = True
        sim_state, _ = gs.step_simulation(sim_state, bm.Action(type=bm.ActionType.PLAY_HAND))
        self.assertEqual(len(sim_state.full_deck), 53)
        self.assertEqual(sim_state.deck_cards_count, 53)

    def test_dna_blueprint_synergy(self):
        """Blueprint copying DNA produces two cloned cards in full_deck and hand."""
        dna = bm.Joker(name=bm.JokerName.DNA)
        bp = bm.Joker(name=bm.JokerName.BLUEPRINT)
        card = bm.PlayingCard(rank=bm.Rank.KING, suit=bm.Suit.HEARTS)

        sim_state = gs.GameState(
            inventory=bm.Inventory(jokers=[bp, dna]),
            hand=bm.Hand(cards=[bm.HandCard(card=card, selected=True)])
        )
        sim_state, _ = gs.step_simulation(sim_state, bm.Action(type=bm.ActionType.PLAY_HAND))

        # Both DNA and Blueprint copy the King: +2 cards in deck and hand
        self.assertEqual(len(sim_state.full_deck), 54)
        self.assertEqual(sim_state.deck_cards_count, 54)
        self.assertEqual(len(sim_state.hand.cards), 2)
        for hc in sim_state.hand.cards:
            self.assertEqual(hc.card.rank, bm.Rank.KING)
            self.assertFalse(hc.selected)

    def test_trading_card_destruction_and_discard_pile(self):
        """Trading Card destroys single card on first discard: removes from full_deck and keeps out of discard_pile."""
        tc = bm.Joker(name=bm.JokerName.TRADING_CARD)
        canio = bm.Joker(name=bm.JokerName.CANIO)
        face_card = bm.PlayingCard(rank=bm.Rank.QUEEN, suit=bm.Suit.DIAMONDS)
        other_card = bm.PlayingCard(rank=bm.Rank.TWO, suit=bm.Suit.CLUBS)

        sim_state = gs.GameState(
            inventory=bm.Inventory(jokers=[tc, canio]),
            hand=bm.Hand(cards=[
                bm.HandCard(card=face_card, selected=True),
                bm.HandCard(card=other_card, selected=False),
            ])
        )
        initial_money = sim_state.stats.money

        # Discard the single Queen
        sim_state, _ = gs.step_simulation(sim_state, bm.Action(type=bm.ActionType.DISCARD))

        # Money increases by 3
        self.assertEqual(sim_state.stats.money, initial_money + 3)
        # Queen is destroyed: decremented deck count and removed from full_deck
        self.assertEqual(sim_state.deck_cards_count, 51)
        self.assertNotIn(face_card, sim_state.full_deck)
        # Queen must NOT be in the discard pile
        self.assertNotIn(face_card, sim_state.discard_pile)
        self.assertEqual(len(sim_state.discard_pile), 0)
        # Queen was a face card: Canio triggered and gained +1x
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
            ])
        )

        # Discard c1
        sim_state, _ = gs.step_simulation(sim_state, bm.Action(type=bm.ActionType.DISCARD))
        self.assertIn(c1, sim_state.discard_pile)
        self.assertEqual(len(sim_state.discard_pile), 1)

        # Play c2
        sim_state.hand.cards[0].selected = True  # c2
        sim_state, _ = gs.step_simulation(sim_state, bm.Action(type=bm.ActionType.PLAY_HAND))
        self.assertIn(c2, sim_state.played_cards_this_round)
        self.assertEqual(len(sim_state.played_cards_this_round), 1)

        # Selecting next blind clears discard_pile and played_cards_this_round
        sim_state, _ = gs.step_simulation(sim_state, bm.Action(type=bm.ActionType.SELECT_BLIND))
        self.assertEqual(len(sim_state.discard_pile), 0)
        self.assertEqual(len(sim_state.played_cards_this_round), 0)

    def test_sixth_sense_destruction(self):
        """Sixth Sense destroys a single played 6 on first hand from hand and deck."""
        sixth = bm.Joker(name=bm.JokerName.SIXTH_SENSE)
        c6 = bm.PlayingCard(rank=bm.Rank.SIX, suit=bm.Suit.SPADES)

        sim_state = gs.GameState(
            inventory=bm.Inventory(jokers=[sixth]),
            hand=bm.Hand(cards=[bm.HandCard(card=c6, selected=True)])
        )
        sim_state, _ = gs.step_simulation(sim_state, bm.Action(type=bm.ActionType.PLAY_HAND))

        # 6 is destroyed: removed from full_deck, hand, and not in played_cards_this_round
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
        initial_money = sim_state.stats.money

        # Standard deck has 4 nines
        sim_state, _ = gs.step_simulation(sim_state, bm.Action(type=bm.ActionType.CASH_OUT))
        self.assertEqual(sim_state.stats.money, initial_money + 4)

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
        ]
        for name, expected_type in action_cases:
            action = parse_lua_action(name, {}, frame_idx=1, timestamp=0.5)
            self.assertEqual(action.type, expected_type)
            self.assertEqual(action.frame_idx, 1)
            self.assertEqual(action.timestamp, 0.5)

    def test_telemetry_file_parity_if_present(self):
        """Runs full parity validation if run_telemetry.jsonl is present."""
        telemetry_paths = [
            REPO_ROOT / "run_telemetry.jsonl",
            REPO_ROOT / "output" / "run_telemetry.jsonl",
        ]
        for p in telemetry_paths:
            if p.exists():
                errors = test_trajectory_parity(p)
                self.assertEqual(errors, 0, f"Encountered {errors} telemetry parity errors in {p}")


def run_synthetic_test() -> None:
    """Convenience function running the synthetic test directly."""
    test = TestEngine()
    test.test_synthetic_simulation()
    print("Synthetic integration test passed successfully!")


if __name__ == "__main__":
    unittest.main()
