"""Round-trip and sparse-delta tests for `game/trajectory_codec.py`."""

from __future__ import annotations

import json
import unittest

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
    HandCard,
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
    Spectral,
    SpectralName,
    Stake,
    Suit,
    Tarot,
    TarotName,
    Voucher,
    VoucherName,
)
from game.gamestate import (
    _TRACKED_STATE_FIELDS,
    GameState,
    RunTrajectory,
    on_action,
)
from game.trajectory_codec import (
    rebuild_gamestate,
    rebuild_trajectory,
    serialize_trajectory,
)


def _assert_states_equivalent(tc: unittest.TestCase, expected: GameState, actual: GameState) -> None:
    """Asserts full field-by-field equivalence between two GameState instances."""
    tc.assertEqual(actual.step_idx, expected.step_idx)
    tc.assertEqual(actual.frame_idx, expected.frame_idx)
    tc.assertAlmostEqual(actual.timestamp, expected.timestamp, places=9)

    for attr in _TRACKED_STATE_FIELDS:
        tc.assertEqual(
            getattr(actual, attr),
            getattr(expected, attr),
            msg=f"Mismatch on GameState.{attr} at step {expected.step_idx}",
        )

    tc.assertEqual(actual.inventory.joker_slots, expected.inventory.joker_slots)
    tc.assertEqual(actual.inventory.consumable_slots, expected.inventory.consumable_slots)
    tc.assertEqual(actual.inventory.jokers, expected.inventory.jokers)
    for ej, aj in zip(expected.inventory.jokers, actual.inventory.jokers):
        tc.assertEqual(aj.id, ej.id)
        tc.assertEqual(aj.state, ej.state)
    tc.assertEqual(actual.inventory.consumables, expected.inventory.consumables)

    tc.assertEqual(actual.hand.max_size, expected.hand.max_size)
    tc.assertEqual(actual.hand.cards, expected.hand.cards)
    for ehc, ahc in zip(expected.hand.cards, actual.hand.cards):
        tc.assertEqual(ahc.card.id, ehc.card.id)
        tc.assertEqual(ahc.card.debuffed, ehc.card.debuffed)
        tc.assertEqual(ahc.card.destroyed, ehc.card.destroyed)
        tc.assertEqual(ahc._real_card, ehc._real_card)


class TestTrajectoryCodec(unittest.TestCase):
    """Verifies sparse delta serialization and full GameState reconstruction."""

    def _step(
        self,
        state: GameState,
        traj: RunTrajectory,
        action: Action,
        mutate=None,
    ) -> None:
        state.frame_idx = action.frame_idx
        state.timestamp = action.timestamp
        if mutate is not None:
            mutate(state)
        on_action(state, action)
        traj.record_step(state, action)

    def test_sparse_delta_omits_unchanged_fields_and_captures_expressive_details(self):
        state = GameState(
            seed="SPARSE1",
            deck=Deck.BLUE_DECK,
            stake=Stake.GOLD,
            ante=1,
            round=1,
            current_blind=BlindName.SMALL,
            location=Location.SELECT,
            apply_deck_stake_on_init=True,
        )
        traj = RunTrajectory()
        traj.record_step(state, action=None)

        # Step 1: Select Blind
        self._step(state, traj, Action(type=ActionType.SELECT_BLIND, frame_idx=60, timestamp=1.0))

        # Step 2: Populate hand & play a pair; reroll_cost does NOT change here.
        c1 = PlayingCard(
            rank=Rank.ACE,
            suit=Suit.SPADES,
            enhancement=Enhancement.MULT,
            seal=Seal.RED,
            edition=Edition.FOIL,
            id=101,
        )
        c2 = PlayingCard(rank=Rank.ACE, suit=Suit.HEARTS, id=102)
        c3 = PlayingCard(rank=Rank.KING, suit=Suit.CLUBS, id=103)
        hand_cards = [
            HandCard(card=c1, selected=True, slot_uid=1),
            HandCard(card=c2, selected=True, slot_uid=2),
            HandCard(card=c3, selected=False, slot_uid=3),
        ]

        def _set_hand_and_score(st: GameState) -> None:
            st.hand.cards = list(hand_cards)
            st.current_chips = 180

        self._step(
            state,
            traj,
            Action(
                type=ActionType.PLAY_HAND,
                frame_idx=180,
                timestamp=3.0,
                selected_cards=[c1, c2],
            ),
            mutate=_set_hand_and_score,
        )

        # Step 3: End round -> Cash out -> Shop with sticker/edition joker
        self._step(
            state,
            traj,
            Action(type=ActionType.END_ROUND, frame_idx=240, timestamp=4.0),
            mutate=lambda st: setattr(st, "current_chips", 350),
        )
        self._step(
            state,
            traj,
            Action(type=ActionType.CASH_OUT, frame_idx=300, timestamp=5.0),
        )

        sticker_joker = Joker(
            name=JokerName.BLUEPRINT,
            edition=Edition.POLYCHROME,
            eternal=True,
            perishable=True,
            perish_tally=3,
            rental=True,
            extra_value=4,
            state={"copied": "Joker"},
            id=77,
        )
        plain_joker = Joker(name=JokerName.JOKER, id=78)
        self._step(
            state,
            traj,
            Action(type=ActionType.BUY_CARD, frame_idx=360, timestamp=6.0, target_idx=0),
            mutate=lambda st: setattr(st, "shop_cards", [sticker_joker, plain_joker]),
        )

        payload = json.loads(json.dumps(serialize_trajectory(state, traj)))
        steps = payload["steps"]

        # Step 2 (PLAY_HAND): action.selected_cards lists played cards (with edition/seal/enhancement),
        # hand_diff lists remaining cards in hand, and unchanged reroll_cost/ante are NOT written.
        play_step = steps[2]
        self.assertIn("hand_diff", play_step)
        self.assertEqual(len(play_step["hand_diff"]), 1)
        self.assertEqual(len(play_step["action"]["selected_cards"]), 2)
        self.assertEqual(play_step["action"]["selected_cards"][0]["edition"], "foil")
        self.assertEqual(play_step["action"]["selected_cards"][0]["seal"], "red")
        self.assertEqual(play_step["action"]["selected_cards"][0]["enhancement"], "mult")
        self.assertNotIn("reroll_cost", play_step.get("field_diffs", {}))
        self.assertNotIn("ante", play_step)

        # Step 5 (BUY_CARD): jokers_diff lists sticker/edition fields on sticker_joker, omits defaults on plain_joker.
        buy_step = steps[5]
        self.assertIn("jokers_diff", buy_step)
        j0 = buy_step["jokers_diff"][0]
        self.assertEqual(j0["name"], "Blueprint")
        self.assertEqual(j0["edition"], "polychrome")
        self.assertTrue(j0["eternal"])
        self.assertTrue(j0["perishable"])
        self.assertEqual(j0["perish_tally"], 3)
        self.assertTrue(j0["rental"])
        self.assertEqual(j0["extra_value"], 4)
        self.assertEqual(j0["state"], {"copied": "Joker"})

        # Verify full round-trip equivalence across every step
        rebuilt_traj = rebuild_trajectory(payload)
        self.assertEqual(len(rebuilt_traj.deltas), len(traj.deltas))
        for i in range(len(traj.deltas)):
            _assert_states_equivalent(self, traj.get_state(i), rebuilt_traj.get_state(i))

        rebuilt_final = rebuild_gamestate(payload)
        _assert_states_equivalent(self, state, rebuilt_final)

    def test_full_multi_ante_trajectory_roundtrip_equivalence(self):
        state = GameState(
            seed="ROUNDTRIP",
            deck=Deck.PLASMA_DECK,
            stake=Stake.PURPLE,
            ante=1,
            round=1,
            current_blind=BlindName.SMALL,
            location=Location.SELECT,
            apply_deck_stake_on_init=True,
        )
        traj = RunTrajectory(keyframe_interval=4)
        traj.record_step(state, action=None)

        # Skip small blind with a tag
        state.small_blind_tag = SkipTag.COUPON_TAG
        self._step(state, traj, Action(type=ActionType.SKIP_BLIND, frame_idx=30, timestamp=0.5))

        # Select big blind -> discard -> play -> end round -> cash out
        self._step(state, traj, Action(type=ActionType.SELECT_BLIND, frame_idx=60, timestamp=1.0))

        d_card = PlayingCard(rank=Rank.TWO, suit=Suit.DIAMONDS, id=1)
        p_card = PlayingCard(rank=Rank.QUEEN, suit=Suit.HEARTS, enhancement=Enhancement.GLASS, id=2)
        fd_card = HandCard(
            card=PlayingCard(rank=Rank.NONE, suit=Suit.NONE),
            facing_back=True,
            slot_uid=9,
        )
        self._step(
            state,
            traj,
            Action(type=ActionType.DISCARD, frame_idx=90, timestamp=1.5, selected_cards=[d_card]),
            mutate=lambda st: setattr(
                st.hand,
                "cards",
                [HandCard(card=d_card, selected=True, slot_uid=1), HandCard(card=p_card, slot_uid=2), fd_card],
            ),
        )
        self._step(
            state,
            traj,
            Action(type=ActionType.PLAY_HAND, frame_idx=120, timestamp=2.0, selected_cards=[p_card]),
            mutate=lambda st: setattr(st, "current_chips", 1000),
        )
        self._step(
            state,
            traj,
            Action(type=ActionType.END_ROUND, frame_idx=150, timestamp=2.5),
            mutate=lambda st: setattr(st, "boss_blind", BlindName.THE_HOOK),
        )
        self._step(state, traj, Action(type=ActionType.CASH_OUT, frame_idx=180, timestamp=3.0))

        # Shop: buy consumable, reroll, redeem voucher, open pack, use in pack, next round
        tarot = Tarot(name=TarotName.FOOL, edition=Edition.NEGATIVE)
        planet = Planet(name=PlanetName.JUPITER)
        spectral = Spectral(name=SpectralName.ECTOPLASM)
        voucher = Voucher(name=VoucherName.OVERSTOCK)
        pack = BoosterPack(
            booster_type=BoosterType.ARCANA,
            size=BoosterSize.JUMBO,
            cost=6,
            choose=1,
            extra=5,
            couponed=True,
        )
        self._step(
            state,
            traj,
            Action(type=ActionType.BUY_CARD, frame_idx=210, timestamp=3.5, target_idx=0),
            mutate=lambda st: (
                setattr(st, "shop_cards", [tarot]),
                setattr(st, "shop_vouchers", [voucher]),
                setattr(st, "shop_packs", [pack]),
            ),
        )
        self._step(state, traj, Action(type=ActionType.REROLL_SHOP, frame_idx=240, timestamp=4.0))
        self._step(state, traj, Action(type=ActionType.REDEEM_VOUCHER, frame_idx=270, timestamp=4.5, target_idx=0))
        self._step(state, traj, Action(type=ActionType.OPEN_PACK, frame_idx=300, timestamp=5.0, target_idx=0))
        self._step(
            state,
            traj,
            Action(type=ActionType.USE_IN_PACK, frame_idx=330, timestamp=5.5, target_idx=0),
            mutate=lambda st: setattr(st, "pack_cards", [planet, spectral]),
        )
        self._step(state, traj, Action(type=ActionType.USE_CONSUMABLE, frame_idx=360, timestamp=6.0, target_idx=0))
        self._step(state, traj, Action(type=ActionType.NEXT_ROUND, frame_idx=390, timestamp=6.5))

        traj.impute_retroactive_boss_blinds()

        json_str = json.dumps(serialize_trajectory(state, traj))
        reloaded = json.loads(json_str)

        rebuilt_traj = rebuild_trajectory(reloaded, keyframe_interval=4)
        self.assertEqual(len(rebuilt_traj.deltas), len(traj.deltas))
        for i in range(len(traj.deltas)):
            _assert_states_equivalent(self, traj.get_state(i), rebuilt_traj.get_state(i))

        # Check step_idx selection on rebuild_gamestate
        _assert_states_equivalent(self, traj.get_state(3), rebuild_gamestate(reloaded, step_idx=3))
        _assert_states_equivalent(self, state, rebuild_gamestate(reloaded))


if __name__ == "__main__":
    unittest.main()
