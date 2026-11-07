"""Comprehensive Unit Tests for Balatro Deterministic Simulation Engine.

Verifies:
1. Exact bit-level parity of pseudohash, pseudoseed, Tw223, and LCG step against
   reference test vectors from Immolate and LuaJIT PRNG.
2. Reproducibility for seed 'HELLO' across multiple runs.
3. Step-dependent state advancement and PRNG channel consumption.
4. Certificate joker card & seal generation (both ante-scoped and `balatro_source` parity).
5. Lucky Cards (+20 Mult and +$20 Cash) probability mechanics and Oops! All 6s scaling.
6. Mail-in Rebate dynamic target selection and discard evaluations.
7. 1-in-N outcome resolvers (Wheel of Fortune, Space Joker, Bloodstone, Glass, Business Card).
8. Deck mutations and exact `sort_id` ordering (`C_2..C_T, D_2..D_T, H_2..H_T, S_2..S_T`).
9. Castle suit selection per round (`reset_castle_card`, deck composition & Stone Card filtering).
10. Ancient Joker suit selection per round (`reset_ancient_card`, 4-suit init & 3-suit non-repeat).
11. The Idol card selection per round (`reset_idol_card`, deck composition & Stone Card filtering).
12. To Do List poker hand selection (`'to_do'`, initial spawn vs end-of-round non-repeating reset, secret hand unlocks).
13. Cartomancer (`'Tarotcar{ante}'`) and Vagabond (`'Tarotvag{ante}'`) Tarot generation, active Tarot blocking, and Showman override.
14. Marble Joker / Stone Joker (`'marb_fr'`) hidden suit & rank generation and reveal upon re-enhancement.
15. Boss Blind queue (`'boss'`), including `min_ante` locks, Ante 8 Showdown bosses, boss rerolls, Hieroglyph/Petroglyph (`-1 ante` vouchers at Ante 1 and Ante 8), and `bosses_used` cycle wrap-around.
16. Skip Tag queue (`'Tag{ante}'`), including Ante 1 `min_ante = 2` resampling, `-1 ante` vouchers, and Orbital Tag (`'orbital'`) poker hand selection.
17. Voucher queue (`'Voucher{ante}'` vs `'Voucher_fromtag'`), including unbought voucher re-appearances, Tier-1 -> Tier-2 prerequisite unlocking, starting deck vouchers (Magic/Nebula/Zodiac), and Voucher Tag global stream behavior.
"""

from __future__ import annotations

import math
import unittest

from balatro_sim.api import BalatroRun, validate_seed
from balatro_sim.events import (
    P_BOSS_BLINDS,
    P_CARDS_KEYS,
    P_TAGS,
    P_TAROTS,
    P_VOUCHERS,
    BossBlindPrediction,
    CertificateResult,
    IdolTarget,
    LuckyHitResult,
    RebateEvaluation,
    StoneCardResult,
    TagPrediction,
    TarotPrediction,
    VoucherPrediction,
    evaluate_lucky_hit,
    evaluate_mail_in_rebate_discard,
    get_mail_in_rebate_target,
    predict_certificate,
)
from balatro_sim.rng import (
    BalatroRNG,
    Tw223,
    advance,
    double_to_u64,
    predict_seed,
    pseudohash,
    pseudoseed,
    truncate13,
    u64_to_double,
)
from balatro_sim.state import (
    Card,
    Deck,
    Edition,
    Enhancement,
    GameState,
    JokerInstance,
    Rank,
    Seal,
    Stake,
    Suit,
)


class TestRNGExactParity(unittest.TestCase):
    """Assert bit-exact parity of PRNG math against reference fixtures."""

    def test_pseudohash_reference_vectors(self) -> None:
        """Verify pseudohash reproduces exact 64-bit IEEE-754 floats."""
        cases = [
            ("", 0x3FF0000000000000),             # 1.0
            ("A", 0x3FE4DB26AE2C1A00),            # single char
            ("TESTSEED", 0x3FD46EF4279F6C00),     # standard test seed
            ("TUTORIAL", 0x3FDABFBA38E62000),     # tutorial seed
            ("A3K9NZ2B", 0x3FEE48C7B2F08D00),
            ("7LB2WVPK", 0x3FC6A501398D4000),
            ("shuffleTESTSEED", 0x3FE3C1F649809000),
            ("Joker1sho1TESTSEED", 0x3FE50C82E208F000),
        ]
        for s, expected_u64 in cases:
            h = pseudohash(s.encode("utf-8"))
            actual_u64 = double_to_u64(h)
            self.assertEqual(
                actual_u64,
                expected_u64,
                f"pseudohash('{s}') mismatch: 0x{actual_u64:016x} != 0x{expected_u64:016x}",
            )

        # Test NaN case (division by zero in backwards hash loop)
        nan_hash = pseudohash("erratic7LB2WVPK".encode("utf-8"))
        self.assertTrue(math.isnan(nan_hash), "Expected NaN for 'erratic7LB2WVPK'")

    def test_predict_seed_reference_vectors(self) -> None:
        """Verify predict_seed produces exact channel initial seed values."""
        cases = [
            ("erratic", "P1HNLKJI", 0x3FE1E058575684B4),
            ("boss", "D5APLKJI", 0x3FDA8FD973A59632),
            ("shuffle", "293QLKJI", 0x3FD5E3161FC17942),
            ("Joker4", "QCVQLKJI", 0x3FDB31341C86132E),
            ("erratic", "EGNRLKJI", 0x3FE282D3B590D8E1),
        ]
        for key, seed, expected_u64 in cases:
            s_val = predict_seed(key, seed)
            actual_u64 = double_to_u64(s_val)
            self.assertEqual(
                actual_u64,
                expected_u64,
                f"predict_seed('{key}', '{seed}') mismatch: 0x{actual_u64:016x} != 0x{expected_u64:016x}",
            )

    def test_stream_interleaved_operations(self) -> None:
        """Verify multi-stream step progression against reference TESTSEED sequence."""
        rng = BalatroRNG("TESTSEED")

        # 1. ['range', 'cdt1', 1, 52, 17]
        r1 = rng.randint("cdt1", 1, 52)
        self.assertEqual(r1, 17)

        # 2. ['range', 'rarity1sho', 1, 52, 7]
        r2 = rng.randint("rarity1sho", 1, 52)
        self.assertEqual(r2, 7)

        # 3. ['random', 'lucky_mult', '3fd496c83c669c70']
        f3 = rng.random("lucky_mult")
        self.assertEqual(double_to_u64(f3), 0x3FD496C83C669C70)

        # 4. ['range', 'rarity1sho', 1, 52, 41]
        r4 = rng.randint("rarity1sho", 1, 52)
        self.assertEqual(r4, 41)

        # 5. ['range', 'anc1', 1, 52, 36]
        r5 = rng.randint("anc1", 1, 52)
        self.assertEqual(r5, 36)

    def test_tausworthe_tw223_seeded_draws(self) -> None:
        """Verify Tw223 seeded generator matches LuaJIT math.random exactly."""
        tw = Tw223(1.0)
        expected_hex_floats = [
            0x3FD4B94FEFE0FFCC,
            0x3F8BE15720593380,
            0x3FED35050026251A,
            0x3FD31DF64372CCA8,
        ]
        for i, exp in enumerate(expected_hex_floats):
            val = tw.random()
            self.assertEqual(
                double_to_u64(val),
                exp,
                f"Tw223(1.0) float[{i}] mismatch: 0x{double_to_u64(val):016x} != 0x{exp:016x}",
            )


class TestReproducibility(unittest.TestCase):
    """Assert 100% deterministic reproducibility for seed 'HELLO' across runs."""

    def test_reproducible_hello_sequences(self) -> None:
        run1 = BalatroRun(seed="HELLO")
        run2 = BalatroRun(seed="HELLO")

        streams = ["lucky_mult", "lucky_money", "cert_fr", "wheel_of_fortune", "boss"]
        for stream in streams:
            for _ in range(20):
                v1 = run1.rng.random(stream)
                v2 = run2.rng.random(stream)
                self.assertEqual(double_to_u64(v1), double_to_u64(v2))

    def test_reproducible_event_evaluations(self) -> None:
        run1 = BalatroRun(seed="HELLO")
        run2 = BalatroRun(seed="HELLO")

        c1 = run1.predict_certificate(ante=6, add_to_deck=False)
        c2 = run2.predict_certificate(ante=6, add_to_deck=False)
        self.assertEqual(c1.card, c2.card)
        self.assertEqual(c1.front_key, c2.front_key)
        self.assertEqual(c1.seal_key, c2.seal_key)

        for _ in range(5):
            h1 = run1.evaluate_lucky_hit()
            h2 = run2.evaluate_lucky_hit()
            self.assertEqual(h1, h2)

        for _ in range(5):
            w1 = run1.evaluate_wheel_of_fortune()
            w2 = run2.evaluate_wheel_of_fortune()
            self.assertEqual(w1, w2)


class TestStepDependentAdvancement(unittest.TestCase):
    """Verify that querying outcomes accurately advances PRNG channel states."""

    def test_lucky_hit_channel_consumption(self) -> None:
        run = BalatroRun(seed="HELLO")

        self.assertEqual(run.rng.get_channel_call_count("lucky_mult"), 0)
        self.assertEqual(run.rng.get_channel_call_count("lucky_money"), 0)

        run.evaluate_lucky_hit()
        self.assertEqual(run.rng.get_channel_call_count("lucky_mult"), 1)
        self.assertEqual(run.rng.get_channel_call_count("lucky_money"), 1)

        run.evaluate_lucky_hit()
        self.assertEqual(run.rng.get_channel_call_count("lucky_mult"), 2)
        self.assertEqual(run.rng.get_channel_call_count("lucky_money"), 2)

    def test_certificate_step_advancement(self) -> None:
        run = BalatroRun(seed="HELLO")

        run.predict_certificate(ante=1, add_to_deck=False)
        self.assertEqual(run.rng.get_channel_call_count("cert_fr1"), 1)
        self.assertEqual(run.rng.get_channel_call_count("cert_sl1"), 1)

        run.predict_certificate(ante=1, add_to_deck=False)
        self.assertEqual(run.rng.get_channel_call_count("cert_fr1"), 2)
        self.assertEqual(run.rng.get_channel_call_count("cert_sl1"), 2)

        self.assertEqual(run.rng.get_channel_call_count("cert_fr2"), 0)

    def test_round_transition_advancement(self) -> None:
        run = BalatroRun(seed="HELLO")
        run.set_state(ante=1, round_num=1)
        self.assertEqual(run.state.round, 1)

        run.end_round()
        run.start_round(2)
        self.assertEqual(run.state.round, 2)
        self.assertEqual(run.state.total_rounds_played, 1)


class TestCertificateMechanics(unittest.TestCase):
    """Test Certificate joker card and seal generation."""

    def test_certificate_seed_hello_ante_6(self) -> None:
        """Seed 'HELLO' at Ante 6 generates Queen of Clubs with Purple Seal."""
        run = BalatroRun(seed="HELLO")
        run.set_state(ante=6, round_num=2)

        initial_deck_len = len(run.state.deck)
        cert_result = run.predict_certificate()

        self.assertEqual(cert_result.suit, Suit.CLUBS)
        self.assertEqual(cert_result.rank, Rank.QUEEN)
        self.assertEqual(cert_result.seal, Seal.PURPLE)
        self.assertEqual(cert_result.front_key, "C_Q")
        self.assertEqual(cert_result.seal_key, "Purple")

        self.assertEqual(len(run.state.deck), initial_deck_len + 1)
        self.assertIn(cert_result.card, run.state.deck)
        self.assertIn(cert_result.card, run.state.hand)

    def test_certificate_source_parity_mode(self) -> None:
        """Verify `ante_scoped=False` uses `'cert_fr'` and `'certsl'` float thresholds."""
        run = BalatroRun(seed="HELLO")
        res = run.predict_certificate(ante_scoped=False, add_to_deck=False)
        self.assertEqual(run.get_stream_call_count("cert_fr"), 1)
        self.assertEqual(run.get_stream_call_count("certsl"), 1)
        self.assertIn(res.seal, [Seal.RED, Seal.BLUE, Seal.GOLD, Seal.PURPLE])

    def test_certificate_batch_prediction_no_mutation(self) -> None:
        run = BalatroRun(seed="HELLO")
        initial_deck_size = len(run.state.deck)

        batch = run.batch_predict_certificates(antes=[1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(len(batch), 8)
        self.assertEqual(len(run.state.deck), initial_deck_size)
        ante_6_result = batch[5]
        self.assertEqual(ante_6_result.ante, 6)
        self.assertEqual(ante_6_result.rank, Rank.QUEEN)
        self.assertEqual(ante_6_result.suit, Suit.CLUBS)
        self.assertEqual(ante_6_result.seal, Seal.PURPLE)


class TestLuckyCards(unittest.TestCase):
    """Test Lucky Card probability mechanics and Oops! All 6s scaling."""

    def test_lucky_hit_return_structure(self) -> None:
        run = BalatroRun(seed="HELLO")
        payout = run.evaluate_lucky_hit()
        self.assertIsInstance(payout, dict)
        self.assertIn("mult", payout)
        self.assertIn("money", payout)
        self.assertIsInstance(payout["mult"], bool)
        self.assertIsInstance(payout["money"], bool)

    def test_lucky_hit_oops_scaling(self) -> None:
        run = BalatroRun(seed="HELLO")
        self.assertEqual(run.state.oops_count, 0)
        self.assertEqual(run.state.probability_modifier, 1.0)

        run.add_joker("Oops! All 6s")
        self.assertEqual(run.state.oops_count, 1)
        self.assertEqual(run.state.probability_modifier, 2.0)

        run.add_joker("Oops! All 6s")
        self.assertEqual(run.state.oops_count, 2)
        self.assertEqual(run.state.probability_modifier, 4.0)

        hits = [run.evaluate_lucky_hit() for _ in range(50)]
        mult_hits = sum(1 for h in hits if h["mult"])
        money_hits = sum(1 for h in hits if h["money"])
        self.assertGreater(mult_hits, 25)
        self.assertGreater(money_hits, 0)


class TestMailInRebate(unittest.TestCase):
    """Test Mail-in Rebate target rank selection and discard payout."""

    def test_mail_in_rebate_target_from_standard_deck(self) -> None:
        run = BalatroRun(seed="HELLO")
        target = run.get_mail_in_rebate_target(ante=1)
        self.assertIsInstance(target, Rank)
        self.assertEqual(target, Rank.SIX)

    def test_mail_in_rebate_custom_deck_composition(self) -> None:
        custom_deck = Deck([
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.SPADES, Rank.ACE),
            Card(Suit.CLUBS, Rank.SEVEN),
        ])
        run = BalatroRun(seed="HELLO", deck=custom_deck)
        for ante in range(1, 21):
            target = run.get_mail_in_rebate_target(ante=ante)
            self.assertIn(target, [Rank.ACE, Rank.SEVEN])

    def test_mail_in_rebate_discard_evaluation(self) -> None:
        run = BalatroRun(seed="HELLO")
        target = Rank.FOUR

        discards = [
            Card(Suit.SPADES, Rank.FOUR),
            Card(Suit.HEARTS, Rank.FOUR),
            Card(Suit.CLUBS, Rank.KING),
            Card(Suit.DIAMONDS, Rank.ACE),
        ]
        result = run.evaluate_mail_in_rebate_discard(discards, target_rank=target)
        self.assertEqual(result.matches, 2)
        self.assertEqual(result.payout, 10)
        self.assertEqual(len(result.matching_cards), 2)


class TestNewRoundTargetJokers(unittest.TestCase):
    """Test Castle, Ancient Joker, The Idol, and To Do List."""

    def test_deck_sort_id_order_matches_balatro_source(self) -> None:
        """Standard 52-card deck must match `game.lua:2367-2373` lexicographical `s..r` order."""
        deck = Deck.create_standard_52()
        keys = [c.card_key for c in deck.cards]
        self.assertEqual(keys, P_CARDS_KEYS)
        self.assertEqual([c.id for c in deck.cards], list(range(1, 53)))

    def test_castle_suit_selection_and_deck_dependency(self) -> None:
        run = BalatroRun(seed="HELLO")
        # In Ante 1 for seed HELLO, the 3 rounds (Small, Big, Boss) pull 3 times from 'cas1'
        ante1_suits = run.predict_castle_suits_for_ante(ante=1, rounds=3)
        self.assertEqual(ante1_suits, [Suit.DIAMONDS, Suit.SPADES, Suit.HEARTS])

        # Edge Case 1: Checkered Deck (only Spades and Hearts)
        checkered_run = BalatroRun(seed="HELLO", deck="Checkered Deck")
        for ante in range(1, 9):
            for s in checkered_run.predict_castle_suits_for_ante(ante=ante, rounds=3):
                self.assertIn(s, [Suit.SPADES, Suit.HEARTS])

        # Edge Case 2: All Stone Cards in deck -> defaults to Suit.SPADES without crashing
        all_stone_deck = Deck([
            Card(Suit.HEARTS, Rank.KING, enhancement=Enhancement.STONE),
            Card(Suit.DIAMONDS, Rank.ACE, enhancement=Enhancement.STONE),
        ])
        stone_run = BalatroRun(seed="HELLO", deck=all_stone_deck)
        self.assertEqual(stone_run.get_castle_suit(ante=1), Suit.SPADES)

    def test_ancient_joker_suit_selection_and_non_repeat(self) -> None:
        run = BalatroRun(seed="HELLO")
        suits_by_ante = run.predict_ancient_joker_suits(antes=range(1, 9), rounds_per_ante=3)

        # Flatten all 24 rounds across Antes 1..8 and verify no two consecutive rounds ever share a suit!
        flat_suits = [s for a in range(1, 9) for s in suits_by_ante[a]]
        self.assertEqual(len(flat_suits), 24)
        for i in range(1, len(flat_suits)):
            self.assertNotEqual(
                flat_suits[i],
                flat_suits[i - 1],
                f"Ancient Joker repeated {flat_suits[i]} at round index {i}",
            )

        # Verify Ante 1 and Ante 2 exact sequences for seed HELLO
        self.assertEqual(suits_by_ante[1], [Suit.SPADES, Suit.HEARTS, Suit.SPADES])
        self.assertEqual(suits_by_ante[2], [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS])

        # Edge case: Skipping a blind in Ante 1 reduces 'anc1' calls from 3 to 2, changing the suit
        # carried into Ante 2 Small Blind!
        with_skip = run.predict_ancient_joker_suits(
            antes=[1, 2], rounds_per_ante=3, skips_per_ante={1: 1}
        )
        self.assertEqual(len(with_skip[1]), 2)
        self.assertEqual(with_skip[1], [Suit.SPADES, Suit.HEARTS])
        # Because Ante 1 ended on Hearts instead of Spades, Ante 2 Round 1 excludes Hearts and picks Spades!
        self.assertEqual(with_skip[2][0], Suit.SPADES)

    def test_idol_card_selection_and_deck_dependency(self) -> None:
        run = BalatroRun(seed="HELLO")
        ante1_idols = run.predict_idol_cards_for_ante(ante=1, rounds=3)
        self.assertEqual(len(ante1_idols), 3)
        self.assertEqual((ante1_idols[0].rank, ante1_idols[0].suit), (Rank.TWO, Suit.CLUBS))
        self.assertEqual((ante1_idols[1].rank, ante1_idols[1].suit), (Rank.NINE, Suit.SPADES))
        self.assertEqual((ante1_idols[2].rank, ante1_idols[2].suit), (Rank.FOUR, Suit.HEARTS))

        # Edge Case: Deck thinned to only Ace of Spades and Stone Cards -> Idol always picks Ace of Spades
        thin_deck = Deck([
            Card(Suit.SPADES, Rank.ACE),
            Card(Suit.HEARTS, Rank.TWO, enhancement=Enhancement.STONE),
            Card(Suit.CLUBS, Rank.KING, enhancement=Enhancement.STONE),
        ])
        thin_run = BalatroRun(seed="HELLO", deck=thin_deck)
        for _ in range(5):
            target = thin_run.get_idol_card(ante=1)
            self.assertEqual(target.rank, Rank.ACE)
            self.assertEqual(target.suit, Suit.SPADES)

    def test_to_do_list_hand_selection_and_secret_hands(self) -> None:
        run = BalatroRun(seed="HELLO")
        hands = run.predict_to_do_list_hands(rounds=10)
        self.assertEqual(len(hands), 10)
        # First draw on seed HELLO from 9 visible hands is 'Straight'
        self.assertEqual(hands[0], "Straight")
        self.assertEqual(hands[:4], ["Straight", "Flush", "Three of a Kind", "Four of a Kind"])
        # Subsequent end-of-round draws exclude the current hand, so no two consecutive rounds match
        for i in range(1, len(hands)):
            self.assertNotEqual(hands[i], hands[i - 1])

        # Edge Case: Unlocking secret hands ('Five of a Kind', 'Flush Five') mid-run expands the pool
        hands_with_secret = run.predict_to_do_list_hands(
            rounds=4,
            secret_hand_unlocks={2: ["Five of a Kind", "Flush Five"]},
        )
        self.assertEqual(hands_with_secret, ["Straight", "Flush", "Three of a Kind", "Straight Flush"])


class TestCartomancerVagabondAndStoneJoker(unittest.TestCase):
    """Test Cartomancer, Vagabond, and Marble/Stone Joker card predictions."""

    def test_cartomancer_and_vagabond_tarot_predictions(self) -> None:
        run = BalatroRun(seed="HELLO")

        # Cartomancer uses 'Tarotcar{ante}'
        c1 = run.predict_cartomancer_tarot(ante=1)
        self.assertEqual(c1.key, "c_moon")
        self.assertEqual(c1.name, "The Moon")
        self.assertEqual(c1.stream_key, "Tarotcar1")
        self.assertEqual(c1.resample_count, 0)

        # Vagabond uses 'Tarotvag{ante}'
        v1 = run.predict_vagabond_tarot(ante=1)
        self.assertEqual(v1.key, "c_empress")
        self.assertEqual(v1.name, "The Empress")
        self.assertEqual(v1.stream_key, "Tarotvag1")
        self.assertEqual(v1.resample_count, 0)

        # Edge Case 1: Holding 'The Moon' in consumables blocks Cartomancer from rolling it again
        # (unless Showman is owned), triggering 'Tarotcar1_resample2'!
        run_blocked = BalatroRun(seed="HELLO")
        c1_blocked = run_blocked.predict_cartomancer_tarot(
            ante=1, active_tarots={"The Moon"}
        )
        self.assertNotEqual(c1_blocked.key, "c_moon")
        self.assertGreaterEqual(c1_blocked.resample_count, 1)
        self.assertEqual(run_blocked.get_stream_call_count("Tarotcar1_resample2"), 1)

        # Edge Case 2: With Showman owned, holding 'The Moon' does NOT block it!
        run_showman = BalatroRun(seed="HELLO")
        run_showman.add_joker("Showman")
        c1_showman = run_showman.predict_cartomancer_tarot(
            ante=1, active_tarots={"The Moon"}
        )
        self.assertEqual(c1_showman.key, "c_moon")
        self.assertEqual(c1_showman.resample_count, 0)

        # Edge Case 3: If all 22 Tarots are held/banned without Showman, fallback is 'c_strength' ("Strength")
        all_22 = {t.key for t in P_TAROTS}
        run_all_blocked = BalatroRun(seed="HELLO")
        fallback_t = run_all_blocked.predict_cartomancer_tarot(ante=1, active_tarots=all_22)
        self.assertEqual(fallback_t.key, "c_strength")
        self.assertEqual(fallback_t.name, "Strength")

    def test_stone_joker_hidden_suit_and_rank(self) -> None:
        run = BalatroRun(seed="HELLO")
        initial_len = len(run.state.deck)

        stone_res = run.predict_stone_joker_card(add_to_deck=True)
        self.assertEqual(len(run.state.deck), initial_len + 1)
        self.assertEqual(stone_res.card.enhancement, Enhancement.STONE)
        self.assertEqual(stone_res.front_key, "S_6")
        self.assertEqual(stone_res.hidden_rank, Rank.SIX)
        self.assertEqual(stone_res.hidden_suit, Suit.SPADES)
        self.assertEqual(stone_res.card.id, 53)

        # While it is a Stone Card, `is_suit` and `is_rank` return False and it is excluded from Idol/Mail/Castle
        self.assertFalse(stone_res.card.is_suit(Suit.SPADES))
        self.assertFalse(stone_res.card.is_rank(Rank.SIX))
        self.assertNotIn(stone_res.card, run.state.deck.get_non_stone_cards())

        # When enhanced into a Mult Card (e.g. via The Empress), its hidden 6 of Spades is revealed!
        run.state.deck.convert_card(stone_res.card, new_enhancement=Enhancement.MULT)
        self.assertTrue(stone_res.card.is_suit(Suit.SPADES))
        self.assertTrue(stone_res.card.is_rank(Rank.SIX))
        self.assertIn(stone_res.card, run.state.deck.get_non_stone_cards())


class TestBossBlindsTagsAndVouchers(unittest.TestCase):
    """Test Boss Blinds, Skip Tags, and Vouchers queues and edge cases."""

    def test_boss_blind_sequence_and_edge_cases(self) -> None:
        run = BalatroRun(seed="HELLO")

        # 1. Standard Antes 1..8 sequence
        bosses = run.predict_boss_blinds(max_ante=8)
        self.assertEqual(len(bosses), 8)
        # Ante 1..7 must be Normal bosses; Ante 8 must be a Showdown boss
        for b in bosses[:7]:
            self.assertFalse(b.is_showdown)
        self.assertTrue(bosses[7].is_showdown)
        self.assertEqual(bosses[0].key, "bl_goad")
        self.assertEqual(bosses[7].key, "bl_final_leaf")
        # No normal boss repeats in Antes 1..7
        normal_keys = [b.key for b in bosses[:7]]
        self.assertEqual(len(normal_keys), len(set(normal_keys)))

        # 2. Edge Case: Reroll boss at Ante 1 (Director's Cut / Boss Tag)
        with_reroll = run.predict_boss_blinds(max_ante=8, rerolls_at_step={1: 1})
        self.assertEqual(len(with_reroll), 9)
        self.assertEqual(with_reroll[0].key, "bl_goad")
        self.assertTrue(with_reroll[1].is_reroll)
        self.assertEqual(with_reroll[1].ante, 1)
        self.assertNotEqual(with_reroll[1].key, "bl_goad")

        # 3. Edge Case: Buying Hieroglyph in Ante 1 (step 1) drops ante to 0, so defeating Ante 1's
        # boss brings ante back to 1 (`ease_ante(1)`), rolling a second Ante 1 boss!
        with_hiero_a1 = run.predict_boss_blinds(
            max_ante=8, minus_ante_vouchers_at_step={1: 1}
        )
        self.assertEqual(len(with_hiero_a1), 9)
        self.assertEqual(with_hiero_a1[0].ante, 1)
        self.assertEqual(with_hiero_a1[1].ante, 1)
        self.assertNotEqual(with_hiero_a1[0].key, with_hiero_a1[1].key)

        # 4. Edge Case: Buying Hieroglyph in Ante 8 (step 8) AND rerolling the boss in Ante 8
        # AFTER buying Hieroglyph! Because Hieroglyph immediately sets `ante = 7`, rerolling the
        # boss in Ante 8 after buying Hieroglyph rolls a NORMAL boss (`ante = 7`) instead of a
        # Showdown boss, and then the next ante is Ante 8 again with a new Showdown boss!
        hiero_a8_reroll = run.predict_boss_blinds(
            max_ante=8,
            minus_ante_vouchers_at_step={8: 1},
            rerolls_after_voucher_at_step={8: 1},
        )
        # Step 8 initial boss is Showdown (Verdant Leaf at ante=8)
        self.assertEqual(hiero_a8_reroll[7].ante, 8)
        self.assertTrue(hiero_a8_reroll[7].is_showdown)
        # Reroll after Hieroglyph in step 8 is rolled at ante=7 -> Normal boss!
        self.assertEqual(hiero_a8_reroll[8].ante, 7)
        self.assertTrue(hiero_a8_reroll[8].is_reroll)
        self.assertFalse(hiero_a8_reroll[8].is_showdown)
        # Then step 9 is Ante 8 again -> Showdown boss (and not Verdant Leaf, since Verdant Leaf was already used)!
        self.assertEqual(hiero_a8_reroll[9].ante, 8)
        self.assertTrue(hiero_a8_reroll[9].is_showdown)
        self.assertNotEqual(hiero_a8_reroll[9].key, hiero_a8_reroll[7].key)

        # 5. Edge Case: Cycle exhaustion across 48 antes (6 Showdown bosses at Antes 8, 16, 24, 32, 40, 48)
        # Since there are 5 Showdown bosses, Antes 8..40 exhaust all 5 without repeats, and Ante 48
        # resets `min_use` to 1!
        endless_bosses = run.predict_boss_blinds(max_ante=48)
        showdowns = [b for b in endless_bosses if b.is_showdown]
        self.assertEqual(len(showdowns), 6)
        first_five_keys = {b.key for b in showdowns[:5]}
        self.assertEqual(len(first_five_keys), 5)
        self.assertEqual(showdowns[5].min_use_tier, 1)

    def test_skip_tag_sequence_and_edge_cases(self) -> None:
        run = BalatroRun(seed="HELLO")
        tags_per_ante = run.predict_skip_tags(max_ante=8, include_orbital_hands=True)
        self.assertEqual(len(tags_per_ante), 8)

        # Verify Ante 1 tags never include any `min_ante = 2` tag
        min_ante_2_keys = {t.key for t in P_TAGS if t.min_ante == 2}
        for t in tags_per_ante[0]:
            self.assertNotIn(t.key, min_ante_2_keys)
        self.assertEqual(tags_per_ante[0][0].key, "tag_voucher")
        self.assertEqual(tags_per_ante[0][1].key, "tag_voucher")

        # Edge Case: Redeeming Hieroglyph in Ante 1 causes Ante 1 to be visited twice;
        # the second visit to Ante 1 pulls the 3rd & 4th draws of 'Tag1', which hit a `min_ante = 2`
        # slot and resample via 'Tag1_resample2'!
        tags_with_hiero_a1 = run.predict_skip_tags(
            max_ante=2, minus_ante_vouchers_at_step={1: 1}
        )
        self.assertEqual(len(tags_with_hiero_a1), 3)
        for pair in tags_with_hiero_a1[:2]:
            for t in pair:
                self.assertEqual(t.ante, 1)
                self.assertNotIn(t.key, min_ante_2_keys)

        # Edge Case: Redeeming Hieroglyph in Ante 2 causes Ante 2 to be visited twice,
        # pulling tags 1 & 2 from 'Tag2' on visit 1, and tags 3 & 4 from 'Tag2' on visit 2!
        tags_with_hiero = run.predict_skip_tags(
            max_ante=3, minus_ante_vouchers_at_step={2: 1}
        )
        self.assertEqual(len(tags_with_hiero), 4)
        self.assertEqual(tags_with_hiero[1][0].ante, 2)
        self.assertEqual(tags_with_hiero[2][0].ante, 2)
        self.assertNotEqual(
            (tags_with_hiero[1][0].key, tags_with_hiero[1][1].key),
            (tags_with_hiero[2][0].key, tags_with_hiero[2][1].key),
        )

    def test_voucher_sequence_purchase_adaptation_and_voucher_tag_global_queue(self) -> None:
        run = BalatroRun(seed="HELLO")

        # 1. If NO vouchers are purchased (`purchase_plan=False`), Tier-2 vouchers never appear,
        # and unbought Tier-1 vouchers can appear multiple times across Antes 1..8!
        unbought_seq = run.predict_voucher_sequence(max_ante=8, purchase_plan=False)
        tier2_keys = {v.key for v in P_VOUCHERS if v.requires}
        for v in unbought_seq:
            self.assertNotIn(v.key, tier2_keys)
        # In seed HELLO, 'v_grabber' (Grabber) appears in Ante 2 AND again in Ante 5 if not bought in Ante 2!
        self.assertEqual(unbought_seq[1].key, "v_grabber")
        self.assertEqual(unbought_seq[4].key, "v_grabber")

        # 2. If vouchers ARE purchased (`purchase_plan=True`), 'v_grabber' bought in Ante 2 cannot
        # reappear in Ante 5 (Ante 5 resamples to 'v_wasteful'), and Tier-2 upgrades unlock!
        bought_seq = run.predict_voucher_sequence(max_ante=8, purchase_plan=True)
        self.assertEqual(bought_seq[1].key, "v_grabber")
        self.assertEqual(bought_seq[4].key, "v_wasteful")
        self.assertGreaterEqual(bought_seq[4].resample_count, 1)
        # No voucher ever repeats when bought!
        bought_keys = [v.key for v in bought_seq]
        self.assertEqual(len(bought_keys), len(set(bought_keys)))

        # 3. Voucher Tag (`_from_tag = true`) pulls from the GLOBAL `'Voucher_fromtag'` stream,
        # NOT the ante-specific `'Voucher{ante}'` stream!
        # Verify that taking 2 Voucher Tags in Ante 1 vs taking 1 in Ante 1 and 1 in Ante 2
        # pulls the exact same sequence from `'Voucher_fromtag'` (when no conflicting vouchers are bought).
        seq_two_in_a1 = run.predict_voucher_sequence(
            max_ante=2, purchase_plan=False, voucher_tags_at_step={1: 2}
        )
        tag_vouchers_a1 = [v for v in seq_two_in_a1 if v.from_tag]
        self.assertEqual(len(tag_vouchers_a1), 2)
        self.assertEqual(tag_vouchers_a1[0].stream_key, "Voucher_fromtag")
        self.assertEqual(tag_vouchers_a1[1].stream_key, "Voucher_fromtag")

        seq_split_a1_a2 = run.predict_voucher_sequence(
            max_ante=2, purchase_plan=False, voucher_tags_at_step={1: 1, 2: 1}
        )
        tag_vouchers_split = [v for v in seq_split_a1_a2 if v.from_tag]
        self.assertEqual(tag_vouchers_a1[0].key, tag_vouchers_split[0].key)
        self.assertEqual(tag_vouchers_a1[1].key, tag_vouchers_split[1].key)

        # 4. Starting Deck Vouchers (Zodiac Deck starts with Tarot Merchant, Planet Merchant, Overstock)
        zodiac_run = BalatroRun(seed="HELLO", deck="Zodiac Deck")
        self.assertIn("v_overstock_norm", zodiac_run.state.used_vouchers)
        self.assertIn("v_tarot_merchant", zodiac_run.state.used_vouchers)
        self.assertIn("v_planet_merchant", zodiac_run.state.used_vouchers)
        # In Seed HELLO, Ante 1 normal voucher is 'v_planet_merchant' on Red Deck,
        # but on Zodiac Deck 'v_planet_merchant' is already owned at run start, so Ante 1 resamples!
        zodiac_v1 = zodiac_run.get_next_voucher(ante=1)
        self.assertNotEqual(zodiac_v1.key, "v_planet_merchant")
        self.assertGreaterEqual(zodiac_v1.resample_count, 1)


class TestEventResolvers(unittest.TestCase):
    """Test 1-in-N outcome resolutions for probability jokers and cards."""

    def test_outcome_resolutions(self) -> None:
        run = BalatroRun(seed="HELLO")

        w = run.evaluate_wheel_of_fortune()
        self.assertIsInstance(w, bool)
        self.assertEqual(run.rng.get_channel_call_count("wheel_of_fortune"), 1)

        s = run.evaluate_space_joker()
        self.assertIsInstance(s, bool)
        self.assertEqual(run.rng.get_channel_call_count("space_joker"), 1)

        b = run.evaluate_bloodstone()
        self.assertIsInstance(b, bool)
        self.assertEqual(run.rng.get_channel_call_count("bloodstone"), 1)

        g = run.evaluate_glass_shatter()
        self.assertIsInstance(g, bool)
        self.assertEqual(run.rng.get_channel_call_count("glass"), 1)

        bc = run.evaluate_business_card()
        self.assertIsInstance(bc, bool)
        self.assertEqual(run.rng.get_channel_call_count("business_card"), 1)

    def test_oops_modifier_scaling(self) -> None:
        run = BalatroRun(seed="HELLO")
        for _ in range(3):
            run.add_joker("Oops! All 6s")
        self.assertEqual(run.state.probability_modifier, 8.0)

        self.assertTrue(run.evaluate_wheel_of_fortune())
        self.assertTrue(run.evaluate_space_joker())
        self.assertTrue(run.evaluate_bloodstone())


class TestDeckAndStateMutations(unittest.TestCase):
    """Test GameState transitions and deck card mutations."""

    def test_deck_mutations(self) -> None:
        deck = Deck.create_standard_52()
        self.assertEqual(len(deck), 52)

        custom_card = Card(
            suit=Suit.HEARTS,
            rank=Rank.ACE,
            enhancement=Enhancement.STEEL,
            edition=Edition.FOIL,
            seal=Seal.RED,
        )
        deck.add_card(custom_card)
        self.assertEqual(len(deck), 53)
        self.assertIn(custom_card, deck)

        destroyed = deck.destroy_card(custom_card)
        self.assertTrue(destroyed)
        self.assertEqual(len(deck), 52)
        self.assertNotIn(custom_card, deck)

        target_card = deck[0]
        converted = deck.convert_card(
            target_card,
            new_suit=Suit.SPADES,
            new_rank=Rank.KING,
            new_enhancement=Enhancement.GLASS,
            new_edition=Edition.POLYCHROME,
            new_seal=Seal.GOLD,
        )
        self.assertEqual(converted.suit, Suit.SPADES)
        self.assertEqual(converted.rank, Rank.KING)
        self.assertEqual(converted.enhancement, Enhancement.GLASS)
        self.assertEqual(converted.edition, Edition.POLYCHROME)
        self.assertEqual(converted.seal, Seal.GOLD)

    def test_joker_management(self) -> None:
        run = BalatroRun(seed="HELLO")
        run.add_joker("Blueprint")
        run.add_joker("Brainstorm")
        self.assertTrue(run.has_joker("Blueprint"))
        self.assertTrue(run.has_joker("Brainstorm"))
        self.assertEqual(len(run.state.owned_jokers), 2)

        removed = run.remove_joker("Blueprint")
        self.assertIsNotNone(removed)
        self.assertFalse(run.has_joker("Blueprint"))
        self.assertEqual(len(run.state.owned_jokers), 1)


class TestCheckpointAndFork(unittest.TestCase):
    """Test state checkpointing, restoring, and forking."""

    def test_checkpoint_and_restore(self) -> None:
        run = BalatroRun(seed="HELLO")
        run.set_state(ante=3, round_num=1)
        run.add_joker("Joker")

        cp = run.checkpoint()

        run.set_state(ante=7, round_num=4)
        run.predict_certificate()
        run.evaluate_lucky_hit()
        self.assertEqual(run.state.ante, 7)

        run.restore(cp)
        self.assertEqual(run.state.ante, 3)
        self.assertEqual(run.state.round, 1)
        self.assertTrue(run.has_joker("Joker"))
        self.assertEqual(run.rng.get_channel_call_count("cert_fr7"), 0)

    def test_fork(self) -> None:
        run = BalatroRun(seed="HELLO")
        run.set_state(ante=4, round_num=2)

        forked = run.fork()
        self.assertEqual(forked.state.ante, 4)

        forked.set_state(ante=8)
        forked.predict_certificate()

        self.assertEqual(run.state.ante, 4)
        self.assertEqual(run.rng.get_channel_call_count("cert_fr8"), 0)


class TestPublicAPIAndValidation(unittest.TestCase):
    """Test top-level BalatroRun interface and seed validation."""

    def test_seed_validation(self) -> None:
        valid_seeds = ["HELLO", "TEST1234", "1A2B3C4D", "A", "12345678"]
        for s in valid_seeds:
            ok, res = validate_seed(s)
            self.assertTrue(ok, f"Seed '{s}' should be valid")

        ok, res = validate_seed("hello")
        self.assertTrue(ok)
        self.assertEqual(res, "HELLO")

        invalid_seeds = [
            ("", "empty"),
            ("TOOLONGSEED", "length > 8"),
            ("HELL0", "contains '0'"),
            ("TEST-12", "contains hyphen"),
            ("SEED!", "contains punctuation"),
        ]
        for s, desc in invalid_seeds:
            ok, msg = validate_seed(s)
            self.assertFalse(ok, f"Seed '{s}' ({desc}) should be invalid")

        with self.assertRaises(ValueError):
            BalatroRun(seed="INVALID_SEED_TOOLONG")

    def test_prompt_example_workflow(self) -> None:
        run = BalatroRun(seed="HELLO")
        run.set_state(ante=6, round_num=2)

        cert_result = run.predict_certificate()
        self.assertEqual(cert_result.rank, Rank.QUEEN)
        self.assertEqual(cert_result.suit, Suit.CLUBS)
        self.assertEqual(cert_result.seal, Seal.PURPLE)

        lucky_payout = run.evaluate_lucky_hit()
        self.assertIn("mult", lucky_payout)
        self.assertIn("money", lucky_payout)

        rebate_target = run.get_mail_in_rebate_target()
        self.assertIsInstance(rebate_target, Rank)


if __name__ == "__main__":
    unittest.main()
