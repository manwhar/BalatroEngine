"""Comprehensive Unit Tests for Balatro Deterministic Simulation Engine.

Verifies:
1. Exact bit-level parity of pseudohash, pseudoseed, Tw223, and LCG step against
   reference test vectors from Immolate and Jackdaw (LuaJIT PRNG).
2. Reproducibility for seed 'HELLO' across multiple runs.
3. Step-dependent state advancement and PRNG channel consumption.
4. Certificate joker card & seal generation (including seed 'HELLO' at Ante 6).
5. Lucky Cards (+20 Mult and +$20 Cash) probability mechanics and Oops! All 6s scaling.
6. Mail-in Rebate dynamic target selection and discard evaluations.
7. 1-in-N outcome resolvers (Wheel of Fortune, Space Joker, Bloodstone, Glass, Business Card).
8. Deck mutations (add, remove, convert, destroy).
9. Public Python API (BalatroRun, batch methods, seed validation, checkpoint/restore/fork).
"""

from __future__ import annotations

import math
import unittest

from balatro_sim.api import BalatroRun, validate_seed
from balatro_sim.events import (
    CertificateResult,
    LuckyHitResult,
    RebateEvaluation,
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
        # Seeding with 1.0 (u64 = 0x3ff0000000000000)
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

        # Test PRNG streams across 100 calls
        streams = ["lucky_mult", "lucky_money", "cert_fr", "wheel_of_fortune", "boss"]
        for stream in streams:
            for _ in range(20):
                v1 = run1.rng.random(stream)
                v2 = run2.rng.random(stream)
                self.assertEqual(double_to_u64(v1), double_to_u64(v2))

    def test_reproducible_event_evaluations(self) -> None:
        run1 = BalatroRun(seed="HELLO")
        run2 = BalatroRun(seed="HELLO")

        # Both predict Certificate at Ante 6
        c1 = run1.predict_certificate(ante=6, add_to_deck=False)
        c2 = run2.predict_certificate(ante=6, add_to_deck=False)
        self.assertEqual(c1.card, c2.card)
        self.assertEqual(c1.front_key, c2.front_key)
        self.assertEqual(c1.seal_key, c2.seal_key)

        # Both evaluate Lucky Hits
        for _ in range(5):
            h1 = run1.evaluate_lucky_hit()
            h2 = run2.evaluate_lucky_hit()
            self.assertEqual(h1, h2)

        # Both evaluate Wheel of Fortune
        for _ in range(5):
            w1 = run1.evaluate_wheel_of_fortune()
            w2 = run2.evaluate_wheel_of_fortune()
            self.assertEqual(w1, w2)


class TestStepDependentAdvancement(unittest.TestCase):
    """Verify that querying outcomes accurately advances PRNG channel states."""

    def test_lucky_hit_channel_consumption(self) -> None:
        run = BalatroRun(seed="HELLO")

        # Before any roll, channel call count is 0
        self.assertEqual(run.rng.get_channel_call_count("lucky_mult"), 0)
        self.assertEqual(run.rng.get_channel_call_count("lucky_money"), 0)

        # 1st roll consumes 1 step each
        run.evaluate_lucky_hit()
        self.assertEqual(run.rng.get_channel_call_count("lucky_mult"), 1)
        self.assertEqual(run.rng.get_channel_call_count("lucky_money"), 1)

        # 2nd roll consumes another step
        run.evaluate_lucky_hit()
        self.assertEqual(run.rng.get_channel_call_count("lucky_mult"), 2)
        self.assertEqual(run.rng.get_channel_call_count("lucky_money"), 2)

    def test_certificate_step_advancement(self) -> None:
        run = BalatroRun(seed="HELLO")

        # Calling Certificate for ante 1 consumes 1 step from cert_fr1 and cert_sl1
        run.predict_certificate(ante=1, add_to_deck=False)
        self.assertEqual(run.rng.get_channel_call_count("cert_fr1"), 1)
        self.assertEqual(run.rng.get_channel_call_count("cert_sl1"), 1)

        # Calling Certificate again for ante 1 advances cert_fr1 and cert_sl1 to call count 2
        card2 = run.predict_certificate(ante=1, add_to_deck=False)
        self.assertEqual(run.rng.get_channel_call_count("cert_fr1"), 2)
        self.assertEqual(run.rng.get_channel_call_count("cert_sl1"), 2)

        # Ante 2 channel remains untouched until called
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

        # Check exact card front and seal
        self.assertEqual(cert_result.suit, Suit.CLUBS)
        self.assertEqual(cert_result.rank, Rank.QUEEN)
        self.assertEqual(cert_result.seal, Seal.PURPLE)
        self.assertEqual(cert_result.front_key, "C_Q")
        self.assertEqual(cert_result.seal_key, "Purple")

        # Verify added to deck and hand
        self.assertEqual(len(run.state.deck), initial_deck_len + 1)
        self.assertIn(cert_result.card, run.state.deck)
        self.assertIn(cert_result.card, run.state.hand)

    def test_certificate_batch_prediction_no_mutation(self) -> None:
        run = BalatroRun(seed="HELLO")
        initial_deck_size = len(run.state.deck)

        batch = run.batch_predict_certificates(antes=[1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(len(batch), 8)
        # Verify deck was not mutated during batch prediction
        self.assertEqual(len(run.state.deck), initial_deck_size)
        # Ante 6 should match Queen of Clubs, Purple Seal
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
        """Oops! All 6s doubles normal odds per active joker."""
        run = BalatroRun(seed="HELLO")

        # Base odds: modifier = 1.0 (0 Oops)
        self.assertEqual(run.state.oops_count, 0)
        self.assertEqual(run.state.probability_modifier, 1.0)

        # Add 1 Oops! All 6s: modifier = 2.0
        run.add_joker("Oops! All 6s")
        self.assertEqual(run.state.oops_count, 1)
        self.assertEqual(run.state.probability_modifier, 2.0)

        # Add 2nd Oops! All 6s: modifier = 4.0
        run.add_joker("Oops! All 6s")
        self.assertEqual(run.state.oops_count, 2)
        self.assertEqual(run.state.probability_modifier, 4.0)

        # Evaluate lucky hits over 50 rolls with 4x multiplier
        hits = [run.evaluate_lucky_hit() for _ in range(50)]
        mult_hits = sum(1 for h in hits if h["mult"])
        money_hits = sum(1 for h in hits if h["money"])

        # With 4x multiplier, Mult is 4 in 5 (80%), so mult_hits should be substantial
        self.assertGreater(mult_hits, 25)
        # Money is 4 in 15 (~26.6%), should produce at least one hit in 50 rolls
        self.assertGreater(money_hits, 0)


class TestMailInRebate(unittest.TestCase):
    """Test Mail-in Rebate target rank selection and discard payout."""

    def test_mail_in_rebate_target_from_standard_deck(self) -> None:
        run = BalatroRun(seed="HELLO")
        target = run.get_mail_in_rebate_target(ante=1)
        self.assertIsInstance(target, Rank)
        self.assertIn(target, list(Rank))

    def test_mail_in_rebate_custom_deck_composition(self) -> None:
        """Target rank must only be chosen from ranks present in the active deck."""
        custom_deck = Deck([
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.SPADES, Rank.ACE),
            Card(Suit.CLUBS, Rank.SEVEN),
        ])
        run = BalatroRun(seed="HELLO", deck=custom_deck)
        # Across 20 ante queries, the target rank should always be either ACE or SEVEN
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
        self.assertEqual(result.payout, 10)  # 2 * $5 = $10
        self.assertEqual(len(result.matching_cards), 2)

        # No matches
        non_matching = [Card(Suit.SPADES, Rank.TEN), Card(Suit.HEARTS, Rank.TWO)]
        result_zero = run.evaluate_mail_in_rebate_discard(non_matching, target_rank=target)
        self.assertEqual(result_zero.matches, 0)
        self.assertEqual(result_zero.payout, 0)


class TestEventResolvers(unittest.TestCase):
    """Test 1-in-N outcome resolutions for probability jokers and cards."""

    def test_outcome_resolutions(self) -> None:
        run = BalatroRun(seed="HELLO")

        # Test wheel of fortune (1 in 4)
        w = run.evaluate_wheel_of_fortune()
        self.assertIsInstance(w, bool)
        self.assertEqual(run.rng.get_channel_call_count("wheel_of_fortune"), 1)

        # Test space joker (1 in 4)
        s = run.evaluate_space_joker()
        self.assertIsInstance(s, bool)
        self.assertEqual(run.rng.get_channel_call_count("space_joker"), 1)

        # Test bloodstone (1 in 2)
        b = run.evaluate_bloodstone()
        self.assertIsInstance(b, bool)
        self.assertEqual(run.rng.get_channel_call_count("bloodstone"), 1)

        # Test glass card shatter (1 in 4)
        g = run.evaluate_glass_shatter()
        self.assertIsInstance(g, bool)
        self.assertEqual(run.rng.get_channel_call_count("glass"), 1)

        # Test business card (1 in 2)
        bc = run.evaluate_business_card()
        self.assertIsInstance(bc, bool)
        self.assertEqual(run.rng.get_channel_call_count("business_card"), 1)

    def test_oops_modifier_scaling(self) -> None:
        run = BalatroRun(seed="HELLO")
        # Add 3 Oops! All 6s (modifier = 8.0)
        for _ in range(3):
            run.add_joker("Oops! All 6s")
        self.assertEqual(run.state.probability_modifier, 8.0)

        # With 8x modifier, 1 in 4 chance becomes 8/4 = 2.0 (guaranteed True)
        self.assertTrue(run.evaluate_wheel_of_fortune())
        self.assertTrue(run.evaluate_space_joker())
        self.assertTrue(run.evaluate_bloodstone())


class TestDeckAndStateMutations(unittest.TestCase):
    """Test GameState transitions and deck card mutations."""

    def test_deck_mutations(self) -> None:
        deck = Deck.create_standard_52()
        self.assertEqual(len(deck), 52)

        # Add custom card
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

        # Destroy card
        destroyed = deck.destroy_card(custom_card)
        self.assertTrue(destroyed)
        self.assertEqual(len(deck), 52)
        self.assertNotIn(custom_card, deck)

        # Convert card
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

        # Take checkpoint
        cp = run.checkpoint()

        # Advance state and PRNG
        run.set_state(ante=7, round_num=4)
        run.predict_certificate()
        run.evaluate_lucky_hit()
        self.assertEqual(run.state.ante, 7)

        # Restore checkpoint
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

        # Advance fork
        forked.set_state(ante=8)
        forked.predict_certificate()

        # Main run remains untouched
        self.assertEqual(run.state.ante, 4)
        self.assertEqual(run.rng.get_channel_call_count("cert_fr8"), 0)


class TestPublicAPIAndValidation(unittest.TestCase):
    """Test top-level BalatroRun interface and seed validation."""

    def test_seed_validation(self) -> None:
        valid_seeds = ["HELLO", "TEST1234", "1A2B3C4D", "A", "12345678"]
        for s in valid_seeds:
            ok, res = validate_seed(s)
            self.assertTrue(ok, f"Seed '{s}' should be valid")

        # Lowercase should be sanitized to uppercase
        ok, res = validate_seed("hello")
        self.assertTrue(ok)
        self.assertEqual(res, "HELLO")

        # Invalid cases
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

        # BalatroRun with invalid seed should raise ValueError when validate=True
        with self.assertRaises(ValueError):
            BalatroRun(seed="INVALID_SEED_TOOLONG")

    def test_prompt_example_workflow(self) -> None:
        """Verify the exact example workflow requested in the prompt runs cleanly."""
        run = BalatroRun(seed="HELLO")

        # Fast-forward / configure path state
        run.set_state(ante=6, round_num=2)

        # Query exact deterministic outcomes
        cert_result = run.predict_certificate()
        self.assertEqual(cert_result.rank, Rank.QUEEN)
        self.assertEqual(cert_result.suit, Suit.CLUBS)
        self.assertEqual(cert_result.seal, Seal.PURPLE)

        # Predict dynamic mid-round triggers
        lucky_payout = run.evaluate_lucky_hit()  # returns {'mult': bool, 'money': bool}
        self.assertIn("mult", lucky_payout)
        self.assertIn("money", lucky_payout)

        rebate_target = run.get_mail_in_rebate_target()
        self.assertIsInstance(rebate_target, Rank)


if __name__ == "__main__":
    unittest.main()

