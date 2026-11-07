"""Public Python API for Balatro Deterministic Simulation.

Provides a clean, expressive top-level interface:
```python
from balatro_sim.api import BalatroRun

run = BalatroRun(seed="HELLO")

# Fast-forward / configure path state
run.set_state(ante=6, round_num=2)

# Query exact deterministic outcomes
cert_result = run.predict_certificate()
print(f"Added Card: {cert_result.rank} of {cert_result.suit}, Seal: {cert_result.seal}")

# Predict dynamic mid-round triggers
lucky_payout = run.evaluate_lucky_hit()  # returns {'mult': bool, 'money': bool}
rebate_target = run.get_mail_in_rebate_target()
```
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from balatro_sim.events import (
    CertificateResult,
    EventResolver,
    LuckyHitResult,
    RebateEvaluation,
    evaluate_lucky_hit,
    evaluate_mail_in_rebate_discard,
    get_mail_in_rebate_target,
    predict_certificate,
)
from balatro_sim.rng import BalatroRNG, pseudohash, pseudoseed
from balatro_sim.state import Card, Deck, Edition, Enhancement, GameState, JokerInstance, Rank, Seal, Stake, Suit

# Allowed Balatro seed characters: 1-9 and A-Z (excluding 0, though we sanitize)
BALATRO_SEED_PATTERN = re.compile(r"^[1-9A-Z]{1,8}$")


def validate_seed(seed: str) -> tuple[bool, str]:
    """Validate and sanitize a Balatro seed.
    
    Balatro run seeds are strings up to 8 characters composed of '1'-'9' and 'A'-'Z'.
    Lowercase characters are automatically capitalized.
    
    Returns (is_valid, sanitized_seed_or_error_message).
    """
    clean = seed.strip().upper()
    # Replace '0' with 'O' or check validity
    if not clean:
        return False, "Seed cannot be empty."
    if len(clean) > 8:
        return False, f"Seed length cannot exceed 8 characters (got {len(clean)})."
    # Check characters
    invalid_chars = [c for c in clean if not (c.isdigit() or c.isalpha()) or c == "0"]
    if invalid_chars:
        return False, f"Seed contains invalid character(s): {set(invalid_chars)}. Allowed: 1-9, A-Z."
    return True, clean


class BalatroRun:
    """Deterministic simulation interface for a Balatro run."""

    def __init__(
        self,
        seed: str = "HELLO",
        stake: Stake | str = Stake.WHITE,
        deck: Deck | None = None,
        validate: bool = True,
    ) -> None:
        if validate:
            is_valid, sanitized = validate_seed(seed)
            if not is_valid:
                raise ValueError(f"Invalid Balatro seed '{seed}': {sanitized}")
            self.seed = sanitized
        else:
            self.seed = str(seed).strip().upper()

        self.rng = BalatroRNG(self.seed)
        self.state = GameState(
            seed=self.seed,
            stake=Stake(stake) if isinstance(stake, str) else stake,
            deck=deck if deck is not None else Deck.create_standard_52(),
        )
        self.events = EventResolver(self.rng)

    # -----------------------------------------------------------------------
    # State Configuration & Fast-Forwarding
    # -----------------------------------------------------------------------

    def set_state(
        self,
        ante: int | None = None,
        round_num: int | None = None,
        total_rounds: int | None = None,
        money: int | None = None,
        stake: Stake | str | None = None,
    ) -> None:
        """Fast-forward / configure path state parameters."""
        if ante is not None:
            self.state.ante = int(ante)
        if round_num is not None:
            self.state.round_within_ante = int(round_num)
        if total_rounds is not None:
            self.state.total_rounds = int(total_rounds)
        if money is not None:
            self.state.money = int(money)
        if stake is not None:
            self.state.stake = Stake(stake) if isinstance(stake, str) else stake

    def start_ante(self, ante_num: int) -> None:
        """Transition into a new ante."""
        self.state.start_ante(ante_num)

    def start_round(self, round_num: int | None = None) -> None:
        """Begin a new round within the current ante."""
        self.state.start_round(round_num)

    def end_round(self) -> None:
        """Conclude the active round."""
        self.state.end_round()

    # -----------------------------------------------------------------------
    # Joker Management
    # -----------------------------------------------------------------------

    def add_joker(
        self,
        name: str,
        edition: Edition | str = Edition.NONE,
        eternal: bool = False,
        perishable: bool = False,
        rental: bool = False,
        state: dict[str, Any] | None = None,
    ) -> JokerInstance:
        """Add an active joker to the owned jokers."""
        ed = Edition(edition) if isinstance(edition, str) else edition
        return self.state.add_joker(
            name=name,
            edition=ed,
            eternal=eternal,
            perishable=perishable,
            rental=rental,
            state=state,
        )

    def remove_joker(self, target: int | str) -> JokerInstance | None:
        """Remove a joker by index or name."""
        return self.state.remove_joker(target)

    def has_joker(self, name: str) -> bool:
        """Check whether an active joker is present."""
        return self.state.has_joker(name)

    # -----------------------------------------------------------------------
    # Event Predictions & Roll Solvers
    # -----------------------------------------------------------------------

    def predict_certificate(
        self,
        ante: int | None = None,
        add_to_deck: bool = True,
    ) -> CertificateResult:
        """Query exact deterministic Certificate joker outcome for the ante.
        
        Pulls card front from G.P_CARDS using 'cert_fr' + str(ante)
        and card seal from G.P_SEALS using 'cert_sl' + str(ante).
        """
        target_ante = ante if ante is not None else self.state.ante
        return predict_certificate(
            rng=self.rng,
            ante=target_ante,
            state=self.state,
            add_to_state=add_to_deck,
        )

    def evaluate_lucky_hit(self) -> dict[str, bool]:
        """Predict dynamic mid-round triggers for a scored Lucky Card.
        
        Evaluates +$20 cash (1 in 15 base) and +20 Mult (1 in 5 base)
        using 'lucky_money' and 'lucky_mult' stream keys, scaled by Oops! All 6s.
        
        Returns:
            dict with keys 'mult' and 'money' (e.g. {'mult': False, 'money': True})
        """
        modifier = self.state.probability_modifier
        res = evaluate_lucky_hit(
            rng=self.rng,
            probability_modifier=modifier,
        )
        return res.to_dict()

    def get_mail_in_rebate_target(self, ante: int | None = None) -> Rank:
        """Query dynamically chosen Mail-in Rebate target rank for the current deck."""
        target_ante = ante if ante is not None else self.state.ante
        return get_mail_in_rebate_target(
            rng=self.rng,
            deck=self.state.deck,
            ante=target_ante,
        )

    def evaluate_mail_in_rebate_discard(
        self,
        discarded_cards: Sequence[Card],
        target_rank: Rank | None = None,
    ) -> RebateEvaluation:
        """Evaluate payout from discarding candidates with Mail-in Rebate active."""
        rank = target_rank if target_rank is not None else self.get_mail_in_rebate_target()
        return evaluate_mail_in_rebate_discard(
            target_rank=rank,
            discarded_cards=discarded_cards,
        )

    def evaluate_wheel_of_fortune(self) -> bool:
        """Resolve The Wheel of Fortune (1 in 4 base chance)."""
        return self.events.wheel_of_fortune(self.state.probability_modifier)

    def evaluate_space_joker(self) -> bool:
        """Resolve Space Joker hand level upgrade (1 in 4 base chance)."""
        return self.events.space_joker(self.state.probability_modifier)

    def evaluate_bloodstone(self, base_denominator: int = 2) -> bool:
        """Resolve Bloodstone extra mult check (1 in 2 base chance)."""
        return self.events.bloodstone(
            oops_multiplier=self.state.probability_modifier,
            base_denominator=base_denominator,
        )

    def evaluate_glass_shatter(self) -> bool:
        """Resolve Glass Card shatter upon scoring (1 in 4 base chance)."""
        return self.events.glass_card_shatter(self.state.probability_modifier)

    def evaluate_business_card(self) -> bool:
        """Resolve Business Card $2 payout on face card scoring (1 in 2 base chance)."""
        return self.events.business_card(self.state.probability_modifier)

    def evaluate_gros_michel(self) -> bool:
        """Resolve Gros Michel extinction check at end of round (1 in 6 base chance)."""
        return self.events.gros_michel_extinct(self.state.probability_modifier)

    def evaluate_cavendish(self) -> bool:
        """Resolve Cavendish extinction check at end of round (1 in 1000 base chance)."""
        return self.events.cavendish_extinct(self.state.probability_modifier)

    # -----------------------------------------------------------------------
    # Batch Lookup Functions
    # -----------------------------------------------------------------------

    def batch_predict_certificates(self, antes: list[int]) -> list[CertificateResult]:
        """Predict Certificate outcomes across multiple future antes without mutating deck."""
        results: list[CertificateResult] = []
        for ante in antes:
            # Predict using a fork so we do not mutate the main RNG stream or state
            forked_rng = self.rng.fork()
            res = predict_certificate(rng=forked_rng, ante=ante, add_to_state=False)
            results.append(res)
        return results

    def batch_evaluate_lucky_hits(self, count: int) -> list[dict[str, bool]]:
        """Evaluate a sequence of consecutive Lucky Card scoring checks."""
        return [self.evaluate_lucky_hit() for _ in range(count)]

    def batch_evaluate_wheel_of_fortune(self, count: int) -> list[bool]:
        """Evaluate a sequence of consecutive Wheel of Fortune usages."""
        return [self.evaluate_wheel_of_fortune() for _ in range(count)]

    def batch_evaluate_space_joker(self, count: int) -> list[bool]:
        """Evaluate a sequence of consecutive Space Joker played hand checks."""
        return [self.evaluate_space_joker() for _ in range(count)]

    def batch_evaluate_glass_shatter(self, count: int) -> list[bool]:
        """Evaluate a sequence of consecutive Glass Card scoring checks."""
        return [self.evaluate_glass_shatter() for _ in range(count)]

    def batch_evaluate_business_card(self, count: int) -> list[bool]:
        """Evaluate a sequence of consecutive Business Card face card scoring checks."""
        return [self.evaluate_business_card() for _ in range(count)]

    # -----------------------------------------------------------------------
    # Stream Inspection, Advancing, & Checkpointing
    # -----------------------------------------------------------------------

    def advance_stream(self, key: str, steps: int = 1) -> float:
        """Arbitrarily advance a named PRNG stream channel."""
        return self.rng.advance(key, steps)

    def get_stream_state(self, key: str) -> float | None:
        """Inspect the raw LCG state of a PRNG stream channel."""
        return self.rng.inspect_state(key)

    def get_stream_call_count(self, key: str) -> int:
        """Get the number of times a PRNG stream channel has been called."""
        return self.rng.get_channel_call_count(key)

    def checkpoint(self) -> dict[str, Any]:
        """Save a complete snapshot checkpoint of the run state and RNG pipeline."""
        return {
            "seed": self.seed,
            "rng": self.rng.checkpoint(),
            "state": self.state.checkpoint(),
        }

    def restore(self, snap: dict[str, Any]) -> None:
        """Restore the run state and RNG pipeline from a checkpoint snapshot."""
        self.seed = snap["seed"]
        self.rng.restore(snap["rng"])
        self.state.restore(snap["state"])
        self.events = EventResolver(self.rng)

    def fork(self) -> BalatroRun:
        """Create an independent deep copy of this run simulation."""
        cloned = BalatroRun(self.seed, validate=False)
        cloned.rng = self.rng.fork()
        cloned.state = self.state.clone()
        cloned.events = EventResolver(cloned.rng)
        return cloned

    def clone(self) -> BalatroRun:
        """Alias for `fork()`."""
        return self.fork()

