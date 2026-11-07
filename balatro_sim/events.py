"""Event Mechanics & Roll Solvers for Balatro Simulation.

Implements exact deterministic roll evaluations for Balatro events:
- Certificate: generates card front from G.P_CARDS ('cert_fr' + ante)
  and seal from G.P_SEALS ('cert_sl' + ante), adding to hand/deck
- Lucky Cards: evaluates +$20 cash (1 in 15 base) and +20 Mult (1 in 5 base)
  via 'lucky_money' and 'lucky_mult' stream keys, respecting Oops! All 6s
- Mail-in Rebate: dynamically chooses target rank from deck composition
  and evaluates discard payout ($5 per match)
- 1-in-N Outcome Resolvers: Wheel of Fortune, Space Joker, Bloodstone,
  Glass Cards, Business Card, Gros Michel, Cavendish
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from balatro_sim.rng import BalatroRNG, Tw223
from balatro_sim.state import Card, Deck, Edition, Enhancement, GameState, Rank, Seal, Suit

# ---------------------------------------------------------------------------
# G.P_CARDS Table (52 Standard Playing Card Fronts in Balatro Lexicographical Order)
# ---------------------------------------------------------------------------

P_CARDS_KEYS: list[str] = [
    "C_2", "C_3", "C_4", "C_5", "C_6", "C_7", "C_8", "C_9", "C_A", "C_J", "C_K", "C_Q", "C_T",
    "D_2", "D_3", "D_4", "D_5", "D_6", "D_7", "D_8", "D_9", "D_A", "D_J", "D_K", "D_Q", "D_T",
    "H_2", "H_3", "H_4", "H_5", "H_6", "H_7", "H_8", "H_9", "H_A", "H_J", "H_K", "H_Q", "H_T",
    "S_2", "S_3", "S_4", "S_5", "S_6", "S_7", "S_8", "S_9", "S_A", "S_J", "S_K", "S_Q", "S_T",
]

_SUIT_LOOKUP: dict[str, Suit] = {
    "C": Suit.CLUBS,
    "D": Suit.DIAMONDS,
    "H": Suit.HEARTS,
    "S": Suit.SPADES,
}

_RANK_LOOKUP: dict[str, Rank] = {
    "2": Rank.TWO,
    "3": Rank.THREE,
    "4": Rank.FOUR,
    "5": Rank.FIVE,
    "6": Rank.SIX,
    "7": Rank.SEVEN,
    "8": Rank.EIGHT,
    "9": Rank.NINE,
    "T": Rank.TEN,
    "J": Rank.JACK,
    "Q": Rank.QUEEN,
    "K": Rank.KING,
    "A": Rank.ACE,
}

P_SEALS_KEYS: list[Seal] = [
    Seal.BLUE,
    Seal.GOLD,
    Seal.PURPLE,
    Seal.RED,
]


def parse_card_key(key: str) -> tuple[Suit, Rank]:
    """Parse Balatro card front key into (Suit, Rank), e.g. 'C_Q' -> (Clubs, Queen)."""
    parts = key.split("_")
    suit = _SUIT_LOOKUP[parts[0]]
    rank = _RANK_LOOKUP[parts[1]]
    return suit, rank


# ---------------------------------------------------------------------------
# Certificate Event
# ---------------------------------------------------------------------------

@dataclass
class CertificateResult:
    """Result of Certificate joker start-of-round card generation."""

    card: Card
    front_key: str
    seal_key: str
    ante: int

    @property
    def suit(self) -> Suit:
        return self.card.suit

    @property
    def rank(self) -> Rank:
        return self.card.rank

    @property
    def seal(self) -> Seal:
        return self.card.seal


def predict_certificate(
    rng: BalatroRNG,
    ante: int,
    state: GameState | None = None,
    add_to_state: bool = True,
) -> CertificateResult:
    """Predict and generate Certificate joker card for the specified ante.
    
    Balatro mechanics:
    - Card front is pulled from G.P_CARDS using stream key 'cert_fr' + str(ante)
    - Card seal is pulled from G.P_SEALS using stream key 'cert_sl' + str(ante)
    - Appends the generated card directly into state's hand and deck if state is provided
    """
    key_fr = f"cert_fr{ante}"
    key_sl = f"cert_sl{ante}"

    # Draw card front
    chosen_front_key, _ = rng.choice(key_fr, P_CARDS_KEYS)
    suit, rank = parse_card_key(chosen_front_key)

    # Draw seal
    chosen_seal, _ = rng.choice(key_sl, P_SEALS_KEYS)

    generated_card = Card(suit=suit, rank=rank, seal=chosen_seal)

    if state is not None and add_to_state:
        state.deck.add_card(generated_card)
        state.hand.append(generated_card)

    return CertificateResult(
        card=generated_card,
        front_key=chosen_front_key,
        seal_key=chosen_seal.value,
        ante=ante,
    )


# ---------------------------------------------------------------------------
# Lucky Card Evaluation
# ---------------------------------------------------------------------------

@dataclass
class LuckyHitResult:
    """Outcome of Lucky Card probability checks."""

    mult: bool  # 1 in 5 base (+20 Mult)
    money: bool  # 1 in 15 base (+$20 cash)
    mult_roll: float
    money_roll: float
    oops_multiplier: float

    @property
    def any_hit(self) -> bool:
        return self.mult or self.money

    def to_dict(self) -> dict[str, bool]:
        return {"mult": self.mult, "money": self.money}


def evaluate_lucky_hit(
    rng: BalatroRNG,
    oops_count: int = 0,
    probability_modifier: float | None = None,
) -> LuckyHitResult:
    """Evaluate deterministic mid-round triggers for a scored Lucky Card.
    
    Balatro rolls separate checks:
    - +20 Mult: 1 in 5 base chance on 'lucky_mult'
    - +$20 Cash: 1 in 15 base chance on 'lucky_money'
    Both are scaled by Oops! All 6s (doubles probability per active joker).
    """
    modifier = (
        probability_modifier
        if probability_modifier is not None
        else float(2 ** oops_count)
    )

    roll_mult = rng.random("lucky_mult")
    roll_money = rng.random("lucky_money")

    mult_success = roll_mult < ((1.0 * modifier) / 5.0)
    money_success = roll_money < ((1.0 * modifier) / 15.0)

    return LuckyHitResult(
        mult=mult_success,
        money=money_success,
        mult_roll=roll_mult,
        money_roll=roll_money,
        oops_multiplier=modifier,
    )


# ---------------------------------------------------------------------------
# Mail-in Rebate Event
# ---------------------------------------------------------------------------

@dataclass
class RebateEvaluation:
    """Outcome of Mail-in Rebate discard evaluation."""

    target_rank: Rank
    matching_cards: list[Card]
    matches: int
    payout: int
    payout_per_card: int = 5


def get_mail_in_rebate_target(
    rng: BalatroRNG,
    deck: Deck,
    ante: int | None = None,
    stream_key: str | None = None,
) -> Rank:
    """Dynamically determine Mail-in Rebate target rank for the round.
    
    Balatro chooses proportionally from the actual card ranks present in the active deck:
    - If deck has no valid cards with rank (e.g. 100% stone cards or empty), defaults to Rank.ACE
    - Otherwise, picks via pseudorandom_element on current deck valid cards
    """
    valid_cards = deck.get_valid_rank_cards()
    if not valid_cards:
        return Rank.ACE

    # Standard stream key in Balatro is 'mail' + str(ante) or 'mail'
    key = stream_key if stream_key is not None else (f"mail{ante}" if ante is not None else "mail")

    chosen_card, _ = rng.choice(key, valid_cards)
    return chosen_card.rank


def evaluate_mail_in_rebate_discard(
    target_rank: Rank,
    discarded_cards: Sequence[Card],
    payout_per_card: int = 5,
) -> RebateEvaluation:
    """Evaluate payout for discarding cards when Mail-in Rebate is active."""
    matches = [c for c in discarded_cards if c.is_rank(target_rank)]
    count = len(matches)
    total_payout = count * payout_per_card
    return RebateEvaluation(
        target_rank=target_rank,
        matching_cards=matches,
        matches=count,
        payout=total_payout,
        payout_per_card=payout_per_card,
    )


# ---------------------------------------------------------------------------
# 1-in-N Outcome Resolvers
# ---------------------------------------------------------------------------

class EventResolver:
    """Deterministic roll solvers for probability-based Balatro jokers and cards."""

    def __init__(self, rng: BalatroRNG) -> None:
        self.rng = rng

    def check_odds(
        self,
        stream_key: str,
        numerator: int,
        denominator: int,
        oops_multiplier: float = 1.0,
    ) -> bool:
        """Resolve a deterministic 1-in-N outcome using stream_key."""
        roll = self.rng.random(stream_key)
        threshold = (numerator * oops_multiplier) / float(denominator)
        return roll < threshold

    def wheel_of_fortune(self, oops_multiplier: float = 1.0) -> bool:
        """Resolve The Wheel of Fortune (1 in 4 base chance). Stream: 'wheel_of_fortune'."""
        return self.check_odds("wheel_of_fortune", 1, 4, oops_multiplier)

    def space_joker(self, oops_multiplier: float = 1.0) -> bool:
        """Resolve Space Joker hand level upgrade (1 in 4 base chance). Stream: 'space_joker'."""
        return self.check_odds("space_joker", 1, 4, oops_multiplier)

    def bloodstone(self, oops_multiplier: float = 1.0, base_denominator: int = 2) -> bool:
        """Resolve Bloodstone extra mult check (1 in 2 base in current Balatro). Stream: 'bloodstone'."""
        return self.check_odds("bloodstone", 1, base_denominator, oops_multiplier)

    def glass_card_shatter(self, oops_multiplier: float = 1.0) -> bool:
        """Resolve Glass Card shatter check upon scoring (1 in 4 base chance). Stream: 'glass'."""
        return self.check_odds("glass", 1, 4, oops_multiplier)

    def business_card(self, oops_multiplier: float = 1.0) -> bool:
        """Resolve Business Card $2 payout on face card scoring (1 in 2 base chance). Stream: 'business_card'."""
        return self.check_odds("business_card", 1, 2, oops_multiplier)

    def gros_michel_extinct(self, oops_multiplier: float = 1.0) -> bool:
        """Resolve Gros Michel extinction check at end of round (1 in 6 base chance). Stream: 'gros_michel'."""
        return self.check_odds("gros_michel", 1, 6, oops_multiplier)

    def cavendish_extinct(self, oops_multiplier: float = 1.0) -> bool:
        """Resolve Cavendish extinction check at end of round (1 in 1000 base chance). Stream: 'cavendish'."""
        return self.check_odds("cavendish", 1, 1000, oops_multiplier)

