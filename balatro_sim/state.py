"""Game State & Deck Tracking for Balatro Simulation.

Tracks active run parameters:
- seed (e.g. "HELLO")
- ante (1-based)
- round within ante (1=Small, 2=Big, 3=Boss) and total round count
- deck: list of active card instances (Suit, Rank, Enhancement, Edition, Seal)
  supporting add, remove, convert, destroy mutations
- owned_jokers: list of active jokers and their positions
- stake: modifiers affecting probabilities/costs
- round lifecycle transitions: start_ante, start_round, trigger_start_of_round_effects, end_round
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Any, Callable


class Suit(StrEnum):
    NONE = "None"
    SPADES = "Spades"
    HEARTS = "Hearts"
    CLUBS = "Clubs"
    DIAMONDS = "Diamonds"

    @property
    def short_code(self) -> str:
        match self:
            case Suit.SPADES:
                return "S"
            case Suit.HEARTS:
                return "H"
            case Suit.CLUBS:
                return "C"
            case Suit.DIAMONDS:
                return "D"
            case _:
                return "N"

    @classmethod
    def from_str(cls, s: str) -> Suit:
        val = s.strip().lower()
        if val in ("s", "spades", "spade"):
            return cls.SPADES
        if val in ("h", "hearts", "heart"):
            return cls.HEARTS
        if val in ("c", "clubs", "club"):
            return cls.CLUBS
        if val in ("d", "diamonds", "diamond"):
            return cls.DIAMONDS
        return cls.NONE


class Rank(StrEnum):
    NONE = "None"
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "10"
    JACK = "Jack"
    QUEEN = "Queen"
    KING = "King"
    ACE = "Ace"

    @property
    def short_code(self) -> str:
        match self:
            case Rank.TEN:
                return "T"
            case Rank.JACK:
                return "J"
            case Rank.QUEEN:
                return "Q"
            case Rank.KING:
                return "K"
            case Rank.ACE:
                return "A"
            case Rank.NONE:
                return "N"
            case _:
                return self.value

    @property
    def is_face(self) -> bool:
        return self in (Rank.JACK, Rank.QUEEN, Rank.KING)

    @property
    def nominal_chips(self) -> int:
        match self:
            case Rank.ACE:
                return 11
            case Rank.KING | Rank.QUEEN | Rank.JACK | Rank.TEN:
                return 10
            case Rank.NONE:
                return 0
            case _:
                return int(self.value)

    @classmethod
    def from_str(cls, s: str) -> Rank:
        val = s.strip().upper()
        mapping = {
            "2": cls.TWO,
            "3": cls.THREE,
            "4": cls.FOUR,
            "5": cls.FIVE,
            "6": cls.SIX,
            "7": cls.SEVEN,
            "8": cls.EIGHT,
            "9": cls.NINE,
            "10": cls.TEN,
            "T": cls.TEN,
            "J": cls.JACK,
            "JACK": cls.JACK,
            "Q": cls.QUEEN,
            "QUEEN": cls.QUEEN,
            "K": cls.KING,
            "KING": cls.KING,
            "A": cls.ACE,
            "ACE": cls.ACE,
        }
        return mapping.get(val, cls.NONE)


class Enhancement(StrEnum):
    NONE = "none"
    BONUS = "bonus"
    MULT = "mult"
    WILD = "wild"
    GLASS = "glass"
    STEEL = "steel"
    STONE = "stone"
    GOLD = "gold"
    LUCKY = "lucky"


class Edition(StrEnum):
    NONE = "none"
    FOIL = "foil"
    HOLOGRAPHIC = "holographic"
    POLYCHROME = "polychrome"
    NEGATIVE = "negative"


class Seal(StrEnum):
    NONE = "none"
    RED = "red"
    BLUE = "blue"
    GOLD = "gold"
    PURPLE = "purple"

    @classmethod
    def from_str(cls, s: str) -> Seal:
        val = s.strip().lower()
        if "red" in val:
            return cls.RED
        if "blue" in val:
            return cls.BLUE
        if "gold" in val:
            return cls.GOLD
        if "purple" in val:
            return cls.PURPLE
        return cls.NONE


class Stake(StrEnum):
    WHITE = "White Stake"
    RED = "Red Stake"
    GREEN = "Green Stake"
    BLACK = "Black Stake"
    BLUE = "Blue Stake"
    PURPLE = "Purple Stake"
    ORANGE = "Orange Stake"
    GOLD = "Gold Stake"


@dataclass
class Card:
    """Card instance representation."""

    suit: Suit
    rank: Rank
    enhancement: Enhancement = Enhancement.NONE
    edition: Edition = Edition.NONE
    seal: Seal = Seal.NONE
    debuffed: bool = False
    bonus_chips: int = 0
    id: int = 0

    @property
    def is_face(self) -> bool:
        """Face cards are Jack, Queen, King (unless Stone card)."""
        if self.enhancement == Enhancement.STONE:
            return False
        return self.rank.is_face

    @property
    def card_key(self) -> str:
        """Balatro key representation like 'C_Q' or 'S_A'."""
        return f"{self.suit.short_code}_{self.rank.short_code}"

    def is_suit(self, suit: Suit | str) -> bool:
        """Check card suit, taking Wild Card enhancement into account."""
        if self.enhancement == Enhancement.STONE:
            return False
        if self.enhancement == Enhancement.WILD:
            return True
        target = Suit.from_str(str(suit)) if isinstance(suit, str) else suit
        return self.suit == target

    def is_rank(self, rank: Rank | str) -> bool:
        """Check card rank (Stone card masks rank)."""
        if self.enhancement == Enhancement.STONE:
            return False
        target = Rank.from_str(str(rank)) if isinstance(rank, str) else rank
        return self.rank == target

    def clone(self) -> Card:
        return copy.copy(self)

    def __repr__(self) -> str:
        enh = f", {self.enhancement.value}" if self.enhancement != Enhancement.NONE else ""
        edi = f", {self.edition.value}" if self.edition != Edition.NONE else ""
        sl = f", {self.seal.value} seal" if self.seal != Seal.NONE else ""
        return f"Card({self.rank.value} of {self.suit.value}{enh}{edi}{sl})"


@dataclass
class JokerInstance:
    """Active Joker card in a run."""

    name: str
    position: int = 0
    edition: Edition = Edition.NONE
    eternal: bool = False
    perishable: bool = False
    rental: bool = False
    state: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> JokerInstance:
        return copy.deepcopy(self)


class Deck:
    """Active deck representation with full mutation capabilities."""

    def __init__(self, cards: list[Card] | None = None) -> None:
        self.cards: list[Card] = cards if cards is not None else []
        self._next_id: int = max([c.id for c in self.cards], default=0) + 1

    @classmethod
    def create_standard_52(cls) -> Deck:
        """Create a standard Balatro 52-card deck."""
        cards: list[Card] = []
        card_id = 1
        for s in [Suit.SPADES, Suit.HEARTS, Suit.CLUBS, Suit.DIAMONDS]:
            for r in [
                Rank.TWO,
                Rank.THREE,
                Rank.FOUR,
                Rank.FIVE,
                Rank.SIX,
                Rank.SEVEN,
                Rank.EIGHT,
                Rank.NINE,
                Rank.TEN,
                Rank.JACK,
                Rank.QUEEN,
                Rank.KING,
                Rank.ACE,
            ]:
                cards.append(Card(suit=s, rank=r, id=card_id))
                card_id += 1
        return cls(cards)

    def add_card(self, card: Card) -> Card:
        """Add a card to the deck."""
        if card.id == 0:
            card.id = self._next_id
            self._next_id += 1
        self.cards.append(card)
        return card

    def remove_card(self, card_or_id: Card | int) -> Card | None:
        """Remove a card from the deck by instance or ID."""
        target_id = card_or_id.id if isinstance(card_or_id, Card) else card_or_id
        for i, c in enumerate(self.cards):
            if c.id == target_id:
                return self.cards.pop(i)
        return None

    def destroy_card(self, card_or_id: Card | int) -> Card | None:
        """Destroy a card (alias for remove_card with logging/tracking)."""
        return self.remove_card(card_or_id)

    def convert_card(
        self,
        card_or_id: Card | int,
        new_suit: Suit | str | None = None,
        new_rank: Rank | str | None = None,
        new_enhancement: Enhancement | str | None = None,
        new_edition: Edition | str | None = None,
        new_seal: Seal | str | None = None,
    ) -> Card | None:
        """Mutate an existing card's attributes in place."""
        target_id = card_or_id.id if isinstance(card_or_id, Card) else card_or_id
        target = next((c for c in self.cards if c.id == target_id), None)
        if target is None:
            return None
        if new_suit is not None:
            target.suit = Suit.from_str(str(new_suit)) if isinstance(new_suit, str) else new_suit
        if new_rank is not None:
            target.rank = Rank.from_str(str(new_rank)) if isinstance(new_rank, str) else new_rank
        if new_enhancement is not None:
            target.enhancement = (
                Enhancement(new_enhancement)
                if isinstance(new_enhancement, str)
                else new_enhancement
            )
        if new_edition is not None:
            target.edition = (
                Edition(new_edition) if isinstance(new_edition, str) else new_edition
            )
        if new_seal is not None:
            target.seal = Seal.from_str(str(new_seal)) if isinstance(new_seal, str) else new_seal
        return target

    def get_valid_rank_cards(self) -> list[Card]:
        """Cards that possess a valid rank (non-Stone cards)."""
        return [c for c in self.cards if c.enhancement != Enhancement.STONE and c.rank != Rank.NONE]

    def get_rank_histogram(self) -> dict[Rank, int]:
        """Return counts of each rank in current deck."""
        hist: dict[Rank, int] = Counter()
        for c in self.get_valid_rank_cards():
            hist[c.rank] += 1
        return dict(hist)

    def clone(self) -> Deck:
        cloned_cards = [c.clone() for c in self.cards]
        deck = Deck(cloned_cards)
        deck._next_id = self._next_id
        return deck

    def __len__(self) -> int:
        return len(self.cards)


@dataclass
class GameState:
    """Active game state tracking full run parameters and lifecycle transitions."""

    seed: str
    ante: int = 1
    round_within_ante: int = 1  # 1=Small Blind, 2=Big Blind, 3=Boss Blind
    total_rounds: int = 1
    deck: Deck = field(default_factory=Deck.create_standard_52)
    owned_jokers: list[JokerInstance] = field(default_factory=list)
    stake: Stake = Stake.WHITE
    hand: list[Card] = field(default_factory=list)
    discards_made_this_round: int = 0
    hands_played_this_round: int = 0
    money: int = 4
    round_active: bool = True
    start_of_round_hooks: list[Callable[[GameState], None]] = field(default_factory=list)

    @property
    def oops_count(self) -> int:
        """Count how many active Oops! All 6s jokers are owned."""
        return sum(
            1 for j in self.owned_jokers if "oops" in j.name.lower()
        )

    @property
    def probability_modifier(self) -> float:
        """Probability multiplier from Oops! All 6s (doubles per joker)."""
        return float(2 ** self.oops_count)

    def add_joker(
        self,
        name: str,
        edition: Edition = Edition.NONE,
        eternal: bool = False,
        perishable: bool = False,
        rental: bool = False,
        state: dict[str, Any] | None = None,
    ) -> JokerInstance:
        """Add a joker to the owned jokers list."""
        pos = len(self.owned_jokers)
        joker = JokerInstance(
            name=name,
            position=pos,
            edition=edition,
            eternal=eternal,
            perishable=perishable,
            rental=rental,
            state=state if state is not None else {},
        )
        self.owned_jokers.append(joker)
        return joker

    def remove_joker(self, target: int | str) -> JokerInstance | None:
        """Remove a joker by index or name."""
        if isinstance(target, int):
            if 0 <= target < len(self.owned_jokers):
                removed = self.owned_jokers.pop(target)
                for idx, j in enumerate(self.owned_jokers):
                    j.position = idx
                return removed
            return None
        target_name = target.lower()
        for idx, j in enumerate(self.owned_jokers):
            if j.name.lower() == target_name:
                removed = self.owned_jokers.pop(idx)
                for i, remaining in enumerate(self.owned_jokers):
                    remaining.position = i
                return removed
        return None

    def find_joker(self, name: str) -> list[JokerInstance]:
        """Find active jokers matching name."""
        target = name.lower()
        return [j for j in self.owned_jokers if target in j.name.lower()]

    def has_joker(self, name: str) -> bool:
        """Check whether any active joker matches name."""
        return len(self.find_joker(name)) > 0

    # -----------------------------------------------------------------------
    # Lifecycle Transitions
    # -----------------------------------------------------------------------

    def start_ante(self, ante_num: int) -> None:
        """Transition into a new ante."""
        self.ante = ante_num
        self.round_within_ante = 0

    def start_round(self, round_num: int | None = None) -> None:
        """Begin a new round within current ante."""
        if round_num is not None:
            self.round_within_ante = round_num
        else:
            self.round_within_ante += 1
        self.total_rounds += 1
        self.discards_made_this_round = 0
        self.hands_played_this_round = 0
        self.round_active = True

        self.trigger_start_of_round_effects()

    def trigger_start_of_round_effects(self) -> None:
        """Fire all registered start-of-round callbacks and hooks."""
        for hook in list(self.start_of_round_hooks):
            hook(self)

    def end_round(self) -> None:
        """Conclude the active round."""
        self.round_active = False

    # -----------------------------------------------------------------------
    # Checkpointing & Forking
    # -----------------------------------------------------------------------

    def checkpoint(self) -> dict[str, Any]:
        """Produce a serializable snapshot of the game state."""
        return {
            "seed": self.seed,
            "ante": self.ante,
            "round_within_ante": self.round_within_ante,
            "total_rounds": self.total_rounds,
            "stake": self.stake.value,
            "money": self.money,
            "round_active": self.round_active,
            "discards_made_this_round": self.discards_made_this_round,
            "hands_played_this_round": self.hands_played_this_round,
            "deck": [
                {
                    "suit": c.suit.value,
                    "rank": c.rank.value,
                    "enhancement": c.enhancement.value,
                    "edition": c.edition.value,
                    "seal": c.seal.value,
                    "debuffed": c.debuffed,
                    "bonus_chips": c.bonus_chips,
                    "id": c.id,
                }
                for c in self.deck.cards
            ],
            "owned_jokers": [
                {
                    "name": j.name,
                    "position": j.position,
                    "edition": j.edition.value,
                    "eternal": j.eternal,
                    "perishable": j.perishable,
                    "rental": j.rental,
                    "state": copy.deepcopy(j.state),
                }
                for j in self.owned_jokers
            ],
        }

    def restore(self, snap: dict[str, Any]) -> None:
        """Restore state from snapshot."""
        self.seed = snap["seed"]
        self.ante = snap["ante"]
        self.round_within_ante = snap["round_within_ante"]
        self.total_rounds = snap["total_rounds"]
        self.stake = Stake(snap["stake"])
        self.money = snap["money"]
        self.round_active = snap["round_active"]
        self.discards_made_this_round = snap.get("discards_made_this_round", 0)
        self.hands_played_this_round = snap.get("hands_played_this_round", 0)

        restored_cards = [
            Card(
                suit=Suit(c["suit"]),
                rank=Rank(c["rank"]),
                enhancement=Enhancement(c["enhancement"]),
                edition=Edition(c["edition"]),
                seal=Seal(c["seal"]),
                debuffed=c.get("debuffed", False),
                bonus_chips=c.get("bonus_chips", 0),
                id=c.get("id", 0),
            )
            for c in snap["deck"]
        ]
        self.deck = Deck(restored_cards)

        self.owned_jokers = [
            JokerInstance(
                name=j["name"],
                position=j["position"],
                edition=Edition(j["edition"]),
                eternal=j.get("eternal", False),
                perishable=j.get("perishable", False),
                rental=j.get("rental", False),
                state=copy.deepcopy(j.get("state", {})),
            )
            for j in snap["owned_jokers"]
        ]

    def clone(self) -> GameState:
        """Create an independent deep copy of this game state."""
        cloned = GameState(
            seed=self.seed,
            ante=self.ante,
            round_within_ante=self.round_within_ante,
            total_rounds=self.total_rounds,
            deck=self.deck.clone(),
            owned_jokers=[j.clone() for j in self.owned_jokers],
            stake=self.stake,
            hand=[c.clone() for c in self.hand],
            discards_made_this_round=self.discards_made_this_round,
            hands_played_this_round=self.hands_played_this_round,
            money=self.money,
            round_active=self.round_active,
        )
        return cloned

