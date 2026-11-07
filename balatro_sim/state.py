"""Game State & Deck Tracking for Balatro Simulation.

Tracks active run parameters with 100% parity to `balatro_source`:
- seed (e.g. "HELLO")
- ante (1-based)
- round within ante (1=Small, 2=Big, 3=Boss) and total round count
- deck: list of active card instances (Suit, Rank, Enhancement, Edition, Seal)
  ordered by Balatro's exact `sort_id` (`C_2..C_T, D_2..D_T, H_2..H_T, S_2..S_T`)
  and supporting add, remove, convert, destroy mutations
- owned_jokers: list of active jokers and their positions
- used_vouchers & shop_vouchers: tracks redeemed and currently visible shop vouchers
- bosses_used: tracks boss blind generation counts for `get_new_boss()` parity
- visible_hands: tracks visible poker hands (`G.GAME.hands`) for To Do List & Orbital Tag
- active_tarots / banned_keys: tracks item availability for `get_current_pool()` parity
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from balatro_sim.rng import BalatroRNG


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

    @property
    def balatro_name(self) -> str:
        match self:
            case Seal.RED:
                return "Red"
            case Seal.BLUE:
                return "Blue"
            case Seal.GOLD:
                return "Gold"
            case Seal.PURPLE:
                return "Purple"
            case _:
                return "None"

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


# Default poker hands order in G.GAME.hands (game.lua:2001-2014)
ALL_POKER_HANDS_ORDER: list[str] = [
    "Flush Five",       # order = 1, visible = False by default
    "Flush House",      # order = 2, visible = False by default
    "Five of a Kind",   # order = 3, visible = False by default
    "Straight Flush",   # order = 4, visible = True
    "Four of a Kind",   # order = 5, visible = True
    "Full House",       # order = 6, visible = True
    "Flush",            # order = 7, visible = True
    "Straight",         # order = 8, visible = True
    "Three of a Kind",  # order = 9, visible = True
    "Two Pair",         # order = 10, visible = True
    "Pair",             # order = 11, visible = True
    "High Card",        # order = 12, visible = True
]

DEFAULT_VISIBLE_POKER_HANDS: list[str] = [
    "Straight Flush",
    "Four of a Kind",
    "Full House",
    "Flush",
    "Straight",
    "Three of a Kind",
    "Two Pair",
    "Pair",
    "High Card",
]

# Lexicographical rank order used by Balatro's card_protos sort (`a.s .. a.r`):
# '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'J', 'K', 'Q', 'T'
LEXICOGRAPHICAL_RANKS: list[Rank] = [
    Rank.TWO,
    Rank.THREE,
    Rank.FOUR,
    Rank.FIVE,
    Rank.SIX,
    Rank.SEVEN,
    Rank.EIGHT,
    Rank.NINE,
    Rank.ACE,
    Rank.JACK,
    Rank.KING,
    Rank.QUEEN,
    Rank.TEN,
]

# Lexicographical suit order used by Balatro's card_protos sort (`a.s .. a.r`):
# 'C', 'D', 'H', 'S'
LEXICOGRAPHICAL_SUITS: list[Suit] = [
    Suit.CLUBS,
    Suit.DIAMONDS,
    Suit.HEARTS,
    Suit.SPADES,
]


@dataclass
class Card:
    """Card instance representation matching Balatro's `Card` object."""

    suit: Suit
    rank: Rank
    enhancement: Enhancement = Enhancement.NONE
    edition: Edition = Edition.NONE
    seal: Seal = Seal.NONE
    debuffed: bool = False
    bonus_chips: int = 0
    id: int = 0  # Corresponds to Balatro's `sort_id`

    @property
    def sort_id(self) -> int:
        return self.id

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
        """Check card suit, taking Wild Card and Stone Card enhancements into account."""
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
    """Active deck representation (`G.playing_cards`) with exact `sort_id` ordering."""

    def __init__(self, cards: list[Card] | None = None) -> None:
        self.cards: list[Card] = []
        self._next_id: int = 1
        if cards is not None:
            for c in cards:
                self.add_card(c)

    @classmethod
    def create_standard_52(cls) -> Deck:
        """Create a standard 52-card deck sorted in Balatro's exact `s..r` `sort_id` order.
        
        In `balatro_source/game.lua:2367-2373`, `card_protos` are sorted by
        `((a.s or '')..(a.r or '')...) < ((b.s or '')..(b.r or '')...)`,
        producing Clubs (2..9, A, J, K, Q, T), Diamonds, Hearts, Spades with `sort_id` 1..52.
        """
        cards: list[Card] = []
        card_id = 1
        for s in LEXICOGRAPHICAL_SUITS:
            for r in LEXICOGRAPHICAL_RANKS:
                cards.append(Card(suit=s, rank=r, id=card_id))
                card_id += 1
        return cls(cards)

    @classmethod
    def create_abandoned_40(cls) -> Deck:
        """Create an Abandoned Deck (40 cards, no Jack/Queen/King) in exact `sort_id` order."""
        cards: list[Card] = []
        card_id = 1
        for s in LEXICOGRAPHICAL_SUITS:
            for r in LEXICOGRAPHICAL_RANKS:
                if r not in (Rank.JACK, Rank.QUEEN, Rank.KING):
                    cards.append(Card(suit=s, rank=r, id=card_id))
                    card_id += 1
        return cls(cards)

    @classmethod
    def create_checkered_52(cls) -> Deck:
        """Create a Checkered Deck (26 Spades, 26 Hearts) in exact `sort_id` order.
        
        In `balatro_source/back.lua:239-253`, Clubs (`sort_id` 1..13) are converted to Spades
        and Diamonds (`sort_id` 14..26) are converted to Hearts.
        """
        deck = cls.create_standard_52()
        for c in deck.cards:
            if c.suit == Suit.CLUBS:
                c.suit = Suit.SPADES
            elif c.suit == Suit.DIAMONDS:
                c.suit = Suit.HEARTS
        return deck

    @classmethod
    def create_erratic_52(cls, rng: BalatroRNG) -> Deck:
        """Create an Erratic Deck (52 randomized cards using stream `'erratic'`) in exact `sort_id` order."""
        from balatro_sim.events import P_CARDS_KEYS, parse_card_key

        protos: list[str] = []
        for _ in range(52):
            k, _ = rng.choice("erratic", P_CARDS_KEYS)
            protos.append(k)
        # Sort by s..r just like game.lua:2367 (`C_2` < `C_3` < ... < `S_T`)
        protos.sort(key=lambda key: key.replace("_", ""))
        cards: list[Card] = []
        for idx, k in enumerate(protos, start=1):
            suit, rank = parse_card_key(k)
            cards.append(Card(suit=suit, rank=rank, id=idx))
        return cls(cards)

    def add_card(self, card: Card) -> Card:
        """Add a card to the deck, assigning the next `sort_id` if unset."""
        if card.id <= 0:
            card.id = self._next_id
            self._next_id += 1
        else:
            if card.id >= self._next_id:
                self._next_id = card.id + 1
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
        """Destroy a card (alias for remove_card)."""
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
        """Mutate an existing card's attributes in place (preserving its `sort_id`)."""
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

    def get_non_stone_cards(self) -> list[Card]:
        """Return all cards where `v.ability.effect ~= 'Stone Card'`, sorted by `sort_id`.
        
        Used by `reset_idol_card()`, `reset_mail_rank()`, and `reset_castle_card()`,
        since `pseudorandom_element` sorts `Card` objects by `a.v.sort_id < b.v.sort_id`.
        """
        valid = [
            c
            for c in self.cards
            if c.enhancement != Enhancement.STONE and c.rank != Rank.NONE and c.suit != Suit.NONE
        ]
        valid.sort(key=lambda c: c.id)
        return valid

    def get_valid_rank_cards(self) -> list[Card]:
        """Alias for `get_non_stone_cards()`."""
        return self.get_non_stone_cards()

    def get_rank_histogram(self) -> dict[Rank, int]:
        """Return counts of each rank in current deck (excluding Stone cards)."""
        hist: Counter[Rank] = Counter()
        for c in self.get_non_stone_cards():
            hist[c.rank] += 1
        return dict(hist)

    def get_suit_histogram(self) -> dict[Suit, int]:
        """Return counts of each base suit in current deck (excluding Stone cards)."""
        hist: Counter[Suit] = Counter()
        for c in self.get_non_stone_cards():
            hist[c.suit] += 1
        return dict(hist)

    def clone(self) -> Deck:
        cloned_cards = [c.clone() for c in self.cards]
        deck = Deck(cloned_cards)
        deck._next_id = self._next_id
        return deck

    def __len__(self) -> int:
        return len(self.cards)

    def __iter__(self) -> Iterator[Card]:
        return iter(self.cards)

    def __getitem__(self, index: int) -> Card:
        return self.cards[index]

    def __contains__(self, item: object) -> bool:
        return item in self.cards


@dataclass
class GameState:
    """Active game state tracking full run parameters and lifecycle transitions."""

    seed: str
    ante: int = 1
    round_within_ante: int = 1  # 1=Small Blind, 2=Big Blind, 3=Boss Blind
    total_rounds: int = 0
    deck: Deck = field(default_factory=Deck.create_standard_52)
    deck_type: str = "Red Deck"
    owned_jokers: list[JokerInstance] = field(default_factory=list)
    stake: Stake = Stake.WHITE
    hand: list[Card] = field(default_factory=list)
    discards_made_this_round: int = 0
    hands_played_this_round: int = 0
    money: int = 4
    round_active: bool = True
    win_ante: int = 8

    # Dynamic round targets (G.GAME.current_round)
    ancient_suit: Suit | None = None
    castle_suit: Suit = Suit.SPADES
    idol_card: tuple[Rank, Suit] = (Rank.ACE, Suit.SPADES)
    mail_rank: Rank = Rank.ACE
    to_do_hand: str | None = None

    # Global pool / progression tracking (G.GAME)
    used_vouchers: set[str] = field(default_factory=set)
    shop_vouchers: list[str] = field(default_factory=list)
    bosses_used: dict[str, int] = field(default_factory=dict)
    current_boss: str | None = None
    banned_keys: set[str] = field(default_factory=set)
    active_tarots: set[str] = field(default_factory=set)
    visible_hands: list[str] = field(
        default_factory=lambda: list(DEFAULT_VISIBLE_POKER_HANDS)
    )
    pool_flags: dict[str, bool] = field(default_factory=dict)

    start_of_round_hooks: list[Callable[[GameState], None]] = field(default_factory=list)

    @property
    def round(self) -> int:
        """Alias for `round_within_ante`."""
        return self.round_within_ante

    @round.setter
    def round(self, value: int) -> None:
        self.round_within_ante = value

    @property
    def total_rounds_played(self) -> int:
        """Alias for `total_rounds`."""
        return self.total_rounds

    @total_rounds_played.setter
    def total_rounds_played(self, value: int) -> None:
        self.total_rounds = value

    @property
    def oops_count(self) -> int:
        """Count how many active Oops! All 6s jokers are owned."""
        return sum(1 for j in self.owned_jokers if "oops" in j.name.lower())

    @property
    def has_showman(self) -> bool:
        """Check whether Showman ('Ring Master') is currently owned."""
        return any(
            "showman" in j.name.lower() or "ring_master" in j.name.lower()
            for j in self.owned_jokers
        )

    @property
    def probability_modifier(self) -> float:
        """Probability multiplier from Oops! All 6s (doubles per joker)."""
        return float(2 ** self.oops_count)

    def unlock_secret_hand(self, hand_name: str) -> None:
        """Make a secret poker hand ('Flush Five', 'Flush House', 'Five of a Kind') visible.
        
        Preserves Balatro's `G.GAME.hands` canonical order.
        """
        if hand_name not in ALL_POKER_HANDS_ORDER:
            raise ValueError(f"Unknown poker hand: {hand_name}")
        if hand_name not in self.visible_hands:
            new_visible_set = set(self.visible_hands) | {hand_name}
            self.visible_hands = [
                h for h in ALL_POKER_HANDS_ORDER if h in new_visible_set
            ]

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
            "deck_type": self.deck_type,
            "stake": self.stake.value,
            "money": self.money,
            "round_active": self.round_active,
            "win_ante": self.win_ante,
            "discards_made_this_round": self.discards_made_this_round,
            "hands_played_this_round": self.hands_played_this_round,
            "ancient_suit": self.ancient_suit.value if self.ancient_suit else None,
            "castle_suit": self.castle_suit.value,
            "idol_card": [self.idol_card[0].value, self.idol_card[1].value],
            "mail_rank": self.mail_rank.value,
            "to_do_hand": self.to_do_hand,
            "used_vouchers": sorted(self.used_vouchers),
            "shop_vouchers": list(self.shop_vouchers),
            "bosses_used": dict(self.bosses_used),
            "current_boss": self.current_boss,
            "banned_keys": sorted(self.banned_keys),
            "active_tarots": sorted(self.active_tarots),
            "visible_hands": list(self.visible_hands),
            "pool_flags": dict(self.pool_flags),
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
        self.deck_type = snap.get("deck_type", "Red Deck")
        self.stake = Stake(snap["stake"])
        self.money = snap["money"]
        self.round_active = snap["round_active"]
        self.win_ante = snap.get("win_ante", 8)
        self.discards_made_this_round = snap.get("discards_made_this_round", 0)
        self.hands_played_this_round = snap.get("hands_played_this_round", 0)

        anc = snap.get("ancient_suit")
        self.ancient_suit = Suit(anc) if anc else None
        self.castle_suit = Suit(snap.get("castle_suit", Suit.SPADES.value))
        idol_raw = snap.get("idol_card", [Rank.ACE.value, Suit.SPADES.value])
        self.idol_card = (Rank(idol_raw[0]), Suit(idol_raw[1]))
        self.mail_rank = Rank(snap.get("mail_rank", Rank.ACE.value))
        self.to_do_hand = snap.get("to_do_hand")

        self.used_vouchers = set(snap.get("used_vouchers", []))
        self.shop_vouchers = list(snap.get("shop_vouchers", []))
        self.bosses_used = dict(snap.get("bosses_used", {}))
        self.current_boss = snap.get("current_boss")
        self.banned_keys = set(snap.get("banned_keys", []))
        self.active_tarots = set(snap.get("active_tarots", []))
        self.visible_hands = list(snap.get("visible_hands", DEFAULT_VISIBLE_POKER_HANDS))
        self.pool_flags = dict(snap.get("pool_flags", {}))

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
        cloned = GameState(seed=self.seed)
        cloned.restore(self.checkpoint())
        cloned.hand = [c.clone() for c in self.hand]
        cloned.start_of_round_hooks = list(self.start_of_round_hooks)
        return cloned
