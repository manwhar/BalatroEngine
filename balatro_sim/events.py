"""Event Mechanics & Roll Solvers for Balatro Simulation.

Verified against `balatro_source` (`card.lua`, `game.lua`, `tag.lua`, `back.lua`,
`functions/common_events.lua`, `functions/state_events.lua`, `functions/UI_definitions.lua`):
- Castle: suit selected each round (`reset_castle_card`, stream `'cas' + str(ante)`)
- Ancient Joker: suit selected each round (`reset_ancient_card`, stream `'anc' + str(ante)`)
- The Idol: card (rank + suit) selected each round (`reset_idol_card`, stream `'idol' + str(ante)`)
- Mail-in Rebate: rank selected each round (`reset_mail_rank`, stream `'mail' + str(ante)`)
- To Do List: poker hand selected on init and end of round (`'to_do'`)
- Cartomancer: Tarot card generated on blind select (`'Tarotcar' + str(ante)`)
- Vagabond: Tarot card generated when hand played with <= $4 (`'Tarotvag' + str(ante)`)
- Marble Joker / Stone Joker: hidden suit + rank of generated Stone Card (`'marb_fr'`)
- Boss Blinds: full run boss blind queue (`get_new_boss`, stream `'boss'`), including
  boss rerolls, Hieroglyph/Petroglyph (-1 ante vouchers), Showdown bosses, and `bosses_used` cycle
- Skip Tags: Small & Big blind tags per ante (`get_next_tag_key`, stream `'Tag' + str(ante)`),
  including `min_ante` resampling, `-1 ante` vouchers, and Orbital Tag hand selection (`'orbital'`)
- Vouchers: shop vouchers (`'Voucher' + str(ante)`) and Voucher Tag skips (`'Voucher_fromtag'`),
  adapting dynamically to purchased/redeemed vouchers, tier-2 prerequisites, and starting decks
- Certificate, Lucky Cards, Wheel of Fortune, Space Joker, Bloodstone, Glass, Business Card, etc.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from balatro_sim.rng import BalatroRNG
from balatro_sim.state import (
     DEFAULT_VISIBLE_POKER_HANDS,
    Card,
    Deck,
    Edition,
    Enhancement,
    GameState,
    Rank,
    Seal,
    Suit,
)

# ---------------------------------------------------------------------------
# 1. G.P_CARDS Table (52 Standard Playing Card Fronts in Lexicographical Key Order)
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
# 2. G.P_CENTER_POOLS['Tarot'] (22 Tarots sorted by order 1..22, game.lua:533-554)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TarotProto:
    key: str
    name: str
    order: int


P_TAROTS: list[TarotProto] = [
    TarotProto("c_fool", "The Fool", 1),
    TarotProto("c_magician", "The Magician", 2),
    TarotProto("c_high_priestess", "The High Priestess", 3),
    TarotProto("c_empress", "The Empress", 4),
    TarotProto("c_emperor", "The Emperor", 5),
    TarotProto("c_heirophant", "The Hierophant", 6),
    TarotProto("c_lovers", "The Lovers", 7),
    TarotProto("c_chariot", "The Chariot", 8),
    TarotProto("c_justice", "Justice", 9),
    TarotProto("c_hermit", "The Hermit", 10),
    TarotProto("c_wheel_of_fortune", "The Wheel of Fortune", 11),
    TarotProto("c_strength", "Strength", 12),
    TarotProto("c_hanged_man", "The Hanged Man", 13),
    TarotProto("c_death", "Death", 14),
    TarotProto("c_temperance", "Temperance", 15),
    TarotProto("c_devil", "The Devil", 16),
    TarotProto("c_tower", "The Tower", 17),
    TarotProto("c_star", "The Star", 18),
    TarotProto("c_moon", "The Moon", 19),
    TarotProto("c_sun", "The Sun", 20),
    TarotProto("c_judgement", "Judgement", 21),
    TarotProto("c_world", "The World", 22),
]

TAROT_BY_KEY: dict[str, TarotProto] = {t.key: t for t in P_TAROTS}
TAROT_BY_NAME: dict[str, TarotProto] = {t.name.lower(): t for t in P_TAROTS}


def normalize_tarot_key(key_or_name: str) -> str:
    """Convert a Tarot key ('c_fool') or name ('The Fool') to its canonical 'c_...' key."""
    clean = key_or_name.strip()
    if clean in TAROT_BY_KEY:
        return clean
    lower = clean.lower()
    if lower in TAROT_BY_NAME:
        return TAROT_BY_NAME[lower].key
    if f"the {lower}" in TAROT_BY_NAME:
        return TAROT_BY_NAME[f"the {lower}"].key
    return clean


# ---------------------------------------------------------------------------
# 3. G.P_CENTER_POOLS['Voucher'] (32 Vouchers sorted by order 1..32, game.lua:592-624)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VoucherProto:
    key: str
    name: str
    order: int
    requires: tuple[str, ...] = ()


P_VOUCHERS: list[VoucherProto] = [
    VoucherProto("v_overstock_norm", "Overstock", 1),
    VoucherProto("v_overstock_plus", "Overstock Plus", 2, ("v_overstock_norm",)),
    VoucherProto("v_clearance_sale", "Clearance Sale", 3),
    VoucherProto("v_liquidation", "Liquidation", 4, ("v_clearance_sale",)),
    VoucherProto("v_hone", "Hone", 5),
    VoucherProto("v_glow_up", "Glow Up", 6, ("v_hone",)),
    VoucherProto("v_reroll_surplus", "Reroll Surplus", 7),
    VoucherProto("v_reroll_glut", "Reroll Glut", 8, ("v_reroll_surplus",)),
    VoucherProto("v_crystal_ball", "Crystal Ball", 9),
    VoucherProto("v_omen_globe", "Omen Globe", 10, ("v_crystal_ball",)),
    VoucherProto("v_telescope", "Telescope", 11),
    VoucherProto("v_observatory", "Observatory", 12, ("v_telescope",)),
    VoucherProto("v_grabber", "Grabber", 13),
    VoucherProto("v_nacho_tong", "Nacho Tong", 14, ("v_grabber",)),
    VoucherProto("v_wasteful", "Wasteful", 15),
    VoucherProto("v_recyclomancy", "Recyclomancy", 16, ("v_wasteful",)),
    VoucherProto("v_tarot_merchant", "Tarot Merchant", 17),
    VoucherProto("v_tarot_tycoon", "Tarot Tycoon", 18, ("v_tarot_merchant",)),
    VoucherProto("v_planet_merchant", "Planet Merchant", 19),
    VoucherProto("v_planet_tycoon", "Planet Tycoon", 20, ("v_planet_merchant",)),
    VoucherProto("v_seed_money", "Seed Money", 21),
    VoucherProto("v_money_tree", "Money Tree", 22, ("v_seed_money",)),
    VoucherProto("v_blank", "Blank", 23),
    VoucherProto("v_antimatter", "Antimatter", 24, ("v_blank",)),
    VoucherProto("v_magic_trick", "Magic Trick", 25),
    VoucherProto("v_illusion", "Illusion", 26, ("v_magic_trick",)),
    VoucherProto("v_hieroglyph", "Hieroglyph", 27),
    VoucherProto("v_petroglyph", "Petroglyph", 28, ("v_hieroglyph",)),
    VoucherProto("v_directors_cut", "Director's Cut", 29),
    VoucherProto("v_retcon", "Retcon", 30, ("v_directors_cut",)),
    VoucherProto("v_paint_brush", "Paint Brush", 31),
    VoucherProto("v_palette", "Palette", 32, ("v_paint_brush",)),
]

VOUCHER_BY_KEY: dict[str, VoucherProto] = {v.key: v for v in P_VOUCHERS}
VOUCHER_BY_NAME: dict[str, VoucherProto] = {v.name.lower(): v for v in P_VOUCHERS}

DECK_STARTING_VOUCHERS: dict[str, tuple[str, ...]] = {
    "magic deck": ("v_crystal_ball",),
    "b_magic": ("v_crystal_ball",),
    "nebula deck": ("v_telescope",),
    "b_nebula": ("v_telescope",),
    "zodiac deck": ("v_tarot_merchant", "v_planet_merchant", "v_overstock_norm"),
    "b_zodiac": ("v_tarot_merchant", "v_planet_merchant", "v_overstock_norm"),
}


def normalize_voucher_key(key_or_name: str) -> str:
    """Convert a Voucher key ('v_overstock_norm') or name ('Overstock') to its 'v_...' key."""
    clean = key_or_name.strip()
    if clean in VOUCHER_BY_KEY:
        return clean
    lower = clean.lower()
    if lower in VOUCHER_BY_NAME:
        return VOUCHER_BY_NAME[lower].key
    return clean


# ---------------------------------------------------------------------------
# 4. G.P_CENTER_POOLS['Tag'] (24 Skip Tags sorted by order 1..24, game.lua:224-249)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TagProto:
    key: str
    name: str
    order: int
    min_ante: int | None = None
    requires: str | None = None


P_TAGS: list[TagProto] = [
    TagProto("tag_uncommon", "Uncommon Tag", 1, None),
    TagProto("tag_rare", "Rare Tag", 2, None, "j_blueprint"),
    TagProto("tag_negative", "Negative Tag", 3, 2, "e_negative"),
    TagProto("tag_foil", "Foil Tag", 4, None, "e_foil"),
    TagProto("tag_holo", "Holographic Tag", 5, None, "e_holo"),
    TagProto("tag_polychrome", "Polychrome Tag", 6, None, "e_polychrome"),
    TagProto("tag_investment", "Investment Tag", 7, None),
    TagProto("tag_voucher", "Voucher Tag", 8, None),
    TagProto("tag_boss", "Boss Tag", 9, None),
    TagProto("tag_standard", "Standard Tag", 10, 2),
    TagProto("tag_charm", "Charm Tag", 11, None),
    TagProto("tag_meteor", "Meteor Tag", 12, 2),
    TagProto("tag_buffoon", "Buffoon Tag", 13, 2),
    TagProto("tag_handy", "Handy Tag", 14, 2),
    TagProto("tag_garbage", "Garbage Tag", 15, 2),
    TagProto("tag_ethereal", "Ethereal Tag", 16, 2),
    TagProto("tag_coupon", "Coupon Tag", 17, None),
    TagProto("tag_double", "Double Tag", 18, None),
    TagProto("tag_juggle", "Juggle Tag", 19, None),
    TagProto("tag_d_six", "D6 Tag", 20, None),
    TagProto("tag_top_up", "Top-up Tag", 21, 2),
    TagProto("tag_skip", "Skip Tag", 22, None),
    TagProto("tag_orbital", "Orbital Tag", 23, 2),
    TagProto("tag_economy", "Economy Tag", 24, None),
]

TAG_BY_KEY: dict[str, TagProto] = {t.key: t for t in P_TAGS}
TAG_BY_NAME: dict[str, TagProto] = {t.name.lower(): t for t in P_TAGS}


def normalize_tag_key(key_or_name: str) -> str:
    """Convert a Tag key ('tag_orbital') or name ('Orbital Tag') to its 'tag_...' key."""
    clean = key_or_name.strip()
    if clean in TAG_BY_KEY:
        return clean
    lower = clean.lower()
    if lower in TAG_BY_NAME:
        return TAG_BY_NAME[lower].key
    if f"{lower} tag" in TAG_BY_NAME:
        return TAG_BY_NAME[f"{lower} tag"].key
    return clean


# ---------------------------------------------------------------------------
# 5. G.P_BLINDS Boss Blinds (28 Bosses, game.lua:263-296)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BossBlindProto:
    key: str
    name: str
    order: int
    min_ante: int
    showdown: bool = False


P_BOSS_BLINDS: list[BossBlindProto] = [
    BossBlindProto("bl_arm", "The Arm", 8, 2, False),
    BossBlindProto("bl_club", "The Club", 9, 1, False),
    BossBlindProto("bl_eye", "The Eye", 16, 3, False),
    BossBlindProto("bl_final_acorn", "Amber Acorn", 26, 10, True),
    BossBlindProto("bl_final_bell", "Cerulean Bell", 30, 10, True),
    BossBlindProto("bl_final_heart", "Crimson Heart", 29, 10, True),
    BossBlindProto("bl_final_leaf", "Verdant Leaf", 27, 10, True),
    BossBlindProto("bl_final_vessel", "Violet Vessel", 28, 10, True),
    BossBlindProto("bl_fish", "The Fish", 10, 2, False),
    BossBlindProto("bl_flint", "The Flint", 24, 2, False),
    BossBlindProto("bl_goad", "The Goad", 12, 1, False),
    BossBlindProto("bl_head", "The Head", 22, 1, False),
    BossBlindProto("bl_hook", "The Hook", 3, 1, False),
    BossBlindProto("bl_house", "The House", 5, 2, False),
    BossBlindProto("bl_manacle", "The Manacle", 15, 1, False),
    BossBlindProto("bl_mark", "The Mark", 25, 2, False),
    BossBlindProto("bl_mouth", "The Mouth", 17, 2, False),
    BossBlindProto("bl_needle", "The Needle", 21, 2, False),
    BossBlindProto("bl_ox", "The Ox", 4, 6, False),
    BossBlindProto("bl_pillar", "The Pillar", 20, 1, False),
    BossBlindProto("bl_plant", "The Plant", 18, 4, False),
    BossBlindProto("bl_psychic", "The Psychic", 11, 1, False),
    BossBlindProto("bl_serpent", "The Serpent", 19, 5, False),
    BossBlindProto("bl_tooth", "The Tooth", 23, 3, False),
    BossBlindProto("bl_wall", "The Wall", 6, 2, False),
    BossBlindProto("bl_water", "The Water", 13, 2, False),
    BossBlindProto("bl_wheel", "The Wheel", 7, 2, False),
    BossBlindProto("bl_window", "The Window", 14, 1, False),
]

BOSS_BY_KEY: dict[str, BossBlindProto] = {b.key: b for b in P_BOSS_BLINDS}
BOSS_BY_NAME: dict[str, BossBlindProto] = {b.name.lower(): b for b in P_BOSS_BLINDS}


def normalize_boss_key(key_or_name: str) -> str:
    """Convert a Boss Blind key ('bl_hook') or name ('The Hook') to its 'bl_...' key."""
    clean = key_or_name.strip()
    if clean in BOSS_BY_KEY:
        return clean
    lower = clean.lower()
    if lower in BOSS_BY_NAME:
        return BOSS_BY_NAME[lower].key
    if f"the {lower}" in BOSS_BY_NAME:
        return BOSS_BY_NAME[f"the {lower}"].key
    return clean


# ---------------------------------------------------------------------------
# 6. Castle, Ancient Joker, The Idol, Mail-in Rebate, To Do List
# ---------------------------------------------------------------------------

def get_castle_suit(
    rng: BalatroRNG,
    deck: Deck,
    ante: int = 1,
    state: GameState | None = None,
    stream_key: str | None = None,
) -> Suit:
    """Determine the suit Castle selects for the round (`reset_castle_card`, common_events.lua:2312-2324).
    
    Balatro mechanics:
    - Sets default suit to `'Spades'`.
    - Filters `G.playing_cards` for non-Stone cards (`v.ability.effect ~= 'Stone Card'`),
      sorted by `sort_id`.
    - Draws a card using `pseudorandom_element(valid_castle_cards, pseudoseed('cas' .. ante))`
      and selects `castle_card.base.suit`.
    """
    valid_cards = deck.get_non_stone_cards()
    if not valid_cards:
        suit = Suit.SPADES
    else:
        key = stream_key if stream_key is not None else f"cas{ante}"
        chosen_card, _ = rng.choice(key, valid_cards)
        suit = chosen_card.suit

    if state is not None:
        state.castle_suit = suit
    return suit


def get_ancient_joker_suit(
    rng: BalatroRNG,
    ante: int = 1,
    current_suit: Suit | str | None = None,
    state: GameState | None = None,
    stream_key: str | None = None,
) -> Suit:
    """Determine the suit Ancient Joker selects for the round (`reset_ancient_card`, common_events.lua:2303-2310).
    
    Balatro mechanics:
    - Independent of deck composition!
    - Iterates `{'Spades', 'Hearts', 'Clubs', 'Diamonds'}` in order and keeps suits
      `~= G.GAME.current_round.ancient_card.suit`.
    - At the very start of the run (`game.lua:2387`), `ancient_card.suit` is `nil`, so Round 1
      draws from all 4 suits. Every subsequent round excludes the previous round's suit
      (so 3 suits are in `ancient_suits`, and Ancient Joker never repeats back-to-back).
    - Draws using `pseudorandom_element(ancient_suits, pseudoseed('anc' .. ante))`.
    """
    if current_suit is not None:
        prev_suit: Suit | None = (
            Suit.from_str(current_suit) if isinstance(current_suit, str) else current_suit
        )
    elif state is not None:
        prev_suit = state.ancient_suit
    else:
        prev_suit = None

    base_order = [Suit.SPADES, Suit.HEARTS, Suit.CLUBS, Suit.DIAMONDS]
    ancient_suits = [s for s in base_order if s != prev_suit]

    key = stream_key if stream_key is not None else f"anc{ante}"
    chosen_suit, _ = rng.choice(key, ancient_suits)

    if state is not None:
        state.ancient_suit = chosen_suit
    return chosen_suit


@dataclass
class IdolTarget:
    """Target card (rank and suit) selected by The Idol for the round."""

    rank: Rank
    suit: Suit
    card: Card | None = None
    ante: int = 1

    def __repr__(self) -> str:
        return f"IdolTarget({self.rank.value} of {self.suit.value}, ante={self.ante})"


def get_idol_card(
    rng: BalatroRNG,
    deck: Deck,
    ante: int = 1,
    state: GameState | None = None,
    stream_key: str | None = None,
) -> IdolTarget:
    """Determine the card (rank + suit) The Idol selects for the round (`reset_idol_card`, common_events.lua:2271-2286).
    
    Balatro mechanics:
    - Sets default to `Ace of Spades`.
    - Filters `G.playing_cards` for non-Stone cards (`v.ability.effect ~= 'Stone Card'`),
      sorted by `sort_id`.
    - Draws `idol_card = pseudorandom_element(valid_idol_cards, pseudoseed('idol' .. ante))`
      and sets `rank = idol_card.base.value`, `suit = idol_card.base.suit`.
    """
    valid_cards = deck.get_non_stone_cards()
    if not valid_cards:
        target = IdolTarget(rank=Rank.ACE, suit=Suit.SPADES, card=None, ante=ante)
    else:
        key = stream_key if stream_key is not None else f"idol{ante}"
        chosen_card, _ = rng.choice(key, valid_cards)
        target = IdolTarget(
            rank=chosen_card.rank,
            suit=chosen_card.suit,
            card=chosen_card,
            ante=ante,
        )

    if state is not None:
        state.idol_card = (target.rank, target.suit)
    return target


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
    state: GameState | None = None,
    stream_key: str | None = None,
) -> Rank:
    """Dynamically determine Mail-in Rebate target rank for the round (`reset_mail_rank`, common_events.lua:2288-2301).
    
    Balatro chooses proportionally from the non-Stone cards in `G.playing_cards` (sorted by `sort_id`):
    - If deck has no non-Stone cards, defaults to `Rank.ACE`.
    - Otherwise, picks via `pseudorandom_element(valid_mail_cards, pseudoseed('mail' .. ante))`.
    """
    valid_cards = deck.get_non_stone_cards()
    if not valid_cards:
        rank = Rank.ACE
    else:
        key = (
            stream_key
            if stream_key is not None
            else (f"mail{ante}" if ante is not None else "mail1")
        )
        chosen_card, _ = rng.choice(key, valid_cards)
        rank = chosen_card.rank

    if state is not None:
        state.mail_rank = rank
    return rank


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


def get_to_do_list_hand(
    rng: BalatroRNG,
    current_hand: str | None = None,
    visible_hands: Sequence[str] | None = None,
    state: GameState | None = None,
    is_initial_spawn: bool = False,
    stream_key: str = "to_do",
) -> str:
    """Determine the poker hand To Do List selects (`card.lua:311-323`, `2975-2984`).
    
    Balatro mechanics:
    1. When `To Do List` is first created/spawned (`Card:set_ability`, `card.lua:311-323`):
       - Builds `_poker_hands` from all `v.visible` hands in `G.GAME.hands`.
       - Draws `pseudorandom_element(_poker_hands, pseudoseed('to_do'))` (repeating if it matches
         `old_hand`, which is `nil` on initial creation -> 1 draw from 9 visible hands).
    2. At the end of each round while `To Do List` is owned (`context.end_of_round`, `card.lua:2975-2984`):
       - Builds `_poker_hands` from all `v.visible` hands where `k ~= self.ability.to_do_poker_hand`
         (8 hands out of 9 by default).
       - Draws `self.ability.to_do_poker_hand = pseudorandom_element(_poker_hands, pseudoseed('to_do'))`
         (single draw, never repeats the previous round's hand!).
    3. Note: Scoring the target hand mid-round pays $4 each time it is played in that round
       and does NOT change the target hand until `end_of_round` (`card.lua:3491-3500`).
    """
    hands_pool = (
        list(visible_hands)
        if visible_hands is not None
        else (
            list(state.visible_hands)
            if state is not None
            else list(DEFAULT_VISIBLE_POKER_HANDS)
        )
    )
    prev_hand = (
        current_hand
        if current_hand is not None
        else (state.to_do_hand if state is not None else None)
    )

    if is_initial_spawn or prev_hand is None:
        # Card:set_ability (card.lua:311-323)
        while True:
            chosen, _ = rng.choice(stream_key, hands_pool)
            if chosen != prev_hand or len(hands_pool) <= 1:
                break
    else:
        # context.end_of_round (card.lua:2975-2984): excludes prev_hand before polling
        filtered_pool = [h for h in hands_pool if h != prev_hand]
        if not filtered_pool:
            filtered_pool = hands_pool
        chosen, _ = rng.choice(stream_key, filtered_pool)

    if state is not None:
        state.to_do_hand = chosen
    return chosen


# ---------------------------------------------------------------------------
# 7. Cartomancer & Vagabond Tarot Generation (`get_current_pool('Tarot', ...)`)
# ---------------------------------------------------------------------------

@dataclass
class TarotPrediction:
    """Result of a deterministic Tarot card generation (Cartomancer, Vagabond, etc.)."""

    key: str
    name: str
    ante: int
    source: str
    stream_key: str
    resample_count: int = 0

    def __repr__(self) -> str:
        res = f", resamples={self.resample_count}" if self.resample_count else ""
        return f"TarotPrediction({self.name!r} [{self.key}], ante={self.ante}, stream={self.stream_key!r}{res})"


def poll_tarot_pool(
    rng: BalatroRNG,
    ante: int = 1,
    key_append: str = "",
    state: GameState | None = None,
    active_tarots: set[str] | None = None,
    banned_keys: set[str] | None = None,
    has_showman: bool = False,
    source_label: str = "Tarot",
) -> TarotPrediction:
    """Poll `G.P_CENTER_POOLS['Tarot']` matching `get_current_pool('Tarot', nil, nil, key_append)`
    and `create_card('Tarot', ...)` (`common_events.lua:1963-2053, 2082-2122`).
    
    Balatro mechanics:
    - `_pool_key = 'Tarot' .. (key_append or '') .. G.GAME.round_resets.ante`
      - Cartomancer (`key_append = 'car'`): `'Tarotcar' .. ante`
      - Vagabond (`key_append = 'vag'`): `'Tarotvag' .. ante`
    - Iterates the 22 Tarots in `order` 1..22.
    - If a Tarot is currently in `G.GAME.used_jokers` (`active_tarots`) and the player does NOT
      have Showman (`j_ring_master`), or if it is in `banned_keys`, `_pool[i] = 'UNAVAILABLE'`.
      Otherwise `_pool[i] = v.key`.
    - If all 22 Tarots are unavailable (`_pool_size == 0`), fallback pool is `{'c_strength'}`.
    - Draws `center = pseudorandom_element(_pool, pseudoseed(_pool_key))`.
    - While `center == 'UNAVAILABLE'`, increments `it` (`2, 3, ...`) and redraws with
      `pseudorandom_element(_pool, pseudoseed(_pool_key .. '_resample' .. it))`.
    """
    eff_ante = max(1, ante)
    pool_key = f"Tarot{key_append}{eff_ante}"

    used = set()
    if state is not None:
        used.update(normalize_tarot_key(x) for x in state.active_tarots)
    if active_tarots is not None:
        used.update(normalize_tarot_key(x) for x in active_tarots)

    banned = set()
    if state is not None:
        banned.update(normalize_tarot_key(x) for x in state.banned_keys)
    if banned_keys is not None:
        banned.update(normalize_tarot_key(x) for x in banned_keys)

    showman = has_showman or (state.has_showman if state is not None else False)

    pool: list[str] = []
    pool_size = 0
    for t in P_TAROTS:
        is_avail = True
        if t.key in banned:
            is_avail = False
        elif not showman and t.key in used:
            is_avail = False

        if is_avail:
            pool.append(t.key)
            pool_size += 1
        else:
            pool.append("UNAVAILABLE")

    if pool_size == 0:
        pool = ["c_strength"]

    center, _ = rng.choice(pool_key, pool)
    it = 1
    resamples = 0
    while center == "UNAVAILABLE":
        it += 1
        resamples += 1
        center, _ = rng.choice(f"{pool_key}_resample{it}", pool)

    proto = TAROT_BY_KEY[center]
    return TarotPrediction(
        key=proto.key,
        name=proto.name,
        ante=ante,
        source=source_label,
        stream_key=pool_key,
        resample_count=resamples,
    )


def predict_cartomancer_tarot(
    rng: BalatroRNG,
    ante: int = 1,
    state: GameState | None = None,
    active_tarots: set[str] | None = None,
    has_showman: bool = False,
) -> TarotPrediction:
    """Predict which Tarot card Cartomancer gives when a blind is selected (`card.lua:2551`).
    
    Uses `create_card('Tarot', G.consumeables, nil, nil, nil, nil, nil, 'car')`
    -> stream `'Tarotcar' + str(ante)`.
    """
    return poll_tarot_pool(
        rng=rng,
        ante=ante,
        key_append="car",
        state=state,
        active_tarots=active_tarots,
        has_showman=has_showman,
        source_label="Cartomancer",
    )


def predict_vagabond_tarot(
    rng: BalatroRNG,
    ante: int = 1,
    state: GameState | None = None,
    active_tarots: set[str] | None = None,
    has_showman: bool = False,
) -> TarotPrediction:
    """Predict which Tarot card Vagabond gives when a hand is played with <= $4 (`card.lua:3750`).
    
    Uses `create_card('Tarot', G.consumeables, nil, nil, nil, nil, nil, 'vag')`
    -> stream `'Tarotvag' + str(ante)`.
    """
    return poll_tarot_pool(
        rng=rng,
        ante=ante,
        key_append="vag",
        state=state,
        active_tarots=active_tarots,
        has_showman=has_showman,
        source_label="Vagabond",
    )


# ---------------------------------------------------------------------------
# 8. Marble Joker / Stone Joker Card Generation (`card.lua:2580-2601`)
# ---------------------------------------------------------------------------

@dataclass
class StoneCardResult:
    """Result of Marble Joker ("Stone Joker") adding a Stone Card when a blind is selected."""

    card: Card
    front_key: str

    @property
    def hidden_suit(self) -> Suit:
        """The underlying suit revealed if the Stone Card is given a different enhancement or vampired."""
        return self.card.suit

    @property
    def hidden_rank(self) -> Rank:
        """The underlying rank revealed if the Stone Card is given a different enhancement or vampired."""
        return self.card.rank

    @property
    def suit(self) -> Suit:
        return self.card.suit

    @property
    def rank(self) -> Rank:
        return self.card.rank

    def __repr__(self) -> str:
        return f"StoneCardResult(hidden={self.hidden_rank.value} of {self.hidden_suit.value} [{self.front_key}], sort_id={self.card.id})"


def predict_stone_joker_card(
    rng: BalatroRNG,
    state: GameState | None = None,
    add_to_deck: bool = True,
    stream_key: str = "marb_fr",
) -> StoneCardResult:
    """Predict the card Marble Joker ("Stone Joker") adds to the deck on blind select (`card.lua:2580-2601`).
    
    Balatro mechanics:
    - Calls `local front = pseudorandom_element(G.P_CARDS, pseudoseed('marb_fr'))`.
    - Creates `Card(..., front, G.P_CENTERS.m_stone, ...)` with the Stone Card enhancement (`m_stone`),
      preserving the underlying `front` suit and rank (`base.suit` and `base.value`).
    - If a different enhancement (e.g. The Empress, The Magician, The Lovers) or Vampire removes
      `m_stone`, the underlying `hidden_rank` and `hidden_suit` are revealed!
    - Note: While it remains a Stone Card (`m_stone`), `reset_idol_card`, `reset_mail_rank`, and
      `reset_castle_card` skip it (`v.ability.effect ~= 'Stone Card'`). If enhanced into a non-Stone
      card, it participates in those rolls at its assigned `sort_id` (`card.id`).
    """
    chosen_front_key, _ = rng.choice(stream_key, P_CARDS_KEYS)
    suit, rank = parse_card_key(chosen_front_key)

    stone_card = Card(
        suit=suit,
        rank=rank,
        enhancement=Enhancement.STONE,
    )

    if state is not None and add_to_deck:
        state.deck.add_card(stone_card)

    return StoneCardResult(
        card=stone_card,
        front_key=chosen_front_key,
    )


# Alias for Marble Joker
predict_marble_joker_card = predict_stone_joker_card


# ---------------------------------------------------------------------------
# 9. Order of Boss Blinds (`get_new_boss`, common_events.lua:2338-2383)
# ---------------------------------------------------------------------------

@dataclass
class BossBlindPrediction:
    """Result of rolling a Boss Blind via `get_new_boss()`."""

    key: str
    name: str
    ante: int
    is_showdown: bool
    is_reroll: bool = False
    eligible_pool_size: int = 0
    min_use_tier: int = 0

    def __repr__(self) -> str:
        kind = "Showdown" if self.is_showdown else "Normal"
        rr = ", reroll=True" if self.is_reroll else ""
        return f"BossBlindPrediction({self.name!r} [{self.key}], ante={self.ante}, {kind}{rr})"


def get_next_boss_blind(
    rng: BalatroRNG,
    ante: int = 1,
    state: GameState | None = None,
    bosses_used: dict[str, int] | None = None,
    banned_keys: set[str] | None = None,
    win_ante: int = 8,
    is_reroll: bool = False,
    stream_key: str = "boss",
) -> BossBlindPrediction:
    """Roll the next Boss Blind matching `get_new_boss()` (`common_events.lua:2338-2383`).
    
    Balatro mechanics:
    1. Eligibility for `ante = G.GAME.round_resets.ante` (with `win_ante = 8`):
       - Normal boss (`not v.boss.showdown`):
         `v.boss.min <= math.max(1, ante)` AND (`math.max(1, ante) % win_ante ~= 0` OR `ante < 2`)
       - Showdown boss (`v.boss.showdown`):
         `ante % win_ante == 0` AND `ante >= 2`
       - Excludes any boss in `G.GAME.banned_keys`.
    2. `bosses_used` cycle tracking:
       - Finds `min_use = min(G.GAME.bosses_used[k] for k in eligible_bosses)`.
       - Keeps only bosses where `G.GAME.bosses_used[k] == min_use` (so no boss can repeat
         until every eligible boss in that pool has been rolled `min_use` times!).
    3. Draws `_, boss = pseudorandom_element(eligible_bosses, pseudoseed('boss'))` (where keys
       are sorted lexicographically by `'bl_...'`) and increments `G.GAME.bosses_used[boss] += 1`.
    """
    if state is not None:
        used_map = state.bosses_used
        banned = state.banned_keys
        eff_win_ante = state.win_ante
    else:
        used_map = bosses_used if bosses_used is not None else {}
        banned = banned_keys if banned_keys is not None else set()
        eff_win_ante = win_ante

    # Ensure all boss keys exist in used_map initialized to 0 (game.lua:1864-1866)
    for b in P_BOSS_BLINDS:
        if b.key not in used_map:
            used_map[b.key] = 0

    eff_ante = max(1, ante)
    eligible: list[BossBlindProto] = []
    for b in P_BOSS_BLINDS:
        if b.key in banned:
            continue
        if not b.showdown:
            if b.min_ante <= eff_ante and (eff_ante % eff_win_ante != 0 or ante < 2):
                eligible.append(b)
        else:
            if ante % eff_win_ante == 0 and ante >= 2:
                eligible.append(b)

    if not eligible:
        raise ValueError(f"No eligible boss blinds found for ante={ante}")

    min_use = min(used_map[b.key] for b in eligible)
    filtered = [b for b in eligible if used_map[b.key] <= min_use]
    # Sort lexicographically by key ('bl_...') matching pseudorandom_element on key-value table
    filtered.sort(key=lambda b: b.key)

    chosen_proto, _ = rng.choice(stream_key, filtered)
    used_map[chosen_proto.key] += 1

    if state is not None:
        state.current_boss = chosen_proto.key

    return BossBlindPrediction(
        key=chosen_proto.key,
        name=chosen_proto.name,
        ante=ante,
        is_showdown=chosen_proto.showdown,
        is_reroll=is_reroll,
        eligible_pool_size=len(filtered),
        min_use_tier=min_use,
    )


# ---------------------------------------------------------------------------
# 10. Order of Skip Tags (`get_next_tag_key`, common_events.lua:1914-1925, 1982-1986)
# ---------------------------------------------------------------------------

@dataclass
class TagPrediction:
    """Result of rolling a Skip Tag via `get_next_tag_key()`."""

    key: str
    name: str
    ante: int
    blind_type: str  # "Small" or "Big"
    stream_key: str
    resample_count: int = 0
    orbital_hand: str | None = None

    def __repr__(self) -> str:
        orb = f", orbital_hand={self.orbital_hand!r}" if self.orbital_hand else ""
        res = f", resamples={self.resample_count}" if self.resample_count else ""
        return f"TagPrediction({self.name!r} [{self.key}], ante={self.ante}, blind={self.blind_type!r}{orb}{res})"


def get_next_tag(
    rng: BalatroRNG,
    ante: int = 1,
    blind_type: str = "Small",
    state: GameState | None = None,
    banned_keys: set[str] | None = None,
    roll_orbital_choice: bool = False,
) -> TagPrediction:
    """Roll the next Skip Tag matching `get_next_tag_key()` (`common_events.lua:1914-1925`, `1982-1986`).
    
    Balatro mechanics:
    - Calls `get_current_pool('Tag')` with `_pool_key = 'Tag' .. G.GAME.round_resets.ante`.
    - Iterates the 24 tags in `G.P_CENTER_POOLS['Tag']` (sorted by `order` 1..24).
    - A tag `v` is available if `not v.min_ante or v.min_ante <= G.GAME.round_resets.ante`
      (and not in `banned_keys`). Note: In Ante 1 (`ante <= 1`), the 9 tags with `min_ante = 2`
      are placed in `_pool` as `'UNAVAILABLE'` at their exact indices, preserving the 24-slot array!
    - If all tags are unavailable, fallback is `'tag_handy'`.
    - Draws `center = pseudorandom_element(_pool, pseudoseed(_pool_key))`.
    - While `center == 'UNAVAILABLE'`, increments `it` (`2, 3, ...`) and redraws with
      `pseudorandom_element(_pool, pseudoseed(_pool_key .. '_resample' .. it))`.
    """
    eff_ante = max(1, ante)
    pool_key = f"Tag{eff_ante}"

    banned = set()
    if state is not None:
        banned.update(normalize_tag_key(x) for x in state.banned_keys)
    if banned_keys is not None:
        banned.update(normalize_tag_key(x) for x in banned_keys)

    pool: list[str] = []
    pool_size = 0
    for t in P_TAGS:
        is_avail = True
        if t.key in banned:
            is_avail = False
        elif t.min_ante is not None and t.min_ante > ante:
            is_avail = False

        if is_avail:
            pool.append(t.key)
            pool_size += 1
        else:
            pool.append("UNAVAILABLE")

    if pool_size == 0:
        pool = ["tag_handy"]

    center, _ = rng.choice(pool_key, pool)
    it = 1
    resamples = 0
    while center == "UNAVAILABLE":
        it += 1
        resamples += 1
        center, _ = rng.choice(f"{pool_key}_resample{it}", pool)

    orbital_hand: str | None = None
    if roll_orbital_choice:
        hands = state.visible_hands if state is not None else DEFAULT_VISIBLE_POKER_HANDS
        orbital_hand, _ = rng.choice("orbital", hands)

    proto = TAG_BY_KEY[center]
    return TagPrediction(
        key=proto.key,
        name=proto.name,
        ante=ante,
        blind_type=blind_type,
        stream_key=pool_key,
        resample_count=resamples,
        orbital_hand=orbital_hand,
    )


def get_orbital_tag_hands_for_ante(
    rng: BalatroRNG,
    visible_hands: Sequence[str] | None = None,
    state: GameState | None = None,
) -> dict[str, str]:
    """Roll `G.GAME.orbital_choices[ante]` for `'Small'`, `'Big'`, and `'Boss'` (`UI_definitions.lua:1506-1516`).
    
    When the Blind Select UI is built at the start of an ante, Balatro populates
    `G.GAME.orbital_choices[ante][type]` for `'Small'`, `'Big'`, and `'Boss'` in order
    using `pseudorandom_element(_poker_hands, pseudoseed('orbital'))`.
    """
    hands = (
        list(visible_hands)
        if visible_hands is not None
        else (
            list(state.visible_hands)
            if state is not None
            else list(DEFAULT_VISIBLE_POKER_HANDS)
        )
    )
    small_hand, _ = rng.choice("orbital", hands)
    big_hand, _ = rng.choice("orbital", hands)
    boss_hand, _ = rng.choice("orbital", hands)
    return {
        "Small": small_hand,
        "Big": big_hand,
        "Boss": boss_hand,
    }


# ---------------------------------------------------------------------------
# 11. Order of Vouchers (`get_next_voucher_key`, common_events.lua:1901-1912, 1989-2007)
# ---------------------------------------------------------------------------

@dataclass
class VoucherPrediction:
    """Result of rolling a Voucher via `get_next_voucher_key(_from_tag)`."""

    key: str
    name: str
    ante: int
    from_tag: bool
    stream_key: str
    resample_count: int = 0

    def __repr__(self) -> str:
        src = "VoucherTag" if self.from_tag else f"Ante {self.ante} Shop"
        res = f", resamples={self.resample_count}" if self.resample_count else ""
        return f"VoucherPrediction({self.name!r} [{self.key}], {src}, stream={self.stream_key!r}{res})"


def get_next_voucher(
    rng: BalatroRNG,
    ante: int = 1,
    from_tag: bool = False,
    state: GameState | None = None,
    used_vouchers: set[str] | None = None,
    shop_vouchers: Sequence[str] | None = None,
    banned_keys: set[str] | None = None,
    add_to_shop_vouchers: bool = False,
) -> VoucherPrediction:
    """Roll the next Voucher matching `get_next_voucher_key(_from_tag)` (`common_events.lua:1901-1912`, `1989-2007`).
    
    Balatro mechanics:
    1. Stream key (`common_events.lua:1902-1903, 2052`):
       - Normal ante shop voucher (`_from_tag = false`):
         `_pool_key = 'Voucher' .. G.GAME.round_resets.ante` (ante-specific stream, e.g. `'Voucher1'`, `'Voucher2'`).
       - Voucher Tag skip (`_from_tag = true`, `tag.lua:306`):
         `_pool_key = 'Voucher_fromtag'` (GLOBAL run-wide stream across all Voucher Tags, NOT ante-specific!).
    2. Pool construction (`G.P_CENTER_POOLS['Voucher']`, ordered 1..32):
       - A voucher `v` is eligible iff:
         a) `not G.GAME.used_vouchers[v.key]` (has not been purchased/redeemed yet)
         b) Not currently present in `G.shop_vouchers.cards` (`shop_vouchers`)
         c) All prerequisites in `v.requires` ARE in `G.GAME.used_vouchers`
         d) Not in `G.GAME.banned_keys`
       - Ineligible vouchers are placed at their exact 1..32 index in `_pool` as `'UNAVAILABLE'`.
       - If all 32 vouchers are unavailable, fallback is `{'v_blank'}`.
    3. Resampling:
       - Draws `center = pseudorandom_element(_pool, pseudoseed(_pool_key))`.
       - While `center == 'UNAVAILABLE'`, increments `it` (`2, 3, ...`) and redraws with
         `pseudorandom_element(_pool, pseudoseed(_pool_key .. '_resample' .. it))`.
    """
    eff_ante = max(1, ante)
    pool_key = "Voucher_fromtag" if from_tag else f"Voucher{eff_ante}"

    redeemed: set[str] = set()
    if state is not None:
        redeemed.update(normalize_voucher_key(x) for x in state.used_vouchers)
    if used_vouchers is not None:
        redeemed.update(normalize_voucher_key(x) for x in used_vouchers)

    in_shop: set[str] = set()
    if state is not None:
        in_shop.update(normalize_voucher_key(x) for x in state.shop_vouchers)
    if shop_vouchers is not None:
        in_shop.update(normalize_voucher_key(x) for x in shop_vouchers)

    banned: set[str] = set()
    if state is not None:
        banned.update(normalize_voucher_key(x) for x in state.banned_keys)
    if banned_keys is not None:
        banned.update(normalize_voucher_key(x) for x in banned_keys)

    pool: list[str] = []
    pool_size = 0
    for v in P_VOUCHERS:
        is_avail = True
        if v.key in redeemed or v.key in in_shop or v.key in banned:
            is_avail = False
        elif v.requires:
            for req in v.requires:
                if req not in redeemed:
                    is_avail = False
                    break

        if is_avail:
            pool.append(v.key)
            pool_size += 1
        else:
            pool.append("UNAVAILABLE")

    if pool_size == 0:
        pool = ["v_blank"]

    center, _ = rng.choice(pool_key, pool)
    it = 1
    resamples = 0
    while center == "UNAVAILABLE":
        it += 1
        resamples += 1
        center, _ = rng.choice(f"{pool_key}_resample{it}", pool)

    if state is not None and add_to_shop_vouchers:
        if center not in state.shop_vouchers:
            state.shop_vouchers.append(center)

    proto = VOUCHER_BY_KEY[center]
    return VoucherPrediction(
        key=proto.key,
        name=proto.name,
        ante=ante,
        from_tag=from_tag,
        stream_key=pool_key,
        resample_count=resamples,
    )


# ---------------------------------------------------------------------------
# 12. Certificate Event (`card.lua:2463-2475`)
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
    ante: int = 1,
    state: GameState | None = None,
    add_to_state: bool = True,
    ante_scoped: bool = True,
) -> CertificateResult:
    """Predict and generate Certificate joker card.
    
    Supports two modes:
    - `ante_scoped=True` (default for backwards compatibility with Prompt 1):
      Pulls card front from `G.P_CARDS` using `'cert_fr' + str(ante)` and seal from
      `G.P_SEALS` using `'cert_sl' + str(ante)`.
    - `ante_scoped=False` (exact `balatro_source/card.lua:2467-2469` parity):
      Pulls card front from `G.P_CARDS` using `'cert_fr'` and seal via float roll on `'certsl'`
      (`> 0.75` Red, `> 0.5` Blue, `> 0.25` Gold, else Purple).
    """
    if ante_scoped:
        key_fr = f"cert_fr{ante}"
        key_sl = f"cert_sl{ante}"
        chosen_front_key, _ = rng.choice(key_fr, P_CARDS_KEYS)
        suit, rank = parse_card_key(chosen_front_key)
        chosen_seal, _ = rng.choice(key_sl, P_SEALS_KEYS)
        seal_label = chosen_seal.balatro_name
    else:
        chosen_front_key, _ = rng.choice("cert_fr", P_CARDS_KEYS)
        suit, rank = parse_card_key(chosen_front_key)
        seal_roll = rng.random("certsl")
        if seal_roll > 0.75:
            chosen_seal = Seal.RED
        elif seal_roll > 0.5:
            chosen_seal = Seal.BLUE
        elif seal_roll > 0.25:
            chosen_seal = Seal.GOLD
        else:
            chosen_seal = Seal.PURPLE
        seal_label = chosen_seal.balatro_name

    generated_card = Card(suit=suit, rank=rank, seal=chosen_seal)

    if state is not None and add_to_state:
        state.deck.add_card(generated_card)
        state.hand.append(generated_card)

    return CertificateResult(
        card=generated_card,
        front_key=chosen_front_key,
        seal_key=seal_label,
        ante=ante,
    )


# ---------------------------------------------------------------------------
# 13. Lucky Card Evaluation (`card.lua:1292-1299`)
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
    """Evaluate deterministic mid-round triggers for a scored Lucky Card (`card.lua:1292-1299`).
    
    Balatro rolls separate checks:
    - +20 Mult: 1 in 5 base chance on `'lucky_mult'`
    - +$20 Cash: 1 in 15 base chance on `'lucky_money'`
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
# 14. 1-in-N Outcome Resolvers
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
        """Resolve a deterministic 1-in-N outcome using `stream_key`."""
        roll = self.rng.random(stream_key)
        threshold = (numerator * oops_multiplier) / float(denominator)
        return roll < threshold

    def wheel_of_fortune(self, oops_multiplier: float = 1.0) -> bool:
        """Resolve The Wheel of Fortune (1 in 4 base chance). Stream: `'wheel_of_fortune'`."""
        return self.check_odds("wheel_of_fortune", 1, 4, oops_multiplier)

    def space_joker(
        self, oops_multiplier: float = 1.0, stream_key: str = "space_joker"
    ) -> bool:
        """Resolve Space Joker hand level upgrade (1 in 4 base chance).
        
        Note: `balatro_source/card.lua:3420` uses `'space'`; defaults to `'space_joker'`
        or accepts `stream_key='space'`.
        """
        return self.check_odds(stream_key, 1, 4, oops_multiplier)

    def bloodstone(self, oops_multiplier: float = 1.0, base_denominator: int = 2) -> bool:
        """Resolve Bloodstone extra X1.5 mult check (1 in 2 base in Balatro 1.0.1f). Stream: `'bloodstone'`."""
        return self.check_odds("bloodstone", 1, base_denominator, oops_multiplier)

    def glass_card_shatter(self, oops_multiplier: float = 1.0) -> bool:
        """Resolve Glass Card shatter check upon scoring (1 in 4 base chance). Stream: `'glass'`."""
        return self.check_odds("glass", 1, 4, oops_multiplier)

    def business_card(
        self, oops_multiplier: float = 1.0, stream_key: str = "business_card"
    ) -> bool:
        """Resolve Business Card $2 payout on face card scoring (1 in 2 base chance).
        
        Note: `balatro_source/card.lua:3177` uses `'business'`; defaults to `'business_card'`
        or accepts `stream_key='business'`.
        """
        return self.check_odds(stream_key, 1, 2, oops_multiplier)

    def gros_michel_extinct(self, oops_multiplier: float = 1.0) -> bool:
        """Resolve Gros Michel extinction check at end of round (1 in 6 base chance). Stream: `'gros_michel'`."""
        return self.check_odds("gros_michel", 1, 6, oops_multiplier)

    def cavendish_extinct(self, oops_multiplier: float = 1.0) -> bool:
        """Resolve Cavendish extinction check at end of round (1 in 1000 base chance). Stream: `'cavendish'`."""
        return self.check_odds("cavendish", 1, 1000, oops_multiplier)

    def reserved_parking(self, oops_multiplier: float = 1.0) -> bool:
        """Resolve Reserved Parking $1 payout on held face card (1 in 2 base chance). Stream: `'parking'`."""
        return self.check_odds("parking", 1, 2, oops_multiplier)

    def eight_ball(self, oops_multiplier: float = 1.0) -> bool:
        """Resolve 8 Ball Tarot generation on scored 8 (1 in 4 base chance). Stream: `'8ball'`."""
        return self.check_odds("8ball", 1, 4, oops_multiplier)

    def hallucination(self, oops_multiplier: float = 1.0, ante: int = 1) -> bool:
        """Resolve Hallucination Tarot generation on opening pack (1 in 2 base chance). Stream: `'halu' + str(ante)`."""
        return self.check_odds(f"halu{ante}", 1, 2, oops_multiplier)
