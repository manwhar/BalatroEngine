import copy
from dataclasses import dataclass, field
from typing import Any

try:
    from game.balatro_models import (
        Action,
        ActionType,
        BlindName,
        BoosterPack,
        Card,
        CardBounds,
        Deck,
        Edition,
        Enhancement,
        Hand,
        HandCard,
        Inventory,
        Joker,
        JokerName,
        Location,
        PlanetName,
        PlayingCard,
        PokerHand,
        PokerHandLevel,
        Rank,
        Seal,
        SkipTag,
        Stake,
        Suit,
        TarotName,
        Voucher,
        VoucherName,
    )
    from game.engine.rules import on_action
except ImportError:
    from balatro_models import (
        Action,
        ActionType,
        BlindName,
        BoosterPack,
        Card,
        CardBounds,
        Deck,
        Edition,
        Enhancement,
        Hand,
        HandCard,
        Inventory,
        Joker,
        JokerName,
        Location,
        PlanetName,
        PlayingCard,
        PokerHand,
        PokerHandLevel,
        Rank,
        Seal,
        SkipTag,
        Stake,
        Suit,
        TarotName,
        Voucher,
        VoucherName,
    )
    from engine import on_action


def create_standard_deck() -> list[PlayingCard]:
    """Generates a standard 52-card deck of PlayingCard objects in Balatro's exact sort_id order (1..52)."""
    suits = [Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS, Suit.SPADES]
    ranks = [
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
    deck = []
    card_id = 1
    for s in suits:
        for r in ranks:
            deck.append(PlayingCard(rank=r, suit=s, id=card_id))
            card_id += 1
    return deck


def _cards_match(a: PlayingCard, b: PlayingCard) -> bool:
    """Checks whether two PlayingCard instances represent the same card instance or identical card."""
    if a is b:
        return True
    if a.id and b.id and a.id == b.id:
        return True
    return (
        a.rank == b.rank
        and a.suit == b.suit
        and a.enhancement == b.enhancement
        and a.edition == b.edition
        and a.seal == b.seal
    )


def get_rank_sort_nominal(card: PlayingCard) -> float:
    """Calculates Balatro's exact nominal value for 'Sort by Rank' (descending).

    Nominals:
      Ace: 11.4, King: 10.3, Queen: 10.2, Jack: 10.1, 10: 10.0, 9..2: 9.0..2.0
      Suit nominals: Spades: +0.04, Hearts: +0.03, Clubs: +0.02, Diamonds: +0.01
      Stone card: -1000.0 (always sorts to rightmost end)
    """
    if card.enhancement == Enhancement.STONE:
        return -1000.0

    rank_nominals = {
        Rank.TWO: (2.0, 0.0),
        Rank.THREE: (3.0, 0.0),
        Rank.FOUR: (4.0, 0.0),
        Rank.FIVE: (5.0, 0.0),
        Rank.SIX: (6.0, 0.0),
        Rank.SEVEN: (7.0, 0.0),
        Rank.EIGHT: (8.0, 0.0),
        Rank.NINE: (9.0, 0.0),
        Rank.TEN: (10.0, 0.0),
        Rank.JACK: (10.0, 0.1),
        Rank.QUEEN: (10.0, 0.2),
        Rank.KING: (10.0, 0.3),
        Rank.ACE: (11.0, 0.4),
    }
    suit_nominals = {
        Suit.DIAMONDS: 0.01,
        Suit.CLUBS: 0.02,
        Suit.HEARTS: 0.03,
        Suit.SPADES: 0.04,
    }
    base_nom, face_nom = rank_nominals.get(card.rank, (0.0, 0.0))
    suit_nom = suit_nominals.get(card.suit, 0.0)
    tie_break = (card.id * 1e-6) if card.id else 0.0
    return base_nom + face_nom + suit_nom + tie_break


def get_suit_sort_nominal(card: PlayingCard) -> float:
    """Calculates Balatro's exact nominal value for 'Sort by Suit' (suit desc).

    Nominals:
      Spades block: 40.0 + rank_nominal
      Hearts block: 30.0 + rank_nominal
      Clubs block: 20.0 + rank_nominal
      Diamonds block: 10.0 + rank_nominal
      Stone card: -1000.0
    """
    if card.enhancement == Enhancement.STONE:
        return -1000.0

    rank_nominals = {
        Rank.TWO: (2.0, 0.0),
        Rank.THREE: (3.0, 0.0),
        Rank.FOUR: (4.0, 0.0),
        Rank.FIVE: (5.0, 0.0),
        Rank.SIX: (6.0, 0.0),
        Rank.SEVEN: (7.0, 0.0),
        Rank.EIGHT: (8.0, 0.0),
        Rank.NINE: (9.0, 0.0),
        Rank.TEN: (10.0, 0.0),
        Rank.JACK: (10.0, 0.1),
        Rank.QUEEN: (10.0, 0.2),
        Rank.KING: (10.0, 0.3),
        Rank.ACE: (11.0, 0.4),
    }
    suit_nominals = {
        Suit.DIAMONDS: 0.01,
        Suit.CLUBS: 0.02,
        Suit.HEARTS: 0.03,
        Suit.SPADES: 0.04,
    }
    base_nom, face_nom = rank_nominals.get(card.rank, (0.0, 0.0))
    suit_nom = suit_nominals.get(card.suit, 0.0)
    tie_break = (card.id * 1e-6) if card.id else 0.0
    return base_nom + face_nom + (suit_nom * 1000.0) + tie_break


@dataclass(kw_only=True)
class GameState:
    """Full snapshot of the game state at a specific frame/decision boundary."""
    step_idx: int = 0
    frame_idx: int = 0
    timestamp: float = 0.0
    location: Location = Location.BLIND
    stake: Stake = Stake.WHITE
    deck: Deck = Deck.RED_DECK
    ante: int = 1
    round: int = 1
    current_blind: BlindName = BlindName.SMALL
    target_chips: int = 300
    current_chips: int = 0

    # Economy & Round Counters (formerly RunStats)
    money: int = 4
    hands_left: int = 4
    discards_left: int = 3
    hands_played_this_round: int = 0
    discards_used_this_round: int = 0
    reroll_cost: int = 5
    free_rerolls: int = 0
    inflation: int = 0
    bankrupt_at: int = 0

    # Inventory & Hand
    inventory: Inventory = field(default_factory=Inventory)
    hand: Hand = field(default_factory=Hand)

    # Shop Offerings (formerly ShopState)
    shop_cards: list[Card] = field(default_factory=list)
    shop_vouchers: list[Voucher] = field(default_factory=list)
    shop_packs: list[BoosterPack] = field(default_factory=list)

    # Booster Pack Opening State (Location.PACK)
    active_pack: BoosterPack | None = None
    pack_cards: list[Card] = field(default_factory=list)
    pack_choices_remaining: int = 0

    # Skip Tags & Skip History
    small_blind_tag: SkipTag | None = None
    big_blind_tag: SkipTag | None = None
    active_tags: list[SkipTag] = field(default_factory=list)
    blinds_skipped: int = 0

    # Vouchers, Consumable Memory & Scaling Modifiers
    vouchers_redeemed: list[VoucherName] = field(default_factory=list)
    last_tarot_planet: TarotName | PlanetName | None = None
    ectoplasm_minus: int = 1
    unique_planets_used: set[str] = field(default_factory=set)
    tarots_used: int = 0
    unused_discards: int = 0
    gros_michel_extinct: bool = False
    pending_cv_checks: set[str] = field(default_factory=set)

    # Per-Round / Per-Ante Dynamic Targets (G.GAME.current_round)
    idol_card: tuple[Rank, Suit] = (Rank.ACE, Suit.SPADES)
    mail_card_rank: Rank = Rank.ACE
    ancient_suit: Suit = Suit.SPADES
    castle_suit: Suit = Suit.SPADES
    to_do_poker_hand: PokerHand = PokerHand.HIGH_CARD

    # Boss Blind & Hand/Card Constraint State
    boss_blind: BlindName | None = None
    boss_disabled: bool = False
    round_won: bool = False
    hand_levels: dict[PokerHand, int] = field(default_factory=dict)
    hand_play_counts: dict[PokerHand, int] = field(default_factory=dict)
    hands_played_this_round_by_type: set[PokerHand] = field(default_factory=set)
    locked_hand_type: PokerHand | None = None
    ox_target_hand: PokerHand | None = None

    # Deck & Pile Tracking
    deck_cards_count: int = 52
    starting_deck_size: int = 52
    full_deck: list[PlayingCard] = field(default_factory=create_standard_deck)
    discard_pile: list[PlayingCard] = field(default_factory=list)
    played_cards_this_round: list[PlayingCard] = field(default_factory=list)
    played_cards_this_ante: list[PlayingCard] = field(default_factory=list)

    # PRNG Engine State
    seed: str = ""
    rng: Any = None
    resolver: Any = None
    _next_slot_uid: int = field(default=1, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.seed and self.rng is None:
            self.set_seed(self.seed)

    def set_seed(self, seed: str) -> None:
        """Initializes or resets the deterministic BalatroRNG and EventResolver on this GameState."""
        self.seed = str(seed).strip().upper()
        try:
            from game.engine.rng_sim.rng import BalatroRNG
            from game.engine.rng_sim.events import EventResolver
        except ImportError:
            from engine.rng_sim.rng import BalatroRNG
            from engine.rng_sim.events import EventResolver
        self.rng = BalatroRNG(self.seed)
        self.resolver = EventResolver(self.rng)

    @property
    def discount_percent(self) -> int:
        """Shop discount percentage from Clearance Sale (25%) or Liquidation (50%)."""
        if VoucherName.LIQUIDATION in self.vouchers_redeemed:
            return 50
        if VoucherName.CLEARANCE_SALE in self.vouchers_redeemed:
            return 25
        return 0

    def add_card_to_deck(self, card: PlayingCard, add_to_hand: bool = False) -> None:
        """Adds a card to the run's full deck (and optionally hand) and triggers Hologram."""
        if card.id == 0:
            max_id = max((c.id for c in self.full_deck), default=0)
            card.id = max_id + 1
        self.full_deck.append(card)
        self.deck_cards_count = len(self.full_deck)
        if add_to_hand:
            self.hand.cards.append(HandCard(card=card, selected=False))
        if self.inventory:
            for j in self.inventory.jokers:
                if j.name == JokerName.HOLOGRAM and not j.debuffed:
                    j.state["x_mult"] = round(j.state.get("x_mult", 1.0) + 0.25, 2)

    def remove_card_from_deck(self, card: PlayingCard, shattered: bool = False) -> bool:
        """Removes a single matching card from full deck, discard pile, and hand when destroyed."""
        removed = False
        for i, c in enumerate(self.full_deck):
            if _cards_match(c, card):
                self.full_deck.pop(i)
                removed = True
                break
        if not removed:
            for i, c in enumerate(self.full_deck):
                if c.rank == card.rank and c.suit == card.suit:
                    self.full_deck.pop(i)
                    removed = True
                    break
        if not removed and self.full_deck:
            self.full_deck.pop()
            removed = True
        for i, c in enumerate(self.discard_pile):
            if _cards_match(c, card):
                self.discard_pile.pop(i)
                break
        for i, hc in enumerate(self.hand.cards):
            if _cards_match(hc.card, card):
                self.hand.cards.pop(i)
                break
        card.destroyed = True
        if removed:
            self.deck_cards_count = len(self.full_deck)
        else:
            self.deck_cards_count = max(0, self.deck_cards_count - 1)
        if self.inventory:
            has_pareidolia = any(
                j.name == JokerName.PAREIDOLIA and not j.debuffed
                for j in self.inventory.jokers
            )
            is_face = (
                card.enhancement != Enhancement.STONE
                and card.rank != Rank.NONE
                and not card.debuffed
                and (card.is_face or has_pareidolia)
            )
            for j in self.inventory.jokers:
                if j.debuffed:
                    continue
                if is_face and j.name == JokerName.CANIO:
                    j.state["x_mult"] = round(j.state.get("x_mult", 1.0) + 1.0, 2)
                elif (
                    card.enhancement == Enhancement.GLASS
                    and shattered
                    and j.name == JokerName.GLASS_JOKER
                ):
                    j.state["x_mult"] = round(j.state.get("x_mult", 1.0) + 0.75, 2)
        return removed

    @property
    def remaining_deck(self) -> list[PlayingCard]:
        """Cards remaining in draw pile: full_deck minus hand, discard_pile, and played_cards_this_round."""
        drawn_or_used = [hc.card for hc in self.hand.cards] + self.discard_pile + self.played_cards_this_round
        remaining = list(self.full_deck)
        for card in drawn_or_used:
            for i, rem_card in enumerate(remaining):
                if _cards_match(rem_card, card):
                    remaining.pop(i)
                    break
        return remaining

    def _get_next_slot_uid(self) -> int:
        uid = self._next_slot_uid
        self._next_slot_uid += 1
        return uid

    def ensure_slot_uids(self) -> None:
        """Assigns persistent slot UIDs and initializes CardBounds for any unassigned hand cards."""
        for hc in self.hand.cards:
            if hc.slot_uid == 0:
                hc.slot_uid = self._get_next_slot_uid()
            if hc.facing_back:
                if hc.bounds is None:
                    hc.bounds = CardBounds(is_face_down=True)
                if not hc.bounds.candidate_card_ids:
                    candidates = self.get_face_down_candidate_pool(hc)
                    hc.bounds.set_candidates(candidates)

    def get_face_down_candidate_pool(self, target_hc: HandCard | None = None) -> list[PlayingCard]:
        """Gathers available candidate PlayingCard objects that could occupy a face-down slot."""
        resolved_in_hand_ids = {
            c.card.id
            for c in self.hand.cards
            if c is not target_hc
            and (not c.facing_back or (c.bounds and c.bounds.is_resolved))
            and c.card.id != 0
        }
        unavailable_ids = (
            {c.id for c in self.discard_pile if c.id}
            | {c.id for c in self.played_cards_this_round if c.id}
            | resolved_in_hand_ids
        )

        base_pool = list(self.remaining_deck)
        if not base_pool and self.full_deck:
            base_pool = [c for c in self.full_deck if c.id not in unavailable_ids]

        pool: list[PlayingCard] = []
        for c in base_pool:
            if c.id not in unavailable_ids:
                pool.append(c)

        if target_hc is not None:
            if target_hc._real_card and not any(_cards_match(c, target_hc._real_card) for c in pool):
                if target_hc._real_card.id not in unavailable_ids:
                    pool.append(target_hc._real_card)
            elif target_hc.bounds and target_hc.bounds.candidate_cards:
                for c in target_hc.bounds.candidate_cards:
                    if c.id not in unavailable_ids and not any(_cards_match(p, c) for p in pool):
                        pool.append(c)

        return pool

    def apply_sort_observation(
        self, sort_type: str = "rank", selected_slot_uid: int | None = None
    ) -> None:
        """Updates hand order and narrows CardBounds for all face-down cards from sort order."""
        self.ensure_slot_uids()
        norm_type = sort_type.lower()
        is_suit_sort = "suit" in norm_type
        nominal_fn = get_suit_sort_nominal if is_suit_sort else get_rank_sort_nominal

        # 1. Deterministic sort if simulation or cards have true values
        can_simulate_sort = all(
            (c._real_card is not None or c.card.rank != Rank.NONE or c.enhancement == Enhancement.STONE)
            for c in self.hand.cards
        )
        if can_simulate_sort:
            self.hand.cards.sort(
                key=lambda hc: nominal_fn(hc._real_card if hc._real_card else hc.card),
                reverse=True,
            )

        # 2. Iterate deduction passes until fixpoint is reached
        changed = True
        while changed:
            changed = False
            for i, hc in enumerate(self.hand.cards):
                if (hc.bounds and hc.bounds.is_resolved) or (not hc.facing_back and hc.bounds is None):
                    continue

                if hc.bounds is None:
                    hc.bounds = CardBounds(is_face_down=True)

                # Determine upper nominal bound from nearest known face-up card to the left
                upper_nominal = float("inf")
                for left_i in range(i - 1, -1, -1):
                    left_card = self.hand.cards[left_i]
                    if not left_card.facing_back and left_card.card.rank != Rank.NONE:
                        upper_nominal = nominal_fn(left_card.card)
                        break

                # Determine lower nominal bound from nearest known face-up card to the right
                lower_nominal = float("-inf")
                for right_i in range(i + 1, len(self.hand.cards)):
                    right_card = self.hand.cards[right_i]
                    if not right_card.facing_back and right_card.card.rank != Rank.NONE:
                        lower_nominal = nominal_fn(right_card.card)
                        break

                # Filter candidates against bounds
                pool = self.get_face_down_candidate_pool(hc)
                valid_candidates: list[PlayingCard] = []
                for cand in pool:
                    cand_nom = nominal_fn(cand)
                    if cand_nom > upper_nominal + 1e-5 or cand_nom < lower_nominal - 1e-5:
                        continue
                    if cand.rank not in hc.bounds.possible_ranks:
                        continue
                    if cand.suit not in hc.bounds.possible_suits:
                        continue
                    if cand.enhancement not in hc.bounds.possible_enhancements:
                        continue
                    if cand.edition not in hc.bounds.possible_editions:
                        continue
                    if cand.seal not in hc.bounds.possible_seals:
                        continue
                    valid_candidates.append(cand)

                # Update bounds
                prev_ids = set(hc.bounds.candidate_card_ids)
                hc.bounds.set_candidates(valid_candidates)

                if prev_ids != hc.bounds.candidate_card_ids:
                    changed = True

                # Check if exactly 1 candidate remains -> resolve!
                if hc.bounds.is_resolved and valid_candidates:
                    resolved = valid_candidates[0]
                    hc.resolve_face_down(resolved)
                    changed = True
                    break  # Restart loop with new resolved card as face-up boundary

        self.reconcile_face_down_adjacencies()

    def reconcile_face_down_adjacencies(self) -> None:
        """Enforces visual adjacency constraints and human memory bounds for face-down cards.

        In Balatro's UI, cards in hand are displayed in a horizontal sequence:
        1. Isolated face-down cards (separated from other face-down cards by at least one
           face-up card) remain visually distinct landmarks and retain all specific
           information known about them (e.g. from Tarot flip-reveals or deductions).
        2. Contiguous adjacent face-down cards without face-up dividers that share the same
           visual signature (selection highlight, seal sticker, or edition sheen on the back)
           are visually indistinguishable to human perception. They lose individual positional
           certainty and conflate their candidate pools to the union of their possibilities,
           forgetting specific single-card identity.
        """
        cards = self.hand.cards
        if len(cards) <= 1:
            return

        # 1. Partition hand into contiguous segments of face-down cards
        segments: list[list[int]] = []
        current_segment: list[int] = []

        for idx, hc in enumerate(cards):
            is_fd = hc.facing_back or (hc.bounds is not None and hc.bounds.is_face_down)
            if is_fd:
                current_segment.append(idx)
            else:
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
        if current_segment:
            segments.append(current_segment)

        # 2. Process each contiguous segment
        for segment in segments:
            # An isolated face-down card (length 1) has visual landmarks on both sides -> retains identity!
            if len(segment) < 2:
                continue

            # Group adjacent cards by visual appearance on the card back:
            # - selected state (raised vs resting)
            # - visible seal (e.g. Red, Blue, Gold, Purple seal sticker on back)
            # - visible edition (e.g. Foil, Holographic, Polychrome sheen on back)
            vis_groups: dict[tuple, list[int]] = {}
            for idx in segment:
                hc = cards[idx]
                seal_vis = hc.card.seal
                if seal_vis == Seal.NONE and hc._real_card and hc._real_card.seal != Seal.NONE:
                    seal_vis = hc._real_card.seal
                ed_vis = hc.card.edition
                if ed_vis == Edition.NONE and hc._real_card and hc._real_card.edition != Edition.NONE:
                    ed_vis = hc._real_card.edition

                vis_sig = (hc.selected, seal_vis, ed_vis)
                vis_groups.setdefault(vis_sig, []).append(idx)

            for vis_sig, group_indices in vis_groups.items():
                if len(group_indices) >= 2:
                    pooled_candidates: list[PlayingCard] = []
                    seen_ids: set[int] = set()

                    for idx in group_indices:
                        hc = cards[idx]
                        if hc.bounds and hc.bounds.candidate_cards:
                            cands = hc.bounds.candidate_cards
                        else:
                            cands = self.get_face_down_candidate_pool(hc)

                        for c in cands:
                            cid = c.id if c.id else id(c)
                            if cid not in seen_ids:
                                seen_ids.add(cid)
                                pooled_candidates.append(c)

                    for idx in group_indices:
                        hc = cards[idx]
                        if hc._real_card:
                            cid = (
                                hc._real_card.id
                                if hc._real_card.id
                                else id(hc._real_card)
                            )
                            if cid not in seen_ids:
                                seen_ids.add(cid)
                                pooled_candidates.append(hc._real_card)

                    if len(pooled_candidates) >= 2:
                        for idx in group_indices:
                            hc = cards[idx]
                            hc.blur_candidates(pooled_candidates)

    def update_face_down_candidates(self) -> None:
        """Prunes candidate pools of unresolved face-down cards against remaining deck."""
        changed = True
        while changed:
            changed = False
            for hc in self.hand.cards:
                if (hc.bounds and hc.bounds.is_resolved) or (not hc.facing_back and hc.bounds is None):
                    continue
                if hc.bounds is None:
                    hc.bounds = CardBounds(is_face_down=True)
                pool = self.get_face_down_candidate_pool(hc)
                if not hc.bounds.candidate_card_ids:
                    valid = pool
                else:
                    valid = [c for c in pool if c.id in hc.bounds.candidate_card_ids]
                prev_len = len(hc.bounds.candidate_card_ids)
                hc.bounds.set_candidates(valid)
                if len(hc.bounds.candidate_card_ids) != prev_len:
                    changed = True
                if hc.bounds.is_resolved and valid:
                    hc.resolve_face_down(valid[0])
                    changed = True
                    break

    def draw_cards(self, count: int | None = None, face_down: bool = False) -> list[HandCard]:
        """Draws cards from remaining deck into hand, appending them strictly to the right."""
        self.ensure_slot_uids()
        needed = (self.hand.max_size - len(self.hand.cards)) if count is None else count
        if needed <= 0:
            return []

        rem = self.remaining_deck
        drawn_playing_cards = rem[:needed]
        new_hand_cards: list[HandCard] = []

        for c in drawn_playing_cards:
            uid = self._get_next_slot_uid()
            if face_down:
                hc = HandCard(card=c, facing_back=True, slot_uid=uid)
                hc.bounds = CardBounds(is_face_down=True)
                hc.bounds.set_candidates(self.remaining_deck)
            else:
                hc = HandCard(card=c, facing_back=False, slot_uid=uid)
            self.hand.cards.append(hc)
            new_hand_cards.append(hc)

        self.reconcile_face_down_adjacencies()
        return new_hand_cards

    def copy(self) -> "GameState":
        return copy.deepcopy(self)


# Scalar/collection fields on GameState that are diffed directly by name in scalar_diffs
_TRACKED_STATE_FIELDS: tuple[str, ...] = (
    "location",
    "stake",
    "deck",
    "ante",
    "round",
    "current_blind",
    "target_chips",
    "current_chips",
    "money",
    "hands_left",
    "discards_left",
    "hands_played_this_round",
    "discards_used_this_round",
    "reroll_cost",
    "free_rerolls",
    "inflation",
    "bankrupt_at",
    "shop_cards",
    "shop_vouchers",
    "shop_packs",
    "active_pack",
    "pack_cards",
    "pack_choices_remaining",
    "small_blind_tag",
    "big_blind_tag",
    "active_tags",
    "blinds_skipped",
    "vouchers_redeemed",
    "last_tarot_planet",
    "ectoplasm_minus",
    "unique_planets_used",
    "tarots_used",
    "unused_discards",
    "gros_michel_extinct",
    "pending_cv_checks",
    "idol_card",
    "mail_card_rank",
    "ancient_suit",
    "castle_suit",
    "to_do_poker_hand",
    "boss_blind",
    "boss_disabled",
    "round_won",
    "hand_levels",
    "hand_play_counts",
    "hands_played_this_round_by_type",
    "locked_hand_type",
    "ox_target_hand",
    "deck_cards_count",
    "starting_deck_size",
    "full_deck",
    "discard_pile",
    "played_cards_this_round",
    "played_cards_this_ante",
    "seed",
)


@dataclass(kw_only=True)
class GameStateDelta:
    """Sparse delta between two GameStates (P-frame). None fields indicate no change."""
    step_idx: int
    frame_idx: int
    timestamp: float
    action: Action | None = None

    location: Location | None = None
    ante: int | None = None
    round: int | None = None
    current_blind: BlindName | None = None
    boss_blind: BlindName | None = None
    target_chips: int | None = None
    current_chips: int | None = None
    deck_cards_count: int | None = None
    full_deck_diff: list[PlayingCard] | None = None
    discard_pile_diff: list[PlayingCard] | None = None
    played_cards_diff: list[PlayingCard] | None = None

    field_diffs: dict[str, Any] | None = None
    jokers_diff: list[Joker] | None = None
    consumables_diff: list[Any] | None = None
    joker_slots_diff: int | None = None
    consumable_slots_diff: int | None = None
    hand_diff: list[HandCard] | None = None
    hand_max_size_diff: int | None = None
    vouchers_diff: list[VoucherName] | None = None
    hand_levels_diff: dict[PokerHand, int] | None = None
    hand_play_counts_diff: dict[PokerHand, int] | None = None
    unique_planets_diff: set[str] | None = None


def diff_states(prev: GameState, curr: GameState, action: Action | None = None) -> GameStateDelta:
    """Generates the minimal GameStateDelta between prev and curr."""
    delta = GameStateDelta(
        step_idx=curr.step_idx,
        frame_idx=curr.frame_idx,
        timestamp=curr.timestamp,
        action=action,
    )

    if prev.location != curr.location:
        delta.location = curr.location
    if prev.ante != curr.ante:
        delta.ante = curr.ante
    if prev.round != curr.round:
        delta.round = curr.round
    if prev.current_blind != curr.current_blind:
        delta.current_blind = curr.current_blind
    if prev.boss_blind != curr.boss_blind:
        delta.boss_blind = curr.boss_blind
    if prev.target_chips != curr.target_chips:
        delta.target_chips = curr.target_chips
    if prev.current_chips != curr.current_chips:
        delta.current_chips = curr.current_chips

    field_diffs: dict[str, Any] = {}
    for attr in _TRACKED_STATE_FIELDS:
        prev_val = getattr(prev, attr)
        curr_val = getattr(curr, attr)
        if prev_val != curr_val:
            field_diffs[attr] = copy.deepcopy(curr_val)
    if field_diffs:
        delta.field_diffs = field_diffs

    if prev.inventory.jokers != curr.inventory.jokers:
        delta.jokers_diff = copy.deepcopy(curr.inventory.jokers)
    if prev.inventory.consumables != curr.inventory.consumables:
        delta.consumables_diff = copy.deepcopy(curr.inventory.consumables)
    if prev.inventory.joker_slots != curr.inventory.joker_slots:
        delta.joker_slots_diff = curr.inventory.joker_slots
    if prev.inventory.consumable_slots != curr.inventory.consumable_slots:
        delta.consumable_slots_diff = curr.inventory.consumable_slots

    if prev.hand.cards != curr.hand.cards:
        delta.hand_diff = copy.deepcopy(curr.hand.cards)
    if prev.hand.max_size != curr.hand.max_size:
        delta.hand_max_size_diff = curr.hand.max_size

    if prev.vouchers_redeemed != curr.vouchers_redeemed:
        delta.vouchers_diff = list(curr.vouchers_redeemed)

    if prev.deck_cards_count != curr.deck_cards_count:
        delta.deck_cards_count = curr.deck_cards_count
    if prev.full_deck != curr.full_deck:
        delta.full_deck_diff = copy.deepcopy(curr.full_deck)
    if prev.discard_pile != curr.discard_pile:
        delta.discard_pile_diff = copy.deepcopy(curr.discard_pile)
    if prev.played_cards_this_round != curr.played_cards_this_round:
        delta.played_cards_diff = copy.deepcopy(curr.played_cards_this_round)
    if prev.hand_levels != curr.hand_levels:
        delta.hand_levels_diff = dict(curr.hand_levels)
    if prev.hand_play_counts != curr.hand_play_counts:
        delta.hand_play_counts_diff = dict(curr.hand_play_counts)
    if prev.unique_planets_used != curr.unique_planets_used:
        delta.unique_planets_diff = set(curr.unique_planets_used)

    return delta


def apply_delta(base: GameState, delta: GameStateDelta) -> GameState:
    """Applies a GameStateDelta onto a base GameState to produce the updated GameState."""
    next_state = base.copy()
    next_state.step_idx = delta.step_idx
    next_state.frame_idx = delta.frame_idx
    next_state.timestamp = delta.timestamp

    if delta.field_diffs:
        for k, v in delta.field_diffs.items():
            setattr(next_state, k, copy.deepcopy(v))

    if delta.location is not None:
        next_state.location = delta.location
    if delta.ante is not None:
        next_state.ante = delta.ante
    if delta.round is not None:
        next_state.round = delta.round
    if delta.current_blind is not None:
        next_state.current_blind = delta.current_blind
    if delta.boss_blind is not None:
        next_state.boss_blind = delta.boss_blind
    if delta.target_chips is not None:
        next_state.target_chips = delta.target_chips
    if delta.current_chips is not None:
        next_state.current_chips = delta.current_chips
    if delta.deck_cards_count is not None:
        next_state.deck_cards_count = delta.deck_cards_count
    if delta.full_deck_diff is not None:
        next_state.full_deck = copy.deepcopy(delta.full_deck_diff)
    if delta.discard_pile_diff is not None:
        next_state.discard_pile = copy.deepcopy(delta.discard_pile_diff)
    if delta.played_cards_diff is not None:
        next_state.played_cards_this_round = copy.deepcopy(delta.played_cards_diff)

    if delta.jokers_diff is not None:
        next_state.inventory.jokers = copy.deepcopy(delta.jokers_diff)
    if delta.consumables_diff is not None:
        next_state.inventory.consumables = copy.deepcopy(delta.consumables_diff)
    if delta.joker_slots_diff is not None:
        next_state.inventory.joker_slots = delta.joker_slots_diff
    if delta.consumable_slots_diff is not None:
        next_state.inventory.consumable_slots = delta.consumable_slots_diff

    if delta.hand_diff is not None:
        next_state.hand.cards = copy.deepcopy(delta.hand_diff)
    if delta.hand_max_size_diff is not None:
        next_state.hand.max_size = delta.hand_max_size_diff

    if delta.vouchers_diff is not None:
        next_state.vouchers_redeemed = list(delta.vouchers_diff)

    if delta.hand_levels_diff is not None:
        next_state.hand_levels = dict(delta.hand_levels_diff)

    if delta.hand_play_counts_diff is not None:
        next_state.hand_play_counts = dict(delta.hand_play_counts_diff)

    if delta.unique_planets_diff is not None:
        next_state.unique_planets_used = set(delta.unique_planets_diff)

    return next_state


@dataclass(kw_only=True)
class RunTrajectory:
    """Sequence of (GameState, Action) pairs stored as Keyframes + Deltas."""
    keyframe_interval: int = 20
    keyframes: dict[int, GameState] = field(default_factory=dict)
    deltas: list[GameStateDelta] = field(default_factory=list)

    def record_step(self, state: GameState, action: Action | None = None) -> GameStateDelta:
        """Records a new state step into the trajectory."""
        step = len(self.deltas)
        state.step_idx = step

        if step % self.keyframe_interval == 0:
            self.keyframes[step] = state.copy()

        prev_state = self.get_state(step - 1) if step > 0 else state
        delta = diff_states(prev_state, state, action)
        self.deltas.append(delta)
        return delta

    def get_state(self, step: int) -> GameState:
        """Reconstructs state at step N by seeking to nearest keyframe and replaying deltas."""
        if step < 0 or step >= len(self.deltas):
            raise IndexError(f"Step {step} out of range (total steps: {len(self.deltas)})")

        kf_step = max(k for k in self.keyframes if k <= step)
        curr = self.keyframes[kf_step].copy()

        for delta in self.deltas[kf_step + 1 : step + 1]:
            curr = apply_delta(curr, delta)

        return curr

    def impute_retroactive_boss_blinds(self):
        """Backward-propagates discovered boss blinds to all preceding steps in the same ante."""
        ante_bosses: dict[int, BlindName] = {}
        for s in range(len(self.deltas)):
            st = self.get_state(s)
            if st.boss_blind:
                ante_bosses[st.ante] = st.boss_blind

        for delta in self.deltas:
            step_state = self.get_state(delta.step_idx)
            if step_state.ante in ante_bosses and not delta.boss_blind:
                delta.boss_blind = ante_bosses[step_state.ante]

        for step, kf in self.keyframes.items():
            if kf.ante in ante_bosses:
                kf.boss_blind = ante_bosses[kf.ante]


def step_simulation(state: GameState, action: Action) -> tuple[GameState, GameStateDelta]:
    """Advances state using deterministic rules & trigger dispatcher, returning new state + delta."""
    prev = state.copy()
    on_action(state, action)
    state.step_idx += 1
    state.frame_idx = action.frame_idx
    state.timestamp = action.timestamp
    delta = diff_states(prev, state, action)
    return state, delta
