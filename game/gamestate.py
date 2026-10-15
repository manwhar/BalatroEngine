import copy
from dataclasses import dataclass, field
from typing import Any

try:
    from game.balatro_models import (
        Action,
        ActionType,
        BlindName,
        Deck,
        Enhancement,
        Hand,
        HandCard,
        Inventory,
        Joker,
        Location,
        PlayingCard,
        PokerHand,
        PokerHandLevel,
        Rank,
        RunStats,
        Seal,
        ShopState,
        Stake,
        Suit,
        Voucher,
        VoucherName,
    )
    from game.engine.rules import on_action
except ImportError:
    from balatro_models import (
        Action,
        ActionType,
        BlindName,
        Deck,
        Enhancement,
        Hand,
        HandCard,
        Inventory,
        Joker,
        Location,
        PlayingCard,
        PokerHand,
        PokerHandLevel,
        Rank,
        RunStats,
        Seal,
        ShopState,
        Stake,
        Suit,
        Voucher,
        VoucherName,
    )
    from engine import on_action


def create_standard_deck() -> list[PlayingCard]:
    """Generates a standard 52-card deck of PlayingCard objects."""
    ranks = [
        Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX,
        Rank.SEVEN, Rank.EIGHT, Rank.NINE, Rank.TEN,
        Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE,
    ]
    suits = [Suit.SPADES, Suit.HEARTS, Suit.CLUBS, Suit.DIAMONDS]
    return [PlayingCard(rank=r, suit=s) for s in suits for r in ranks]


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
    
    stats: RunStats = field(default_factory=RunStats)
    inventory: Inventory = field(default_factory=Inventory)
    hand: Hand = field(default_factory=Hand)
    shop: ShopState | None = None
    vouchers_redeemed: list[VoucherName] = field(default_factory=list)
    hand_levels: dict[PokerHand, int] = field(default_factory=dict)
    boss_blind: BlindName | None = None
    hand_play_counts: dict[PokerHand, int] = field(default_factory=dict)
    unique_planets_used: set[str] = field(default_factory=set)
    deck_cards_count: int = 52
    starting_deck_size: int = 52
    full_deck: list[PlayingCard] = field(default_factory=create_standard_deck)
    discard_pile: list[PlayingCard] = field(default_factory=list)
    played_cards_this_round: list[PlayingCard] = field(default_factory=list)

    def add_card_to_deck(self, card: PlayingCard, add_to_hand: bool = False) -> None:
        """Adds a card to the run's full deck (and optionally hand)."""
        self.full_deck.append(card)
        self.deck_cards_count = len(self.full_deck)
        if add_to_hand:
            self.hand.cards.append(HandCard(card=card, selected=False))

    def remove_card_from_deck(self, card: PlayingCard) -> bool:
        """Removes a card from full deck, discard pile, and hand when destroyed."""
        removed = False
        for i, c in enumerate(self.full_deck):
            if c is card or (c.rank == card.rank and c.suit == card.suit and c.enhancement == card.enhancement and c.seal == card.seal):
                self.full_deck.pop(i)
                removed = True
                break
        for i, c in enumerate(self.discard_pile):
            if c is card or (c.rank == card.rank and c.suit == card.suit and c.enhancement == card.enhancement and c.seal == card.seal):
                self.discard_pile.pop(i)
                break
        self.hand.cards = [
            hc for hc in self.hand.cards
            if hc.card is not card and not (hc.card.rank == card.rank and hc.card.suit == card.suit)
        ]
        card.destroyed = True
        if removed:
            self.deck_cards_count = len(self.full_deck)
        else:
            self.deck_cards_count = max(0, self.deck_cards_count - 1)
        return removed

    @property
    def remaining_deck(self) -> list[PlayingCard]:
        """Cards remaining in draw pile: full_deck minus hand, discard_pile, and played_cards_this_round."""
        drawn_or_used = [hc.card for hc in self.hand.cards] + self.discard_pile + self.played_cards_this_round
        remaining = list(self.full_deck)
        for card in drawn_or_used:
            if card in remaining:
                remaining.remove(card)
        return remaining

    def copy(self) -> "GameState":
        return copy.deepcopy(self)


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
    
    stats_diff: dict[str, Any] | None = None
    jokers_diff: list[Joker] | None = None
    consumables_diff: list[Any] | None = None
    hand_diff: list[HandCard] | None = None
    shop_diff: ShopState | None = None
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

    stats_diff = {}
    for attr in ("money", "hands_left", "discards_left", "hands_played_this_round", "discards_used_this_round", "reroll_cost"):
        prev_val = getattr(prev.stats, attr)
        curr_val = getattr(curr.stats, attr)
        if prev_val != curr_val:
            stats_diff[attr] = curr_val
    if stats_diff:
        delta.stats_diff = stats_diff

    if prev.inventory.jokers != curr.inventory.jokers:
        delta.jokers_diff = curr.inventory.jokers
    if prev.inventory.consumables != curr.inventory.consumables:
        delta.consumables_diff = curr.inventory.consumables

    if prev.hand.cards != curr.hand.cards:
        delta.hand_diff = curr.hand.cards

    if prev.shop != curr.shop:
        delta.shop_diff = curr.shop

    if prev.vouchers_redeemed != curr.vouchers_redeemed:
        delta.vouchers_diff = curr.vouchers_redeemed

    if prev.deck_cards_count != curr.deck_cards_count:
        delta.deck_cards_count = curr.deck_cards_count
    if prev.full_deck != curr.full_deck:
        delta.full_deck_diff = curr.full_deck
    if prev.discard_pile != curr.discard_pile:
        delta.discard_pile_diff = curr.discard_pile
    if prev.played_cards_this_round != curr.played_cards_this_round:
        delta.played_cards_diff = curr.played_cards_this_round
    if prev.hand_play_counts != curr.hand_play_counts:
        delta.hand_play_counts_diff = curr.hand_play_counts
    if prev.unique_planets_used != curr.unique_planets_used:
        delta.unique_planets_diff = curr.unique_planets_used

    return delta


def apply_delta(base: GameState, delta: GameStateDelta) -> GameState:
    """Applies a GameStateDelta onto a base GameState to produce the updated GameState."""
    next_state = base.copy()
    next_state.step_idx = delta.step_idx
    next_state.frame_idx = delta.frame_idx
    next_state.timestamp = delta.timestamp

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
        next_state.full_deck = delta.full_deck_diff
    if delta.discard_pile_diff is not None:
        next_state.discard_pile = delta.discard_pile_diff
    if delta.played_cards_diff is not None:
        next_state.played_cards_this_round = delta.played_cards_diff

    if delta.stats_diff:
        for k, v in delta.stats_diff.items():
            setattr(next_state.stats, k, v)

    if delta.jokers_diff is not None:
        next_state.inventory.jokers = delta.jokers_diff
    if delta.consumables_diff is not None:
        next_state.inventory.consumables = delta.consumables_diff

    if delta.hand_diff is not None:
        next_state.hand.cards = delta.hand_diff

    if delta.shop_diff is not None:
        next_state.shop = delta.shop_diff

    if delta.vouchers_diff is not None:
        next_state.vouchers_redeemed = delta.vouchers_diff

    if delta.hand_levels_diff is not None:
        next_state.hand_levels = delta.hand_levels_diff

    if delta.hand_play_counts_diff is not None:
        next_state.hand_play_counts = delta.hand_play_counts_diff

    if delta.unique_planets_diff is not None:
        next_state.unique_planets_used = delta.unique_planets_diff

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
