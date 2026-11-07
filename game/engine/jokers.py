"""
Complete 150-Joker Engine implementation.
Connects all 150 Balatro Jokers to active state changes, accumulators,
game triggers, passive deck modifiers, and CV observation hooks.
"""

import copy
import random
from typing import Callable

try:
    from game.balatro_models import (
        BlindName,
        Card,
        CardType,
        Consumable,
        Edition,
        Enhancement,
        Joker,
        JokerName,
        JokerRarity,
        PlayingCard,
        PokerHand,
        Rank,
        Seal,
        SkipTag,
        Suit,
    )
except ImportError:
    from balatro_models import (
        BlindName,
        Card,
        CardType,
        Consumable,
        Edition,
        Enhancement,
        Joker,
        JokerName,
        JokerRarity,
        PlayingCard,
        PokerHand,
        Rank,
        Seal,
        SkipTag,
        Suit,
    )
from .context import TriggerContext, TriggerEvent
from .cv_stubs import (
    cv_detect_boss_blind,
    cv_detect_card_destroyed,
    cv_detect_glass_card_shattered,
    cv_detect_lucky_card_triggered,
    cv_detect_target_poker_hand,
    cv_detect_target_rank,
    cv_detect_target_suit,
)
from .poker_hands import evaluate_poker_hand


JOKER_EFFECTS: dict[JokerName, Callable[[Joker, TriggerContext], None]] = {}


def joker_hook(name: JokerName):
    """Registers a handler for a specific Joker."""
    def decorator(fn: Callable[[Joker, TriggerContext], None]):
        JOKER_EFFECTS[name] = fn
        return fn
    return decorator


# =========================================================================
# HELPER FUNCTIONS (Face / Suit / Retriggers / Passive Deck Modifiers)
# =========================================================================

def is_card_face(card: PlayingCard, jokers: list[Joker] | None = None) -> bool:
    """Checks if a PlayingCard counts as a face card, respecting debuffs, Stone, and Pareidolia."""
    if getattr(card, "debuffed", False):
        return False
    if card.enhancement == Enhancement.STONE or card.rank == Rank.NONE:
        return False
    if jokers and any(j.name == JokerName.PAREIDOLIA and not j.debuffed for j in jokers):
        return True
    return card.is_face


def card_matches_suit(
    card: PlayingCard,
    target_suit: Suit,
    jokers: list[Joker] | None = None,
    bypass_debuff: bool = False,
) -> bool:
    """Checks if a PlayingCard matches target_suit, respecting Stone, Wild, debuffs, and Smeared Joker."""
    if card.enhancement == Enhancement.STONE or card.rank == Rank.NONE:
        return False
    if getattr(card, "debuffed", False) and not bypass_debuff:
        return False
    if card.enhancement == Enhancement.WILD:
        return target_suit != Suit.NONE
    if jokers and any(j.name == JokerName.SMEARED_JOKER and not j.debuffed for j in jokers):
        if target_suit in (Suit.HEARTS, Suit.DIAMONDS) and card.suit in (Suit.HEARTS, Suit.DIAMONDS):
            return True
        if target_suit in (Suit.SPADES, Suit.CLUBS) and card.suit in (Suit.SPADES, Suit.CLUBS):
            return True
        return False
    return card.suit == target_suit


def get_card_repetitions(card: PlayingCard, scoring_idx: int, state) -> int:
    """Calculates total scoring triggers (1 + retriggers) for a scoring card at scoring_idx."""
    if getattr(card, "debuffed", False):
        return 0
    triggers = 1
    if card.seal == Seal.RED:
        triggers += 1

    jokers = state.inventory.jokers if state and state.inventory else []
    for j in jokers:
        if j.debuffed:
            continue
        eff = resolve_copy_target(j, jokers) if j.name in (JokerName.BLUEPRINT, JokerName.BRAINSTORM) else j
        if eff is None or eff.debuffed:
            continue
        if eff.name == JokerName.HANGING_CHAD and scoring_idx == 0:
            triggers += 2
        elif eff.name == JokerName.HACK:
            if card.enhancement != Enhancement.STONE and card.rank in (
                Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE
            ):
                triggers += 1
        elif eff.name == JokerName.SOCK_AND_BUSKIN:
            if is_card_face(card, jokers):
                triggers += 1
        elif eff.name == JokerName.SELTZER:
            triggers += 1
        elif eff.name == JokerName.DUSK:
            if state.hands_left == 0:
                triggers += 1
    return triggers


def init_joker_default_state(state, joker: Joker) -> None:
    """Populates default keys in joker.state if not already initialized."""
    match joker.name:
        case JokerName.ANCIENT_JOKER:
            joker.state.setdefault("target_suit", state.ancient_suit)
        case JokerName.TO_DO_LIST:
            joker.state.setdefault("target_hand", state.to_do_poker_hand)
        case JokerName.THE_IDOL:
            joker.state.setdefault("target_rank", state.idol_card[0])
            joker.state.setdefault("target_suit", state.idol_card[1])
        case JokerName.CASTLE:
            joker.state.setdefault("chips", 0)
            joker.state.setdefault("target_suit", state.castle_suit)
        case JokerName.MAIL_IN_REBATE:
            joker.state.setdefault("target_rank", state.mail_card_rank)
        case (
            JokerName.CEREMONIAL_DAGGER
            | JokerName.FLASH_CARD
            | JokerName.RED_CARD
            | JokerName.GREEN_JOKER
            | JokerName.RIDE_THE_BUS
            | JokerName.SPARE_TROUSERS
        ):
            joker.state.setdefault("mult", 0)
        case JokerName.SQUARE_JOKER | JokerName.RUNNER | JokerName.WEE_JOKER:
            joker.state.setdefault("chips", 0)
        case (
            JokerName.CAMPFIRE
            | JokerName.CONSTELLATION
            | JokerName.LUCKY_CAT
            | JokerName.HOLOGRAM
            | JokerName.MADNESS
            | JokerName.VAMPIRE
            | JokerName.OBELISK
            | JokerName.GLASS_JOKER
            | JokerName.CANIO
            | JokerName.HIT_THE_ROAD
        ):
            joker.state.setdefault("x_mult", 1.0)
        case JokerName.YORICK:
            joker.state.setdefault("x_mult", 1.0)
            joker.state.setdefault("discards", 23)
        case JokerName.RAMEN:
            joker.state.setdefault("x_mult", 2.0)
        case JokerName.LOYALTY_CARD:
            joker.state.setdefault("remaining", 5)
        case JokerName.POPCORN:
            joker.state.setdefault("mult", 20)
        case JokerName.ICE_CREAM:
            joker.state.setdefault("chips", 100)
        case JokerName.TURTLE_BEAN:
            joker.state.setdefault("h_size", 5)
        case JokerName.SELTZER:
            joker.state.setdefault("hands_left", 10)
        case JokerName.INVISIBLE_JOKER:
            joker.state.setdefault("rounds", 0)
        case JokerName.ROCKET:
            joker.state.setdefault("payout", 1)


def on_joker_added(state, joker: Joker) -> None:
    """Applies Card:add_to_deck passive effects when a Joker is acquired."""
    init_joker_default_state(state, joker)
    if joker.debuffed:
        return
    match joker.name:
        case JokerName.JUGGLER:
            state.hand.max_size += 1
        case JokerName.DRUNKARD:
            state.discards_left += 1
        case JokerName.MERRY_ANDY:
            state.discards_left += 3
            state.hand.max_size = max(0, state.hand.max_size - 1)
        case JokerName.TROUBADOUR:
            state.hand.max_size += 2
            state.hands_left = max(1, state.hands_left - 1)
        case JokerName.STUNTMAN:
            state.hand.max_size = max(0, state.hand.max_size - 2)
        case JokerName.TURTLE_BEAN:
            state.hand.max_size += joker.state.get("h_size", 5)
        case JokerName.CREDIT_CARD:
            state.bankrupt_at -= 20
        case JokerName.CHAOS_THE_CLOWN:
            state.free_rerolls += 1
        case JokerName.CHICOT:
            if state.current_blind not in (BlindName.SMALL, BlindName.BIG):
                state.boss_disabled = True


def on_joker_removed(state, joker: Joker) -> None:
    """Reverses Card:remove_from_deck passive effects when a Joker is sold or destroyed."""
    if joker.debuffed:
        return
    match joker.name:
        case JokerName.JUGGLER:
            state.hand.max_size = max(0, state.hand.max_size - 1)
        case JokerName.DRUNKARD:
            state.discards_left = max(0, state.discards_left - 1)
        case JokerName.MERRY_ANDY:
            state.discards_left = max(0, state.discards_left - 3)
            state.hand.max_size += 1
        case JokerName.TROUBADOUR:
            state.hand.max_size = max(0, state.hand.max_size - 2)
            state.hands_left += 1
        case JokerName.STUNTMAN:
            state.hand.max_size += 2
        case JokerName.TURTLE_BEAN:
            state.hand.max_size = max(0, state.hand.max_size - joker.state.get("h_size", 5))
        case JokerName.CREDIT_CARD:
            state.bankrupt_at += 20
        case JokerName.CHAOS_THE_CLOWN:
            state.free_rerolls = max(0, state.free_rerolls - 1)


def sync_dynamic_joker_states(state) -> None:
    """Synchronizes dynamic state values on all Jokers at each action boundary."""
    if not state or not state.inventory:
        return
    jokers = state.inventory.jokers
    for j in jokers:
        init_joker_default_state(state, j)
        match j.name:
            case JokerName.SWASHBUCKLER:
                j.state["mult"] = sum(other.sell_value for other in jokers if other is not j)
            case JokerName.ABSTRACT_JOKER:
                j.state["mult"] = 3 * len(jokers)
            case JokerName.JOKER_STENCIL:
                non_neg = sum(1 for other in jokers if other.edition != Edition.NEGATIVE)
                empty_slots = max(0, state.inventory.joker_slots - non_neg)
                stencils = sum(
                    1 for other in jokers if other.name == JokerName.JOKER_STENCIL and not other.debuffed
                )
                j.state["x_mult"] = float(max(1, empty_slots + stencils))
            case JokerName.FORTUNE_TELLER:
                j.state["mult"] = state.tarots_used
            case JokerName.THROWBACK:
                j.state["x_mult"] = round(1.0 + 0.25 * state.blinds_skipped, 2)
            case JokerName.STONE_JOKER:
                j.state["chips"] = 25 * sum(
                    1 for c in state.full_deck if c.enhancement == Enhancement.STONE
                )
            case JokerName.STEEL_JOKER:
                j.state["x_mult"] = round(
                    1.0 + 0.2 * sum(1 for c in state.full_deck if c.enhancement == Enhancement.STEEL),
                    2,
                )
            case JokerName.DRIVER:
                tally = sum(1 for c in state.full_deck if c.enhancement != Enhancement.NONE)
                j.state["driver_tally"] = tally
                j.state["x_mult"] = 3.0 if tally >= 16 else 1.0
            case JokerName.BLUE_JOKER:
                j.state["chips"] = 2 * len(state.remaining_deck)
            case JokerName.EROSION:
                j.state["mult"] = max(0, 4 * (state.starting_deck_size - len(state.full_deck)))
            case JokerName.CLOUD_9:
                j.state["nine_tally"] = sum(
                    1
                    for c in state.full_deck
                    if c.rank == Rank.NINE and c.enhancement != Enhancement.STONE
                )
            case JokerName.BULL:
                j.state["chips"] = max(0, 2 * state.money)
            case JokerName.BOOTSTRAPS:
                j.state["mult"] = 2 * max(0, state.money // 5)
            case JokerName.BANNER:
                j.state["chips"] = 30 * state.discards_left
            case JokerName.MYSTIC_SUMMIT:
                j.state["mult"] = 15 if state.discards_left == 0 else 0
            case JokerName.BASEBALL_CARD:
                j.state["uncommon_count"] = sum(
                    1 for other in jokers if other.info.rarity == JokerRarity.UNCOMMON
                )


# =========================================================================
# 1. TARGET-SHIFTING & ROTATING JOKERS
# =========================================================================

@joker_hook(JokerName.ANCIENT_JOKER)
def handle_ancient_joker(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event in (TriggerEvent.SETTING_BLIND, TriggerEvent.END_OF_ROUND):
        suit = cv_detect_target_suit(None, joker)
        ctx.state.ancient_suit = suit
        joker.state["target_suit"] = suit


@joker_hook(JokerName.TO_DO_LIST)
def handle_to_do_list(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.SETTING_BLIND and not ctx.is_blueprint:
        if "target_hand" not in joker.state:
            hand = cv_detect_target_poker_hand(None, joker)
            ctx.state.to_do_poker_hand = hand
            joker.state["target_hand"] = hand
    elif ctx.event == TriggerEvent.BEFORE_PLAY:
        target = joker.state.get("target_hand", ctx.state.to_do_poker_hand)
        played_hand = ctx.poker_hand or evaluate_poker_hand(
            ctx.full_hand or ctx.state.hand.selected_cards,
            ctx.state.inventory.jokers,
        )
        if played_hand == target:
            ctx.state.money += 4
    elif ctx.event == TriggerEvent.END_OF_ROUND and not ctx.is_blueprint:
        next_hand = cv_detect_target_poker_hand(None, joker)
        ctx.state.to_do_poker_hand = next_hand
        joker.state["target_hand"] = next_hand


@joker_hook(JokerName.THE_IDOL)
def handle_the_idol(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event in (TriggerEvent.SETTING_BLIND, TriggerEvent.END_OF_ROUND):
        rank = cv_detect_target_rank(None, joker)
        suit = cv_detect_target_suit(None, joker)
        ctx.state.idol_card = (rank, suit)
        joker.state["target_rank"] = rank
        joker.state["target_suit"] = suit


@joker_hook(JokerName.CASTLE)
def handle_castle(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.DISCARD:
        target_suit = joker.state.get("target_suit", ctx.state.castle_suit)
        discarded = ctx.full_hand or (ctx.state.hand.selected_cards if ctx.state and ctx.state.hand else [])
        jokers = ctx.state.inventory.jokers if ctx.state and ctx.state.inventory else None
        matching = sum(1 for c in discarded if card_matches_suit(c, target_suit, jokers))
        if matching > 0:
            joker.state["chips"] = joker.state.get("chips", 0) + (matching * 3)
    elif ctx.event == TriggerEvent.END_OF_ROUND:
        suit = cv_detect_target_suit(None, joker)
        ctx.state.castle_suit = suit
        joker.state["target_suit"] = suit


@joker_hook(JokerName.MAIL_IN_REBATE)
def handle_mail_in_rebate(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.DISCARD:
        target_rank = joker.state.get("target_rank", ctx.state.mail_card_rank)
        discarded = ctx.full_hand or (ctx.state.hand.selected_cards if ctx.state and ctx.state.hand else [])
        matches = sum(
            1
            for c in discarded
            if not getattr(c, "debuffed", False)
            and c.enhancement != Enhancement.STONE
            and c.rank == target_rank
        )
        if matches > 0:
            ctx.state.money += matches * 5
    elif ctx.event == TriggerEvent.END_OF_ROUND and not ctx.is_blueprint:
        rank = cv_detect_target_rank(None, joker)
        ctx.state.mail_card_rank = rank
        joker.state["target_rank"] = rank


# =========================================================================
# 2. ACCUMULATOR & STATE-SCALING JOKERS
# =========================================================================

@joker_hook(JokerName.CEREMONIAL_DAGGER)
def handle_ceremonial_dagger(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.SETTING_BLIND:
        jokers = ctx.state.inventory.jokers
        if joker in jokers:
            idx = jokers.index(joker)
            if idx + 1 < len(jokers):
                sliced = jokers[idx + 1]
                if not sliced.eternal:
                    on_joker_removed(ctx.state, sliced)
                    jokers.pop(idx + 1)
                    added_mult = sliced.sell_value * 2
                    joker.state["mult"] = joker.state.get("mult", 0) + added_mult


@joker_hook(JokerName.GIFT_CARD)
def handle_gift_card(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.END_OF_ROUND:
        for c in ctx.state.inventory.jokers:
            c.extra_value += 1
        for c in ctx.state.inventory.consumables:
            c.extra_value += 1


@joker_hook(JokerName.EGG)
def handle_egg(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.END_OF_ROUND:
        joker.extra_value += 3


@joker_hook(JokerName.FLASH_CARD)
def handle_flash_card(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.REROLL_SHOP:
        joker.state["mult"] = joker.state.get("mult", 0) + 2


@joker_hook(JokerName.RED_CARD)
def handle_red_card(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.SKIP_BOOSTER:
        joker.state["mult"] = joker.state.get("mult", 0) + 3


@joker_hook(JokerName.GREEN_JOKER)
def handle_green_joker(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.BEFORE_PLAY:
        joker.state["mult"] = joker.state.get("mult", 0) + 1
    elif ctx.event == TriggerEvent.DISCARD:
        joker.state["mult"] = max(0, joker.state.get("mult", 0) - 1)


@joker_hook(JokerName.RIDE_THE_BUS)
def handle_ride_the_bus(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.BEFORE_PLAY:
        jokers = ctx.state.inventory.jokers if ctx.state and ctx.state.inventory else None
        scoring = ctx.scoring_cards or (
            ctx.state.hand.get_scoring_cards(jokers) if ctx.state and ctx.state.hand else []
        )
        if any(is_card_face(c, jokers) for c in scoring):
            joker.state["mult"] = 0
        else:
            joker.state["mult"] = joker.state.get("mult", 0) + 1


@joker_hook(JokerName.SQUARE_JOKER)
def handle_square_joker(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.BEFORE_PLAY:
        played = ctx.full_hand or (ctx.state.hand.selected_cards if ctx.state and ctx.state.hand else [])
        if len(played) == 4:
            joker.state["chips"] = joker.state.get("chips", 0) + 4


@joker_hook(JokerName.RUNNER)
def handle_runner(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.BEFORE_PLAY:
        hand = ctx.poker_hand or evaluate_poker_hand(
            ctx.full_hand or (ctx.state.hand.selected_cards if ctx.state and ctx.state.hand else []),
            ctx.state.inventory.jokers if ctx.state else None,
        )
        if hand in (PokerHand.STRAIGHT, PokerHand.STRAIGHT_FLUSH):
            joker.state["chips"] = joker.state.get("chips", 0) + 15


@joker_hook(JokerName.WEE_JOKER)
def handle_wee_joker(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.INDIVIDUAL_CARD_SCORED:
        if (
            ctx.target_card
            and not getattr(ctx.target_card, "debuffed", False)
            and ctx.target_card.enhancement != Enhancement.STONE
            and ctx.target_card.rank == Rank.TWO
        ):
            joker.state["chips"] = joker.state.get("chips", 0) + 8
    elif ctx.event == TriggerEvent.BEFORE_PLAY:
        scoring = ctx.scoring_cards or (
            ctx.state.hand.get_scoring_cards(ctx.state.inventory.jokers)
            if ctx.state and ctx.state.hand
            else []
        )
        twos = sum(
            1
            for c in scoring
            if not getattr(c, "debuffed", False)
            and c.enhancement != Enhancement.STONE
            and c.rank == Rank.TWO
        )
        if twos > 0 and ctx.extra and ctx.extra.get("batch_wee"):
            joker.state["chips"] = joker.state.get("chips", 0) + (twos * 8)


@joker_hook(JokerName.SPARE_TROUSERS)
def handle_spare_trousers(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.BEFORE_PLAY:
        hand = ctx.poker_hand or evaluate_poker_hand(
            ctx.full_hand or (ctx.state.hand.selected_cards if ctx.state and ctx.state.hand else []),
            ctx.state.inventory.jokers if ctx.state else None,
        )
        if hand in (PokerHand.TWO_PAIR, PokerHand.FULL_HOUSE, PokerHand.FLUSH_HOUSE):
            joker.state["mult"] = joker.state.get("mult", 0) + 2


@joker_hook(JokerName.CAMPFIRE)
def handle_campfire(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.CARD_SOLD:
        joker.state["x_mult"] = round(joker.state.get("x_mult", 1.0) + 0.25, 2)
    elif ctx.event == TriggerEvent.END_OF_ROUND:
        if ctx.state.current_blind not in (BlindName.SMALL, BlindName.BIG):
            joker.state["x_mult"] = 1.0


@joker_hook(JokerName.CONSTELLATION)
def handle_constellation(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.USE_CONSUMABLE:
        if ctx.target_card and ctx.target_card.type == CardType.PLANET:
            joker.state["x_mult"] = round(joker.state.get("x_mult", 1.0) + 0.1, 2)
            ctx.state.unique_planets_used.add(str(getattr(ctx.target_card, "name", "")))


@joker_hook(JokerName.LUCKY_CAT)
def handle_lucky_cat(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.INDIVIDUAL_CARD_SCORED:
        if ctx.target_card and not getattr(ctx.target_card, "debuffed", False):
            mult_trig, money_trig = cv_detect_lucky_card_triggered(None)
            if ctx.extra and ctx.extra.get("lucky_triggered"):
                mult_trig = True
            if mult_trig or money_trig:
                joker.state["x_mult"] = round(joker.state.get("x_mult", 1.0) + 0.25, 2)
            if ctx.target_card.enhancement == Enhancement.LUCKY:
                ctx.state.pending_cv_checks.add("money")


@joker_hook(JokerName.HIKER)
def handle_hiker(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.INDIVIDUAL_CARD_SCORED:
        if ctx.target_card and not getattr(ctx.target_card, "debuffed", False):
            ctx.target_card.bonus_chips += 5
    elif ctx.event == TriggerEvent.AFTER_PLAY:
        if ctx.extra and ctx.extra.get("batch_hiker"):
            scoring = ctx.scoring_cards or (
                ctx.state.hand.get_scoring_cards(ctx.state.inventory.jokers)
                if ctx.state and ctx.state.hand
                else []
            )
            for card in scoring:
                if not getattr(card, "debuffed", False):
                    card.bonus_chips += 5


@joker_hook(JokerName.HOLOGRAM)
def handle_hologram(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.CARD_BOUGHT:
        if ctx.extra and ctx.extra.get("already_added_to_deck"):
            return
        if ctx.target_card and ctx.target_card.type == CardType.PLAYING_CARD:
            joker.state["x_mult"] = round(joker.state.get("x_mult", 1.0) + 0.25, 2)


@joker_hook(JokerName.MADNESS)
def handle_madness(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.SETTING_BLIND:
        if ctx.state.current_blind in (BlindName.SMALL, BlindName.BIG):
            joker.state["x_mult"] = round(joker.state.get("x_mult", 1.0) + 0.5, 2)
            destructible = [
                j for j in ctx.state.inventory.jokers if j is not joker and not j.eternal
            ]
            if destructible:
                ctx.state.pending_cv_checks.add("jokers")


@joker_hook(JokerName.VAMPIRE)
def handle_vampire(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.BEFORE_PLAY:
        scoring = ctx.scoring_cards or (
            ctx.state.hand.get_scoring_cards(ctx.state.inventory.jokers)
            if ctx.state and ctx.state.hand
            else []
        )
        for card in scoring:
            if card.enhancement != Enhancement.NONE and not getattr(card, "debuffed", False):
                card.enhancement = Enhancement.NONE
                joker.state["x_mult"] = round(joker.state.get("x_mult", 1.0) + 0.1, 2)


@joker_hook(JokerName.OBELISK)
def handle_obelisk(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.BEFORE_PLAY:
        hand = ctx.poker_hand or evaluate_poker_hand(
            ctx.full_hand or ctx.state.hand.selected_cards,
            ctx.state.inventory.jokers,
        )
        counts = ctx.state.hand_play_counts
        play_more_than = counts.get(hand, 0) if counts else 0
        # In Balatro (card.lua:3545), Obelisk resets iff play_more_than > 0 and NO other hand has played >= play_more_than
        if play_more_than > 0 and not any(k != hand and v >= play_more_than for k, v in counts.items()):
            joker.state["x_mult"] = 1.0
        else:
            joker.state["x_mult"] = round(joker.state.get("x_mult", 1.0) + 0.2, 2)


@joker_hook(JokerName.GLASS_JOKER)
def handle_glass_joker(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.AFTER_PLAY:
        if cv_detect_glass_card_shattered(None):
            joker.state["x_mult"] = round(joker.state.get("x_mult", 1.0) + 0.75, 2)


@joker_hook(JokerName.CANIO)
def handle_canio(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event in (TriggerEvent.DISCARD, TriggerEvent.AFTER_PLAY):
        if cv_detect_card_destroyed(None):
            joker.state["x_mult"] = round(joker.state.get("x_mult", 1.0) + 1.0, 2)


@joker_hook(JokerName.YORICK)
def handle_yorick(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.DISCARD:
        discarded = ctx.full_hand or (ctx.state.hand.selected_cards if ctx.state and ctx.state.hand else [])
        if not discarded:
            return
        discards = joker.state.get("discards", 23) - len(discarded)
        while discards <= 0:
            joker.state["x_mult"] = round(joker.state.get("x_mult", 1.0) + 1.0, 2)
            discards += 23
        joker.state["discards"] = discards


@joker_hook(JokerName.RAMEN)
def handle_ramen(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.DISCARD:
        discarded = ctx.full_hand or (ctx.state.hand.selected_cards if ctx.state and ctx.state.hand else [])
        count = len(discarded)
        if count > 0:
            current_xmult = joker.state.get("x_mult", 2.0)
            new_xmult = round(current_xmult - (count * 0.01), 2)
            if new_xmult <= 1.0:
                if joker in ctx.state.inventory.jokers:
                    on_joker_removed(ctx.state, joker)
                    ctx.state.inventory.jokers.remove(joker)
            else:
                joker.state["x_mult"] = new_xmult


@joker_hook(JokerName.LOYALTY_CARD)
def handle_loyalty_card(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.BEFORE_PLAY:
        rem = joker.state.get("remaining", 5) - 1
        if rem < 0:
            rem = 5
        joker.state["remaining"] = rem


# =========================================================================
# 3. COUNTDOWN & EXPIRY JOKERS
# =========================================================================

@joker_hook(JokerName.POPCORN)
def handle_popcorn(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.END_OF_ROUND:
        current_mult = joker.state.get("mult", 20)
        new_mult = current_mult - 4
        if new_mult <= 0:
            if joker in ctx.state.inventory.jokers:
                on_joker_removed(ctx.state, joker)
                ctx.state.inventory.jokers.remove(joker)
        else:
            joker.state["mult"] = new_mult


@joker_hook(JokerName.ICE_CREAM)
def handle_ice_cream(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.AFTER_PLAY:
        current_chips = joker.state.get("chips", 100)
        new_chips = max(0, current_chips - 5)
        if new_chips <= 0:
            if joker in ctx.state.inventory.jokers:
                on_joker_removed(ctx.state, joker)
                ctx.state.inventory.jokers.remove(joker)
        else:
            joker.state["chips"] = new_chips


@joker_hook(JokerName.TURTLE_BEAN)
def handle_turtle_bean(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.END_OF_ROUND:
        current_size = joker.state.get("h_size", 5)
        new_size = current_size - 1
        ctx.state.hand.max_size = max(0, ctx.state.hand.max_size - 1)
        if new_size <= 0:
            joker.state["h_size"] = 0
            if joker in ctx.state.inventory.jokers:
                ctx.state.inventory.jokers.remove(joker)
        else:
            joker.state["h_size"] = new_size


@joker_hook(JokerName.SELTZER)
def handle_seltzer(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.AFTER_PLAY:
        remaining = joker.state.get("hands_left", 10) - 1
        if remaining <= 0:
            if joker in ctx.state.inventory.jokers:
                on_joker_removed(ctx.state, joker)
                ctx.state.inventory.jokers.remove(joker)
        else:
            joker.state["hands_left"] = remaining


@joker_hook(JokerName.INVISIBLE_JOKER)
def handle_invisible_joker(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.END_OF_ROUND:
        joker.state["rounds"] = joker.state.get("rounds", 0) + 1
    elif ctx.event == TriggerEvent.CARD_SOLD:
        if ctx.target_card == joker and joker.state.get("rounds", 0) >= 2:
            ctx.state.pending_cv_checks.add("jokers")


@joker_hook(JokerName.GROS_MICHEL)
def handle_gros_michel(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.END_OF_ROUND:
        ctx.state.pending_cv_checks.add("jokers")


@joker_hook(JokerName.CAVENDISH)
def handle_cavendish(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.END_OF_ROUND:
        ctx.state.pending_cv_checks.add("jokers")


# =========================================================================
# 4. ROUND & BLIND START JOKERS (Hands, Discards, Spawns)
# =========================================================================

@joker_hook(JokerName.BURGLAR)
def handle_burglar(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.SETTING_BLIND:
        ctx.state.hands_left += 3
        ctx.state.discards_left = 0


@joker_hook(JokerName.DRUNKARD)
def handle_drunkard(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.SETTING_BLIND:
        ctx.state.discards_left += 1


@joker_hook(JokerName.MERRY_ANDY)
def handle_merry_andy(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.SETTING_BLIND:
        ctx.state.discards_left += 3


@joker_hook(JokerName.TROUBADOUR)
def handle_troubadour(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.SETTING_BLIND:
        ctx.state.hands_left = max(1, ctx.state.hands_left - 1)


@joker_hook(JokerName.MARBLE_JOKER)
def handle_marble_joker(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.SETTING_BLIND:
        stone_card = PlayingCard(
            rank=Rank.NONE,
            suit=Suit.NONE,
            enhancement=Enhancement.STONE,
        )
        ctx.state.add_card_to_deck(stone_card, add_to_hand=False)


@joker_hook(JokerName.CARTOMANCER)
def handle_cartomancer(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.SETTING_BLIND:
        if ctx.state.inventory and ctx.state.inventory.can_add_consumable:
            ctx.state.pending_cv_checks.add("consumables")


@joker_hook(JokerName.RIFF_RAFF)
def handle_riff_raff(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.SETTING_BLIND:
        if ctx.state.inventory and ctx.state.inventory.can_add_joker:
            ctx.state.pending_cv_checks.add("jokers")


@joker_hook(JokerName.CERTIFICATE)
def handle_certificate(joker: Joker, ctx: TriggerContext):
    if ctx.event in (TriggerEvent.FIRST_HAND_DRAWN, TriggerEvent.SETTING_BLIND):
        ranks = [
            Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX,
            Rank.SEVEN, Rank.EIGHT, Rank.NINE, Rank.TEN,
            Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE,
        ]
        suits = [Suit.SPADES, Suit.HEARTS, Suit.CLUBS, Suit.DIAMONDS]
        seals = [Seal.RED, Seal.BLUE, Seal.GOLD, Seal.PURPLE]
        cert_card = PlayingCard(
            rank=random.choice(ranks),
            suit=random.choice(suits),
            seal=random.choice(seals),
        )
        ctx.state.add_card_to_deck(cert_card, add_to_hand=True)
        ctx.state.pending_cv_checks.add("hand")


@joker_hook(JokerName.CHICOT)
def handle_chicot(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.SETTING_BLIND:
        if ctx.state.current_blind not in (BlindName.SMALL, BlindName.BIG):
            ctx.state.boss_disabled = True


@joker_hook(JokerName.LUCHADOR)
def handle_luchador(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.CARD_SOLD:
        if ctx.target_card == joker and ctx.state.current_blind not in (BlindName.SMALL, BlindName.BIG):
            ctx.state.boss_disabled = True
            for j in ctx.state.inventory.jokers:
                if not (j.perishable and j.perish_tally <= 0):
                    j.debuffed = False
            for hc in ctx.state.hand.cards:
                hc.card.debuffed = False


@joker_hook(JokerName.DIET_COLA)
def handle_diet_cola(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.CARD_SOLD:
        if ctx.target_card == joker or ctx.is_blueprint:
            ctx.state.active_tags.append(SkipTag.DOUBLE_TAG)


# =========================================================================
# 5. DISCARD-TRIGGERED JOKERS
# =========================================================================

@joker_hook(JokerName.TRADING_CARD)
def handle_trading_card(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.DISCARD:
        discarded = ctx.full_hand or (ctx.state.hand.selected_cards if ctx.state and ctx.state.hand else [])
        if ctx.state.discards_used_this_round == 1 and len(discarded) == 1:
            target = discarded[0]
            ctx.state.money += 3
            ctx.state.remove_card_from_deck(target)


@joker_hook(JokerName.FACELESS_JOKER)
def handle_faceless_joker(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.DISCARD:
        discarded = ctx.full_hand or (ctx.state.hand.selected_cards if ctx.state and ctx.state.hand else [])
        jokers = ctx.state.inventory.jokers if ctx.state and ctx.state.inventory else None
        faces = sum(1 for c in discarded if is_card_face(c, jokers))
        if faces >= 3:
            ctx.state.money += 5


@joker_hook(JokerName.BURNT_JOKER)
def handle_burnt_joker(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.DISCARD:
        if ctx.state.discards_used_this_round == 1:
            discarded = ctx.full_hand or (ctx.state.hand.selected_cards if ctx.state and ctx.state.hand else [])
            hand_type = evaluate_poker_hand(discarded, ctx.state.inventory.jokers)
            ctx.state.hand_levels[hand_type] = ctx.state.hand_levels.get(hand_type, 1) + 1


# =========================================================================
# 6. ECONOMY & PAYOUT JOKERS
# =========================================================================

@joker_hook(JokerName.GOLDEN_JOKER)
def handle_golden_joker(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.END_OF_ROUND:
        ctx.state.money += 4


@joker_hook(JokerName.DELAYED_GRATIFICATION)
def handle_delayed_gratification(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.END_OF_ROUND:
        if ctx.state.discards_used_this_round == 0:
            ctx.state.money += ctx.state.discards_left * 2


@joker_hook(JokerName.ROCKET)
def handle_rocket(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.END_OF_ROUND:
        payout = joker.state.get("payout", 1)
        if ctx.state.current_blind not in (BlindName.SMALL, BlindName.BIG):
            payout += 2
            joker.state["payout"] = payout
        ctx.state.money += payout


@joker_hook(JokerName.CLOUD_9)
def handle_cloud_9(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.END_OF_ROUND:
        deck_to_check = ctx.state.full_deck if ctx.state.full_deck else [c.card for c in ctx.state.hand.cards]
        nines = sum(
            1 for c in deck_to_check if c.rank == Rank.NINE and c.enhancement != Enhancement.STONE
        )
        ctx.state.money += nines


@joker_hook(JokerName.SATELLITE)
def handle_satellite(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.END_OF_ROUND:
        ctx.state.money += len(ctx.state.unique_planets_used)


@joker_hook(JokerName.TO_THE_MOON)
def handle_to_the_moon(joker: Joker, ctx: TriggerContext):
    pass


@joker_hook(JokerName.CHAOS_THE_CLOWN)
def handle_chaos_the_clown(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.END_OF_ROUND:
        ctx.state.free_rerolls += 1


# =========================================================================
# 7. SPECIAL PLAY / SPAWN / CARD MODIFIER JOKERS
# =========================================================================

@joker_hook(JokerName.PERKEO)
def handle_perkeo(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.LEAVE_SHOP:
        if ctx.state.inventory and ctx.state.inventory.consumables:
            ctx.state.pending_cv_checks.add("consumables")


@joker_hook(JokerName.DNA)
def handle_dna(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.BEFORE_PLAY:
        played = ctx.full_hand or (ctx.state.hand.selected_cards if ctx.state and ctx.state.hand else [])
        if ctx.state.hands_played_this_round <= 1 and len(played) == 1:
            copied_card = copy.deepcopy(played[0])
            copied_card.id = 0
            ctx.state.add_card_to_deck(copied_card, add_to_hand=True)


@joker_hook(JokerName.SIXTH_SENSE)
def handle_sixth_sense(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.BEFORE_PLAY:
        played = ctx.full_hand or (ctx.state.hand.selected_cards if ctx.state and ctx.state.hand else [])
        if ctx.state.hands_played_this_round <= 1 and len(played) == 1:
            c = played[0]
            if c.rank == Rank.SIX and c.enhancement != Enhancement.STONE:
                ctx.state.remove_card_from_deck(c)
                if ctx.state.inventory and ctx.state.inventory.can_add_consumable:
                    try:
                        from game.balatro_models import Spectral, SpectralName
                    except ImportError:
                        from balatro_models import Spectral, SpectralName
                    ctx.state.inventory.consumables.append(Spectral(name=SpectralName.ANKH))
                    ctx.state.pending_cv_checks.add("consumables")


@joker_hook(JokerName.SEANCE)
def handle_seance(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.BEFORE_PLAY:
        hand = ctx.poker_hand or evaluate_poker_hand(
            ctx.full_hand or ctx.state.hand.selected_cards,
            ctx.state.inventory.jokers,
        )
        if hand == PokerHand.STRAIGHT_FLUSH and ctx.state.inventory.can_add_consumable:
            ctx.state.pending_cv_checks.add("consumables")


@joker_hook(JokerName.SUPERPOSITION)
def handle_superposition(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.BEFORE_PLAY:
        hand = ctx.poker_hand or evaluate_poker_hand(
            ctx.full_hand or ctx.state.hand.selected_cards,
            ctx.state.inventory.jokers,
        )
        scoring = ctx.scoring_cards or ctx.state.hand.get_scoring_cards(ctx.state.inventory.jokers)
        has_ace = any(c.rank == Rank.ACE and c.enhancement != Enhancement.STONE for c in scoring)
        if hand in (PokerHand.STRAIGHT, PokerHand.STRAIGHT_FLUSH) and has_ace:
            if ctx.state.inventory.can_add_consumable:
                ctx.state.pending_cv_checks.add("consumables")


@joker_hook(JokerName.VAGABOND)
def handle_vagabond(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.BEFORE_PLAY:
        if ctx.state.money <= 4 and ctx.state.inventory.can_add_consumable:
            ctx.state.pending_cv_checks.add("consumables")


@joker_hook(JokerName.HALLUCINATION)
def handle_hallucination(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.OPEN_BOOSTER:
        if ctx.state.inventory and ctx.state.inventory.can_add_consumable:
            ctx.state.pending_cv_checks.add("consumables")


@joker_hook(JokerName.EIGHT_BALL)
def handle_eight_ball(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.INDIVIDUAL_CARD_SCORED:
        if (
            ctx.target_card
            and not getattr(ctx.target_card, "debuffed", False)
            and ctx.target_card.enhancement != Enhancement.STONE
            and ctx.target_card.rank == Rank.EIGHT
            and ctx.state.inventory.can_add_consumable
        ):
            ctx.state.pending_cv_checks.add("consumables")


@joker_hook(JokerName.SPACE_JOKER)
def handle_space_joker(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.BEFORE_PLAY:
        ctx.state.pending_cv_checks.add("hand_levels")


@joker_hook(JokerName.BUSINESS_CARD)
def handle_business_card(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.INDIVIDUAL_CARD_SCORED:
        jokers = ctx.state.inventory.jokers if ctx.state and ctx.state.inventory else None
        if ctx.target_card and is_card_face(ctx.target_card, jokers):
            ctx.state.pending_cv_checks.add("money")


@joker_hook(JokerName.RESERVED_PARKING)
def handle_reserved_parking(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.BEFORE_PLAY:
        jokers = ctx.state.inventory.jokers if ctx.state and ctx.state.inventory else None
        held = [hc.card for hc in ctx.state.hand.cards if not hc.selected]
        if any(is_card_face(c, jokers) for c in held):
            ctx.state.pending_cv_checks.add("money")


@joker_hook(JokerName.MATADOR)
def handle_matador(joker: Joker, ctx: TriggerContext):
    if ctx.event in (TriggerEvent.BEFORE_PLAY, TriggerEvent.DISCARD):
        if (
            ctx.state.current_blind not in (BlindName.SMALL, BlindName.BIG)
            and not ctx.state.boss_disabled
        ):
            ctx.state.pending_cv_checks.add("money")


@joker_hook(JokerName.MIDAS_MASK)
def handle_midas_mask(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.AFTER_PLAY:
        jokers = ctx.state.inventory.jokers if ctx.state and ctx.state.inventory else None
        scoring = ctx.scoring_cards or (
            ctx.state.hand.get_scoring_cards(jokers) if ctx.state and ctx.state.hand else []
        )
        for c in scoring:
            if is_card_face(c, jokers):
                c.enhancement = Enhancement.GOLD


@joker_hook(JokerName.ROUGH_GEM)
def handle_rough_gem(joker: Joker, ctx: TriggerContext):
    jokers = ctx.state.inventory.jokers if ctx.state and ctx.state.inventory else None
    if ctx.event == TriggerEvent.INDIVIDUAL_CARD_SCORED:
        if ctx.target_card and card_matches_suit(ctx.target_card, Suit.DIAMONDS, jokers):
            ctx.state.money += 1
    elif ctx.event == TriggerEvent.AFTER_PLAY:
        if ctx.extra and ctx.extra.get("batch_payout"):
            scoring = ctx.scoring_cards or (
                ctx.state.hand.get_scoring_cards(jokers) if ctx.state and ctx.state.hand else []
            )
            diamonds = sum(1 for c in scoring if card_matches_suit(c, Suit.DIAMONDS, jokers))
            ctx.state.money += diamonds


@joker_hook(JokerName.GOLDEN_TICKET)
def handle_golden_ticket(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.INDIVIDUAL_CARD_SCORED:
        if (
            ctx.target_card
            and not getattr(ctx.target_card, "debuffed", False)
            and ctx.target_card.enhancement == Enhancement.GOLD
        ):
            ctx.state.money += 4
    elif ctx.event == TriggerEvent.AFTER_PLAY:
        if ctx.extra and ctx.extra.get("batch_payout"):
            scoring = ctx.scoring_cards or (
                ctx.state.hand.get_scoring_cards(ctx.state.inventory.jokers)
                if ctx.state and ctx.state.hand
                else []
            )
            golds = sum(
                1
                for c in scoring
                if not getattr(c, "debuffed", False) and c.enhancement == Enhancement.GOLD
            )
            ctx.state.money += golds * 4


@joker_hook(JokerName.HIT_THE_ROAD)
def handle_hit_the_road(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.DISCARD:
        discarded = ctx.full_hand or (ctx.state.hand.selected_cards if ctx.state and ctx.state.hand else [])
        jacks = sum(
            1
            for c in discarded
            if not getattr(c, "debuffed", False)
            and c.enhancement != Enhancement.STONE
            and c.rank == Rank.JACK
        )
        if jacks > 0:
            joker.state["x_mult"] = round(joker.state.get("x_mult", 1.0) + (jacks * 0.5), 2)
    elif ctx.event == TriggerEvent.END_OF_ROUND:
        joker.state["x_mult"] = 1.0


# =========================================================================
# 8. PURE PASSIVE SCORING & STATIC JOKERS
# =========================================================================
STATIC_SCORING_JOKERS = [
    JokerName.JOKER, JokerName.GREEDY_JOKER, JokerName.LUSTY_JOKER, JokerName.WRATHFUL_JOKER,
    JokerName.GLUTTONOUS_JOKER, JokerName.JOLLY_JOKER, JokerName.ZANY_JOKER, JokerName.MAD_JOKER,
    JokerName.CRAZY_JOKER, JokerName.DROLL_JOKER, JokerName.SLY_JOKER, JokerName.WILY_JOKER,
    JokerName.CLEVER_JOKER, JokerName.DEVIOUS_JOKER, JokerName.CRAFTY_JOKER, JokerName.HALF_JOKER,
    JokerName.JOKER_STENCIL, JokerName.FOUR_FINGERS, JokerName.MIME, JokerName.CREDIT_CARD,
    JokerName.BANNER, JokerName.MYSTIC_SUMMIT, JokerName.MISPRINT, JokerName.DUSK,
    JokerName.RAISED_FIST, JokerName.CHAOS_THE_CLOWN, JokerName.FIBONACCI, JokerName.STEEL_JOKER,
    JokerName.SCARY_FACE, JokerName.ABSTRACT_JOKER, JokerName.HACK, JokerName.PAREIDOLIA,
    JokerName.EVEN_STEVEN, JokerName.ODD_TODD, JokerName.SCHOLAR, JokerName.BUSINESS_CARD,
    JokerName.SUPERNOVA, JokerName.SPACE_JOKER, JokerName.BLACKBOARD, JokerName.SPLASH,
    JokerName.BLUE_JOKER, JokerName.CARD_SHARP, JokerName.SHORTCUT, JokerName.BARON,
    JokerName.PHOTOGRAPH, JokerName.EROSION, JokerName.RESERVED_PARKING, JokerName.FORTUNE_TELLER,
    JokerName.JUGGLER, JokerName.DRUNKARD, JokerName.STONE_JOKER, JokerName.BASEBALL_CARD,
    JokerName.BULL, JokerName.WALKIE_TALKIE, JokerName.SMILEY_FACE,
    JokerName.MR_BONES, JokerName.ACROBAT, JokerName.SOCK_AND_BUSKIN, JokerName.SWASHBUCKLER,
    JokerName.TROUBADOUR, JokerName.SMEARED_JOKER, JokerName.THROWBACK, JokerName.HANGING_CHAD,
    JokerName.BLOODSTONE, JokerName.ARROWHEAD, JokerName.ONYX_AGATE, JokerName.SHOWMAN,
    JokerName.FLOWER_POT, JokerName.MERRY_ANDY, JokerName.OOPS_ALL_6S, JokerName.SEEING_DOUBLE,
    JokerName.MATADOR, JokerName.THE_DUO, JokerName.THE_TRIO, JokerName.THE_FAMILY,
    JokerName.THE_ORDER, JokerName.THE_TRIBE, JokerName.STUNTMAN, JokerName.SHOOT_THE_MOON,
    JokerName.DRIVER, JokerName.ASTRONOMER, JokerName.BOOTSTRAPS, JokerName.TRIBOULET,
    JokerName.BLUEPRINT, JokerName.BRAINSTORM,
]


def _noop_hook(joker: Joker, ctx: TriggerContext):
    pass


for static_joker in STATIC_SCORING_JOKERS:
    if static_joker not in JOKER_EFFECTS:
        JOKER_EFFECTS[static_joker] = _noop_hook


# =========================================================================
# CENTRAL DISPATCHER & BLUEPRINT / BRAINSTORM RESOLUTION
# =========================================================================

def resolve_copy_target(joker: Joker, jokers: list[Joker], visited: set[int] | None = None) -> Joker | None:
    """Resolves the ultimate concrete Joker that Blueprint or Brainstorm copies.

    Follows copy chains (e.g. Blueprint copying Brainstorm) with cycle protection.
    Returns None if:
    - Target slot does not exist (Blueprint on right edge, or Brainstorm on slot 0 targeting itself).
    - Chain forms a cycle.
    - Target joker has blueprint_compat=False.
    """
    if visited is None:
        visited = set()

    j_id = id(joker)
    if j_id in visited:
        return None
    visited.add(j_id)

    if joker.name == JokerName.BLUEPRINT:
        if joker not in jokers:
            return None
        idx = jokers.index(joker)
        if idx + 1 >= len(jokers):
            return None
        candidate = jokers[idx + 1]
    elif joker.name == JokerName.BRAINSTORM:
        if not jokers or jokers[0] == joker:
            return None
        candidate = jokers[0]
    else:
        candidate = joker

    # If the candidate is also a copy Joker, recursively resolve down the chain
    if candidate.name in (JokerName.BLUEPRINT, JokerName.BRAINSTORM):
        return resolve_copy_target(candidate, jokers, visited)

    # Check whether the target Joker is compatible with Blueprint / Brainstorm
    if not candidate.blueprint_compat:
        return None

    return candidate


def dispatch_jokers(ctx: TriggerContext):
    """
    Dispatches the event to all Jokers currently in inventory,
    fully respecting Blueprint and Brainstorm copy targets, compatibility rules, and debuff status.
    """
    jokers = list(ctx.state.inventory.jokers)
    for j in jokers:
        init_joker_default_state(ctx.state, j)
        if j.debuffed:
            continue
        if j.name in (JokerName.BLUEPRINT, JokerName.BRAINSTORM):
            target = resolve_copy_target(j, jokers)
            if target and not target.debuffed and target.blueprint_compat and target.name in JOKER_EFFECTS:
                copy_ctx = TriggerContext(
                    event=ctx.event,
                    state=ctx.state,
                    action=ctx.action,
                    target_card=ctx.target_card,
                    scoring_cards=ctx.scoring_cards,
                    full_hand=ctx.full_hand,
                    poker_hand=ctx.poker_hand,
                    extra=ctx.extra,
                    is_blueprint=True,
                )
                JOKER_EFFECTS[target.name](target, copy_ctx)
        elif j.name in JOKER_EFFECTS:
            JOKER_EFFECTS[j.name](j, ctx)
    sync_dynamic_joker_states(ctx.state)
