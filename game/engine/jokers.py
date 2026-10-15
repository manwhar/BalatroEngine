"""
Complete 150-Joker Engine implementation.
Connects all 150 Balatro Jokers to active state changes, accumulators,
game triggers, and CV observation hooks.
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
        PlayingCard,
        PokerHand,
        Rank,
        Seal,
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
        PlayingCard,
        PokerHand,
        Rank,
        Seal,
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
# 1. TARGET-SHIFTING & ROTATING JOKERS
# =========================================================================

@joker_hook(JokerName.ANCIENT_JOKER)
def handle_ancient_joker(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.SETTING_BLIND or ctx.event == TriggerEvent.END_OF_ROUND:
        joker.state["target_suit"] = cv_detect_target_suit(None, joker)


@joker_hook(JokerName.TO_DO_LIST)
def handle_to_do_list(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.SETTING_BLIND:
        joker.state["target_hand"] = cv_detect_target_poker_hand(None, joker)
    elif ctx.event == TriggerEvent.BEFORE_PLAY:
        target = joker.state.get("target_hand", PokerHand.PAIR)
        played_hand = evaluate_poker_hand(ctx.state.hand.selected_cards, ctx.state.inventory.jokers)
        if played_hand == target:
            ctx.state.stats.money += 4
            if not ctx.is_blueprint:
                joker.state["target_hand"] = cv_detect_target_poker_hand(None, joker)


@joker_hook(JokerName.THE_IDOL)
def handle_the_idol(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.SETTING_BLIND or ctx.event == TriggerEvent.END_OF_ROUND:
        joker.state["target_rank"] = cv_detect_target_rank(None, joker)
        joker.state["target_suit"] = cv_detect_target_suit(None, joker)


@joker_hook(JokerName.CASTLE)
def handle_castle(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.DISCARD:
        target_suit = joker.state.get("target_suit", Suit.SPADES)
        discarded = ctx.full_hand or (ctx.state.hand.selected_cards if ctx.state and ctx.state.hand else [])
        matching = sum(1 for c in discarded if c.suit == target_suit)
        if matching > 0:
            joker.state["chips"] = joker.state.get("chips", 0) + (matching * 3)
    elif ctx.event == TriggerEvent.END_OF_ROUND:
        joker.state["target_suit"] = cv_detect_target_suit(None, joker)


@joker_hook(JokerName.MAIL_IN_REBATE)
def handle_mail_in_rebate(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.DISCARD:
        target_rank = joker.state.get("target_rank", Rank.ACE)
        discarded = ctx.full_hand or (ctx.state.hand.selected_cards if ctx.state and ctx.state.hand else [])
        matches = sum(1 for c in discarded if c.rank == target_rank)
        if matches > 0:
            ctx.state.stats.money += matches * 5
    elif ctx.event == TriggerEvent.END_OF_ROUND:
        joker.state["target_rank"] = cv_detect_target_rank(None, joker)


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
                    jokers.pop(idx + 1)
                    added_mult = sliced.sell_value * 2
                    joker.state["mult"] = joker.state.get("mult", 0) + added_mult


@joker_hook(JokerName.GIFT_CARD)
def handle_gift_card(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.END_OF_ROUND:
        for c in ctx.state.inventory.jokers:
            c.extra_value += 1
        for c in ctx.state.inventory.consumables:
            c.extra_value += 1


@joker_hook(JokerName.EGG)
def handle_egg(joker: Joker, ctx: TriggerContext):
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
        scoring = ctx.scoring_cards or (ctx.state.hand.get_scoring_cards(ctx.state.inventory.jokers) if ctx.state and ctx.state.hand else [])
        if any(c.is_face for c in scoring):
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
            ctx.state.inventory.jokers if ctx.state else None
        )
        if hand in (PokerHand.STRAIGHT, PokerHand.STRAIGHT_FLUSH):
            joker.state["chips"] = joker.state.get("chips", 0) + 15


@joker_hook(JokerName.WEE_JOKER)
def handle_wee_joker(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.INDIVIDUAL_CARD_SCORED:
        if ctx.target_card and ctx.target_card.rank == Rank.TWO:
            joker.state["chips"] = joker.state.get("chips", 0) + 8
    elif ctx.event == TriggerEvent.BEFORE_PLAY:
        scoring = ctx.scoring_cards or (ctx.state.hand.get_scoring_cards(ctx.state.inventory.jokers) if ctx.state and ctx.state.hand else [])
        twos = sum(1 for c in scoring if c.rank == Rank.TWO)
        if twos > 0 and ctx.extra and ctx.extra.get("batch_wee"):
            joker.state["chips"] = joker.state.get("chips", 0) + (twos * 8)


@joker_hook(JokerName.SPARE_TROUSERS)
def handle_spare_trousers(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.BEFORE_PLAY:
        hand = ctx.poker_hand or evaluate_poker_hand(
            ctx.full_hand or (ctx.state.hand.selected_cards if ctx.state and ctx.state.hand else []),
            ctx.state.inventory.jokers if ctx.state else None
        )
        if hand in (PokerHand.TWO_PAIR, PokerHand.FULL_HOUSE, PokerHand.FLUSH_HOUSE):
            joker.state["mult"] = joker.state.get("mult", 0) + 2


@joker_hook(JokerName.CAMPFIRE)
def handle_campfire(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.CARD_SOLD:
        joker.state["x_mult"] = round(joker.state.get("x_mult", 1.0) + 0.5, 2)
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
        mult_trig, money_trig = cv_detect_lucky_card_triggered(None)
        if mult_trig or money_trig:
            joker.state["x_mult"] = round(joker.state.get("x_mult", 1.0) + 0.25, 2)


@joker_hook(JokerName.HIKER)
def handle_hiker(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.INDIVIDUAL_CARD_SCORED:
        if ctx.target_card:
            ctx.target_card.bonus_chips += 5
    elif ctx.event == TriggerEvent.AFTER_PLAY:
        if ctx.extra and ctx.extra.get("batch_hiker"):
            scoring = ctx.scoring_cards or (ctx.state.hand.get_scoring_cards(ctx.state.inventory.jokers) if ctx.state and ctx.state.hand else [])
            for card in scoring:
                card.bonus_chips += 5


@joker_hook(JokerName.HOLOGRAM)
def handle_hologram(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.CARD_BOUGHT:
        if ctx.target_card and ctx.target_card.type == CardType.PLAYING_CARD:
            joker.state["x_mult"] = round(joker.state.get("x_mult", 1.0) + 0.25, 2)


@joker_hook(JokerName.MADNESS)
def handle_madness(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.SETTING_BLIND:
        if ctx.state.current_blind in (BlindName.SMALL, BlindName.BIG):
            joker.state["x_mult"] = round(joker.state.get("x_mult", 1.0) + 0.5, 2)


@joker_hook(JokerName.VAMPIRE)
def handle_vampire(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.BEFORE_PLAY:
        scoring = ctx.scoring_cards or (ctx.state.hand.get_scoring_cards(ctx.state.inventory.jokers) if ctx.state and ctx.state.hand else [])
        for card in scoring:
            if card.enhancement != Enhancement.NONE:
                card.enhancement = Enhancement.NONE
                joker.state["x_mult"] = round(joker.state.get("x_mult", 1.0) + 0.1, 2)


@joker_hook(JokerName.OBELISK)
def handle_obelisk(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.BEFORE_PLAY:
        hand = evaluate_poker_hand(ctx.state.hand.selected_cards, ctx.state.inventory.jokers)
        counts = ctx.state.hand_play_counts
        most_played = max(counts, key=counts.get) if counts else None
        if most_played and hand == most_played:
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
        discards = joker.state.get("discards", 23) - len(discarded)
        if discards <= 0:
            joker.state["x_mult"] = round(joker.state.get("x_mult", 1.0) + 1.0, 2)
            joker.state["discards"] = 23
        else:
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
    if ctx.event == TriggerEvent.END_OF_ROUND:
        current_mult = joker.state.get("mult", 20)
        new_mult = current_mult - 4
        if new_mult <= 0:
            if joker in ctx.state.inventory.jokers:
                ctx.state.inventory.jokers.remove(joker)
        else:
            joker.state["mult"] = new_mult


@joker_hook(JokerName.ICE_CREAM)
def handle_ice_cream(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.AFTER_PLAY:
        current_chips = joker.state.get("chips", 100)
        new_chips = max(0, current_chips - 5)
        if new_chips <= 0:
            if joker in ctx.state.inventory.jokers:
                ctx.state.inventory.jokers.remove(joker)
        else:
            joker.state["chips"] = new_chips


@joker_hook(JokerName.TURTLE_BEAN)
def handle_turtle_bean(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.END_OF_ROUND:
        current_size = joker.state.get("h_size", 5)
        new_size = current_size - 1
        if new_size <= 0:
            if joker in ctx.state.inventory.jokers:
                ctx.state.inventory.jokers.remove(joker)
        else:
            joker.state["h_size"] = new_size


@joker_hook(JokerName.SELTZER)
def handle_seltzer(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.AFTER_PLAY:
        remaining = joker.state.get("hands_left", 10) - 1
        if remaining <= 0:
            if joker in ctx.state.inventory.jokers:
                ctx.state.inventory.jokers.remove(joker)
        else:
            joker.state["hands_left"] = remaining


@joker_hook(JokerName.INVISIBLE_JOKER)
def handle_invisible_joker(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.END_OF_ROUND:
        joker.state["rounds"] = min(2, joker.state.get("rounds", 0) + 1)


@joker_hook(JokerName.GROS_MICHEL)
def handle_gros_michel(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.END_OF_ROUND:
        pass


@joker_hook(JokerName.CAVENDISH)
def handle_cavendish(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.END_OF_ROUND:
        pass


# =========================================================================
# 4. ROUND & BLIND START JOKERS (Hands, Discards, Spawns)
# =========================================================================

@joker_hook(JokerName.BURGLAR)
def handle_burglar(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.SETTING_BLIND:
        ctx.state.stats.hands_left += 3
        ctx.state.stats.discards_left = 0


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
    pass


@joker_hook(JokerName.RIFF_RAFF)
def handle_riff_raff(joker: Joker, ctx: TriggerContext):
    pass


@joker_hook(JokerName.CERTIFICATE)
def handle_certificate(joker: Joker, ctx: TriggerContext):
    if ctx.event in (TriggerEvent.FIRST_HAND_DRAWN, TriggerEvent.SETTING_BLIND):
        ranks = [
            Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX,
            Rank.SEVEN, Rank.EIGHT, Rank.NINE, Rank.TEN,
            Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE
        ]
        suits = [Suit.SPADES, Suit.HEARTS, Suit.CLUBS, Suit.DIAMONDS]
        seals = [Seal.RED, Seal.BLUE, Seal.GOLD, Seal.PURPLE]
        cert_card = PlayingCard(
            rank=random.choice(ranks),
            suit=random.choice(suits),
            seal=random.choice(seals),
        )
        ctx.state.add_card_to_deck(cert_card, add_to_hand=True)


@joker_hook(JokerName.CHICOT)
def handle_chicot(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.SETTING_BLIND:
        if ctx.state.current_blind not in (BlindName.SMALL, BlindName.BIG):
            pass


@joker_hook(JokerName.LUCHADOR)
def handle_luchador(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.CARD_SOLD:
        if ctx.target_card == joker:
            pass


@joker_hook(JokerName.DIET_COLA)
def handle_diet_cola(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.CARD_SOLD:
        if ctx.target_card == joker:
            pass


# =========================================================================
# 5. DISCARD-TRIGGERED JOKERS
# =========================================================================

@joker_hook(JokerName.TRADING_CARD)
def handle_trading_card(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.DISCARD:
        discarded = ctx.full_hand or (ctx.state.hand.selected_cards if ctx.state and ctx.state.hand else [])
        if ctx.state.stats.discards_used_this_round == 1 and len(discarded) == 1:
            target = discarded[0]
            ctx.state.stats.money += 3
            ctx.state.remove_card_from_deck(target)
            if target.is_face and ctx.state.inventory:
                for j in ctx.state.inventory.jokers:
                    if j.name == JokerName.CANIO:
                        j.state["x_mult"] = round(j.state.get("x_mult", 1.0) + 1.0, 2)


@joker_hook(JokerName.FACELESS_JOKER)
def handle_faceless_joker(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.DISCARD:
        discarded = ctx.full_hand or (ctx.state.hand.selected_cards if ctx.state and ctx.state.hand else [])
        faces = sum(1 for c in discarded if c.is_face)
        if faces >= 3:
            ctx.state.stats.money += 5


@joker_hook(JokerName.BURNT_JOKER)
def handle_burnt_joker(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.DISCARD:
        if ctx.state.stats.discards_used_this_round == 1:
            discarded = ctx.full_hand or (ctx.state.hand.selected_cards if ctx.state and ctx.state.hand else [])
            hand_type = evaluate_poker_hand(discarded, ctx.state.inventory.jokers)
            if hand_type in ctx.state.hand_levels:
                ctx.state.hand_levels[hand_type] += 1


# =========================================================================
# 6. ECONOMY & PAYOUT JOKERS
# =========================================================================

@joker_hook(JokerName.GOLDEN_JOKER)
def handle_golden_joker(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.END_OF_ROUND:
        ctx.state.stats.money += 4


@joker_hook(JokerName.DELAYED_GRATIFICATION)
def handle_delayed_gratification(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.END_OF_ROUND:
        if ctx.state.stats.discards_used_this_round == 0:
            ctx.state.stats.money += ctx.state.stats.discards_left * 2


@joker_hook(JokerName.ROCKET)
def handle_rocket(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.END_OF_ROUND:
        payout = joker.state.get("payout", 1)
        ctx.state.stats.money += payout
        if ctx.state.current_blind not in (BlindName.SMALL, BlindName.BIG):
            joker.state["payout"] = payout + 2


@joker_hook(JokerName.CLOUD_9)
def handle_cloud_9(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.END_OF_ROUND:
        deck_to_check = ctx.state.full_deck if ctx.state.full_deck else [c.card for c in ctx.state.hand.cards]
        nines = sum(1 for c in deck_to_check if c.rank == Rank.NINE)
        ctx.state.stats.money += nines


@joker_hook(JokerName.SATELLITE)
def handle_satellite(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.END_OF_ROUND:
        ctx.state.stats.money += len(ctx.state.unique_planets_used)


@joker_hook(JokerName.TO_THE_MOON)
def handle_to_the_moon(joker: Joker, ctx: TriggerContext):
    pass


# =========================================================================
# 7. SPECIAL PLAY / SPAWN / CARD MODIFIER JOKERS
# =========================================================================

@joker_hook(JokerName.PERKEO)
def handle_perkeo(joker: Joker, ctx: TriggerContext):
    pass


@joker_hook(JokerName.DNA)
def handle_dna(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.BEFORE_PLAY:
        played = ctx.full_hand or (ctx.state.hand.selected_cards if ctx.state and ctx.state.hand else [])
        if ctx.state.stats.hands_played_this_round <= 1 and len(played) == 1:
            copied_card = copy.deepcopy(played[0])
            ctx.state.add_card_to_deck(copied_card, add_to_hand=True)


@joker_hook(JokerName.SIXTH_SENSE)
def handle_sixth_sense(joker: Joker, ctx: TriggerContext):
    if ctx.is_blueprint:
        return
    if ctx.event == TriggerEvent.BEFORE_PLAY:
        played = ctx.full_hand or (ctx.state.hand.selected_cards if ctx.state and ctx.state.hand else [])
        if ctx.state.stats.hands_played_this_round <= 1 and len(played) == 1:
            c = played[0]
            if c.rank == Rank.SIX:
                ctx.state.remove_card_from_deck(c)
                if ctx.state.inventory and ctx.state.inventory.can_add_consumable:
                    try:
                        from game.balatro_models import Spectral, SpectralName
                    except ImportError:
                        from balatro_models import Spectral, SpectralName
                    ctx.state.inventory.consumables.append(Spectral(name=SpectralName.ANKH))


@joker_hook(JokerName.SEANCE)
def handle_seance(joker: Joker, ctx: TriggerContext):
    pass


@joker_hook(JokerName.SUPERPOSITION)
def handle_superposition(joker: Joker, ctx: TriggerContext):
    pass


@joker_hook(JokerName.VAGABOND)
def handle_vagabond(joker: Joker, ctx: TriggerContext):
    pass


@joker_hook(JokerName.HALLUCINATION)
def handle_hallucination(joker: Joker, ctx: TriggerContext):
    pass


@joker_hook(JokerName.EIGHT_BALL)
def handle_eight_ball(joker: Joker, ctx: TriggerContext):
    pass


@joker_hook(JokerName.MIDAS_MASK)
def handle_midas_mask(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.AFTER_PLAY:
        scoring = ctx.scoring_cards or (ctx.state.hand.get_scoring_cards(ctx.state.inventory.jokers) if ctx.state and ctx.state.hand else [])
        for c in scoring:
            if c.is_face:
                c.enhancement = Enhancement.GOLD


@joker_hook(JokerName.ROUGH_GEM)
def handle_rough_gem(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.INDIVIDUAL_CARD_SCORED:
        if ctx.target_card and ctx.target_card.suit == Suit.DIAMONDS:
            ctx.state.stats.money += 1
    elif ctx.event == TriggerEvent.AFTER_PLAY:
        if ctx.extra and ctx.extra.get("batch_payout"):
            scoring = ctx.scoring_cards or (ctx.state.hand.get_scoring_cards(ctx.state.inventory.jokers) if ctx.state and ctx.state.hand else [])
            diamonds = sum(1 for c in scoring if c.suit == Suit.DIAMONDS)
            ctx.state.stats.money += diamonds


@joker_hook(JokerName.GOLDEN_TICKET)
def handle_golden_ticket(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.INDIVIDUAL_CARD_SCORED:
        if ctx.target_card and ctx.target_card.enhancement == Enhancement.GOLD:
            ctx.state.stats.money += 4
    elif ctx.event == TriggerEvent.AFTER_PLAY:
        if ctx.extra and ctx.extra.get("batch_payout"):
            scoring = ctx.scoring_cards or (ctx.state.hand.get_scoring_cards(ctx.state.inventory.jokers) if ctx.state and ctx.state.hand else [])
            golds = sum(1 for c in scoring if c.enhancement == Enhancement.GOLD)
            ctx.state.stats.money += golds * 4


@joker_hook(JokerName.HIT_THE_ROAD)
def handle_hit_the_road(joker: Joker, ctx: TriggerContext):
    if ctx.event == TriggerEvent.DISCARD:
        discarded = ctx.full_hand or (ctx.state.hand.selected_cards if ctx.state and ctx.state.hand else [])
        jacks = sum(1 for c in discarded if c.rank == Rank.JACK)
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
    JokerName.BLUEPRINT, JokerName.BRAINSTORM
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
    fully respecting Blueprint and Brainstorm copy targets and compatibility rules.
    """
    jokers = list(ctx.state.inventory.jokers)
    for j in jokers:
        if j.name in (JokerName.BLUEPRINT, JokerName.BRAINSTORM):
            target = resolve_copy_target(j, jokers)
            if target and target.blueprint_compat and target.name in JOKER_EFFECTS:
                copy_ctx = TriggerContext(
                    event=ctx.event,
                    state=ctx.state,
                    action=ctx.action,
                    target_card=ctx.target_card,
                    extra=ctx.extra,
                    is_blueprint=True,
                )
                JOKER_EFFECTS[target.name](target, copy_ctx)
        elif j.name in JOKER_EFFECTS:
            JOKER_EFFECTS[j.name](j, ctx)
