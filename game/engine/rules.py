from .context import TriggerContext, TriggerEvent
from .jokers import dispatch_jokers
from .poker_hands import evaluate_poker_hand, get_scoring_cards
try:
    from game.balatro_models import Action, ActionType, Location, BlindName
except ImportError:
    from balatro_models import Action, ActionType, Location, BlindName


def on_action(state, action: Action):
    """Executes deterministic game engine state transitions and triggers."""
    match action.type:
        case ActionType.SELECT_CARD:
            if action.target_idx is not None:
                state.hand.select(action.target_idx)

        case ActionType.PLAY_HAND:
            state.stats.hands_left = max(0, state.stats.hands_left - 1)
            state.stats.hands_played_this_round += 1
            
            played_cards = [c.card for c in state.hand.cards if c.selected]
            jokers = state.inventory.jokers if state.inventory else None
            scoring_cards = get_scoring_cards(played_cards, jokers)
            poker_hand = evaluate_poker_hand(played_cards, jokers)

            # 1. Before play
            ctx = TriggerContext(
                event=TriggerEvent.BEFORE_PLAY,
                state=state,
                action=action,
                scoring_cards=scoring_cards,
                full_hand=played_cards,
                poker_hand=poker_hand,
            )
            dispatch_jokers(ctx)
            
            # 2. Individual card scored (for each active scoring card)
            for card in scoring_cards:
                ctx_ind = TriggerContext(
                    event=TriggerEvent.INDIVIDUAL_CARD_SCORED,
                    state=state,
                    action=action,
                    target_card=card,
                    scoring_cards=scoring_cards,
                    full_hand=played_cards,
                    poker_hand=poker_hand,
                )
                dispatch_jokers(ctx_ind)

            # 3. After play
            ctx_after = TriggerContext(
                event=TriggerEvent.AFTER_PLAY,
                state=state,
                action=action,
                scoring_cards=scoring_cards,
                full_hand=played_cards,
                poker_hand=poker_hand,
            )
            dispatch_jokers(ctx_after)
            
            state.hand.cards = [c for c in state.hand.cards if not c.selected]
            for card in played_cards:
                if not getattr(card, "destroyed", False):
                    state.played_cards_this_round.append(card)

        case ActionType.DISCARD:
            state.stats.discards_left = max(0, state.stats.discards_left - 1)
            state.stats.discards_used_this_round += 1
            
            discarded_cards = [c.card for c in state.hand.cards if c.selected]
            ctx = TriggerContext(
                event=TriggerEvent.DISCARD,
                state=state,
                action=action,
                full_hand=discarded_cards,
            )
            dispatch_jokers(ctx)
            
            state.hand.cards = [c for c in state.hand.cards if not c.selected]
            for card in discarded_cards:
                if not getattr(card, "destroyed", False):
                    state.discard_pile.append(card)

        case ActionType.REROLL_SHOP:
            state.stats.money -= state.stats.reroll_cost
            state.stats.reroll_cost += 1
            
            ctx = TriggerContext(event=TriggerEvent.REROLL_SHOP, state=state, action=action)
            dispatch_jokers(ctx)

        case ActionType.BUY_CARD:
            if state.shop and action.target_idx is not None and 0 <= action.target_idx < len(state.shop.cards):
                card = state.shop.cards.pop(action.target_idx)
                cost = card.get_cost()
                state.stats.money -= cost
                if card.type.value == "joker":
                    state.inventory.jokers.append(card)
                elif card.type.value in ("tarot", "planet", "spectral"):
                    state.inventory.consumables.append(card)
                ctx = TriggerContext(event=TriggerEvent.CARD_BOUGHT, state=state, action=action, target_card=card)
                dispatch_jokers(ctx)

        case ActionType.SELL_CARD:
            sold_card = None
            if action.target_idx is not None:
                if action.target_idx < len(state.inventory.jokers):
                    sold_card = state.inventory.jokers.pop(action.target_idx)
                else:
                    c_idx = action.target_idx - len(state.inventory.jokers)
                    if 0 <= c_idx < len(state.inventory.consumables):
                        sold_card = state.inventory.consumables.pop(c_idx)
            if sold_card:
                state.stats.money += sold_card.sell_value
                ctx = TriggerContext(event=TriggerEvent.CARD_SOLD, state=state, action=action, target_card=sold_card)
                dispatch_jokers(ctx)

        case ActionType.CASH_OUT:
            ctx = TriggerContext(event=TriggerEvent.END_OF_ROUND, state=state, action=action)
            dispatch_jokers(ctx)
            state.location = Location.SHOP
            state.stats.reroll_cost = 5

        case ActionType.SELECT_BLIND:
            state.location = Location.BLIND
            state.stats.hands_played_this_round = 0
            state.stats.discards_used_this_round = 0
            state.stats.hands_left = 4
            state.stats.discards_left = 3
            state.discard_pile.clear()
            state.played_cards_this_round.clear()
            
            ctx = TriggerContext(event=TriggerEvent.SETTING_BLIND, state=state, action=action)
            dispatch_jokers(ctx)

        case ActionType.REORDER_JOKERS:
            if action.joker_order:
                current_jokers = state.inventory.jokers
                if len(action.joker_order) == len(current_jokers):
                    state.inventory.jokers = [current_jokers[i] for i in action.joker_order]
