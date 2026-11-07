from balatro_models import *
from gamestate import GameState

J = JokerName


def discard_castle(joker: Joker, game_state: GameState, discarded):

    joker.state["chips"]


def green_joker_on_hand_played(joker: Joker, game_state: GameState):
    joker.state["mult"] = joker.state.get("mult", 0) + 1


HAND_PLAYED_HANDLERS = {JokerName.GREEN_JOKER: green_joker_on_hand_played}

# def play_hand(hand: PokerHand, scoring_cards: list[PlayingCard], inventory: Inventory):
#     """Updates all game values as a result of playing a hand."""
#     jokers = inventory.jokers

#     def j(joker: JokerName):
#         return joker in jokers

#     if j(J.LOYALTY_CARD):
