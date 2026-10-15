"""
CV Model Query Stubs:
Acts as the interface for computer vision observations that cannot be determined
solely from button clicks or deterministic rules.

RUNNING LIST OF UNIMPLEMENTED CV CALLS (Strictly Necessary):
1. cv_detect_hand_cards(frame) -> Reads full list of cards in hand on deal/return
2. cv_detect_shop_cards(frame) -> Reads items in the shop after reroll or phase transition
3. cv_detect_joker_text_value(frame, joker, pattern) -> OCRs dynamic tooltip/badge numbers when hovered
4. cv_detect_target_suit(frame, joker) -> Reads target suit (Ancient Joker, Castle)
5. cv_detect_target_rank(frame, joker) -> Reads target rank (Mail-In Rebate, The Idol)
6. cv_detect_target_poker_hand(frame, joker) -> Reads target poker hand (To Do List)
7. cv_detect_lucky_card_triggered(frame) -> Observes if Lucky card rolled mult/money during scoring
8. cv_detect_glass_card_shattered(frame) -> Observes if Glass card shattered (Glass Joker)
9. cv_detect_card_destroyed(frame) -> Identifies if face card was destroyed during play/discard (Canio)
10. cv_detect_boss_blind(frame) -> OCRs boss blind name from Blind Select or Run Info menu
"""

from typing import Any
try:
    from game.balatro_models import (
        BlindName,
        HandCard,
        Joker,
        PokerHand,
        Rank,
        Suit,
    )
except ImportError:
    from balatro_models import (
        BlindName,
        HandCard,
        Joker,
        PokerHand,
        Rank,
        Suit,
    )


def cv_detect_hand_cards(frame: Any = None) -> list[HandCard]:
    """Evaluates the player's card hand ROI."""
    return []


def cv_detect_shop_cards(frame: Any = None) -> list[Any]:
    """Evaluates the shop ROI for cards and packs."""
    return []


def cv_detect_joker_text_value(frame: Any, joker: Joker, pattern: str = "") -> int | float | None:
    """OCRs numeric values from a joker's badge or hovered description."""
    return None


def cv_detect_target_suit(frame: Any = None, joker: Joker | None = None) -> Suit:
    """Reads the active target suit for Ancient Joker or Castle."""
    return Suit.HEARTS


def cv_detect_target_rank(frame: Any = None, joker: Joker | None = None) -> Rank:
    """Reads the active target rank for Mail-In Rebate or The Idol."""
    return Rank.ACE


def cv_detect_target_poker_hand(frame: Any = None, joker: Joker | None = None) -> PokerHand:
    """Reads the active target poker hand for To Do List."""
    return PokerHand.PAIR


def cv_detect_lucky_card_triggered(frame: Any = None) -> tuple[bool, bool]:
    """Detects if a Lucky Card triggered +20 Mult and/or +$20 during scoring."""
    return False, False


def cv_detect_glass_card_shattered(frame: Any = None) -> bool:
    """Detects if a Glass Card shattered on play."""
    return False


def cv_detect_card_destroyed(frame: Any = None) -> bool:
    """Detects if a face card was destroyed during play/discard/event."""
    return False


def cv_detect_boss_blind(frame: Any = None) -> BlindName | None:
    """Reads upcoming Boss Blind from Run Info screen or Blind Select screen."""
    return None
