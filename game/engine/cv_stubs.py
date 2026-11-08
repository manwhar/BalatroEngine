"""
CV Model Query Stubs:
Acts as the interface for computer vision observations of static, visible on-screen UI elements.
All non-visible state, RNG rolls, card creation, and rotating targets are deterministically
modeled via `game.engine.rng_sim`.

STATIC VISIBLE CV QUERIES:
1. cv_detect_hand_cards(frame) -> Reads cards in player hand (rank, suit, enhancements, seals, editions)
2. cv_detect_shop_cards(frame) -> Reads items in the shop (jokers, packs, vouchers, consumables)
3. cv_detect_boss_blind(frame) -> Reads boss blind name from HUD / Blind Select screen
"""

from typing import Any

try:
    from game.balatro_models import BlindName, HandCard
except ImportError:
    from balatro_models import BlindName, HandCard


def cv_detect_hand_cards(frame: Any = None) -> list[HandCard]:
    """Evaluates the player's card hand ROI."""
    return []


def cv_detect_shop_cards(frame: Any = None) -> list[Any]:
    """Evaluates the shop ROI for cards and packs."""
    return []


def cv_detect_boss_blind(frame: Any = None) -> BlindName | None:
    """Reads upcoming Boss Blind from Run Info screen or Blind Select screen."""
    return None
