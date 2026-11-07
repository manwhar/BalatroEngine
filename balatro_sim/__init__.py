"""Balatro Deterministic PRNG Simulation Engine.

A native Python implementation of Balatro's exact deterministic PRNG pipeline,
game-state tracking, and event roll solvers modeled after Immolate / Balatro source.
"""

from balatro_sim.api import BalatroRun, validate_seed
from balatro_sim.events import (
    CertificateResult,
    EventResolver,
    LuckyHitResult,
    RebateEvaluation,
    evaluate_lucky_hit,
    evaluate_mail_in_rebate_discard,
    get_mail_in_rebate_target,
    predict_certificate,
)
from balatro_sim.rng import (
    BalatroRNG,
    PRNGChannel,
    Tw223,
    double_to_u64,
    predict_seed,
    pseudohash,
    pseudoseed,
    truncate13,
    u64_to_double,
)
from balatro_sim.state import (
    Card,
    Deck,
    Edition,
    Enhancement,
    GameState,
    JokerInstance,
    Rank,
    Seal,
    Stake,
    Suit,
)

__all__ = [
    # Top-level API
    "BalatroRun",
    "validate_seed",
    # RNG Engine
    "BalatroRNG",
    "PRNGChannel",
    "Tw223",
    "pseudohash",
    "pseudoseed",
    "predict_seed",
    "truncate13",
    "double_to_u64",
    "u64_to_double",
    # Game State & Entities
    "Card",
    "Deck",
    "GameState",
    "JokerInstance",
    "Suit",
    "Rank",
    "Enhancement",
    "Edition",
    "Seal",
    "Stake",
    # Events & Solvers
    "CertificateResult",
    "LuckyHitResult",
    "RebateEvaluation",
    "EventResolver",
    "predict_certificate",
    "evaluate_lucky_hit",
    "get_mail_in_rebate_target",
    "evaluate_mail_in_rebate_discard",
]

