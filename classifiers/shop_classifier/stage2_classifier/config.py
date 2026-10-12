"""
Stage 2 Multi-Head Shop & Item Classifier Configuration
=======================================================
Defines classes, indices, input dimensions, and loss weights for shop items:
- Item Key (152 Jokers + 77 Consumables + unknown = 230 classes)
- Edition (5 classes: None, Foil, Holo, Polychrome, Negative)
- Stickers / Status (Eternal, Perishable, Rental, Debuffed)
"""

import json
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
ALL_KEYS_FILE = CURRENT_DIR / "all_keys.json"

# 1. Item Keys (Jokers + Consumables)
if ALL_KEYS_FILE.exists():
    with open(ALL_KEYS_FILE, "r", encoding="utf-8") as f:
        ITEM_KEYS = json.load(f)
else:
    ITEM_KEYS = ["unknown", "j_joker"]

KEY_TO_IDX = {k: i for i, k in enumerate(ITEM_KEYS)}
IDX_TO_KEY = {i: k for i, k in enumerate(ITEM_KEYS)}

# 2. Editions (5 classes: None, Foil, Holo, Polychrome, Negative)
EDITIONS = ["None", "Foil", "Holo", "Polychrome", "Negative"]
EDITION_TO_IDX = {e: i for i, e in enumerate(EDITIONS)}
IDX_TO_EDITION = {i: e for i, e in enumerate(EDITIONS)}

# Standard Balatro card aspect ratio (71 x 95)
# Crop dimensions for classifier input (width, height)
CROP_WIDTH = 128
CROP_HEIGHT = 192

# Head configuration for MultiHeadItemClassifier
HEAD_CONFIG = {
    "key": len(ITEM_KEYS),
    "edition": len(EDITIONS),     # 5
    "eternal": 1,                 # binary logit (BCE)
    "perishable": 1,              # binary logit (BCE)
    "rental": 1,                  # binary logit (BCE)
    "debuffed": 1,                # binary logit (BCE)
}
