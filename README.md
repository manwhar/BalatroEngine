# BalatroEngine

Computer vision pipeline and gamestate recognition system for **Balatro**, designed to automatically parse and extract gamestates from recorded video gameplay (e.g. YouTube runs by high-level players like Balatro University) to support imitation learning, move prediction, and Monte-Carlo Tree Search (MCTS).

---

## Architecture Overview

1. **Synthetic Dataset Generation Mod (`balatro_mod/`)**: An in-engine Balatro mod (Steamodded / SMODS) that generates perfectly annotated screenshot datasets with pixel-accurate bounding boxes and card metadata.
2. **Card Classifier Pipeline (`classifiers/card_classifier/`)**: A two-stage detection and classification pipeline that detects playing cards in the hand and predicts rank, suit, edition, enhancement, seal, debuff, face-down status, and collab deck skins.
3. **Shop & Joker Dataset Engine (`balatro_mod/src/shop_gen.lua`)**: Generates multi-region training data across the top Joker bar, shop slots, booster packs, and consumable slots, accounting for editions, stickers, debuffs, and YouTube mod variants.
4. **Frame Extraction & ROI Tools (`tools/` & root utilities)**: Geometry, shadow evaluation, and ROI cropping utilities to extract gameplay frames and align detection areas from YouTube videos.

---

## Primary Components & Features

### 1. In-Game Dataset Generator Mod (`balatro_mod/`)
A modular Steamodded mod installed into Balatro to capture synchronized, perfectly labeled CV datasets.

- **Operation Modes**: Configurable from the in-game mod settings menu between **Off**, **Card Classifier**, and **Shop Classifier**.
- **Hand Card Generator (`src/cards_gen.lua`)**: Generates randomized hands with configurable distributions for card ranks, suits, enhancements, editions (Foil/Holo/Poly/Negative), seals, face-down flips, and collab deck skins.
- **Shop & Booster Pack Generator (`src/shop_gen.lua`)**:
  - Cycles between **Main Shop** (2–4 slots) and **Booster Packs** (Buffoon, Arcana, Spectral, Celestial) across normal and Mega sizes.
  - Generates top **Joker Bar** setups (0–9 jokers) with combinations of Eternal, Perishable, Rental stickers, Editions, and Boss Blind debuffs.
  - Generates top **Consumable Slots** (Tarot/Planet/Spectral) with normal and **Perkeo duplicate modes** (supporting negative consumables).
  - Enforces vanilla item pools (Legendary jokers excluded from shop/buffoon packs; *The Soul* and *Black Hole* appropriately allowed in booster packs).
- **In-Engine Bounding Box Calibration (`src/ui.lua`)**: Provides an interactive on-screen editor with hotkeys (`Tab`, `Shift+Tab`, arrow keys, `+`/`-`, `[`/`]`) to align detection ROIs and card bounding boxes in real-time.
- **Automated Capture Engine (`src/capture.lua`)**: Hooks Love2D rendering to compute true bounding boxes from card transformation matrices, saving cropped ROI images alongside YOLO `.txt` labels and structured JSON metadata.

---

### 2. Balatro University Mod (BUMod) Art Compatibility
High-level video data frequently features custom card art from the popular [OceanRamen/BUMod](https://github.com/OceanRamen/BUMod).

- **Historical Texture Integration**: Textures across 2+ years of BUMod commit revisions (e.g. Showman, Scholar, Space Joker, Vampire, Rocket) are extracted into `balatro_mod/assets/bu_variants/`.
- **Dynamic In-Engine Swapping**: During dataset generation, the mod uniformly rolls between vanilla art and historical BUMod sprite revisions, ensuring editions, stickers, and debuff shaders render directly on top of all custom textures.
- **Update Utility (`update_bumod_variants.py`)**: A standalone script that queries the BUMod repository, pulls newly updated Joker sprites into the mod, and regenerates `bu_variants_manifest.lua` for future mod releases.

---

### 3. Two-Stage Card Classifier (`classifiers/card_classifier/`)
A dual-stage neural network architecture optimized for precision on complex card overlaps.

- **Stage 1 (Card Detector)**: A YOLO-based object detector that locates bounding boxes for cards within the player's hand ROI.
- **Stage 2 (Multi-Task Classifier)**: A ResNet/CNN classifier that ingests cropped card images and predicts:
  - **Rank & Suit**: Identifies standard ranks (2–Ace, Stone) and suits, including High Contrast and collab deck variants.
  - **Modifiers**: Identifies card Enhancement (Bonus, Mult, Wild, Glass, Steel, Stone, Gold, Lucky), Edition (Foil, Holo, Polychrome, Negative), and Seal (Gold, Red, Blue, Purple).
  - **Status**: Identifies whether the card is Selected (highlighted upward), Debuffed, or Face-down.
- **End-to-End Pipeline (`pipeline.py`)**: Chains Stage 1 detection and Stage 2 classification into a single inference interface.
- **Automated Training Daemon (`auto_retrain.py`)**: Watches the dataset folder for new captures, prepares training splits, trains both stages, and evaluates checkpoint metrics.

---

### 4. Frame Scraping & Computer Vision Utilities (`tools/`)
- **`tools/scrape_frame.py`**: Extracts frames from local gameplay video files at specified timestamps.
- **`tools/adjust_rois.py` & `tools/grab_rois.py`**: Interactive visual tools to fine-tune region-of-interest coordinates across different video resolutions.
- **`tools/assign_test_images.py`**: Tooling to curate test evaluation splits from actual YouTube gameplay footage.
- **`frame.py`, `geometry.py`, `matcher.py`**: Foundational geometry and template matching helpers for UI alignment and button detection.

---

## Usage Quickstart

### Updating BUMod Textures
```powershell
python update_bumod_variants.py
```

### Training Card Classifier
```powershell
# Prepare dataset splits from raw mod captures
python classifiers/card_classifier/prepare_dataset.py

# Run Stage 1 detector and Stage 2 multi-task classifier training
python classifiers/card_classifier/train.py

# Or launch continuous background retrainer:
python auto_retrain.py
```

### Running Pipeline on Video Frames
```powershell
python classifiers/card_classifier/test_video_frame.py --input path/to/frame.png
```
