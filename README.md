# BalatroEngine

Computer vision pipeline and gamestate recognition system for **Balatro**, designed to automatically parse and extract gamestates and player actions from recorded video gameplay (e.g. YouTube runs by high-level players like Balatro University) to support imitation learning, move prediction, and Monte-Carlo Tree Search (MCTS).

---

## Overview

1. **Template Matching Action Classifier (`main.py`, `matcher.py`, `eval_template.py`)**: A high-speed UI state detector that locates interactive elements (buttons, blinds, shops, booster packs) and evaluates user clicks via physical shadow displacement.
2. **Synthetic Dataset Generation Mod (`balatro_mod/`)**: An in-engine Balatro mod (Steamodded / SMODS) that generates perfectly annotated screenshot datasets with pixel-accurate bounding boxes and card metadata.
3. **Card Classifier Pipeline (`classifiers/card_classifier/`)**: A two-stage detection and classification pipeline that detects playing cards in the hand and predicts rank, suit, edition, enhancement, seal, debuff, face-down status, and collab deck skins.
4. **Shop & Joker Dataset Engine (`balatro_mod/src/shop_gen.lua`)**: Generates multi-region training data across the top Joker bar, shop slots, booster packs, and consumable slots, accounting for editions, stickers, debuffs, and YouTube mod variants.
5. **Frame Extraction & Calibration Tools (`tools/` & root utilities)**: Geometry, shadow evaluation, and ROI cropping utilities to extract gameplay frames and align detection areas across video resolutions.

---

## Main Stuff

### 1. Template Matching & Action Classifier
The primary pipeline for extracting player decisions directly from recorded gameplay videos.

- **High-Speed Template Matching (`main.py`, `matcher.py`)**: Scans video frames within targeted Region of Interest (ROI) bounding boxes using OpenCV template matching with normalized cross-correlation and masking.
- **Button Click Detection via Shadow Analysis (`shadow_evaluator.py`, `eval_template.py`)**: Balatro buttons render a distinct drop shadow beneath them when resting, which disappears when clicked/depressed. A Sobel horizontal edge filter and median gradient analyzer detect the presence or collapse of this shadow to register whether a button was actively clicked.
- **Interactive Calibration & Debug Plotter (`debug_plotter.py`)**: An interactive OpenCV/matplotlib interface that renders match confidence heatmaps, ROI boundaries, and shadow analysis slices in real time, allowing hotkey-based alignment of shadow offsets (`shadow_dy`, `shadow_dw`, `shadow_dh`).
- **Batch Multiprocessing Engine (`main.py`)**: Parallelizes video decoding and frame evaluation using Python `concurrent.futures.ProcessPoolExecutor` with configurable frame skipping (`skip_frames`) for high throughput across multi-hour videos.
- **Resolution-Independent Geometry (`geometry.py`)**: Normalized and pixel-level bounding box primitives (`NormalizedROI`, `PixelROI`) that scale UI coordinates seamlessly across 720p, 1080p, and 1440p video sources.

---

### 2. In-Game Dataset Generator Mod (`balatro_mod/`)
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

### 3. BUMod Art Integration
Balatro University's gameplay features custom card art from the [OceanRamen/BUMod](https://github.com/OceanRamen/BUMod).

- **Historical Texture Integration**: Textures across 2+ years of BUMod commit revisions (e.g. Showman, Scholar, Space Joker, Vampire, Rocket) are extracted into `balatro_mod/assets/bu_variants/`.
- **Dynamic In-Engine Swapping**: During dataset generation, the mod uniformly rolls between vanilla art and historical BUMod sprite revisions, ensuring editions, stickers, and debuff shaders render directly on top of all custom textures.
- **Update Utility (`update_bumod_variants.py`)**: A standalone script that queries the BUMod repository, pulls newly updated Joker sprites into the mod, and regenerates `bu_variants_manifest.lua` for future mod releases.

---

### 4. Card Classifier (`classifiers/card_classifier/`)
A dual-stage neural network architecture optimized for precision on complex card overlaps.

- **Stage 1 (Card Detector)**: A YOLO-based object detector that locates bounding boxes for cards within the player's hand ROI.
- **Stage 2 (Multi-Task Classifier)**: A ResNet/CNN classifier that ingests cropped card images and predicts:
  - **Rank & Suit**: Identifies standard ranks (2–Ace, Stone) and suits, including High Contrast and collab deck variants.
  - **Modifiers**: Identifies card Enhancement (Bonus, Mult, Wild, Glass, Steel, Stone, Gold, Lucky), Edition (Foil, Holo, Polychrome, Negative), and Seal (Gold, Red, Blue, Purple).
  - **Status**: Identifies whether the card is Selected (highlighted upward), Debuffed, or Face-down.
- **End-to-End Pipeline (`pipeline.py`)**: Chains Stage 1 detection and Stage 2 classification into a single inference interface.
- **Automated Training Daemon (`auto_retrain.py`)**: Watches the dataset folder for new captures, prepares training splits, trains both stages, and evaluates checkpoint metrics.

---

### 5. Video Scraping & Auxiliary CV Utilities (`tools/`)
- **`tools/scrape_frame.py`**: Extracts frames from local gameplay video files at specified timestamps.
- **`tools/adjust_rois.py` & `tools/grab_rois.py`**: Interactive visual tools to fine-tune region-of-interest coordinates across different video resolutions.
- **`tools/assign_test_images.py`**: Tooling to curate test evaluation splits from actual YouTube gameplay footage.
- **`tools/crop_transparent_borders.py`**: Cleans and normalizes alpha channels on raw UI template crops.

---

## Usage Quickstart

### Running the Template Matching Pipeline on Gameplay Video
```powershell
# Run the action classifier over an input gameplay video
python main.py

# Launch interactive calibration / debug plotting mode for template ROIs and shadows
python main.py --setup_mode --debug_plot
```

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
python auto_retrain.py --model card
```

### Training Shop & Item Classifier
```powershell
# Sync latest shop dataset from mod and inspect statistics
python classifiers/shop_classifier/prepare_dataset.py

# Run Stage 1 detector (YOLO) and Stage 2 multi-head item classifier
python classifiers/shop_classifier/train.py --sync --stage all

# Or launch continuous background retrainer (monitors dataset_shop and auto-trains):
python auto_retrain.py --model shop --target 10000
```

### Running Pipeline on Video Frames
```powershell
python classifiers/card_classifier/test_video_frame.py --input path/to/frame.png
```
