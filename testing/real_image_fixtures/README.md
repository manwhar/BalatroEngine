# Real-Image CV Test Fixtures

Drop real Balatro gameplay screenshots (full game window / 16:9 1080p frames recommended; PNG or JPG) into the category folders below and record expected detections in `ground_truth.json`.

## Folders & What Screenshots to Collect

| Folder | CV Function Tested | What Screenshots to Capture |
|---|---|---|
| `hand/` | `cv_detect_hand_cards` | Active blind frames (some with 0 selected cards, some with 1–5 raised/selected cards, stone cards, debuffed cards, editions/enhancements/seals, face-down cards; optional Arcana/Spectral pack frames with visible hand). |
| `shop/` | `cv_detect_shop_cards` | Main shop frames before opening any pack (include Jokers, Tarots, Planets, Vouchers, Booster Packs on the shelves, plus top Joker bar & Consumables if populated). |
| `pack/` | `cv_detect_pack_cards` | Open booster pack frames covering all 5 pack types: Arcana, Celestial, Spectral, Buffoon, and Standard (PlayingCards with rank/suit/enhancements/seals). |
| `joker_bar/` | `cv_detect_joker_bar` | Top Joker bar frames across different screens (`select`, `blind`, cash-out screen with `round_won: true`, `shop`, `pack`), including empty bar, single/multiple Jokers, editions (`foil`, `holographic`, `polychrome`, `negative`), and stickers (`eternal`, `perishable`, `rental`, `debuffed`). |

## Annotating `ground_truth.json`

Add entries to the `"hand"`, `"shop"`, `"pack"`, and `"joker_bar"` arrays in `ground_truth.json` following `schema_examples_reference_only` (left-to-right order). Only fields you include in each expected item dict are asserted.

## Running the Suite

```powershell
python -m unittest testing/test_cv_real_images.py -v
```
