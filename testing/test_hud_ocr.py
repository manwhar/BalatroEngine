"""Unit tests for classifiers.hud_ocr (synthetic generator, CRNN+CTC model, training, pipeline)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import tempfile
import unittest

import cv2
import torch

from classifiers.hud_ocr.config import (
    CHAR_TO_IDX,
    FIELD_ALLOWED_CHARS,
    HUD_ROI_FIELDS,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    NUM_CLASSES,
)
from classifiers.hud_ocr.dataset import ModHudOcrDataset, SyntheticHudOcrDataset, collate_ocr_batch
from classifiers.hud_ocr.model import HudOcrCRNN, compute_cer, ctc_greedy_decode
from classifiers.hud_ocr.pipeline import HudOcrPipeline, parse_hud_value
from classifiers.hud_ocr.synth import (
    BalatroHudSynthesizer,
    corrupt_hud_crop_to_empty,
    resolve_font_path,
)
from classifiers.hud_ocr.train import train_ocr


class TestHudOcr(unittest.TestCase):
    def test_synthesizer_uses_m6x11plus_and_renders_all_fields(self) -> None:
        font_path = resolve_font_path()
        self.assertIsNotNone(font_path)
        self.assertEqual(font_path.name, "m6x11plus.ttf")

        synth = BalatroHudSynthesizer(font_path=font_path)
        rng = random.Random(123)
        for field_name in HUD_ROI_FIELDS:
            crop, text, chosen = synth.render_sample(rng, field_name=field_name, augment=False)
            self.assertEqual(chosen, field_name)
            self.assertEqual(crop.shape, (INPUT_HEIGHT, INPUT_WIDTH, 3))
            self.assertEqual(str(crop.dtype), "uint8")
            self.assertTrue(len(text) > 0)
            allowed = set(FIELD_ALLOWED_CHARS[field_name])
            self.assertTrue(all(ch in allowed for ch in text), f"Unexpected char in {text!r} for {field_name}")

        # Augmented renders include both non-empty and empty ("") negative examples
        seen_empty = False
        for _ in range(120):
            crop, text, _ = synth.render_sample(rng, augment=True)
            self.assertEqual(crop.shape, (INPUT_HEIGHT, INPUT_WIDTH, 3))
            if text == "":
                seen_empty = True
                break
        self.assertTrue(seen_empty)

    def test_model_forward_decode_and_parser(self) -> None:
        model = HudOcrCRNN()
        x = torch.randn(4, 3, INPUT_HEIGHT, INPUT_WIDTH)
        log_probs = model(x)
        self.assertEqual(tuple(log_probs.shape), (4, 48, NUM_CLASSES))

        decoded = ctc_greedy_decode(log_probs)
        self.assertEqual(len(decoded), 4)

        # Verify per-field allowed-character masking prevents non-digit tokens on digit-only fields
        forced = torch.full((1, 10, NUM_CLASSES), -20.0)
        forced[0, 0:2, CHAR_TO_IDX["$"]] = 0.0
        forced[0, 2:4, CHAR_TO_IDX["4"]] = -1.0
        forced[0, 4:, 0] = 0.0  # blank
        masked_out = ctc_greedy_decode(forced, allowed_chars_per_sample=[FIELD_ALLOWED_CHARS["hud_hands"]])
        self.assertEqual(masked_out[0][0], "4")
        self.assertAlmostEqual(compute_cer(["123", "$5"], ["123", "$5"]), 0.0)
        self.assertGreater(compute_cer(["124", "$5"], ["123", "$5"]), 0.0)

        self.assertEqual(parse_hud_value("hud_round_score", "12,450"), 12450)
        self.assertEqual(parse_hud_value("hud_round_score", "1.25e+12"), 1_250_000_000_000)
        self.assertEqual(parse_hud_value("hud_money", "-$14"), -14)
        self.assertEqual(parse_hud_value("hud_ante", "3/8"), 3)
        self.assertEqual(parse_hud_value("hud_reroll_cost", "$7"), 7)
        self.assertIsNone(parse_hud_value("hud_money", ""))

    def test_dataset_collate_train_and_pipeline_smoke(self) -> None:
        ds = SyntheticHudOcrDataset(num_samples=8, seed=7, fixed=True, augment=False)
        imgs, targets, target_lengths, texts = collate_ocr_batch([ds[i] for i in range(4)])
        self.assertEqual(tuple(imgs.shape), (4, 3, INPUT_HEIGHT, INPUT_WIDTH))
        self.assertEqual(int(target_lengths.sum().item()), int(targets.numel()))
        self.assertEqual(len(texts), 4)

        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = Path(tmpdir) / "hud_ocr_test.pt"
            args = argparse.Namespace(
                epochs=1,
                batch_size=16,
                train_samples=32,
                val_samples=16,
                lr=1e-3,
                weight_decay=1e-4,
                rnn_hidden=64,
                rnn_layers=1,
                dropout=0.1,
                num_workers=0,
                seed=11,
                device="cpu",
                no_amp=True,
                font_path=None,
                dataset_dir=None,
                synth_only=True,
                synth_samples=None,
                manifest_dir=None,
                val_real_only=False,
                weights_out=str(weights_path),
                runs_dir=str(Path(tmpdir) / "runs"),
            )
            saved_path = train_ocr(args)
            self.assertTrue(saved_path.exists())

            pipe = HudOcrPipeline(weights_path=saved_path, device="cpu", batch_size=8)
            synth = BalatroHudSynthesizer()
            rng = random.Random(99)
            crops = [synth.render_sample(rng, field_name="hud_money", augment=False)[0] for _ in range(3)]
            results = pipe.predict_crops(crops, field_names=["hud_money"] * 3)
            self.assertEqual(len(results), 3)
            for res in results:
                self.assertEqual(res.field_name, "hud_money")
                self.assertGreaterEqual(res.confidence, 0.0)

    def test_mod_dataset_and_mixed_empty_training(self) -> None:
        synth = BalatroHudSynthesizer()
        rng = random.Random(2026)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "dataset_hud_ocr"
            (root / "images").mkdir(parents=True, exist_ok=True)
            (root / "labels").mkdir(parents=True, exist_ok=True)

            # Create 6 synthetic mod-formatted frames alternating blind & shop (with empty fields)
            for frame_idx in range(6):
                phase = "blind" if frame_idx % 2 == 0 else "shop"
                fields_meta = {
                    "hud_round_score": "0" if frame_idx == 0 else "12,450",
                    "hud_money": "-5" if frame_idx == 1 else "42",
                    "hud_hands": "4",
                    "hud_discards": "3",
                    "hud_ante": "7",
                    "hud_round": "12",
                    "hud_reroll_cost": "5" if phase == "shop" else "",
                    "hud_blind_target": "300" if phase == "blind" else "",
                }
                for fname, val in fields_meta.items():
                    img_rgb, _, _ = synth.render_sample(
                        rng,
                        field_name=fname,
                        text_override=val if val != "" else None,
                        augment=False,
                    )
                    cv2.imwrite(
                        str(root / "images" / f"frame_{frame_idx:05d}_{fname}.png"),
                        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                    )
                with (root / "labels" / f"frame_{frame_idx:05d}.json").open("w", encoding="utf-8") as f:
                    json.dump({"frame_index": frame_idx, "phase": phase, "fields": fields_meta}, f)

            train_ds = ModHudOcrDataset(root, split="train", val_fraction=0.2, seed=42, augment=True)
            val_ds = ModHudOcrDataset(root, split="val", val_fraction=0.2, seed=42, augment=False)
            self.assertEqual(len(train_ds) + len(val_ds), 6 * len(HUD_ROI_FIELDS))
            self.assertTrue(any(val_ds[i][2] == "" for i in range(len(val_ds))))

            weights_path = Path(tmpdir) / "hud_ocr_mod_test.pt"
            args = argparse.Namespace(
                epochs=1,
                batch_size=16,
                train_samples=32,
                val_samples=16,
                lr=1e-3,
                weight_decay=1e-4,
                rnn_hidden=64,
                rnn_layers=1,
                dropout=0.1,
                num_workers=0,
                seed=42,
                device="cpu",
                no_amp=True,
                font_path=None,
                dataset_dir=str(root),
                synth_only=False,
                synth_samples=8,
                manifest_dir=None,
                val_real_only=False,
                weights_out=str(weights_path),
                runs_dir=str(Path(tmpdir) / "runs"),
            )
            saved_path = train_ocr(args)
            self.assertTrue(saved_path.exists())

    def test_corrupt_hud_crop_to_empty_and_dataset_negative_augs(self) -> None:
        synth = BalatroHudSynthesizer()
        rng = random.Random(314)
        crop, text, _ = synth.render_sample(
            rng, field_name="hud_money", text_override="42", augment=False
        )
        self.assertEqual(text, "42")
        for seed in range(25):
            corrupted = corrupt_hud_crop_to_empty(crop, random.Random(seed))
            self.assertEqual(corrupted.shape, (INPUT_HEIGHT, INPUT_WIDTH, 3))
            self.assertEqual(str(corrupted.dtype), "uint8")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "dataset_hud_ocr"
            (root / "images").mkdir(parents=True, exist_ok=True)
            (root / "labels").mkdir(parents=True, exist_ok=True)
            for idx in range(2):
                cv2.imwrite(
                    str(root / "images" / f"frame_{idx:05d}_hud_money.png"),
                    cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                )
                with (root / "labels" / f"frame_{idx:05d}.json").open("w", encoding="utf-8") as f:
                    json.dump({"frame_index": idx, "phase": "shop", "fields": {"hud_money": "42"}}, f)

            ds = ModHudOcrDataset(root, split="all", augment=True)
            seen_texts = {ds[0][2] for _ in range(150)}
            self.assertIn("42", seen_texts)
            self.assertIn("", seen_texts)


if __name__ == "__main__":
    unittest.main()
