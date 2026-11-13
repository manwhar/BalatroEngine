"""Real-image integration test suite scaffolding for all CV detectors.

Evaluates:
  - `cv_detect_hand_cards`
  - `cv_detect_shop_cards`
  - `cv_detect_pack_cards`
  - `cv_detect_joker_bar`

Screenshots and expected outputs are user-supplied in:
  testing/real_image_fixtures/ground_truth.json
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path
from typing import Any

import cv2

import game.balatro_models as bm
import game.engine.cv_stubs as cv_stubs
import game.gamestate as gs


FIXTURES_DIR = Path(__file__).resolve().parent / "real_image_fixtures"
GROUND_TRUTH_PATH = FIXTURES_DIR / "ground_truth.json"

REQUIRED_SECTIONS = ("hand", "shop", "pack", "joker_bar")


def _norm(value: Any) -> str:
    """Normalizes enum names and human labels for flexible ground-truth comparison."""
    text = value.value if hasattr(value, "value") else str(value)
    return re.sub(r"[^a-z0-9]", "", text.strip().lower())


def _load_ground_truth() -> dict[str, Any]:
    with GROUND_TRUTH_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_frame(rel_path: str) -> Any:
    img_path = FIXTURES_DIR / rel_path
    if not img_path.exists():
        raise FileNotFoundError(
            f"Ground-truth image missing: '{img_path}'. Place the screenshot file there."
        )
    frame = cv2.imread(str(img_path))
    if frame is None:
        raise ValueError(f"OpenCV failed to decode image '{img_path}'.")
    return frame


def _location_from_label(label: str | None, default: bm.Location) -> bm.Location:
    if not label:
        return default
    key = _norm(label)
    for loc in bm.Location:
        if _norm(loc.value) == key or _norm(loc.name) == key:
            return loc
    raise ValueError(f"Unknown Location '{label}' in ground_truth.json")


class TestRealImageCVFixtures(unittest.TestCase):
    """Validates CV stubs against user-provided screenshots and JSON ground truth."""

    def _assert_hand_card_matches(
        self, actual: bm.HandCard, expected: dict[str, Any], ctx: str
    ) -> None:
        if "rank" in expected:
            self.assertEqual(_norm(actual.card.rank), _norm(expected["rank"]), msg=f"{ctx} rank")
        if "suit" in expected:
            self.assertEqual(_norm(actual.card.suit), _norm(expected["suit"]), msg=f"{ctx} suit")
        if "enhancement" in expected:
            self.assertEqual(
                _norm(actual.card.enhancement), _norm(expected["enhancement"]), msg=f"{ctx} enhancement"
            )
        if "edition" in expected:
            self.assertEqual(_norm(actual.card.edition), _norm(expected["edition"]), msg=f"{ctx} edition")
        if "seal" in expected:
            self.assertEqual(_norm(actual.card.seal), _norm(expected["seal"]), msg=f"{ctx} seal")
        if "selected" in expected:
            self.assertEqual(bool(actual.selected), bool(expected["selected"]), msg=f"{ctx} selected")
        if "debuffed" in expected:
            self.assertEqual(
                bool(actual.card.debuffed), bool(expected["debuffed"]), msg=f"{ctx} debuffed"
            )
        if "facing_back" in expected:
            self.assertEqual(
                bool(actual.facing_back), bool(expected["facing_back"]), msg=f"{ctx} facing_back"
            )

    def _assert_shop_like_item_matches(
        self, actual: Any, expected: dict[str, Any], ctx: str
    ) -> None:
        if "type" in expected:
            self.assertEqual(_norm(type(actual).__name__), _norm(expected["type"]), msg=f"{ctx} type")
        if "name" in expected:
            actual_name = getattr(actual, "name", None)
            self.assertIsNotNone(actual_name, msg=f"{ctx} missing .name on {type(actual).__name__}")
            self.assertEqual(_norm(actual_name), _norm(expected["name"]), msg=f"{ctx} name")
        if "edition" in expected:
            self.assertEqual(
                _norm(getattr(actual, "edition", bm.Edition.NONE)),
                _norm(expected["edition"]),
                msg=f"{ctx} edition",
            )
        for bool_field in ("eternal", "perishable", "rental", "debuffed"):
            if bool_field in expected:
                self.assertEqual(
                    bool(getattr(actual, bool_field, False)),
                    bool(expected[bool_field]),
                    msg=f"{ctx} {bool_field}",
                )
        if "booster_type" in expected:
            self.assertEqual(
                _norm(getattr(actual, "booster_type", "")),
                _norm(expected["booster_type"]),
                msg=f"{ctx} booster_type",
            )
        if "size" in expected:
            self.assertEqual(
                _norm(getattr(actual, "size", "")),
                _norm(expected["size"]),
                msg=f"{ctx} size",
            )
        if "rank" in expected:
            self.assertEqual(
                _norm(getattr(actual, "rank", bm.Rank.NONE)),
                _norm(expected["rank"]),
                msg=f"{ctx} rank",
            )
        if "suit" in expected:
            self.assertEqual(
                _norm(getattr(actual, "suit", bm.Suit.NONE)),
                _norm(expected["suit"]),
                msg=f"{ctx} suit",
            )
        if "enhancement" in expected:
            self.assertEqual(
                _norm(getattr(actual, "enhancement", bm.Enhancement.NONE)),
                _norm(expected["enhancement"]),
                msg=f"{ctx} enhancement",
            )
        if "seal" in expected:
            self.assertEqual(
                _norm(getattr(actual, "seal", bm.Seal.NONE)),
                _norm(expected["seal"]),
                msg=f"{ctx} seal",
            )

    def _assert_item_sequence(
        self,
        actual_items: list[Any],
        expected_items: list[dict[str, Any]],
        matcher: Any,
        ctx: str,
    ) -> None:
        self.assertEqual(
            len(actual_items),
            len(expected_items),
            msg=f"{ctx}: expected {len(expected_items)} items, detected {len(actual_items)}: {actual_items}",
        )
        for idx, (act_item, exp_item) in enumerate(zip(actual_items, expected_items)):
            matcher(act_item, exp_item, f"{ctx}[{idx}]")

    def test_ground_truth_schema_is_valid(self):
        """Always verifies ground_truth.json structure and fixture directories exist."""
        self.assertTrue(GROUND_TRUTH_PATH.exists(), f"Missing {GROUND_TRUTH_PATH}")
        data = _load_ground_truth()
        for section in REQUIRED_SECTIONS:
            self.assertTrue((FIXTURES_DIR / section).is_dir(), f"Missing folder {FIXTURES_DIR / section}")
            self.assertIn(section, data, f"ground_truth.json missing '{section}' list")
            self.assertIsInstance(data[section], list, f"'{section}' must be a list")

    def test_cv_detect_hand_cards_real_images(self):
        cases = _load_ground_truth().get("hand", [])
        if not cases:
            self.skipTest(
                "No real-image fixtures configured in testing/real_image_fixtures/ground_truth.json['hand']."
            )

        for case in cases:
            rel_path = case["image"]
            with self.subTest(image=rel_path):
                frame = _load_frame(rel_path)
                state = gs.GameState(
                    location=_location_from_label(case.get("location"), bm.Location.BLIND),
                    round_won=bool(case.get("round_won", False)),
                )
                detected = cv_stubs.cv_detect_hand_cards(
                    state, frame, conf_threshold=float(case.get("conf_threshold", 0.50))
                )
                self._assert_item_sequence(
                    detected, case["cards"], self._assert_hand_card_matches, f"hand:{rel_path}"
                )

    def test_cv_detect_shop_cards_real_images(self):
        cases = _load_ground_truth().get("shop", [])
        if not cases:
            self.skipTest(
                "No real-image fixtures configured in testing/real_image_fixtures/ground_truth.json['shop']."
            )

        for case in cases:
            rel_path = case["image"]
            with self.subTest(image=rel_path):
                frame = _load_frame(rel_path)
                state = gs.GameState(location=bm.Location.SHOP)
                detected = cv_stubs.cv_detect_shop_cards(
                    state, frame, conf_threshold=float(case.get("conf_threshold", 0.35))
                )
                shelf_pairs = [
                    ("shop_cards", list(detected)),
                    ("vouchers", detected.vouchers),
                    ("packs", detected.packs),
                    ("joker_bar", detected.joker_bar),
                    ("consumables", detected.consumables),
                ]
                for key, actual_list in shelf_pairs:
                    if key in case:
                        self._assert_item_sequence(
                            actual_list,
                            case[key],
                            self._assert_shop_like_item_matches,
                            f"shop:{rel_path}:{key}",
                        )

    def test_cv_detect_pack_cards_real_images(self):
        cases = _load_ground_truth().get("pack", [])
        if not cases:
            self.skipTest(
                "No real-image fixtures configured in testing/real_image_fixtures/ground_truth.json['pack']."
            )

        for case in cases:
            rel_path = case["image"]
            with self.subTest(image=rel_path):
                frame = _load_frame(rel_path)
                state = gs.GameState(
                    location=_location_from_label(case.get("location"), bm.Location.PACK)
                )
                detected = cv_stubs.cv_detect_pack_cards(
                    state, frame, conf_threshold=float(case.get("conf_threshold", 0.35))
                )
                self._assert_item_sequence(
                    detected, case["cards"], self._assert_shop_like_item_matches, f"pack:{rel_path}"
                )

    def test_cv_detect_joker_bar_real_images(self):
        cases = _load_ground_truth().get("joker_bar", [])
        if not cases:
            self.skipTest(
                "No real-image fixtures configured in testing/real_image_fixtures/ground_truth.json['joker_bar']."
            )

        for case in cases:
            rel_path = case["image"]
            with self.subTest(image=rel_path):
                frame = _load_frame(rel_path)
                state = gs.GameState(
                    location=_location_from_label(case.get("location"), bm.Location.BLIND),
                    round_won=bool(case.get("round_won", False)),
                )
                detected = cv_stubs.cv_detect_joker_bar(
                    state, frame, conf_threshold=float(case.get("conf_threshold", 0.35))
                )
                self._assert_item_sequence(
                    detected,
                    case["jokers"],
                    self._assert_shop_like_item_matches,
                    f"joker_bar:{rel_path}",
                )


if __name__ == "__main__":
    unittest.main()
