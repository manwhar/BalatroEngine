"""Unit tests for coarse screen-state template gating and Windows path handling."""

import json
from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from game.balatro_models import BlindName, BoosterPack, BoosterSize, BoosterType, Deck, Location
from game.gamestate import GameState
import main as main_module
from screen_state import (
    ALLOWED_TEMPLATES_BY_STATE,
    CoarseScreenState,
    ScreenStateTracker,
    is_action_clickable_in_gamestate,
    validate_templates_covered,
)


class TestScreenStateTracker(unittest.TestCase):
    def test_all_registered_templates_covered(self):
        templates_path = REPO_ROOT / "assets" / "templates" / "drspectred" / "templates.json"
        with open(templates_path, "r", encoding="utf-8") as f:
            registered = list(json.load(f).keys())
        self.assertEqual(validate_templates_covered(registered), [])

    def test_pre_game_to_blind_to_cash_out_to_shop(self):
        tracker = ScreenStateTracker()
        self.assertEqual(tracker.state, CoarseScreenState.PRE_GAME)
        pre_allowed = tracker.allowed_template_names()
        self.assertIn("Play Game", pre_allowed)
        self.assertIn("Select Blind", pre_allowed)
        self.assertIn("Skip Blind", pre_allowed)
        self.assertNotIn("Play Hand", pre_allowed)
        self.assertNotIn("Buy Card", pre_allowed)
        self.assertNotIn("Use In Pack", pre_allowed)

        # Main menu PLAY -> Run setup PLAY -> Blind Select
        self.assertEqual(tracker.on_template_clicked("Play Game"), CoarseScreenState.SELECT)
        self.assertIn("Play Game", tracker.allowed_template_names())
        self.assertEqual(tracker.on_template_clicked("Play Game"), CoarseScreenState.SELECT)
        self.assertIn("Select Blind", tracker.allowed_template_names())

        # Select Blind -> Blind (Play/Discard/Sort AND Cash Out for implicit round win!)
        self.assertEqual(tracker.on_template_clicked("Select Blind"), CoarseScreenState.BLIND)
        blind_allowed = tracker.allowed_template_names()
        self.assertNotIn("Play Game", blind_allowed)
        self.assertNotIn("Select Blind", blind_allowed)
        self.assertNotIn("Use In Pack", blind_allowed)
        for btn in ("Play Hand", "Discard", "Sort by Rank", "Sort by Suit", "Cash Out", "Sell Card", "Use Consumable"):
            self.assertIn(btn, blind_allowed)

        # Cash Out -> Shop
        self.assertEqual(tracker.on_template_clicked("Cash Out"), CoarseScreenState.SHOP)
        shop_allowed = tracker.allowed_template_names()
        self.assertIn("Buy Card", shop_allowed)
        self.assertIn("Open Pack", shop_allowed)
        self.assertIn("Next Round", shop_allowed)
        self.assertNotIn("Play Hand", shop_allowed)
        self.assertNotIn("Use In Pack", shop_allowed)

    def test_shop_pack_and_mega_pack_hybrid_states(self):
        tracker = ScreenStateTracker()
        tracker.on_template_clicked("Select Blind")
        tracker.on_template_clicked("Cash Out")

        # Open Pack from Shop -> strict PACK state (no Sort/Play/Buy/Cash Out)
        self.assertEqual(tracker.on_template_clicked("Open Pack"), CoarseScreenState.PACK)
        pack_allowed = tracker.allowed_template_names()
        self.assertEqual(
            pack_allowed,
            {
                "Use In Pack",
                "Select Card",
                "Skip Pack",
                "Sell Card",
                "Use Consumable",
                "Open Options Menu",
                "Open Run Info Menu",
            },
        )

        # Choosing a card in pack enters PACK_OR_SHOP (supports both choose=1 exit & choose=2 Mega!)
        self.assertEqual(tracker.on_template_clicked("Use In Pack"), CoarseScreenState.PACK_OR_SHOP)
        hybrid_allowed = tracker.allowed_template_names()
        self.assertIn("Use In Pack", hybrid_allowed)
        self.assertIn("Skip Pack", hybrid_allowed)
        self.assertIn("Buy Card", hybrid_allowed)
        self.assertIn("Next Round", hybrid_allowed)
        self.assertNotIn("Play Hand", hybrid_allowed)

        # If it was Mega, second Skip Pack returns cleanly to SHOP
        self.assertEqual(tracker.on_template_clicked("Skip Pack"), CoarseScreenState.SHOP)

    def test_skip_blind_tag_pack_and_double_tag_caution(self):
        tracker = ScreenStateTracker()
        tracker.on_template_clicked("Play Game")
        # Skip Blind could stay on SELECT (economy/investment tag) OR open a pack (Charm/Meteor tag)
        self.assertEqual(tracker.on_template_clicked("Skip Blind"), CoarseScreenState.SELECT_OR_PACK)
        allowed = tracker.allowed_template_names()
        self.assertIn("Select Blind", allowed)
        self.assertIn("Skip Blind", allowed)
        self.assertIn("Use In Pack", allowed)
        self.assertIn("Skip Pack", allowed)
        self.assertNotIn("Play Hand", allowed)

        # Choosing in pack -> PACK_OR_SELECT
        self.assertEqual(tracker.on_template_clicked("Select Card"), CoarseScreenState.PACK_OR_SELECT)
        # Skip Pack from SELECT keeps SELECT_OR_PACK cleanly in case Double Tag queued another pack
        self.assertEqual(tracker.on_template_clicked("Skip Pack"), CoarseScreenState.SELECT_OR_PACK)
        # Select Blind resolves back to BLIND
        self.assertEqual(tracker.on_template_clicked("Select Blind"), CoarseScreenState.BLIND)

    def test_options_allows_new_run_play_game_and_end_round_narrows_to_cash_out(self):
        tracker = ScreenStateTracker()
        tracker.on_template_clicked("Select Blind")
        self.assertNotIn("Play Game", tracker.allowed_template_names())
        tracker.on_template_clicked("Open Options Menu")
        self.assertIn("Play Game", tracker.allowed_template_names())
        self.assertIn("Play Hand", tracker.allowed_template_names())

        # Normal gameplay resumes clearing allow_play_game
        tracker.on_template_clicked("Play Hand")
        self.assertNotIn("Play Game", tracker.allowed_template_names())

        # Explicit temporal END_ROUND narrows BLIND -> CASH_OUT
        self.assertEqual(tracker.on_end_round(), CoarseScreenState.CASH_OUT)
        self.assertNotIn("Play Hand", tracker.allowed_template_names())
        self.assertIn("Cash Out", tracker.allowed_template_names())

    def test_consistency_with_gamestate(self):
        s_blind = GameState(location=Location.BLIND, hands_left=4, discards_left=3)
        for tmpl in ("Play Hand", "Discard", "Sort by Rank", "Sort by Suit", "Sell Card", "Use Consumable"):
            self.assertTrue(is_action_clickable_in_gamestate(s_blind, tmpl))
            self.assertIn(tmpl, ALLOWED_TEMPLATES_BY_STATE[CoarseScreenState.BLIND])

        s_shop = GameState(location=Location.SHOP)
        for tmpl in ("Buy Card", "Buy and Use", "Open Pack", "Redeem Voucher", "Reroll Shop", "Next Round"):
            self.assertTrue(is_action_clickable_in_gamestate(s_shop, tmpl))
            self.assertIn(tmpl, ALLOWED_TEMPLATES_BY_STATE[CoarseScreenState.SHOP])

        s_pack = GameState(
            location=Location.PACK,
            active_pack=BoosterPack(
                booster_type=BoosterType.ARCANA, size=BoosterSize.REGULAR, cost=4, choose=1, extra=3
            ),
        )
        for tmpl in ("Use In Pack", "Select Card", "Skip Pack", "Sell Card", "Use Consumable"):
            self.assertTrue(is_action_clickable_in_gamestate(s_pack, tmpl))
            self.assertIn(tmpl, ALLOWED_TEMPLATES_BY_STATE[CoarseScreenState.PACK])
        for forbidden in ("Play Hand", "Sort by Rank", "Buy Card", "Select Blind"):
            self.assertFalse(is_action_clickable_in_gamestate(s_pack, forbidden))
            self.assertNotIn(forbidden, ALLOWED_TEMPLATES_BY_STATE[CoarseScreenState.PACK])

    def test_windows_and_posix_video_output_path(self):
        win_out = main_module.build_output_json_path(
            "outputs", r"C:\Users\thatp\Videos\balatro_run_1.mp4", timestamp=123456
        )
        self.assertEqual(Path(win_out), Path("outputs") / "balatro_run_1_123456.json")
        posix_out = main_module.build_output_json_path(
            "outputs", "/home/user/videos/balatro_run_2.mkv", timestamp=123456
        )
        self.assertEqual(Path(posix_out), Path("outputs") / "balatro_run_2_123456.json")


if __name__ == "__main__":
    unittest.main()
