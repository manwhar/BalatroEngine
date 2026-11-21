"""Unit tests for vod_pipeline.metadata and vod_pipeline.runner."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from game.balatro_models import Deck, Stake
from vod_pipeline.metadata import extract_run_metadata, extract_video_id
from vod_pipeline.runner import PipelineConfig, run_batch_pipeline


class TestVodMetadataAndBatchPipeline(unittest.TestCase):
    def test_extract_run_metadata_on_all_19_saved_sample_inputs(self) -> None:
        meta_dir = Path("assets/sample_inputs/metadata")
        self.assertTrue(meta_dir.exists(), f"Expected saved metadata in {meta_dir}")
        index_path = meta_dir / "index.json"
        self.assertTrue(index_path.exists())

        expected_ready = {
            "xPGzf7YJlxM": (Deck.ERRATIC_DECK, "Z1DCPJSE", Stake.GOLD),
            "tIu3nyD178A": (Deck.ANAGLYPH_DECK, "5CT15E6W", Stake.GOLD),
            "_ZuzKjGVRPs": (Deck.MAGIC_DECK, "27T3PBQG", Stake.GOLD),
            "BlJRaf_XTYU": (Deck.MAGIC_DECK, "ISAD3FV6", Stake.GOLD),
            "1166hAVgn9M": (Deck.BLACK_DECK, "W3RIR565", Stake.GOLD),
            "WTS86AYxU6o": (Deck.PAINTED_DECK, "QDZMEKC3", Stake.GOLD),
            "QAi7tTvPXhc": (Deck.NEBULA_DECK, "42REVY7N", Stake.GOLD),
            "YpR4WuvVjbU": (Deck.PAINTED_DECK, "CM78H84Y", Stake.GOLD),
            "zqINP6NEwkQ": (Deck.BLUE_DECK, "PMLR4BVQ", Stake.GOLD),
            "sVuPmDmTyL4": (Deck.ZODIAC_DECK, "FNSD85BC", Stake.GOLD),
            "1HrCULIV_cg": (Deck.ZODIAC_DECK, "KX6LKY4S", Stake.GOLD),
            "Nj_k0gWhRH4": (Deck.ANAGLYPH_DECK, "EXESWZR2", Stake.GOLD),
            "GaIkZ2p07ZM": (Deck.GHOST_DECK, "M4FT2H9Y", Stake.GOLD),
            "BzMdXrJJGZI": (Deck.CHECKERED_DECK, "L68SU9WS", Stake.GOLD),
            "QRvxSAY-s1M": (Deck.PAINTED_DECK, "X37S2WYQ", Stake.GOLD),
            "gJVogwKQulY": (Deck.GHOST_DECK, "8B6YD9FJ", Stake.GOLD),
        }
        expected_skipped = {
            "_MDwyFmIo5s": ("missing_deck", "BZTGPV12", Stake.GOLD),
            "9vDE86eVa0Y": ("missing_deck", "UXS1G65E", Stake.GOLD),
            "KwcBB0pDDJo": ("missing_deck", "VUCDKYZ6", Stake.GOLD),
        }

        for vid, (exp_deck, exp_seed, exp_stake) in expected_ready.items():
            raw = json.loads((meta_dir / f"{vid}.json").read_text(encoding="utf-8"))
            parsed = extract_run_metadata(raw)
            self.assertTrue(parsed.ready, f"{vid} should be ready: {parsed}")
            self.assertIsNone(parsed.skip_reason)
            self.assertEqual(parsed.deck, exp_deck, vid)
            self.assertEqual(parsed.seed, exp_seed, vid)
            self.assertEqual(parsed.stake, exp_stake, vid)

        for vid, (exp_reason, exp_seed, exp_stake) in expected_skipped.items():
            raw = json.loads((meta_dir / f"{vid}.json").read_text(encoding="utf-8"))
            parsed = extract_run_metadata(raw)
            self.assertFalse(parsed.ready, f"{vid} should be skipped: {parsed}")
            self.assertEqual(parsed.skip_reason, exp_reason, vid)
            self.assertIsNone(parsed.deck, vid)
            self.assertEqual(parsed.seed, exp_seed, vid)
            self.assertEqual(parsed.stake, exp_stake, vid)

    def test_metadata_extraction_edge_cases_and_ambiguity_guards(self) -> None:
        self.assertEqual(
            extract_video_id("https://www.youtube.com/watch?v=abc123_X&t=10s"),
            "abc123_X",
        )
        self.assertEqual(extract_video_id("https://youtu.be/xyz987-Q?t=5"), "xyz987-Q")

        # Title White Stake overrides Gold Stake description boilerplate
        m1 = extract_run_metadata(
            {
                "id": "stake_override",
                "title": "White Stake Chill Run",
                "description": "Plasma Deck:\n\nAB12CD34\nDaily Gold Stake gameplay.",
            }
        )
        self.assertTrue(m1.ready)
        self.assertEqual(m1.deck, Deck.PLASMA_DECK)
        self.assertEqual(m1.seed, "AB12CD34")
        self.assertEqual(m1.stake, Stake.WHITE)

        # Missing stake defaults to Gold Stake; bare color words ("Red Card", "Blue Seal") ignored
        m2 = extract_run_metadata(
            {
                "id": "default_gold",
                "title": "Red Card + Blue Seal Combo!",
                "description": "Abandoned Deck: \n\n7K9MP2QT\nJust fun Balatro.",
            }
        )
        self.assertTrue(m2.ready)
        self.assertEqual(m2.deck, Deck.ABANDONED_DECK)
        self.assertEqual(m2.seed, "7K9MP2QT")
        self.assertEqual(m2.stake, Stake.GOLD)
        self.assertEqual(m2.stake_source, "default_gold")

        # Conflicting decks between title and description header -> skip safely
        m3 = extract_run_metadata(
            {
                "id": "conflict",
                "title": "Checkered Deck madness",
                "description": "Yellow Deck:\n\n2P3Q4R5S",
            }
        )
        self.assertFalse(m3.ready)
        self.assertEqual(m3.skip_reason, "ambiguous_deck_conflict")

        # Multiple seeds -> skip safely
        m4 = extract_run_metadata(
            {
                "id": "multiseed",
                "title": "Two Green Deck runs",
                "description": "Seed: 1A2B3C4D\nSeed: 5E6F7G8H",
            }
        )
        self.assertFalse(m4.ready)
        self.assertEqual(m4.skip_reason, "ambiguous_multiple_seeds")

        # URLs containing 8-char alphanumeric strings must never match as seeds
        m5 = extract_run_metadata(
            {
                "id": "url_only",
                "title": "Plasma Deck Only",
                "description": "Check https://bit.ly/4qXxtdN2 and https://example.com/1234ABCD",
            }
        )
        self.assertFalse(m5.ready)
        self.assertEqual(m5.skip_reason, "missing_seed")

    def test_batch_pipeline_overlaps_bounds_backlog_skips_and_deletes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir) / "batch_runs"
            urls = [f"https://www.youtube.com/watch?v=vid{i}" for i in range(7)]

            # vid1 missing deck -> must be skipped BEFORE downloading
            # vid4 missing seed -> must be skipped BEFORE downloading
            # vid5 fails during processing -> video MUST still be deleted
            def fake_fetch(input_urls, _cfg):
                out = []
                for idx, u in enumerate(input_urls):
                    vid = f"vid{idx}"
                    if idx == 1:
                        raw = {"id": vid, "title": "Red Card only", "description": "Seed: 12AB34CD"}
                    elif idx == 4:
                        raw = {"id": vid, "title": "Anaglyph Deck", "description": "No seed here"}
                    else:
                        raw = {
                            "id": vid,
                            "title": "Purple Stake Magic Deck",
                            "description": f"Magic Deck:\n\n2B3C4D5{idx}",
                        }
                    out.append((raw, extract_run_metadata(raw, url=u)))
                return out

            state_lock = threading.Lock()
            downloaded_vids: list[str] = []
            processed_vids: list[str] = []
            active_downloads = [0]
            max_active_downloads = [0]
            max_files_on_disk = [0]

            def _count_disk_videos(ddir: Path) -> int:
                return len(list(ddir.glob("*.mp4"))) if ddir.exists() else 0

            def fake_download(_url, video_id, downloads_dir, _cfg, _stop_event):
                with state_lock:
                    downloaded_vids.append(video_id)
                    active_downloads[0] += 1
                    max_active_downloads[0] = max(max_active_downloads[0], active_downloads[0])
                time.sleep(0.06)
                p = downloads_dir / f"{video_id}.mp4"
                p.write_bytes(b"fake_mp4")
                with state_lock:
                    active_downloads[0] -= 1
                    max_files_on_disk[0] = max(max_files_on_disk[0], _count_disk_videos(downloads_dir))
                return p

            def fake_process(video_path, meta, _cfg):
                with state_lock:
                    max_files_on_disk[0] = max(
                        max_files_on_disk[0], _count_disk_videos(video_path.parent)
                    )
                    processed_vids.append(meta.video_id)
                time.sleep(0.05)
                if meta.video_id == "vid5":
                    raise RuntimeError("simulated processing failure")
                traj_out = work_dir / "trajectories" / f"{meta.video_id}_trajectory.json"
                traj_out.write_text("{}", encoding="utf-8")
                return {
                    "actions_json": str(work_dir / "actions" / f"{meta.video_id}_actions.json"),
                    "trajectory_json": str(traj_out),
                    "num_steps": 12,
                    "final_ante": 8,
                    "final_round": 24,
                    "elapsed_sec": 0.05,
                }

            cfg = PipelineConfig(
                work_dir=work_dir,
                max_backlog=2,
                download_workers=2,
                keep_videos=False,
            )
            report = run_batch_pipeline(
                urls,
                cfg,
                fetch_metadata_fn=fake_fetch,
                download_fn=fake_download,
                process_fn=fake_process,
            )

            self.assertEqual(report["counts"]["total"], 7)
            self.assertEqual(report["counts"]["eligible"], 5)
            self.assertEqual(report["counts"]["processed"], 4)
            self.assertEqual(report["counts"]["skipped"], 2)
            self.assertEqual(report["counts"]["failed"], 1)

            self.assertNotIn("vid1", downloaded_vids)
            self.assertNotIn("vid4", downloaded_vids)
            self.assertEqual(set(downloaded_vids), {"vid0", "vid2", "vid3", "vid5", "vid6"})
            self.assertGreaterEqual(max_active_downloads[0], 2)
            self.assertLessEqual(max_files_on_disk[0], 2)
            # All processed & failed videos deleted after completion
            self.assertEqual(list((work_dir / "downloads").glob("*.mp4")), [])
            self.assertTrue(Path(report["report_path"]).exists())
            self.assertTrue((work_dir / "reports" / "manifest.jsonl").exists())

            # Re-running with resume=True skips the 4 completed videos without re-downloading
            downloaded_vids.clear()
            report_resume = run_batch_pipeline(
                urls,
                cfg,
                fetch_metadata_fn=fake_fetch,
                download_fn=fake_download,
                process_fn=fake_process,
            )
            self.assertEqual(report_resume["counts"]["eligible"], 1)  # only previously failed vid5
            self.assertEqual(downloaded_vids, ["vid5"])


if __name__ == "__main__":
    unittest.main()
