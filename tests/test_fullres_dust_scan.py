import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from negpy.desktop.controller import AppController
from negpy.desktop.session import AppState, DesktopSessionManager
from negpy.features.retouch.logic import (
    compute_dust_stats,
    detect_luma_score,
    dust_detections_to_heal_strokes,
    filter_strokes_by_score,
    strokes_to_score,
)
from negpy.features.retouch.models import HEAL_SIZE_MAX, HEAL_SIZE_MIN
from negpy.services.rendering.preview_manager import PreviewManager


def _grainy(h, w, level=0.5, sigma=0.01, seed=21):
    rng = np.random.default_rng(seed)
    return (np.full((h, w, 3), level) + rng.normal(0, sigma, (h, w, 3))).astype(np.float32)


def test_empty_detection_converts_to_no_strokes():
    assert dust_detections_to_heal_strokes(None, [], (100, 100)) == []
    assert dust_detections_to_heal_strokes(np.ones((100, 100), dtype=np.float32), [], (100, 100)) == []


def test_speck_becomes_one_centred_spot_heal():
    score = np.ones((1600, 1600), dtype=np.float32)
    score[798:803, 798:803] = 0.02  # at-floor defect block
    strokes = dust_detections_to_heal_strokes(score, [], (1600, 1600))
    assert len(strokes) == 1
    (points, size, dx, dy) = strokes[0]
    assert dx == 0.0 and dy == 0.0
    assert HEAL_SIZE_MIN <= size <= HEAL_SIZE_MAX
    (nx, ny) = points[0]
    assert 798 / 1600 <= nx <= 802 / 1600
    assert 798 / 1600 <= ny <= 802 / 1600


def test_hair_mask_becomes_strokes_on_the_defect():
    hair = np.zeros((100, 100), dtype=np.uint8)
    hair[50, 10:90] = 1  # long thin hair
    strokes = dust_detections_to_heal_strokes(None, [hair], (100, 100))
    assert strokes
    for points, size, _dx, _dy in strokes:
        (nx, ny) = points[0]
        assert 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0
        assert HEAL_SIZE_MIN <= size <= HEAL_SIZE_MAX
        # Spots sit on the hair row (within one brush diameter of it).
        assert abs(ny * 100 - 50) <= size


def test_stroke_cap_keeps_the_largest_first():
    score = np.ones((200, 200), dtype=np.float32)
    score[10:14, 10:14] = 0.02  # small speck
    score[100:160, 100:160] = 0.02  # large block (tiles, capped)
    strokes = dust_detections_to_heal_strokes(score, [], (200, 200), max_strokes=3)
    assert len(strokes) == 3
    # The large block tiles first (sorted largest-first).
    (nx, _ny) = strokes[0][0][0]
    assert nx >= 100 / 200


def test_fullres_roundtrip_repairs_a_detected_speck():
    """detect → convert → manual bake finds something: the placed heal covers its speck."""
    rng = np.random.default_rng(42)
    img = (np.full((160, 160, 3), 0.18) * (1.0 + rng.normal(0, 0.02, (160, 160, 3)))).astype(np.float32)
    img[80:83, 80:83] = 0.005  # dust blocks scanner light: dark on the scan
    stats = compute_dust_stats(img, 4)
    score, hairs = detect_luma_score(img, 0.66, 4, stats=stats, hair_threshold=1.0)
    assert score is not None
    strokes = dust_detections_to_heal_strokes(score, [], img.shape[:2])
    assert strokes
    manual = strokes_to_score(img, strokes, [])
    assert manual is not None and (manual < 1.0).any()


def test_filter_keeps_strokes_the_manual_gate_refires():
    rng = np.random.default_rng(42)
    img = (np.full((160, 160, 3), 0.18) * (1.0 + rng.normal(0, 0.02, (160, 160, 3)))).astype(np.float32)
    img[80:83, 80:83] = 0.005
    stats = compute_dust_stats(img, 4)
    score, _ = detect_luma_score(img, 0.66, 4, stats=stats, hair_threshold=1.0)
    strokes = dust_detections_to_heal_strokes(score, [], img.shape[:2])
    assert strokes
    manual = strokes_to_score(img, strokes, [])
    kept, dropped = filter_strokes_by_score(strokes, manual, img.shape[:2])
    assert len(kept) == len(strokes) and not dropped


def test_filter_drops_strokes_on_clean_film():
    img = _grainy(160, 160)
    phantom = [([[0.5, 0.5]], 6.0, 0.0, 0.0)]
    manual = strokes_to_score(img, phantom, [])
    assert manual is None
    kept, dropped = filter_strokes_by_score(phantom, manual, img.shape[:2])
    assert kept == [] and dropped == phantom


class TestDustScanFinished(unittest.TestCase):
    """The finished slot files hits as manual heals and drops stale runs."""

    def setUp(self):
        self.mock_session_manager = MagicMock(spec=DesktopSessionManager)
        self.mock_session_manager.state = AppState()
        self.mock_session_manager.repo = MagicMock()

        with (
            patch("negpy.desktop.controller.RenderWorker") as mock_rw_class,
            patch("negpy.desktop.controller.PreviewManager") as mock_pm_class,
        ):
            mock_rw_class.return_value = MagicMock()
            mock_pm_class.return_value = MagicMock(spec=PreviewManager)
            mock_pm_class.return_value.load_linear_preview.return_value = (None, (0, 0), {})
            self.controller = AppController(self.mock_session_manager)
        self.controller.request_render = MagicMock()
        self.controller.state.current_file_path = "/rolls/frame.tif"
        self.controller.state.current_file_hash = "frame-hash"

    def tearDown(self):
        import gc

        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
            self.controller.prefetch_load_thread,
            self.controller.scan_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    def test_finished_appends_heals_and_renders(self):
        self.controller._dust_scan_running = True
        strokes = [([[0.5, 0.5]], 6.0, 0.0, 0.0), ([[0.2, 0.2]], 6.0, 0.0, 0.0)]

        self.controller._on_dust_scan_finished(strokes, 0, "frame-hash")

        self.assertFalse(self.controller.is_dust_scan_running)
        saved = self.mock_session_manager.update_config.call_args.args[0]
        self.assertEqual(list(saved.retouch.manual_heal_strokes), strokes)
        self.assertTrue(self.mock_session_manager.update_config.call_args.kwargs.get("persist"))
        self.controller.request_render.assert_called_once()

    def test_stale_run_is_dropped(self):
        self.controller._dust_scan_running = True

        self.controller._on_dust_scan_finished([([[0.5, 0.5]], 6.0, 0.0, 0.0)], 0, "other-hash")

        self.assertFalse(self.controller.is_dust_scan_running)
        self.mock_session_manager.update_config.assert_not_called()
        self.controller.request_render.assert_not_called()

    def test_empty_run_writes_nothing(self):
        self.controller._dust_scan_running = True

        self.controller._on_dust_scan_finished([], 0, "frame-hash")

        self.assertFalse(self.controller.is_dust_scan_running)
        self.mock_session_manager.update_config.assert_not_called()
        self.controller.request_render.assert_not_called()

    def test_all_dropped_reports_without_writing(self):
        self.controller._dust_scan_running = True
        msgs = []
        self.controller.status_message_requested.connect(lambda text, _ms, kind: msgs.append(text))

        self.controller._on_dust_scan_finished([], 3, "frame-hash")

        self.assertFalse(self.controller.is_dust_scan_running)
        self.mock_session_manager.update_config.assert_not_called()
        self.controller.request_render.assert_not_called()
        self.assertTrue(any("3 marks" in m for m in msgs))

    def test_rescan_skips_already_placed_centres(self):
        from dataclasses import replace

        placed = [([[0.5, 0.5]], 6.0, 0.0, 0.0)]
        retouch = replace(self.controller.state.config.retouch, manual_heal_strokes=placed)
        self.controller.state.config = replace(self.controller.state.config, retouch=retouch)
        self.controller._dust_scan_running = True

        self.controller._on_dust_scan_finished(
            [([[0.5, 0.5]], 6.0, 0.0, 0.0), ([[0.7, 0.7]], 6.0, 0.0, 0.0)],
            2,
            "frame-hash",
        )

        saved = self.mock_session_manager.update_config.call_args.args[0]
        self.assertEqual(
            [tuple(p[0]) for p in (s[0] for s in saved.retouch.manual_heal_strokes)],
            [(0.5, 0.5), (0.7, 0.7)],
        )
