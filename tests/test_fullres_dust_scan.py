import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from negpy.desktop.controller import AppController, heal_bypass_config
from negpy.desktop.session import AppState, DesktopSessionManager, ToolMode
from negpy.features.retouch.logic import (
    compute_dust_stats,
    detect_bright_luma_score,
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


def test_hair_mask_becomes_one_traced_stroke():
    hair = np.zeros((1600, 1600), dtype=np.uint8)
    hair[800:802, 600:800] = 1  # long thin hair
    strokes = dust_detections_to_heal_strokes(None, [hair], (1600, 1600))
    assert len(strokes) == 1
    points, size, _dx, _dy = strokes[0]
    assert len(points) >= 2, "a trace, not a spot"
    assert HEAL_SIZE_MIN <= size <= HEAL_SIZE_MAX
    xs = sorted(p[0] * 1600 for p in points)
    assert xs[0] <= 601.0 and xs[-1] >= 799.0, "the trace spans the hair"
    assert all(abs(p[1] * 1600 - 801) <= 3.0 for p in points), "the trace hugs the hair"


def test_minor_hairline_traces_instead_of_circling():
    # A 1 px, 15 px hairline: below the detector's hair-routing minimum area, so the
    # old shape gate encircled it. Tracing needs no minimum.
    hair = np.zeros((1600, 1600), dtype=np.uint8)
    hair[900, 700:715] = 1
    strokes = dust_detections_to_heal_strokes(None, [hair], (1600, 1600))
    assert len(strokes) == 1
    points, _size, _dx, _dy = strokes[0]
    assert len(points) >= 2
    xs = sorted(p[0] * 1600 for p in points)
    assert xs[0] <= 701.0 and xs[-1] >= 714.0


def test_bent_hair_converts_to_closed_loop():
    hair = np.zeros((1600, 1600), dtype=np.uint8)
    hair[500:700, 800:803] = 1
    hair[697:700, 800:1000] = 1  # an L bend
    strokes = dust_detections_to_heal_strokes(None, [hair], (1600, 1600))
    assert len(strokes) == 1
    points, _size, _dx, _dy = strokes[0]
    assert len(points) > 2, "a bend survives as a loop, not two ends"
    assert points[0] == points[-1], "the loop is closed for the open-chain bake"
    assert all(500 - 5 <= p[1] * 1600 <= 700 + 5 and 800 - 5 <= p[0] * 1600 <= 1000 + 5 for p in points)


def test_hair_brush_is_sized_to_thickness():
    hair = np.zeros((1600, 1600), dtype=np.uint8)
    hair[800:803, 600:1000] = 1  # 3 px thick, 400 long
    strokes = dust_detections_to_heal_strokes(None, [hair], (1600, 1600))
    assert len(strokes) == 1
    _points, size, _dx, _dy = strokes[0]
    # Thickness plus skirt pad, not the 400 px bounding length.
    assert size < 20.0


def test_hair_roundtrip_repairs_through_the_trace():
    rng = np.random.default_rng(11)
    h = w = 800
    img = (np.full((h, w, 3), 0.18) * (1.0 + rng.normal(0, 0.02, (h, w, 3)))).astype(np.float32)
    img[400:402, 300:450] = 0.005
    _score, hair = detect_luma_score(img, 0.66, 4, hair_threshold=0.01)
    assert hair is not None
    strokes = dust_detections_to_heal_strokes(None, [hair], img.shape[:2])
    assert len(strokes) == 1 and len(strokes[0][0]) >= 2
    manual = strokes_to_score(img, strokes, [])
    assert manual is not None and (manual < 1.0).any()
    kept, dropped = filter_strokes_by_score(strokes, manual, img.shape[:2])
    assert kept == strokes and not dropped


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


def _unexposed_speck_source(h=160, w=160, seed=42):
    rng = np.random.default_rng(seed)
    img = (np.full((h, w, 3), 0.18) * (1.0 + rng.normal(0, 0.02, (h, w, 3)))).astype(np.float32)
    img[80:83, 80:83] = 0.70  # unexposed: clear on the negative, bright on the scan
    return img


def test_detect_bright_luma_score_finds_unexposed_speck():
    img = _unexposed_speck_source()
    score = detect_bright_luma_score(img, dust_threshold=0.66, dust_size=4)
    assert score is not None
    assert score[80:83, 80:83].max() < 1.0, "the unexposed speck must be scored as a defect"
    ys, xs = np.where(score < 1.0)
    assert abs(float(xs.mean()) - 81.5) < 5.0 and abs(float(ys.mean()) - 81.5) < 5.0


def test_detect_bright_luma_score_clean_frame_is_empty():
    assert detect_bright_luma_score(_grainy(160, 160), 0.66, 4) is None


def test_detect_bright_luma_score_threshold_off_is_empty():
    assert detect_bright_luma_score(_unexposed_speck_source(), 1.0, 4) is None


def test_bright_roundtrip_repairs_a_detected_speck():
    """Bright speck → convert → manual bake finds something, like the dark side."""
    img = _unexposed_speck_source()
    score = detect_bright_luma_score(img, 0.66, 4)
    assert score is not None
    strokes = dust_detections_to_heal_strokes(score, [], img.shape[:2])
    assert strokes
    manual = strokes_to_score(img, strokes, [])
    assert manual is not None and (manual < 1.0).any()
    kept, dropped = filter_strokes_by_score(strokes, manual, img.shape[:2])
    assert len(kept) == len(strokes) and not dropped


def test_merged_dark_and_bright_scores_convert_both():
    """The worker's minimum-merge of both sides converts every component."""
    rng = np.random.default_rng(9)
    h = w = 1600
    img = (np.full((h, w, 3), 0.18) * (1.0 + rng.normal(0, 0.02, (h, w, 3)))).astype(np.float32)
    img[400:403, 400:403] = 0.005  # scanner dust: dark on the scan
    img[1200:1203, 1200:1203] = 0.70  # unexposed dust: bright on the scan
    stats = compute_dust_stats(img, 4)
    dark, _ = detect_luma_score(img, 0.66, 4, stats=stats, hair_threshold=1.0)
    bright = detect_bright_luma_score(img, 0.66, 4)
    assert dark is not None and bright is not None
    merged = np.minimum(dark, bright).astype(np.float32)
    strokes = dust_detections_to_heal_strokes(merged, [], img.shape[:2])
    assert len(strokes) == 2
    manual = strokes_to_score(img, strokes, [])
    kept, dropped = filter_strokes_by_score(strokes, manual, img.shape[:2])
    assert len(kept) == 2 and not dropped


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


def test_heal_bypass_config_lifts_heals_keeps_clones():
    from dataclasses import replace

    from negpy.domain.models import WorkspaceConfig

    config = WorkspaceConfig()
    config = replace(
        config,
        retouch=replace(
            config.retouch,
            manual_heal_strokes=[([[0.5, 0.5]], 6.0, 0.0, 0.0)],
            manual_dust_spots=[(0.2, 0.2, 6.0)],
            scratch_lines=[(0.0, 0.0, 1.0, 1.0, 8.0)],
            clone_strokes=[([[0.1, 0.1]], 6.0, 0.0, 0.0, 1.0, 0.5, True)],
        ),
    )
    out = heal_bypass_config(config)
    assert out.retouch.manual_heal_strokes == []
    assert out.retouch.manual_dust_spots == []
    assert out.retouch.scratch_lines == []
    assert out.retouch.clone_strokes == config.retouch.clone_strokes
    assert out.exposure == config.exposure


class TestHealBypass(unittest.TestCase):
    """The Show Original toggle previews the unhealed source without touching the edit."""

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

    def test_toggle_flips_and_renders(self):
        import numpy as np

        self.controller.state.preview_raw = np.zeros((8, 8, 3), dtype=np.float32)
        seen = []
        self.controller.heal_bypass_changed.connect(seen.append)

        self.controller.toggle_heal_bypass()

        self.assertTrue(self.controller.state.heal_bypass)
        self.assertEqual(seen, [True])
        self.controller.request_render.assert_called_once()

    def test_toggle_off_renders_plain(self):
        import numpy as np

        self.controller.state.preview_raw = np.zeros((8, 8, 3), dtype=np.float32)
        self.controller.state.heal_bypass = True

        self.controller.toggle_heal_bypass(force=False)

        self.assertFalse(self.controller.state.heal_bypass)
        self.controller.request_render.assert_called_once()

    def test_bypassed_render_strips_heals_and_skips_memo(self):
        import numpy as np

        from dataclasses import replace

        retouch = replace(self.controller.state.config.retouch, manual_heal_strokes=[([[0.5, 0.5]], 6.0, 0.0, 0.0)])
        self.controller.state.config = replace(self.controller.state.config, retouch=retouch)
        self.controller.state.current_file_path = ""
        self.controller.state.preview_raw = np.zeros((16, 16, 3), dtype=np.float32)
        self.controller.state.heal_bypass = True
        tasks = []
        self.controller.render_requested.connect(tasks.append)

        AppController.request_render(self.controller)

        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(task.config.retouch.manual_heal_strokes, [])
        self.assertEqual(len(self.controller.state.config.retouch.manual_heal_strokes), 1, "the edit keeps its heals")
        self.assertTrue(task.config_override)
        self.assertEqual(task.memo_key, "")
        self.assertFalse(task.readback_metrics)


def test_region_to_pixels_orders_and_clamps():
    from negpy.features.retouch.logic import region_to_pixels

    assert region_to_pixels((0.75, 0.75, 0.25, 0.25), (100, 200)) == (50, 25, 150, 75)
    assert region_to_pixels((-0.5, -0.5, 2.0, 2.0), (100, 200)) == (0, 0, 200, 100)


def test_region_to_pixels_degenerate_is_no_region():
    from negpy.features.retouch.logic import region_to_pixels

    assert region_to_pixels((0.5, 0.5, 0.5, 0.5), (100, 200)) is None
    assert region_to_pixels((0.0, 0.0, 0.001, 0.001), (100, 200)) is None


class TestDustScanDispatch(unittest.TestCase):
    """The scan entry builds the worker task from the dialog choices."""

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

    def _dispatched_task(self):
        tasks = []
        self.controller.dust_scan_requested.connect(tasks.append)
        return tasks

    def test_defaults_repeat_the_plain_scan(self):
        tasks = self._dispatched_task()

        self.controller.scan_fullres_dust_to_heals()

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].polarity, "both")
        self.assertIsNone(tasks[0].region)
        self.assertEqual(self.controller._last_scan_polarity, "both")

    def test_polarity_and_region_reach_the_task(self):
        tasks = self._dispatched_task()

        self.controller.scan_fullres_dust_to_heals(polarity="print_dark", region=(0.1, 0.2, 0.4, 0.5))

        self.assertEqual(tasks[0].polarity, "print_dark")
        self.assertEqual(tasks[0].region, (0.1, 0.2, 0.4, 0.5))
        self.assertEqual(self.controller._last_scan_polarity, "print_dark")

    def test_unknown_polarity_falls_back_to_both(self):
        tasks = self._dispatched_task()

        self.controller.scan_fullres_dust_to_heals(polarity="sideways")

        self.assertEqual(tasks[0].polarity, "both")

    def test_degenerate_region_falls_back_to_whole(self):
        tasks = self._dispatched_task()

        self.controller.scan_fullres_dust_to_heals(region=(0.5, 0.5, 0.5, 0.5))

        self.assertIsNone(tasks[0].region)

    def test_region_outside_the_frame_clips(self):
        tasks = self._dispatched_task()

        self.controller.scan_fullres_dust_to_heals(region=(-0.5, -0.5, 2.0, 0.4))

        self.assertEqual(tasks[0].region, (0.0, 0.0, 1.0, 0.4))

    def test_second_scan_while_running_is_dropped(self):
        self.controller._dust_scan_running = True
        tasks = self._dispatched_task()

        self.controller.scan_fullres_dust_to_heals()

        self.assertEqual(tasks, [])

    def test_picked_region_returns_to_the_scan_dialog(self):
        self.controller.state.last_metrics["uv_grid"] = MagicMock()
        with (
            patch("negpy.desktop.controller.CoordinateMapping") as mock_map,
            patch.object(self.controller, "_run_dust_scan_dialog") as mock_open,
        ):
            mock_map.map_click_to_raw.side_effect = [(0.1, 0.2), (0.4, 0.5)]
            self.controller.handle_scan_region_picked([(0.0, 0.0), (1.0, 1.0)])

        self.assertEqual(self.controller.state.active_tool, ToolMode.NONE)
        mock_open.assert_called_once_with(region=(0.1, 0.2, 0.4, 0.5))

    def test_region_scan_button_arms_the_pick_tool(self):
        self.controller.open_scan_region_dialog()
        self.assertEqual(self.controller.state.active_tool, ToolMode.SCAN_REGION)
