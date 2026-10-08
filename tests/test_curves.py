"""Display-referred curves, applied after levels as the last step.

The math lives in two places — ``curves.py`` (CPU bake + apply) and
``shaders/curves.wgsl`` (integer-index table application). Both sides index the
same baked tables identically, so parity holds by construction.
"""

import unittest
from dataclasses import replace

import numpy as np

from negpy.domain.models import WorkspaceConfig, flat_master_config
from negpy.features.exposure.curves import (
    CURVE_NODES,
    active_mask,
    apply_curves,
    bake_channel_lut,
    channel_offsets,
    curves_active,
    curves_fields,
    grid_outputs,
)
from negpy.features.exposure.models import ExposureConfig
from negpy.infrastructure.gpu.device import GPUDevice


class TestCurvesMath(unittest.TestCase):
    def test_identity_by_default(self) -> None:
        conf = ExposureConfig()
        self.assertFalse(curves_active(conf))
        self.assertEqual(len(curves_fields()), 4 * CURVE_NODES)
        img = np.linspace(0.0, 1.0, 300, dtype=np.float32).reshape(10, 10, 3)
        np.testing.assert_array_equal(np.asarray(apply_curves(img, conf)), img)

    def test_identity_bake_is_the_exact_ramp(self) -> None:
        np.testing.assert_array_equal(bake_channel_lut((0.0,) * CURVE_NODES), np.arange(256, dtype=np.float32) / 255.0)

    def test_bake_passes_through_every_node(self) -> None:
        # Nodes sit between integer code values, so the bracketing LUT entries
        # straddle each node value within one steep pixel's slope.
        rng = np.random.default_rng(0)
        for _ in range(50):
            off = tuple(rng.uniform(-100, 100, CURVE_NODES))
            lut = bake_channel_lut(off)
            grid = np.maximum.accumulate(np.clip(np.array([i * 255 / 7 for i in range(8)]) + np.array(off), 0, 255))
            for i, gx in enumerate((0, 36, 73, 109, 146, 182, 219, 255)):
                self.assertAlmostEqual(float(lut[gx] * 255), float(grid[i]), delta=2.0)

    def test_bake_stays_monotone_and_bounded(self) -> None:
        rng = np.random.default_rng(1)
        for _ in range(500):
            lut = bake_channel_lut(tuple(rng.uniform(-200, 200, CURVE_NODES)))
            self.assertTrue(bool(np.all(np.diff(lut) >= 0)))
            self.assertGreaterEqual(float(lut.min()), 0.0)
            self.assertLessEqual(float(lut.max()), 1.0)

    def test_midtone_lift(self) -> None:
        conf = replace(ExposureConfig(), curve_3=40.0)
        out = np.asarray(apply_curves(np.full((4, 4, 3), 109.3 / 255, dtype=np.float32), conf))
        self.assertAlmostEqual(float(out[0, 0, 0] * 255), 149.3, delta=1.5)

    def test_per_channel_leaves_siblings_bit_exact(self) -> None:
        conf = replace(ExposureConfig(), curve_5_red=-50.0)
        self.assertEqual(active_mask(conf), (0, 1, 0, 0))
        out = np.asarray(apply_curves(np.full((2, 2, 3), 0.7, dtype=np.float32), conf))
        self.assertLess(float(out[0, 0, 0]), 0.7)
        # Untouched lanes never pass a table: bit-exact float32 passthrough.
        self.assertEqual(float(out[0, 0, 1]), float(np.float32(0.7)))
        self.assertEqual(float(out[0, 0, 2]), float(np.float32(0.7)))

    def test_config_values_are_clamped(self) -> None:
        conf = replace(ExposureConfig(), curve_0_red=9999.0, curve_1_red=-9999.0)
        lo = channel_offsets(conf, "red")
        self.assertEqual((lo[0], lo[1]), (255.0, -255.0))
        out = np.asarray(apply_curves(np.full((2, 2, 3), 0.5, dtype=np.float32), conf))
        self.assertTrue(bool(np.all(out >= 0.0)) and bool(np.all(out <= 1.0)))

    def test_grid_outputs_match_the_bake(self) -> None:
        self.assertEqual(grid_outputs(ExposureConfig(), "global"), (0, 36, 73, 109, 146, 182, 219, 255))
        conf = replace(ExposureConfig(), curve_7_blue=-300.0)
        # Held at the monotone bound: the bake never folds, the marker shows it.
        self.assertEqual(grid_outputs(conf, "blue")[-1], 219)


class TestCurvesPipeline(unittest.TestCase):
    def _processor(self):  # local import: building it pulls the desktop stack
        from negpy.services.rendering.image_processor import ImageProcessor

        return ImageProcessor()

    def _negative(self) -> np.ndarray:
        rng = np.random.default_rng(7)
        h, w = 48, 48
        grad = np.linspace(0.05, 0.9, w, dtype=np.float32)
        img = np.repeat(grad[None, :], h, axis=0)
        img = np.stack([img, img * 0.9, img * 0.8], axis=-1)
        return np.ascontiguousarray(img + rng.uniform(0, 0.005, img.shape).astype(np.float32))

    def _render(self, settings, **kw):
        result, metrics = self._processor().run_pipeline(
            self._negative(), settings, "curves-src", render_size_ref=48.0, prefer_gpu=False, readback_metrics=True, **kw
        )
        return np.asarray(result)[:, :, :3].astype(np.float64), metrics

    def test_curves_move_the_render_and_publish_the_input_histogram(self) -> None:
        base = WorkspaceConfig()
        plain, _ = self._render(base)
        curved_cfg = replace(base, exposure=replace(base.exposure, curve_2=30.0, curve_5=-30.0))
        curved, metrics = self._render(curved_cfg)
        self.assertGreater(float(np.abs(curved - plain).max()), 1e-4)
        hist = metrics.get("curves_input_histogram")
        self.assertIsNotNone(hist)
        assert hist is not None
        self.assertEqual(np.asarray(hist).shape, (4, 256))
        # Curves alone publish no levels histogram.
        self.assertNotIn("levels_input_histogram", metrics)

    def test_no_histogram_keys_without_either_stage(self) -> None:
        _, metrics = self._render(WorkspaceConfig())
        self.assertNotIn("curves_input_histogram", metrics)
        self.assertNotIn("levels_input_histogram", metrics)

    def test_flat_master_bypasses_curves(self) -> None:
        flat = flat_master_config(WorkspaceConfig())
        plain, _ = self._render(flat)
        with_curves, _ = self._render(replace(flat, exposure=replace(flat.exposure, curve_2=60.0)))
        np.testing.assert_array_equal(with_curves, plain)


@unittest.skipUnless(GPUDevice.get().is_available, "GPU not available")
class TestGpuCurvesParity(unittest.TestCase):
    def _render(self, processor, settings, img, prefer_gpu):
        result, _ = processor.run_pipeline(
            img, settings, "curves-parity-src", render_size_ref=float(max(img.shape[:2])), prefer_gpu=prefer_gpu, readback_metrics=False
        )
        if hasattr(result, "readback"):
            arr = np.asarray(result.readback())[:, :, :3]
        else:
            arr = np.asarray(result)[:, :, :3]
        return arr.astype(np.float64)

    def test_cpu_gpu_match_curves(self) -> None:
        from negpy.services.rendering.image_processor import ImageProcessor

        processor = ImageProcessor()
        if processor.engine_gpu is None:
            self.skipTest("GPU engine not initialised")

        rng = np.random.default_rng(3)
        h, w = 64, 64
        grad = np.linspace(0.05, 0.9, w, dtype=np.float32)
        img = np.repeat(grad[None, :], h, axis=0)
        img = np.stack([img, img * 0.9, img * 0.8], axis=-1)
        img = np.ascontiguousarray(img + rng.uniform(0, 0.01, img.shape).astype(np.float32))

        base = WorkspaceConfig()
        settings = replace(
            base,
            exposure=replace(
                base.exposure,
                curve_1=25.0,
                curve_5=-30.0,
                curve_2_red=40.0,
                curve_6_blue=-45.0,
                levels_in_low=10,
                levels_gamma=1.2,
            ),
        )
        cpu = self._render(processor, settings, img, prefer_gpu=False)
        gpu = self._render(processor, settings, img, prefer_gpu=True)

        self.assertEqual(cpu.shape, gpu.shape)
        mad = float(np.mean(np.abs(cpu - gpu)))
        mx = float(np.max(np.abs(cpu - gpu)))
        self.assertLess(mad, 0.01, f"mean abs diff {mad:.4f}")
        self.assertLess(mx, 0.04, f"max abs diff {mx:.4f}")


if __name__ == "__main__":
    unittest.main()
