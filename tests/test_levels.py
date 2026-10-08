"""Display-referred levels (GIMP-style), applied last in the pipeline.

The math lives in two places — ``levels.py`` (CPU) and ``shaders/levels.wgsl``
(GPU). They must agree, or GPU previews drift from CPU exports.
"""

import unittest
from dataclasses import replace

import numpy as np

from negpy.domain.models import WorkspaceConfig, flat_master_config
from negpy.features.exposure.levels import apply_levels, channel_levels, levels_active, levels_fields, uniform_rows
from negpy.features.exposure.models import ExposureConfig
from negpy.infrastructure.gpu.device import GPUDevice


def _gray(*values: float) -> np.ndarray:
    return np.repeat(np.asarray(values, dtype=np.float32)[None, :, None], 3, axis=2)


class TestLevelsMath(unittest.TestCase):
    def test_identity_by_default(self) -> None:
        conf = ExposureConfig()
        self.assertFalse(levels_active(conf))
        img = np.linspace(0.0, 1.0, 300, dtype=np.float32).reshape(10, 10, 3)
        np.testing.assert_array_equal(np.asarray(apply_levels(img, conf)), img)

    def test_every_channel_defaults_to_identity(self) -> None:
        for field in levels_fields():
            default = getattr(ExposureConfig(), field)
            if "gamma" in field:
                self.assertEqual(default, 1.0)
            elif "high" in field:
                self.assertEqual(default, 255)
            else:
                self.assertEqual(default, 0)

    def test_input_window_stretches(self) -> None:
        conf = replace(ExposureConfig(), levels_in_low=51, levels_in_high=204)
        out = np.asarray(apply_levels(_gray(0.2, 0.5, 0.8), conf))[0, :, 0]
        np.testing.assert_allclose(out, [0.0, 0.5, 1.0], atol=1e-6)

    def test_gamma_above_one_holds_highlights(self) -> None:
        conf = replace(ExposureConfig(), levels_gamma=2.0)
        out = float(np.asarray(apply_levels(_gray(0.25), conf))[0, 0, 0])
        self.assertAlmostEqual(out, 0.5, places=5)

    def test_gamma_below_one_deepens_midtones(self) -> None:
        conf = replace(ExposureConfig(), levels_gamma=0.5)
        out = float(np.asarray(apply_levels(_gray(0.25), conf))[0, 0, 0])
        self.assertAlmostEqual(out, 0.0625, places=5)

    def test_output_window_compresses(self) -> None:
        conf = replace(ExposureConfig(), levels_out_low=51, levels_out_high=204)
        out = np.asarray(apply_levels(_gray(0.0, 0.5, 1.0), conf))[0, :, 0]
        np.testing.assert_allclose(out, [0.2, 0.5, 0.8], atol=1e-6)

    def test_degenerate_window_thresholds(self) -> None:
        conf = replace(ExposureConfig(), levels_in_low=128, levels_in_high=128)
        out = np.asarray(apply_levels(_gray(0.2, 0.9), conf))[0, :, 0]
        np.testing.assert_allclose(out, [0.0, 1.0], atol=1e-6)

    def test_per_channel_leaves_other_channels(self) -> None:
        conf = replace(ExposureConfig(), levels_in_low_red=128)
        img = np.full((2, 2, 3), 0.6, dtype=np.float32)
        out = np.asarray(apply_levels(img, conf))
        self.assertLess(float(out[0, 0, 0]), 0.6)
        np.testing.assert_allclose(out[..., 1:], 0.6, atol=1e-6)

    def test_master_applies_before_the_channel(self) -> None:
        conf = replace(ExposureConfig(), levels_in_low=51, levels_in_low_red=51)
        img = np.full((2, 2, 3), 0.6, dtype=np.float32)
        master_only = np.asarray(apply_levels(img, replace(ExposureConfig(), levels_in_low=51)))
        both = np.asarray(apply_levels(img, conf))
        # The red channel passes two windows, green/blue only the master.
        np.testing.assert_allclose(both[..., 1:], master_only[..., 1:], atol=1e-6)
        self.assertLess(float(both[0, 0, 0]), float(master_only[0, 0, 0]))

    def test_config_values_are_clamped(self) -> None:
        conf = replace(
            ExposureConfig(),
            levels_in_low=999,
            levels_gamma=0.0,
            levels_in_high=-99,
            levels_out_low=-5,
            levels_out_high=9999,
        )
        lo, gamma, hi, olo, ohi = channel_levels(conf, "value")
        self.assertEqual((lo, hi, olo, ohi), (255.0, 0.0, 0.0, 255.0))
        self.assertEqual(gamma, 0.1)
        out = np.asarray(apply_levels(_gray(0.0, 0.5, 1.0), conf))
        self.assertTrue(np.all(out >= 0.0) and np.all(out <= 1.0))

    def test_uniform_rows_are_normalized(self) -> None:
        conf = replace(ExposureConfig(), levels_in_low=51, levels_gamma=2.0, levels_in_high_red=128)
        lo, hi, inv, olo, ospan = uniform_rows(conf)
        self.assertEqual(len((lo, hi, inv, olo, ospan)), 5)
        self.assertAlmostEqual(lo[0], 51 / 255)
        self.assertAlmostEqual(inv[0], 0.5)
        self.assertAlmostEqual(hi[1], 128 / 255)
        self.assertAlmostEqual(inv[1], 1.0)


class TestLevelsPipeline(unittest.TestCase):
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
            self._negative(), settings, "levels-src", render_size_ref=48.0, prefer_gpu=False, readback_metrics=True, **kw
        )
        return np.asarray(result)[:, :, :3].astype(np.float64), metrics

    def test_levels_move_the_render_and_publish_the_input_histogram(self) -> None:
        base = WorkspaceConfig()
        plain, _ = self._render(base)
        leveled_cfg = replace(
            base,
            exposure=replace(base.exposure, levels_in_low=20, levels_gamma=1.2, levels_in_high=235),
        )
        leveled, metrics = self._render(leveled_cfg)
        self.assertGreater(float(np.abs(leveled - plain).max()), 1e-4)
        hist = metrics.get("levels_input_histogram")
        self.assertIsNotNone(hist)
        assert hist is not None
        self.assertEqual(np.asarray(hist).shape, (4, 256))

    def test_no_histogram_key_without_levels(self) -> None:
        _, metrics = self._render(WorkspaceConfig())
        self.assertNotIn("levels_input_histogram", metrics)

    def test_flat_master_bypasses_levels(self) -> None:
        flat = flat_master_config(WorkspaceConfig())
        plain, _ = self._render(flat)
        with_levels, _ = self._render(replace(flat, exposure=replace(flat.exposure, levels_in_low=40, levels_gamma=2.0)))
        np.testing.assert_array_equal(with_levels, plain)


@unittest.skipUnless(GPUDevice.get().is_available, "GPU not available")
class TestGpuLevelsParity(unittest.TestCase):
    def _render(self, processor, settings, img, prefer_gpu):
        result, _ = processor.run_pipeline(
            img, settings, "levels-parity-src", render_size_ref=float(max(img.shape[:2])), prefer_gpu=prefer_gpu, readback_metrics=False
        )
        if hasattr(result, "readback"):
            arr = np.asarray(result.readback())[:, :, :3]
        else:
            arr = np.asarray(result)[:, :, :3]
        return arr.astype(np.float64)

    def test_cpu_gpu_match_levels(self) -> None:
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
                levels_in_low=15,
                levels_gamma=1.4,
                levels_in_high=240,
                levels_out_low=5,
                levels_out_high=250,
                levels_in_low_red=30,
                levels_gamma_blue=0.7,
                levels_out_high_green=230,
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
