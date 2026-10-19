import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.features.exposure.normalization import resolve_analysis_region
from negpy.features.exposure.processor import NormalizationProcessor
from negpy.features.process.models import ProcessConfig, ProcessMode


def test_resolve_falls_back_to_roi_and_buffer_when_rect_none():
    roi, buf = resolve_analysis_region((100, 200), (10, 90, 20, 180), 0.05, None)
    assert roi == (10, 90, 20, 180)
    assert buf == 0.05


def test_resolve_rect_overrides_roi_and_zeroes_buffer():
    # Rect is normalized in the transformed image (H=100, W=200).
    roi, buf = resolve_analysis_region((100, 200), (10, 90, 20, 180), 0.05, (0.25, 0.5, 0.75, 1.0))
    assert roi == (50, 100, 50, 150)
    assert buf == 0.0


def test_resolve_ignores_degenerate_rect():
    # Zero-area rect can't blank analysis: fall back to the ROI + buffer.
    roi, buf = resolve_analysis_region((100, 100), None, 0.1, (0.5, 0.5, 0.5, 0.5))
    assert roi is None
    assert buf == 0.1


def _split_negative() -> np.ndarray:
    # Linear "negative": a dense (dark) left half and a thin (bright) right half.
    img = np.empty((200, 200, 3), dtype=np.float32)
    img[:, :100] = 0.15
    img[:, 100:] = 0.85
    return img


def _bounds_for_rect(rect):
    img = _split_negative()
    cfg = ProcessConfig(process_mode=ProcessMode.C41, analysis_buffer=0.0, analysis_rect=rect)
    ctx = PipelineContext(scale_factor=1.0, original_size=(200, 200))
    NormalizationProcessor(cfg).process(img, ctx)
    return ctx.metrics["log_bounds"]


def test_analysis_rect_changes_metered_bounds():
    # Metering only the dense (dark) half vs only the thin (bright) half must yield
    # different normalization bounds — proof the rect is wired through the meters.
    dark = _bounds_for_rect((0.0, 0.0, 0.5, 1.0))
    bright = _bounds_for_rect((0.5, 0.0, 1.0, 1.0))
    assert not np.allclose(dark.floors, bright.floors) or not np.allclose(dark.ceils, bright.ceils)


def test_resolve_reads_the_meter_rect_when_nothing_is_cropped():
    roi, buf = resolve_analysis_region((100, 200), None, 0.05, None, (0.25, 0.1, 0.75, 0.9))
    assert roi == (10, 90, 50, 150)
    assert buf == 0.05


def test_resolve_crop_and_freehand_region_outrank_the_meter_rect():
    meter = (0.25, 0.1, 0.75, 0.9)
    assert resolve_analysis_region((100, 200), (5, 95, 5, 195), 0.05, None, meter) == ((5, 95, 5, 195), 0.05)
    assert resolve_analysis_region((100, 200), None, 0.05, (0.0, 0.0, 0.5, 1.0), meter) == ((0, 100, 0, 100), 0.0)


# The picture box inside the opaque holder of _holder_frame, normalized.
_PICTURE = (0.2, 0.17, 0.8, 0.83)


def _holder_frame() -> np.ndarray:
    """A negative in an opaque holder: the holder reads denser than any film."""
    img = np.full((300, 450, 3), 3e-4, dtype=np.float32)
    yy, xx = np.mgrid[0:200, 0:270]
    t = 0.15 + 0.5 * xx / 270.0 + 0.15 * np.sin(yy / 12.0)
    img[50:250, 90:360] = np.stack([0.7 * 10 ** (-1.2 * t), 0.45 * 10 ** (-1.3 * t), 0.25 * 10 ** (-1.4 * t)], -1)
    return img


def _holder_bounds(ctx=None, **process):
    ctx = ctx or PipelineContext(scale_factor=1.0, original_size=(300, 450))
    cfg = ProcessConfig(process_mode=ProcessMode.C41, analysis_buffer=0.0, **process)
    NormalizationProcessor(cfg).process(_holder_frame(), ctx)
    return ctx.metrics["log_bounds"]


def test_meter_rect_keeps_the_holder_out_of_the_bounds():
    metered = _holder_bounds(meter_rect=_PICTURE)
    picture_only = _holder_bounds(analysis_rect=_PICTURE)
    whole_frame = _holder_bounds()
    np.testing.assert_allclose(metered.floors, picture_only.floors, atol=1e-6)
    np.testing.assert_allclose(metered.ceils, picture_only.ceils, atol=1e-6)
    assert min(whole_frame.floors) < min(metered.floors) - 0.5


def test_a_new_meter_rect_re_meters_bounds_cached_in_the_metrics():
    ctx = PipelineContext(scale_factor=1.0, original_size=(300, 450))
    _holder_bounds(ctx)
    cached = _holder_bounds(ctx, meter_rect=_PICTURE)
    fresh = _holder_bounds(meter_rect=_PICTURE)
    np.testing.assert_allclose(cached.floors, fresh.floors, atol=1e-6)
