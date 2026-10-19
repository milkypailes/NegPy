from dataclasses import replace

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.features.geometry.logic import autocrop_detection_key, meter_detection_key, resolve_meter_rect
from negpy.features.geometry.models import GeometryConfig, crops_frame
from negpy.services.rendering.image_processor import MeterBox, _resolve_meter_rect

_STORED = (-2.0, -2.1, -2.2), (-0.5, -0.6, -0.7)


def _negative(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pic = np.stack([0.70 * 10 ** (-1.2 * t), 0.45 * 10 ** (-1.3 * t), 0.25 * 10 ** (-1.4 * t)], -1)
    return (pic * rng.uniform(0.95, 1.05, pic.shape)).astype(np.float32)


def _frame(in_holder: bool) -> np.ndarray:
    """A C-41 negative, either in an opaque holder with a clear rebate ring or filling the frame."""
    rng = np.random.default_rng(0)
    h, w = 600, 900
    img = np.full((h, w, 3), 3e-4, np.float32)
    y0, y1, x0, x1 = (80, 520, 150, 750) if in_holder else (0, h, 0, w)
    img[y0:y1, x0:x1] = (0.70, 0.45, 0.25)
    inset = 18 if in_holder else 0
    yy, xx = np.mgrid[y0 + inset : y1 - inset, x0 + inset : x1 - inset]
    t = 0.15 + 0.5 * (xx - xx.min()) / np.ptp(xx) + 0.15 * np.sin(yy / 23.0)
    img[y0 + inset : y1 - inset, x0 + inset : x1 - inset] = _negative(t, rng)
    return img


def _settings(**process) -> WorkspaceConfig:
    base = WorkspaceConfig()
    return replace(base, process=replace(base.process, local_floors=_STORED[0], local_ceils=_STORED[1], **process))


def test_the_meter_rect_is_the_picture_inside_a_holder():
    x1, y1, x2, y2 = resolve_meter_rect(_frame(True), GeometryConfig(), 1600.0)
    assert 150 / 900 <= x1 < x2 <= 750 / 900
    assert 80 / 600 <= y1 < y2 <= 520 / 600
    assert (x2 - x1) * (y2 - y1) > 0.6 * (564 / 900) * (404 / 600)


def test_a_frame_the_film_fills_has_no_meter_rect():
    assert resolve_meter_rect(_frame(False), GeometryConfig(), 1600.0) is None


def test_the_meter_key_follows_the_geometry_and_the_source_shape():
    g = GeometryConfig()
    assert meter_detection_key(g, (600, 900)) != meter_detection_key(replace(g, rotation=1), (600, 900))
    assert meter_detection_key(g, (600, 900)) != meter_detection_key(g, (600, 450))
    assert meter_detection_key(g, (600, 900)) == meter_detection_key(g, (1200, 1800, 3))
    assert meter_detection_key(g, (600, 900)).startswith(autocrop_detection_key(g))


def test_crops_frame_matches_the_branches_that_set_a_crop_roi():
    g = GeometryConfig()
    assert not crops_frame(g)
    assert crops_frame(replace(g, crop_rect=(0.1, 0.1, 0.9, 0.9)))
    assert crops_frame(replace(g, crop_to_valid=True))
    assert not crops_frame(replace(g, crop_to_valid=True, crop_from_auto=True))
    assert crops_frame(replace(g, autocrop_offset=4))


def test_a_frame_with_no_crop_resolves_its_meter_rect_and_re_meters():
    img = _frame(True)
    settings, box = _resolve_meter_rect(img, _settings())
    key = meter_detection_key(GeometryConfig(), img.shape)
    assert box == MeterBox(settings.process.meter_rect, key, autocrop_detection_key(settings.geometry))
    assert settings.process.meter_rect is not None and settings.process.meter_key == key
    assert not settings.process.is_local_initialized


def test_a_resolved_meter_rect_is_not_detected_again():
    img = _frame(True)
    resolved, _ = _resolve_meter_rect(img, _settings())
    again, box = _resolve_meter_rect(img, resolved)
    assert box is None and again is resolved


def test_a_cropped_frame_or_a_freehand_region_needs_no_meter_rect():
    img = _frame(True)
    cropped = replace(_settings(), geometry=replace(GeometryConfig(), crop_rect=(0.2, 0.2, 0.8, 0.8)))
    assert _resolve_meter_rect(img, cropped) == (cropped, None)
    freehand = _settings(analysis_rect=(0.3, 0.3, 0.7, 0.7))
    assert _resolve_meter_rect(img, freehand) == (freehand, None)
    armed = replace(_settings(), geometry=replace(GeometryConfig(), crop_from_auto=True))
    assert _resolve_meter_rect(img, armed) == (armed, None)


def test_a_full_bleed_frame_keeps_its_bounds():
    img = _frame(False)
    settings, box = _resolve_meter_rect(img, _settings())
    assert box is not None and box.rect is None
    assert (settings.process.local_floors, settings.process.local_ceils) == _STORED


def test_locked_bounds_survive_a_new_meter_rect():
    settings, _ = _resolve_meter_rect(_frame(True), _settings(lock_bounds=True))
    assert settings.process.meter_rect is not None
    assert (settings.process.local_floors, settings.process.local_ceils) == _STORED


def test_a_holder_frame_with_no_crop_prints_like_its_auto_crop(tmp_path):
    """The holder is denser than any film: metered, it takes the print's white point and the picture prints dark."""
    import cv2
    import tifffile

    from negpy.features.geometry.models import with_auto_crop
    from negpy.services.rendering.image_processor import ImageProcessor

    path = str(tmp_path / "holder.tif")
    tifffile.imwrite(path, (np.clip(_frame(True), 0, 1) * 65535).astype(np.uint16), photometric="rgb")
    proc = ImageProcessor()

    def picture_lightness(cfg: WorkspaceConfig, box=(0.0, 0.0, 1.0, 1.0)) -> float:
        """Mean L* of the middle of `box` (normalized in the render), the same film in either render."""
        out = proc.render_display_array(path, cfg, path, 900, prefer_gpu=False)
        lab = cv2.cvtColor(out.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
        h, w = lab.shape[:2]
        x1, y1, x2, y2 = box
        ys = slice(int((y1 + 0.4 * (y2 - y1)) * h), int((y1 + 0.6 * (y2 - y1)) * h))
        xs = slice(int((x1 + 0.4 * (x2 - x1)) * w), int((x1 + 0.6 * (x2 - x1)) * w))
        return float(lab[ys, xs, 0].mean())

    picture = resolve_meter_rect(_frame(True), GeometryConfig(), 1600.0)
    no_crop = picture_lightness(WorkspaceConfig(), picture)
    auto_crop = picture_lightness(with_auto_crop(WorkspaceConfig(), True))
    assert abs(no_crop - auto_crop) < 1.0


def test_a_source_is_detected_once_across_renders_that_never_froze(monkeypatch):
    """A drag before the freeze lands, or a caller with no controller, renders the same unfrozen edit again."""
    import negpy.services.rendering.image_processor as ip

    calls = []
    real = ip.resolve_meter_rect
    monkeypatch.setattr(ip, "resolve_meter_rect", lambda *a, **k: calls.append(1) or real(*a, **k))
    proc = ip.ImageProcessor()
    img = _frame(True)
    for density in (1.0, 1.2, 1.4):
        cfg = WorkspaceConfig()
        cfg = replace(cfg, exposure=replace(cfg.exposure, density=density))
        proc.run_pipeline(img, cfg, "meter-memo", render_size_ref=900.0, prefer_gpu=False)
    assert len(calls) == 1
