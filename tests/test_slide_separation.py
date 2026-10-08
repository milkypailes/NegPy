"""The slide renders live in features/transparency and the negative's print never reaches
into them. Every exposure control declares what it does on a slide, and the declaration
is checked by rendering: a print control added without a slide decision fails here."""

import ast
import dataclasses
import pathlib
from dataclasses import replace

import numpy as np
import pytest

from negpy.domain.interfaces import PipelineContext
from negpy.features.exposure.models import ExposureConfig
from negpy.features.process.models import ProcessMode, cast_removal_for_mode
from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG
from negpy.services.rendering.engine import base_processor, exposure_processor

_TRIMS = ("_trim_red", "_trim_green", "_trim_blue")


def _with_trims(*names: str) -> set:
    return {n + t for n in names for t in ("", *_TRIMS)}


# Drives the transfer curve on a raw slide and on a Positive frame.
TRANSFER = (
    {"density", "grade", "wb_cyan", "wb_magenta", "wb_yellow", "shadow_density", "highlight_density", "cast_removal_strength"}
    | {f"{zone}_{dye}" for zone in ("shadow", "highlight") for dye in ("cyan", "magenta", "yellow")}
    | _with_trims("toe", "shoulder", "toe_width", "shoulder_width", "dye_separation")
    | {"separation_damping"}
    # Display-referred levels run after the output transform on every path, so they
    # move a slide exactly like a print.
    | {
        f"levels_{name}{suffix}"
        for suffix in ("", "_red", "_green", "_blue")
        for name in ("in_low", "gamma", "in_high", "out_low", "out_high")
    }
)
# Meter the frame on a raw slide and on a Positive frame alike; a slide starts with both off.
METERS = {"auto_exposure", "auto_normalize_contrast"}
# The paper model's own controls; the transfer curve has no paper to apply them to.
PRINT_ONLY = (
    {"contrast_mask", "mask_spacer", "paper_dmin", "paper_black", "paper_profile", "preflash"}
    | {n + t for n in ("grade",) for t in _TRIMS}
    | _with_trims("shadow_grade", "highlight_grade", "midtone_gamma")
)
# Chooses the renderer rather than shaping one (process.path.render_path).
ROUTING = {"render_intent"}


def test_every_exposure_control_declares_its_slide_behaviour():
    fields = {f.name for f in dataclasses.fields(ExposureConfig)}
    groups = (TRANSFER, METERS, PRINT_ONLY, ROUTING)
    assert set().union(*groups) == fields, "classify the new field in this module"
    assert sum(len(g) for g in groups) == len(fields), "a field sits in two groups"


def test_the_negative_package_never_imports_the_slide_package():
    root = pathlib.Path(__file__).resolve().parent.parent / "negpy" / "features" / "exposure"
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else []
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            assert not any(n.startswith("negpy.features.transparency") for n in names), path.name


def _moved(field: str, value):
    if isinstance(value, bool):
        return not value
    if field == "paper_profile":
        return "kodak_endura"
    if field.startswith("levels_"):
        # Integer 0-255 bounds: ±25 stays in range and off the clamp for both ends.
        if "gamma" in field:
            return value + 2.0
        return value - 25.0 if "high" in field else value + 25.0
    if "grade" in field:
        return value + 15.0
    if field in ("toe_width", "shoulder_width", "mask_spacer"):
        return value + 1.5
    if field == "preflash":
        return 0.8
    return value + 0.25


def _image() -> np.ndarray:
    rng = np.random.default_rng(3)
    v = np.geomspace(5e-4, 0.9, 64 * 64).astype(np.float32).reshape(64, 64)
    img = np.stack([v, v * 0.82, v * 0.62], axis=-1)
    return np.ascontiguousarray(img + rng.uniform(0, 1e-4, img.shape).astype(np.float32))


def _config(positive: bool):
    """Toe, Shoulder and Dye Separation off neutral, so their widths, trims and damping bite."""
    cfg = DEFAULT_WORKSPACE_CONFIG
    exposure = replace(
        cfg.exposure,
        cast_removal_strength=cast_removal_for_mode(ProcessMode.E6, cfg.exposure.cast_removal_strength),
        toe=0.4,
        shoulder=-0.3,
        dye_separation=1.3,
        auto_exposure=False,
        auto_normalize_contrast=False,
    )
    return replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.E6, positive_source=positive), exposure=exposure)


def _render(cfg) -> np.ndarray:
    """Base + exposure stages, then the engine tail (finish, output transform,
    levels), so end-of-pipeline controls like levels read a delta too."""
    from negpy.features.exposure.levels import apply_levels
    from negpy.features.finish.processor import FinishProcessor
    from negpy.kernel.image.logic import working_oetf_encode

    img = _image()
    ctx = PipelineContext(original_size=img.shape[:2], scale_factor=1.0, process_mode=ProcessMode.E6, wants_uv_grid=False)
    norm = base_processor(cfg).process(img, ctx)
    lin = np.asarray(exposure_processor(cfg).process(norm, ctx), dtype=np.float32)
    finished = np.asarray(FinishProcessor(cfg.finish, cfg.export.export_print_size, (1.0, 1.0, 1.0), None).process(lin, ctx))
    encoded = np.asarray(working_oetf_encode(finished))
    return np.asarray(apply_levels(encoded, cfg.exposure), dtype=np.float64)


def _delta(field: str, positive: bool) -> float:
    cfg = _config(positive)
    moved = replace(cfg, exposure=replace(cfg.exposure, **{field: _moved(field, getattr(cfg.exposure, field))}))
    return float(np.abs(_render(moved) - _render(cfg)).max())


@pytest.mark.parametrize("field", sorted(TRANSFER))
def test_transfer_controls_move_a_slide(field):
    assert _delta(field, positive=False) > 1e-4


@pytest.mark.parametrize("field", sorted(METERS))
def test_meters_move_a_raw_slide_and_a_positive_frame(field):
    assert _delta(field, positive=False) > 1e-4
    assert _delta(field, positive=True) > 1e-4


@pytest.mark.parametrize("field", sorted(PRINT_ONLY))
def test_print_controls_leave_a_slide_alone(field):
    assert _delta(field, positive=False) == 0.0
    assert _delta(field, positive=True) == 0.0
