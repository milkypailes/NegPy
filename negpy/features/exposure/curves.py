"""Display-referred curves, GIMP-style, applied after levels as the last step.

Each of the four channels (the Global master, then Red, Green, Blue) holds a
fixed grid of control points as offsets from the identity line, in code values.
The offsets bake through a monotone cubic (Fritsch-Carlson PCHIP) into a
256-entry table, and both engines apply the table by integer indexing, so CPU
and GPU agree by construction rather than by mirrored math. A fixed grid keeps
the config bounded and every node draggable vertically only, with its
neighbours clamping the drag so the curve can never fold.
"""

from typing import Any

import numpy as np

from negpy.domain.types import ImageBuffer
from negpy.kernel.image.validation import ensure_image

CURVE_NODES = 8
CURVE_OFFSET_MIN = -255.0
CURVE_OFFSET_MAX = 255.0

# Fixed node inputs, evenly spaced over the code-value range.
CURVE_INPUTS = tuple(i * 255.0 / (CURVE_NODES - 1) for i in range(CURVE_NODES))

# Default node x, the grid rounded to code values.
CURVE_DEFAULT_X = tuple(int(round(v)) for v in CURVE_INPUTS)


def _node_fields(channel: str) -> tuple[str, ...]:
    """Offset field names for `channel`, node 0 (shadows) to node N (highlights)."""
    suffix = "" if channel == "global" else f"_{channel}"
    return tuple(f"curve_{i}{suffix}" for i in range(CURVE_NODES))


def _node_x_fields(channel: str) -> tuple[str, ...]:
    """Position field names for `channel`, in code values, same node order."""
    suffix = "" if channel == "global" else f"_{channel}"
    return tuple(f"curve_x_{i}{suffix}" for i in range(CURVE_NODES))


def curves_fields() -> tuple[str, ...]:
    """Every ExposureConfig field curves owns: offsets then positions per channel."""
    from negpy.features.exposure.levels import LEVELS_CHANNELS

    return tuple(f for ch in LEVELS_CHANNELS for f in (*_node_fields(ch), *_node_x_fields(ch)))


def curves_defaults() -> dict[str, Any]:
    """Identity value per curves field: zero offsets on the default grid."""
    defaults: dict[str, Any] = {}
    from negpy.features.exposure.levels import LEVELS_CHANNELS

    for ch in LEVELS_CHANNELS:
        defaults.update({f: 0.0 for f in _node_fields(ch)})
        defaults.update({f: x for f, x in zip(_node_x_fields(ch), CURVE_DEFAULT_X)})
    return defaults


def channel_offsets(config: Any, channel: str) -> tuple[float, ...]:
    """Clamped offsets for `channel`, in code values."""
    return tuple(float(min(max(float(getattr(config, f)), CURVE_OFFSET_MIN), CURVE_OFFSET_MAX)) for f in _node_fields(channel))


def node_positions(config: Any, channel: str) -> tuple[int, ...]:
    """Clamped node x for `channel`, in code values."""
    return tuple(int(min(max(round(float(getattr(config, f))), 0), 255)) for f in _node_x_fields(channel))


def curves_active(config: Any) -> bool:
    """True when any node leaves the identity line."""
    return any(o != 0.0 for ch in ("global", "red", "green", "blue") for o in channel_offsets(config, ch))


def active_mask(config: Any) -> tuple[int, int, int, int]:
    """Per-lane apply flags over (global, red, green, blue). An identity table
    still quantizes to 8 bits, so untouched lanes are skipped on both engines
    and stay bit-exact. Single source for the CPU application and the GPU pack."""
    return tuple(1 if any(channel_offsets(config, ch)) else 0 for ch in ("global", "red", "green", "blue"))


def without_curves(config: Any) -> Any:
    """`config` with every curves field at its default, so a curves-only edit skips
    the exposure cache like a levels-only one does (compose with without_levels)."""
    from dataclasses import replace

    try:
        return replace(config, **curves_defaults())
    except TypeError:
        return config


def _pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fritsch-Carlson monotone cubic slopes: the interpolant through
    non-decreasing data never overshoots. No scipy dependency by design."""
    n = len(x)
    if n < 2:
        return np.zeros(max(n, 1), dtype=np.float64)
    h = np.diff(x)
    delta = np.diff(y) / h
    if n == 2:
        return np.array([delta[0], delta[0]], dtype=np.float64)
    m = np.empty(n, dtype=np.float64)
    m[0], m[-1] = delta[0], delta[-1]
    for k in range(1, n - 1):
        if delta[k - 1] == 0.0 or delta[k] == 0.0 or (delta[k - 1] < 0.0) != (delta[k] < 0.0):
            m[k] = 0.0
        else:
            w1 = 2.0 * h[k] + h[k - 1]
            w2 = h[k] + 2.0 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])
    for end in (0, -1):
        d0, d1 = (delta[0], delta[1]) if end == 0 else (delta[-1], delta[-2])
        h0, h1 = (h[0], h[1]) if end == 0 else (h[-1], h[-2])
        edge = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if (edge < 0.0) != (d0 < 0.0):
            edge = 0.0
        elif abs(edge) > 3.0 * abs(d0):
            edge = 3.0 * d0
        m[end] = edge
    return m


def bake_channel_lut(offsets: Any, x_positions: Any = None) -> np.ndarray:
    """(256,) float32 output code values for node `offsets` at `x_positions`
    (default the fixed grid): identity plus the offset, ordered and held
    monotone, through PCHIP, sampled per code value. A degenerate grid falls
    back to identity rather than dividing by zero."""
    if x_positions is None:
        x_positions = CURVE_INPUTS
    xs = np.asarray(list(x_positions), dtype=np.float64)
    ys = np.clip(xs + np.asarray(list(offsets), dtype=np.float64), 0.0, 255.0)
    order = np.argsort(xs, kind="stable")
    xs, ys = xs[order], np.maximum.accumulate(ys[order])
    _, first = np.unique(xs, return_index=True)
    xs, ys = xs[np.sort(first)], ys[np.sort(first)]
    if len(xs) < 2:
        return (np.arange(256, dtype=np.float32) / 255.0)
    slopes = _pchip_slopes(xs, ys)
    xe = np.arange(256, dtype=np.float64)
    k = np.clip(np.searchsorted(xs, xe, side="right") - 1, 0, len(xs) - 2)
    h = xs[k + 1] - xs[k]
    t = (xe - xs[k]) / h
    t2, t3 = t * t, t * t * t
    y = (
        (2.0 * t3 - 3.0 * t2 + 1.0) * ys[k]
        + (t3 - 2.0 * t2 + t) * h * slopes[k]
        + (-2.0 * t3 + 3.0 * t2) * ys[k + 1]
        + (t3 - t2) * h * slopes[k + 1]
    )
    return np.clip(y / 255.0, 0.0, 1.0).astype(np.float32)


def bake_config_luts(config: Any) -> np.ndarray:
    """(4, 256) float32 tables over (global, red, green, blue). Single source for
    the CPU application and the GPU storage-buffer upload."""
    from negpy.features.exposure.levels import LEVELS_CHANNELS

    return np.stack([bake_channel_lut(channel_offsets(config, ch), node_positions(config, ch)) for ch in LEVELS_CHANNELS])


def node_points(config: Any, channel: str) -> tuple[tuple[int, int], ...]:
    """The 8 (x, y) node positions the panel draws, in code values: what the
    bake holds, so the markers never disagree with the render."""
    xs = np.asarray(node_positions(config, channel), dtype=np.float64)
    ys = np.clip(xs + np.asarray(channel_offsets(config, channel)), 0.0, 255.0)
    order = np.argsort(xs, kind="stable")
    xs, ys = xs[order], np.maximum.accumulate(ys[order])
    return tuple((int(x), int(round(y))) for x, y in zip(xs, ys))


def _apply_lut_plane(plane: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Integer-index table application in float32, mirroring the WGSL lookup:
    `u32(clamp(x * 255 + 0.5, 0, 255))`."""
    idx = np.clip((plane * np.float32(255.0) + np.float32(0.5)).astype(np.int32), 0, 255)
    return lut[idx]


def apply_curves(image: ImageBuffer, config: Any) -> ImageBuffer:
    """Per-channel curves on a display-encoded buffer, Global master first, then
    each channel's own table. Identity (every offset 0) returns the input
    untouched, since even an identity table would quantize to 8 bits."""
    if not curves_active(config):
        return image
    mask = active_mask(config)
    tables = bake_config_luts(config)
    arr = np.asarray(image, dtype=np.float32)
    mono = arr.ndim == 2
    res = arr[..., None].copy() if mono else arr.copy()
    if mask[0]:
        for c in range(res.shape[-1]):
            res[..., c] = _apply_lut_plane(res[..., c], tables[0])
    for c in range(min(3, res.shape[-1])):
        if mask[c + 1]:
            res[..., c] = _apply_lut_plane(res[..., c], tables[c + 1])
    res = np.clip(res, 0.0, 1.0)
    if mono:
        res = res[..., 0]
    return ensure_image(res.astype(np.float32))
