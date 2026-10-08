"""Display-referred levels, GIMP-style, as the last pipeline step.

Each of the four channels (the Global master, then Red, Green, Blue) carries its
own input low/gamma/high and output low/high. The master applies equally to all
three channels first, then each per-channel curve trims on top. GIMP applies
``pow(t, 1/gamma)``; higher gamma holds more high-level intensities.
"""

from typing import Any

import numpy as np

from negpy.domain.types import ImageBuffer
from negpy.kernel.image.validation import ensure_image

LEVELS_CHANNELS = ("global", "red", "green", "blue")

LEVELS_IN_MIN = 0
LEVELS_IN_MAX = 255
LEVELS_GAMMA_MIN = 0.1
LEVELS_GAMMA_MAX = 10.0
# Share of pixels clipped at each end by Auto Input Levels, as in GIMP.
LEVELS_AUTO_CLIP = 0.006


def _channel_fields(channel: str) -> tuple[str, str, str, str, str]:
    """(in_low, gamma, in_high, out_low, out_high) field names for `channel`."""
    suffix = "" if channel == "global" else f"_{channel}"
    return (
        f"levels_in_low{suffix}",
        f"levels_gamma{suffix}",
        f"levels_in_high{suffix}",
        f"levels_out_low{suffix}",
        f"levels_out_high{suffix}",
    )


def levels_fields() -> tuple[str, ...]:
    """Every ExposureConfig field levels owns, master first, then R/G/B."""
    return tuple(f for ch in LEVELS_CHANNELS for f in _channel_fields(ch))


def channel_levels(config: Any, channel: str) -> tuple[float, float, float, float, float]:
    """Clamped (in_low, gamma, in_high, out_low, out_high) for `channel`."""
    in_low_f, gamma_f, in_high_f, out_low_f, out_high_f = _channel_fields(channel)
    in_low = float(min(max(round(float(getattr(config, in_low_f))), LEVELS_IN_MIN), LEVELS_IN_MAX))
    gamma = float(min(max(float(getattr(config, gamma_f)), LEVELS_GAMMA_MIN), LEVELS_GAMMA_MAX))
    in_high = float(min(max(round(float(getattr(config, in_high_f))), LEVELS_IN_MIN), LEVELS_IN_MAX))
    out_low = float(min(max(round(float(getattr(config, out_low_f))), LEVELS_IN_MIN), LEVELS_IN_MAX))
    out_high = float(min(max(round(float(getattr(config, out_high_f))), LEVELS_IN_MIN), LEVELS_IN_MAX))
    return in_low, gamma, in_high, out_low, out_high


def levels_active(config: Any) -> bool:
    """True when any channel deviates from the identity mapping."""
    for channel in LEVELS_CHANNELS:
        in_low, gamma, in_high, out_low, out_high = channel_levels(config, channel)
        if in_low != 0.0 or in_high != 255.0 or out_low != 0.0 or out_high != 255.0 or gamma != 1.0:
            return True
    return False


def auto_input_window(counts: Any) -> tuple[int, int]:
    """GIMP Auto Input Levels for one 256-bin input histogram: the input bounds
    at the first bin past a 0.6% tail each end. A degenerate frame falls back to
    identity rather than posterizing on a threshold."""
    hist = np.asarray(counts, dtype=np.float64).ravel()
    total = float(hist.sum())
    if total <= 0.0 or hist.shape[0] < 256:
        return 0, 255
    lo, hi = 0, 255
    acc = 0.0
    for i in range(255):
        acc += float(hist[i])
        if abs(acc / total - LEVELS_AUTO_CLIP) < abs((acc + float(hist[i + 1])) / total - LEVELS_AUTO_CLIP):
            lo = i + 1
            break
    acc = 0.0
    for i in range(255, 0, -1):
        acc += float(hist[i])
        if abs(acc / total - LEVELS_AUTO_CLIP) < abs((acc + float(hist[i - 1])) / total - LEVELS_AUTO_CLIP):
            hi = i - 1
            break
    if hi <= lo:
        return 0, 255
    return lo, hi


def auto_channel_levels(counts: Any) -> tuple[int, float, int, int, int]:
    """One channel's Auto Input Levels: gamma 1 and the full output range over
    the auto input window, as GIMP resets the other terms on Auto."""
    lo, hi = auto_input_window(counts)
    return (lo, 1.0, hi, 0, 255)


def without_levels(config: Any) -> Any:
    """`config` with every levels field at its default, so a levels-only edit
    re-runs just the final stage instead of the exposure cache behind it."""
    from dataclasses import replace

    defaults = {}
    for f in levels_fields():
        if "gamma" in f:
            defaults[f] = 1.0
        elif "high" in f:
            defaults[f] = 255
        else:
            defaults[f] = 0
    try:
        return replace(config, **defaults)
    except TypeError:
        return config


def uniform_rows(config: Any) -> tuple[tuple[float, float, float, float], ...]:
    """Five vec4s over (global, red, green, blue): input low/high, inverse gamma,
    output low and output span, all normalized. Single source for the GPU pack."""
    lo, hi, inv, olo, orange = [], [], [], [], []
    for channel in LEVELS_CHANNELS:
        in_low, gamma, in_high, out_low, out_high = channel_levels(config, channel)
        lo.append(in_low / 255.0)
        hi.append(in_high / 255.0)
        inv.append(1.0 / gamma)
        olo.append(out_low / 255.0)
        orange.append((out_high - out_low) / 255.0)
    return (tuple(lo), tuple(hi), tuple(inv), tuple(olo), tuple(orange))


def _remap_plane(
    plane: np.ndarray,
    in_low: float,
    gamma: float,
    in_high: float,
    out_low: float,
    out_high: float,
) -> np.ndarray:
    """One channel's input range through gamma onto the output range, all in 0-255."""
    lo, hi = in_low / 255.0, in_high / 255.0
    if hi > lo:
        t = (plane - lo) / (hi - lo)
    else:
        # Degenerate window: the low marker alone thresholds, as in GIMP, where
        # the unnormalized distance is clamped straight onto [0, 1].
        t = (plane - lo) * 255.0
    t = np.clip(t, 0.0, 1.0)
    if gamma != 1.0:
        t = np.power(t, 1.0 / gamma, dtype=np.float32)
    return (out_low + (out_high - out_low) * t) / 255.0


def apply_levels(image: ImageBuffer, config: Any) -> ImageBuffer:
    """GIMP levels on a display-encoded buffer, master first, then per-channel.

    Runs on float32 directly rather than through an OpenCV LUT: the pipeline is
    float throughout and a 256-entry uint8 table would quantize the mapping.
    Identity (every channel at defaults) returns the input untouched.
    """
    if not levels_active(config):
        return image
    arr = np.asarray(image, dtype=np.float32)
    mono = arr.ndim == 2
    img = arr[..., None] if mono else arr
    res = img.astype(np.float32, copy=True)
    master = channel_levels(config, "global")
    if master != (0.0, 1.0, 255.0, 0.0, 255.0):
        for c in range(res.shape[-1]):
            res[..., c] = _remap_plane(res[..., c], *master)
    for c, channel in enumerate(("red", "green", "blue")):
        if c >= res.shape[-1]:
            break
        params = channel_levels(config, channel)
        if params != (0.0, 1.0, 255.0, 0.0, 255.0):
            res[..., c] = _remap_plane(res[..., c], *params)
    res = np.clip(res, 0.0, 1.0)
    if mono:
        res = res[..., 0]
    return ensure_image(res.astype(np.float32))
