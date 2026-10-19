import gc
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from negpy.domain.models import WorkspaceConfig
from negpy.kernel.system.i18n import tr
from negpy.services.rendering.image_processor import ImageProcessor


@dataclass(frozen=True)
class DustScanTask:
    """One full-resolution optical dust scan.

    ``config`` is the edit at dispatch: decode params plus the dust thresholds, size
    and exclusion strokes the scan runs with. ``key`` identifies the frame the scan
    left (content hash + fork suffix); the controller drops results that arrive after
    the user moved on. ``half`` is the active half-frame slice, if any. ``polarity``
    is print-domain ("print_bright", "print_dark", "both"). ``region`` is an optional
    raw-normalized rect covered instead of the whole image; coordinates follow the
    half slice, like the strokes they produce.
    """

    file_path: str
    config: WorkspaceConfig
    key: str
    half: Optional[tuple] = None
    polarity: str = "both"
    region: Optional[tuple] = None


class DustScanWorker(QObject):
    """Optical dust detection on the full-resolution decode, off the UI thread.

    The interactive render detects at preview scale, where a sub-pixel speck dilutes
    below the grain. This runs the optical gates at 1:1 — the dark-side gate for
    scanner dust and the bright-side gate for unexposed dust the camera never lit —
    and returns manual heal strokes, so each hit stays individually
    deletable. No config is written here; the controller appends the strokes.
    """

    started = pyqtSignal()
    finished = pyqtSignal(list, int, str)  # (heal strokes, dropped phantoms, task key)
    error = pyqtSignal(str, str)  # (message, task key)

    def __init__(self) -> None:
        super().__init__()
        self._processor = ImageProcessor(use_gpu=False)

    @pyqtSlot(object)
    def run(self, task: DustScanTask) -> None:
        from negpy.features.retouch.logic import (
            compute_dust_stats,
            detect_bright_luma_score,
            detect_luma_score,
            detection_margin,
            drop_exclusions,
            dust_detections_to_heal_strokes,
            filter_strokes_by_score,
            region_to_pixels,
            strokes_to_score,
        )

        try:
            self.started.emit()
            ret = task.config.retouch
            f32, _ir, _cs = self._processor._load_source_f32(task.file_path, task.config)
            if task.half is not None:
                half, split_x, crop_rect, gutter, axis = task.half
                if half:
                    f32, _ir = ImageProcessor._slice_half_source(f32, _ir, half, split_x, crop_rect, gutter, axis)
            h, w = f32.shape[:2]
            # A region scans a crop alone, padded past the detection windows so its rim reads
            # the film the full frame would, then pastes the verdict back onto the full frame:
            # conversion and verification stay in raw-frame coordinates, and a defect
            # straddling the rim keeps the repair on the side inside. The full frame is the
            # reference, so the crop's windows and normalization match a whole-image scan.
            detect_img = f32
            reference = None
            region = None
            if task.region is not None:
                bounds = region_to_pixels(task.region, f32.shape[:2])
                if bounds is not None:
                    x0, y0, x1, y1 = bounds
                    margin = detection_margin(f32.shape[:2], int(ret.dust_size))
                    px0, py0 = max(0, x0 - margin), max(0, y0 - margin)
                    px1, py1 = min(w, x1 + margin), min(h, y1 + margin)
                    detect_img = np.ascontiguousarray(f32[py0:py1, px0:px1])
                    reference = f32
                    region = (x0, y0, x1, y1, px0, py0)
            stats = compute_dust_stats(detect_img, int(ret.dust_size), reference=reference)
            dark, hair = None, None
            if task.polarity in ("print_bright", "both"):
                dark, hair = detect_luma_score(
                    detect_img,
                    float(ret.dust_threshold),
                    int(ret.dust_size),
                    stats=stats,
                    hair_threshold=float(ret.dust_hair_threshold),
                    reference=reference,
                )
            bright = None
            bright_hair = None
            if task.polarity in ("print_dark", "both"):
                bright, bright_hair = detect_bright_luma_score(
                    detect_img,
                    float(ret.dust_threshold),
                    int(ret.dust_size),
                    hair_threshold=float(ret.dust_hair_threshold),
                    reference=reference,
                )
            sides = [s for s in (dark, bright) if s is not None]
            score = sides[0] if len(sides) == 1 else (np.minimum(*sides).astype(np.float32) if sides else None)
            for side_hair in (hair, bright_hair):
                if side_hair is None:
                    continue
                hair = side_hair if hair is None else np.maximum(hair, np.asarray(side_hair))
            if region is not None:
                x0, y0, x1, y1, px0, py0 = region
                sy0, sx0 = y0 - py0, x0 - px0
                if score is not None:
                    score = np.ascontiguousarray(score[sy0 : sy0 + (y1 - y0), sx0 : sx0 + (x1 - x0)])
                if hair is not None:
                    hair = np.ascontiguousarray(np.asarray(hair)[sy0 : sy0 + (y1 - y0), sx0 : sx0 + (x1 - x0)])
                full = np.ones((h, w), dtype=np.float32)
                if score is not None:
                    full[y0:y1, x0:x1] = score
                score = full
                if hair is not None:
                    plane = np.zeros((h, w), dtype=np.uint8)
                    plane[y0:y1, x0:x1] = (np.asarray(hair) > 0).astype(np.uint8)
                    hair = plane
            score, hair = drop_exclusions(score, hair, list(ret.dust_exclusion_strokes))
            hairs: List = [hair] if hair is not None else []
            strokes = dust_detections_to_heal_strokes(score, hairs, f32.shape[:2])
            # The manual bake re-gates every stroke by local contrast on this same
            # buffer, so verify here and drop what it would skip: a placed mark that
            # repairs nothing reads as a broken promise on the canvas.
            manual = strokes_to_score(f32, strokes, [])
            kept, dropped = filter_strokes_by_score(strokes, manual, f32.shape[:2])
            self.finished.emit(kept, len(dropped), task.key)
        except Exception as e:
            self.error.emit(tr("Full-res dust scan failed: {e}", e=e), task.key)
        finally:
            self._processor.cleanup(release_source_cache=True, collect=False)
            gc.collect()
