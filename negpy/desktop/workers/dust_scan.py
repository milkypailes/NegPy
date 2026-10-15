import gc
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from negpy.domain.models import WorkspaceConfig
from negpy.services.rendering.image_processor import ImageProcessor


@dataclass(frozen=True)
class DustScanTask:
    """One full-resolution optical dust scan.

    ``config`` is the edit at dispatch: decode params plus the dust thresholds, size
    and exclusion strokes the scan runs with. ``key`` identifies the frame the scan
    left (content hash + fork suffix); the controller drops results that arrive after
    the user moved on. ``half`` is the active half-frame slice, if any.
    """

    file_path: str
    config: WorkspaceConfig
    key: str
    half: Optional[tuple] = None


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
            drop_exclusions,
            dust_detections_to_heal_strokes,
            filter_strokes_by_score,
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
            stats = compute_dust_stats(f32, int(ret.dust_size))
            dark, hair = detect_luma_score(
                f32,
                float(ret.dust_threshold),
                int(ret.dust_size),
                stats=stats,
                hair_threshold=float(ret.dust_hair_threshold),
            )
            bright = detect_bright_luma_score(f32, float(ret.dust_threshold), int(ret.dust_size))
            sides = [s for s in (dark, bright) if s is not None]
            score = sides[0] if len(sides) == 1 else (np.minimum(*sides).astype(np.float32) if sides else None)
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
            self.error.emit(f"Full-res dust scan failed: {e}", task.key)
        finally:
            self._processor.cleanup(release_source_cache=True, collect=False)
            gc.collect()
