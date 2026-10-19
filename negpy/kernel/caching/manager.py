from typing import Optional, Tuple
from negpy.kernel.caching.logic import CacheEntry


class PipelineCache:
    """
    Stores intermediate stage results for the active image.
    """

    source_hash: str = ""
    process_mode: str = ""

    # Checkpoints
    base: Optional[CacheEntry] = None
    exposure: Optional[CacheEntry] = None
    clahe: Optional[CacheEntry] = None
    lab: Optional[CacheEntry] = None
    # The clear base, read once per source: every edit reuses it.
    base_measured: bool = False
    base_log: Optional[Tuple[float, float, float]] = None

    def clear(self) -> None:
        self.base = None
        self.exposure = None
        self.clahe = None
        self.lab = None
        self.base_measured = False
        self.base_log = None
        self.source_hash = ""
