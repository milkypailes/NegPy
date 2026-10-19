"""Drops per-frame bounds saved under an older BOUNDS_VERSION, so each frame meters again on its
next render. A frame with Lock Bounds keeps its own. Roll baselines stay until Roll Analysis runs.
Best-effort; it never raises into app startup, and a failed sweep runs again at the next start.
"""

import json
import sqlite3
from contextlib import closing

from negpy.features.exposure.normalization import BOUNDS_VERSION
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

_VERSION_KEY = "bounds_version"
_TABLES = (
    ("file_settings", ("file_hash",)),
    ("edit_history", ("file_hash", "step_index")),
    ("work_prints", ("file_hash", "name")),
)


def _drop_stale_bounds(conn: sqlite3.Connection, table: str, key_cols: tuple[str, ...]) -> None:
    cols = ", ".join(key_cols)
    where = " AND ".join(f"{c} = ?" for c in key_cols)
    for *keys, settings_json in conn.execute(f"SELECT {cols}, settings_json FROM {table}").fetchall():
        try:
            data = json.loads(settings_json or "")
        except (ValueError, TypeError):
            continue
        if not isinstance(data, dict) or data.get("lock_bounds") or not any(data.get("local_floors") or ()):
            continue
        data["local_floors"] = [0.0, 0.0, 0.0]
        data["local_ceils"] = [0.0, 0.0, 0.0]
        conn.execute(f"UPDATE {table} SET settings_json = ? WHERE {where}", (json.dumps(data, default=str), *keys))


def migrate_stale_frame_bounds(repo) -> None:
    if int(repo.get_global_setting(_VERSION_KEY) or 0) >= BOUNDS_VERSION:
        return
    try:
        with closing(sqlite3.connect(repo.edits_db_path)) as conn, conn:
            for table, key_cols in _TABLES:
                _drop_stale_bounds(conn, table, key_cols)
    except Exception:
        logger.exception("Stale-bounds migration failed; continuing without it")
        return
    repo.save_global_setting(_VERSION_KEY, BOUNDS_VERSION)
