"""Request-scoped plumbing: the per-source HistoryDB registry.

Framework-free (no FastAPI imports) so the registry can back any server
shape — the FastAPI dependency that wraps it lives in :mod:`.api`.
"""

import threading
from dataclasses import dataclass
from pathlib import Path

from ..config import AnySource, Config
from ..query import HistoryDB
from ..views import SourceStyle, style_for


@dataclass
class _Entry:
    db: HistoryDB
    style: SourceStyle
    stat_key: tuple[int, int]


class SourceRegistry:
    """One open :class:`HistoryDB` per source, refreshed when its artifact does.

    ``build_meta.parquet`` is always the artifact's final write (both full
    builds and incremental appends commit by writing it last), so its
    ``(mtime_ns, size)`` identifies a complete artifact: when the stat
    changes, the old handle is closed and a fresh one opened. Each request
    gets a :meth:`HistoryDB.fork` of the cached handle, safe for
    concurrent use, and closes it when done.
    """

    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._entries: dict[str, _Entry] = {}
        self._lock = threading.Lock()

    def acquire(self, name: str) -> tuple[HistoryDB, SourceStyle]:
        """A forked handle + style for the named source.

        Raises :class:`obohog.config.ConfigError` for unknown names and
        :class:`~obohog.query.ArtifactNotFound` /
        :class:`~obohog.query.SchemaMismatch` for unusable artifacts —
        the web layer maps those to responses.
        """
        source = self._cfg.get_source(name)
        key = _stat_key(source)
        with self._lock:
            entry = self._entries.get(name)
            if entry is None or entry.stat_key != key:
                if entry is not None:
                    entry.db.close()
                entry = _Entry(HistoryDB(source.db_dir), style_for(source), key)
                self._entries[name] = entry
            return entry.db.fork(), entry.style

    def close(self) -> None:
        with self._lock:
            for entry in self._entries.values():
                entry.db.close()
            self._entries.clear()


def _stat_key(source: AnySource) -> tuple[int, int]:
    meta = Path(source.db_dir) / "build_meta.parquet"
    try:
        st = meta.stat()
    except FileNotFoundError:
        return (0, 0)
    return (st.st_mtime_ns, st.st_size)
