"""图片索引存储层。

用一份独立的 SQLite 文件记录会话中出现过的图片元数据，不污染 AstrBot 主库。
所有阻塞的 sqlite 调用都通过 asyncio.to_thread 丢到线程池，避免卡住事件循环。

图片文件本身由 main.py 复制到插件数据目录后，把路径写进这里。
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    umo        TEXT    NOT NULL,
    seq        INTEGER NOT NULL,
    sender_id  TEXT    NOT NULL DEFAULT '',
    file_path  TEXT    NOT NULL,
    caption    TEXT    NOT NULL DEFAULT '',
    mime_type  TEXT    NOT NULL DEFAULT 'image/jpeg',
    size       INTEGER NOT NULL DEFAULT 0,
    created_at REAL    NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_images_umo_seq  ON images(umo, seq);
CREATE INDEX        IF NOT EXISTS idx_images_umo_time ON images(umo, created_at DESC);
CREATE INDEX        IF NOT EXISTS idx_images_time     ON images(created_at);
"""


@dataclass
class ImageRecord:
    """一条图片索引记录。"""

    id: int
    umo: str
    """会话标识，即 event.unified_msg_origin。"""

    seq: int
    """该会话内的图片序号，从 1 递增，供模型精确定位。"""

    sender_id: str
    file_path: str
    caption: str
    """同一次消息里伴随的文字，是关键词检索的主要依据。"""

    mime_type: str
    size: int
    created_at: float

    @property
    def time_str(self) -> str:
        return time.strftime("%m-%d %H:%M", time.localtime(self.created_at))

    @property
    def short_caption(self) -> str:
        cap = (self.caption or "").strip()
        if not cap:
            return "（无文字说明）"
        return cap if len(cap) <= 40 else cap[:40] + "…"


def guess_mime(path: str | Path) -> str:
    """按文件后缀猜 MIME 类型，猜不出按 jpeg 处理。"""
    return _MIME_BY_SUFFIX.get(Path(path).suffix.lower(), "image/jpeg")


class ImageStore:
    """会话历史图片的元数据索引。"""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ 底层

    @contextmanager
    def _session(self):
        """取得一个连接，正常退出提交、异常回滚、最终关闭。"""
        conn = sqlite3.connect(self.db_path, timeout=15)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _to_record(row: sqlite3.Row) -> ImageRecord:
        return ImageRecord(
            id=row["id"],
            umo=row["umo"],
            seq=row["seq"],
            sender_id=row["sender_id"],
            file_path=row["file_path"],
            caption=row["caption"],
            mime_type=row["mime_type"],
            size=row["size"],
            created_at=row["created_at"],
        )

    # -------------------------------------------------------- 同步实现（线程池）

    def _init_sync(self) -> None:
        with self._session() as conn:
            conn.executescript(_SCHEMA)

    def _add_sync(
        self,
        umo: str,
        sender_id: str,
        file_path: str,
        caption: str,
        mime_type: str,
        size: int,
    ) -> int:
        with self._session() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 AS s FROM images WHERE umo = ?",
                (umo,),
            ).fetchone()
            seq = int(row["s"])
            conn.execute(
                "INSERT INTO images"
                " (umo, seq, sender_id, file_path, caption, mime_type, size, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (umo, seq, sender_id, file_path, caption, mime_type, size, time.time()),
            )
            return seq

    def _recent_sync(self, umo: str, limit: int) -> list[ImageRecord]:
        with self._session() as conn:
            rows = conn.execute(
                "SELECT * FROM images WHERE umo = ? ORDER BY id DESC LIMIT ?",
                (umo, limit),
            ).fetchall()
        return [self._to_record(r) for r in rows]

    def _get_by_seq_sync(self, umo: str, seq: int) -> ImageRecord | None:
        with self._session() as conn:
            row = conn.execute(
                "SELECT * FROM images WHERE umo = ? AND seq = ?", (umo, seq)
            ).fetchone()
        return self._to_record(row) if row else None

    def _search_sync(
        self, umo: str, tokens: list[str], limit: int
    ) -> list[ImageRecord]:
        """按关键词命中数给候选图片打分，命中越多越靠前。

        只在最近若干条里找，避免全表扫描。
        """
        if not tokens:
            return []
        with self._session() as conn:
            rows = conn.execute(
                "SELECT * FROM images WHERE umo = ? ORDER BY id DESC LIMIT 300",
                (umo,),
            ).fetchall()

        scored: list[tuple[int, int, ImageRecord]] = []
        for r in rows:
            caption = (r["caption"] or "").lower()
            hit = sum(1 for t in tokens if t in caption)
            if hit:
                scored.append((hit, r["id"], self._to_record(r)))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        return [rec for _, _, rec in scored[:limit]]

    def _count_sync(self, umo: str | None = None) -> int:
        with self._session() as conn:
            if umo:
                row = conn.execute(
                    "SELECT COUNT(*) AS c FROM images WHERE umo = ?", (umo,)
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) AS c FROM images").fetchone()
        return int(row["c"])

    def _cleanup_sync(self, max_per_session: int, max_days: float) -> list[str]:
        """清掉过期与超额的记录，返回需要一并删除的磁盘文件路径。"""
        removed: list[str] = []
        with self._session() as conn:
            if max_days > 0:
                cutoff = time.time() - max_days * 86400
                rows = conn.execute(
                    "SELECT id, file_path FROM images WHERE created_at < ?", (cutoff,)
                ).fetchall()
                for r in rows:
                    removed.append(r["file_path"])
                    conn.execute("DELETE FROM images WHERE id = ?", (r["id"],))

            if max_per_session > 0:
                umos = [
                    r[0] for r in conn.execute("SELECT DISTINCT umo FROM images").fetchall()
                ]
                for umo in umos:
                    rows = conn.execute(
                        "SELECT id, file_path FROM images WHERE umo = ?"
                        " ORDER BY id DESC LIMIT -1 OFFSET ?",
                        (umo, max_per_session),
                    ).fetchall()
                    for r in rows:
                        removed.append(r["file_path"])
                        conn.execute("DELETE FROM images WHERE id = ?", (r["id"],))
        return removed

    def _clear_session_sync(self, umo: str) -> list[str]:
        """删除某会话的全部记录，返回需要一并删除的磁盘文件路径。"""
        with self._session() as conn:
            rows = conn.execute(
                "SELECT file_path FROM images WHERE umo = ?", (umo,)
            ).fetchall()
            conn.execute("DELETE FROM images WHERE umo = ?", (umo,))
        return [r["file_path"] for r in rows]

    # ---------------------------------------------------------------- 异步接口

    async def init(self) -> None:
        await asyncio.to_thread(self._init_sync)

    async def add(
        self,
        umo: str,
        sender_id: str,
        file_path: str,
        caption: str,
        mime_type: str,
        size: int,
    ) -> int:
        return await asyncio.to_thread(
            self._add_sync, umo, sender_id, file_path, caption, mime_type, size
        )

    async def recent(self, umo: str, limit: int) -> list[ImageRecord]:
        return await asyncio.to_thread(self._recent_sync, umo, limit)

    async def get_by_seq(self, umo: str, seq: int) -> ImageRecord | None:
        return await asyncio.to_thread(self._get_by_seq_sync, umo, seq)

    async def search(self, umo: str, tokens: list[str], limit: int) -> list[ImageRecord]:
        return await asyncio.to_thread(self._search_sync, umo, tokens, limit)

    async def count(self, umo: str | None = None) -> int:
        return await asyncio.to_thread(self._count_sync, umo)

    async def cleanup(self, max_per_session: int, max_days: float) -> list[str]:
        return await asyncio.to_thread(self._cleanup_sync, max_per_session, max_days)

    async def clear_session(self, umo: str) -> list[str]:
        return await asyncio.to_thread(self._clear_session_sync, umo)
