"""按需回看会话历史图片的工具。

原理：工具返回 mcp.types.CallToolResult，其中携带 ImageContent（base64 图片）。
AstrBot 的 tool_loop_agent_runner 会把这类图片落盘缓存，并紧接着追加一条
携带真实图片的 user 消息，因此模型在下一轮能真正「看到」这张图。

前置条件：当前会话使用的模型供应商需支持 image 模态，否则 runner 只保留
路径文本，模型看不到图像内容。
"""

from __future__ import annotations

import asyncio
import base64
import io
import re
from pathlib import Path
from typing import Any

import mcp
from pydantic import Field
from pydantic.dataclasses import dataclass

from astrbot import logger
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext

from storage import ImageRecord

_DESCRIPTION = (
    "回看本会话此前出现过的图片，并把图片原图返回给你直接观看。"
    "当用户的问题涉及之前发过的图片内容时必须调用，例如「刚才那张是什么颜色」"
    "「对比一下前面两张」「这个柜门适合贴哪种膜」「图上那个地方能不能处理」。"
    "可用图片清单见上下文中的「视觉记忆」索引，用 seq 可精确定位某一张；"
    "也可以用 query 关键词检索图片发送时伴随的文字。"
    "只能看到本会话真实发送过的图片，看不到从未发过的图。"
    "如果确认用户的问题与任何历史图片无关，不要调用。"
)

_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "seq": {
            "type": "integer",
            "description": "图片序号，来自上下文「视觉记忆」索引里的 #N。指定后精确返回该张，忽略 query。",
        },
        "query": {
            "type": "string",
            "description": "描述你要找的图的关键词，会与图片发送时伴随的文字匹配，例如「柜门」「红色」。留空表示取最近的一张。",
        },
        "limit": {
            "type": "integer",
            "description": "最多返回几张，默认 1。需要对比多张时可设为 2 或 3。",
            "default": 1,
        },
    },
    "required": [],
}

_CJK = re.compile(r"[\u4e00-\u9fff]+")
_ALNUM = re.compile(r"[a-z0-9]+")


def _tokenize(query: str) -> list[str]:
    """把查询切成匹配用的词元。

    中文没有空格，整词之外再补 2-gram，提高部分匹配的召回率。
    """
    q = (query or "").strip().lower()
    if not q:
        return []

    tokens: list[str] = []
    for part in _CJK.findall(q):
        tokens.append(part)
        if len(part) > 2:
            tokens.extend(part[i : i + 2] for i in range(len(part) - 1))
    tokens.extend(_ALNUM.findall(q))

    seen: set[str] = set()
    ordered: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered


@dataclass
class RecallImageTool(FunctionTool[AstrAgentContext]):
    """让模型按需回看会话历史图片。"""

    name: str = "recall_image"
    description: str = _DESCRIPTION
    parameters: dict = Field(default_factory=lambda: dict(_PARAMETERS))

    # 以下字段不参与工具 schema，仅用于依赖注入
    store: Any = None
    conf: Any = None

    # ------------------------------------------------------------------ 工具逻辑

    async def call(
        self,
        context: ContextWrapper[AstrAgentContext],
        **kwargs,
    ) -> ToolExecResult:
        agent_ctx = context.context if context else None
        event = getattr(agent_ctx, "event", None)
        if event is None or self.store is None:
            return "当前环境无法定位会话，无法回看图片。"

        umo = event.unified_msg_origin

        seq = kwargs.get("seq")
        query = (kwargs.get("query") or "").strip()
        limit = self._safe_limit(kwargs.get("limit"))

        records = await self._locate(umo, seq, query, limit)
        if not records:
            return (
                "本会话没有找到匹配的图片。注意：只能回看本会话此前真实发送过的图片，"
                "无法查看从未发送过的图。若用户确实发过但索引里没有，可能是该图早于本插件安装时间。"
            )

        content: list[mcp.types.TextContent | mcp.types.ImageContent] = []
        for rec in records:
            path = Path(rec.file_path)
            if not path.exists():
                content.append(
                    mcp.types.TextContent(
                        type="text",
                        text=f"#{rec.seq} 的图片文件已不在磁盘上（可能已被清理）：{rec.file_path}",
                    )
                )
                continue
            try:
                raw, mime = await asyncio.to_thread(
                    self._load_image, path, rec.mime_type
                )
            except Exception as e:  # 单张失败不影响其余
                logger.warning(f"[visual_recall] 读取图片失败 {path}: {e}")
                content.append(
                    mcp.types.TextContent(type="text", text=f"#{rec.seq} 读取失败：{e}")
                )
                continue

            content.append(
                mcp.types.TextContent(
                    type="text",
                    text=f"#{rec.seq} [{rec.time_str}] {rec.short_caption}",
                )
            )
            content.append(
                mcp.types.ImageContent(
                    type="image",
                    data=base64.b64encode(raw).decode("utf-8"),
                    mimeType=mime,
                )
            )

        if not content:
            return "找到了图片记录，但所有图片都无法读取。"
        return mcp.types.CallToolResult(content=content)  # type: ignore[arg-type]

    # -------------------------------------------------------------------- 内部

    def _safe_limit(self, raw: Any) -> int:
        cap = 3
        if self.conf:
            try:
                cap = int(self.conf.get("max_images_per_call", 3))
            except (TypeError, ValueError):
                cap = 3
        cap = max(1, min(cap, 6))
        try:
            want = int(raw) if raw is not None else 1
        except (TypeError, ValueError):
            want = 1
        return max(1, min(want, cap))

    async def _locate(
        self, umo: str, seq: Any, query: str, limit: int
    ) -> list[ImageRecord]:
        if seq is not None:
            try:
                rec = await self.store.get_by_seq(umo, int(seq))
            except (TypeError, ValueError):
                rec = None
            return [rec] if rec else []

        if query:
            records = await self.store.search(umo, _tokenize(query), limit)
            if records:
                return records
            # 关键词没命中时退回最近几张，至少给模型一点依据
            return await self.store.recent(umo, limit)

        return await self.store.recent(umo, limit)

    def _load_image(self, path: Path, mime: str) -> tuple[bytes, str]:
        """读图并按需压缩，返回 (字节, mime)。

        压缩只在 Pillow 可用时进行；GIF 动图不做压缩以免丢帧。
        """
        raw = path.read_bytes()
        max_edge = 1280
        if self.conf:
            try:
                max_edge = int(self.conf.get("max_image_edge", 1280))
            except (TypeError, ValueError):
                max_edge = 1280

        if max_edge <= 0 or mime == "image/gif":
            return raw, mime
        if len(raw) <= 800_000:
            # 小图先不解码，省一次开销
            try:
                from PIL import Image  # noqa: F401
            except ImportError:
                return raw, mime

        try:
            from PIL import Image

            img = Image.open(io.BytesIO(raw))
            if max(img.size) <= max_edge and len(raw) <= 800_000:
                return raw, mime

            img.thumbnail((max_edge, max_edge), Image.LANCZOS)
            buf = io.BytesIO()
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            img.save(buf, format="JPEG", quality=82)
            return buf.getvalue(), "image/jpeg"
        except ImportError:
            return raw, mime
        except Exception as e:
            logger.debug(f"[visual_recall] 压缩失败，使用原图 {path}: {e}")
            return raw, mime
