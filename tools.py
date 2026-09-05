"""按需回看会话历史图片的工具。

原理：工具返回 mcp.types.CallToolResult，其中携带 ImageContent（base64 图片）。
AstrBot 的 tool_loop_agent_runner 会把这类图片落盘缓存，并紧接着追加一条
携带真实图片的 user 消息，因此模型在下一轮能真正「看到」这张图。

前置条件：当前会话使用的模型供应商需支持 image 模态，否则 runner 只保留
路径文本，模型看不到图像内容。

GIF 动图：直接吐原 GIF 多半不稳（多数模型只渲染第 1 帧），所以会先按
等间隔抽 N 帧拼成一张 contact sheet 静态预览图。需要 Pillow，没装时回退原图。
"""

from __future__ import annotations

import asyncio
import base64
import io
import math
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

from .storage import ImageRecord

_DESCRIPTION = (
    "回看本会话此前出现过的图片，并把图片原图返回给你直接观看。"
    "当用户的问题涉及之前发过的图片内容时必须调用，例如「刚才那张是什么颜色」"
    "「对比一下前面两张」「这个柜门适合贴哪种膜」「图上那个地方能不能处理」。"
    "可用图片清单见上下文中的「视觉记忆」索引，用 seq 可精确定位某一张；"
    "也可以用 query 关键词检索图片发送时伴随的文字。"
    "只能看到本会话真实发送过的图片，看不到从未发过的图。"
    "如果确认用户的问题与任何历史图片无关，不要调用。"
    "注：GIF 动图会被自动抽帧拼成多帧接触印片，预览图每帧左上角的 #K 即第 K 帧。"
)

_GIF_DESCRIPTION = (
    "专门用于回看本会话发过的 GIF 动图：按等间隔抽 N 帧（默认 8），"
    "拼成一张 contact sheet 静态预览图返回，便于一次看清整个动画过程"
    "（多数模型直接看 GIF 只会渲染第 1 帧）。"
    "seq 来自上下文「视觉记忆」索引的 #N；也可以用 query 关键词匹配 GIF 发送时伴随的文字。"
    "frame_count 控制抽多少帧（1-16），limit 控制一次处理几张 GIF（1-3）。"
    "若是普通图（JPG/PNG/WEBP），用 recall_image 即可，不要调本工具。"
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

_GIF_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "seq": {
            "type": "integer",
            "description": "GIF 图片序号，来自上下文「视觉记忆」索引里的 #N。",
        },
        "query": {
            "type": "string",
            "description": "描述要找的动图的关键词，例如「表情包」「演示动画」「柜门贴膜」。",
        },
        "frame_count": {
            "type": "integer",
            "description": "抽取帧数（1-16），默认取插件配置 gif_frame_count。动作复杂可适当调大。",
            "default": 8,
        },
        "limit": {
            "type": "integer",
            "description": "一次最多处理几张 GIF（1-3），默认 1。",
            "default": 1,
        },
    },
    "required": [],
}

_CJK = re.compile(r"[\u4e00-\u9fff]+")
_ALNUM = re.compile(r"[a-z0-9]+")


# ============================================================================
# GIF 帧抽取 / contact sheet 拼图（插件自带的 Python 工具，不依赖 AstrBot
# 的临时脚本执行通道，按需调用 Pillow 本地库即可）
# ============================================================================


def _extract_gif_frames(path: str | Path, count: int) -> list[Any]:
    """用 Pillow 从 GIF 中抽取 count 帧，按等间隔采样首尾都被采到。

    用 ImageSequence.Iterator 而不是裸 img.seek()，是为了正确处理 GIF
    的 disposal 模式（restore-to-background/previous 等），否则组合帧
    会出现半透明残影。
    必须 .copy() 才能脱离原 Image 的生命周期。
    """
    frames: list[Any] = []
    try:
        from PIL import Image, ImageSequence  # type: ignore
    except ImportError:
        return []

    try:
        with Image.open(str(path)) as img:
            n_total = int(getattr(img, "n_frames", 1) or 1)
            if n_total <= 0:
                return []

            if n_total <= count:
                target_set = set(range(n_total))
            else:
                # 等间隔采样：保证首末都被采到，中间尽量均匀
                step = (n_total - 1) / max(count - 1, 1)
                target_set = {int(round(i * step)) for i in range(count)}
                target_set = {i for i in target_set if 0 <= i < n_total}

            if not target_set:
                return []
            max_target = max(target_set)

            for idx, frame in enumerate(ImageSequence.Iterator(img)):
                if idx > max_target:
                    break
                if idx in target_set:
                    frames.append(frame.convert("RGB").copy())
    except Exception as e:
        logger.debug(f"[visual_recall] GIF 抽帧失败 {path}: {e}")
        return []

    return frames


def _pick_layout(n: int) -> tuple[int, int]:
    """根据帧数自动选 cols×rows 布局。列数 ≤4，倾向于尽量方正。"""
    if n <= 1:
        return (1, 1)
    cols = max(1, min(4, int(math.ceil(math.sqrt(n)))))
    rows = math.ceil(n / cols)
    return (cols, rows)


def _compose_contact_sheet(
    frames: list[Any],
    max_edge: int = 640,
    padding: int = 6,
    bg_color: tuple[int, int, int] = (245, 245, 245),
    label_color: tuple[int, int, int] = (220, 38, 38),
    label: bool = True,
) -> Any:
    """把若干帧拼成 contact sheet 静态预览图。

    每帧等比缩到 max_edge 内，按 cols×rows 网格铺开；左上角标 #K 即第 K 帧，
    方便模型引用具体帧（如「第 3 帧出现的那个字」）。
    """
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except ImportError:
        raise

    if not frames:
        raise ValueError("no frames to compose")

    scaled: list[tuple[Any, tuple[int, int]]] = []
    for f in frames:
        thumb = f.copy()
        thumb.thumbnail((max_edge, max_edge), Image.LANCZOS)
        scaled.append((thumb, thumb.size))

    cols, rows = _pick_layout(len(scaled))
    cell_w = max(s[1][0] for s in scaled)
    cell_h = max(s[1][1] for s in scaled)
    sheet_w = cols * cell_w + (cols + 1) * padding
    sheet_h = rows * cell_h + (rows + 1) * padding

    sheet = Image.new("RGB", (sheet_w, sheet_h), bg_color)
    draw = ImageDraw.Draw(sheet) if label else None

    # Pillow 10+ 才支持 load_default(size=)，老版本只能用 load_default()
    font: Any = None
    if draw is not None:
        try:
            font = ImageFont.load_default(size=14)
        except TypeError:
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None

    for idx, (frame, size) in enumerate(scaled):
        r, c = divmod(idx, cols)
        cell_x = padding + c * (cell_w + padding)
        cell_y = padding + r * (cell_h + padding)
        cx = cell_x + (cell_w - size[0]) // 2
        cy = cell_y + (cell_h - size[1]) // 2
        sheet.paste(frame, (cx, cy))
        if draw is not None:
            tag = f"#{idx + 1}"
            # 半透明白底，让标签在亮暗背景上都能看清
            try:
                draw.rectangle(
                    [cell_x + 2, cell_y + 2, cell_x + 36, cell_y + 20],
                    fill=(255, 255, 255),
                )
            except Exception:
                pass
            draw.text((cell_x + 5, cell_y + 2), tag, fill=label_color, font=font)

    return sheet


def _make_gif_contact_sheet_jpeg(
    path: str | Path, frame_count: int, max_edge: int
) -> bytes:
    """抽帧 + 拼图，返回 JPEG 字节。失败由调用方决定如何回退。"""
    frames = _extract_gif_frames(path, frame_count)
    if not frames:
        raise ValueError("no frames extracted")
    sheet = _compose_contact_sheet(frames, max_edge=max_edge, label=True)
    buf = io.BytesIO()
    sheet.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _gif_note_text(path: Path, sampled: int, total: int, cols: int, rows: int, source: str) -> str:
    """生成告诉模型「这张是 GIF 抽帧预览」的文字说明。

    source 标注上下文（recall_image 自动 / recall_gif_frames 显式），方便调试。
    """
    return (
        f"[GIF 抽帧预览 · 来源 {source}] 原图共 {total} 帧，"
        f"已等间隔抽取 {sampled} 帧，按 {cols}×{rows} 网格拼成静态预览图，"
        "预览图每帧左上角的 #K 即原始 GIF 的第 K 帧。"
    )


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

            gif_note: str | None = None

            try:
                raw, mime, gif_processed, gif_meta = await asyncio.to_thread(
                    self._load_image, path, rec.mime_type
                )
                if gif_processed and gif_meta is not None:
                    gif_note = _gif_note_text(
                        path,
                        gif_meta["sampled"],
                        gif_meta["total"],
                        gif_meta["cols"],
                        gif_meta["rows"],
                        source="recall_image 自动",
                    )
            except Exception as e:  # 单张失败不影响其余
                logger.warning(f"[visual_recall] 读取图片失败 {path}: {e}")
                content.append(
                    mcp.types.TextContent(type="text", text=f"#{rec.seq} 读取失败：{e}")
                )
                continue

            if gif_note:
                content.append(mcp.types.TextContent(type="text", text=gif_note))
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

    def _gif_frame_count(self) -> int:
        default = 8
        if not self.conf:
            return default
        try:
            return max(1, min(int(self.conf.get("gif_frame_count", default)), 16))
        except (TypeError, ValueError):
            return default

    def _gif_max_edge(self) -> int:
        default = 640
        if not self.conf:
            return default
        try:
            return max(64, min(int(self.conf.get("gif_max_edge", default)), 2048))
        except (TypeError, ValueError):
            return default

    def _gif_as_contact_sheet(self) -> bool:
        if not self.conf:
            return True
        try:
            return bool(self.conf.get("gif_as_contact_sheet", True))
        except Exception:
            return True

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

    def _load_image(self, path: Path, mime: str) -> tuple[bytes, str, bool, dict | None]:
        """读图、按需压缩或 GIF 抽帧，返回 (字节, mime, 是否走 GIF 抽帧, 元信息)。

        GIF 动图：根据 gif_as_contact_sheet 配置走 contact sheet 分支，
        Pillow 缺失或解码失败时回退到原图。
        其他格式：按 max_image_edge 等比压缩。
        """
        # ---- GIF 抽帧分支
        if mime == "image/gif" and self._gif_as_contact_sheet():
            try:
                count = self._gif_frame_count()
                edge = self._gif_max_edge()

                # 先用 head info 取总帧数和预览布局，得到 gif_meta 用于说明文案
                gif_meta = self._gif_meta(path, count)
                if gif_meta is None:
                    raise ValueError("gif meta unavailable")
                raw = _make_gif_contact_sheet_jpeg(path, count, edge)
                return raw, "image/jpeg", True, gif_meta
            except ImportError:
                logger.debug("[visual_recall] Pillow 未安装，GIF 回退到原图")
            except Exception as e:
                logger.warning(f"[visual_recall] GIF 抽帧失败，回退原图 {path}: {e}")

        # ---- 普通分支（兼容老逻辑）
        raw = path.read_bytes()
        max_edge = 1280
        if self.conf:
            try:
                max_edge = int(self.conf.get("max_image_edge", 1280))
            except (TypeError, ValueError):
                max_edge = 1280

        if max_edge <= 0:
            return raw, mime, False, None
        if len(raw) <= 800_000:
            try:
                from PIL import Image  # noqa: F401
            except ImportError:
                return raw, mime, False, None

        try:
            from PIL import Image

            img = Image.open(io.BytesIO(raw))
            if max(img.size) <= max_edge and len(raw) <= 800_000:
                return raw, mime, False, None

            img.thumbnail((max_edge, max_edge), Image.LANCZOS)
            buf = io.BytesIO()
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            img.save(buf, format="JPEG", quality=82)
            return buf.getvalue(), "image/jpeg", False, None
        except ImportError:
            return raw, mime, False, None
        except Exception as e:
            logger.debug(f"[visual_recall] 压缩失败，使用原图 {path}: {e}")
            return raw, mime, False, None

    @staticmethod
    def _gif_meta(path: Path, count: int) -> dict | None:
        """读取 GIF 头部信息，计算真正会被抽到的索引集合，给说明文案用。"""
        try:
            from PIL import Image
        except ImportError:
            return None
        try:
            with Image.open(str(path)) as img:
                n_total = int(getattr(img, "n_frames", 1) or 1)
        except Exception:
            return None
        if n_total <= 0:
            return None
        if n_total <= count:
            sampled = n_total
        else:
            step = (n_total - 1) / max(count - 1, 1)
            sampled = len({int(round(i * step)) for i in range(count)})
        cols, rows = _pick_layout(sampled)
        return {
            "total": n_total,
            "sampled": sampled,
            "cols": cols,
            "rows": rows,
        }


@dataclass
class RecallGifFramesTool(FunctionTool[AstrAgentContext]):
    """让模型按需回看本会话中发过的 GIF，并显式控制抽多少帧。

    与 recall_image 的差别：本工具只处理 GIF，不返回原图；模型可以指定
    frame_count 拿更多帧（动作复杂时需要），或一次处理多张 GIF 做对比。
    """

    name: str = "recall_gif_frames"
    description: str = _GIF_DESCRIPTION
    parameters: dict = Field(default_factory=lambda: dict(_GIF_PARAMETERS))

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
            return "当前环境无法定位会话，无法回看 GIF。"

        umo = event.unified_msg_origin

        seq = kwargs.get("seq")
        query = (kwargs.get("query") or "").strip()
        frame_count = self._safe_frame_count(kwargs.get("frame_count"))
        limit = self._safe_limit(kwargs.get("limit"))

        records = await self._locate(umo, seq, query, limit)
        if not records:
            return (
                "本会话没有找到匹配的图。注意：只能回看本会话此前真实发送过的图片。"
                "如果索引里这张图是 JPG/PNG/WEBP，请改用 recall_image。"
            )

        try:
            from PIL import Image  # noqa: F401
        except ImportError:
            return (
                "本环境未安装 Pillow，无法抽 GIF 帧。"
                "请在部署环境 pip install Pillow，或改用 recall_image 直接看首帧。"
            )

        gif_records = [r for r in records if r.mime_type == "image/gif"]
        if not gif_records:
            sample = ", ".join(f"#{r.seq}({r.mime_type})" for r in records[:5])
            return (
                f"已找到 {len(records)} 张候选图（{sample}{'...' if len(records) > 5 else ''}），"
                "但都不是 GIF。本工具只处理 GIF；普通图片请用 recall_image。"
            )

        content: list[mcp.types.TextContent | mcp.types.ImageContent] = []
        for rec in gif_records:
            path = Path(rec.file_path)
            if not path.exists():
                content.append(
                    mcp.types.TextContent(
                        type="text",
                        text=f"#{rec.seq} 文件已不在磁盘上：{rec.file_path}",
                    )
                )
                continue

            meta = RecallImageTool._gif_meta(path, frame_count)
            try:
                raw = await asyncio.to_thread(
                    _make_gif_contact_sheet_jpeg, path, frame_count, self._gif_max_edge()
                )
            except Exception as e:
                logger.warning(f"[visual_recall] GIF 抽帧失败 {path}: {e}")
                content.append(
                    mcp.types.TextContent(type="text", text=f"#{rec.seq} 抽帧失败：{e}")
                )
                continue

            if meta is not None:
                content.append(
                    mcp.types.TextContent(
                        type="text",
                        text=_gif_note_text(
                            path,
                            meta["sampled"],
                            meta["total"],
                            meta["cols"],
                            meta["rows"],
                            source="recall_gif_frames",
                        ),
                    )
                )
            content.append(
                mcp.types.TextContent(
                    type="text",
                    text=f"#{rec.seq} [{rec.time_str}] {rec.short_caption} (GIF 预览)",
                )
            )
            content.append(
                mcp.types.ImageContent(
                    type="image",
                    data=base64.b64encode(raw).decode("utf-8"),
                    mimeType="image/jpeg",
                )
            )

        if not content:
            return "找到了 GIF 记录，但都未能抽取帧。"
        return mcp.types.CallToolResult(content=content)  # type: ignore[arg-type]

    # -------------------------------------------------------------------- 内部

    def _safe_frame_count(self, raw: Any) -> int:
        default = self._gif_frame_count_default()
        try:
            want = int(raw) if raw is not None else default
        except (TypeError, ValueError):
            want = default
        return max(1, min(want, 16))

    def _gif_frame_count_default(self) -> int:
        if not self.conf:
            return 8
        try:
            return max(1, min(int(self.conf.get("gif_frame_count", 8)), 16))
        except (TypeError, ValueError):
            return 8

    def _gif_max_edge(self) -> int:
        default = 640
        if not self.conf:
            return default
        try:
            return max(64, min(int(self.conf.get("gif_max_edge", default)), 2048))
        except (TypeError, ValueError):
            return default

    def _safe_limit(self, raw: Any) -> int:
        cap = 3
        if self.conf:
            try:
                cap = int(self.conf.get("max_images_per_call", 3))
            except (TypeError, ValueError):
                cap = 3
        cap = max(1, min(cap, 3))
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
            return await self.store.recent(umo, limit)

        return await self.store.recent(umo, limit)
